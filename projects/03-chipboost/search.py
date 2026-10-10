#!/usr/bin/env python3
"""
search.py -- arm (c) of the three-arm comparison: random search over the expert matmul's block caps,
no AI. Same referee and same budget as the model arms, so if random search wins anywhere we can say so.

    python search.py --budget 24 --seed 0                         # in the seat pod, real referee
    python search.py --budget 24 --seed 1 --run-id matmul-random_search-s1-rep2
    python search.py --stub --budget 5 --out /tmp/x.jsonl         # laptop: a FAKE referee
    python search.py --dry-run --budget 24 --seed 0               # the candidate list, nothing runs

    # the exhaustive sweep, the ground truth for arm c: every triple that fits, split over cores
    CHIPBOOST_CORE=2 python search.py --exhaustive --shard 0/2      # attempt 0 + half the space
    CHIPBOOST_CORE=3 python search.py --exhaustive --shard 1/2      # the other half
    python search.py --summarize logs/seat-102/sweep*.jsonl logs/seat-102/attempts.jsonl

The space is every (tm, tn, tk) that divides the primary shape's tile counts (shapes.cases("matmul",
"timing")[0]), minus what will not fit in SBUF. Attempt 0 is kernels/matmul_expert.py as shipped, then
budget - 1 distinct random triples. Each candidate is that file with its three cap lines rewritten,
judged by speedcheck.check_isolated in a fresh process, and APPENDED to --out (default
logs/seat-<N>/attempts.jsonl) as one schema record, like every attempts.jsonl.

The budget is the team's: one referee evaluation = one attempt = one unit of --budget, whatever the
verdict, as in agent.py. Held-out shapes are the referee's job (the hardened one draws three for every
would-be "faster") and heldout_grid.py's at the end; this file has no held-out logic. When the REFEREE
itself fails (no free core, broken baseline) check_isolated returns None: that is retried, and if it
keeps failing the candidate is skipped and reported, never logged as a verdict on the kernel.

--stub (also used, with a warning, when speedcheck cannot be imported) is a fake referee for testing
without NKI. Its records carry source "sim" and "(stub)" messages, and its output is fenced by STUB
banners, so it cannot pass for a result.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import socket
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import schema  # noqa: E402
import shapes  # noqa: E402  also puts ../02-kernel-agent on sys.path

TEMPLATE = "kernels/matmul_expert.py"
CAP_NAMES = ("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N", "TILES_IN_BLOCK_K")
TILE_M, TILE_N, TILE_K = 128, 512, 128   # the kernel's nl.tile_size values: tiles are M/128, N/512, K/128

# Stub mode only, while kernels/matmul_expert.py does not exist: the three cap lines, nothing runnable.
STUB_TEMPLATE = '''"""(stub) stand-in for kernels/matmul_expert.py, which does not exist yet. Not a kernel."""
TILES_IN_BLOCK_M = 16   # search.py rewrites these three lines
TILES_IN_BLOCK_N = 2
TILES_IN_BLOCK_K = 8
'''
STUB_BASELINE_US = 1000.0   # round on purpose: no real baseline looks like this
STUB_BANNER = ("#" * 78 + "\n"
               "#  STUB REFEREE: every verdict, time and speedup below is FAKE. Nothing was\n"
               "#  compiled, simulated or timed. Never report these numbers.\n" + "#" * 78)


def sbuf_bytes_per_partition(tm, tn, tk, M, itemsize=2, acc_bytes=4):
    block_m, block_n = tm * 128, tn * 512
    return (tk * block_m * itemsize          # lhsT block
            + tk * block_n * itemsize        # rhs block
            + (M // 128) * block_n * acc_bytes)  # result accumulator


SBUF_LIMIT = int(0.75 * 192 * 1024)   # 192 KiB per partition, 25% headroom for the compiler

# Candidate files live OUTSIDE the referee's folder. The hardened referee hashes every .py under it before
# and after a check and calls any change tampering: with several searches running at once, one search's
# fresh candidate file turned every other check into `rules` (measured on seat-102, 4 cores).
RUNS_ROOT = os.environ.get("CHIPBOOST_RUNS", os.path.join(tempfile.gettempdir(), "chipboost_runs"))


# ---------------------------------------------------------------- the space

def tile_counts(case):
    """(M, N, K) tile counts of a matmul case: the numbers the caps must divide."""
    return case["M"] // TILE_M, case["N"] // TILE_N, case["K"] // TILE_K


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def candidate_space(tiles):
    """Every (tm, tn, tk) dividing the tile counts. Each is a distinct kernel at this shape; any other
    cap would just round down to one of these."""
    tm_n, tn_n, tk_n = tiles
    return [(tm, tn, tk) for tm in divisors(tm_n) for tn in divisors(tn_n) for tk in divisors(tk_n)]


def effective(caps, tiles):
    """What the kernel runs: per dimension the largest divisor of the tile count <= the cap, as the
    loop in kernels/matmul_expert.py computes it (1 if none)."""
    return tuple(max([d for d in divisors(n) if d <= cap] or [1]) for cap, n in zip(caps, tiles))


def plan(pool, first, budget, seed):
    """Attempt 0 is `first`, then budget - 1 distinct triples of `pool` in random.Random(seed) order.
    Sorted before the shuffle, so the order depends on the seed alone. Never silently shortened: an
    equal budget is the point of the comparison."""
    pool = sorted(pool)
    if budget < 1:
        raise ValueError("--budget must be at least 1: attempt 0 is the expert kernel as shipped")
    if budget - 1 > len(pool):
        raise ValueError(f"--budget {budget} needs {budget - 1} random triples, but only {len(pool)} are "
                         f"left after the SBUF filter and the exclusion: use --budget {len(pool) + 1} or less")
    random.Random(seed).shuffle(pool)
    return [tuple(first)] + pool[:budget - 1]


# ---------------------------------------------------------------- the candidate file

def _cap_pattern(name):
    return re.compile(rf"^({name}\s*=\s*)(\d+)", re.M)   # module level only: no indent


def read_caps(src):
    """The three caps as the template ships them."""
    caps = []
    for name in CAP_NAMES:
        found = _cap_pattern(name).findall(src)
        if len(found) != 1:
            raise ValueError(f"expected exactly one module-level line `{name} = <int>` in the template, "
                             f"found {len(found)}")
        caps.append(int(found[0][1]))
    return tuple(caps)


def rewrite_caps(src, caps):
    """The template with its three cap lines set to `caps`. Fails unless each line is found exactly
    once: a silent miss would time the shipped kernel under another triple's name."""
    for name, value in zip(CAP_NAMES, caps):
        src, n = _cap_pattern(name).subn(rf"\g<1>{int(value)}", src)
        if n != 1:
            raise ValueError(f"expected exactly one module-level line `{name} = <int>` in the template, "
                             f"found {n}")
    return src


# ---------------------------------------------------------------- referees

def stub_referee(caps, runs_as, seed, tiles, M):
    """A FAKE referee. Deterministic from the triple and the seed (hashlib: hash() changes per process),
    about 20% wrong, the rest a smooth function of the blocks plus +-2% noise. sim_ok and chip_ok stay
    None because nothing ran."""
    h = hashlib.sha256(f"stub {tuple(caps)} {seed}".encode()).digest()
    u_wrong = int.from_bytes(h[:8], "big") / 2.0 ** 64
    u_noise = int.from_bytes(h[8:16], "big") / 2.0 ** 64
    rec = {k: None for k in schema.ATTEMPT_FIELDS}
    rec.update(kernel="matmul", source="sim",
               instruction_given="(stub) FAKE referee: nothing was measured, so there is no instruction")
    if u_wrong < 0.2:
        rec.update(verdict="wrong", referee_message="(stub) FAKE 'wrong', a dice roll: no kernel was "
                                                    "compiled, simulated or timed")
        return rec
    tm, tn, tk = runs_as
    reuse = math.log2(tm * tn * tk) / math.log2(max(2, tiles[0] * tiles[1] * tiles[2]))   # 0 .. 1
    pressure = sbuf_bytes_per_partition(tm, tn, tk, M) / SBUF_LIMIT
    speedup = (0.8 + 2.0 * reuse - 0.8 * pressure ** 2) * (1.0 + 0.04 * (u_noise - 0.5))
    t = STUB_BASELINE_US / speedup
    rec.update(verdict="faster" if speedup >= 1.05 else "slower",
               referee_message=f"(stub) FAKE: nothing ran; {t:.1f} us vs a made-up {STUB_BASELINE_US:.1f} us "
                               f"= {speedup:.3f}x",
               time_us_median=t, time_us_iqr=0.02 * t, baseline_us_same_session=STUB_BASELINE_US,
               speedup=STUB_BASELINE_US / t)
    return rec


def finish(rec, seat, run_id, attempt_no, src):
    """The referee's record with the fields this arm owns overwritten; raises if it is not valid."""
    rec = dict(rec)
    rec.update(seat=seat, kernel="matmul", arm="random_search", run_id=run_id, attempt_no=attempt_no,
               round=attempt_no, prompt_tokens=None, prompt=None, response=None, code=src,
               # sha1 cut to 12 hex digits, the referee's own code_hash format, so arms join on it
               code_hash=hashlib.sha1(src.encode()).hexdigest()[:12], timestamp=time.time())
    problems = schema.validate(rec)
    if problems:
        raise ValueError(f"attempt {attempt_no}: invalid record: {problems}")
    return rec


BASELINE = os.path.join(HERE, "kernels", "matmul_start.py")


class _Isolated:
    """check_isolated behind RefereeWorker's interface, for a referee without the worker."""

    def __init__(self, speedcheck):
        self.speedcheck, self.last_error = speedcheck, None

    def check(self, path):
        return self.speedcheck.check_isolated(path, op="matmul", baseline=BASELINE)

    def close(self):
        pass


def open_referee(speedcheck):
    """P1's RefereeWorker when the referee has it: one process holds the NeuronCore (CHIPBOOST_CORE, inherited)
    across candidates, skipping the 6-13 s runtime start check_isolated pays each time. Same contract: a
    record, or None when the REFEREE failed."""
    if hasattr(speedcheck, "RefereeWorker"):
        return speedcheck.RefereeWorker(op="matmul", baseline=BASELINE), "RefereeWorker, one process for the run"
    return _Isolated(speedcheck), "check_isolated, one fresh process per candidate"


def call_referee(referee, path, tries=3, wait=10):
    """referee.check(path), retried while the REFEREE fails (it returns None): a busy core or a broken
    baseline is not a verdict on the kernel. None after `tries` attempts means skip the candidate."""
    for i in range(tries):
        rec = referee.check(path)
        if rec is not None:
            return rec
        why = (getattr(referee, "last_error", None) or "no reason given").splitlines()[0][:160]
        if i < tries - 1:
            print(f"     referee failed (not a verdict on the kernel: {why}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    return None


# ---------------------------------------------------------------- cli

def caps_str(t):
    return f"m{t[0]} n{t[1]} k{t[2]}"


def seat_from_hostname():
    """seat-102 -> 102, as agent.py does: the pod's hostname is its seat."""
    h = socket.gethostname()
    tail = h.split("-", 1)[1] if h.startswith("seat-") else ""
    return int(tail) if tail.isdigit() else None


def summarize(paths):
    """Arm c's report from sweep and attempt logs. The sweep (run ids matmul-sweep-*) is the ground truth:
    every SBUF-fitting triple timed once. Each random-search run is then placed against it: its best
    verified triple, the attempt it came on, and that triple's rank among all of them. Triples are keyed by
    what they RUN as at the primary shape (the expert's caps 16, 2, 8 run as 2, 2, 8)."""
    import glob
    tiles = tile_counts(shapes.cases("matmul", "timing")[0])
    files = sorted({f for p in paths for f in glob.glob(p)})
    runs = {}
    for path in files:
        for line in open(path):
            try:
                rec = json.loads(line)
                caps = effective(read_caps(rec["code"]), tiles)
            except (ValueError, KeyError, TypeError):
                continue
            if rec.get("kernel") == "matmul" and rec.get("arm") == "random_search":
                runs.setdefault(rec.get("run_id") or "?", []).append(dict(rec, caps=caps))

    truth = {}
    for rid, recs in runs.items():
        if rid.startswith("matmul-sweep-"):
            for r in recs:
                truth[r["caps"]] = r
    timed = sorted((r for r in truth.values() if r.get("speedup") is not None
                    and r.get("verdict") in ("faster", "no_gain", "slower")), key=lambda r: -r["speedup"])
    rank = {r["caps"]: i for i, r in enumerate(timed, 1)}
    print(f"{len(files)} file(s); the sweep covers {len(truth)} triple(s), {len(timed)} of them timed")
    for i, r in enumerate(timed[:10], 1):
        print(f"  sweep #{i:<3} {caps_str(r['caps']):<12} {r['verdict']:<8} {r['time_us_median']:8.1f} us "
              f"{r['speedup']:6.3f}x")
    expert = effective((16, 2, 8), tiles)
    shipped = rank.get(expert) and timed[rank[expert] - 1]
    if timed:
        print(f"  ground truth: {caps_str(timed[0]['caps'])} at {timed[0]['speedup']:.3f}x; the expert as shipped "
              f"({caps_str(expert)}) ranks #{rank.get(expert, '?')} of {len(timed)}"
              + (f"; the best is {timed[0]['speedup'] / shipped['speedup']:.3f}x over it" if shipped else ""))

    # Attempt 0 is the expert as shipped: AWS's design, not a discovery. What the search found is its best
    # over that attempt 0, both timed in the run's own session (the dashboard's tuning study does the same).
    print()
    bests, gains = [], []
    for rid, recs in sorted(runs.items()):
        if rid.startswith("matmul-sweep-"):
            continue
        recs.sort(key=lambda r: r.get("attempt_no") or 0)
        ver = [r for r in recs if r.get("verdict") == "faster" and r.get("speedup") is not None]
        if not ver:
            print(f"  {rid}: {len(recs)} evaluations, no verified speedup")
            continue
        b = max(ver, key=lambda r: r["speedup"])
        bests.append(b["speedup"])
        a0 = recs[0] if recs[0].get("attempt_no") == 0 and recs[0]["caps"] == expert else None
        gain = ""
        if a0 and a0.get("speedup"):
            gains.append(b["speedup"] / a0["speedup"])
            gain = (f", {gains[-1]:.3f}x over its attempt 0 (the expert as shipped, {a0['speedup']:.3f}x)"
                    if b is not a0 else ", nothing beat its attempt 0 (the expert as shipped)")
        where = f"rank #{rank[b['caps']]} of {len(timed)} in the sweep" if b["caps"] in rank else "not in the sweep"
        share = f", {b['speedup'] / timed[0]['speedup']:.1%} of the ground truth" if timed else ""
        print(f"  {rid}: {len(recs)} evaluations; best verified {caps_str(b['caps'])} {b['speedup']:.3f}x "
              f"at attempt {b.get('attempt_no')}{gain}; {where}{share}")
    if bests:
        bests.sort()
        print(f"  random search, {len(bests)} run(s): best verified vs start min {bests[0]:.3f}x, "
              f"median {bests[len(bests) // 2]:.3f}x, max {bests[-1]:.3f}x")
    if gains:
        gains.sort()
        print(f"  the search's own share, over the expert as shipped: min {gains[0]:.3f}x, "
              f"median {gains[len(gains) // 2]:.3f}x, max {gains[-1]:.3f}x")
    return runs


def resolve_seat(seat, required):
    """--seat, else $CHIPBOOST_SEAT, else the pod's hostname. A real run must name its seat (it labels
    every record and picks the log); a stub run falls back to 100, since its records are fake anyway."""
    if seat is not None:
        return seat, "--seat"
    if os.environ.get("CHIPBOOST_SEAT"):
        return int(os.environ["CHIPBOOST_SEAT"]), "CHIPBOOST_SEAT"
    if seat_from_hostname() is not None:
        return seat_from_hostname(), "hostname"
    if required:
        sys.exit("no seat given: pass --seat N or set CHIPBOOST_SEAT (the hostname is not seat-N), so the "
                 "records name the seat that ran them.")
    return 100, "default: no --seat or CHIPBOOST_SEAT"


def main():
    ap = argparse.ArgumentParser(description="arm (c): random search over matmul_expert.py's block caps")
    ap.add_argument("--budget", type=int, default=24, help="referee calls including attempt 0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seat", type=int, help="overwrites each record's seat; default $CHIPBOOST_SEAT")
    ap.add_argument("--out", help="appended to; default logs/seat-<seat>/attempts.jsonl")
    ap.add_argument("--run-id", help="default matmul-random_search-s<seed>-<HHMMSS>")
    ap.add_argument("--stub", action="store_true", help="FAKE referee, for testing without NKI")
    ap.add_argument("--dry-run", action="store_true", help="print the candidate list and exit")
    ap.add_argument("--exhaustive", action="store_true",
                    help="every triple that fits (ignores --budget); logs to sweep*.jsonl, not attempts")
    ap.add_argument("--shard", default="0/1", metavar="I/N", help="with --exhaustive: take every N-th candidate")
    ap.add_argument("--summarize", nargs="+", metavar="JSONL", help="rank the triples in these logs and exit")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
        return 0
    try:
        shard_i, shard_n = (int(v) for v in a.shard.split("/"))
        assert shard_n >= 1 and 0 <= shard_i < shard_n
    except (ValueError, AssertionError):
        sys.exit(f"--shard must look like I/N with 0 <= I < N, got {a.shard!r}")
    out = os.path.abspath(a.out) if a.out else None   # relative to where it was typed, before the chdir
    os.chdir(HERE)   # check_isolated's child inherits our cwd and resolves the relative paths against it

    speedcheck, stub, fell_back = None, a.stub, None
    if not a.dry_run and not stub:
        try:
            import speedcheck
        except Exception as e:   # ImportError, or a SyntaxError/TypeError from newer-Python code
            stub, fell_back = True, f"{type(e).__name__}: {e}"
        if not stub:
            try:
                import nki  # noqa: F401
            except ImportError:
                # Without it the referee still returns a valid record ("rules", "failed to import")
                # that would look like a real rejection, so refuse rather than log it.
                print("nki is not installed here: run this in the seat pod, or pass --stub for a fake "
                      "referee.")
                return 2
            if not os.path.exists("kernels/matmul_start.py"):
                # the referee would silently time against reference_level4.py instead
                sys.exit("kernels/matmul_start.py not found: it is the baseline every speedup is against.")

    if os.path.exists(TEMPLATE):
        template, template_name = open(TEMPLATE).read(), TEMPLATE
    elif stub or a.dry_run:
        template, template_name = STUB_TEMPLATE, "an inline stub template (kernels/matmul_expert.py is missing)"
    else:
        sys.exit(f"{TEMPLATE} not found: search.py rewrites its three cap lines.")

    case = shapes.cases("matmul", "timing")[0]   # the primary shape
    tiles, M = tile_counts(case), case["M"]
    space = candidate_space(tiles)
    fit = [t for t in space if sbuf_bytes_per_partition(*t, M) <= SBUF_LIMIT]
    try:
        caps0 = read_caps(template)
        repeat = effective(caps0, tiles)         # attempt 0 as it runs here: never draw it again
        pool = [t for t in fit if t != repeat]
        if a.exhaustive:
            # Attempt 0 first, then every other triple that fits, in a fixed order; a shard takes every
            # N-th, so the shards together cover the space exactly once.
            order = ([tuple(caps0)] + sorted(pool))[shard_i::shard_n]
        else:
            order = plan(pool, caps0, a.budget, a.seed)
            # The same seeded order continued: when the referee fails on a candidate, the next unused
            # triple takes its place, so the run still spends its whole budget.
            reserve = plan(pool, caps0, len(pool) + 1, a.seed)[a.budget:]
    except ValueError as e:
        sys.exit(str(e))

    mark = "STUB " if stub else ""
    if fell_back:
        print(f"WARNING: `import speedcheck` failed ({fell_back}), so this run uses the FAKE stub referee.")
    if stub:
        print(STUB_BANNER)
    print(f"{mark}primary shape {shapes.label('matmul', case)}: tile counts M {tiles[0]}, N {tiles[1]}, "
          f"K {tiles[2]} -> {len(space)} triples")
    print(f"{mark}SBUF filter: dropped {len(space) - len(fit)} of {len(space)} (estimate over {SBUF_LIMIT:,} "
          f"B per partition), {len(fit)} fit")
    if a.exhaustive:
        print(f"{mark}EXHAUSTIVE sweep, shard {shard_i}/{shard_n}: {len(order)} of the {len(pool) + 1} candidates "
              f"(the expert as shipped, caps {caps_str(caps0)}, then every other triple that fits; "
              f"the expert is in shard 0). Not an arm run: logged to sweep-*.jsonl")
    else:
        print(f"{mark}attempt 0: {template_name} as shipped, caps {caps_str(caps0)}, which run as "
              f"{caps_str(repeat)} here, so {repeat} is excluded from the random draws (no second timing)")
        print(f"{mark}attempts 1-{len(order) - 1}: distinct random triples of the {len(pool)} left, "
              f"random.Random({a.seed})" if len(order) > 1 else f"{mark}no random attempts (--budget 1)")

    if a.dry_run:
        for n, caps in enumerate(order):
            runs = effective(caps, tiles)
            b = sbuf_bytes_per_partition(*runs, M)
            note = f" (runs as {caps_str(runs)})" if runs != tuple(caps) else ""
            print(f"  #{n:<3} caps {caps_str(caps):<12}{note:<22} SBUF {b:>7,} B per partition "
                  f"({b / SBUF_LIMIT:4.0%} of the limit)")
        print("dry run: no referee calls")
        return 0

    seat, seat_from = resolve_seat(a.seat, required=not stub)
    run_id = a.run_id or (f"matmul-sweep-{shard_i}of{shard_n}-{time.strftime('%H%M%S')}" if a.exhaustive
                          else f"matmul-random_search-s{a.seed}-{time.strftime('%H%M%S')}")
    run_dir = os.path.join(RUNS_ROOT, run_id)
    # The sweep is not an equal-budget arm run, so it never goes where the dashboard reads attempts.
    out = out or os.path.join(HERE, "logs", f"seat-{seat}",
                              f"sweep-{shard_i}of{shard_n}.jsonl" if a.exhaustive else "attempts.jsonl")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    referee, referee_name = (None, "FAKE stub") if stub else open_referee(speedcheck)
    print(f"{mark}run {run_id}: seat {seat} ({seat_from}), core {os.environ.get('CHIPBOOST_CORE', 'auto')}, "
          f"referee {referee_name}")
    print(f"{mark}candidates in {run_dir} (outside the referee's folder), records appended to {out}")

    best, best_caps = None, None          # best verified: verdict "faster", the dashboard's definition
    timed, timed_caps = None, None        # best correct and timed, any verdict
    skipped, failing = [], 0
    queue = list(order)
    reserve = [] if a.exhaustive else reserve
    n = 0   # attempts logged; a candidate the referee could not judge gets no number
    while queue:
        caps = queue.pop(0)
        src = rewrite_caps(template, caps)
        path = os.path.join(run_dir, f"cand_m{caps[0]}_n{caps[1]}_k{caps[2]}.py")
        with open(path, "w") as f:
            f.write(src)
        rec = (stub_referee(caps, effective(caps, tiles), a.seed, tiles, M) if stub
               else call_referee(referee, path))
        if rec is None:
            skipped.append(caps)
            failing += 1
            if n == 0 and not a.exhaustive:
                print(f"{mark}attempt 0 (the expert as shipped) could not be judged: the referee failed 3 "
                      f"times. Stopping; fix the referee (a free core?) and rerun.", flush=True)
                break
            swap = reserve.pop(0) if reserve else None
            if swap is not None:
                queue.append(swap)
            print(f"{mark}     caps {caps_str(caps):<12} SKIPPED: the referee failed 3 times, not logged"
                  + (f"; caps {caps_str(swap)} takes its place" if swap is not None else ""), flush=True)
            if failing >= 3:
                print(f"{mark}the referee failed on 3 candidates in a row: stopping rather than burn the "
                      f"rest. Check the cores (REFEREE.md section 7) and rerun.", flush=True)
                break
            continue
        failing = 0
        rec = finish(rec, seat, run_id, n, src)
        n += 1
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        s, t = rec["speedup"], rec["time_us_median"]
        if rec["verdict"] == "faster" and s is not None and (best is None or s > best):
            best, best_caps = s, caps
        if rec["verdict"] in ("faster", "slower", "no_gain") and s is not None and (timed is None or s > timed):
            timed, timed_caps = s, caps
        t_s = f"{t:.1f}us" if t is not None else "-"
        s_s = f"{s:.2f}x" if s is not None else "-"
        b_s = f"{best:.2f}x" if best is not None else "-"
        print(f"{mark}#{n - 1:<3} caps {caps_str(caps):<12} {rec['verdict']:<12} {t_s:>9}  {s_s:>6}  best {b_s}",
              flush=True)

    if referee is not None:
        referee.close()
    if best is None:
        print(f"{mark}best verified (faster): none"
              + (f"; best timed: caps {caps_str(timed_caps)}, {timed:.2f}x" if timed is not None else ""))
    else:
        print(f"{mark}best verified (faster): caps {caps_str(best_caps)} (runs as "
              f"{caps_str(effective(best_caps, tiles))}), {best:.2f}x")
    print(f"{mark}evaluated {n} of {len(order)}"
          + (f" ({len(skipped)} skipped: the referee failed, not the kernel: {skipped}; replaced from the same "
             f"seeded order where any were left)" if skipped else "")
          + f"; filtered: SBUF dropped {len(space) - len(fit)} of {len(space)}, {repeat} excluded as "
            f"attempt 0's twin")
    if stub:
        print(STUB_BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
