"""kopt.py -- the kernel optimisation loop. Small models only; the chip is the judge.

Three roles, as the track asks:

  workload    kernels/matmul_blocked.py, a matmul kernel with three block-size lines
  checker     reads the chip's profile of the latest attempt and writes ONE instruction
  optimiser   reads that instruction and rewrites the three lines

Both agents are the same <=8B model behind an OpenAI-compatible endpoint. Every attempt is compiled,
run on a NeuronCore, checked against NumPy and profiled (chip.py); nothing is scored on the simulator.

Arms, so the result is a comparison and not an anecdote:

  random     no model: an untried setting drawn uniformly from the legal grid
  raw        optimiser only, shown the raw profile numbers of every attempt so far
  template   optimiser + a fixed code-written sentence built from the profile (no checker model)
  checker    optimiser + the checker model's instruction
  checker3   the checker model picks one move from a code-built menu of untried single-setting changes,
             each annotated with what the measurements say about that direction
  checker4   checker3, plus the optimiser's edit is compared with the instruction before any chip time
             is spent and sent back if it changed the wrong line (built after reading the v3 log)
  checker5   checker4, with far moves added to the menu and the answer given by naming the move
  greedy5    greedy on checker5's menu
  greedy     no model: the first menu option whose direction has a faster-than-slower record, else a
             random menu option. The ablation for checker3 -- does the model choosing add anything?
  checker2   as checker, but the checker model is shown measured single-change comparisons and the
             legal values instead of raw profile lines (built after v1 lost to random; see evidence_text)

    python kopt.py --arm checker --runs 10 --attempts 10 --out runs/checker.jsonl

Measurements are cached by setting (results/grid_<size>.jsonl): the chip repeats to under 1%, so a
setting measured once is not measured again. Anything not in the cache is measured live.
"""

import argparse
import json
import os
import random
import re
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chip  # noqa: E402

KNOBS = chip.KNOBS

# What every model arm is told about the kernel. Deliberately says what each line DOES and not
# what value is good: the direction has to come from the measurement.
CARD = """The kernel multiplies a {M}x{K} matrix by a {K}x{N} matrix on an AI accelerator. It works in blocks. Three lines set the block sizes:

TILES_IN_BLOCK_M=<int>   how many 128-wide tiles of the left matrix are loaded per transfer
TILES_IN_BLOCK_N=<int>   how many 512-wide tiles of the right matrix are loaded per transfer
TILES_IN_BLOCK_K=<int>   how many 128-deep tiles of the shared dimension are kept on-chip and reused

Larger blocks mean fewer, bigger transfers and more reuse of data already on the chip, but they need more on-chip buffer, which is limited. M must divide {m_tiles}, N must divide {n_tiles}, K must divide {k_tiles}."""


# ---------------------------------------------------------------- measurement with a cache

class Bench:
    def __init__(self, size, cache_path, live=True):
        self.shape = chip.parse_shape(size)
        self.size, self.live, self.cache_path, self.cache = chip.shape_tag(self.shape), live, cache_path, {}
        self.tiles = (self.shape[1] // 128, self.shape[2] // 512, self.shape[0] // 128)   # M, N, K
        self.base = open(os.path.join(HERE, "kernels", "matmul_blocked.py")).read()
        if os.path.exists(cache_path):
            for line in open(cache_path):
                if line.strip():
                    r = json.loads(line)
                    self.cache[tuple(r["knobs"])] = r

    def legal(self):
        d = lambda n: [v for v in range(1, n + 1) if n % v == 0]
        return [(m, n, k) for m in d(self.tiles[0]) for n in d(self.tiles[1]) for k in d(self.tiles[2])]

    def measure(self, knobs):
        knobs = tuple(knobs)
        if knobs not in self.cache:
            if not self.live:
                raise KeyError(f"{knobs} is not in the cache and live measurement is off")
            rec = chip.measure(chip.set_knobs(self.base, *knobs), self.shape, reps=3)
            rec["knobs"] = list(knobs)
            self.cache[knobs] = rec
            open(self.cache_path, "a").write(json.dumps(rec) + "\n")
        return self.cache[knobs]


def stage_of(rec):
    if rec.get("time_us"):
        return "timed"
    return "numerics" if "WRONG RESULT" in (rec.get("error") or "") else "compile"


def short_error(rec):
    """The part of a failure a model can act on. The compiler's message is long; keep its reason."""
    e = rec.get("error") or ""
    if ("State buffer allocation failed" in e or "exceeds state buffer capacity" in e
            or "Allocated memory out of bound" in e):
        return "FAILED TO COMPILE: the blocks need more on-chip buffer than the chip has."
    m = re.search(r"assertion failed: (.*)", e) or re.search(r"AssertionError: (?!error)(.*)", e)
    if m:
        return f"FAILED TO COMPILE: {m.group(1)[:160]}"
    return ("FAILED: " + e.replace("\n", " "))[:220]


def profile_line(rec):
    """The raw numbers, one line. This is everything the `raw` arm and the checker model see."""
    if not rec.get("time_us"):
        return short_error(rec)
    return (f"time {rec['time_us']:.1f} microseconds; compute engine busy {100 * rec['te_busy']:.1f}%; "
            f"data movement busy {100 * rec['dma_busy']:.1f}%; {rec['bytes']:,} bytes moved "
            f"(minimum possible {rec['floor_bytes']:,}) in {rec['transfers']} transfers "
            f"averaging {rec['avg_transfer_bytes']:,.0f} bytes")


def knob_text(knobs):
    return "\n".join(f"{n}={v}" for n, v in zip(KNOBS, knobs))


def history_text(hist):
    return "\n".join(f"attempt {i + 1}: M={h['knobs'][0]} N={h['knobs'][1]} K={h['knobs'][2]} -> {profile_line(h['rec'])}"
                     for i, h in enumerate(hist))


# ---------------------------------------------------------------- the two agents

def ask(a, prompt, max_tokens):
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
                temperature=a.temperature, top_p=0.95, chat_template_kwargs={"enable_thinking": False})
    for attempt in range(3):
        try:
            r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body, timeout=180)
            r.raise_for_status()
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:  # the endpoint is shared by parallel runs; one retry is usually enough
            err = e
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"model endpoint failed three times: {err}")


def template_instruction(hist):
    """Code-written feedback from the latest profile. The ablation for the checker MODEL: same
    numbers, fixed sentences, no model."""
    rec = hist[-1]["rec"]
    if not rec.get("time_us"):
        return short_error(rec) + " Use smaller blocks than the last attempt."
    out = []
    if rec["bytes"] > 1.05 * rec["floor_bytes"]:
        out.append(f"The kernel moves {rec['bytes'] / rec['floor_bytes']:.2f}x the minimum bytes, so data is "
                   f"being loaded more than once. Keep more of it on the chip between uses.")
    if rec["dma_busy"] > rec["te_busy"]:
        out.append("Data movement is busier than the compute engine, so the engine waits for data.")
    if rec["transfers"] > 200:
        out.append(f"There are {rec['transfers']} separate transfers. Fewer, larger transfers cost less.")
    return " ".join(out) or "The compute engine is busy almost all the time. Little is left to gain."


def checker_instruction(a, hist, card):
    """The checker agent: profile in, one instruction out."""
    prompt = (f"{card}\n\nYou are the CHECKER. You do not write code. You read measurements taken on the real "
              f"chip and tell the engineer what to change next.\n\nAttempts so far (best time wins):\n"
              f"{history_text(hist)}\n\nThe latest attempt is the last line. Reply with ONE instruction of at "
              f"most two sentences: which of M, N or K to change, in which direction, and the reason taken from "
              f"the numbers. Do not repeat a setting that was already tried.")
    return ask(a, prompt, 120)


def evidence_text(hist, legal):
    """What the measurements say, worked out in code so the checker cannot argue past it.

    Written after checker v1 lost to random choice at the adapter shape. Its log showed the model
    repeating "increase N, fewer transfers" for ten attempts while every N it raised got slower: it
    was given the numbers and reasoned from a rule of thumb instead. So v2 is handed the comparison
    already made -- for each pair of attempts that differ in ONE setting, which way the time went --
    plus the legal values, because v1 also kept proposing 16 and 18 for a setting that must divide 24.
    It still says only what was measured. It never names a setting that has not been tried.
    """
    timed = [h for h in hist if h["rec"].get("time_us")]
    best = min(timed, key=lambda h: h["rec"]["time_us"])
    names = ("M", "N", "K")
    lines = []
    for j, name in enumerate(names):
        for a_ in timed:
            for b_ in timed:
                ka, kb = a_["knobs"], b_["knobs"]
                if ka[j] < kb[j] and all(ka[i] == kb[i] for i in range(3) if i != j):
                    ta, tb = a_["rec"]["time_us"], b_["rec"]["time_us"]
                    rest = ", ".join(f"{names[i]}={ka[i]}" for i in range(3) if i != j)
                    verdict = (f"{tb / ta:.2f}x SLOWER" if tb > ta * 1.01 else
                               f"{ta / tb:.2f}x faster" if ta > tb * 1.01 else "no real change")
                    lines.append((abs(tb / ta - 1), f"raising {name} from {ka[j]} to {kb[j]} (with {rest}): "
                                                    f"{ta:.0f} -> {tb:.0f} microseconds, {verdict}"))
    lines = [t for _, t in sorted(lines, reverse=True)[:8]]
    failed = [h for h in hist if not h["rec"].get("time_us")]
    out = [f"Best so far: M={best['knobs'][0]} N={best['knobs'][1]} K={best['knobs'][2]} at {best['rec']['time_us']:.0f} microseconds "
           f"(compute engine busy {100 * best['rec']['te_busy']:.0f}%)."]
    out.append("Measured effect of single changes so far:\n" + ("\n".join("- " + l for l in lines) if lines else
               "- none yet: no two attempts differ in exactly one setting"))
    if failed:
        out.append("Did not compile: " + "; ".join(f"M={h['knobs'][0]} N={h['knobs'][1]} K={h['knobs'][2]}" for h in failed[-4:]))
    out.append("Legal values -- M: " + " ".join(map(str, sorted({k[0] for k in legal}))) +
               " | N: " + " ".join(map(str, sorted({k[1] for k in legal}))) +
               " | K: " + " ".join(map(str, sorted({k[2] for k in legal}))))
    out.append("Already tried (M,N,K): " + " ".join(f"({h['knobs'][0]},{h['knobs'][1]},{h['knobs'][2]})" for h in hist))
    return "\n".join(out)


def checker2_instruction(a, hist, card, legal):
    """Checker v2: same model, but it is shown the measured comparisons instead of raw profiles."""
    prompt = (f"{card}\n\nYou are the CHECKER. You do not write code. You decide the next change from measurements "
              f"taken on the real chip.\n\n{evidence_text(hist, legal)}\n\nTrust the measured lines over any general "
              f"rule: if raising a setting made the kernel slower, lower it, and the other way round. Change ONE "
              f"setting of the best-so-far, to a legal value, so that the result is not in the already-tried list. "
              f"Reply in this form and nothing else:\nSet <M or N or K> to <value>. Reason: <one sentence citing a measured line, "
              f"or saying no measurement exists yet>.")
    return ask(a, prompt, 90)


def menu(hist, legal, far=False):
    """Checker v3's question to the model: a lettered list of single-setting moves from the best so far.

    v2 fixed the direction (it stopped pushing N up) but then tied itself in knots -- "raising N from 6
    to 12 made it slower, so lowering it back to 12" -- and re-proposed tried settings. So v3 removes
    what the model was bad at and keeps what it is for: code lists the moves that are legal and
    untried, each with the measured record for that direction, and the model CHOOSES. Each option is
    the nearest untried legal value below or above the best-so-far in one setting.

    far=True (checker v5) adds, per direction, the furthest untried legal value as well. The v4 log
    showed why: it walked N down 24, 12, 8, 6, 4, 3, 2, 1 with the evidence agreeing at every step,
    and from a distant start the ten attempts ran out before it arrived.
    """
    names = ("M", "N", "K")
    timed = [h for h in hist if h["rec"].get("time_us")]
    best = min(timed, key=lambda h: h["rec"]["time_us"])
    tried = {tuple(h["knobs"]) for h in hist}
    legal_set = set(legal)
    record = {}   # (knob, "raise"/"lower") -> [faster, slower, same]
    for j in range(3):
        up = [0, 0, 0]
        for a_ in timed:
            for b_ in timed:
                ka, kb = a_["knobs"], b_["knobs"]
                if ka[j] < kb[j] and all(ka[i] == kb[i] for i in range(3) if i != j):
                    ta, tb = a_["rec"]["time_us"], b_["rec"]["time_us"]
                    up[0 if tb < ta * 0.99 else 1 if tb > ta * 1.01 else 2] += 1
        record[(j, "raise")] = up
        record[(j, "lower")] = [up[1], up[0], up[2]]
    options = []
    for j in range(3):
        values = sorted({k[j] for k in legal})
        for direction, seq in (("lower", [v for v in reversed(values) if v < best["knobs"][j]]),
                               ("raise", [v for v in values if v > best["knobs"][j]])):
            open_ = [v for v in seq if tuple(v if i == j else best["knobs"][i] for i in range(3)) in legal_set - tried]
            picks = open_[:1] + (open_[-1:] if far and len(open_) > 1 else [])   # nearest, then furthest
            for v in picks:
                cand = tuple(v if i == j else best["knobs"][i] for i in range(3))
                f, sl, same = record[(j, direction)]
                n = f + sl + same
                ev = ("no measurement of this direction yet" if n == 0 else
                      f"measured so far: {'raising' if direction == 'raise' else 'lowering'} {names[j]} made it faster {f} of {n} times, slower {sl} of {n}")
                options.append(dict(knobs=cand, knob=names[j], value=v,
                                    text=f"Set {names[j]} to {v} ({direction} it from {best['knobs'][j]}) -- {ev}"))
    head = (f"Best so far: M={best['knobs'][0]} N={best['knobs'][1]} K={best['knobs'][2]} at "
            f"{best['rec']['time_us']:.0f} microseconds (compute engine busy {100 * best['rec']['te_busy']:.0f}%, "
            f"data movement busy {100 * best['rec']['dma_busy']:.0f}%).")
    return head, options


def checker3_instruction(a, hist, card, legal):
    """Checker v3: the model picks one move from a measured menu. Returns (instruction, chosen knobs)."""
    head, options = menu(hist, legal)
    if not options:
        return "", None
    letters = "ABCDEF"
    listing = "\n".join(f"{letters[i]}. {o['text']}" for i, o in enumerate(options))
    prompt = (f"{card}\n\nYou are the CHECKER. You decide the next change from measurements taken on the real chip.\n\n"
              f"{head}\n\nUntried moves:\n{listing}\n\nPick the ONE move most likely to make the kernel faster. Prefer a "
              f"direction the measurements show is faster; avoid one they show is slower; if nothing is measured for any "
              f"move, pick any. Reply with the letter, then one sentence of reason.")
    reply = ask(a, prompt, 70)
    m = re.match(r"\W*([A-F])\b", reply.strip())
    if not m or letters.index(m.group(1)) >= len(options):
        return reply, None
    o = options[letters.index(m.group(1))]
    return o["text"].split(" -- ")[0] + ". " + reply.strip()[:200], o["knobs"]


def checker5_instruction(a, hist, card, legal):
    """Checker v5: v4's menu with far moves, answered by NAMING the move instead of a letter.

    From the v4 log: "Raising N has historically made the kernel slower in all 15 previous attempts,
    so it's best to avoid this direction" -- and the letter it gave was the one for raising N. The
    reasoning was right and the letter was not, so the answer is now the move itself, which has to
    match a listed move or it is asked again.
    """
    head, options = menu(hist, legal, far=True)
    if not options:
        return "", None
    listing = "\n".join("- " + o["text"] for o in options)
    prompt = (f"{card}\n\nYou are the CHECKER. You decide the next change from measurements taken on the real chip.\n\n"
              f"{head}\n\nUntried moves:\n{listing}\n\nPick the ONE move most likely to make the kernel faster. Prefer a "
              f"direction the measurements show is faster, and when they agree strongly a bigger step is fine; avoid a "
              f"direction they show is slower; if nothing is measured for any move, pick any. Reply in exactly this form:\n"
              f"Set <M or N or K> to <value>. Reason: <one sentence>.")
    reply = ""
    for _ in range(2):
        reply = ask(a, prompt, 70)
        m = re.search(r"Set\s+([MNK])\s+to\s+(\d+)", reply)
        hit = [o for o in options if m and o["knob"] == m.group(1) and o["value"] == int(m.group(2))]
        if hit:
            return hit[0]["text"].split(" -- ")[0] + ". " + reply.strip()[:200], hit[0]["knobs"]
        prompt += "\n\nYour last reply did not name one of the listed moves. Copy one of them exactly."
    return reply, None


def apply_move(a, current, move, want, tries=3):
    """Optimiser for checker v4: make the one-line edit, and have the edit CHECKED before chip time.

    Found in the v3 log: told "Set K to 1" with the lines at M=2, N=24, K=2, the optimiser returned
    M=4, N=24, K=1 -- it changed a line it was not asked to touch, which produced a setting already
    tried, and it did the same on every retry of that attempt. An edit that does not match its
    instruction is a checker failure like any other, and it is the cheapest one to catch: compare
    before spending a compile. On a mismatch the optimiser is told exactly which line is wrong.
    Returns (knobs or None, reply, number of edits rejected).
    """
    prompt = (f"These three lines set a kernel's block sizes:\n\n{knob_text(current)}\n\nInstruction: {move}\n"
              f"Change only that one line. Reply with all three lines and nothing else.")
    reply, got = "", None
    for n in range(tries):
        reply = ask(a, prompt, 60)
        vals = [re.search(rf"{k}\s*=\s*(\d+)", reply) for k in KNOBS]
        got = tuple(int(v.group(1)) for v in vals) if all(vals) else None
        if got == tuple(want):
            return got, reply, n
        wrong = ("your reply did not contain the three lines" if got is None else
                 "; ".join(f"{k} should be {w} but you wrote {g}" for k, w, g in zip(KNOBS, want, got) if w != g))
        prompt = (f"These three lines set a kernel's block sizes:\n\n{knob_text(current)}\n\nInstruction: {move}\n"
                  f"Your last edit was rejected: {wrong}. Change only the one line the instruction names. "
                  f"Reply with all three lines and nothing else.")
    return None, reply, tries


def optimiser(a, card, current, instruction=None, hist=None):
    """The optimiser agent: rewrites the three lines. Sees the instruction, or the raw history."""
    parts = [card, "\nYou are the OPTIMISER. The three lines are currently:\n\n" + knob_text(current)]
    if hist is not None:
        parts.append("\nMeasurements on the real chip so far (lower time is better):\n" + history_text(hist))
    if instruction:
        parts.append("\nSettings already tried: " + "; ".join(
            f"M={h['knobs'][0]} N={h['knobs'][1]} K={h['knobs'][2]}" for h in instruction["hist"]))
        parts.append("\nInstruction from the checker:\n" + instruction["text"])
    parts.append("\nReply with the three lines only, with new values that have not been tried, to make the kernel faster.")
    reply = ask(a, "\n".join(parts), 80)
    vals = [re.search(rf"{k}\s*=\s*(\d+)", reply) for k in KNOBS]
    return (tuple(int(v.group(1)) for v in vals) if all(vals) else None), reply


# ---------------------------------------------------------------- one run

def run_once(a, bench, arm, run_id, log, start=(1, 1, 1)):
    rng = random.Random(a.seed * 1000 + run_id)
    card = CARD.format(K=bench.shape[0], M=bench.shape[1], N=bench.shape[2], m_tiles=bench.tiles[0],
                       n_tiles=bench.tiles[1], k_tiles=bench.tiles[2])
    hist = [dict(knobs=start, rec=bench.measure(start))]
    best = hist[0]
    pool = [k for k in bench.legal() if k != start]
    rng.shuffle(pool)

    def emit(attempt, knobs, stage, rec=None, instruction="", reply=""):
        row = dict(arm=arm, run=run_id, size=bench.size, start=list(start), attempt=attempt, knobs=list(knobs) if knobs else None,
                   stage_reached=stage, correct=bool(rec and rec.get("correct")),
                   time_us=rec.get("time_us") if rec else None, te_busy=rec.get("te_busy") if rec else None,
                   dma_busy=rec.get("dma_busy") if rec else None, bytes=rec.get("bytes") if rec else None,
                   transfers=rec.get("transfers") if rec else None, error=short_error(rec) if rec and not rec.get("time_us") else None,
                   instruction=instruction, optimiser_reply=reply, best_time_us_so_far=best["rec"]["time_us"])
        log.write(json.dumps(row) + "\n")
        log.flush()

    emit(0, start, "timed", hist[0]["rec"])
    for attempt in range(1, a.attempts + 1):
        instruction, reply = "", ""
        if arm == "random":
            knobs = pool.pop()
        elif arm in ("greedy", "greedy5"):
            _, options = menu(hist, bench.legal(), far=(arm == "greedy5"))
            if not options:
                knobs = pool.pop()
            else:
                def score(o):
                    m = re.search(r"faster (\d+) of (\d+) times, slower (\d+)", o["text"])
                    return (int(m.group(1)) - int(m.group(3))) if m else 0
                top = max(score(o) for o in options)
                pick = rng.choice([o for o in options if score(o) == top])
                knobs, instruction = pick["knobs"], pick["text"]
        else:
            if arm == "raw":
                knobs, reply = optimiser(a, card, best["knobs"], hist=hist)
            elif arm in ("checker4", "checker5"):
                instruction, want = (checker3_instruction if arm == "checker4" else checker5_instruction)(
                    a, hist, card, bench.legal())
                if want:
                    knobs, reply, rejected = apply_move(a, best["knobs"], instruction.split(". ")[0] + ".", want)
                    if knobs is None:
                        emit(attempt, None, "misapplied", instruction=instruction, reply=reply)
                        continue
                    if rejected:
                        instruction += f" [optimiser edit rejected {rejected}x before it matched]"
                else:
                    knobs = None
            elif arm == "checker3":
                instruction, want = checker3_instruction(a, hist, card, bench.legal())
                knobs, reply = (optimiser(a, card, best["knobs"], instruction=dict(text=instruction.split(". ")[0] + ".", hist=hist))
                                if want else (None, ""))
            else:
                instruction = (checker_instruction(a, hist, card) if arm == "checker" else
                               checker2_instruction(a, hist, card, bench.legal()) if arm == "checker2" else
                               template_instruction(hist))
            if arm not in ("raw", "checker3", "checker4", "checker5"):
                knobs, reply = optimiser(a, card, best["knobs"], instruction=dict(text=instruction, hist=hist))
        if knobs is None:
            emit(attempt, None, "unreadable", instruction=instruction, reply=reply)
            continue
        if any(tuple(h["knobs"]) == knobs for h in hist):
            emit(attempt, knobs, "duplicate", instruction=instruction, reply=reply)
            continue
        rec = bench.measure(knobs)
        hist.append(dict(knobs=knobs, rec=rec))
        if rec.get("time_us") and rec["time_us"] < best["rec"]["time_us"]:
            best = hist[-1]
        emit(attempt, knobs, stage_of(rec), rec, instruction, reply)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=("random", "raw", "template", "checker", "checker2", "checker3", "checker4", "checker5", "greedy", "greedy5"))
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--attempts", type=int, default=10, help="chip attempts per run, after the starting point")
    ap.add_argument("--size", default="2048", help="2048 for square, or K4096_M512_N12288")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--model", default=os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--base", default=os.environ.get("KOPT_BASE_URL", os.environ.get("KERNEL_AGENT_BASE_URL", "")))
    ap.add_argument("--cache", help="measurement cache (default results/grid_<size>.jsonl)")
    ap.add_argument("--no-live", action="store_true", help="fail instead of measuring a setting missing from the cache")
    ap.add_argument("--starts", default="worst", choices=("worst", "spread", "unblocked"),
                    help="worst: run i starts from the i-th slowest measured setting; unblocked: always 1,1,1")
    ap.add_argument("--nstarts", type=int, default=0, help="number of distinct starts (default: one per run)")
    ap.add_argument("--first-run", type=int, default=0, help="run ids to skip, to split one experiment over servers")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    bench = Bench(a.size, a.cache or os.path.join(HERE, "results", f"grid_{a.size}.jsonl"), live=not a.no_live)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    # Measured: the served model answers the same prompt almost identically each time (two runs from
    # 1,1,1 matched for seven attempts), so repeats from one start are not independent. Runs differ
    # by where they START instead: the slowest settings on the grid.
    slow = sorted((r for r in bench.cache.values() if r.get("time_us")), key=lambda r: -r["time_us"])
    n = a.nstarts or a.runs      # more runs than starts: the starts are cycled (for the no-model arms)
    if a.starts == "worst":
        starts = [tuple(r["knobs"]) for r in slow[:n]]
    elif a.starts == "spread":   # evenly through the slowest 90% of the grid, by rank
        starts = [tuple(slow[int(i * (len(slow) - 1) * 0.9 / max(n - 1, 1))]["knobs"]) for i in range(n)]
    else:
        starts = [(1, 1, 1)]
    finals = []
    with open(a.out, "a") as log:
        for run_id in range(a.first_run, a.runs):
            best = run_once(a, bench, a.arm, run_id, log, start=starts[run_id % len(starts)])
            finals.append(best["rec"]["time_us"])
            print(f"{a.arm} run {run_id}: best {best['rec']['time_us']:.1f} us at M,N,K={best['knobs']}", flush=True)
    finals.sort()
    print(f"{a.arm}: {len(finals)} runs, median {finals[len(finals) // 2]:.1f}, best {finals[0]:.1f}, worst {finals[-1]:.1f} us")


if __name__ == "__main__":
    main()
