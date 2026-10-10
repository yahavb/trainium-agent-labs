"""Second stage: make a verified kernel cheaper without ever making it wrong.

The harness already keeps an efficiency ledger (FLOPs, instructions, HBM bytes, each against the
op's minimum). This loop turns the worst excess into ONE optimisation directive, asks the model for
a rewrite, and accepts it only if it still passes dev AND holdout with no rule violation AND is
measurably cheaper. Correctness never regresses: a rejected rewrite leaves the best kernel in place.

Lives outside kagent/ so the correctness benchmark's code version is unchanged.

    python3 scripts/optimize.py --from runs/ours-L1234-seat-130.jsonl --rounds 4 --out runs/opt.jsonl
    python3 scripts/optimize.py --kernel 5=kernels/hand/l5_softmax.py --rounds 4
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kagent.agent import LiveModel, ScriptedModel, account, render  # noqa: E402
from kagent.extract import extract_code  # noqa: E402
from kagent.harness import verify  # noqa: E402
from kagent.levels import LEVELS  # noqa: E402
from kagent.rules import TILE_C, TILE_R  # noqa: E402
from roofline import measure  # noqa: E402  (scripts/ is on sys.path when run as a script)

TILE = TILE_R * TILE_C
# accept a rewrite only if roofline time (scripts/roofline.py) drops by IMPROVE; these ops are
# memory bound, so trading FLOPs for fewer passes (online softmax) is a win
IMPROVE = 0.05      # a rewrite must cut roofline time by at least 5%
MAX_OPS_GROWTH = 2.0  # instruction count is mostly an artefact of the column-loop rules: reported, capped

# v2 (E16): an optimisation directive gets the same treatment as a repair directive: one concrete,
# rule-safe change with the code to write, aimed at a diagnosed pattern, and never a target the
# tile rules make impossible. Abstract hints ("fewer instructions") made the model reach for
# np.sum/np.max (6 of 13 rewrites broke a rule).
ROW_REDUCE = {"row_sum": ("acc = np.zeros(i1 - i0, dtype=np.float64)", "acc += x[i0:i1, j]"),
              "row_max": ("acc = np.full(i1 - i0, -np.inf, dtype=np.float32)",
                          "acc = np.maximum(acc, x[i0:i1, j])")}
ROW_NORM = {"rmsnorm", "softmax", "layernorm"}
NO_REDUCTIONS = "Keep reductions as explicit loops over columns; do not call np.sum, np.max or np.mean."

ONLINE_SOFTMAX = (
    "Use an online softmax so x is read twice instead of three times. For each row tile keep "
    "m = np.full(i1 - i0, -np.inf) and s = np.zeros(i1 - i0, dtype=np.float64). In the FIRST pass "
    "over the columns j do: col = x[i0:i1, j].astype(np.float64); m_new = np.maximum(m, col); "
    "s = s * np.exp(m - m_new) + np.exp(col - m_new); m = m_new. In the SECOND pass write "
    "out[i0:i1, j] = np.exp(x[i0:i1, j].astype(np.float64) - m) / s. Delete the separate max pass.")


def better(new, old):
    """Roofline time down by IMPROVE (instructions may grow, within reason), or time unchanged and
    instructions down by IMPROVE (per-element loops -> column vectors)."""
    faster = new["time_x"] <= old["time_x"] * (1 - IMPROVE) and new["ops"] <= old["ops"] * MAX_OPS_GROWTH
    leaner = new["time_x"] <= old["time_x"] * 1.01 and new["ops"] <= old["ops"] * (1 - IMPROVE)
    return faster or leaner


def _case(level, name):
    return next(c for c in level.get_cases() if c.name == name)


def read_floor(level, m):
    """The fewest input reads the tile rules allow. A row-wise normalisation needs the whole row's
    statistic before it can write, and a row wider than one tile cannot be held, so x is read twice;
    a tiled matmul re-reads a once per output column tile and b once per output row tile."""
    args = _case(level, m["case"]).args
    if level.name in ROW_NORM:
        return 1.0 if args[0].shape[1] <= TILE_C else 2.0
    if level.name == "matmul":
        a, b = args
        n_tiles, m_tiles = -(-b.shape[1] // TILE_C), -(-a.shape[0] // TILE_R)
        return (a.nbytes * n_tiles + b.nbytes * m_tiles) / (a.nbytes + b.nbytes)
    if level.name in ("band_attention", "conv1d"):
        return m["reads_x"]          # no closed form here: do not ask for the impossible
    return 1.0


def directives(level, m):
    """Diagnosed inefficiencies, most costly first, each with the code to write."""
    out = []
    if m["elems_per_op"] < 4:
        head = (f"It issues {m['ops']} instructions of about {m['elems_per_op']:.0f} element each: "
                f"it loops over single elements x[r, c]. Work on column vectors instead: ")
        if level.name in ROW_REDUCE:
            init, step = ROW_REDUCE[level.name]
            out.append(("per-element", head +
                        f"for each row tile i0:i1 start `{init}`, then for each column j do "
                        f"`{step}` (one instruction for that column of the whole tile), and after "
                        f"the column loop write `out[i0:i1] = acc`. " + NO_REDUCTIONS))
        elif level.name in ROW_NORM:
            out.append(("per-element", head +
                        "keep each per-row statistic as a vector of length i1 - i0 and update it "
                        "with whole columns, e.g. `col = x[i0:i1, j].astype(np.float64)`, then "
                        "write the output one column at a time, e.g. "
                        "`out[i0:i1, j] = x[i0:i1, j] * rinv * g[j]`. " + NO_REDUCTIONS))
        elif level.name == "relu_affine":
            out.append(("per-element", head +
                        "apply it to the whole tile at once: "
                        "`out[i0:i1, j0:j1] = np.maximum(x[i0:i1, j0:j1] * a + b, 0.0)`."))
        else:
            out.append(("per-element", head + "operate on whole column slices x[i0:i1, j] and "
                                              "whole tiles where the work is elementwise. "
                        + NO_REDUCTIONS))
    floor = read_floor(level, m)
    if m["reads_x"] > floor * (1 + IMPROVE):
        hint = (ONLINE_SOFTMAX if level.name == "softmax" else
                "Fuse the passes that read the same tiles, and keep values you already hold "
                "instead of reloading them.")
        out.append(("reads", f"It loads each input element {m['reads_x']:.2f} times from HBM; "
                             f"{floor:.2f} is possible under the tile rules. " + hint))
    if (m["flops_x"] is not None and m["flops_x"] > 1 + IMPROVE and level.name != "softmax"):
        # softmax: the online form trades FLOPs for a pass over x, which is the win
        out.append(("flops", f"It performs {m['flops_x']:.2f}x the minimal FLOPs: some values are "
                             f"computed more than once. Compute each value once and reuse it."))
    if level.name == "relu_affine" and m["elems_per_op"] >= 4 and m["elems_per_op"] < TILE / 32:
        out.append(("ops", f"It issues {m['ops']} instructions of about {m['elems_per_op']:.0f} "
                           f"elements; one tile holds {TILE}. Apply the elementwise work to whole "
                           f"tiles: `out[i0:i1, j0:j1] = np.maximum(x[i0:i1, j0:j1] * a + b, 0.0)`."))
    return out


def opt_prompt(level, code, m, text):
    return [("task", f"This numpy function `{level.signature}` is correct and must stay correct: "
                     f"{level.task} Tiles are at most {TILE_R} rows x {TILE_C} columns."),
            ("code", f"```python\n{code}```"),
            ("feedback", f"Efficiency on the largest test ({m['case']}): {text}"),
            ("task", "Rewrite it to be cheaper; you may restructure the loops as needed. Reply with the complete "
                     "function in one ```python code block.")]


@dataclass
class Step:
    level: int
    source: str
    round: int
    target: str
    outcome: str        # accepted | not-cheaper | wrong | holdout-fail | violation | no-code
    before: dict
    after: dict | None
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    code: str | None = None


def optimise(level, code, model, rounds, source, log, say=print):
    if not verify(level, code).passed:
        say(f"L{level.num} {source}: input kernel does not pass; skipped")
        return None
    best, m = code, measure(level, code)
    first = dict(m)
    say(f"L{level.num} {source}: start  time {m['time_x']:.2f}x min  reads {m['reads_x']:.2f}x  "
        f"flops {(m['flops_x'] or 0):.2f}x  intensity {m['intensity']:.2f} FLOP/B  ops {m['ops']}")
    tried = set()
    for rnd in range(1, rounds + 1):
        todo = [d for d in directives(level, m) if d[0] not in tried]
        if not todo:
            say("   nothing left to optimise" if not tried else "   every direction tried")
            break
        target, text = todo[0]
        parts = opt_prompt(level, best, m, text)
        account(model, parts)
        r = model.complete(render(parts), f"opt:{source}:{rnd}")
        new = extract_code(r.content) if r.finish_reason == "stop" else None
        after, outcome = None, "no-code"
        if new:
            nrep = verify(level, new)
            if nrep.violations:
                outcome = "violation"
            elif not nrep.passed:
                outcome = "wrong"
            elif not verify(level, new, holdout=True).passed:
                outcome = "holdout-fail"
            else:
                after = measure(level, new)
                outcome = "accepted" if better(after, m) else "not-cheaper"
        log.write(json.dumps(asdict(Step(level.num, source, rnd, target, outcome, m, after,
                                         r.prompt_tokens, r.completion_tokens, r.latency_s,
                                         new))) + "\n")
        log.flush()
        say(f"   #{rnd} target={target:5s} -> {outcome}"
            + (f"  time {after['time_x']:.2f}x reads {after['reads_x']:.2f}x flops "
               f"{(after['flops_x'] or 0):.2f}x ops {after['ops']}" if after else ""))
        if outcome == "accepted":
            best, m = new, after
        else:
            tried.add(target)
    say(f"   final: time {first['time_x']:.2f}x -> {m['time_x']:.2f}x of the roofline minimum "
        f"(reads {first['reads_x']:.2f}x -> {m['reads_x']:.2f}x, flops {(first['flops_x'] or 0):.2f}x "
        f"-> {(m['flops_x'] or 0):.2f}x, ops {first['ops']} -> {m['ops']})")
    return first, m, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", help="agent JSONL: optimise every verified kernel in it")
    ap.add_argument("--kernel", action="append", default=[], help="LEVEL=path to a kernel file")
    ap.add_argument("--levels", type=int, nargs="*", help="only these levels")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--scripted", help="JSON list of replies, for testing without a model")
    ap.add_argument("--out", default=f"runs/opt-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    a = ap.parse_args()

    jobs = []
    if a.src:
        for line in Path(a.src).read_text().splitlines():
            r = json.loads(line)
            if r.get("type") == "result" and r["status"] == "verified":
                jobs.append((r["level"], r["code"], f"run{r['run']}"))
    for spec in a.kernel:
        n, path = spec.split("=", 1)
        jobs.append((int(n), Path(path).read_text(), Path(path).stem))
    if a.levels:
        jobs = [j for j in jobs if j[0] in a.levels]

    model = ScriptedModel(json.loads(Path(a.scripted).read_text())) if a.scripted else LiveModel()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as log:
        for n, code, source in jobs:
            optimise(LEVELS[n], code, model, a.rounds, source, log)
    print(f"log: {a.out}")


if __name__ == "__main__":
    main()
