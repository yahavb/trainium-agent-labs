"""Pick the submission kernel for each level: every kernel the agent verified in any run (plus
optimiser-accepted rewrites), re-checked with the organizers' kernelbench.py; among those that pass,
the one with the lowest roofline time, then the fewest instructions. Writes kernels/final/ and
FINAL.md.

    KERNELBENCH_DIR=/path/to/02-kernel-agent uv run python scripts/final_kernels.py
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from kagent import judge  # noqa: E402
from kagent.harness import verify  # noqa: E402
from kagent.levels import LEVELS  # noqa: E402
from roofline import measure  # noqa: E402

if not judge.available():
    sys.exit("organizers' kernelbench.py not found: set KERNELBENCH_DIR")

cands = []
for path in sorted(glob.glob(str(ROOT / "results/**/*.jsonl"), recursive=True)):
    for line in open(path):
        r = json.loads(line)
        src = None
        if r.get("type") == "result" and r["status"] in ("verified", "dev-only") and r.get("code"):
            src, lv, tag = r["code"], r["level"], f"{r.get('mode', '?')} v{r.get('version', '?')[:6]} run{r['run'] + 1}"
        elif (r.get("outcome") == "accepted" and r.get("code")
              and r["source"].startswith("run")):     # optimiser rewrites of agent-written kernels only
            src, lv, tag = r["code"], r["level"], f"agent run{int(r['source'][3:]) + 1}, then optimised"
        if src:
            cands.append((lv, tag, Path(path).relative_to(ROOT).as_posix(), src))

best, tried, seen = {}, {}, set()
for lv, tag, path, src in cands:
    if (lv, src) in seen:          # repeats of one prompt are often byte-identical (notes E27)
        continue
    seen.add((lv, src))
    # both checkers: ours (rules incl. the PR #10 column-slice fix, dev and holdout) and theirs
    if not (verify(LEVELS[lv], src).passed and verify(LEVELS[lv], src, holdout=True).passed):
        tried.setdefault(lv, [0, 0])[0] += 1
        continue
    ok, msg = judge.check(lv, src)
    tried.setdefault(lv, [0, 0])
    tried[lv][0] += 1
    if not ok:
        continue
    tried[lv][1] += 1
    try:
        m = measure(LEVELS[lv], src)
    except TypeError:      # our harness cannot run it (e.g. pre-v4 signature): cost unknown
        continue
    key = (round(m["time_x"], 2), m["ops"])
    if lv not in best or key < best[lv][0]:
        best[lv] = (key, tag, path, src, m, msg)

out = ROOT / "kernels/final"
out.mkdir(parents=True, exist_ok=True)
for old in out.glob("l*.py"):
    old.unlink()
rows = ["# Final kernels", "",
        "One kernel per level, chosen by `scripts/final_kernels.py`: written by the agent, passing our current",
        "harness (rules, dev and holdout) and the organizers' `kernelbench.py`; among those, the lowest roofline time, then",
        "the fewest instructions. Kernels seeded from our hand-written references (e.g. the optimised",
        "softmax of notes E19) are excluded: only kernels the agent wrote count.", "",
        "| level | kernel file | from | organizers' checker | roofline time | instructions | candidates passing / checked |",
        "|---|---|---|---|---|---|---|"]
for lv in sorted(LEVELS):
    n_t, n_ok = tried.get(lv, [0, 0])
    if lv not in best:
        rows.append(f"| L{lv} {LEVELS[lv].name} | — | none passes yet | — | — | — | {n_ok}/{n_t} |")
        continue
    _, tag, path, src, m, msg = best[lv]
    name = f"l{lv}_{LEVELS[lv].name}.py"
    (out / name).write_text(f"# L{lv} {LEVELS[lv].name}: {tag} ({path}); {msg}\n" + src)
    rows.append(f"| L{lv} {LEVELS[lv].name} | `kernels/final/{name}` | {tag} | {msg.split(': ')[-1]} | "
                f"{m['time_x']:.2f}x | {m['ops']} | {n_ok}/{n_t} |")
(ROOT / "FINAL.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
