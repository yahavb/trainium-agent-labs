"""Run every kernel in kernels/{hand,broken,redteam} through kagent.harness and the
organizers' kernelbench.py, and report where the two verdicts disagree."""
import sys, glob, os, re, runpy, contextlib, io
import numpy as np
REPO, KB = sys.argv[1], sys.argv[2]
sys.path[:0] = [REPO, KB]
from kagent import harness
from kagent.levels import LEVELS
import kernelbench as kb

# Before v4 our L4/L8 kernels took a gain (and bias); kernelbench's take only x. Adapt only
# when the checked-out harness still uses the old signature.
_OLD = {
    4: lambda k: (lambda x: k(x, np.ones(x.shape[1], np.float32), 1e-6)),
    8: lambda k: (lambda x: k(x, np.ones(x.shape[1], np.float32), np.zeros(x.shape[1], np.float32), 1e-5)),
}
ADAPT = {n: f for n, f in _OLD.items() if ", g," in LEVELS[n].signature}
extra = sys.argv[3:]
files = sorted(glob.glob(f"{REPO}/kernels/*/l*.py")) + extra
rows = []
for f in files:
    n = int(re.match(r"l(\d+)_", os.path.basename(f)).group(1))
    src = open(f).read()
    with contextlib.redirect_stdout(io.StringIO()):
        dev = harness.verify(LEVELS[n], src)
        hold = harness.verify(LEVELS[n], src, holdout=True)
    ours = dev.passed and hold.passed
    rules = kb.check_rules(src, n)
    try:
        k = runpy.run_path(f)["kernel"]
        k = ADAPT.get(n, lambda k: k)(k)
        r = kb.verify(k, n, stop_early=False)
        kb_num, first = f"{r['passed']}/{r['total']}", (r["failures"][0][1].splitlines()[0] if r["failures"] else "")
        theirs = r["ok"] and not rules
    except Exception as e:
        kb_num, first, theirs = "load-err", str(e)[:80], False
    tag = "AGREE" if ours == theirs else ("OURS-PASS/KB-FAIL" if ours else "OURS-FAIL/KB-PASS")
    rows.append((tag, os.path.relpath(f, REPO) if f.startswith(REPO) else os.path.basename(f),
                 f"dev {dev.n_pass}/{len(dev.results)} hold {hold.n_pass}/{len(hold.results)} viol={len(dev.violations)+len(hold.violations)}",
                 f"kb {kb_num} rules={len(rules)}", first[:110]))
for r in sorted(rows, key=lambda r: r[0] == "AGREE"):
    print(" | ".join(r))
