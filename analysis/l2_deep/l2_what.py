"""What do level-2 kernels that score 0.50 (or anything below 1 that runs) actually compute? Run each distinct
kernel on level 2's loop shapes with arange inputs (every element distinct) and match its output against
named wrong transforms. Writes /tmp/l2_what.json."""
import collections, hashlib, io, json, os, sys
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
sys.path.insert(0, "/h/tal-deliv/projects/02-kernel-agent")
import nkibench

R = "/h/trainium-agent-labs/runs"; H = "/h/tal-deliv/analysis/logs"
SRC = {"baseline": [f"{H}/baseline/attempts.jsonl", f"{H}/replica_seat119/attempts.jsonl"],
       "v7": [f"{R}/seat-116/Ev7_L2/Ev7_L2.jsonl"],
       "v8.1": [f"{R}/seat-116/v81_partial/v81_L2.jsonl"],
       "v8.2": [f"{R}/seat-116/v82/v82_L2_s116.jsonl", f"{R}/seat-119/v82/v82_L2_s119.jsonl"]}


def transforms(x, s2):
    P, F = x.shape; A, B = s2
    out = {"correct": x.reshape(P, A, B).transpose(0, 2, 1).reshape(P, F),
           "identity (x unchanged)": x.copy(),
           "block read as BxA (A and B swapped)": x.reshape(P, B, A).transpose(0, 2, 1).reshape(P, F)}
    if P * F == F * P:
        out["whole x transposed, flattened back"] = x.T.reshape(P, F)
    return out


def classify(got, x, s2):
    got = np.asarray(got)
    if got.shape != x.shape:
        if got.size == x.size and np.array_equal(got.ravel(), x.T.ravel()):
            return "whole x transposed (wrong shape)"
        return f"wrong shape {got.shape}"
    if np.all(got == 0):
        return "all zeros"
    for name, t in transforms(x, s2).items():
        if np.array_equal(got, t):
            return name
    c = transforms(x, s2)["correct"]
    frac = float((got == c).mean())
    rows_ok = int(np.all(got == c, axis=1).sum())
    return f"partly right ({frac:.0%} of elements, {rows_ok}/{x.shape[0]} rows)"


sh = lambda c: hashlib.sha1(c.encode()).hexdigest()[:10]
kern = {}          # sha -> dict(code, uses=[(version, file, line, run, round, reward)])
for ver, files in SRC.items():
    for f in files:
        for n, line in enumerate(open(f), 1):
            r = json.loads(line)
            if r.get("level") != 2 or r["reward"] >= 1 - 1e-9 or not r["parts"].get("runs"):
                continue
            k = kern.setdefault(sh(r["code"]), dict(code=r["code"], uses=[]))
            k["uses"].append((ver, os.path.basename(f), n, r.get("run"), r["round"], round(r["reward"], 2)))
print("distinct level-2 kernels that run but are not correct:", len(kern))
spec = nkibench.LEVELS[2]
res = {}
for s, k in kern.items():
    path = nkibench.candidate_path(f"_l2what_{s}")
    open(path, "w").write(k["code"])
    out = []
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            kernel = nkibench.load_kernel(path, spec["entry"])
    except Exception as e:
        res[s] = dict(uses=k["uses"], shapes=[f"load error {type(e).__name__}"]); continue
    for case in spec["shapes"]:
        P, F = case["shape"]; x = np.arange(P * F, dtype=np.float32).reshape(P, F)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                got, _ = nkibench.simulate_and_count(kernel, (x, case["shape2D"]))
            out.append(classify(got, x, case["shape2D"]))
        except Exception as e:
            out.append(f"raises {type(e).__name__}")
    res[s] = dict(uses=k["uses"], shapes=out)
json.dump(res, open("/tmp/l2_what.json", "w"), indent=1)
for s, v in sorted(res.items(), key=lambda kv: -len(kv[1]["uses"])):
    vers = collections.Counter(u[0] for u in v["uses"])
    print(s, dict(vers), "|", " ; ".join(v["shapes"]))
