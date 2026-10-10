import io, json, sys, hashlib, collections
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
sys.path.insert(0, "/h/tal-deliv/projects/02-kernel-agent"); sys.path.insert(0, "/h/tal-deliv/scripts")
import nkibench, taxonomy
res = json.load(open("/tmp/l2_what.json"))
R = "/h/trainium-agent-labs/runs"; H = "/h/tal-deliv/analysis/logs"
files = [f"{H}/baseline/attempts.jsonl", f"{H}/replica_seat119/attempts.jsonl", f"{R}/seat-116/Ev7_L2/Ev7_L2.jsonl",
         f"{R}/seat-116/v81_partial/v81_L2.jsonl", f"{R}/seat-116/v82/v82_L2_s116.jsonl", f"{R}/seat-119/v82/v82_L2_s119.jsonl"]
codes = {}
for f in files:
    for l in open(f):
        r = json.loads(l); codes.setdefault(hashlib.sha1(r["code"].encode()).hexdigest()[:10], r["code"])
def cat(s):
    path = nkibench.candidate_path(f"_cat_{s}"); open(path, "w").write(codes[s])
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            k = nkibench.load_kernel(path, "tensor_transpose2D_kernel_")
            P, F, A, B = 32, 12, 3, 4
            x = np.arange(P * F, dtype=np.float32).reshape(P, F)
            got, _ = nkibench.simulate_and_count(k, (x, (A, B)))
    except Exception as e:
        return "raises on (32,12) as 3x4"
    got = np.asarray(got, dtype=np.float64)
    if got.shape != x.shape: return "wrong shape"
    if np.array_equal(got, x): return "identity: x returned unchanged"
    ok = np.isfinite(got) & (np.abs(got) < 1e6)
    src = np.where(ok, got - np.arange(P)[:, None] * F, -1)
    kk = np.arange(F)
    if not ok.all() and ok.any(): return "only part of the output written (rest uninitialised)"
    if np.all(src == kk // B): return "one loop variable dropped: out[i*B+j] = x[i]"
    if np.all(src == kk % A): return "one loop variable dropped: out[k] = x[k mod A]"
    return "other permutation"
out = collections.defaultdict(lambda: collections.Counter())
runs = collections.defaultdict(set)
for s, v in res.items():
    c = cat(s)
    for ver, f, n, run, rnd, rw in v["uses"]:
        if abs(rw - 0.5) < 1e-6:
            out[c][ver] += 1; runs[c].add((ver, f, run))
print("0.50 attempts by what they compute:")
for c, vs in sorted(out.items(), key=lambda kv: -sum(kv[1].values())):
    print(f"  {sum(vs.values()):>3}  {c}  {dict(vs)}  runs={len(runs[c])}")
json.dump({c: dict(v) for c, v in out.items()}, open("/tmp/l2_cat.json", "w"))
# the failed v8.2 run on seat-116
rows = [json.loads(l) for l in open(f"{R}/seat-116/v82/v82_L2_s116.jsonl")]
eps = taxonomy.load_attempts([f"{R}/seat-116/v82/v82_L2_s116.jsonl"])
for (fi, run, lv), a in eps.items():
    best = max(r["reward"] for r in a)
    rounds = collections.OrderedDict()
    for r in a: rounds.setdefault(r["round"], []).append(r)
    print(f"\nseat-116 v8.2 run {run+1}: best {best:.2f}, rounds {len(rounds)}")
    if best < 1:
        for rd, rs in rounds.items():
            sc = [round(r["reward"], 2) for r in rs]
            top = max(rs, key=lambda r: r["reward"])
            print(f"  r{rd} samples {sc} (s1=repair, s2-4 fresh) | carried s{rs.index(top)+1}: {taxonomy.error_line(top['feedback'])[:150]}")
