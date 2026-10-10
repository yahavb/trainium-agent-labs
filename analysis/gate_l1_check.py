"""Run gate_nki's rules (89417dd) over logged kernels. New rules: partition_fixes + dst_shape_fixes
(the level-1 fix); old rules: whatever else fixes() returns. Writes /tmp/gate_check.json."""
import hashlib, json, sys
sys.path.insert(0, "/h/tal-deliv/projects/02-kernel-agent")
import gate_nki

H, R = "/h/tal-deliv", "/h/trainium-agent-labs/runs"
SOLVED = [  # (label, level, path)
    ("v7 round 2", 3, f"{R}/seat-117/Ev7_L3/Ev7_L3.jsonl"),
    ("v7 round 2", 4, f"{R}/seat-118/Ev7_L4/Ev7_L4.jsonl"),
    ("E-div", 3, f"{R}/seat-119/Ediv_L3/Ediv_L3.jsonl"),
    ("E-div", 4, f"{R}/seat-118/Ediv_L4/Ediv_L4.jsonl"),
    ("baseline", 2, f"{H}/analysis/logs/baseline/attempts.jsonl"),
    ("replica", 2, f"{H}/analysis/logs/replica_seat119/attempts.jsonl"),
    ("E-v3", 4, f"{H}/analysis/logs/ev3_L4/attempts_v3_L4.jsonl"),
    ("teammate 117", 2, f"{H}/analysis/logs/teammate_all_seat117/attempts.jsonl"),
]
sha = lambda c: hashlib.sha1(c.encode()).hexdigest()[:10]


def rules(code):
    new = gate_nki.partition_fixes(code) + gate_nki.dst_shape_fixes(code)
    allf = gate_nki.fixes(code)
    return len(new), len(allf) - len(new), [f[4][:90] for f in new]


out = dict(solved=[], l1=[])
for label, lv, p in SOLVED:
    seen = {}
    for n, line in enumerate(open(p), 1):
        r = json.loads(line)
        if r["level"] == lv and r["reward"] >= 1 - 1e-9:
            seen.setdefault(sha(r["code"]), (n, r["code"]))
    for s, (n, code) in seen.items():
        new, old, why = rules(code)
        out["solved"].append(dict(label=label, level=lv, file=p.replace("/h/", ""), line=n, sha=s,
                                  new=new, old=old, why=why))
p = f"{R}/seat-119/Ev7_L1/Ev7_L1.jsonl"
for n, line in enumerate(open(p), 1):
    r = json.loads(line)
    new, old, why = rules(r["code"])
    out["l1"].append(dict(line=n, run=r.get("run"), round=r["round"], reward=r["reward"], sha=sha(r["code"]),
                          new=new, old=old, why=why, feedback=r["feedback"][:120]))
json.dump(out, open("/tmp/gate_check.json", "w"), indent=1)
s = out["solved"]
print(f"solved kernels (distinct per source): {len(s)}; new rule fires on {sum(x['new'] > 0 for x in s)}; "
      f"old rules on {sum(x['old'] > 0 for x in s)}")
for x in s:
    print(f"  {x['label']:<13} L{x['level']} {x['sha']} line {x['line']:>3}: new {x['new']} old {x['old']} {x['why'][:1]}")
l1 = out["l1"]
fired = [x for x in l1 if x["new"]]
print(f"v7 L1 attempts: {len(l1)}; new rule fires on {len(fired)} ({len({x['sha'] for x in fired})} distinct kernels)")
for x in fired:
    print(f"  line {x['line']:>2} run {x['run']} round {x['round']} reward {x['reward']:.2f} {x['sha']}: {x['why'][0][:80]}")
