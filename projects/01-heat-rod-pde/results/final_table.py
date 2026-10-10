"""Final table: original baseline (as shipped, seed 0) vs final Idea 2 (A+B, seed 0), per problem."""
import glob, json, re, statistics as st
os_ = __import__("os"); os_.chdir("/workspace/projects/01-heat-rod-pde")

# as shipped: attempts log rows 0-27 are the first full run (level 0.1, then level 1 all); timings from run.log
ship = [json.loads(l) for l in open("results/asshipped-attempts.jsonl")][:28]
times = {}
cur = None
for line in open("results/asshipped-run.log"):
    m = re.match(r"=+ (level\S+) =+", line)
    if m: cur = m.group(1)
    m = re.match(r"round (\d+):.*\(([\d.]+)s\)", line)
    if m and cur: times.setdefault(cur, 0.0); times[cur] += float(m.group(2))
base = {}
for p in sorted({r["problem"] for r in ship}):
    rs = [r for r in ship if r["problem"] == p]
    solved = [r["round"] for r in rs if r["reward"] == 1.0]
    base[p] = dict(best=max(r["reward"] for r in rs), mean=st.mean(r["reward"] for r in rs),
                   rounds=str(min(solved) + 1) if solved else f">{max(r['round'] for r in rs) + 1}",
                   tokens=None, wall=times.get(p))

final = {}
for f in sorted(glob.glob("results/FINAL-2*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    for p in sorted({r["problem"] for r in rows if r["kind"] == "cand"}):
        rs = [r for r in rows if r["kind"] == "cand" and r["problem"] == p and r.get("source", "model") == "model"]
        rd = [r for r in rows if r["kind"] == "round" and r["problem"] == p]
        solved = [r["round"] for r in rs if r["reward"] == 1.0]
        final[p] = dict(best=max(r["reward"] for r in rs), mean=st.mean(r["reward"] for r in rs),
                        rounds=str(min(solved) + 1) if solved else f">{max(r['round'] for r in rs) + 1}",
                        tokens=sum(r.get("completion_tokens", 0) for r in rs), wall=sum(r["wall_s"] for r in rd))

def cell(d, k, fmt):
    if d is None: return "not run"
    v = d.get(k)
    return "n/a" if v is None else fmt(v)

print("| problem | baseline best | final best | baseline mean | final mean | baseline rounds | final rounds | final tokens | baseline time | final time |")
print("|---|---|---|---|---|---|---|---|---|---|")
for p in sorted(set(base) | set(final)):
    b, f = base.get(p), final.get(p)
    print(f"| {p} | {cell(b,'best',lambda v:f'{v:.1f}')} | {cell(f,'best',lambda v:f'{v:.1f}')} | "
          f"{cell(b,'mean',lambda v:f'{v:.2f}')} | {cell(f,'mean',lambda v:f'{v:.2f}')} | "
          f"{cell(b,'rounds',str)} | {cell(f,'rounds',str)} | {cell(f,'tokens',str)} | "
          f"{cell(b,'wall',lambda v:f'{v:.0f} s')} | {cell(f,'wall',lambda v:f'{v:.0f} s')} |")
