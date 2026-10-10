"""E2E rate: per run, a FRESH unstaged plan from Qwen -> loop_tool staged -> stock nkibench check."""
import argparse, ast, json, os, re, subprocess, sys
sys.path.insert(0, "/workspace/projects/02-kernel-agent")
os.chdir("/workspace/projects/02-kernel-agent")
import plan_check, plan_stage


def canon(x):
    if isinstance(x, str):
        return ast.unparse(ast.parse(x, mode="eval"))
    if isinstance(x, list):
        return [canon(y) for y in x]
    if isinstance(x, dict):
        return {k: canon(v) for k, v in x.items()}
    return x


ap = argparse.Namespace(offline=False, rounds=8, wording="steps", give_up_after=0, plan_max_tokens=800,
                        plan_out=None, context=8192, model=plan_stage.agent.MODEL, think=False,
                        max_tokens=800, base=os.environ["KERNEL_AGENT_BASE_URL"].strip().rstrip("/"))
start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
summary = []
for run in range(start, start + n):
    print(f"\n######## e2e run {run} ########", flush=True)
    with open("attempts-plan-stage-e2e-rate.jsonl", "a") as log:
        out = plan_stage.run_whole(ap, log, run)
    row = dict(run=run, plan_rounds=out["rounds"]["all"], plan_ok=out["plan_ok"])
    if not out["plan_ok"]:
        row.update(solved=False, why="no plan")
        summary.append(row); print(row, flush=True); continue
    plan = out["plan"]
    row["equals_reference"] = canon(plan) == canon(plan_check.REFERENCE)
    pf = f"plan-e2e-rate-run{run}.json"
    json.dump(plan, open(pf, "w"), indent=1)
    lt_log = f"attempts-plan-stage-e2e-rate-body-run{run}.jsonl"
    p = subprocess.run([sys.executable, "loop_tool.py", "--given-plan", pf, "--staged", "--wording", "steps",
                        "--structure-check", "--rounds", "8", "--samples", "1", "--context", "8192",
                        "--repeat", "1", "--log", lt_log], capture_output=True, text=True)
    open(f"run-e2e-rate-body-run{run}.log", "w").write(p.stdout + p.stderr)
    m = re.search(r"reward ([\d.]+)", p.stdout)
    row["body_reward"] = float(m.group(1)) if m else None
    rows = [json.loads(l) for l in open(lt_log)]
    row["body_rounds"] = len(rows)
    row["body_rounds_by_stage"] = {s: sum(r.get("staged") == s for r in rows) for s in ("A", "B")}
    good = [r for r in rows if r.get("staged") == "B" and r["reward"] >= 1 - 1e-9]
    if not good:
        row.update(solved=False, why="body")
        summary.append(row); print(row, flush=True); continue
    kf = f"plan-e2e-rate-kernel-run{run}.py"
    open(kf, "w").write(good[0]["code"])
    c = subprocess.run([sys.executable, "nkibench.py", "--level", "4", "--check", kf],
                       capture_output=True, text=True)
    txt = c.stdout + c.stderr
    rules = re.search(r"rules\s+(\S+)", txt)
    num = re.search(r"numerics\s+(\d+/\d+)", txt)
    row.update(nkibench_rules=rules and rules.group(1), nkibench_numerics=num and num.group(1))
    row["solved"] = bool(rules and rules.group(1) == "clean" and num and num.group(1) == "4/4")
    summary.append(row); print(row, flush=True)
print("\nSUMMARY")
for r in summary:
    print(json.dumps(r))
print(f"e2e solved {sum(r['solved'] for r in summary)}/{len(summary)} (this batch)")
