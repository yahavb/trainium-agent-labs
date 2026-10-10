"""The checker: decides whether a decode-megakernel candidate is accepted, and says WHY when it is not.

Gates, cheapest first. Each prints PASS/FAIL with a reason that names the change to make, not just a
number. A failing gate stops the run unless --keep-going.

  layout    float32 torch reference vs HuggingFace Qwen3 (tiny config). If this fails, every later
            comparison is meaningless: the reference itself has the wrong layout or convention.
  tiny      layer kernel vs the reference on a tiny config, against the error bf16 rounding alone
            produces. On failure, the kernel output is compared with deliberately wrong references
            (interleaved RoPE, no QK-norm, mask counting the active token twice, no RoPE, attention
            switched off) and the closest one is reported: "your kernel implements X".
  head      embedding row gather (must be bit-exact) and LM head + argmax at the real vocabulary size.
  real      real Qwen3-8B weights, all 36 layers, greedy, teacher-forced against HF float32: every step
            where the kernel's argmax differs from HF's must be explained by bf16 noise (HF's own margin
            between the two tokens below 3 sigma of the kernel's logit error at that step).
  speed     full decode step, one launch vs per-op torch.compile of the same math on the same core.
            Rejected if not faster than per-op; the HBM floor is reported for context.

  NEURON_RT_VISIBLE_CORES=2 python check.py                      # all gates (~25 min cold, ~5 min cached)
  NEURON_RT_VISIBLE_CORES=2 python check.py --gates layout,tiny,head
  NEURON_RT_VISIBLE_CORES=2 python check.py --layer-kernel qwen3_decode_layers_sbuf --gates tiny
"""
import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
K = os.path.join(HERE, "kernels")
R = os.path.join(HERE, "results")
ap = argparse.ArgumentParser()
ap.add_argument("--gates", default="layout,tiny,head,real,speed")
ap.add_argument("--layer-kernel", default="qwen3_decode_layers", help="function in kernels/qwen3_megakernel.py")
ap.add_argument("--resident", action="store_true", help="check qwen3_decode_step_resident in the real/speed gates")
ap.add_argument("--control", choices=["mask_active_zero", "no_rope_input"], default=None,
                help="negative control for the tiny gate: inject a known bug into the kernel's inputs; the gate must FAIL and name it")
ap.add_argument("--keep-going", action="store_true")
ap.add_argument("--out", default=os.path.join(R, "check_log.jsonl"), help="every check run is appended here")
args = ap.parse_args()
env = dict(os.environ)
env.setdefault("NEURON_RT_VISIBLE_CORES", "2")


def run(cmd, timeout=3600):
    t0 = time.time()
    p = subprocess.run([sys.executable, *cmd], cwd=HERE, env=env, capture_output=True, text=True, timeout=timeout)
    out = p.stdout + p.stderr
    return out, p.returncode, time.time() - t0


def last_json(path):
    with open(path) as f:
        return json.loads(f.readlines()[-1])


def gate_layout_and_tiny(do_tiny):
    out, rc, dt = run([os.path.join(K, "test_tiny.py"), "--layers", "2", "--kernel", args.layer_kernel]
                      + ([] if do_tiny else ["--skip-device"]) + (["--control", args.control] if args.control else []))
    res = {}
    m = re.search(r"\[step 1\] reference vs HuggingFace: max \|diff\| / max \|hf\| = ([0-9.e+-]+)", out)
    if not m:
        res["layout"] = dict(ok=False, reason=f"test_tiny did not reach step 1 (rc={rc}): {out[-400:]}")
        return res
    e = float(m.group(1))
    res["layout"] = dict(ok=e < 1e-5, value=e,
                         reason="reference matches HF" if e < 1e-5 else
                         f"reference differs from HF by {e:.1e}: fix qwen3_ref.py (QKV packing, QK-norm, RoPE, GQA) first")
    if not do_tiny:
        return res
    m = re.search(r"kernel vs fp32 ref: ([0-9.e+-]+)\s+kernel vs bf16-input ref: ([0-9.e+-]+)\s+\(bf16 rounding alone: ([0-9.e+-]+)\)", out)
    if not m:
        res["tiny"] = dict(ok=False, reason=f"kernel did not compile/run (rc={rc}): {out[-600:]}")
        return res
    e32, e16, ebase = map(float, m.groups())
    limit = max(3 * ebase, 2e-2)
    ok = e32 <= limit
    diag = {k: float(v) for k, v in re.findall(r"\[diag\] rel-L2 kernel vs variant (\S+)\s*: ([0-9.e+-]+)", out)}
    l2 = re.search(r"\[diag\] rel-L2 kernel vs fp32 ref: ([0-9.e+-]+)", out)
    l2 = float(l2.group(1)) if l2 else None
    if ok:
        reason = f"max error {e32:.2e} within {limit:.1e} (3x the {ebase:.1e} that bf16 rounding alone gives)"
    else:
        best = min(diag, key=diag.get) if diag else None
        if best is not None and l2 is not None and diag[best] < l2:
            reason = (f"max error {e32:.2e} > {limit:.1e}; the output is closer to the '{best}' variant "
                      f"(rel-L2 {diag[best]:.2e}) than to the correct reference ({l2:.2e}): the kernel implements {best}")
        else:
            reason = (f"max error {e32:.2e} > {limit:.1e} and no known wrong convention explains it; rerun "
                      f"kernels/test_tiny.py --zero attn / --zero mlp to see which block carries the error")
    res["tiny"] = dict(ok=ok, max_err=e32, bf16_floor=ebase, limit=limit, variants=diag, reason=reason)
    return res


def gate_head():
    out, rc, dt = run([os.path.join(K, "test_head.py")])
    exact = "gathered exactly: True" in out
    m = re.search(r"logits rel-L2 vs fp32 ref: ([0-9.e+-]+); kernel argmax (\d+).*kernel-logits argmax (\d+)", out)
    if not m:
        return dict(ok=False, reason=f"head kernel did not compile/run (rc={rc}): {out[-600:]}")
    rel, a, b = float(m.group(1)), int(m.group(2)), int(m.group(3))
    reasons = []
    if not exact:
        reasons.append("embedding row is not bit-exact: the indirect DMA reads the wrong row or stride")
    if rel >= 2e-2:
        reasons.append(f"LM-head logits off by {rel:.1e}: check pack_lm_head's row permutation against rmsnorm_tkg's SBUF layout")
    if a != b:
        reasons.append(f"argmax index {a} is not the argmax of the kernel's own logits ({b}): cascaded_max index bookkeeping")
    return dict(ok=not reasons, logits_rel_l2=rel, embed_exact=exact,
                reason="; ".join(reasons) or f"embedding exact, logits rel-L2 {rel:.1e}, argmax consistent")


def gate_real():
    out_path = os.path.join(R, "generate.jsonl")
    cmd = [os.path.join(K, "generate.py"), "--layers", "36", "--new", "32", "--teacher-force", "--timing-iters", "0",
           "--out", out_path] + (["--resident"] if args.resident else [])
    out, rc, dt = run(cmd, timeout=5400)
    if rc != 0:
        return dict(ok=False, reason=f"generate.py failed (rc={rc}): {out[-600:]}")
    r = last_json(out_path)
    dis = r.get("teacher_forced_disagreements", {})
    unexplained = {i: d for i, d in dis.items() if not d["explained_by_bf16_noise"]}
    ok = not unexplained
    if ok:
        reason = (f"teacher-forced argmax agrees on {r['tokens_match']}/{r['new_tokens']} steps; every disagreement is a near-tie "
                  f"inside bf16 noise ({', '.join(f'step {i}: HF margin {d['hf_margin']} < 3 sigma {d['three_sigma']}' for i, d in dis.items()) or 'none'})")
    else:
        i, d = next(iter(unexplained.items()))
        reason = (f"step {i}: kernel picked {d['kernel']!r} over HF's {d['hf']!r} with HF margin {d['hf_margin']} >= 3 sigma "
                  f"{d['three_sigma']}: a real error, not rounding. Bisect layers with kernels/test_qwen3_8b.py --layers N")
    return dict(ok=ok, agreement=f"{r['tokens_match']}/{r['new_tokens']}", disagreements=dis, reason=reason)


def gate_speed():
    out_path = os.path.join(R, "bench_step.jsonl")
    out, rc, dt = run([os.path.join(K, "bench_step.py"), "--layers", "36", "--out", out_path]
                      + (["--resident"] if args.resident else []), timeout=5400)
    if rc != 0:
        return dict(ok=False, reason=f"bench_step.py failed (rc={rc}): {out[-600:]}")
    r = last_json(out_path)
    ok = r["mega_ms"] < r["baseline_ms"]
    return dict(ok=ok, mega_ms=r["mega_ms"], baseline_ms=r["baseline_ms"], floor_ms=r["hbm_floor_ms_at_736GBps"],
                reason=(f"{r['mega_ms']:.2f} ms/token vs {r['baseline_ms']:.2f} per-op ({r['baseline_ms'] / r['mega_ms']:.2f}x), "
                        f"{100 * r['hbm_floor_ms_at_736GBps'] / r['mega_ms']:.0f}% of the HBM floor")
                if ok else f"one launch ({r['mega_ms']:.2f} ms) is not faster than per-op ({r['baseline_ms']:.2f} ms): nothing gained")


gates = args.gates.split(",")
verdict = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), layer_kernel=args.layer_kernel, resident=args.resident,
               control=args.control, gates={})
plan = []
if "layout" in gates or "tiny" in gates:
    plan.append(("layout+tiny", lambda: gate_layout_and_tiny("tiny" in gates)))
for g, fn in (("head", gate_head), ("real", gate_real), ("speed", gate_speed)):
    if g in gates:
        plan.append((g, fn))
all_ok = True
for name, fn in plan:
    t0 = time.time()
    res = fn()
    results = res if name == "layout+tiny" else {name: res}
    for g, r in results.items():
        verdict["gates"][g] = r
        all_ok &= r["ok"]
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {g:7s} {r['reason']}  ({time.time() - t0:.0f}s)", flush=True)
    if not all_ok and not args.keep_going:
        break
verdict["accepted"] = all_ok and all(g in verdict["gates"] for g in gates)
print("ACCEPTED" if verdict["accepted"] else "REJECTED")
with open(args.out, "a") as f:
    f.write(json.dumps(verdict) + "\n")
