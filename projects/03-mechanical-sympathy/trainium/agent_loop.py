#!/usr/bin/env python3
"""agent_loop.py: Qwen3-8B (vLLM on this chip, cores 0-1) writes an NKI kernel for Samudra's hottest
non-conv op, our checker grades it, the reason goes back as an instruction, it tries again.

Target op: fused InstanceNorm2d (affine=False, eps=1e-5) + CappedGELU (exact GELU, then min(., 10)).
It follows every 3x3 conv in every ConvNeXt block of the Samudra UNet (2 per block, 9 blocks).
In kernel layout one sample is x[C, H*W]: channels on the partition axis, pixels on the free axis,
so the norm is a per-partition reduction over the free axis.

Checker stages (cheap -> expensive; reward weights match the organizers' agent.py):
  parses 0.1   python parses
  rules  0.2   only nki / nki.language / nki.isa; entry `instnorm_gelu_kernel(x)` with @nki.jit
  runs   0.2   nki simulator runs on every test shape without raising
  correct 0.5  rel RMS error <= tol vs torch on every shape (incl. partial 128-tiles and outliers
               that must hit the cap)
Stage 2 (on-chip speed vs the compiler's own InstanceNorm+GELU) is device_check.py, run only on
kernels that reach correct.

    python agent_loop.py                       # 8 rounds x 2 samples
    python agent_loop.py --rounds 8 --samples 4 --repeat 3   # a solve rate over 3 runs
    python agent_loop.py --selftest            # prove the checker first (no model, ~1 min)
Every attempt is one JSON line in attempts_samudra.jsonl (the attempt-log deliverable).
"""
import argparse, ast, json, os, re, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

KA = os.environ.get("KERNEL_AGENT_DIR", "/workspace/projects/02-kernel-agent")
sys.path.insert(0, KA)
import nkibench            # noqa: E402  organizers' harness: simulator wrapper, error enrichment
import agent as ka         # noqa: E402  organizers' agent: ask(), extract_code(), enrich()

ENTRY = "instnorm_gelu_kernel"
EPS, CAP = 1e-5, 10.0
WEIGHTS = dict(parses=0.1, rules=0.2, runs=0.2, correct=0.5)
sys.dont_write_bytecode = True   # never run a stale compiled candidate (see grade())
_N_GRADED = 0
FEEDBACK = "v1"   # v1 = organizers' enrich() only; v2 = + the failing line + task-specific fixes
# (C, HW): full tiles, a partial last partition tile, and Samudra-like widths (280 ch).
SHAPES = [(128, 512), (200, 1000), (280, 2048)]
ALLOWED_IMPORTS = {"nki", "nki.language", "nki.isa"}


# ------------------------------------------------------------------ reference + inputs
def make_input(C, HW, seed=0):
    r = np.random.default_rng(seed + C + HW)
    x = (r.standard_normal((C, HW)) * r.uniform(0.5, 3, (C, 1)) + r.uniform(-2, 2, (C, 1)))
    # outliers: after normalization these are ~sqrt(HW) >> 10, so the cap must clip them
    for c in range(0, C, 37):
        x[c, (c * 7) % HW] += 40 * x[c].std()
    return x.astype(np.float32)


def reference(x):
    import torch
    t = torch.from_numpy(x)[None, :, :, None]                 # [1, C, HW, 1] as an image
    y = torch.nn.functional.instance_norm(t, eps=EPS)
    y = torch.clamp(torch.nn.functional.gelu(y), max=CAP)     # exact (erf) GELU, as Samudra
    return y[0, :, :, 0].numpy()


# ------------------------------------------------------------------ checker
def check_rules(src):
    tree = ast.parse(src)
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            bad += [a.name for a in n.names if a.name not in ALLOWED_IMPORTS]
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "") not in ALLOWED_IMPORTS and not (n.module == "nki" and all(
                    a.name in ("language", "isa") for a in n.names)):
                bad.append(n.module or "?")
    v = []
    if bad:
        v.append(f"Imports {sorted(set(bad))} are not allowed; use only `import nki`, "
                 f"`import nki.language as nl`, `import nki.isa as nisa`.")
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == ENTRY]
    if not fns:
        v.append(f"There is no top-level function named {ENTRY}. Start it with exactly "
                 f"`@nki.jit` then `def {ENTRY}(x):`.")
    elif not any("jit" in ast.unparse(d) for d in fns[0].decorator_list):
        v.append(f"Put @nki.jit on the line above def {ENTRY}(x):.")
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    fw = used & nkibench.FRAMEWORK_MODULES
    if fw:
        v.append(f"The kernel uses {sorted(fw)}; all math must use nl./nisa. calls on tiles.")
    return v


def simulate(kernel, x):
    try:
        got, _ = nkibench.simulate_and_count(kernel, [x])
    except TypeError:                       # older/newer harness signature: plain simulator
        import nki
        run, _ = nkibench._simulator(nki, kernel)
        got = run(x)
    return np.asarray(got, dtype=np.float32)


def diagnose(got, want, x, tol):
    """Turn a mismatch into an instruction about the likely bug, not just a number."""
    if got.shape != want.shape:
        return False, (f"Output shape is {got.shape} but must be {want.shape}, the same as x. "
                       f"Allocate the result with nl.ndarray(x.shape, dtype=x.dtype, "
                       f"buffer=nl.shared_hbm) and store every tile.")
    if not np.isfinite(got).all():
        return False, ("The output contains NaN/inf. Add eps=1e-5 to the variance before the "
                       "square root, and compute variance as mean((x-mean)^2), never negative.")
    err = got - want
    rel = float(np.sqrt((err ** 2).mean() / (want ** 2).mean()))
    if rel <= tol:
        return True, f"rel RMS error {rel:.2e}"
    C = x.shape[0]
    bad_rows = np.where(np.abs(err).max(1) > 0.05)[0]
    hints = []
    if np.allclose(got, x, rtol=1e-4, atol=1e-5):
        return False, (f"Wrong values: the output equals the input x, so nothing was computed. "
                       f"Normalize each row (subtract its mean, divide by sqrt(var + 1e-5)), apply "
                       f"gelu, then minimum(., {CAP}), and store that result.")
    capped = np.minimum(got, CAP)
    if got.max() > CAP + 1e-3 and np.sqrt(((capped - want) ** 2).mean() / (want ** 2).mean()) <= tol:
        hints.append(f"Everything is right except the cap: apply y = minimum(gelu(xn), {CAP}).")
    elif got.max() > CAP + 1e-3:
        hints.append(f"Values exceed {CAP}: apply the cap, y = minimum(gelu(xn), {CAP}).")
    if len(bad_rows) and bad_rows.min() >= 128 and C > 128:
        hints.append(f"Rows 0-127 are right but rows >= 128 are wrong (C={C}): loop over "
                     f"partition tiles of 128 and mask the last partial tile (C % 128 = {C % 128}).")
    xm = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + EPS)
    xm_unb = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, ddof=1, keepdims=True) + EPS)
    g = lambda z: np.minimum(0.5 * z * (1 + np.tanh(0.79788456 * (z + 0.044715 * z ** 3))), CAP)
    if np.abs(got - g(xm_unb)).mean() < np.abs(got - g(xm)).mean() * 0.5:
        hints.append("The variance looks unbiased (divided by N-1); InstanceNorm divides by N.")
    if np.abs(got - np.minimum(xm, CAP)).mean() < np.abs(err).mean() * 0.5:
        hints.append("The output looks like the normalized x without GELU; apply gelu after the norm.")
    if not hints:
        r = int(np.abs(err).max(1).argmax()); c = int(np.abs(err[r]).argmax())
        hints.append(f"Largest error at row {r}, column {c}: got {got[r, c]:.4f}, want "
                     f"{want[r, c]:.4f}. Each row (channel) must be normalized by its own mean "
                     f"and variance over all {x.shape[1]} columns.")
    return False, f"Wrong values: rel RMS error {rel:.3f} > {tol}. " + " ".join(hints)


REDUCTIONS = ("nl.sum(", "nl.mean(", "nl.max(", "nl.min(", "nl.prod(", "nisa.tensor_reduce(")


def add_keepdims(line):
    """If the line calls a reduction without keepdims, return the line with keepdims=True added."""
    for r in REDUCTIONS:
        i = line.find(r)
        if i < 0:
            continue
        depth, j = 0, i + len(r) - 1
        for j in range(i + len(r) - 1, len(line)):
            depth += {"(": 1, ")": -1}.get(line[j], 0)
            if depth == 0:
                break
        call = line[i:j + 1]
        if depth != 0 or "keepdims" in call:
            return None
        return line[:j] + ", keepdims=True" + line[j:]
    return None


ONE_D_ALLOC = re.compile(r"(nl\.ndarray\(\s*\(\s*)([^,()]+?)(\s*,\s*\))")


def fix_one_d(line):
    """Return the line with every 1-D-tile cause fixed, or None if it has none."""
    new = ONE_D_ALLOC.sub(lambda m: f"{m.group(1)}{m.group(2)}, 1)", line)
    new = add_keepdims(new) or new
    return new if new != line else None


FAKE_CONST = re.compile(r"nl\.(constant|scalar|value|const|full_like|number)\(([^()]*)\)")


def fix_line(line):
    """All known rewrites for one line (1-D tiles, invented constant wrappers), or None."""
    new = fix_one_d(line) or line
    new = FAKE_CONST.sub(lambda m: m.group(2).strip(), new)
    return new if new != line else None


def one_d_fixes(src, limit=6):
    """Every line in the kernel that would make a 1-D tile, with its exact fix."""
    out = []
    for n, line in enumerate(src.splitlines(), 1):
        code = line.split("#")[0].rstrip()
        fixed = fix_one_d(code)
        if fixed:
            out.append(f"line {n}: `{code.strip()}` -> `{fixed.strip()}`")
    return out[:limit]


INVENTED = re.compile(r"\b(nl|nisa)\.(\w+)\(")


def signature_match(line, missing):
    """The model called a function that doesn't exist. Find real functions in the same module
    whose parameters cover every keyword the model passed; return the line rewritten to the best."""
    import difflib, inspect
    mods = {}
    try:
        import nki.language as _nl, nki.isa as _nisa
        mods = {"nl": _nl, "nisa": _nisa}
    except ImportError:
        return None, []
    call = next((m for m in INVENTED.finditer(line) if m.group(2) == missing), None)
    if not call:
        return None, []
    prefix, mod = call.group(1), mods[call.group(1)]
    kws = set(re.findall(r"(\w+)\s*=(?!=)", line[call.end():]))
    cands = []
    for name in dir(mod):
        fn = getattr(mod, name)
        if name.startswith("_") or not callable(fn):
            continue
        try:
            params = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            continue
        if kws and kws <= params:
            cands.append(name)
    if not cands:
        return None, []
    best = max(cands, key=lambda n: difflib.SequenceMatcher(None, n, missing).ratio())
    return line.replace(f"{prefix}.{missing}(", f"{prefix}.{best}(", 1), cands


def v2_hints(e, path):
    """Feedback v2: name the exact line that failed, and for known errors the exact change."""
    out = ""
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == path]
    if frames:
        f = frames[-1]
        out += f" The line of your kernel that failed is line {f.lineno}: `{(f.line or '').strip()}`."
    m = str(e)
    line = (frames[-1].line or "").strip() if frames else ""
    fixes = one_d_fixes(Path(path).read_text()) if "at least 2 dimensions" in m else []
    if fixes:
        out += (" Per-row values (sums, means, variances) must be (rows, 1) tiles: allocate them with "
                "shape (rows, 1) and pass keepdims=True to every reduction. Fix ALL of these lines now, "
                "not just the first: " + "; ".join(fixes) + ".")
    elif "at least 2 dimensions" in m:
        out += (" In this task the usual cause is the per-row statistics: mean and variance hold one "
                "value per row, so allocate them as (rows, 1) tiles, for example "
                "nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf), and keep that second axis "
                "when you reduce over the free axis (keepdims=True). Never allocate a tile of shape (rows,).")
    elif "has no attribute" in m and FAKE_CONST.search(line):
        out += (f" NKI has no wrapper for constants: pass a plain Python number (an int or float, "
                f"e.g. N or 1.0 / N) directly as the operand. Change the line to `{fix_line(line).strip()}`.")
    elif "has no attribute" in m and re.search(r"has no attribute '(\w+)'", m):
        missing = re.search(r"has no attribute '(\w+)'", m).group(1)
        fixed, cands = signature_match(line, missing)
        if fixed:
            src_lines = Path(path).read_text().splitlines()
            calls = [n for n, l in enumerate(src_lines, 1) if f".{missing}(" in l]
            best = re.search(r"\.(\w+)\(", fixed[fixed.find("."):]).group(1) if "." in fixed else cands[0]
            out += (f" `{missing}` does not exist, but the arguments you passed match "
                    f"{', '.join('`' + c + '`' for c in cands[:3])} exactly. Change the line to `{fixed.strip()}`.")
            if len(calls) > 1:
                out += (f" Your kernel calls `{missing}` {len(calls)} times (lines "
                        f"{', '.join(map(str, calls[:12]))}{', ...' if len(calls) > 12 else ''}): "
                        f"rename every one of them, and remember `data` must be a tile, so a constant "
                        f"goes in `operand0` as a plain number.")
    elif "partition" in m and ("128" in m or "exceeds" in m):
        out += (" A tile has at most 128 rows: loop over the C rows in blocks of 128 with "
                "nl.affine_range, and give the last block its real size, min(128, C - start).")
    return out


def grade(src, tol, shapes=None):
    parts = dict(parses=False, rules=False, runs=False, correct=False)
    score = lambda: sum(WEIGHTS[k] for k, v in parts.items() if v)
    if not src.strip():
        return 0.0, parts, "No code came back. Reply with exactly one python code block."
    try:
        compile(src, "<candidate>", "exec"); parts["parses"] = True
    except SyntaxError as e:
        return 0.0, parts, f"The code does not parse: {e.msg} on line {e.lineno}."
    v = check_rules(src)
    if v:
        return score(), parts, "Rule violations, fix exactly these: " + " ".join(v)
    parts["rules"] = True
    # A fresh file name per candidate: with one shared name, Python's bytecode cache (keyed on file
    # size + mtime to the second) could run the PREVIOUS candidate when two have the same size.
    global _N_GRADED
    _N_GRADED += 1
    path = f"/tmp/_samudra_cand_{os.getpid()}_{_N_GRADED}.py"
    Path(path).write_text(src)
    try:
        kernel = nkibench.load_kernel(path, ENTRY)
    except Exception as e:
        return score(), parts, ka.enrich(f"The file could not be loaded: {type(e).__name__}: {e}")
    fails, ran = [], True
    for C, HW in (shapes or SHAPES):
        x = make_input(C, HW)
        try:
            got = simulate(kernel, x)
        except Exception as e:
            ran = False
            msg = f"{type(e).__name__}: {str(e)[:600]}"
            fb = ka.enrich(msg)
            if FEEDBACK == "v2":
                fb += v2_hints(e, path)
            fails.append(f"[x {C}x{HW}] raised " + fb)
            break                                   # same error on every shape; save time
        ok, msg = diagnose(got, reference(x), x, tol)
        if not ok:
            fails.append(f"[x {C}x{HW}] {msg}")
    parts["runs"] = ran
    if not fails:
        parts["correct"] = True
        return score(), parts, "Correct on every shape."
    return score(), parts, " ".join(fails[:2])


# ------------------------------------------------------------------ prompts
def example_kernel():
    p = Path(KA) / "reference_level1.py"
    return p.read_text() if p.exists() else ""


def first_prompt():
    return f"""Write an NKI kernel for AWS Trainium2 (nki 0.6).

Task: fused InstanceNorm + capped GELU, used inside an ocean model.
Input x: float32 array of shape [C, N] in HBM (C channels, N pixels; C may exceed 128 and need not
be a multiple of 128; N is up to 16200). For every row c independently:
    mean = sum(x[c, :]) / N
    var  = sum((x[c, :] - mean)^2) / N          # biased, divide by N
    xn   = (x[c, :] - mean) / sqrt(var + 1e-5)
    y[c, :] = minimum(gelu(xn), 10.0)           # GELU, then cap at 10
Return y with the same shape and dtype as x.

Put channels on the partition axis (at most 128 rows per tile) and pixels on the free axis.
Requirements: start with exactly
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def {ENTRY}(x):
No numpy or torch inside the kernel. Allocate the output with
nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm) and return it.

Here is a working NKI kernel for a different operation, showing the API that exists in this
version (loading, tiling, storing, allocation):
```python
{example_kernel()}
```
Reply with one python code block containing the whole kernel and nothing else."""


def repair_prompt(src, feedback):
    return f"""Your NKI kernel for fused InstanceNorm + capped GELU (x [C, N] -> y, per-row
mean/biased variance over N, eps 1e-5, then minimum(gelu(xn), 10.0)) was checked:

{feedback}

Your kernel:
```python
{src}
```
Fix it. Keep `@nki.jit def {ENTRY}(x):` and only the imports nki, nki.language as nl, nki.isa as
nisa. Reply with one python code block containing the whole corrected kernel."""


# ------------------------------------------------------------------ checker self-test
# Prove the checker before trusting it: planted bugs must be caught with the right instruction,
# and bugs too small to matter must pass. The correct answer used here is the torch reference, so
# no reference kernel exists anywhere the model could see.

COPY_KERNEL = """import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def instnorm_gelu_kernel(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.dma_copy(dst=out, src=t)
    return out
"""


def _norm(x, ddof=0):
    m = x.mean(1, keepdims=True)
    return (x - m) / np.sqrt(x.var(1, ddof=ddof, keepdims=True) + EPS)


def _gelu_tanh(z):
    return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))


def selftest():
    """Returns True if every planted case gets the expected verdict and feedback."""
    ok_all = True

    def show(name, ok, got, want):
        nonlocal ok_all
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {got[:110]}" + ("" if ok else f"   (expected: {want})"))

    print("1. static rules (no simulator)")
    rule_cases = [
        ("syntax error", "def instnorm_gelu_kernel(x)\n  return x", 0.0, "does not parse"),
        ("torch import", COPY_KERNEL.replace("import nki.isa as nisa", "import nki.isa as nisa\nimport torch"),
         0.1, "not allowed"),
        ("missing @nki.jit", COPY_KERNEL.replace("@nki.jit\n", ""), 0.1, "@nki.jit"),
        ("wrong function name", COPY_KERNEL.replace("def instnorm_gelu_kernel", "def kernel"), 0.1,
         "no top-level function"),
        ("empty answer", "", 0.0, "No code"),
    ]
    for name, src, want_r, want_fb in rule_cases:
        r, parts, fb = grade(src, 2e-2)
        show(name, abs(r - want_r) < 1e-9 and want_fb.lower() in fb.lower(), f"reward {r:.1f}: {fb}",
             f"reward {want_r}, feedback containing '{want_fb}'")

    print("2. numeric diagnosis (planted outputs vs the torch reference)")
    planted = [
        # name, fn(x) -> output, should pass?, feedback keyword
        ("correct (torch reference)", lambda x: reference(x), True, ""),
        ("tanh-approx GELU (within tol)", lambda x: np.minimum(_gelu_tanh(_norm(x)), CAP), True, ""),
        ("variance / (N-1) (within tol)", lambda x: np.minimum(_gelu_tanh(_norm(x, 1)), CAP), True, ""),
        ("cap missing", lambda x: _gelu_tanh(_norm(x)), False, "except the cap"),
        ("GELU missing", lambda x: np.minimum(_norm(x), CAP), False, "without GELU"),
        ("rows >= 128 never written", lambda x: np.concatenate(
            [reference(x)[:128], np.zeros_like(x[128:])]), False, "rows >= 128"),
        ("NaN (eps missing / uninit)", lambda x: reference(x) * np.nan, False, "NaN"),
        ("wrong output shape", lambda x: reference(x)[:, :-1], False, "shape"),
        ("plain copy of x", lambda x: x.copy(), False, "equals the input"),
    ]
    for name, fn, want_pass, kw in planted:
        verdicts = []
        for C, HW in SHAPES:
            x = make_input(C, HW)
            verdicts.append((C, diagnose(np.asarray(fn(x), np.float32), reference(x), x, 2e-2)))
        if want_pass:
            ok = all(v[0] for _, v in verdicts)
            show(name, ok, "passes on all shapes" if ok else str([v for _, v in verdicts if not v[0]][:1]),
                 "pass on every shape")
        else:
            fails = [(C, v[1]) for C, v in verdicts if not v[0]]
            hit = any(kw.lower() in m.lower() for _, m in fails)
            show(name, bool(fails) and hit, fails[0][1] if fails else "PASSED (should fail)",
                 f"fail, with feedback containing '{kw}'")

    print("3. end to end through the NKI simulator (a copy kernel: runs, but wrong values)")
    try:
        r, parts, fb = grade(COPY_KERNEL, 2e-2, shapes=[(128, 512)])
        show("copy kernel, 128x512", parts.get("runs") and not parts.get("correct") and abs(r - 0.5) < 1e-9,
             f"reward {r:.1f} {parts}: {fb}", "reward 0.5, runs=True, correct=False")
    except Exception as e:
        show("copy kernel, 128x512", False, f"{type(e).__name__}: {e}", "the simulator to run")
    print("SELFTEST", "OK" if ok_all else "FAILED")
    return ok_all


# ------------------------------------------------------------------ loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=ka.MODEL)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--samples", type=int, default=2, help="answers per round (vLLM max-num-seqs 4)")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--think", type=int, default=0)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--log", default="attempts_samudra.jsonl")
    ap.add_argument("--repeat", type=int, default=1, help="independent runs, for a solve rate")
    ap.add_argument("--feedback", default="v1", choices=["v1", "v2"],
                    help="v1 = organizers' enrich() only; v2 = + failing line + task-specific fixes")
    ap.add_argument("--selftest", action="store_true", help="prove the checker on planted cases, no model")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    a.think = bool(a.think)
    global FEEDBACK
    FEEDBACK = a.feedback
    solved = []
    for rep in range(a.repeat):
        run_id = time.strftime("%H%M%S") + (f"-r{rep + 1}" if a.repeat > 1 else "")
        solved.append(solve(a, run_id))
    if a.repeat > 1:
        print(f"\nSOLVE RATE: {sum(r is not None for r in solved)}/{a.repeat} runs "
              f"(rounds to solve: {[r for r in solved if r is not None]})")


def solve(a, run_id):
    """One run of the loop. Returns the round it solved in, or None."""
    best = (-1.0, "", "")
    prompt = first_prompt()
    Path("kernels").mkdir(exist_ok=True)
    print(f"run {run_id}: {a.rounds} rounds x {a.samples} samples, feedback {FEEDBACK}, "
          f"model {a.model} at {a.base}")
    for rnd in range(1, a.rounds + 1):
        t0 = time.time()
        try:
            answers = ka.ask_parallel(a, prompt, a.samples)
        except Exception:
            answers = [ka.ask(a, prompt) for _ in range(a.samples)]
        gen_s = time.time() - t0
        results = []
        for i, ans in enumerate(answers):
            src = ka.extract_code(ans or "")
            t1 = time.time()
            try:
                reward, parts, fb = grade(src, a.tol)
            except Exception as e:  # a checker crash must never be blamed on the model
                reward, parts, fb = 0.0, {}, f"CHECKER ERROR (not the model's fault): {e}"
                traceback.print_exc()
            rec = dict(run=run_id, round=rnd, sample=i, reward=round(reward, 3), parts=parts,
                       feedback=fb, gen_s=round(gen_s, 1), check_s=round(time.time() - t1, 1),
                       chars=len(src), prompt_chars=len(prompt), reply_chars=len(ans or ""),
                       feedback_version=FEEDBACK,
                       source=src, time=time.strftime("%H:%M:%S"))
            with open(a.log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            results.append((reward, src, fb))
            print(f"  round {rnd} sample {i}: reward {reward:.2f}  {fb[:160]}")
        rb = max(results, key=lambda r: r[0])
        if rb[0] > best[0]:
            best = rb
        if rb[0] >= 1.0:
            out = Path("kernels") / f"instnorm_gelu_{run_id}.py"
            out.write_text(rb[1])
            print(f"CORRECT kernel in round {rnd}: saved {out}")
            return rnd
        # Repair the best answer of THIS round (as the organizers' agent.py does): repairing the
        # all-time best made their loop resend the same answer forever. No code at all -> start over.
        prompt = repair_prompt(rb[1], rb[2]) if rb[1].strip() else first_prompt()
    print(f"no correct kernel after {a.rounds} rounds; best reward {best[0]:.2f}: {best[2][:300]}")
    return None


if __name__ == "__main__":
    main()
