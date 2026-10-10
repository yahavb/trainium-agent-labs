#!/usr/bin/env python3
"""
verifier.py — grades one candidate kernel. THE component: the loop is only as good as
the feedback this returns, which is why it is built before the agent and self-tested
against planted bugs (selftest.py).

Order, cheapest first -- the same gate shape as project 02's nkibench, ported to the
NumPy stage:

  1. parses?          compile(), milliseconds
  2. rules clean?     AST scan: banned calls, fancy indexing, whole-array arithmetic,
                      missing tile loop. Milliseconds, no execution.
  3. runs?            executed per test case, inputs snapshotted (a kernel that writes
                      into its argument PASSED once in project 02 because the reference
                      happened to run first -- luck is not correctness)
  4. correct?         against the reference, on hostile shapes and values, with the
                      level's stated tolerance; every mismatch localised to an element
                      and classified ragged-edge vs partial-coverage vs core arithmetic

Every failure comes back as {case, verdict, taxonomy, INSTRUCTION} -- the instruction is
what the model is fed, and errors.instruction_from_failure() builds it. The reward is
partial credit (0.1 parse, 0.2 rules, 0.2 runs, 0.5 correct) so the loop has a gradient
to climb and near-misses stay distinguishable from nonsense.
"""

import ast
import signal
import time
import traceback

import numpy as np

import errors
import ladder

WEIGHTS = dict(parses=0.1, rules=0.2, runs=0.2, correct=0.5)
FULL_REWARD = sum(WEIGHTS.values())

# Absolute floor: an all-zero reference (the 'zeros' cases) has RMS 0, so the
# RMS-relative test needs something to compare against. 1e-5 is far under any real
# computation's drift and far over float32 denormal noise.
ABS_FLOOR = 1e-5

EXEC_TIMEOUT_S = 10          # per test case; a tiled elementwise op on 257x61 is microseconds
NONDETERMINISM_CHECK = True  # run case 0 twice; catches hidden global state


# ---------------------------------------------------------------- 1. parse

def check_parses(src):
    if not (src or "").strip():
        return None, "No code came back. Reply with one python code block containing the kernel and nothing else."
    try:
        compile(src, "<candidate>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"The code does not parse: {e.msg} on line {e.lineno}."


# ---------------------------------------------------------------- 2. static rules

def _dotted(node):
    parts, cur = [], node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def check_rules(src, level_n):
    """A scan, not a proof -- same caveat as project 02: it catches the cheats visible in
    the text. Returns (violations, taxonomy_label) with label None when clean or
    ambiguous (a plain syntax problem is reported by the parse gate, not here)."""
    spec = ladder.LEVELS[level_n]
    bad = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], None  # the parse gate owns syntax

    # entry-point bookkeeping
    entry = "kernel"
    entry_node, decorated = None, False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            entry_node = node
            for d in node.decorator_list:
                if "jit" in ast.unparse(d):
                    decorated = True
    if entry_node is None:
        return [f"no function named `{entry}` is defined -- that is the entry point this "
                f"level is checked through"], "parse-error"

    array_params = [a.arg for i, a in enumerate(entry_node.args.args)
                    if i in spec["array_args"]]

    banned = set(spec["ban"])
    for name in list(banned):          # 'add.reduce' style dotted bans -> leaf name too
        banned.add(name.split(".")[-1])

    def _is_param(node):
        return isinstance(node, ast.Name) and node.id in array_params

    def _fancy(slice_node):
        """Boolean masks and integer-array subscripts anywhere in the index expression.
        Tuples of slices/constants (x[r0:r0+128, c0:c0+512]) are the tiling idiom and are
        fine -- only a bare List or a Compare inside the index is a cheat."""
        return any(isinstance(n, (ast.Compare, ast.List)) for n in ast.walk(slice_node))

    has_loop = False
    tagged = []            # (message, taxonomy) -- each violation carries its own label
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            has_loop = True

        if isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            leaf = dotted.split(".")[-1] if dotted else ""
            if leaf in banned:
                # The ban targets handing the WHOLE operation to the framework
                # (rejecting a correct kernel is worse than missing a cheat -- measured in
                # project 02). Function form: banned when an argument IS the whole ndarray
                # parameter (np.sum(x)); method form (x.sum()) when the receiver is.
                # np.sum(t, axis=1) on a tile slice is the intended route and stays legal.
                if isinstance(node.func, ast.Attribute) and _is_param(node.func.value):
                    on_whole_input = True
                else:
                    on_whole_input = any(_is_param(a) for a in node.args
                                         if isinstance(a, ast.Name))
                if on_whole_input:
                    tagged.append((f"line {node.lineno}: calls `{dotted}` on the whole "
                                   f"input, which does this whole operation. The kernel "
                                   f"has to compute it (on a tile slice it would be "
                                   f"legal).", "banned-call"))

        if isinstance(node, ast.Subscript) and _fancy(node.slice):
            tagged.append((f"line {node.lineno}: fancy/boolean indexing is not allowed -- "
                           f"slices and arithmetic on slices only.", "fancy-indexing"))

        # whole-array arithmetic on an ndarray parameter: the broadcast trick
        if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.MatMult)):
            for side in (node.left, node.right):
                if _is_param(side):
                    tagged.append((f"line {node.lineno}: arithmetic on `{side.id}` (the "
                                   f"whole input) hides the tiling. Slice a tile and "
                                   f"operate on that.", "whole-array-op"))

        # np.transpose(x) as a CALL is banned per-level; x.T as an attribute on a param
        if isinstance(node, ast.Attribute) and node.attr == "T" \
                and isinstance(node.value, ast.Name) and node.value.id in array_params \
                and "transpose" in banned:
            tagged.append((f"line {node.lineno}: `{node.value.id}.T` transposes the whole "
                           f"input in one library step.", "banned-call"))

    if ladder.tile_loop_required(level_n) and not has_loop:
        tagged.append(("no explicit loop anywhere -- this level's shapes do not fit in "
                       "one 128x512 tile, so the work must be wrapped in `for` loops "
                       "over tiles.", "no-tile-loop"))

    if not tagged:
        return [], None
    # One label for the attempt: the most actionable violation wins.
    PRIORITY = ["banned-call", "fancy-indexing", "whole-array-op", "no-tile-loop"]
    tax = next((t for p in PRIORITY for _m, t in tagged if t == p), "banned-call")
    return sorted({m for m, _t in tagged}), tax


# ---------------------------------------------------------------- 3+4. execution

class _Timeout(Exception):
    pass


def _guard(seconds):
    """signal-based per-case timeout. Verifier runs in the main thread of agent.py; when
    it does not (dashboard replay, threads), the guard is skipped rather than fatal."""
    def deco(fn):
        def run(*a, **k):
            use_signal = hasattr(signal, "setitimer")
            if use_signal:
                try:
                    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))
                    signal.setitimer(signal.ITIMER_REAL, seconds)
                except ValueError:      # not the main thread
                    use_signal = False
            try:
                return fn(*a, **k)
            finally:
                if use_signal:
                    signal.setitimer(signal.ITIMER_REAL, 0)
        return run
    return deco


def _snapshot(args):
    return [a.copy() if isinstance(a, np.ndarray) else a for a in args]


def _stat_scope(got, want, err, tol):
    """Detect the per-tile-statistic signature. Returns (taxonomy, hint) or None.

    Fits a per-(row, column-tile) least-squares scale s and asks how much of the error
    it explains. Two diagnoses come out:
      stat-scope      -- scales DIFFER between tiles: the statistic was computed per
                         tile (softmax's 1-wide tile returning 1.0 is the loud case)
      core-arithmetic -- one uniform scale explains everything: a wrong leading
                         constant or epsilon, not a scope problem
    """
    if got.ndim != 2 or got.shape[1] <= ladder.TILE_COLS:
        return None
    wrong = int((err > tol).sum())
    if not wrong:
        return None
    explained = 0
    scales = []
    for c0 in range(0, got.shape[1], ladder.TILE_COLS):
        c1 = min(c0 + ladder.TILE_COLS, got.shape[1])
        m = err[:, c0:c1] > tol
        if not m.any():
            continue
        g, w = got[:, c0:c1], want[:, c0:c1]
        dn = (w * w).sum(axis=1, keepdims=True)
        num = (g * w).sum(axis=1, keepdims=True)
        s = np.where(dn > 1e-15, num / np.where(dn > 1e-15, dn, 1.0), 1.0)
        resid = np.abs(g - w * s)
        good = (resid <= tol * 0.5) & m & (np.abs(s) > 0.2)
        explained += int(good.sum())
        for r in range(got.shape[0]):
            if m[r, 0] or m[r, -1]:
                scales.append(float(s[r, 0]))
    if explained < 0.6 * wrong or not scales:
        return None
    arr = np.asarray(scales)
    varies = arr.size > 1 and (arr.max() / max(arr.min(), 1e-9)) > 1.05 \
        and not np.allclose(arr, arr[0], rtol=0.02)
    if varies:
        return ("stat-scope",
                "the error is a per-TILE scaling: your normalising statistic (max, sum, "
                "mean, RMS) was computed per TILE, but the reference computes it over "
                "the WHOLE row. Restructure to two passes -- pass 1 accumulates the "
                "per-row statistic across all of the row's column tiles, pass 2 walks "
                "the tiles again and applies it. (before answering, run: DOCS: row "
                "statistics that span column tiles)")
    return ("core-arithmetic",
            "every wrong element is the reference times a NEAR-CONSTANT factor, so the "
            "shape of the computation is right and one constant in it is wrong (a "
            "missing epsilon, a scale factor, a sqrt). Check the formula's constants.")


def describe_mismatch(got, want, rtol):
    """The message the agent learns from. Localises: element, direction, which tile.
    Returns (verdict, taxonomy) or (None, None)."""
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)

    if got.shape != want.shape:
        return (f"WRONG SHAPE: returned {got.shape}, reference is {want.shape}. "
                f"Check the output-size arithmetic, not the values.", "wrong-shape")

    if not np.all(np.isfinite(got)):
        n_nan, n_inf = int(np.isnan(got).sum()), int(np.isinf(got).sum())
        where = np.argwhere(~np.isfinite(got))[0]
        return (f"NON-FINITE OUTPUT: {n_nan} NaN and {n_inf} Inf, first at "
                f"{tuple(int(i) for i in where)}. Usually overflow (softmax needs the row "
                f"max subtracted before exp) or an uninitialised output element being read.",
                "non-finite")

    scale = float(np.sqrt((want ** 2).mean()))
    tol = max(rtol * scale, ABS_FLOOR)
    err = np.abs(got - want)
    worst = float(err.max())
    if worst <= tol:
        return None, None

    # An all-zero output is its own diagnosis and the most common one: the kernel ran,
    # so it scores for running, but the results never reached the output.
    zero_frac = float((np.abs(got) < ABS_FLOOR).mean())
    want_zero_frac = float((np.abs(want) < ABS_FLOOR).mean())
    if zero_frac > 0.9 and want_zero_frac < 0.5:
        # When the array itself does not fill one tile (or its dims are not tile
        # multiples), an all-zero result is the ragged-edge signature -- the whole output
        # IS the final partial tile. Reserve partial-coverage for genuinely tiled arrays.
        shape_is_ragged = (got.shape[0] < ladder.TILE_ROWS
                           or got.shape[-1] < ladder.TILE_COLS
                           or got.shape[0] % ladder.TILE_ROWS != 0
                           or got.shape[-1] % ladder.TILE_COLS != 0)
        tax = "ragged-edge" if shape_is_ragged else "partial-coverage"
        return (f"OUTPUT IS {zero_frac:.0%} ZEROS while the reference is not. The kernel ran "
                f"but its results never reached the output array. Check that every tile of "
                f"your loop writes to ITS OWN slice of the output", tax)
    if zero_frac > 0.25 and want_zero_frac < 0.05:
        zi = np.argwhere(np.abs(got) < ABS_FLOOR)
        lo, hi = zi.min(axis=0), zi.max(axis=0)
        return (f"{zero_frac:.0%} of the output is zero, in the block from "
                f"{tuple(int(v) for v in lo)} to {tuple(int(v) for v in hi)}, while the "
                f"reference has no zeros there. Some tiles were written and others were "
                f"not: check the loop bounds against the output shape.", "partial-coverage")

    i = int(np.argmax(err))
    idx = np.unravel_index(i, got.shape)
    msg = [f"NUMERICAL MISMATCH: worst error {worst:.3g} of the output's RMS ({scale:.4g}), "
           f"tolerance {tol:.3g}.",
           f"  at index {tuple(int(v) for v in idx)}: expected {want.flat[i]:+.6g}, "
           f"got {got.flat[i]:+.6g}",
           f"  {float((err > tol).mean()):.1%} of elements are outside tolerance"]
    tax = "core-arithmetic"

    # Statistic-scope signature (measured twice in the first live baseline): when the
    # error is explained, per (row, column tile), by got ~= want * s -- i.e. the output
    # is the reference times a factor that DIFFERS between tiles -- the kernel computed
    # its normalising statistic per TILE where the reference used the whole row. This is
    # invisible to the ragged-edge location hint, which sends the model clamping slices
    # it does not need to clamp. A UNIFORM factor instead means a plain wrong constant.
    scope = _stat_scope(got, want, err, tol)
    if scope is not None:
        kind, hint = scope
        msg.append("  " + hint)
        return "\n".join(msg), kind

    if got.ndim >= 2:
        r, c = int(idx[0]), int(idx[-1])
        ragged = []
        if r >= (got.shape[0] // ladder.TILE_ROWS) * ladder.TILE_ROWS and got.shape[0] > ladder.TILE_ROWS:
            ragged.append(f"the final partial ROW tile (row {r} of {got.shape[0]})")
        if c >= (got.shape[-1] // ladder.TILE_COLS) * ladder.TILE_COLS and got.shape[-1] > ladder.TILE_COLS:
            ragged.append(f"the final partial COLUMN tile (col {c} of {got.shape[-1]})")
        if ragged:
            msg.append(f"  this is in {' and '.join(ragged)} -- the ragged edge is the "
                       f"likely cause, not the core arithmetic")
            tax = "ragged-edge"
        elif float((err > tol).mean()) > 0.5:
            msg.append("  most elements are wrong, so this is the core arithmetic, not "
                       "an edge case")
        else:
            msg.append(f"  row {r} of {got.shape[0]}, col {c} of {got.shape[-1]}, inside "
                       f"a full tile")
    return "\n".join(msg), tax


@_guard(EXEC_TIMEOUT_S)
def _run_case(kernel_fn, args):
    return kernel_fn(*args)


def check(src, level_n, with_nondeterminism=NONDETERMINISM_CHECK, max_failures=3):
    """Grade one candidate. Returns a result dict:
       reward, parts, failures[{case, verdict, taxonomy, instruction}], solved, notes
    solved means correct on EVERY case of the level."""
    spec = ladder.LEVELS[level_n]
    parts = dict(parses=False, rules=False, runs=False, correct=False)
    failures = []
    fail = lambda tax, case, verdict, instr=None: failures.append(dict(
        taxonomy=tax, case=case, verdict=verdict,
        instruction=instr or errors.instruction_from_failure(dict(taxonomy=tax, verdict=verdict), case)))

    ok, why = check_parses(src)
    if ok is None:
        return _result(parts, [dict(taxonomy="empty-answer", case="", verdict=why,
                                    instruction=errors.instruction_from_failure(
                                        dict(taxonomy="empty-answer", verdict=""), ""))],
                       notes="empty")
    if not ok:
        fail("parse-error", "", why)
        return _result(parts, failures)

    parts["parses"] = True

    # Signature gate FIRST (cheaper and more specific than the rule scan): a kernel with
    # the wrong arity otherwise dies on case 1 with a TypeError that never mentions the
    # reference's signature, or trips an unrelated static rule.
    ns = {}
    try:
        exec(compile(src, "<candidate>", "exec"), ns)
    except Exception:
        tb = traceback.format_exc(limit=4)
        fail("raised", "import", f"The module raised on import: {tb.strip().splitlines()[-1]}")
        return _result(parts, failures)
    kernel_fn = ns.get("kernel")
    if not callable(kernel_fn):
        fail("parse-error", "", "No callable `kernel` defined. Name the entry point `kernel` "
                                "with the same signature as the reference.")
        return _result(parts, failures)
    import inspect as _inspect
    want_params = list(_inspect.signature(spec["ref"]).parameters)
    got_params = list(_inspect.signature(kernel_fn).parameters)
    if got_params != want_params:
        fail("parse-error", "signature",
             f"kernel{tuple(got_params)} has the wrong signature. Define it EXACTLY as "
             f"kernel{tuple(want_params)} -- same names, same order.")
        return _result(parts, failures)

    violations, vtax = check_rules(src, level_n)
    if violations:
        first = violations[0]
        fail(vtax or "banned-call", f"line {first.split(':')[0] if ':' in first else '?'}",
             "Rule violations (score zero however fast the kernel is): "
             + " ".join(violations[:4]))
        return _result(parts, failures)
    parts["rules"] = True

    cases = spec["build"]()
    passed, ran_any, checked_cases = 0, 0, 0
    for case_i, (label, args) in enumerate(cases):
        try:
            want = spec["ref"](*_snapshot(args))
        except Exception as e:      # a broken reference is OUR bug -- say so, loudly
            failures.append(dict(taxonomy="raised", case=label,
                                 verdict=f"REFERENCE ITSELF FAILED: {e}", instruction=""))
            continue
        checked_cases += 1

        call_args = _snapshot(args)
        try:
            # NumPy prints a RuntimeWarning block per overflow/divide -- dozens per run,
            # all describing the same bug the non-finite check already reports. Capture
            # them; if the numbers came out finite anyway (e.g. exp underflow), they stay
            # captured rather than scrolling the transcript.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = _run_case(kernel_fn, call_args)
        except _Timeout:
            fail("timeout", label, f"exceeded {EXEC_TIMEOUT_S}s on this case -- "
                 f"restructured work needed, a tiled kernel over this shape is sub-second")
            continue
        except Exception as e:
            msg = f"RAISED {type(e).__name__}: {e}"
            fail("raised", label, errors.enrich_exception(msg))
            continue
        ran_any += 1
        parts["runs"] = True

        mut = _input_mutation(args, call_args)
        if mut:
            fail("modified-input", label, mut)
            continue
        verdict, tax = describe_mismatch(got, want, spec["rtol"])
        if verdict:
            fail(tax, label, verdict)
            continue
        passed += 1

        if with_nondeterminism and case_i == 0:
            try:
                again = _run_case(kernel_fn, _snapshot(args))
                v2, t2 = describe_mismatch(again, got, 0.0)
                if v2:
                    fail("nondeterministic", label,
                         "Two runs on the identical input returned different results "
                         "(tolerance 0). You are reading uninitialised memory or keeping "
                         "state between calls.")
            except Exception:
                pass

    if checked_cases and passed == checked_cases:
        parts["correct"] = True
        return _result(parts, failures, solved=True)
    if not ran_any:
        parts["runs"] = False
    return _result(parts, failures,
                   notes=f"{passed}/{checked_cases} cases passed" if checked_cases else "no cases ran")


def _input_mutation(before, after):
    for i, (orig, now) in enumerate(zip(before, after)):
        if isinstance(orig, np.ndarray) and not np.array_equal(orig, np.asarray(now)):
            return (f"THE KERNEL MODIFIED ITS INPUT (argument {i}). Allocate a new output "
                    f"array and write the result there; the arguments belong to the caller.")
    return None


def _result(parts, failures, solved=False, notes=""):
    reward = (WEIGHTS["parses"] * parts["parses"] + WEIGHTS["rules"] * parts["rules"]
              + WEIGHTS["runs"] * parts["runs"] + WEIGHTS["correct"] * parts["correct"])
    return dict(reward=round(reward, 4), parts=parts, failures=failures,
                solved=solved, notes=notes,
                taxonomy=failures[0]["taxonomy"] if failures else None)


# ---------------------------------------------------------------- CLI: --check a file

def verify_file(path, level_n):
    src = open(path).read()
    spec = ladder.LEVELS[level_n]
    r = check(src, level_n)
    print(f"level {level_n}: {spec['name']}   (rtol {spec['rtol']:g} of RMS -- {spec['tol_why'][:60]}...)")
    if r["solved"]:
        print("  VERIFIED: correct on every case.")
        return 0
    print(f"  reward {r['reward']:.2f}   parts {r['parts']}   {r['notes']}")
    for f in r["failures"][:3]:
        print(f"\n  case: {f['case']}   taxonomy: {f['taxonomy']}")
        import textwrap
        print(textwrap.indent(f["verdict"], "    "))
        print(textwrap.indent("INSTRUCTION: " + f["instruction"], "    "))
    print("\n  ^ feed the INSTRUCTION to the model. If it does not locate the bug,")
    print("    improve THIS message before touching any prompt.")
    return 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4 and sys.argv[1] == "--level":
        sys.exit(verify_file(sys.argv[3], int(sys.argv[2])))
    print("usage: python verifier.py --level N --check FILE.py")
