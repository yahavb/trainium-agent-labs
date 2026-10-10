"""agent_feedback_v2.py -- run agent.py with three prompt changes, measured against the baseline.

Place beside agent.py. All agent.py CLI options still work:

    python agent_feedback_v2.py --check-feedback          # print the new prompts, no model
    python agent_feedback_v2.py --all --rounds 8 --samples 4 --context 8192 --repeat 3 \
        --log attempts-feedback-v2.jsonl

What changes, and why (each one traced to a failure in baseline.log):

1. SHAPE CARD. The first prompt never told the model the test shapes, so on level 4 it wrote a
   single-tile kernel, passed K=M=128 and died on K=256 M=256 N=1024 with "partition dimension 256
   exceeds maximum 128" four rounds running. Now both prompts list every shape the checker runs.

2. RESTRUCTURE INSTEAD OF PATCH. repair_prompt says "change exactly what the checker names and
   keep everything else identical", while the checker's hint for the same failure says "loop over
   the partition dimension in chunks". An 8B model obeyed the first and so could not do the second.
   When the feedback asks for a loop-structure change, the repair prompt now says to rewrite.

3. LEVEL-AWARE DIRECTION. Level 1 kept reaching for nc_matmul (transpose_moving=, dst in psum,
   nisa.multiply ...) for a pooling reduction; the matmul levels were never told how K, M, N map
   onto loops. A short paragraph per level names the direction. It is text, not the reference.

v1's targeted tensor_scalar hint for the invented nisa.multiply / nisa.scalar_mul is kept.
"""

import inspect
import re
import sys

# ------------------------------------------------------------------ v1, kept

MISSING_SCALE_API = re.compile(
    r"module ['\"]nki\.isa['\"] has no attribute ['\"](?:multiply|scalar_mul)['\"]"
)

SCALING_HINT = (
    " For elementwise scaling by a constant, use "
    "nisa.tensor_scalar(dst=scaled, data=tile, op0=nl.multiply, operand0=scale). "
    "First allocate scaled = nl.ndarray(tile.shape, dtype=tile.dtype, buffer=nl.sbuf). "
    "Here tile is an existing on-chip tile, scale is a compile-time scalar constant, "
    "and scaled has the same shape as tile. nl.multiply names the operator; "
    "tensor_scalar executes it and writes into dst. Replace the missing call "
    "with this operation, using your own tile and scalar variable names."
)

# ------------------------------------------------------------------ v2

# Feedback that can only be acted on by changing the loop structure. The stock repair prompt
# forbids exactly that, so these switch it to a rewrite prompt.
STRUCTURAL_MARKERS = (
    "exceeds maximum",
    "exceeds pmax",
    "loop over the partition dimension",
    "Split K into chunks",
    "Out-of-bound access",
    "looping over the wrong dimension",
    "covering only part of the output",
)

# Per-level direction. Deliberately prose: it says which axis maps to which loop and which
# instruction does the work, never the kernel itself.
DIRECTION = {
    1: ("This is a REDUCTION, not a matmul: do NOT use nisa.nc_matmul or PSUM anywhere in it. "
        "Do NOT dma_copy one pool window at a time either -- that is thousands of tiny transfers. "
        "Every test shape has C <= 128, so load the WHOLE input once into one sbuf tile of shape "
        "x.shape (C on the partition axis). Then reduce with a strided view: tile.ap(pattern) "
        "makes an N-D view of a tile without moving data, where pattern is a list of "
        "[stride_in_elements, count] pairs, one per axis of the view, the FIRST pair being the "
        "partition axis, and the view's axes come out in the order you list them. Example: a "
        "(P, F) tile with F = F1*F2 is seen as (P, F1, F2) by t.ap([[F, P], [F2, F1], [1, F2]]). "
        "For pooling, build a 5-D view (C, H//p, W//p, p, p) in which the LAST two axes walk the "
        "p x p window -- work the five strides out from the row-major layout of (C, H, W) -- then "
        "s = nl.sum(view, axis=[3, 4]) gives a (C, H//p, W//p) tile. Scale it with "
        "nisa.tensor_scalar(dst=o, data=s, op0=nl.multiply, operand0=1.0/(p*p)) into an sbuf tile "
        "o of the same shape, and dma_copy o into the shared_hbm output."),
    3: ("Layout facts for this matmul: lhsT is [K, M] and rhs is [K, N], so K is the PARTITION axis "
        "of BOTH inputs. The RESULT is [M, N] = (lhsT.shape[1], rhs.shape[1]) -- two dimensions, "
        "with M on the partition axis -- and it needs its OWN tiles: a psum tile of shape (M, N) in "
        "nl.float32, an sbuf tile of shape (M, N) to copy it into, and the shared_hbm output of "
        "shape (M, N). Never reuse an operand tile for the result; (K, N) is not (M, N). One "
        "nc_matmul consumes a stationary tile lhsT[k0:k0+kk, m0:m0+mm] (kk <= 128, mm <= 128) and "
        "a moving tile rhs[k0:k0+kk, n0:n0+nn] (nn <= 512). Allocate every tile with the exact "
        "shape of the slice it holds: when a dimension is smaller than the limit (here M=64), use "
        "the dimension itself, never the limit. Sequence: dma_copy both operands HBM->sbuf, "
        "nc_matmul into psum, tensor_copy psum->sbuf, dma_copy sbuf->the shared_hbm output."),
    4: ("Layout facts for this matmul: lhsT is [K, M] and rhs is [K, N], so K is the PARTITION axis "
        "of BOTH inputs. The RESULT is [M, N] = (lhsT.shape[1], rhs.shape[1]) with M on the "
        "partition axis, and it needs its OWN tiles -- never reuse an operand tile for it. One "
        "nc_matmul consumes a stationary tile lhsT[k0:k0+128, m0:m0+128] and a moving tile "
        "rhs[k0:k0+128, n0:n0+512] and writes a (128, 512) float32 tile in psum. Three dimensions "
        "can exceed one tile, and each one is its own loop:\n"
        "  - M > 128: loop over 128-row blocks of the OUTPUT (m0 = i*128).\n"
        "  - N > 512: loop over 512-column blocks of the OUTPUT (n0 = j*512).\n"
        "  - K > 128: loop over 128-deep chunks of the CONTRACTION (k0 = k*128); allocate ONE psum "
        "tile of shape (128, 512) before this loop and call nc_matmul into that same tile every "
        "iteration -- it accumulates. Only after the K loop tensor_copy psum->sbuf and dma_copy to "
        "out[m0:m0+128, n0:n0+512].\n"
        "Derive every loop count from .shape (M // 128, N // 512, K // 128); the test shapes are all "
        "exact multiples. Allocate operand tiles INSIDE the loops with the slice's exact shape, and "
        "never allocate any tile with more than 128 on its first dimension."),
}
DIRECTION[2] = ""  # level 2 already solves on round 0 in the baseline; leave it alone
for _lv in (5, 6, 7):
    DIRECTION[_lv] = DIRECTION[4]


def shape_card(base, level):
    """Every shape the checker will run, with the output shape, as text for the prompt."""
    nb = base.nkibench
    spec = nb.LEVELS[level]
    params = list(inspect.signature(spec["ref"]).parameters)
    lines = []
    for case in spec["shapes"]:
        args, _ = nb.make_inputs(case, level)
        want = spec["ref"](*args)
        ins = ", ".join(
            f"{p}.shape={tuple(a.shape)}" if hasattr(a, "shape") else f"{p}={a!r}"
            for p, a in zip(params, args))
        lines.append(f"  {ins}  ->  returns shape {tuple(want.shape)}")
    return ("The checker runs the kernel on ALL of these, and every one must pass, so take every "
            "size from the arguments' .shape instead of hard-coding one case:\n"
            + "\n".join(lines))


def level_block(base, level):
    d = DIRECTION.get(level, "")
    card = shape_card(base, level)
    return (d + "\n\n" + card) if d else card


def install_feedback(base):
    """Patch enrich, first_prompt and repair_prompt in the loaded agent module.

    solve() looks these names up as module globals on every call, so rebinding them here is
    enough; nothing in agent.py is edited.
    """
    orig_enrich = base.enrich
    orig_first = base.first_prompt
    orig_repair = base.repair_prompt

    def enrich(error_text):
        if MISSING_SCALE_API.search(error_text):
            return error_text + SCALING_HINT
        return orig_enrich(error_text)

    def first_prompt(level, terse=0):
        p = orig_first(level, terse)
        # Insert before the final "Reply with ..." sentence so the answer format stays last.
        marker = "Reply with"
        i = p.rfind(marker)
        block = level_block(base, level)
        if i < 0:
            return p + "\n\n" + block
        return p[:i] + block + "\n\n" + p[i:]

    def repair_prompt(level, source, feedback):
        op = base.nkibench.LEVELS[level]["op"]
        block = level_block(base, level)
        if any(m in feedback for m in STRUCTURAL_MARKERS):
            return (f"This NKI kernel for {op} is not right yet.\n\n"
                    f"```python\n{source}\n```\n\n"
                    f"A checker reports:\n{feedback}\n\n"
                    f"{block}\n\n"
                    f"The fix is a change of LOOP STRUCTURE, so rewrite the body of the function "
                    f"rather than patching one line. Keep the imports, the @nki.jit decorator, the "
                    f"function name and its arguments exactly as they are. Reply with ONE python "
                    f"code block.")
        return orig_repair(level, source, feedback) + "\n\n" + block

    base.enrich = enrich
    base.first_prompt = first_prompt
    base.repair_prompt = repair_prompt
    return orig_enrich, orig_first, orig_repair


def check_feedback(base):
    """No model, no kernel: print what the model would now see, and assert the routing."""
    originals = install_feedback(base)
    orig_enrich, orig_first, orig_repair = originals
    try:
        err = "raised AttributeError: module 'nki.isa' has no attribute 'multiply'"
        assert base.enrich(err) == err + SCALING_HINT
        other = "raised AssertionError: dst must be in ['psum'], got sbuf"
        assert base.enrich(other) == orig_enrich(other)
        print("enrich: v1 scaling hint routed, other errors unchanged")

        structural = ("1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: "
                      "dma_copy dst partition dimension 256 exceeds maximum 128 A tile may have at "
                      "most 128 rows")
        rp = base.repair_prompt(4, "def f(): pass", structural)
        assert "LOOP STRUCTURE" in rp and "keep everything else identical" not in rp
        plain = base.repair_prompt(4, "def f(): pass", "NUMERICAL MISMATCH: worst error 0.3")
        assert "keep everything else identical" in plain and "returns shape" in plain
        print("repair_prompt: structural failures -> rewrite; others -> original + shape card")

        for lv in (1, 3, 4):
            p = base.first_prompt(lv)
            assert "returns shape" in p and p.rstrip().endswith("No prose.")
            print(f"\n=== first_prompt level {lv}: {len(p)} chars (~{len(p) // 4} tokens) ===")
            start = p.find(DIRECTION[lv][:40]) if DIRECTION.get(lv) else p.find("The checker runs")
            print(p[start:])
        print("\nFEEDBACK CHECK PASSED (prompts and routing only; no kernel/model run)")
    finally:
        base.enrich, base.first_prompt, base.repair_prompt = originals


if __name__ == "__main__":
    import agent as base

    if sys.argv[1:] == ["--check-feedback"]:
        check_feedback(base)
    else:
        install_feedback(base)
        print("Feedback experiment v2: shape card + restructure prompt + level direction",
              flush=True)
        base.main()
