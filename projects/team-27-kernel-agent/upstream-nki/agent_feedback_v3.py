"""agent_feedback_v3.py -- v2 plus two feedback rules: .ap() strides, and nl.shared_hbm is not a tensor.

Place beside agent.py and agent_feedback_v2.py. All agent.py CLI options still work:

    python agent_feedback_v3.py --check-feedback
    python agent_feedback_v3.py --level 1 --rounds 8 --samples 4 --context 8192 --repeat 5 \
        --log attempts-feedback-v3-L1.jsonl

Why. Under v2 the level-1 kernel reached the right structure for the first time -- whole input in
one sbuf tile, a 5-D .ap() view, nl.sum over the window axes, tensor_scalar, dma_copy out -- and
scored 0.50 instead of the baseline's 0.30. Every remaining failure was the same line:

    view = tile.ap([[C, C], [H, H // p], [W, W // p], [p, p], [p, p]])
    -> ap() pattern has invalid partition stride. Partition step 32 must equal tensor free
       dimension size 1024.

The model does not know that a stride is "how many elements one step along this axis skips", and
when told "must equal 1024" it changed 32 to 2. The simulator's message names the bad number but
not the rule, so this rule explains the rule, with the partition pair filled in for the shape at
hand so there is at least one anchor it can copy.

Second rule, from level 3 under v2. The kernel was right down to the last two lines:

    nisa.dma_copy(dst=nl.shared_hbm, src=sbuf_tile)
    return nl.shared_hbm
    -> 'MemoryRegion' object has no attribute 'buffer'

v2's direction text said "dma_copy sbuf -> the shared_hbm output" and the model read nl.shared_hbm
itself as that output. The stock feedback ("a MemoryRegion is not a numpy array") names the type
and not the fix. This rule gives the allocation line, and the v2 direction text for the matmul
levels is reworded to spell out the allocation.

Third, from level 4 under v2 (0.62 -> 0.75, two of four shapes). The K-loop accumulation into one
PSUM tile was right -- the K=512 shape passed -- but v2's "each dimension is its own loop" was read
as three loops one after another, and lhsT was sliced on K only (lhsT[k0:k0+128, :]), so M=256
overflowed the (128, 128) tile. v3 says NESTED, gives the nesting order, and writes both slice
axes out. The dma_copy size-mismatch rule also gains one sentence for the "sliced one axis only"
case, which is what an integer src/dst ratio means.
"""

import re
import sys

import agent_feedback_v2 as v2


def ap_hint(error_text):
    """Explain stride arithmetic for a row-major (C, H, W) tile, with the shape's own numbers."""
    m = re.search(r"tensor shape: \((\d+), (\d+), (\d+)\)", error_text)
    if m:
        C, H, W = (int(g) for g in m.groups())
        anchor = (f" For this tensor shape ({C}, {H}, {W}) the partition pair is "
                  f"[{H * W}, {C}], i.e. [H*W, C].")
    else:
        anchor = " The partition pair is [H*W, C]."
    return (
        " A stride is the number of ELEMENTS skipped when the index of that axis increases by one, "
        "in the tile's row-major (C, H, W) layout: one step along C skips H*W elements, one step "
        "along H skips W elements, one step along W skips 1 element. So an axis of the view that "
        "advances k rows at a time has stride k*W, and one that advances k columns at a time has "
        "stride k." + anchor +
        " Then for the 5-D view (C, H//p, W//p, p, p): the second axis advances p rows per step, "
        "the third advances p columns per step, the fourth advances 1 row per step, the fifth "
        "advances 1 column per step. Write every stride as an expression of H, W and p (not as a "
        "bare number), keep the counts as they are, and change nothing else in the kernel.")


MEMREGION_HINT = (
    " nl.shared_hbm, nl.sbuf and nl.psum are memory REGIONS, not tensors: you cannot copy into "
    "them or return them. Allocate the output tensor first, with the result shape, and use THAT: "
    "out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm); ... "
    "nisa.dma_copy(dst=out, src=result_sbuf_tile); return out. Same for inputs: never pass an "
    "argument tensor (which lives in HBM) straight into nc_matmul; dma_copy it into an sbuf tile "
    "first and pass the tile.")

OUTPUT_LINE = (" Allocate the output ONCE at the top with "
               "out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm) and return out at "
               "the end; nl.shared_hbm by itself is a memory region, not a tensor.")

for _lv in (3, 4):
    v2.DIRECTION[_lv] = (v2.DIRECTION[_lv]
                         .replace("the shared_hbm output of shape (M, N)",
                                  "the output tensor out of shape (M, N)")
                         .replace("dma_copy sbuf->the shared_hbm output slice",
                                  "dma_copy sbuf->out[slice]")
                         .replace("dma_copy sbuf->the shared_hbm output",
                                  "dma_copy sbuf->out")
                         + OUTPUT_LINE)
v2.DIRECTION[4] = (
    "Layout facts for this matmul: lhsT is [K, M] and rhs is [K, N], so K is the PARTITION axis "
    "of BOTH inputs. The RESULT is [M, N] = (lhsT.shape[1], rhs.shape[1]) with M on the partition "
    "axis. Allocate it ONCE at the top with out = nl.ndarray((M, N), dtype=lhsT.dtype, "
    "buffer=nl.shared_hbm) and return out at the end; nl.shared_hbm by itself is a memory region, "
    "not a tensor. One nc_matmul consumes a stationary tile of lhsT that is 128 deep in K and 128 "
    "wide in M, and a moving tile of rhs that is 128 deep in K and 512 wide in N, and writes a "
    "(128, 512) float32 tile in psum.\n"
    "The tiling is THREE NESTED LOOPS, one inside the other, not three loops one after another:\n"
    "  outer loop i over M // 128 (output row block, m0 = i*128)\n"
    "    middle loop j over N // 512 (output column block, n0 = j*512)\n"
    "      allocate ONE psum tile (128, 512) here, for this output block\n"
    "      inner loop k over K // 128 (contraction, k0 = k*128):\n"
    "        sbuf tile a (128, 128) <- dma_copy from lhsT[k0:k0+128, m0:m0+128]  (slice BOTH axes)\n"
    "        sbuf tile b (128, 512) <- dma_copy from rhs[k0:k0+128, n0:n0+512]   (slice BOTH axes)\n"
    "        nc_matmul(dst=psum, stationary=a, moving=b)   -- accumulates across k\n"
    "      after the k loop: tensor_copy psum -> sbuf tile (128, 512), then dma_copy that to "
    "out[m0:m0+128, n0:n0+512]\n"
    "Every slice has two ranges, never a bare ':'. Derive the loop counts from .shape; the test "
    "shapes are exact multiples. Never allocate any tile with more than 128 on its first dimension.")
for _lv in (5, 6, 7):
    v2.DIRECTION[_lv] = v2.DIRECTION[4]

ONE_AXIS_HINT = (" The source has exactly {ratio}x the elements of the tile, which means you sliced "
                 "only ONE axis of the source and left the other as ':'. Slice both axes to the "
                 "tile's shape, e.g. lhsT[k0:k0+128, m0:m0+128] for a (128, 128) tile.")


def install_feedback(base):
    originals = v2.install_feedback(base)
    prev = base.enrich

    def enrich(error_text):
        if "ap() pattern" in error_text:
            return error_text + ap_hint(error_text)
        if "'MemoryRegion' object" in error_text or "got private_hbm" in error_text \
                or "got shared_hbm" in error_text:
            return error_text + MEMREGION_HINT
        m = re.search(r"dma_copy requires src and dst to have the same number of elements, "
                      r"got src=(\d+), dst=(\d+)", error_text)
        if m and int(m.group(2)) and int(m.group(1)) % int(m.group(2)) == 0 \
                and int(m.group(1)) > int(m.group(2)):
            return prev(error_text) + ONE_AXIS_HINT.format(ratio=int(m.group(1)) // int(m.group(2)))
        return prev(error_text)

    base.enrich = enrich
    return originals


def check_feedback(base):
    originals = install_feedback(base)
    try:
        err = ("raised AssertionError: ap() pattern has invalid partition stride. Partition step 32 "
               "must equal tensor free dimension size 1024. Pattern: [[32, 32], [32, 16], [32, 16], "
               "[2, 2], [2, 2]], tensor shape: (32, 32, 32)")
        out = base.enrich(err)
        assert "[1024, 32]" in out and "k*W" in out
        print("enrich: ap() stride rule attached, partition pair resolved to [1024, 32]")
        other = "raised AttributeError: module 'nki.isa' has no attribute 'multiply'"
        assert base.enrich(other).endswith(v2.SCALING_HINT)
        print("enrich: v2/v1 rules still in place")
        for e in ("raised AttributeError: 'MemoryRegion' object has no attribute 'buffer'",
                  "raised AssertionError: stationary must be in ['sbuf'], got private_hbm"):
            assert base.enrich(e).endswith(MEMREGION_HINT)
        print("enrich: MemoryRegion / hbm-operand rule attached")
        for lv in (3, 4):
            d = v2.DIRECTION[lv]
            assert "buffer=nl.shared_hbm) and return out" in d and "the shared_hbm output" not in d
        print("direction: matmul levels now spell out the output allocation")
        assert "THREE NESTED LOOPS" in v2.DIRECTION[4] and "slice BOTH axes" in v2.DIRECTION[4]
        e = ("raised AssertionError: dma_copy requires src and dst to have the same number of "
             "elements, got src=32768, dst=16384")
        assert base.enrich(e).endswith(ONE_AXIS_HINT.format(ratio=2))
        e2 = "raised AssertionError: dma_copy requires src and dst to have the same number of elements, got src=4, dst=16384"
        assert not base.enrich(e2).endswith(ONE_AXIS_HINT.format(ratio=0))
        print("direction: level 4 nested-loop text; enrich: one-axis-slice rule on integer ratios")
        p = base.first_prompt(1)
        assert "returns shape" in p and "REDUCTION" in p
        print("first_prompt: v2 level direction and shape card present")
        print("\n=== the ap() feedback the model would see ===")
        print(out)
        print("\nFEEDBACK CHECK PASSED (routing only; no kernel/model run)")
    finally:
        base.enrich, base.first_prompt, base.repair_prompt = originals


if __name__ == "__main__":
    import agent as base

    if sys.argv[1:] == ["--check-feedback"]:
        check_feedback(base)
    else:
        install_feedback(base)
        print("Feedback experiment v3: v2 + ap() stride rule", flush=True)
        base.main()
