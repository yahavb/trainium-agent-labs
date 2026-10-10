"""Level-specific NKI guidance derived from checker categories and local references.

Category names below are stable labels from the feedback schema. This module does not
classify errors or change checker behavior; it turns known failure modes into generation
constraints and a short private self-check.
"""

GUIDES = {
    1: """Level 1 shape plan (average pooling):
- Input is (C,H,W), pool_size is p, and output must be (C,H//p,W//p). Each output is the sum of its non-overlapping p-by-p input window divided by p*p.
- Use an access-pattern view whose logical shape is (C,H//p,W//p,p,p), then reduce only the final two window axes with nl.sum. Do not call a NumPy/PyTorch pooling or mean operation.
- DMA source and destination must contain the same number of elements. Derive the SBUF tile from the exact input slice/view; never allocate an arbitrary large tile for a small window. Keep C on the partition axis and ensure it does not exceed the supplied partition limit.
- Checker-to-constraint: DMA_SIZE_MISMATCH/TILE_SHAPE_MISMATCH => match source, tile, and destination slice element counts and shapes; PARTITION_LIMIT/OUT_OF_BOUNDS => derive dimensions from input shape and keep all indices in range; NUMERICAL_MISMATCH => verify both p-by-p axes are reduced and divide by p*p.
Before emitting code, privately verify the output shape, the two reduction axes, normalization, and every DMA shape. Return only the requested Python code block.""",
    3: """Level 3 shape and dataflow plan (single-tile matmul):
- lhsT has shape (K,M), rhs has shape (K,N), and result has shape (M,N). The first dimension K is the contraction/partition dimension in both inputs.
- Keep every SBUF and PSUM tile explicitly 2-D: lhs (K,M), rhs (K,N), PSUM/output (M,N). Do not allocate a 1-D PSUM from lhsT.shape[1:].
- Use nisa.dma_copy HBM->SBUF for each operand; nisa.nc_matmul(dst=psum, stationary=lhs_sbuf, moving=rhs_sbuf); nisa.tensor_copy PSUM->SBUF; then nisa.dma_copy SBUF->shared_hbm output. Use keyword arguments for nc_matmul.
- For this level's supplied shape (K=128,M=64,N=512), each operand and result fits one tile under the checker-provided limits; do not pad dimensions or invent a K loop. For other levels, tile each dimension from its actual bounds.
- Checker-to-constraint: TILE_RANK => make each on-chip tile 2-D (partition, free); BUFFER_PLACEMENT => operands in SBUF and nc_matmul destination in PSUM; DMA_SIZE_MISMATCH/TILE_SHAPE_MISMATCH/OUT_OF_BOUNDS => copy matching in-range slices; NUMERICAL_MISMATCH => preserve K contraction and (M,N) result indexing.
Before emitting code, privately verify all tile shapes, buffer regions, the (M,N) result, and the copy sequence. Return only the requested Python code block.""",
}


def guide_for_level(level):
    """Return only guidance relevant to this level."""
    return GUIDES.get(level, "")
