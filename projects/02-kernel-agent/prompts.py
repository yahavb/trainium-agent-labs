"""Select a compact, level-relevant NKI API card for first_prompt."""

API_BY_LEVEL = {
    1: """Relevant NKI APIs: nl.ndarray, tile.ap access-pattern views, nl.sum(view, axis=[...]), nisa.tensor_scalar, nisa.dma_copy, nisa.tensor_copy. Use the supplied reference's supported signatures.""",
    2: """Relevant NKI APIs: nl.ndarray, nl.affine_range, nl.ds, nisa.dma_copy, nisa.tensor_copy. Use the supplied reference's supported signatures.""",
    3: """Relevant NKI APIs: nl.ndarray, nisa.dma_copy, nisa.nc_matmul(dst=, stationary=, moving=), nisa.tensor_copy. nc_matmul reads both operands from SBUF and writes to PSUM.""",
    4: """Relevant NKI APIs: nl.ndarray, nl.affine_range, nisa.dma_copy, nisa.nc_matmul(dst=, stationary=, moving=), nisa.tensor_copy. nc_matmul reads both operands from SBUF and writes to PSUM.""",
    5: """Relevant NKI APIs: nl.ndarray, nl.affine_range, nisa.dma_copy, nisa.nc_matmul(dst=, stationary=, moving=), nisa.tensor_copy. Reuse loaded SBUF tiles across the relevant loop.""",
    6: """Relevant NKI APIs: nl.ndarray, nl.affine_range, nisa.dma_copy, nisa.nc_matmul(dst=, stationary=, moving=), nisa.tensor_copy. Keep reused tiles within the existing SBUF layout and capacity.""",
    7: """Relevant NKI APIs: nl.ndarray, nl.affine_range, nisa.dma_copy, nisa.nc_matmul(dst=, stationary=, moving=), nisa.tensor_copy. Accumulate K tiles into the same PSUM result before copying it out.""",
    8: """Relevant NKI APIs: nl.ndarray, nl.affine_range, nl.max, nl.sum, nl.exp, nisa.dma_copy, nisa.nc_matmul, nisa.tensor_copy. Follow the supplied reference for tile layouts and operations.""",
}


def level_prompt_context(level, pmax, stationary_fmax, moving_fmax, terse=0):
    """Build prompt context without changing first_prompt's interface or output contract."""
    from kernel_guides import guide_for_level

    api = API_BY_LEVEL.get(level, API_BY_LEVEL[3])
    if level == 1:
        api += f" Checker partition limit: at most {pmax}."
    elif level >= 3 and level <= 7:
        api += (
            f" Checker limits: partition <= {pmax}, stationary free dimension <= "
            f"{stationary_fmax}, moving free dimension <= {moving_fmax}."
        )
    guide = guide_for_level(level)
    return "\n\n".join(part for part in (api, guide) if part)
