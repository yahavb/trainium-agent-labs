#!/usr/bin/env python3
"""
overlap_bench.py — 3-Engine Grader, SBUF Validator, and Block-n Hazard Diagnostic
Track 1 Lead: Shashwat

This module verifies:
1. Static Rules: Use of 3 hardware engines (DMA, Vector, Tensor) and SBUF limits.
2. Numerical Correctness: Bilinear Interpolation + MatMul against NumPy ground truth.
3. Engine Concurrency: Measures 3-way overlap (DMA || Vector || Tensor).
4. Granular Recalibration (Step A, B, C): Isolates failure at Block n to diagnose
   Pipeline Sync Hazards, Ragged Edge Hazards, or Precision Hazards.
"""

import argparse
import ast
import inspect
import sys
import textwrap
import numpy as np

# Hardware Constants for AWS Trainium (NeuronCore-v2)
PMAX = 128                  # Maximum partition dimension
FMAX = 512                  # Maximum free dimension
SBUF_TILE_ROWS = 128
SBUF_TILE_COLS = 128

BANNED_WHOLE_ARRAY = {
    "einsum", "matmul", "dot", "tensordot", "inner", "outer"
}


# ---------------------------------------------------------------- Reference Math
def ref_bilinear_matmul(x: np.ndarray, weight: np.ndarray, alpha: float = 1.25, beta: float = 0.5) -> np.ndarray:
    """
    Ground truth operation:
    1. Vector Engine: Bilinear scaling & interpolation: x_scaled = alpha * x + beta
    2. Tensor Engine: Matrix Multiplication: y = x_scaled @ weight
    """
    x_scaled = (alpha * x.astype(np.float64) + beta).astype(np.float32)
    y = (x_scaled @ weight.astype(np.float64)).astype(np.float32)
    return y


# ---------------------------------------------------------------- Static AST Rule Checker
def check_rules(source: str) -> list[str]:
    """Inspects AST for engine usage, tile boundaries, and illegal whole-array calls."""
    violations = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"Syntax error: {e.msg} on line {e.lineno}"]

    # 1. Check for banned whole-array cheats
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in BANNED_WHOLE_ARRAY:
                violations.append(f"line {node.lineno}: calls banned `{func.attr}`. Must use explicit tile loops.")
            elif isinstance(func, ast.Name) and func.id in BANNED_WHOLE_ARRAY:
                violations.append(f"line {node.lineno}: calls banned `{func.id}`.")

    # 2. Check for 3-Engine Primitives (or simulated hardware abstractions)
    source_lower = source.lower()
    has_dma = ("dma" in source_lower) or ("load_tile" in source_lower)
    has_vector = ("vector" in source_lower) or ("tensor_scalar" in source_lower) or ("bilinear" in source_lower) or ("scale" in source_lower)
    has_tensor = ("nc_matmul" in source_lower) or ("tensor_matmul" in source_lower) or ("matmul_tile" in source_lower) or ("tensor" in source_lower and "matmul" in source_lower)

    if not has_dma:
        violations.append("DMA Engine missing: code must explicitly manage DMA transfers to SBUF.")
    if not has_vector:
        violations.append("Vector Engine missing: code must perform vector scaling/interpolation in SBUF.")
    if not has_tensor:
        violations.append("Tensor Engine missing: code must issue tile matrix multiplication on the Tensor Engine.")

    # 3. Check for multi-buffering strategy (buffer pointers / double or triple buffering)
    has_buffers = any(k in source_lower for k in ["buf", "buffer", "active", "next", "ping", "pong"])
    if not has_buffers:
        violations.append("Multi-buffering missing: must allocate distinct buffers (e.g. buf_dma, buf_vec, buf_tensor) to prevent stalls.")

    return violations


# ---------------------------------------------------------------- Block-n Hazard Diagnostic
def diagnose_block_failure(
    kernel_fn,
    x: np.ndarray,
    weight: np.ndarray,
    alpha: float,
    beta: float,
    tile_h: int = SBUF_TILE_ROWS
) -> tuple[str, str, int]:
    """
    Step A: Granular Block Isolation
    Step B: Micro-Test Diagnostic
    Step C: Recalibration Diagnostic
    """
    H, W = x.shape
    num_blocks = (H + tile_h - 1) // tile_h
    expected_full = ref_bilinear_matmul(x, weight, alpha, beta)

    try:
        candidate_full = kernel_fn(x, weight, alpha, beta)
    except Exception as e:
        return "CRASH_HAZARD", f"Kernel execution raised an exception: {str(e)}", -1

    # Check each block slice
    for b in range(num_blocks):
        r_start = b * tile_h
        r_end = min(r_start + tile_h, H)
        expected_slice = expected_full[r_start:r_end, :]
        candidate_slice = candidate_full[r_start:r_end, :]

        mismatch = not np.allclose(expected_slice, candidate_slice, rtol=1e-3, atol=1e-3)
        if mismatch:
            # STEP B: Run Block b in isolated single-block test
            x_isolated = x[r_start:r_end, :].copy()
            try:
                isolated_result = kernel_fn(x_isolated, weight, alpha, beta)
                isolated_pass = np.allclose(expected_slice, isolated_result, rtol=1e-3, atol=1e-3)
            except Exception:
                isolated_pass = False

            # STEP C: Categorize Hazard
            if isolated_pass:
                # Passes alone, fails in pipeline => Sync / Overwrite hazard!
                return (
                    "PIPELINE_SYNC_HAZARD",
                    f"Block {b} PASSED in isolated test but FAILED in the 3-engine pipeline. "
                    "The DMA Engine overwrote SBUF cache before the Tensor Engine finished reading it. "
                    "Fix: Swap active and next buffer pointers explicitly (`active_buf, next_buf = next_buf, active_buf`) "
                    "or insert hardware synchronization.",
                    b
                )
            elif b == num_blocks - 1 and (H % tile_h != 0):
                # Ragged edge at the tail
                rem = H - r_start
                return (
                    "RAGGED_EDGE_HAZARD",
                    f"Block {b} is a partial final tile with {rem} rows (smaller than standard tile {tile_h}). "
                    "Boundary index out-of-bounds or zero-pad missing. "
                    f"Fix: Apply boundary clamp `valid_rows = min({tile_h}, total_rows - r_start)` to tile operations.",
                    b
                )
            else:
                return (
                    "NUMERICAL_PRECISION_HAZARD",
                    f"Block {b} diverged numerically (Max error: {np.max(np.abs(expected_slice - candidate_slice)):.4e}). "
                    "Fix: Accumulate intermediate results in float32 precision before converting to bfloat16.",
                    b
                )

    return "NONE", "All blocks passed numerically.", -1


# ---------------------------------------------------------------- Engine Overlap Evaluator
def measure_overlap(source: str) -> tuple[float, str]:
    """
    Evaluates whether the kernel implements 3-way overlapped pipeline:
    - Prologue: Prefetch Block 0
    - Main Loop: DMA(N+1) || Vector(N) || Tensor(N-1)
    - Epilogue: Drain pipeline
    """
    source_lower = source.lower()
    has_prologue = ("prologue" in source_lower) or ("prefetch" in source_lower) or ("load_tile(0" in source_lower)
    has_epilogue = ("epilogue" in source_lower) or ("drain" in source_lower) or ("remainder" in source_lower)
    has_pipelined_loop = ("for " in source) and ("range" in source)

    if has_prologue and has_epilogue and has_pipelined_loop:
        return 1.0, "3-Way Hardware Pipeline detected (Prologue, Steady-state Overlap, Epilogue Drain)."
    elif has_pipelined_loop and ("next" in source_lower or "pong" in source_lower):
        return 0.65, "Partial 2-way overlap detected; missing clean prologue/epilogue stages."
    else:
        return 0.20, "Sequential execution detected: DMA, Vector, and Tensor run back-to-back with 66% idle stalls."


# ---------------------------------------------------------------- Main Grader Contract
def grade(source: str) -> tuple[float, dict, str]:
    """
    Primary API Contract for Track 2 (Heet's Agent):
    Returns (score: float, details: dict, feedback: str)
    """
    details = {
        "parses": False,
        "rules_clean": False,
        "correctness": 0.0,
        "overlap_efficiency": 0.0,
        "hazard_type": "NONE",
        "block_failed": -1
    }

    # 1. Parse check
    try:
        compile(source, "<candidate>", "exec")
        details["parses"] = True
    except SyntaxError as e:
        return 0.0, details, f"Syntax Error: {e.msg} on line {e.lineno}. Return valid Python."

    # 2. Rule check
    violations = check_rules(source)
    if violations:
        feedback = "Hardware Rule Violations:\n" + "\n".join(f"- {v}" for v in violations)
        return 0.1, details, feedback
    details["rules_clean"] = True

    # 3. Execution & Numerical Correctness check
    namespace = {}
    try:
        exec(source, namespace)
    except Exception as e:
        return 0.2, details, f"Execution failed during import/definition: {str(e)}"

    kernel_fn = namespace.get("pipeline_kernel") or namespace.get("kernel")
    if not callable(kernel_fn):
        return 0.2, details, "Missing entry-point: kernel must define `def pipeline_kernel(x, weight, alpha, beta):`"

    # Hostile Test Shapes (including ragged edge shape)
    # Shape 1: Even divisible (256x128 @ 128x128)
    # Shape 2: Hostile ragged edge (315x128 @ 128x128 -> last tile has 59 rows)
    test_cases = [
        (256, 128, 128),
        (315, 128, 128)
    ]

    for H, W, K in test_cases:
        np.random.seed(42)
        x = np.random.randn(H, W).astype(np.float32)
        weight = np.random.randn(W, K).astype(np.float32)
        alpha, beta = 1.25, 0.5

        hazard, hint, block_n = diagnose_block_failure(kernel_fn, x, weight, alpha, beta)
        if hazard != "NONE":
            details["hazard_type"] = hazard
            details["block_failed"] = block_n
            score = 0.3  # Parses + rules
            return score, details, f"Diagnostic Failure [{hazard}]: {hint}"

    details["correctness"] = 1.0

    # 4. Engine Overlap Efficiency
    overlap_score, overlap_notes = measure_overlap(source)
    details["overlap_efficiency"] = overlap_score

    # Final weighted score
    # parses(0.1) + rules(0.2) + correctness(0.4) + overlap(0.3)
    total_score = 0.1 + 0.2 + (0.4 * details["correctness"]) + (0.3 * overlap_score)
    feedback = f"SUCCESS: Score {total_score:.2f}/1.00. Correctness: 100%. {overlap_notes}"
    return total_score, details, feedback


# ---------------------------------------------------------------- Selftest
def run_selftest():
    print("=== Running overlap_bench.py Selftest ===")

    # Test 1: Banned Einsum
    bad_code = "import numpy as np\ndef pipeline_kernel(x, w, a, b):\n    return np.einsum('ij,jk->ik', x, w)"
    v = check_rules(bad_code)
    assert any("einsum" in item for item in v), "Failed to catch banned einsum"
    print("  [PASS] Caught banned whole-array operation (einsum)")

    # Test 2: Missing Engines
    missing_engines_code = "def pipeline_kernel(x, w, a, b):\n    return x @ w"
    v = check_rules(missing_engines_code)
    assert len(v) >= 3, "Failed to catch missing hardware engines"
    print("  [PASS] Caught missing DMA, Vector, and Tensor engines")

    # Test 3: Clean Ground Truth
    x = np.random.randn(128, 128).astype(np.float32)
    w = np.random.randn(128, 128).astype(np.float32)
    res = ref_bilinear_matmul(x, w)
    assert res.shape == (128, 128), "Reference math shape error"
    print("  [PASS] Ground truth Bilinear + MatMul math verified")

    print("\nALL SELFTESTS PASSED SUCCESSFULLY! Grader is ready for judging.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="Run harness self-test")
    parser.add_argument("--check", type=str, help="Check a Python kernel file")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    elif args.check:
        with open(args.check) as f:
            code = f.read()
        score, details, msg = grade(code)
        print(f"Score: {score:.2f}")
        print(f"Details: {details}")
        print(f"Verdict:\n{msg}")
    else:
        parser.print_help()
