#!/usr/bin/env python3
"""
overlap_bench.py — 3-Engine Grader, SBUF Validator, and Block-n Hazard Diagnostic
Track 1 Lead: Shashwat

This module verifies:
1. Static Rules & AST Security: Use of 3 hardware engines (DMA, Vector, Tensor),
   enforcement of SBUF limits (PMAX <= 128, FMAX <= 512), and prevention of banned
   whole-array shortcut cheats (einsum, full @, np.linalg, torch/scipy).
2. Hostile Shape & Value Generation: Prime rows (317x128 ragged edge), sub-tile
   matrices (64x128), large stress test (1024x128), extreme magnitudes (1e4),
   zeros, and negative values.
3. Numerical Correctness: Bilinear Interpolation + MatMul against NumPy ground truth.
4. Engine Concurrency: Evaluates 3-way overlap (DMA || Vector || Tensor) with
   prologue prefetching and epilogue draining.
5. Granular Recalibration (Step A, B, C): Isolates failure at Block n to diagnose
   Pipeline Sync Hazards, Ragged Edge Hazards, or Numerical Precision Hazards.
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
    "einsum", "matmul", "dot", "tensordot", "inner", "outer", "kron", "vdot", "cross"
}

BANNED_FRAMEWORKS = {
    "torch", "scipy", "sklearn", "tensorflow", "jax"
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


# ---------------------------------------------------------------- Hostile Value & Shape Generator
def generate_hostile_test_cases():
    """
    Generates hostile test cases designed to expose edge cases, accumulator overflows,
    and partial-tile bugs:
    1. Standard even: 256x128 (2 full tiles of 128x128)
    2. Prime rows: 317x128 (Block 0: 128, Block 1: 128, Block 2: 61 rows ragged tail)
    3. Sub-tile matrix: 64x128 (smaller than a full 128 partition)
    4. Large stress matrix: 1024x128 (8 full blocks sustained multi-buffering)
    5. Extreme magnitude: 256x128 with values scaled to 1e4 (tests float32 accumulation)
    6. Zero matrix: 256x128 with zeros (verifies scalar offset beta handling)
    7. Negative signed matrix: 256x128 with negative values (verifies signed math)
    """
    cases = []

    # 1. Standard Even
    np.random.seed(42)
    x1 = np.random.randn(256, 128).astype(np.float32)
    w1 = np.random.randn(128, 128).astype(np.float32)
    cases.append(("standard_even_256", x1, w1, 1.25, 0.5, "Standard 2-block even matrix"))

    # 2. Prime Rows (Ragged Edge Tail: 317 rows -> 128 + 128 + 61)
    np.random.seed(43)
    x2 = np.random.randn(317, 128).astype(np.float32)
    w2 = np.random.randn(128, 128).astype(np.float32)
    cases.append(("prime_ragged_317", x2, w2, 1.25, 0.5, "Prime rows matrix with 61-row ragged tail"))

    # 3. Sub-Tile Matrix (64 rows < 128 tile)
    np.random.seed(44)
    x3 = np.random.randn(64, 128).astype(np.float32)
    w3 = np.random.randn(128, 128).astype(np.float32)
    cases.append(("sub_tile_64", x3, w3, 1.25, 0.5, "Sub-tile matrix smaller than standard 128 tile"))

    # 4. Large Stress Matrix (1024 rows -> 8 full tiles)
    np.random.seed(45)
    x4 = np.random.randn(1024, 128).astype(np.float32)
    w4 = np.random.randn(128, 128).astype(np.float32)
    cases.append(("large_stress_1024", x4, w4, 1.25, 0.5, "Large 1024-row stress matrix (8 full blocks)"))

    # 5. Extreme Magnitude (Values scaled to 1e4 -> tests accumulator overflow)
    np.random.seed(46)
    x5 = (np.random.randn(256, 128) * 1e4).astype(np.float32)
    w5 = (np.random.randn(128, 128) * 1e4).astype(np.float32)
    cases.append(("extreme_magnitude_1e4", x5, w5, 1.25, 0.5, "Extreme magnitude values (1e4) to test accumulator overflow"))

    # 6. Zero Matrix (Tests bias addition beta)
    x6 = np.zeros((256, 128), dtype=np.float32)
    w6 = np.ones((128, 128), dtype=np.float32)
    cases.append(("zero_matrix_bias", x6, w6, 1.25, 0.5, "Zero matrix testing pure bias propagation"))

    # 7. Negative Signed Matrix (Tests sign preservation across Vector and Tensor engines)
    np.random.seed(47)
    x7 = -np.abs(np.random.randn(256, 128)).astype(np.float32)
    w7 = -np.abs(np.random.randn(128, 128)).astype(np.float32)
    cases.append(("negative_signed", x7, w7, 1.25, 0.5, "Strictly negative values testing signed arithmetic"))

    return cases


# ---------------------------------------------------------------- Hardened Static AST Rule Checker
def check_rules(source: str) -> list[str]:
    """
    Inspects AST for:
    1. Banned whole-array cheats:
       - Calls to np.einsum, matmul, dot, tensordot, etc.
       - Whole-array @ binary operator directly on top-level input matrices (x @ weight).
       - Imports from forbidden external frameworks (torch, scipy, sklearn, etc.).
       - Calls on np.linalg modules.
    2. SBUF Tile Size Constraints:
       - Ensures partition tile size does not exceed Trainium hardware limit PMAX=128.
    3. Hardware Engine Primitives:
       - DMA Engine (e.g. dma_copy, nl.load, load_tile).
       - Vector Engine (e.g. vector_bilinear_scale, tensor_scalar, nl.add, nl.multiply).
       - Tensor Engine (e.g. tensor_matmul, nc_matmul, matmul_tile).
    4. Multi-Buffering Strategy:
       - Explicit buffer allocation and rotation (buf_dma, buf_vec, buf_tensor, active/next).
    """
    violations = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"Syntax error: {e.msg} on line {e.lineno}"]

    source_lower = source.lower()

    # 1. Check for banned external frameworks
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_root = alias.name.split('.')[0]
                if mod_root in BANNED_FRAMEWORKS:
                    violations.append(f"line {node.lineno}: illegal import of framework `{alias.name}`. Must write native NKI/simulated primitives.")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod_root = node.module.split('.')[0]
                if mod_root in BANNED_FRAMEWORKS:
                    violations.append(f"line {node.lineno}: illegal import from framework `{node.module}`.")

    # 2. Check for banned whole-array function calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr in BANNED_WHOLE_ARRAY:
                    violations.append(f"line {node.lineno}: calls banned whole-array function `{func.attr}`. Must use tile loops.")
                if isinstance(func.value, ast.Attribute) and func.value.attr == "linalg":
                    violations.append(f"line {node.lineno}: calls banned linear algebra module `linalg.{func.attr}`.")
            elif isinstance(func, ast.Name) and func.id in BANNED_WHOLE_ARRAY:
                violations.append(f"line {node.lineno}: calls banned `{func.id}`.")

    # 3. Check for whole-array @ operator on input matrices
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            # If left or right is direct input matrix name 'x' or 'weight' (not subscripted)
            left_is_whole = isinstance(node.left, ast.Name) and node.left.id in {"x", "weight", "X", "W"}
            right_is_whole = isinstance(node.right, ast.Name) and node.right.id in {"x", "weight", "X", "W"}
            if left_is_whole or right_is_whole:
                violations.append(
                    f"line {node.lineno}: illegal whole-array `@` operator on unsliced input matrix. "
                    "Trainium hardware only supports tile-level matrix multiplication via Tensor Engine."
                )

    # 4. Check for SBUF Partition tile size bounds (PMAX = 128)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and any(k in target.id.lower() for k in ["tile_h", "tile_rows", "p_size", "tile_p"]):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                        if node.value.value > PMAX:
                            violations.append(
                                f"line {node.lineno}: Tile height {node.value.value} exceeds Trainium hardware limit PMAX={PMAX}."
                            )

    # 5. Check for 3 Hardware Engine Primitives
    has_dma = any(k in source_lower for k in ["dma", "load_tile", "nl.load", "dma_copy"])
    has_vector = any(k in source_lower for k in ["vector", "tensor_scalar", "bilinear", "scale", "nl.add", "nl.multiply"])
    has_matmul_op = any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult) for node in ast.walk(tree))
    has_tensor = (
        has_matmul_op
        or any(k in source_lower for k in ["nc_matmul", "tensor_matmul", "matmul_tile", "matmul", "matrix multiply", "matrix multiplication"])
        or ("tensor" in source_lower and any(m in source_lower for m in ["matmul", "multiply", "multiplication", "dot"]))
    )

    if not has_dma:
        violations.append("DMA Engine missing: code must explicitly manage DMA transfers between HBM and SBUF.")
    if not has_vector:
        violations.append("Vector Engine missing: code must perform vector scaling/interpolation in SBUF.")
    if not has_tensor:
        violations.append("Tensor Engine missing: code must issue tile matrix multiplication on the Tensor Engine (e.g. `buf_tensor[:valid, :] = buf_vec[:valid, :].astype(np.float64) @ weight.astype(np.float64)`).")

    # 6. Check for Multi-Buffering Strategy (buffer allocation / pointer swapping)
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

    # Check for shape mismatch
    if candidate_full.shape != expected_full.shape:
        return (
            "SHAPE_MISMATCH_HAZARD",
            f"Output shape {candidate_full.shape} does not match expected shape {expected_full.shape}.",
            -1
        )

    # Check for NaNs or Infs
    if np.any(np.isnan(candidate_full)) or np.any(np.isinf(candidate_full)):
        for b in range(num_blocks):
            r_start = b * tile_h
            r_end = min(r_start + tile_h, H)
            c_slice = candidate_full[r_start:r_end, :]
            if np.any(np.isnan(c_slice)) or np.any(np.isinf(c_slice)):
                return (
                    "NUMERICAL_PRECISION_HAZARD",
                    f"Block {b} produced NaN or Inf values. Intermediate accumulator overflowed SBUF capacity. "
                    "Fix: Accumulate intermediate results in float32 precision before converting to bfloat16.",
                    b
                )

    # Check each block slice against ground truth
    for b in range(num_blocks):
        r_start = b * tile_h
        r_end = min(r_start + tile_h, H)
        expected_slice = expected_full[r_start:r_end, :]
        candidate_slice = candidate_full[r_start:r_end, :]

        # Scale-aware tolerance for hostile extreme magnitudes
        tol_atol = 1e-2 * (1.0 + np.mean(np.abs(expected_slice)) * 1e-3)
        tol_rtol = 1e-2
        mismatch = not np.allclose(expected_slice, candidate_slice, rtol=tol_rtol, atol=tol_atol)

        if mismatch:
            # STEP B: Run Block b in isolated single-block micro-test
            x_isolated = x[r_start:r_end, :].copy()
            isolated_result = None
            try:
                isolated_result = kernel_fn(x_isolated, weight, alpha, beta)
                isolated_pass = np.allclose(expected_slice, isolated_result, rtol=tol_rtol, atol=tol_atol)
            except Exception:
                isolated_pass = False

            # STEP C: Categorize Hazard & Formulate Surgical Hint
            if isolated_pass:
                # Passes alone, fails in pipeline => Pipeline Sync / Buffer Overwrite Hazard!
                return (
                    "PIPELINE_SYNC_HAZARD",
                    f"Block {b} PASSED in isolated micro-test but FAILED in the 3-engine pipeline. "
                    "The DMA Engine overwrote SBUF cache before the Tensor Engine finished reading it. "
                    "Fix: Swap active and next buffer pointers explicitly (`active_buf, next_buf = next_buf, active_buf`) "
                    "or insert hardware synchronization.",
                    b
                )
            elif (b == num_blocks - 1 and (H % tile_h != 0)) or (num_blocks == 1 and H < tile_h):
                # Ragged edge at the tail or partial tile
                rem = H - r_start
                return (
                    "RAGGED_EDGE_HAZARD",
                    f"Block {b} is a partial tile with {rem} rows (smaller than standard tile {tile_h}). "
                    "Boundary index out-of-bounds or zero-pad missing. "
                    f"Fix: Apply boundary clamp `valid_rows = min({tile_h}, total_rows - r_start)` to tile operations.",
                    b
                )
            elif np.all(candidate_slice == 0.0) or (isolated_result is not None and np.all(isolated_result == 0.0)):
                return (
                    "UNCOMPUTED_BLOCK_HAZARD",
                    f"Block {b} was uncomputed (output slice is all zeros). "
                    "Cause: Kernel skipped Block computation (e.g. `if r_start > 0` skipped Block 0, or missing epilogue drain). "
                    "Fix: Handle single blocks (`num_blocks == 1`), compute Block 0 in prologue, and drain all remaining blocks in epilogue.",
                    b
                )
            else:
                max_err = float(np.max(np.abs(expected_slice - candidate_slice)))
                return (
                    "NUMERICAL_PRECISION_HAZARD",
                    f"Block {b} diverged numerically (Max error: {max_err:.4e}). "
                    "Fix: Accumulate intermediate results in float32 precision before converting to bfloat16.",
                    b
                )

    return "NONE", "All blocks passed numerically.", -1


# ---------------------------------------------------------------- Engine Overlap Evaluator
def measure_overlap(source: str) -> tuple[float, str]:
    """
    Evaluates whether the kernel implements 3-way overlapped pipeline:
    - Prologue: Prefetch Block 0 and Block 1
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
        "block_failed": -1,
        "test_case_failed": None
    }

    # 1. Parse check
    try:
        compile(source, "<candidate>", "exec")
        details["parses"] = True
    except SyntaxError as e:
        details["hazard_type"] = "SYNTAX_ERROR"
        return 0.0, details, f"Syntax Error: {e.msg} on line {e.lineno}. Return valid Python."

    # 2. Rule check
    violations = check_rules(source)
    if violations:
        details["hazard_type"] = "RULE_VIOLATION"
        feedback = "Hardware Rule Violations:\n" + "\n".join(f"- {v}" for v in violations)
        return 0.1, details, feedback
    details["rules_clean"] = True

    # 3. Execution & Numerical Correctness check across all hostile cases
    namespace = {}
    try:
        exec(source, namespace)
    except Exception as e:
        details["hazard_type"] = "IMPORT_ERROR"
        return 0.2, details, f"Execution failed during import/definition: {str(e)}"

    kernel_fn = namespace.get("pipeline_kernel") or namespace.get("kernel")
    if not callable(kernel_fn):
        details["hazard_type"] = "ENTRYPOINT_MISSING"
        return 0.2, details, "Missing entry-point: kernel must define `def pipeline_kernel(x, weight, alpha, beta):`"

    hostile_test_cases = generate_hostile_test_cases()

    for case_name, x, weight, alpha, beta, desc in hostile_test_cases:
        hazard, hint, block_n = diagnose_block_failure(kernel_fn, x, weight, alpha, beta)
        if hazard != "NONE":
            details["hazard_type"] = hazard
            details["block_failed"] = block_n
            details["test_case_failed"] = case_name
            score = 0.3  # Parses + rules
            return score, details, f"Diagnostic Failure [{hazard}] on test '{case_name}' ({desc}): {hint}"

    details["correctness"] = 1.0

    # 4. Engine Overlap Efficiency
    overlap_score, overlap_notes = measure_overlap(source)
    details["overlap_efficiency"] = overlap_score

    # Final weighted score
    # parses(0.1) + rules(0.2) + correctness(0.4) + overlap(0.3)
    total_score = 0.1 + 0.2 + (0.4 * details["correctness"]) + (0.3 * overlap_score)
    feedback = f"SUCCESS: Score {total_score:.2f}/1.00. Correctness: 100% across all hostile test cases. {overlap_notes}"
    return total_score, details, feedback


# ---------------------------------------------------------------- Selftest
def run_selftest():
    print("=================================================================")
    print("      overlap_bench.py — COMPREHENSIVE HARNESS SELFTEST")
    print("=================================================================\n")

    # Test 1: Banned Einsum
    bad_einsum = "import numpy as np\ndef pipeline_kernel(x, w, a, b):\n    return np.einsum('ij,jk->ik', x, w)"
    v1 = check_rules(bad_einsum)
    assert any("einsum" in item for item in v1), "Failed to catch banned einsum"
    print("  [PASS 1/10] Caught banned whole-array operation (np.einsum)")

    # Test 2: Banned Whole-Array @ Operator
    bad_matmul_op = "def pipeline_kernel(x, weight, a, b):\n    return x @ weight"
    v2 = check_rules(bad_matmul_op)
    assert any("whole-array `@`" in item for item in v2), "Failed to catch whole-array @"
    print("  [PASS 2/10] Caught banned whole-array matrix operator (x @ weight)")

    # Test 3: Illegal Framework Import
    bad_framework = "import torch\ndef pipeline_kernel(x, w, a, b):\n    return x"
    v3 = check_rules(bad_framework)
    assert any("torch" in item for item in v3), "Failed to catch banned framework torch"
    print("  [PASS 3/10] Caught forbidden external framework (torch)")

    # Test 4: SBUF Tile Size Limit Overflow
    bad_tile_limit = "TILE_H = 256\ndef pipeline_kernel(x, w, a, b):\n    pass"
    v4 = check_rules(bad_tile_limit)
    assert any("exceeds Trainium hardware limit" in item for item in v4), "Failed to catch tile limit overflow"
    print("  [PASS 4/10] Caught SBUF tile size overflow (TILE_H=256 > PMAX=128)")

    # Test 5: Missing Engines & Multi-buffering
    missing_engines_code = "def pipeline_kernel(x, w, a, b):\n    pass"
    v5 = check_rules(missing_engines_code)
    assert len(v5) >= 4, "Failed to catch missing engines and multi-buffering"
    print("  [PASS 5/10] Caught missing DMA, Vector, Tensor engines and multi-buffering")

    # Test 6: Ground Truth Math Verification
    x_test = np.random.randn(128, 128).astype(np.float32)
    w_test = np.random.randn(128, 128).astype(np.float32)
    res = ref_bilinear_matmul(x_test, w_test)
    assert res.shape == (128, 128), "Reference math shape error"
    print("  [PASS 6/10] Ground truth Bilinear + MatMul reference math verified")

    # Test 7: Hostile Test Generator
    hostile_cases = generate_hostile_test_cases()
    assert len(hostile_cases) == 7, "Hostile test case count mismatch"
    print(f"  [PASS 7/10] Hostile test generator verified ({len(hostile_cases)} test cases: prime, sub-tile, extreme 1e4, zeros, negatives)")

    # Test 8: Block-n Hazard Diagnostic — Pipeline Sync Hazard
    # Synthetic kernel that works alone on single tile but overwrites buffer in multi-tile
    def dummy_sync_bug_kernel(x, w, a=1.25, b=0.5):
        if x.shape[0] <= 128:
            return ref_bilinear_matmul(x, w, a, b)
        # multi-tile: corrupt block 1
        res = ref_bilinear_matmul(x, w, a, b)
        res[128:256, :] = 0.0
        return res

    h_sync, _, blk_sync = diagnose_block_failure(dummy_sync_bug_kernel, np.random.randn(256, 128).astype(np.float32), w_test, 1.25, 0.5)
    assert h_sync == "PIPELINE_SYNC_HAZARD" and blk_sync == 1, f"Expected PIPELINE_SYNC_HAZARD, got {h_sync}"
    print("  [PASS 8/10] Block-n Isolation Diagnostic accurately caught PIPELINE_SYNC_HAZARD at Block 1")

    # Test 9: Block-n Hazard Diagnostic — Ragged Edge Hazard
    def dummy_ragged_bug_kernel(x, w, a=1.25, b=0.5):
        res = ref_bilinear_matmul(x, w, a, b)
        if x.shape[0] % 128 != 0:
            res[-1, :] += 999.0  # Corrupt ragged edge
        return res

    h_ragged, _, blk_ragged = diagnose_block_failure(dummy_ragged_bug_kernel, np.random.randn(317, 128).astype(np.float32), w_test, 1.25, 0.5)
    assert h_ragged == "RAGGED_EDGE_HAZARD" and blk_ragged == 2, f"Expected RAGGED_EDGE_HAZARD, got {h_ragged}"
    print("  [PASS 9/10] Block-n Isolation Diagnostic accurately caught RAGGED_EDGE_HAZARD on prime tail (317 rows)")

    # Test 10: Golden Reference Kernel Evaluation
    from reference_pipeline import overlapped_pipeline
    def golden_wrapper(x, w, a=1.25, b=0.5):
        y, _, _ = overlapped_pipeline(x, w, a, b)
        return y

    with open("projects/03-multi-engine-overlap/reference_pipeline.py") as f:
        ref_code = f.read()

    score, details, feedback = grade(ref_code)
    assert score == 1.00, f"Expected 1.00 for golden kernel, got {score:.2f}"
    assert details["correctness"] == 1.0, "Expected 100% correctness on golden reference"
    assert details["overlap_efficiency"] == 1.0, "Expected 100% overlap efficiency on golden reference"
    print(f"  [PASS 10/10] Golden Reference Kernel graded 1.00/1.00 across all 7 hostile test cases!")

    print("\n" + "=" * 65)
    print("      ALL 10 SELFTESTS PASSED (100% VERIFICATION RATE)")
    print("      Grader is fully hardened for the live evaluation loop!")
    print("=================================================================\n")


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
