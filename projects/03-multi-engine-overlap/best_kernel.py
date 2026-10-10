#!/usr/bin/env python3
"""
reference_pipeline.py — Golden Reference Kernels (Sequential vs. 3-Way Overlapped Pipeline)
Track 3 Lead: Tanay

This module provides:
1. `sequential_pipeline`: Naive baseline where DMA, Vector, and Tensor run back-to-back (67% idle stalls).
2. `overlapped_pipeline`: Golden 3-way pipelined kernel overlapping DMA, Vector, and Tensor engines.
3. Accurate cycle/time interval tracing for the Gantt chart visualizer.
"""

import argparse
import time
from dataclasses import dataclass
import numpy as np

# Hardware Engine Simulated Latencies for 128x128 Tile (in microseconds)
LATENCY_DMA_US = 10.0      # HBM -> SBUF transfer
LATENCY_VEC_US = 8.0       # Vector engine element-wise math
LATENCY_TENSOR_US = 12.0   # Tensor engine tile matmul
TILE_H = 128
TILE_W = 128


@dataclass
class EngineEvent:
    engine: str       # 'DMA', 'VECTOR', 'TENSOR'
    start_us: float
    end_us: float
    block_id: int
    note: str = ""


# ---------------------------------------------------------------- Simulated Hardware Primitives
def dma_copy(dst_sbuf: np.ndarray, src_hbm: np.ndarray, valid_rows: int):
    """Simulates nisa.dma_copy moving data from HBM into SBUF."""
    dst_sbuf[:valid_rows, :] = src_hbm[:valid_rows, :]


def vector_bilinear_scale(dst_sbuf: np.ndarray, src_sbuf: np.ndarray, alpha: float, beta: float, valid_rows: int):
    """Simulates Vector Engine (nisa.tensor_scalar) element-wise math."""
    dst_sbuf[:valid_rows, :] = alpha * src_sbuf[:valid_rows, :] + beta


def tensor_matmul(dst_sbuf: np.ndarray, act_sbuf: np.ndarray, wt_sbuf: np.ndarray, valid_rows: int):
    """Simulates Tensor Engine (nisa.nc_matmul) matrix multiplication."""
    # Slices avoid whole-array cheats and operate strictly within tile
    act_slice = act_sbuf[:valid_rows, :].astype(np.float64)
    wt_slice = wt_sbuf.astype(np.float64)
    dst_sbuf[:valid_rows, :] = (act_slice @ wt_slice).astype(np.float32)


# ---------------------------------------------------------------- Naive Baseline (Sequential)
def sequential_pipeline(x: np.ndarray, weight: np.ndarray, alpha: float = 1.25, beta: float = 0.5):
    """
    Naive Sequential Execution:
    For each block:
       DMA (10us) -> Vector (8us) -> Tensor (12us)
    Total block time: 30us. 67% idle stall across engines.
    """
    H, W = x.shape
    _, K = weight.shape
    num_blocks = (H + TILE_H - 1) // TILE_H
    y_out = np.zeros((H, K), dtype=np.float32)

    events: list[EngineEvent] = []
    curr_t = 0.0

    # Single shared buffer in SBUF (no multi-buffering)
    buf_raw = np.zeros((TILE_H, W), dtype=np.float32)
    buf_scaled = np.zeros((TILE_H, W), dtype=np.float32)
    buf_out = np.zeros((TILE_H, K), dtype=np.float32)

    for b in range(num_blocks):
        r_start = b * TILE_H
        r_end = min(r_start + TILE_H, H)
        valid_rows = r_end - r_start

        # 1. DMA Engine Transfer
        dma_start = curr_t
        dma_end = dma_start + LATENCY_DMA_US
        dma_copy(buf_raw, x[r_start:r_end, :], valid_rows)
        events.append(EngineEvent("DMA", dma_start, dma_end, b))
        curr_t = dma_end

        # 2. Vector Engine
        vec_start = curr_t
        vec_end = vec_start + LATENCY_VEC_US
        vector_bilinear_scale(buf_scaled, buf_raw, alpha, beta, valid_rows)
        events.append(EngineEvent("VECTOR", vec_start, vec_end, b))
        curr_t = vec_end

        # 3. Tensor Engine
        ten_start = curr_t
        ten_end = ten_start + LATENCY_TENSOR_US
        tensor_matmul(buf_out, buf_scaled, weight, valid_rows)
        events.append(EngineEvent("TENSOR", ten_start, ten_end, b))
        curr_t = ten_end

        # Write output back to HBM
        y_out[r_start:r_end, :] = buf_out[:valid_rows, :]

    return y_out, events, curr_t


# ---------------------------------------------------------------- Golden 3-Way Overlapped Pipeline
def overlapped_pipeline(x: np.ndarray, weight: np.ndarray, alpha: float = 1.25, beta: float = 0.5):
    """
    3-Way Pipelined Execution with Triple Buffering in SBUF:
    - buf_dma: Receiving block N+1 from HBM
    - buf_vec: Transforming block N with Vector Engine
    - buf_tensor: Computing matmul on block N-1 with Tensor Engine

    Steady-state iteration time = max(10us, 8us, 12us) = 12us.
    Speedup: 30us / 12us = 2.5x speedup!
    """
    H, W = x.shape
    _, K = weight.shape
    num_blocks = (H + TILE_H - 1) // TILE_H
    y_out = np.zeros((H, K), dtype=np.float32)

    events: list[EngineEvent] = []

    # Allocate distinct SBUF buffers for 3-stage pipeline
    # Triple buffers allow zero-copy pointer rotation
    buf_dma = np.zeros((TILE_H, W), dtype=np.float32)
    buf_vec = np.zeros((TILE_H, W), dtype=np.float32)
    buf_tensor = np.zeros((TILE_H, W), dtype=np.float32)
    buf_out = np.zeros((TILE_H, K), dtype=np.float32)

    def get_valid_rows(b_idx: int) -> int:
        r_start = b_idx * TILE_H
        return min(TILE_H, H - r_start)

    t_dma = 0.0
    t_vec = 0.0
    t_ten = 0.0

    # ------------------ PROLOGUE ------------------
    # Step 1: DMA loads Block 0
    if num_blocks > 0:
        v0 = get_valid_rows(0)
        dma_copy(buf_dma, x[0:v0, :], v0)
        events.append(EngineEvent("DMA", t_dma, t_dma + LATENCY_DMA_US, 0, "Prologue"))
        t_dma += LATENCY_DMA_US

    # Step 2: DMA loads Block 1 || Vector computes Block 0
    if num_blocks > 1:
        # Rotate: buf_vec gets Block 0
        buf_vec, buf_dma = buf_dma, buf_vec

        step2_start = t_dma
        # DMA Block 1
        v1 = get_valid_rows(1)
        dma_copy(buf_dma, x[TILE_H:TILE_H + v1, :], v1)
        events.append(EngineEvent("DMA", step2_start, step2_start + LATENCY_DMA_US, 1, "Prologue"))

        # Vector Block 0 (runs concurrently)
        v0 = get_valid_rows(0)
        vector_bilinear_scale(buf_vec, buf_vec, alpha, beta, v0)
        events.append(EngineEvent("VECTOR", step2_start, step2_start + LATENCY_VEC_US, 0, "Prologue"))

        step2_end = step2_start + max(LATENCY_DMA_US, LATENCY_VEC_US)
        t_dma = step2_end
        t_vec = step2_end

    # ------------------ MAIN STEADY-STATE LOOP ------------------
    # Concurrent: DMA(Block i) || Vector(Block i-1) || Tensor(Block i-2)
    for b in range(2, num_blocks):
        # Rotate triple buffers:
        # buf_tensor gets Block b-2 (was buf_vec)
        # buf_vec gets Block b-1 (was buf_dma)
        # buf_dma is freed to receive Block b
        buf_tensor, buf_vec, buf_dma = buf_vec, buf_dma, buf_tensor

        step_start = max(t_dma, t_vec, t_ten)

        # 1. DMA Engine: Fetch Block b
        vb = get_valid_rows(b)
        r_st = b * TILE_H
        dma_copy(buf_dma, x[r_st:r_st + vb, :], vb)
        events.append(EngineEvent("DMA", step_start, step_start + LATENCY_DMA_US, b, "Steady-state"))

        # 2. Vector Engine: Transform Block b-1
        vb_prev = get_valid_rows(b - 1)
        vector_bilinear_scale(buf_vec, buf_vec, alpha, beta, vb_prev)
        events.append(EngineEvent("VECTOR", step_start, step_start + LATENCY_VEC_US, b - 1, "Steady-state"))

        # 3. Tensor Engine: Multiply Block b-2
        vb_prev2 = get_valid_rows(b - 2)
        tensor_matmul(buf_out, buf_tensor, weight, vb_prev2)
        r_out_st = (b - 2) * TILE_H
        y_out[r_out_st:r_out_st + vb_prev2, :] = buf_out[:vb_prev2, :]
        events.append(EngineEvent("TENSOR", step_start, step_start + LATENCY_TENSOR_US, b - 2, "Steady-state"))

        step_dur = max(LATENCY_DMA_US, LATENCY_VEC_US, LATENCY_TENSOR_US)
        t_dma = step_start + step_dur
        t_vec = step_start + step_dur
        t_ten = step_start + step_dur

    # ------------------ EPILOGUE (DRAIN PIPELINE) ------------------
    # Epilogue 1: Finish Tensor on Block num_blocks-2 and Vector on num_blocks-1
    if num_blocks >= 2:
        buf_tensor, buf_vec = buf_vec, buf_tensor
        epi1_start = max(t_vec, t_ten)

        v_penult = get_valid_rows(num_blocks - 2)
        tensor_matmul(buf_out, buf_tensor, weight, v_penult)
        r_st_penult = (num_blocks - 2) * TILE_H
        y_out[r_st_penult:r_st_penult + v_penult, :] = buf_out[:v_penult, :]
        events.append(EngineEvent("TENSOR", epi1_start, epi1_start + LATENCY_TENSOR_US, num_blocks - 2, "Epilogue"))

        v_last = get_valid_rows(num_blocks - 1)
        vector_bilinear_scale(buf_dma, buf_dma, alpha, beta, v_last)
        events.append(EngineEvent("VECTOR", epi1_start, epi1_start + LATENCY_VEC_US, num_blocks - 1, "Epilogue"))

        epi1_end = epi1_start + max(LATENCY_TENSOR_US, LATENCY_VEC_US)
        t_ten = epi1_end

        # Epilogue 2: Finish Tensor on final Block num_blocks-1
        epi2_start = t_ten
        tensor_matmul(buf_out, buf_dma, weight, v_last)
        r_st_last = (num_blocks - 1) * TILE_H
        y_out[r_st_last:r_st_last + v_last, :] = buf_out[:v_last, :]
        events.append(EngineEvent("TENSOR", epi2_start, epi2_start + LATENCY_TENSOR_US, num_blocks - 1, "Epilogue"))
        t_ten = epi2_start + LATENCY_TENSOR_US
    elif num_blocks == 1:
        # Single block fallback
        v0 = get_valid_rows(0)
        vector_bilinear_scale(buf_dma, buf_dma, alpha, beta, v0)
        tensor_matmul(buf_out, buf_dma, weight, v0)
        y_out[0:v0, :] = buf_out[:v0, :]

    total_time = max(t_dma, t_vec, t_ten)
    return y_out, events, total_time


# ---------------------------------------------------------------- Wrapper Entrypoint for Grader
def pipeline_kernel(x: np.ndarray, weight: np.ndarray, alpha: float = 1.25, beta: float = 0.5) -> np.ndarray:
    """Wrapper function matching overlap_bench grading contract."""
    res, _, _ = overlapped_pipeline(x, weight, alpha, beta)
    return res


# ---------------------------------------------------------------- Verification Run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Compare Sequential vs Overlapped outputs")
    args = parser.parse_args()

    np.random.seed(42)
    # Hostile shape with 315 rows (Block 0: 128, Block 1: 128, Block 2: 59 rows)
    X = np.random.randn(315, 128).astype(np.float32)
    W = np.random.randn(128, 128).astype(np.float32)

    seq_out, seq_ev, seq_t = sequential_pipeline(X, W)
    ovl_out, ovl_ev, ovl_t = overlapped_pipeline(X, W)

    diff = np.max(np.abs(seq_out - ovl_out))
    speedup = seq_t / ovl_t

    print("=== Reference Pipeline Benchmark ===")
    print(f"Matrix Shape: {X.shape} @ {W.shape}")
    print(f"Numerical Difference: {diff:.6e} (Pass: {diff < 1e-4})")
    print(f"Naive Sequential Latency:  {seq_t:.1f} us")
    print(f"3-Way Overlapped Latency:  {ovl_t:.1f} us")
    print(f"Hardware Speedup Factor:   {speedup:.2f}x ({((1 - ovl_t/seq_t)*100):.1f}% reduction in stalls)\n")
