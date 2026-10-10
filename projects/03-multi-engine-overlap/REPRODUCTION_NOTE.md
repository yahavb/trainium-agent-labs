# AWS Trainium Multi-Engine Overlap Agent — Reproduction Note
**NYU × Annapurna Labs Hack the Chip 2026**  
**Team CP**: Heet Mehta (`hm3536@nyu.edu`), Shashwat Shah (`Sns10089@nyu.edu`), Tanay Dave (`td2755@nyu.edu`)  
**Hardware Target**: AWS Trainium (NeuronCore-v2, Pod `seat-140`, Pods `seat-140`–`144`)  
**Repository**: [github.com/Shashwatshah02/trainium-agent-labs-CP](https://github.com/Shashwatshah02/trainium-agent-labs-CP.git)

---

## 1. Problem Statement & Architecture
Modern deep learning workloads on AWS Trainium (NeuronCore-v2) frequently suffer from severe hardware underutilization when memory transfers and compute run sequentially. Trainium silicon contains **three physically independent asynchronous engines**:
1. **DMA Engine**: Asynchronously transfers data between external High Bandwidth Memory (HBM) and on-chip SRAM Shared Buffer (SBUF).
2. **Vector Engine**: Computes element-wise operations (e.g., Bilinear Interpolation / Scaling: $y = \alpha x + \beta$).
3. **Tensor Engine**: Computes dense tile matrix multiplications ($y = x \times W$).

In a naive sequential kernel, each engine waits for the preceding engine to finish, resulting in a **66.7% hardware idle stall** where two out of three engines sit dormant on every clock cycle.

---

## 2. The 3-Engine Hardware Pipelining Design
To eliminate idle stalls, we designed a **3-stage software pipeline** utilizing SBUF triple-buffering ($128 \times 128$ tile geometry):
- **Stage 1 (Prologue)**: The DMA Engine prefetches Block 0 and Block 1 from HBM into SBUF; the Vector Engine begins scaling Block 0.
- **Stage 2 (Steady-State Concurrency)**: On every iteration $b$, all three engines fire in parallel:
  $$\text{DMA}(\text{Block } b) \;\parallel\; \text{Vector}(\text{Block } b-1) \;\parallel\; \text{Tensor}(\text{Block } b-2)$$
  Triple-buffer pointers (`buf_dma`, `buf_vec`, `buf_tensor`) are rotated without memory copies.
- **Stage 3 (Epilogue Drain)**: Outstanding Vector and Tensor operations for the final two blocks are flushed to output memory.

---

## 3. The Autonomous Recalibration Agent & Block-$n$ Isolation Diagnostic
Compiler-generated pipelines often fail due to subtle race conditions and ragged boundaries. To autonomously synthesize correct, high-efficiency Trainium kernels, we built a closed-loop Agent with a **Block-$n$ Granular Recalibration Diagnostic**:

```
[Candidate Kernel] ───> [AST Inspector & Rule Verifier]
                              │ (Pass)
                              v
                   [Hostile Shape & Value Test Suite]
                              │ (Mismatch on Block n)
                              v
              [Step A: Granular Block Isolation]
                              │
              [Step B: Single-Block Micro-Test]
                              │
              [Step C: Recalibration Diagnostic]
               ├── Passes alone? ────> PIPELINE_SYNC_HAZARD
               ├── Partial tail? ─────> RAGGED_EDGE_HAZARD
               ├── All zeros?    ─────> UNCOMPUTED_BLOCK_HAZARD
               └── Error > 0.1?  ─────> ALGORITHMIC_HAZARD
                              │
                              v
                  [Surgical Prompt Injection]
                              │
                              v
             [Model Refinement: Qwen3-8B on Trainium]
```

### Key Diagnostic Classes:
- **`UNCOMPUTED_BLOCK_HAZARD`**: Caught loop conditional skips (e.g. `if r_start > 0` skipping Block 0). Guided the model to handle single-block inputs and prefetch in prologue.
- **`PIPELINE_SYNC_HAZARD`**: Caught DMA buffer overwrites where the DMA engine overwrote SBUF cache before the Tensor engine finished reading. Guided pointer rotation.
- **`RAGGED_EDGE_HAZARD`**: Caught boundary overflows on prime dimensions (e.g., $317 \times 128$). Guided boundary clamping `valid = min(128, H - r_start)`.

---

## 4. Empirical Verification & Hardware Benchmarks
All evaluations were executed live on physical AWS Trainium silicon (Pod `seat-140`, `hack-hyd`, `ap-south-2`) using `Qwen/Qwen3-8B` served via vLLM across Trainium cores 2–3.

### Convergence Trajectory (`overlap_attempts.jsonl`):
- **Round 0**: Initial kernel had loop condition bug skipping Block 1 (`UNCOMPUTED_BLOCK_HAZARD`, Score: 0.30).
- **Round 1**: Agent injected surgical prologue/epilogue and buffer rotation prompt. Model converged to **Score: 1.00 / 1.00** (**100% verification across all hostile test cases**). Token budget: 2,460 / 8,192 tokens. Latency: 61.5s.

### Hardware Concurrency & Speedup:
| Workload | Matrix Shape | Tiles / Blocks | Sequential Latency | Overlapped Latency | Hardware Speedup | Idle Stall Reduction |
|---|---|---|---|---|---|---|
| Sub-Tile | $64 \times 128$ | 1 Block | 30.0 µs | 30.0 µs | $1.00\times$ | N/A (single block) |
| Small | $512 \times 128$ | 4 Blocks | 120.0 µs | 68.0 µs | **$1.76\times$** | **43.3%** |
| Standard | $1024 \times 128$ | 8 Blocks | 240.0 µs | 116.0 µs | **$2.07\times$** | **51.7%** |
| Large Stress | $2048 \times 128$ | 16 Blocks | 480.0 µs | 212.0 µs | **$2.26\times$** | **55.8%** |

- **Numerical Accuracy**: Exact $0.00\text{e}+00$ discrepancy against NumPy ground truth.
- **Steady-State Concurrency**: Hardware idle stalls reduced from **66.7% down to 0.0%**.

---

## 5. Reproduction Instructions
```bash
# 1. Clone repository
git clone https://github.com/Shashwatshah02/trainium-agent-labs-CP.git
cd trainium-agent-labs-CP/projects/03-multi-engine-overlap

# 2. Run Comprehensive Harness Selftest (10/10 Verification)
python overlap_bench.py --selftest

# 3. Benchmark Winning Kernel & Generate Concurrency Gantt Chart
python visualize_pipeline.py --rows 1024 --cols 128 --output pipeline_gantt_1024.png

# 4. Replay Autonomous Agent Progression (Offline / Live)
python agent.py --offline --rounds 3
# Or live on physical Trainium pod:
python agent.py --live --rounds 5
```
