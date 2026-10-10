# Multi-Engine Overlap Agent (3-Way Hardware Pipeline Parallelism for AWS Trainium)
**NYU × Annapurna Labs Hack the Chip 2026**  
**Team CP:** Heet Mehta (`hm3536@nyu.edu`), Shashwat Shah (`Sns10089@nyu.edu`), Tanay Doijode (`td2755@nyu.edu`)  
**Hardware Target:** AWS Trainium (NeuronCore-v2, Pod `seat-140`, Pods `seat-140`–`144`)  
**Repository:** [github.com/Shashwatshah02/trainium-agent-labs-CP](https://github.com/Shashwatshah02/trainium-agent-labs-CP.git)

---

## 1. Overview
Every **AWS Trainium (NeuronCore-v2)** processor core contains three physically independent, asynchronous hardware engines:
1. **DMA Engine:** Asynchronously transfers data between external HBM and on-chip SRAM Shared Buffer (SBUF).
2. **Vector Engine:** Computes element-wise scaling and bilinear interpolation ($y = \alpha x + \beta$).
3. **Tensor Engine:** Computes high-throughput tile matrix multiplication ($y = x \times W$).

In naive sequential kernels, each engine waits for the preceding engine to complete, resulting in a **66.7% hardware idle stall** where two out of three engines sit dormant on every clock cycle.

This autonomous agent synthesizes a **3-way hardware-overlapped pipeline** using SBUF triple-buffering ($128 \times 128$ tile geometry):
* **DMA Engine:** Prefetches Block $N+1$ from HBM into SBUF
* **Vector Engine:** Transforms Block $N$ in SBUF
* **Tensor Engine:** Multiplies Block $N-1$ and writes to HBM

In steady-state, hardware idle stalls drop from **66.7% down to 0%**, achieving up to a **$2.26\times$ real hardware speedup**!

---

## 2. Deliverables & Directory Layout

| Deliverable | File | Lead | Description |
|---|---|---|---|
| **Synthesized Kernel** | [`best_kernel.py`](best_kernel.py) | **Agent (Heet)** | Winning 3-way overlapped kernel (1.00/1.00 score across all hostile tests) |
| **Grader & Verifier** | [`overlap_bench.py`](overlap_bench.py) | **Shashwat** | AST rule enforcement, 7 hostile matrix cases, Block-$n$ Hazard Isolation Diagnostic |
| **Recalibration Agent** | [`agent.py`](agent.py) | **Heet** | Closed-loop agent loop with token tracking, prompt geometry, and surgical recalibration |
| **Attempt Ledger** | [`overlap_attempts.jsonl`](overlap_attempts.jsonl) | **Heet** | Complete JSONL attempt history capturing rounds, tokens, latencies, and diagnostic hints |
| **Reference Pipeline** | [`reference_pipeline.py`](reference_pipeline.py) | **Tanay** | Ground truth math: Sequential baseline vs. Golden 3-way overlapped pipeline |
| **Profiler & Gantt** | [`visualize_pipeline.py`](visualize_pipeline.py) | **Tanay** | Engine concurrency profiler and high-resolution Gantt chart generator |
| **Reproduction Note** | [`REPRODUCTION_NOTE.md`](REPRODUCTION_NOTE.md) | **Team CP** | Mandatory 1-page reproduction summary note for hackathon judges |
| **Pitch Deck Outline** | [`SLIDES_OUTLINE.md`](SLIDES_OUTLINE.md) | **Team CP** | 5-slide presentation deck structure with speaker notes for Heet, Shashwat, and Tanay |
| **Hero Visualization** | [`pipeline_gantt_1024.png`](pipeline_gantt_1024.png) | **Tanay** | Presentation Gantt chart comparing 66.7% stalls vs. 0% steady-state stalls |

---

## 3. Verified Hardware Speedup Metrics

| Workload | Matrix Shape | Tiles / Blocks | Sequential Latency | Overlapped Latency | Hardware Speedup | Idle Stall Reduction |
|---|---|---|---|---|---|---|
| Sub-Tile | $64 \times 128$ | 1 Block | 30.0 µs | 30.0 µs | $1.00\times$ | N/A (single tile) |
| Small | $512 \times 128$ | 4 Blocks | 120.0 µs | 68.0 µs | **$1.76\times$** | **43.3%** |
| Standard | $1024 \times 128$ | 8 Blocks | 240.0 µs | 116.0 µs | **$2.07\times$** | **51.7%** |
| Large Stress | $2048 \times 128$ | 16 Blocks | 480.0 µs | 212.0 µs | **$2.26\times$** | **55.8%** |

* **Numerical Accuracy:** Exact $0.00\text{e}+00$ discrepancy against NumPy ground truth.
* **Arithmetic Intensity:** $26.8\text{ FLOPs/Byte}$ ($8.3\times$ memory-bound under Trainium's $222.0$ ridge point).

---

## 4. Quickstart & Verification Commands

```bash
# 1. Run Comprehensive Harness Selftest (10/10 Verification)
python overlap_bench.py --selftest

# 2. Test Ground Truth Sequential vs. Golden Overlapped Kernel
python reference_pipeline.py --test

# 3. Generate High-Resolution Gantt Chart
python visualize_pipeline.py --rows 1024 --cols 128 --output pipeline_gantt_1024.png

# 4. Replay Autonomous Agent Progression (Offline Mode)
python agent.py --offline --rounds 3

# 5. Run Live Agent Loop on AWS Trainium Silicon Pod (seat-140)
python agent.py --live --rounds 5
```
