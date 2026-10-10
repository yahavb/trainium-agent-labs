# Multi-Engine Overlap Agent (3-Way Pipeline Parallelism for AWS Trainium)

## Hackathon Track Overview
Every **AWS Trainium (NeuronCore-v2)** contains three independent hardware engines:
1. **DMA Engine:** Moves data between HBM and SBUF cache (`nisa.dma_copy`).
2. **Vector Engine:** Performs element-wise operations and Bilinear Interpolation (`nisa.tensor_scalar`, `nl.add`, `nl.multiply`).
3. **Tensor Engine:** Performs high-throughput matrix multiplication (`nisa.nc_matmul`).

Naive execution runs these sequentially, leaving engines **idle 66% of the time**.
This agent synthesizes a **3-way overlapped pipeline** where:
* Block $N+1$ is fetched by DMA
* Block $N$ is transformed by the Vector Engine
* Block $N-1$ is multiplied by the Tensor Engine

---

## Directory Layout & Team Responsibilities

| File | Lead | Purpose |
|---|---|---|
| `overlap_bench.py` | **Shashwat** | 3-Engine Grader, SBUF Validator, Block-$n$ Hazard Isolation Diagnostic |
| `agent.py` | **Heet** | LLM Agent Loop, SBUF Geometry Prompt, Recalibration Patcher |
| `reference_pipeline.py` | **Tanay** | Ground truth reference: Naive Sequential vs. Golden Overlapped Pipeline |
| `visualize_pipeline.py` | **Tanay** | Execution timeline tracer, Speedup calculator & Gantt Chart generator |

---

## Quickstart Commands

```bash
# 1. Verify the Grader & Planted Bug Detectors
python overlap_bench.py --selftest

# 2. Test Ground Truth Sequential vs. Golden Overlapped Kernel
python reference_pipeline.py --test

# 3. Generate the 3-Engine Gantt Chart for the Pitch Deck
python visualize_pipeline.py --output gantt_chart.png

# 4. Run the Agent Loop (offline replay mode)
python agent.py --offline

# 5. Run the Agent Loop with live model (Qwen3-8B / GPT-OSS)
python agent.py --rounds 6 --samples 2
```
