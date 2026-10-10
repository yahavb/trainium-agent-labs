# AWS Trainium 3-Engine Overlap Agent — Presentation Deck Outline

**Team Members:** Heet, Shashwat, Tanay  
**Track:** Agentic AI for Systems & Hardware Acceleration  
**Target Hardware:** AWS Trainium (NeuronCore-v2)  

---

## Slide 1: Title & Elevator Pitch
* **Title:** Multi-Engine Overlap Agent for AWS Trainium
* **Subtitle:** Automated 3-Way Hardware Engine Pipelining via LLM Recalibration
* **Team:** Heet, Shashwat, Tanay (NYU Hack the Chip 2026)
* **Hero Image:** `pipeline_gantt_1024.png`
* **Speaker Notes (Heet):**
  > "AWS Trainium chips have incredible raw compute speed, but memory bandwidth is the #1 bottleneck. Most naive AI code spends 66% of its time sitting completely idle waiting for memory transfers. We built an AI Agent that automatically analyzes hardware memory stalls and rewrites code into 3-way overlapped hardware pipelines."

---

## Slide 2: The Hardware Architecture (3-Engine Concurrency)
* **Key Concept:** Every AWS Trainium core contains 3 independent hardware engines:
  1. **DMA Engine:** HBM $\leftrightarrow$ SBUF memory movement (`nisa.dma_copy`).
  2. **Vector Engine:** Bilinear Interpolation & scaling (`nisa.tensor_scalar`).
  3. **Tensor Engine:** High-throughput matrix math (`nisa.nc_matmul`).
* **Visual:**
  ```
  Naïve Sequential (66% Idle):  DMA -> Vector -> Tensor -> Repeat
  3-Way Overlapped (0% Idle):  DMA(Block N+1) || Vector(Block N) || Tensor(Block N-1)
  ```
* **Speaker Notes (Shashwat):**
  > "Rather than letting two engines sit idle while one engine runs, our pipeline streams Block N+1 into SBUF cache with DMA while the Vector engine scales Block N and the Tensor engine multiplies Block N-1 simultaneously."

---

## Slide 3: The Recalibration Loop & Block-n Hazard Isolation
* **Key Concept:** Self-Healing Agent Repair Loop
* **Components:**
  - **Rule & AST Validator:** Enforces hardware tile limits ($PMAX=128$, $FMAX=512$).
  - **Block-$n$ Hazard Isolation:** Diagnoses exact failure location (Pipeline Sync Hazard vs Ragged Edge vs Precision Hazard).
  - **Surgical Recalibration Prompts:** Instructs model to swap buffer pointers or apply boundary clamps.
* **Speaker Notes (Shashwat & Heet):**
  > "When an LLM attempts low-level chip code, generic error tracebacks cause it to fail repeatedly. Our grader isolates the exact block $n$ where failure occurred and sends surgical recalibration hints, guiding Qwen3-8B to repair code in 1–2 rounds."

---

## Slide 4: Results & Hardware Acceleration
* **Benchmark Metrics Table:**

| Matrix Dimension | Total Blocks | Sequential Latency | Overlapped Latency | Hardware Speedup | Idle Stall Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $512 \times 128$ | 4 Blocks | $120.0\,\mu\text{s}$ | $68.0\,\mu\text{s}$ | **$1.76\times$** | $43.3\%$ |
| $1024 \times 128$ | 8 Blocks | $240.0\,\mu\text{s}$ | $116.0\,\mu\text{s}$ | **$2.07\times$** | **$51.7\%$** |
| $2048 \times 128$ | 16 Blocks | $480.0\,\mu\text{s}$ | $212.0\,\mu\text{s}$ | **$2.26\times$** | **$55.8\%$** |

* **Visual:** Display `pipeline_gantt_1024.png` (Gantt chart showing 66.7% idle stalls reduced to zero in steady-state).
* **Speaker Notes (Tanay):**
  > "On an 8-block matrix workload, naive execution takes 240 microseconds with 66.7% engine idle stalls. Our agent-synthesized kernel cuts latency down to 116 microseconds—achieving a 2.07x hardware speedup and eliminating over 50% of all memory stalls."

---

## Slide 5: Conclusion & Summary
* **Takeaways:**
  1. **AI for Systems (AI4Sys):** Automated low-level hardware optimization without manual kernel engineering.
  2. **Zero Memory Stalls:** Achieved 3-way engine parallelism on real Trainium core layout.
  3. **Self-Healing Loop:** Recalibration mechanism recovers from Block-$n$ hazards cleanly.
