# Multi-Engine Overlap Agent (3-Way Pipeline Parallelism for AWS Trainium)
## Hackathon Master Plan & Team Guide

**Event:** Hack the Chip — NYU × Annapurna Labs (AWS Trainium)  
**Team Size:** 3 Members  
**Target Architecture:** AWS Trainium (NeuronCore-v2)  

---

## 1. Executive Summary & Pitch

### The Core Problem
AWS Trainium chips have immense raw compute speed, but **memory bandwidth (moving data between main HBM RAM and fast SBUF cache) is the #1 bottleneck in real-world AI workloads**. Most naive code spends 10% of its time doing math and **90% of its time sitting idle waiting for memory transfers**.

### The Solution
We are building an **Automated Agent Loop** that takes naive LLM-generated code and automatically refactors it into a **3-Way Parallel Hardware Pipeline**. Our agent overlaps **Memory Transfers (DMA Engine)**, **Vector Scaling / Bilinear Interpolation (Vector Engine)**, and **Matrix Multiplication (Tensor Engine)** so that all 3 hardware engines run simultaneously with **zero memory stalls**.

---

## 2. Technical Architecture: 3-Engine Parallelism

Every AWS Trainium core contains **3 independent physical hardware engines** capable of running at the exact same millisecond:

1. **DMA Engine (Memory Movement):** Moves data between main RAM (HBM) and fast cache (SBUF) via `nisa.dma_copy`.
2. **Vector Engine (Element-wise Math):** Performs Bilinear Interpolation, scaling, and additions via `nisa.tensor_scalar`, `nl.add`, `nl.multiply`.
3. **Tensor Engine (Matrix Math):** Performs heavy matrix multiplication via `nisa.nc_matmul`.

```
                    NAIVE EXECUTION (66% Idle Time)
┌──────────────────┬──────────────────────┬──────────────────────┐
│  DMA Engine (N)  │ Vector Engine (N)    │ Tensor Engine (N)    │
│  (Vector/Tensor  │ (DMA/Tensor engines  │ (DMA/Vector engines  │
│   engines idle)  │  idle)               │  idle)               │
└──────────────────┴──────────────────────┴──────────────────────┘

            3-WAY OVERLAPPED ENGINE PIPELINE (0% Idle Time)
┌────────────────────────────────────────────────────────────────┐
│  DMA Engine:    Loads Block N+1 into SBUF Cache                │
│  Vector Engine: Performs Bilinear Interpolation on Block N     │
│  Tensor Engine: Performs Matrix Multiplication on Block N-1    │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Recalibration Mechanism (Failure Handling at Block n)

If execution fails at the $n$-th block during pipeline execution, our system triggers a 3-step self-healing recalibration loop:

```
  Pipeline Execution
  ├── Block 0: PASSED ✅
  ├── Block 1: PASSED ✅
  └── Block n: FAILED ❌ (NaN / Mismatch / Buffer Overwrite / Ragged Edge)
           │
           ▼
  STEP A: Granular Block Isolation
  Isolates Block n input slice & identifies hazard type
           │
           ▼
  STEP B: Micro-Test Diagnostic
  Tests Block n in isolation (Single-Block vs Pipeline Sync check)
           │
           ▼
  STEP C: Recalibration Patch Prompt
  Generates targeted fix instruction for the LLM
```

### The 3 Patch Conditions:
* **Pipeline Sync Hazard at Block $n$:** Swap buffer pointers (`active_buf, next_buf = next_buf, active_buf`) to prevent DMA overwriting data before Tensor Engine finishes.
* **Ragged Edge Boundary Hazard at Block $n$:** Apply boundary clamping (`min(128, remaining_rows)`) for the final partial block.
* **Numerical Instability Hazard at Block $n$:** Upgrade intermediate accumulator precision to `float32` inside SBUF before casting back to `bfloat16`.

---

## 4. Team Work Distribution (3 Parallel Tracks)

### **Track 1: Member 1 — Hardware & Profiler Lead (Grader)**
* **Focus:** Building the 3-Engine Grader and Memory Stall Harness.
* **Key Tasks:**
  1. Build the Grader rules to verify Bilinear Interpolation + MatMul accuracy against NumPy.
  2. Measure 3-engine concurrency (checking if DMA, Vector, and Tensor operations run in parallel).
  3. Format granular recalibration hints when block $n$ fails.

### **Track 2: Member 2 — Agent & Prompt Strategy Lead (AI Brain)**
* **Focus:** Prompts, LLM Strategy, and Cache Geometry Helper.
* **Key Tasks:**
  1. Design LLM prompts that teach `Qwen3-8B` / `GPT-OSS` the 3-Engine Overlap Design Pattern.
  2. Build a 3-Buffer Cache Allocation Helper (`buf_dma`, `buf_vector`, `buf_tensor`) to prevent SBUF cache overflow.
  3. Manage multi-sample candidate ranking (selecting the kernel with highest engine concurrency).

### **Track 3: Member 3 — Benchmarking & Storyteller Lead (Pitch Deck)**
* **Focus:** Speedup evaluation, visualizations, and hackathon presentation.
* **Key Tasks:**
  1. Measure execution speedup (e.g. 3x latency reduction, 0% memory idle time).
  2. Generate a visual **3-Engine Hardware Parallelism Gantt Chart** for slides.
  3. Prepare the 1-Page Report and Presentation Deck for Annapurna Labs judges.

---

## 5. 10-Hour Hackathon Roadmap

| Hours | Target Goal | Deliverable |
| :---: | :--- | :--- |
| **1–3** | **Architecture & Test Harness** | Member 1 builds 3-engine grader; Member 2 creates prompt templates; Member 3 sets up test scripts. |
| **4–6** | **Loop Tuning & Experiments** | Run agent loop on pod (`seat-140`), refine recalibration prompts until agent outputs 100% overlapped code. |
| **7–8** | **Data Collection & Benchmarks** | Member 3 collects latency reduction metrics and generates engine parallelism charts. |
| **9–10** | **Pitch Deck & Final Polish** | Finalize slides, practice presentation, and prepare report for judges. |
