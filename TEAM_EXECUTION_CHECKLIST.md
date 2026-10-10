# Multi-Engine Overlap Agent: Concrete Team Execution Checklist & Dependency Guide

**Project:** 3-Way Hardware Pipelined Kernel Agent for AWS Trainium (NeuronCore-v2)  
**Team Members:** Heet (`heet/agent`), Shashwat (`shashwat/grader`), Tanay (`tanay/benchmark`)  
**Current Time:** 12:55 PM | **Target Pitch Time:** ~6:00 PM  

---

## 1. Synchronization Matrix & Dependencies

This table defines **who can work in parallel** and **where you must wait or help each other**.

| Phase | Shashwat (`shashwat/grader`) | Heet (`heet/agent`) | Tanay (`tanay/benchmark`) | Dependency / Handshake |
|---|---|---|---|---|
| **Phase 1: Setup & Scaffolding** *(1:00 PM – 2:00 PM)* | Build hostile test cases (prime shapes, odd edges) in `overlap_bench.py` | Configure model connection (`seat-140` / `GPTOSS_BASE_URL`) in `agent.py` | Benchmark 8-block & 16-block scaling in `visualize_pipeline.py` | **100% Parallel.** Zero blocking. Everyone works in their own branch. |
| **Phase 2: First Live Loop Handshake** *(2:00 PM – 2:30 PM)* | Verify `grade()` returns consistent diagnostic schema | Feed candidate outputs from live model into `overlap_bench.py` | Build slide wireframe (Title, Problem, 3-Engine Architecture) | **Sync Point 1:** Heet & Shashwat align on `hazard_type` string definitions. |
| **Phase 3: The Recalibration Tuning** *(2:30 PM – 4:00 PM)* | Fine-tune Block-$n$ isolation hints based on model failure modes | Tune prompt compression and SBUF geometry template | Generate high-res Gantt charts and calculate % idle stalls | **Collaboration Point:** Shashwat & Heet iterate together on prompt repairs. Tanay works independently. |
| **Phase 4: Performance Verification** *(4:00 PM – 4:45 PM)* | Run final grading suite against agent's best kernel | Export `overlap_attempts.jsonl` (Attempt deliverable) | Benchmark agent's generated kernel vs Naive baseline | **Sync Point 2:** Tanay waits on Heet to get the highest-scoring kernel for final profiling. |
| **Phase 5: Pitch Deck & 1-Page Report** *(4:45 PM – 6:00 PM)* | Write Failure Taxonomy & Verifier section | Write Agent Loop & Recalibration Strategy section | Insert Gantt charts & Speedup figures into deck | **All 3 Collaborative:** Merge into `master`, rehearse presentation. |

---

## 2. Phase-by-Phase Concrete Checklists

### Phase 1: Deep Track Development (1:00 PM – 2:00 PM)
*Goal: Harden each component independently without waiting on anyone.*

#### Shashwat (`shashwat/grader`):
- [x] Add hostile test dimensions to `overlap_bench.py`:
  - [x] Matrix with prime rows: `317 × 128` (last tile has only 61 rows).
  - [x] Small matrix: `64 × 128` (smaller than a full 128 tile).
  - [x] Large matrix: `1024 × 128` (stress test).
- [x] Implement Hostile Value Generator:
  - [x] Extreme floating-point values ($10^4$) to test accumulator overflow.
  - [x] Zero matrices and negative matrices.
- [x] Harden the 3-engine AST inspector to catch illegal hidden whole-array operations.
- [x] Run `python overlap_bench.py --selftest` and verify 100% pass.
- [x] `git commit -am "Harden hostile test suite and AST grader"` and push to `shashwat/grader`.

#### Heet (`heet/agent`):
- [x] Connect `agent.py` to live model endpoint:
  - [x] Test connection to local pod (`http://localhost:8000/v1` on `seat-140`) or shared `GPTOSS_BASE_URL`.
  - [x] Verify `--live` flag receives responses within timeout.
- [x] Instrument token usage tracker:
  - [x] Track `prompt_tokens` and `completion_tokens` per round.
  - [x] Ensure generation stays strictly below the 8,192 token limit.
- [x] Finalize the SBUF 3-buffer cache geometry prompt (ensuring prompt is concise to prevent model loops).
- [x] `git commit -am "Implement live model endpoint and token instrumentation"` and push to `heet/agent`.

#### Tanay (`tanay/benchmark`):
- [x] Run scaling benchmarks in `visualize_pipeline.py`:
  - [x] Test $512 \times 128$ (4 blocks), $1024 \times 128$ (8 blocks), and $2048 \times 128$ (16 blocks).
  - [x] Record the exact latency numbers and hardware speedup factors ($1.76\times \rightarrow 2.07\times \rightarrow 2.35\times$).
- [x] Polish the Gantt chart visualization:
  - [x] Ensure clear labels ("DMA Engine", "Vector Engine", "Tensor Engine").
  - [x] Highlight the 66.7% idle stalls in sequential vs. 0% in overlapped steady-state.
  - [x] Export `pipeline_gantt_1024.png` (high DPI).
- [x] Start setting up the Google Slides structure based on the Master Plan.
- [x] `git commit -am "Generate multi-block Gantt charts and benchmark metrics"` and push to `tanay/benchmark`.

---

### Phase 2: Live Loop Handshake & Integration (2:00 PM – 2:30 PM)
*Goal: First live integration test where the live model writes code graded by the verifier.*

- [x] **DEPENDENCY CHECK:** Heet pulls latest from `shashwat/grader` (or Shashwat pushes updates to master/shared branch).
- [x] **TEAM TEST:** Heet runs `python agent.py --live --rounds 3`.
- [x] Observe Model Behavior:
  - [x] Did the model output valid Python code block?
  - [x] Did Shashwat's AST checker catch any rule violations?
  - [x] Did the Block-$n$ diagnostic trigger accurately?
- [x] **Troubleshooting Rule:** If the model hallucinates or outputs empty content, shorten the initial prompt immediately.

---

### Phase 3: Recalibration Tuning Loop (2:30 PM – 4:00 PM)
*Goal: Ensure the agent successfully repairs a failed kernel in 1–2 rounds.*

#### Heet & Shashwat (Pairing):
- [x] Test **Sync Hazard Recovery:**
  - [x] When Block $n$ fails due to buffer overwrite, does the model swap buffer pointers?
  - [x] Refined the hint in `overlap_bench.py` and structural pattern in `agent.py`.
- [x] Test **Ragged Edge Recovery:**
  - [x] Verified partial tail block recovery on shape $317 \times 128$ with `valid = min(128, rem)`.
- [x] Log multi-round trajectories into `overlap_attempts.jsonl` (Proof of Learning across offline and live hardware runs).

#### Tanay (Independent):
- [x] Build the **Roofline Model & Arithmetic Intensity Comparison**:
  - [x] Compare HBM traffic: Sequential vs. Overlapped pipeline.
  - [x] Show that DMA prefetching completely hides memory latency behind compute.
- [x] Draft Slide Deck outline (`SLIDES_OUTLINE.md` complete with speaker notes for all 3 members):
  1. *Slide 1: Title & The Memory Wall Problem* (66.7% idle memory stalls on Trainium).
  2. *Slide 2: The Solution: 3-Way Engine Pipelining* (DMA || Vector || Tensor).
  3. *Slide 3: The Recalibration Agent* (Block-$n$ Isolation Diagnostic).
  4. *Slide 4: Results & Hardware Speedup* ($2.07\times$ speedup, Gantt chart).
  5. *Slide 5: Failure Taxonomy & Learnings*.

---

### Phase 4: Final Benchmarking & Data Lock (4:00 PM – 4:45 PM)
*Goal: Freeze code and extract final numbers.*

- [x] **DEPENDENCY:** Heet exports the winning kernel generated by the agent (`best_kernel.py`).
- [x] Tanay runs `visualize_pipeline.py` on the agent's synthesized kernel to confirm it matches the $2.07\times$ speedup.
- [x] Verify Deliverable 1: `overlap_bench.py` passes all 10/10 selftests.
- [x] Verify Deliverable 2: `overlap_attempts.jsonl` is populated with scored attempts, tokens, and latencies.
- [x] Verify Deliverable 3: `pipeline_gantt_1024.png` is generated.
- [x] Merge `shashwat/grader`, `heet/agent`, and `tanay/benchmark` into `master`.

---

### Phase 5: Pitch Deck & 1-Page Summary Note (4:45 PM – 6:00 PM)
*Goal: Win the presentation.*

- [x] **1-Page Reproduction Note** (`REPRODUCTION_NOTE.md` authored and verified):
  - [x] Problem statement & Trainium architecture.
  - [x] The 3-engine concurrency design.
  - [x] Recalibration diagnostic performance (rounds to green).
  - [x] Hardware speedup and idle stall reduction.
- [x] **Slide Deck Finalization:**
  - [x] Add the Gantt chart graphic reference (`pipeline_gantt_1024.png`).
  - [x] Add the attempt ledger table.
- [x] **Presentation Rehearsal (10 mins):**
  - [x] Heet: Introduces the Problem & Agent Loop (1.5 mins).
  - [x] Shashwat: Explains the Hardware Grader & Block-$n$ Hazard Isolation (1.5 mins).
  - [x] Tanay: Presents the Hardware Gantt Chart, Latency Speedup & Conclusion (2 mins).
