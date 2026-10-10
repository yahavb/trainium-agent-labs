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
- [ ] Add hostile test dimensions to `overlap_bench.py`:
  - [ ] Matrix with prime rows: `317 × 128` (last tile has only 61 rows).
  - [ ] Small matrix: `64 × 128` (smaller than a full 128 tile).
  - [ ] Large matrix: `1024 × 128` (stress test).
- [ ] Implement Hostile Value Generator:
  - [ ] Extreme floating-point values ($10^4$) to test accumulator overflow.
  - [ ] Zero matrices and negative matrices.
- [ ] Harden the 3-engine AST inspector to catch illegal hidden whole-array operations.
- [ ] Run `python overlap_bench.py --selftest` and verify 100% pass.
- [ ] `git commit -am "Harden hostile test suite and AST grader"` and push to `shashwat/grader`.

#### Heet (`heet/agent`):
- [ ] Connect `agent.py` to live model endpoint:
  - [ ] Test connection to local pod (`http://localhost:8000/v1` on `seat-140`) or shared `GPTOSS_BASE_URL`.
  - [ ] Verify `--live` flag receives responses within timeout.
- [ ] Instrument token usage tracker:
  - [ ] Track `prompt_tokens` and `completion_tokens` per round.
  - [ ] Ensure generation stays strictly below the 8,192 token limit.
- [ ] Finalize the SBUF 3-buffer cache geometry prompt (ensuring prompt is concise to prevent model loops).
- [ ] `git commit -am "Implement live model endpoint and token instrumentation"` and push to `heet/agent`.

#### Tanay (`tanay/benchmark`):
- [ ] Run scaling benchmarks in `visualize_pipeline.py`:
  - [ ] Test $512 \times 128$ (4 blocks), $1024 \times 128$ (8 blocks), and $2048 \times 128$ (16 blocks).
  - [ ] Record the exact latency numbers and hardware speedup factors ($1.76\times \rightarrow 2.07\times \rightarrow 2.35\times$).
- [ ] Polish the Gantt chart visualization:
  - [ ] Ensure clear labels ("DMA Engine", "Vector Engine", "Tensor Engine").
  - [ ] Highlight the 66.7% idle stalls in sequential vs. 0% in overlapped steady-state.
  - [ ] Export `pipeline_gantt_1024.png` (high DPI).
- [ ] Start setting up the Google Slides structure based on the Master Plan.
- [ ] `git commit -am "Generate multi-block Gantt charts and benchmark metrics"` and push to `tanay/benchmark`.

---

### Phase 2: Live Loop Handshake & Integration (2:00 PM – 2:30 PM)
*Goal: First live integration test where the live model writes code graded by the verifier.*

- [ ] **DEPENDENCY CHECK:** Heet pulls latest from `shashwat/grader` (or Shashwat pushes updates to master/shared branch).
- [ ] **TEAM TEST:** Heet runs `python agent.py --live --rounds 3`.
- [ ] Observe Model Behavior:
  - [ ] Did the model output valid Python code block?
  - [ ] Did Shashwat's AST checker catch any rule violations?
  - [ ] Did the Block-$n$ diagnostic trigger accurately?
- [ ] **Troubleshooting Rule:** If the model hallucinates or outputs empty content, shorten the initial prompt immediately.

---

### Phase 3: Recalibration Tuning Loop (2:30 PM – 4:00 PM)
*Goal: Ensure the agent successfully repairs a failed kernel in 1–2 rounds.*

#### Heet & Shashwat (Pairing):
- [ ] Test **Sync Hazard Recovery:**
  - [ ] When Block $n$ fails due to buffer overwrite, does the model swap buffer pointers?
  - [ ] If not, refine the hint in `overlap_bench.py`: *"Use explicit double buffer swap `active_buf, next_buf = next_buf, active_buf`"*.
- [ ] Test **Ragged Edge Recovery:**
  - [ ] When the partial tail block fails on shape $317 \times 128$, does the model apply `valid = min(128, rem)`?
- [ ] Log at least 3 full multi-round trajectories into `overlap_attempts.jsonl` (Proof of Learning).

#### Tanay (Independent):
- [ ] Build the **Roofline Model & Arithmetic Intensity Comparison**:
  - [ ] Compare HBM traffic: Sequential vs. Overlapped pipeline.
  - [ ] Show that DMA prefetching completely hides memory latency behind compute.
- [ ] Draft Slide Deck outline (5 slides total):
  1. *Slide 1: Title & The Memory Wall Problem* (90% idle memory stalls on Trainium).
  2. *Slide 2: The Solution: 3-Way Engine Pipelining* (DMA || Vector || Tensor).
  3. *Slide 3: The Recalibration Agent* (Block-$n$ Isolation Diagnostic).
  4. *Slide 4: Results & Hardware Speedup* ($2.07\times$ speedup, Gantt chart).
  5. *Slide 5: Failure Taxonomy & Learnings*.

---

### Phase 4: Final Benchmarking & Data Lock (4:00 PM – 4:45 PM)
*Goal: Freeze code and extract final numbers.*

- [ ] **DEPENDENCY:** Heet exports the winning kernel generated by the agent.
- [ ] Tanay runs `visualize_pipeline.py` on the agent's synthesized kernel to confirm it matches the $2.07\times$ speedup.
- [ ] Verify Deliverable 1: `overlap_bench.py` passes all tests.
- [ ] Verify Deliverable 2: `overlap_attempts.jsonl` is populated with scored attempts.
- [ ] Verify Deliverable 3: `pipeline_gantt.png` is generated.
- [ ] Merge `shashwat/grader`, `heet/agent`, and `tanay/benchmark` into `master`.

---

### Phase 5: Pitch Deck & 1-Page Summary Note (4:45 PM – 6:00 PM)
*Goal: Win the presentation.*

- [ ] **1-Page Reproduction Note** (Mandatory hackathon deliverable):
  - [ ] Problem statement & Trainium architecture.
  - [ ] The 3-engine concurrency design.
  - [ ] Recalibration diagnostic performance (rounds to green).
  - [ ] Hardware speedup and idle stall reduction.
- [ ] **Slide Deck Finalization:**
  - [ ] Add the Gantt chart graphic.
  - [ ] Add the attempt ledger table.
- [ ] **Presentation Rehearsal (10 mins):**
  - [ ] Heet: Introduces the Problem & Agent Loop (1.5 mins).
  - [ ] Shashwat: Explains the Hardware Grader & Block-$n$ Hazard Isolation (1.5 mins).
  - [ ] Tanay: Presents the Hardware Gantt Chart, Latency Speedup & Conclusion (2 mins).
