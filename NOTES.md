# Hackathon notes (Trainium agent labs)

**Seats**: one pod per person; replace `<N>` in the commands with **your own seat number** (team lead teoguo = seat-116). Do not go into someone else's seat.

> Sources: `README.md`, `STATE.md`, `projects/02-kernel-agent/README.md`, `projects/02-kernel-agent/CHALLENGE-kernel-agent.md`.
> The core in one sentence: **what makes an agent loop good or bad is the checker, not the model.** "Wrong, off by 341%" is true but useless; "the sin(pi·x) term should not be there at all" is an instruction the model can act on.

---

## 0. Current status (2026-10-10 17:22, updated as we go)

**People**: teoguo (seat-116) + liuyq (experienced) are the two main contributors; three other newcomers help out and are not on the critical path.
**Problem**: we do project 2 (`projects/02-kernel-agent`, NKI kernel agent, runs on the chip). The CHALLENGE (Stage A, `kernelbench.py`) only if time allows, moving the same agent over. Scoring follows the 30/25/25/20 rubric (CHALLENGE line 239 says "Same rubric as every problem").

**Repos and remotes** (local checkout: `trainium-agent-labs/`, working branch `team`)
| remote | repo | use |
|---|---|---|
| `team` | github.com/liuyq123/trainium-agent-labs | **shared repo**; the local `team` branch tracks `team/master`, use `git pull --rebase` / `git push` directly |
| `origin` | github.com/teoguo/trainium-agent-labs | teoguo's fork; PRs to the original repo will go from here later (branch off `master`) |
| `upstream` | github.com/yahavb/trainium-agent-labs | the original repo, read-only |

**Baseline** (seat-116, `--all --rounds 8 --samples 4 --context 8192 --repeat 5`, 10:50–12:43, finished, 424 attempts)
```
          solved   scores of the 5 runs             STATE.md reference
level 1   0/5      [0.30, 0.30, 0.30, 0.30, 0.30]   0/5, all 0.30
level 2   3/5      [1.00, 0.30, 1.00, 0.30, 1.00]   4/5
level 3   0/5      [0.30, 0.30, 0.30, 0.30, 0.30]   0/5, all 0.30
level 4   0/5      [0.62, 0.62, 0.50, 0.62, 0.62]   0/5, all 0.62
```
**Deadline 18:30** (confirmed with the organizers). Seats 115/117/118/119 belong to teammates and can run in parallel; the runbook is the Claude doc "Trainium team member runbook".
Pull logs back to this machine: `scripts/sync.sh 116 pull` → `runs/seat-116/latest/` (gitignored, not committed).
Regenerate the category tables: `.venv/bin/python scripts/attempts_to_csv.py runs/seat-116/latest/projects/02-kernel-agent/attempts.jsonl -o analysis/baseline_seat116`

**Experiment results (maintained by executor 1; all `--rounds 8 --samples 4 --context 8192 --repeat 5`, vLLM TP2/8192/seqs 4, simulation target trn2/gen3)**

> ⚠ **The server decodes greedily** (measured 14:30): the same request at temperature 0.7 gave 4/4 character-identical outputs, the same with `n=4`, and passing `seed` returns HTTP 500 and crashes the engine (117 was restarted once because of this). Outputs change when the batch composition differs, so it is not fully deterministic. Conclusion: `--samples 4` is often just 1 sample, and `--repeat 5` is often the same trajectory replayed 5 times, so the table below has an extra "distinct trajectories" column (the number of runs whose whole-trajectory code sequence differs). Baseline 116: in 89 of 106 rounds the 4 samples were character-identical.

| experiment | commit | level | seat | solved | scores of the 5 runs (round solved) | distinct trajectories | reaudit | decision |
|---|---|---|---|---|---|---|---|---|
| baseline | before 8f1ca41 | L1/L2/L3/L4 | 116 | 0/5, 3/5, 0/5, 0/5 | see the table above | L1 round 0 identical in all 5; L2–L4 round 0 all different | L2 2 kernels PASS | reference |
| baseline replica | upstream 8f1ca41 | L1/L2/L3/L4 | 119 | 0/5, 2/5, 0/5, 0/5 | L2 [1,.3,.3,1,.3], L4 [.62×4,.5] | — | — | consistent with 116 |
| E-A | 5ed7ec2 (26c43ed in the process) | L1 | 116 | 0/5 | all 0.30 | 1–2 (log has no run field) | — | adopted as groundwork |
| E-v3 | c39c0ce | L4 | 117 | 1/2 (stopped at the 3rd run) | 1.0 (round 6), 0.75 | 2 | PASS | superseded by v7 |
| E-F | ac258d2 | L1 | 116 | 0/5 | all 0.30 | **1** | — | **not adopted, reverted (d97b5e4)** |
| **E-v7** | ce0403c | L3 | 117 | **5/5** | all 1.0 (all in round 1) | 4 | PASS (13 solves, 2 distinct kernels) | **adopted (mainline)** |
| **E-v7** | ce0403c | L4 | 118 | **5/5** | all 1.0 (all in round 3) | **1** | PASS (20 solves, 1 kernel) | **adopted (mainline)** |
| E-v7 | ce0403c | L1 | 119 | 0/1 (stopped at the 2nd run for E-div) | 0.50 | — | — | aborted |
| E-v7 | ce0403c (deployed by SUBMISSION's clone steps) | L2 | 116 | in progress (since 14:50) | | | | regression check |
| E-div | 81549a0 (v7 + feedback_v8.py) | L3 / L4 | 119 / 118 | L3 5/5, L4 5/5 | L3 all in round 1, L4 all in round 3 | L3 3, L4 5 | PASS (only 1 kernel per level) | varying the samples gives more trajectories, but they converge to the same solution |
| E-div | 81549a0 | L1 | 117 | 0/1 (stopped at the 2nd run) | 0.50 | — | — | — |
| ablations A1–A5 | ce0403c (v7 base, --repeat 1) | L3 / L4 | 119 / 118 | only A2 (CARD=theirs) failed to solve L3 | A1 L4 round 2; A2, A3 L4 round 5; A4, A5 L4 round 3 | — | — | L3 relies on CARD=category; L4 does not depend on any single switch |
| v8 (SKELETON on) | 9a19bb0 | L4 / L3 / L1 | 118 / 119 / 117, 119 | L4 0/2, L3 1/1 (round 3), L1 0/2 | L4 0.62 ×2 | — | — | **SKELETON not adopted** (the model fills the <…> wrongly, L4 gets stuck out of bounds) |
| v8 (after 81c37cf) | 6f11031 | L1 | 116 | 1/1 (round 1) | 1.0 | — | PASS, lowers to trn2 | first L1 solve in the simulator |
| L2 round-0 diagnosis | 9a19bb0 / original agent.py | L2 r0 ×20 | 116 / 119 | v8 0/20, v8p (original first-round prompt) 0/20, **v8ps (original prompt + original sampling) 2/20, original agent.py 2/20** | — | — | — | **the L2 regression is caused by SAMPLING=qwen** |
| **v8.2 (final candidate)** | **96a9fc9**, md5 c2f81a; V7.md's exports + SKELETON=0 TRUNCFIX=1 L1FIX=1 MIXSAMP=1 L2HINT=1 MIX=1 | **L1** | 117 ×3, 118 ×2 | **5/5** | solved in rounds 2, 2, 0, 4, 2 | 5 kernels | PASS ×5, all lower to trn2 | **adopted** |
| **v8.2** | 96a9fc9 | **L2** | 116 ×3, 119 ×2 | **3/5** | [.5, 1, .5, 1, 1], solved in rounds 4, 2, 1 | 3 kernels | PASS | adopted (≥ baseline; v7 is 0/3) |
| **v8.2** | 96a9fc9 | **L3** | 118 ×3, 119, 116 | **5/5** | solved in rounds 0, 2, 0, 0, 0 | 3 kernels | PASS | adopted |
| **v8.2** | 96a9fc9 | **L4** | 119 ×5 | **5/5** | all in round 2 | 1 kernel (same as v7) | PASS | adopted |
| **v8.3** | **5c3aba2**, md5 87ddb0; v8.2's switches + L2CAT=1 | **L2** | 117 ×5, 119 ×4 | **7/9** | 117: 1,1,1,1,.5; 119: 1,1,.5,1 | — | trn2 full build + birsim all match | L1/L3/L4 request bodies are byte-identical to 96a9fc9's (mock endpoint --all, 48 requests); v8.2's results carry over |
| v8.2 extra runs | 96a9fc9 | L1 / L2 | 118, 117 | L1 +2/2, L2 +2/3 | — | — | — | v8.2 totals: L1 7/7, L2 5/9 |
| final_all | 96a9fc9, `--all --repeat 1` | L1–L4 | 116 | L1 1.0, L2 1.0, **L3 0.30**, L4 1.0 | solved in rounds 6, 7, —, round 2 | — | — | one command runs all 4 levels; this time L3 was not solved (the same failure repeated after 4 rounds, stopped early) |

**Final runs (18:25 stop; logs in `analysis/logs/final/seat-{115..119}/`, flat, one file set per prefix: `<prefix>.jsonl`, `_verdicts.jsonl`, `_nki_verdicts.jsonl`, `_usage.jsonl`, `run_<prefix>.log`; each seat's `v82_queue.log` lists every run's start, end and score). Rounds below are the log's "after N rounds" (N = 0-based round + 1).**

final = **0eb2695** (tag `final`). Runs per level come from 96a9fc9 (L1, L3, L4; prefixes `v82_`, `v82x_`, `final_all`), 5c3aba2 (L2, `v83_`), 0eb2695 (L2 `v85_`, L9-14 `final_L*`) and 15fb0d5 (L5-7, `v84w_`, `warm_`). Their request bodies are byte-identical on the levels each is quoted for: 1c2f915, 4fdeb0e, c860074.

| run set | commit | level | seats | solved | scores (after N rounds) | re-audit |
|---|---|---|---|---|---|---|
| v8.3 L2 (all) | 5c3aba2 | L2 | 117 x6, 119 x5 | **9/11** | 117: 1(3),1(2),1(4),1(5),.5,1(2); 119: 1(3),1(1),.5,1(2),1(2) | PASS (27 kernels over v8.2/v8.3/final_all, 116-119) |
| v8.5 L2 | 0eb2695 | L2 | 117 x3, 118 x3 (+2 partial L2d, stopped) | **4/6** | 117: 1(5),1(3),1(3); 118: .30, .50, 1(2) | not run (out of time) |
| WARM L5-7 | 15fb0d5 | L5 / L6 / L7 | 115-119 | **0** | L5 best .88 (6 runs), L6 best .75 (6), L7 best .75 (6; plus partial warm_L7 on 117) | - |
| held-out L9-14 | 0eb2695 | L9-L14 | 119, 115, 117, 118 | L9 **2/2**, L10 **1/1**, L11 0/1 (.67), L12 0/2 (.50, .30), L13 0/1 (.50), L14 0/1 (.50) | L9 1(3),1(3); L10 1(3) | L9, L10: PASS by a fresh-process `grade()` (plain `scripts/reaudit.py` cannot load levels 9-14: KeyError 9) |

- The extra v8.3 runs count: 117 `v83_L2d`, 119 `v83_L2b` and 119 `v83_L2c` have the same switch line and md5 87ddb0, and each wrote its SOLVED line before anything was stopped (119's at 17:08 and 17:11, per seat-119's `v82_queue.log`). The 17:13 stop hit 119's `v83_L2d`, which has only a start line and is excluded. (Corrected from 8/10 after exec 2 counted the queue log.)
- v8.5's one added sentence (level 2, "returned ()") never fired: no v8.5 attempt hit "returned ()". So the v8.5 L2 runs are v8.4/v8.3 L2 runs, not evidence for the change.
- L9 and L10: held-out verdict UNVERIFIED ("no held-out check ran": the held-out set has no levels 9-14); verdict_nki VERIFIED, 15/15 extra cases, lowers for trn2. L12 stalls on `s = s + 1e-6` TypeError (tile plus Python float).
- Seat-115 served with `--max-num-seqs 8` (others 4); its runs: warm_* and final_L10/L14.
- Why L5-7 stay at 2.00x / 1.56x (found 17:40, not implemented): the traffic hint says hoist loads out of the innermost loop, but the warm kernel's innermost loop is k, where loads are needed. The excess is outer: at K=256 M=512 N=1024 rhs is re-read once per m tile (4x) and lhsT once per n tile (2x), 7.34 MB against a 3.67 MB floor; n outer with the rhs strip kept in SBUF gives about 1.14x. The model mostly resubmitted the warm kernel unchanged. Per-operand traffic in the feedback would need telling lhsT from rhs inside `dma_copy`, where `src.name` is empty in the 0.6 simulator.

**Chip-level verification (17:20)**
- **trn2 full build + birsim** (`check/compile_solves7.py`, CPU only): v8.2's L1 (5 kernels), L2 (3), L3 (3), L4 (1), plus v8.3's L2 solves, **all compile, and birsim MATCHES for all** (max error as a fraction of RMS: L1 about 1.9e-07, L2 0, L3 1.2e-06, L4 2.9e-06). The compiler rejected none of them.
- **Measured on a NeuronCore** (`check/device_check.py`, seat-116, vLLM stopped first and started again afterwards): L1 kernel 53900f4b **matches the reference on the chip for 4/4 shapes**, device error 2.4e-07–4.8e-07. 1.5–2.0 seconds per call, including host overhead; this is not kernel latency.
- Result files: `runs/seat-117/l1_compile/compile.txt`, each seat's `/tmp/cmp/compile_s*.txt`, `/tmp/device_l1.json` (seat-116); all will be copied into `analysis/logs/final/`.

Logs: `runs/seat-<N>/<experiment>/` (not in git). Every 1.0 is re-audited with `scripts/reaudit.py`.

**Confirmed findings**
1. **About 50 seconds per round, spent in model generation, not in scoring; there is no cheap speed-up.** Measured on seat-116 at 12:50: 1 concurrent stream 13.9 tok/s, 4 concurrent streams 22.1 tok/s in total (5.5 each). The model really runs on Trainium (`neuron-ls` shows the process; `PJRT_DEVICE=CPU` is set by vllm_neuron itself). Of the 11-core CPU quota only about 5 cores are used and throttling is about 1%, so the CPU quota is not the bottleneck; `vllm._C` being missing is normal on Neuron. The only knobs are `--optimization-level 3` (default O1) or TP=4; both need a recompile of unknown duration and would invalidate the baseline, so **we decided not to change them**. Countermeasure: run several seats in parallel, testing one level at a time; output tokens are expensive, input tokens are cheap (prefill about 270 tok/s).
2. **Failure categories** (5 runs, 424 attempts, `analysis/baseline_seat116_summary.csv`):
   - copy with mismatched sizes on the two sides 96 (L1 40, L2 30, L3 26)
   - invented functions that do not exist 80 (all in L1: `nisa.multiply`, `nisa.scalar_mul`)
   - index out of bounds 64 (L2/L3/L4)
   - tile over 128 rows 55 (the main blocker on L4)
   - misused reshape 50 (L3 48), 1-D tile 22 (L3)
3. **The harness's feedback already carries fix suggestions** (`enrich()` in `agent.py`), yet the same errors keep recurring, which means the existing suggestions do not work. Changes to the feedback should start there.
4. **No permission for port-forward**; the agent can only run inside the pod; `kubectl cp` / exec work.

**Reference repo: aws-neuron/neuron-agentic-development** (recommended by an AWS engineer, cloned to `../neuron-agentic-development/`, read-only, not put into our repo)
- The most useful part is `skills/neuron-nki-docs/references/`: `indices/symbol-lookup.md` (every NKI symbol and its module), `programming/api/*.md` (API signatures), `debugging/error-codes/`
- Example: `multiply` exists, but it is `nl.multiply`, an **operation type** to be passed to `nisa.tensor_tensor(..., op=nl.multiply)` / `nisa.tensor_scalar(...)`, not a function in `nki.isa` that can be called directly. Right now the harness suggests "scalar_engine..." by letter similarity, which does not help the model
- ⚠️ The docs are for NKI 0.4.0; the pod has **0.6.0**. Names written into the feedback must first be confirmed to exist in the pod
- ⚠️ `references/downloads/*_nki_kernels.py` (average_pool2d, matmul, transpose2d) are essentially the reference answers for levels 1–4; **they must not go into the prompt** (that would leak the answers), they are for humans only

**In progress / next steps**
1. [Experiment A is wired into `KNOWN_FIXES` in `enrich()`, commit 5ed7ec2, to be verified on seat-116 on L1] There are only 4 invented names (nisa.multiply 36, transpose_moving 13, nisa.scalar_mul 12, tile_size() 1); in 0.6.0 `nl.multiply` can be called directly, and `nc_matmul`'s transpose parameter is called `is_transpose`
2. Put a trimmed API card (the dozen or so functions levels 1–4 use, about 300 tokens) into the prompt
3. liuyq's `feedback_v2.py` / `feedback_v3.py` (commit fc137e7) already cover L4's "tile over 128 rows" and "copy size mismatch"; on the 4090 she solved L4 5/10 (original 0/15); to be confirmed on the pod
4. [Hardware legality check: **verified with real nki 0.6.0** (local Docker, per `SETUP_PYTHON.md`)] The simulator does not check limits when allocating tiles; we now record every `nl.ndarray/zeros/ones/full` during simulation and judge partitions >128, PSUM >16 KiB per partition, SBUF >192 KiB per partition as "ILLEGAL ON HARDWARE", ahead of the numeric comparison; spanning several PSUM banks is legal and not flagged. Verification: selftest passes; all 4 reference kernels pass (12/12/5/90 allocations audited); an L4 kernel that "allocates (256,128) SBUF but moves only 128 rows per DMA" gets 4/4 from the real simulator with the audit off (the hole is real) and is judged ILLEGAL with the audit on; a PSUM spanning 2 banks is not falsely flagged. **Fixed a pitfall (13:55 correction: more serious than first stated)**: nki caches by file path, so within one process a later candidate at the same path **with the same byte count** is simulated as if it were the first one -- **both the numbers and the allocations are stale**. Measured: the reference L4 with nc_matmul's two operands swapped (same length; on its own it raises an error) is judged 4/4 when written to the same path right after the reference. So runs that graded consecutively in one process before c39c0ce (baseline, the 4090's L4 5/10, earlier runs on 117/116) may contain **fake solves**; **every 1.0 must first be re-audited with `scripts/reaudit.py`**. Re-audited so far: baseline L2 (2 kernels) PASS; 117's L4 v3 solve PASS. Now `agent.py` / `feedback_v2.py` write each candidate to a unique path (`nkibench.candidate_path()`), and v2's `locate()` still points at the failing line. **26c43ed has this pitfall, do not run experiments with it** (or set `NKIBENCH_NO_ALLOC_AUDIT=1`). The 4090's L4 5/10 was measured without the audit and must be re-audited when reproduced.
   - **13:35 seat usage**: 115 runs `daykit/.../feedback_v7.py` (L2, L9); 116 runs experiment A (L1, `attempts_expA_L1.jsonl`); 117 runs L4 v3 `--repeat 5` (`attempts_v3_L4.jsonl`, started 13:13, old `agent.py`/`feedback_v2.py`, **its solved kernels must be re-audited**); 118 is restarting vLLM (new config, compiling); 119 runs `agent.py --all --repeat 5`.
   - **Re-audit**: `python scripts/reaudit.py <attempts.jsonl ...>` runs `--check` on every 1.0 kernel in a fresh process of its own (needs nki: the pod or SETUP_PYTHON.md's Docker). Baseline re-audited: all 4 L2 solves (2 distinct kernels) PASS; the baseline numbers hold.
5. Add per-segment token accounting to the agent (how much of the prompt is rules / previous code / errors / failure ledger) and confidence output
6. Validate every change with `--level X --repeat 5` and roll back if it gets worse; freeze around 16:30, run the final comparison on several seats in parallel, submit before 18:15

**File index**: `GETTING_STARTED.md` (commands to connect to a seat) · `scripts/connect.sh` · `scripts/sync.sh` · `scripts/attempts_to_csv.py` · `analysis/` (category tables) · `projects/02-kernel-agent/{agent.py,nkibench.py}` (the code to change)

---

## 1. Scoring rubric (the same for every problem)

| weight | item | key points |
|---|---|---|
| **30%** | Correctness | How many levels pass on **the judges' held-out shapes and adversarial values**, not your own tests. **Breaking a rule = 0 points**, not a deduction. |
| **25%** | Deliverables | Levels passed + attempts used per level. Fewer attempts = good agent design, not luck. |
| **25%** | Method and honesty | **Does the agent know when it failed?** Reporting "verified" when it did not pass is **worse** than honestly reporting a failure. Plus token budget accounting and a failure taxonomy. |
| **20%** | Demo and write-up | Show **a failure and a recovery**, not just successes. Can others reproduce it. |

**The two things that set teams apart:**
- **Calibration**: the agent must output a confidence, and it must be accurate. "I cannot verify level 8" is worth more than "done" followed by a lie.
- **Failure taxonomy**: run the whole ladder, collect every wrong kernel, group them into a few named failure modes with counts. Needs no accelerator, and it is what the judges most want to keep.

> A team that "passes 4 levels + gives a rigorous failure taxonomy of the other 6" beats a team that "claims to pass level 9 but cannot show verification" -- that is the rubric itself.

---

## 2. Deliverables

**README Part 4 (required for every project):**
1. **Your checker**, and the reasons for what it accepts/rejects (this is what the organizers want to keep).
2. **attempt log**: every attempt + score (the agent already writes `attempts.jsonl`; `scripts/sync.sh <N> pull` pulls it to the local `runs/`).
3. **One-page write-up**: what was run, on what, and the results -- **including how many runs and the spread**.

**CHALLENGE-kernel-agent.md (when choosing the kernel agent problem):**
1. The agent itself
2. The verification harness -- **state the tolerance and the reason** ("rtol=1e-5" is an answer, "looks about right" is not)
3. **eval set**: shapes and values tested, including adversarial values. **Mandatory**.
4. Failure taxonomy (groups + counts)
5. **Token accounting**: input tokens per attempt and where they go (docs / error context / code). A "token allocation per attempt" chart is the most valuable thing in the demo → matplotlib is installed in the local venv.
6. A one-page reproduction guide

---

## 3. Known pitfalls (all measured)

### Model / prompt
- **Do not turn on thinking (`--think`).** Measured with Qwen3: per round ~8s → **446s**, scores 0.30–0.62 → **0.00**, every sample truncated with `finish_reason=length` at ~9,900 characters with no code. **A bigger token budget does not help; the fix is a shorter prompt.** Check this first when the agent returns nothing.
- **Do not pile rules into the prompt.** Give a list of constraints → the model self-checks item by item, loops, and outputs nothing; give no constraints → it outputs confident but rule-breaking code. **The right way: let generation run free, let the verifier catch violations, then send back a single "change only this one thing" instruction. Constraints belong in the verifier, not in the generation prompt.**
- **verdict ≠ instruction.** Sending back `line 16: calls banned max` as is → the same violation comes back; rewriting it as "replace np.max/np.sum with explicit loops, leave everything else unchanged" → fixed in one round.
- **Do not put target values in the feedback.** Given the correct coefficients, the model copies them and derives nothing. Only say **where it is wrong and in which direction**.
- **Give tools, not hints.** The model cannot compute integrals → give it a calculator it calls itself (project 1's `tool_calc.py`).
- **A plausible prompt improvement can make everything worse.** Adding a "copy in blocks of 128" example: level 2 went from 2/5 → 0/5, level 4 from 0.62 → 0.30; rolled back. **Do not add it back without measuring.**
- **Failures move rather than disappear**; scores can go down as understanding gets better (because of how the weights are split).

### Measurement
- **Report the success rate over `--repeat N`, not the best run.** Level 2 with the same settings 5 times: `[1.00, 1.00, 1.00, 0.50, 1.00]`; luck alone gives 0.5↔1.0.
- Levels 1/3/4 gave identical results in all five runs (0 variance) = **a capability wall**, so the effect of a change can be attributed cleanly; level 2 is **luck-limited**, and single comparisons mean nothing.
- The three walls are really the same NKI idiom -- **tiling**:
  - L1: `SBUF and PSUM tensors must have at least 2 dimensions` (built a 1-D tile)
  - L3: `cannot reshape array of size 32768 into shape (1,64)` (reshape instead of slicing)
  - L4: `dma_copy dst partition dimension 256 exceeds maximum 128` (one tile for the whole tensor) → 0.62, only 1 of 4 shapes passes
- **Say clearly which numbers come from the simulator and which from the device**; layer 2/3 (real latency) is not done yet, all numbers so far are throughput estimates from `nki.simulate`.
- 222 Flops/Byte is the bf16 ridge and the test shapes are float32 → the conclusion is only indicative.

### Parameters
- **`--context 8192`** (the server in the seat pod is 8192). The repair prompt has to hold the previous kernel + checker instructions + the failure ledger; 4096 squeezes out room for the answer.
- Local Qwen3: `--samples 4` (the server runs 4 streams at once, almost free).
- Shared gpt-oss-20b: **greedy decoding** → `--samples 1` (multiple samples = identical answers), `--terse 1` (a long prompt makes it only reason and not answer), retrying the same prompt is pointless, the `tools=` parameter has no effect, input limit 8192 tokens, near the limit it **truncates silently** -- **check `finish_reason` on every call**, `max_tokens` at least 2500. ~4 req/s shared by everyone.
- 8B **beat** 20B on level 4 (0.62 vs 0.30): in 4 of 6 rounds gpt-oss returned empty after ~10,000 characters of hidden reasoning.

### Environment / cluster
- **Credentials are only valid in the current terminal, and they expire.** Paste them again in a new terminal/tab; on `ExpiredToken` paste the latest block from the channel. **Never write them into any file, never commit them.**
- **No `kubectl port-forward` permission** (RBAC only grants pods get/list/watch, pods/log, pods/exec). This machine cannot reach `localhost:8000` inside the pod, **the agent can only run inside the pod**. Measured on seat-116: `can-i create pods/portforward` = no, port-forward reports `cannot create resource "pods/portforward"`. `kubectl cp` works both ways (goes through exec).
- After `kubectl exec -it`, **wait for the `root@seat-N:/workspace#` prompt before typing**, otherwise the input goes to the local shell.
- Long jobs always use `nohup python agent.py ARGS > run.log 2>&1 < /dev/null &` + `tail -f run.log`, otherwise they die when the connection drops. `pgrep -af agent.py` shows whether it is still running.
- **If the pod is replaced, `/workspace` is gone** → push to your own git or `scripts/sync.sh pull` back to this machine in time.
- Before any git command in the pod: `git config --global --add safe.directory /workspace`.
- A NeuronCore cannot be shared by two processes: vLLM uses NC 2–3, NC 0–1 are free; for on-device timing later use `NEURON_RT_VISIBLE_CORES=0,1` (not verified).
- `nki` only exists in the Trainium pod; failing to import it locally is normal. What runs locally: `kernelbench.py` (Stage A, pure NumPy) and `nkibench.py --selftest` (skips the simulation part).

---

## 4. Commands to run next

### Local (every new terminal)
```bash
# 1) paste the macOS/Linux credentials block from the channel (export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN)
# 2) connect to your own seat
scripts/connect.sh <N>
kubectl exec -it seat-<N> -- bash
```

### In the pod, terminal 1: start the model
```bash
neuron-ls
cd /workspace && ./serve.sh                      # ~4 minutes, wait for READY; do not close it
```

### In the pod, terminal 2: run the agent
```bash
git config --global --add safe.directory /workspace
cd /workspace/projects/02-kernel-agent
python nkibench.py --selftest                     # first prove the harness can be trusted
python nkibench.py --level 4 --check reference_level4.py
nohup python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5 > run.log 2>&1 < /dev/null &
tail -f run.log                                   # this is the baseline; record solved x/5 and the spread for each level
```

### Local: sync
```bash
DRY_RUN=1 scripts/sync.sh <N> push             # first see which files would be pushed
scripts/sync.sh <N> push                       # locally changed projects/ files → pod /workspace
scripts/sync.sh <N> pull                       # the pod's run*.log / *.jsonl → runs/seat-<N>/<time>/
```

### Local: Stage A, doable without the cluster
```bash
source .venv/bin/activate
cd projects/02-kernel-agent
python kernelbench.py --selftest
python kernelbench.py --list
python kernelbench.py --level 1 --show
python kernelbench.py --level 1 --check my_kernel.py
```

### Suggested order
1. First run the `--repeat 5` baseline and record the numbers (this is the reference for every later comparison).
2. Change only **the checker's feedback wording** (turn verdicts into instructions), one change at a time, `--repeat 5` every time.
3. Record the token allocation from day one (docs / errors / code / ledger) and plot it at the end.
4. Keep collecting wrong kernels and count them by failure mode.
5. Add a confidence to the agent's output and check whether it is accurate.
