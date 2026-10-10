# Hack the Chip: repo context and hackathon playbook

> Read this first when resuming work. This is a snapshot of the repository plus a recommended plan, not a claim that the proposed improvements have been implemented.
>
> Analyzed: **October 10, 2026**. Base commit: **`8f1ca418827a3b4adc88a766262d065b639b7d46`**. Local analysis machine: macOS; Python 3.14; NumPy 2.4.4; SymPy 1.14.0; httpx 0.28.1. **No NKI SDK, Trainium execution, or live model calls were available during this analysis.** Historical model results below come from the repo's documents, not new experiments.

## 1. What you are actually being asked to do

The event is **Hack the Chip — NYU × Annapurna Labs**. Its central challenge is:

> Build an agent around a small model running on Trainium. Have it attempt a problem, check the attempt, turn the failure into useful feedback, and try again. Prove what the loop accomplishes.

The important artifact is the **checker and the feedback it produces**. A chatbot, a single successful answer, or a screenshot of a model responding does not demonstrate the central idea.

A simple example:

```text
Task: write a numerically stable softmax kernel.
Attempt 1: generates exp(x) / sum(exp(x)).
Checker: detects non-finite results on large inputs.
Repair instruction: subtract the maximum of each row before exponentiation.
Attempt 2: generates a repaired kernel.
Checker: verifies the code against the reference on every evaluation case.
Result: show both attempts, the failure, the repair, and the measured outcome.
```

There is **no training requirement** in the central assignment. These starter agents perform inference and repair; they do not update model weights. Reinforcement learning is an optional extension, and would require substantial additional work.

The root README gives three universal deliverables:

1. **Your checker**, with an explanation of what it accepts and rejects.
2. **Every attempt and its score**, so someone else can inspect the loop.
3. **A one-page reproduction note**: model, hardware, settings, results, number of runs, and spread.

The kernel challenge additionally requests the agent, evaluation cases, failure taxonomy, and token instrumentation. Prepare that superset if choosing kernels.

Sources: [root README](README.md), [kernel challenge](projects/20-kernel-agent/CHALLENGE-kernel-agent.md).

## 2. My recommended project and scope

**Choose Project 2, the kernel agent, if you have a functioning workshop seat and several hours. Build a checker-driven repair controller and a rigorous evaluation report.**

Suggested project pitch:

> “We turn kernel failures into small, targeted repairs under an 8192-token context limit, and measure when those repairs improve correctness and when the agent should stop.”

The starter already provides the model connection, references, grading, and a retry loop. Your contribution should be identifiable: better diagnosis, targeted context selection, stronger checks, or improved recovery rates. Re-running the starter is a baseline, not the finished project.

First target **NKI level 4: tiled matmul**. The historical baseline passes one of four shapes and fails the larger tiled cases, so there is a concrete gap to close. Level 3 is a smaller milestone; level 2 is a working control. Level 1 can be useful if its two-dimensional-buffer failure is easier to fix.

Aim for this minimum credible result:

- A baseline and improved agent evaluated with the same settings.
- At least one clear failure-to-success recovery on a previously failing case or level.
- Complete attempt records, a named failure taxonomy, and explicit verification status.
- Repeated runs and separate evaluation shapes that were not used to tune prompts.
- A short demo and reproduction note, including remaining failures.

**Fallback if infrastructure consumes too much time:** use the NumPy kernel ladder. It works on this laptop. The full automatic agent for that ladder is not supplied, so you must implement the controller or start with the manual driver.

**Fallback with roughly 1–2 hours left:** extend Project 1 with a controlled feedback/tools comparison and new problem seeds. It has fewer environment dependencies, but reproducing its existing six solved problems alone offers little novelty.

A higher level is not automatically a stronger submission. A trustworthy improvement with honest limits aligns better with the documented rubric than an impressive claim without verification. This is a strategy recommendation, not a guarantee of judging outcomes.

## 3. The documented rubric

The following weights are in the **kernel challenge document**. The repository mixes older laptop work and newer NKI work, so confirm with organizers that this is the rubric used for your track today. You can prepare against it while they clarify.

| Weight | Criterion | What to optimize for |
|---|---|---|
| 30% | Correctness | Passing held-back shapes and hostile values; rule violations invalidate a result |
| 25% | Delivered result | Levels cleared and attempts needed to clear them |
| 25% | Method and honesty | Detecting failure, confidence/calibration, token accounting, failure taxonomy |
| 20% | Demo and write-up | A failure followed by recovery, with a reproducible procedure |

The challenge mentions **three held-back operations** for the NumPy ladder. Their implementations and the actual current judging set are not present here. Do not infer that the NKI ladder has the same hidden levels.

The README says teams are typically 3–5 and asks custom-project proposals to be discussed with an engineer before 11:30. Treat that as a documented event instruction, not a verified schedule for today.

## 4. Repo map: where to go for each job

| Path | Purpose | When to open it |
|---|---|---|
| [README.md](README.md) | Assignment, participant access, baseline measurements | Setup and overall task |
| [STATE.md](STATE.md) | Detailed history, known failures, abandoned ideas | Historical context; some setup/status text is stale |
| [serve.sh](serve.sh) | Starts the local Qwen3 model inside your seat pod | Serving and logs |
| [requirements.txt](requirements.txt) | httpx, NumPy, SymPy | Laptop dependencies |
| [Project 1 README](projects/01-heat-rod-pde/README.md) | Heat equation task and lessons | Understand a working checker/repair example |
| [Project 1 agent.py](projects/01-heat-rod-pde/agent.py) | Sampling, calculator exchanges, retry loop | Extend the physics agent |
| [pdecheck.py](projects/01-heat-rod-pde/pdecheck.py) | Physics grades and directional feedback | Improve the physics checker |
| [tool_calc.py](projects/01-heat-rod-pde/tool_calc.py) | SymPy calculator using `COMPUTE:` lines | Tools/no-tools experiments |
| [level0_heatrod.py](projects/01-heat-rod-pde/level0_heatrod.py) | Three easy generated physics problems | Warm-up and checker tests |
| [level1_heatrod.py](projects/01-heat-rod-pde/level1_heatrod.py) | Insulated boundaries and parabola initial conditions | Harder physics cases |
| [Project 2 README](projects/20-kernel-agent/README.md) | NKI project history, references, optimization discussion | Read with the caveats below |
| [CHALLENGE-kernel-agent.md](projects/20-kernel-agent/CHALLENGE-kernel-agent.md) | NumPy challenge, rubric, context-budget problem | Submission design and laptop track |
| [Project 2 agent.py](projects/20-kernel-agent/agent.py) | Automated **NKI** agent | Main modification point for recommended project |
| [nkibench.py](projects/20-kernel-agent/nkibench.py) | **NKI** references, simulator, static checks, traffic counting | NKI verification |
| [kernelbench.py](projects/20-kernel-agent/kernelbench.py) | **NumPy** references, hostile cases, static checks | Laptop verification |
| [try_level.py](projects/20-kernel-agent/try_level.py) | Manual NumPy generation/repair through gpt-oss | Prototype repair prompts |
| `projects/20-kernel-agent/reference_level{1,2,3,4}.py` | Shipped correct NKI tutorial kernels | Validate harness; understand legal API patterns |
| [gptoss/README.md](gptoss/README.md) | Shared endpoint behavior and measurements | Before using the shared model |
| [gptoss/chat.py](gptoss/chat.py), [web.py](gptoss/web.py) | Terminal/browser chat clients | Endpoint smoke test or optional presentation |
| [gptoss/probe.py](gptoss/probe.py), [loadtest.py](gptoss/loadtest.py) | API and performance experiments | Only if endpoint measurement is your project |
| [gptoss/mockserver.py](gptoss/mockserver.py) | Canned local endpoint | Client development; never model evaluation |
| [k8s/workshop-seats.yaml](k8s/workshop-seats.yaml) | Actual participant environment configuration | Resolve defaults and pod behavior |
| `k8s/*job*.yaml`, `k8s/workshop-rbac.yaml` | Cluster jobs and access controls | Primarily facilitator/development infrastructure |
| [workshop/FACILITATOR.md](workshop/FACILITATOR.md) | Organizer setup, seat assignment, credentials | Understand access; participants need not deploy the cluster |

No applicable `AGENTS.md` was found in the repository or checked parent directories during this analysis.

## 5. Crucial distinction: there are TWO kernel ladders

Do not mix their level numbers, rules, signatures, or success claims.

### A. NumPy ladder: `kernelbench.py`

Runs on a CPU with NumPy, without the NKI SDK. Candidates define **`kernel(...)`**, matching the reference's arguments.

| Level | Operation | Main trap |
|---|---|---|
| 1 | `relu(a*x+b)` | Establish that the pipeline works |
| 2 | Row sum | Final partial tile |
| 3 | Row max | Start from `-inf`, not zero |
| 4 | RMSNorm | Accumulation precision |
| 5 | Softmax | Subtract row max before `exp` |
| 6 | Transpose | Non-square, non-divisible shapes |
| 7 | Matmul | K-tail handling and accumulation dtype |
| 8 | LayerNorm | Cancellation in `E[x²] - E[x]²` |
| 9 | Windowed attention | Band boundaries and window larger than sequence |
| 10 | 1D convolution | Stride/dilation output-size arithmetic |

Rules call for explicit tiles no larger than 128 rows × 512 columns, explicit tile loops, no operation-level shortcuts, and handling ragged dimensions.

The standard 2D test builder includes `(129, 513)`, `(1, 7)`, `(7, 1)`, `(31, 127)`, and `(257, 61)`, along with normal, large-offset, repeated-row, and zero inputs. Levels 7, 9, and 10 have their own builders.

Important test-data detail: **“identical rows” means rows repeat the same vector**, not that every element within each row is equal. Add truly constant rows if your operation needs that case. Add all-negative rows to exercise max initialization explicitly.

The default numeric rule is approximately:

```text
max(abs(got - want) / max(abs(want), 1e-30)) <= 1e-4
```

It also checks output shape and finite values. Near-zero expected outputs make that purely relative rule very strict. Any tolerance change must be justified and recorded; do not silently loosen it to turn failures green.

### B. NKI ladder: `nkibench.py` and `agent.py`

Candidates are real NKI source with `@nki.jit`. Their correctness is currently checked using **CPU simulation requiring the Neuron/NKI SDK**. CPU simulation does not require a free Trainium core, but it does require the SDK environment.

| Level | Operation | Required entry point / extra requirement |
|---|---|---|
| 1 | 2D average pooling | `tensor_avgpool_kernel` |
| 2 | Transpose free axes within each partition | `tensor_transpose2D_kernel_` |
| 3 | Single-tile matmul | `nki_matmul_basic_` |
| 4 | Tiled matmul | `nki_matmul_tiled_` |
| 5 | Hoisted-load matmul | `nki_matmul_hoist_load_`; HBM traffic ≤ 1.60× floor |
| 6 | M/N blocked matmul | `nki_matmul_block_free_dimension_`; ≤ 1.25× floor |
| 7 | Fully blocked matmul | `nki_matmul_fully_optimized_`; ≤ 1.05× floor |
| 8 | Single-head attention | `nki_attention_`; integration bug described below |

**`agent.py --all` runs only levels 1–4**, not all eight registered levels. Use `--level 5`, etc., explicitly. Only levels 1–4 have shipped NKI reference kernel files.

NKI level 2 is not an ordinary whole-matrix transpose. Input shape is `[P, F1*F2]`; each partition's flattened `F1 × F2` matrix is transposed while the partition axis stays in place.

NKI matmul receives **`lhsT: [K,M]` and `rhs: [K,N]`**, and returns `[M,N]`. The NumPy truth is `lhsT.T @ rhs`. Layout mistakes here are an easy way to get plausible wrong numbers.

The NKI mismatch test defaults to **maximum absolute error divided by the reference output RMS ≤ 0.02**, with an RMS fallback of 1 for zero outputs. That is not the NumPy ladder's elementwise relative tolerance and not ordinary `np.allclose(rtol=0.02)`.

## 6. What is already solved, and what remains open

### Project 1: heat equation agent

Task: propose a temperature function `u(x,t)` satisfying `u_t = k*u_xx`, two boundary conditions, and a starting temperature profile.

The checker uses SymPy differentiation and numeric evaluation. The PDE and boundaries are evaluated at sampled times/points; the starting shape is compared using a grid-based relative L2 error. **It is not a universal symbolic proof for arbitrary generated expressions.**

Reward:

- 0.4: equation holds.
- 0.2: left boundary holds.
- 0.2: right boundary holds.
- 0.2: initial condition matches within the problem's tolerance.

Level 0 has three sine-wave warm-ups. Level 1 has two insulated-boundary cases and a parabola initial condition. The parabola is approximated by a truncated series; the supplied test verifies that three odd terms meet the 0.5% starting-shape tolerance.

Historical status is “solved 6/6,” with important qualifications:

- An early transcript disclosed target coefficients in feedback, which the model copied.
- Current `REVEAL_COEFFICIENTS = False` uses directional feedback.
- The model can request exact integrals through `COMPUTE:`; the tool returns the value of the integral the model chose.
- The repo reports tools succeeding where no-tools stalls, but those results still need repeated reproduction under your setup.
- Some older text conflates the insulated-boundary problems with the parabola problem. **Current `level1_heatrod.make(3)` uses two Dirichlet boundaries**, not an insulated right boundary.
- A sentence in the Project 1 README misinterprets reward 0.4. Use the actual weights and `parts` output, not that sentence, to explain a score.

Useful extension: compare scalar-only feedback, existing directional feedback, and directional feedback with calculator tools on multiple generated seeds. Keep coefficient disclosure disabled throughout.

### Project 2: historical model baseline

These are **repo-reported measurements**, not results collected in this analysis:

| NKI level | Qwen3-8B, 5 runs | gpt-oss-20b, 3 runs |
|---|---|---|
| 1: pooling | 0/5 solved; best reward 0.30 each run | 0/3; 0.30 each run |
| 2: transpose | 4/5 solved; rewards 0.5–1.0 | 3/3 solved |
| 3: single-tile matmul | 0/5; 0.30 each run | 0/3; 0.30 each run |
| 4: tiled matmul | 0/5; approximately 0.62 each run | 0/3; 0.30 each run |

The top of the Project 2 README says no level was solved and best score was 0.30. That is superseded by later measurements in the same README and `STATE.md`. Level 2 has succeeded; level 4 remains a strong repair target.

An earlier measurement had Qwen level 2 solving 2/5. Do not present 2/5 → 4/5 as an established improvement: settings/feedback and sampling varied, and n=5 is small.

Historical recurring failures:

| Level | Failure | Repair direction |
|---|---|---|
| 1 | SBUF/PSUM allocated as a 1D vector | Preserve a partition axis and a free axis |
| 3 | Reshaping a whole operand into an incompatible tile | Slice the correct operand region |
| 4 | DMA partition dimension exceeds 128 | Tile the large dimension and clamp bounds |

## 7. How the existing NKI agent works

Relevant functions in [agent.py](projects/20-kernel-agent/agent.py):

1. `first_prompt()` includes the reference, entry point, hardware limits, and an API card. `--terse` selects shorter forms.
2. `ask_parallel()` generates multiple candidates concurrently.
3. `extract_code()` extracts fenced code or a plausible module start.
4. `grade()` checks parsing and static rules, loads the kernel, simulates shapes, checks numerics and input mutation, and applies traffic limits.
5. `enrich()` turns known simulator/API errors into more actionable instructions.
6. `solve()` selects the best candidate **within the current round**, remembers the best result overall, but repairs the latest candidate with code.
7. Repeated failures trigger a compressed failure ledger, then a stop condition.
8. Attempts are appended to JSONL. `--repeat` prints an aggregate solve rate.

Reward components are 0.1 parsing, 0.2 rule compliance, 0.2 successful execution on at least one case, and 0.5 prorated shape passes.

| Reward | Typical meaning in current implementation |
|---|---|
| 0.00 | Empty or unparseable code |
| 0.10 | Parses; static rules fail |
| 0.30 | Static rules pass; kernel loading/execution can still fail |
| 0.50 | Executes on at least one case; no cases fully accepted |
| ~0.62 | For a four-shape level, commonly 0.625: one case accepted |
| 1.00 | Current agent checks accepted every configured case |

**A rule violation can still receive 0.1 internal progress reward.** It must remain an invalid final submission. Reward is search guidance, not calibrated confidence and not the event's percentage score.

The agent's `ask()` estimates prompt tokens as `len(prompt)//4`, caps answer tokens, and prints truncation warnings. It does not preserve full API usage or finish metadata in the JSONL. This is an immediate instrumentation opportunity.

## 8. Code-level findings worth fixing first

These are analysis findings, not implemented fixes. They can make your project both more reliable and more distinctive.

### P0: make the final checker agree with the agent

`agent.grade()` checks **input mutation, levels 5–7 traffic bars, and selected hardware-hazard warnings**. `nkibench.verify()`, used by the standalone `--check` CLI, does not apply those same checks.

Consequence: a CLI “pass” and an agent “pass” can mean different things. Reuse a shared structured evaluation function and have both callers consume it. Report separate gates for legality, execution, numerics, mutation, warnings, and traffic.

### P0: strengthen the NumPy legality checks if choosing that track

`kernelbench.check_rules()` explicitly describes itself as an incomplete scan. Local diagnostic calls confirmed:

```text
Level 7: return a @ b       -> no static violation
Level 6: return x.T.copy()  -> no static violation
```

Those shortcuts contradict the intended tiled-kernel task. The scan also does not prove tile limits or loop coverage, and only catches some indexing forms. Add targeted checks, demonstrate them against planted violations, and document what still requires human review. Do not claim the current scan proves legality.

### P1: complete experiment logging

The NKI JSONL currently stores level, round, reward, `parts`, prompt/reply character counts, code, and feedback. It omits the full prompt, repetition ID, sample ID, per-attempt timestamps/latency, token usage, model config, and response finish reason.

Without repetition IDs, rounds restarting at zero across `--repeat` runs are ambiguous. Without actual prompts, the run cannot be reconstructed precisely. Store this metadata at the point where it exists, rather than estimating it later from console text.

### P1: fix the level 8 integration before trying attention

Both `agent.grade()` and `nkibench.verify()` assume every level ≥3 has matmul shape keys `M`, `K`, and `N` when computing intensity. Attention cases have `seq` and `dim`.

A local **stubbed diagnostic**, returning a correct attention reference output and a nonzero byte count, reproduced **`KeyError: 'M'`** in `agent.grade(..., 8)`. This was a controller test, not a simulated NKI success.

Use per-operation FLOP metadata or gate matmul-specific reporting to matmul levels. The terse prompt also applies matmul-specific guidance to every level ≥3; make that operation-specific too.

### P1: improve empty-answer recovery after a previous round emitted code

Prompt shortening is triggered when there is no saved latest code. If a later round returns empty output, an older kernel remains available, so the loop can return to the long repair prompt that caused the empty answer. This is also noted as untried work in `STATE.md`.

Handle empty/truncated responses as an explicit failure class. Shorten the current repair request, preserve the last valid code separately, and log what changed. Do not treat a longer reasoning-only response as progress.

### P1: add evaluation coverage beyond stock NKI inputs

NKI matmul levels 4–7 currently use tile-divisible dimensions. The input generators primarily produce seeded Gaussian float32 arrays. That does not cover the challenge's full hostile-value/ragged-shape ambition.

Add separate development and held-out suites with appropriate ragged dimensions, alternative seeds, zeros, negatives, constant rows, and large values. **The shipped level-4 reference explicitly asserts tile-divisible shapes**, so a ragged suite needs its own compatible oracle/baseline; do not label the supplied reference broken for violating a precondition it declares.

The NKI agent currently calls `make_inputs()` with its default seed; `--repeat` varies model generation but does not diversify evaluation inputs. Distinguish repeated model trials from held-out correctness evaluation.

### P2: make measurement limitations visible

- DMA traffic counting wraps `nisa.dma_copy`. Other transfer routes/aliases may not be fully measured.
- `check_traffic_bar()` skips enforcement if measured bytes are zero. Treat unmeasured traffic as unknown, not evidence of efficiency.
- `simulate_and_count()` temporarily patches a module-global function. Candidate grading is currently sequential; parallelizing simulation without isolation could corrupt counts.
- `grade()` writes `/tmp/_agent_levelN.py`. Concurrent processes evaluating the same level can collide. Use per-run temporary paths.
- Agents do not currently expose calibrated confidence. Add explicit verification statuses first; do not invent a probability from reward.
- The harness executes candidate Python in-process and has no per-candidate timeout. A broken infinite loop can stall the run. A subprocess worker with a timeout is useful if you have time; the AST scan is not an execution sandbox.
- `requirements.txt` permits NumPy 1.26, while the physics checker calls `np.trapezoid`. If that attribute is missing, align the dependency floor or provide a compatibility fallback.

## 9. The project design most likely to pay off

Build on the starter rather than replacing everything.

```text
reference + small operation/API context
              |
           generator
              |
         code extractor
              |
  shared checker: rules -> simulation -> numeric/other gates
              |
 structured failure with category, location, and evidence
              |
 repair planner selects ONE change + relevant API snippet
              |
    compact prompt + latest code + short failed-action ledger
              |
       next attempt or honest stop
```

### Improvement A: structured failures and surgical repairs

Start with a small taxonomy, using examples from actual runs:

| Category | Example | Next action |
|---|---|---|
| `empty_answer` / `truncated` | No code or `finish_reason=length` | Shorten prompt and preserve output headroom |
| `syntax` / `extraction` | Prose or incomplete function | Request a complete function/module |
| `rule_violation` | Framework shortcut | Replace the named shortcut with an allowed primitive/loop |
| `api_misuse` | Nonexistent function or wrong signature | Supply the real signature and minimal usage |
| `buffer_layout` | PSUM/SBUF/HBM placement error | Repair the named buffer allocation/copy path |
| `tile_bounds` | Partition >128 or out-of-bounds | Derive bounds from shape; handle partial tile |
| `accumulation` | Wrong K-loop lifetime | Keep the accumulator across the contraction loop |
| `numerical_stability` | Overflow/cancellation | Apply the operation-specific stable formulation |
| `input_mutation` | Writes into caller-owned input | Allocate and return a separate output |
| `traffic_excess` | Bytes above floor threshold | Remove identified redundant loads/stores |
| `environment` | SDK absent or endpoint unavailable | Stop/recover infrastructure; do not ask model to rewrite arithmetic |

Example repair instruction:

> “The destination tile has 256 partitions; the limit is 128. Split that dimension into chunks bounded by the actual input shape and update the corresponding destination slices. Preserve the arithmetic.”

Use the taxonomy to route a small relevant snippet into the prompt. Do not paste the whole documentation set after every error. The existing `enrich()` already does some of this: your improvement should address a measured missing case, not merely rename the function.

### Improvement B: visible token budgeting

For an 8192-token context, a proposed starting allocation is roughly 4500 input tokens, 3000 output tokens, and 692 tokens of safety margin. **This is a tuning starting point, not a measured optimum.** Reasoning tokens can consume output budget too.

Prioritize: task/signature, current code, one actionable failure, the needed API excerpt, then a short ledger. Drop old full kernels and long traces first. Store everything in the log even when it is omitted from the next prompt.

Use the model's tokenizer where accessible. Otherwise label estimates as estimates, record server-reported usage as authoritative totals when supplied, and keep a conservative margin. Component estimates may not sum exactly to server totals because of chat formatting/tokenization.

A useful demo chart is **input tokens spent on task/docs/code/errors/ledger by attempt**, alongside pass rate or best reward. It directly addresses the challenge's context-management emphasis.

### Improvement C: verification status and stopping

Expose statuses such as:

```text
INVALID_RULES
FAILED_EXECUTION
FAILED_NUMERICS
FAILED_TRAFFIC
VERIFIED_ON_SIMULATOR_CASES
VERIFIED_ON_DEVICE_CASES
UNVERIFIED_ENVIRONMENT
STOPPED_REPEATING_FAILURE
```

Track repeated failure categories and code/prompt hashes, not only exact error strings. A changing index or number can make the same structural failure look new. A stop should preserve the best candidate and explain why it remains unverified.

If you report a numeric confidence, state what event it predicts and evaluate its calibration on held-out cases. With a tiny dataset, a measured false-verification rate and explicit coverage are more defensible than “99% confident.”

## 10. Setup and first commands

### Laptop: access your assigned seat

Install the AWS CLI and kubectl if necessary. Obtain the temporary credentials and **your own seat number** from the organizers. The credentials must be available in each terminal you use; never save them in this guide or commit them.

Commands documented for this cluster:

```bash
aws --version
kubectl version --client
aws eks update-kubeconfig --name hack-hyd --region ap-south-2
kubectl get pod seat-YOUR_NUMBER
kubectl exec -it seat-YOUR_NUMBER -- bash
```

Replace `YOUR_NUMBER`. Wait until the prompt is inside the pod before running the next commands. Only use your assigned seat. `Running 1/1` means the container is up; because its command is `sleep infinity`, it does **not** mean the model is already serving.

### Pod terminal 1: start the local model

```bash
cd /workspace
neuron-ls
MAX_MODEL_LEN=8192 ./serve.sh
```

Wait for `READY`. Startup is documented as about four minutes under favorable conditions; shared downloads/compilation can take longer.

The seat manifest sets `MAX_MODEL_LEN=8192`, but **the script's standalone default is 4096**. Explicitly setting 8192 avoids this mismatch for a fresh server. If a server is already running, `serve.sh` waits for it; it does not restart it with your new settings. Verify the existing configuration before choosing the agent's `--context` value.

The model survives the launching shell exiting (`nohup setsid` in `serve.sh`). Old `STATE.md` instructions involving `docker exec -it vllm bash` describe an earlier setup; current seats already run the model-serving container.

### Pod terminal 2: prove the checker, then establish baseline

```bash
kubectl exec -it seat-YOUR_NUMBER -- bash
```

Once inside:

```bash
cd /workspace
git config --global --add safe.directory /workspace
cd projects/20-kernel-agent
python nkibench.py --selftest
python nkibench.py --list
python nkibench.py --level 4 --check reference_level4.py
python agent.py --level 4 --rounds 8 --samples 4 --context 8192 --log baseline-l4.jsonl
```

For the repeated baseline, run in the background so a disconnected `kubectl exec` does not terminate it:

```bash
nohup python -u agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5 --log baseline-qwen.jsonl > run-baseline-qwen.log 2>&1 < /dev/null &
tail -f run-baseline-qwen.log
```

`Ctrl+C` stops `tail`, not the background run. Reconnect to the same pod and inspect that log. `pgrep -af agent.py` shows running agents. Use distinct filenames for baseline and improved experiments; JSONL output appends by default.

The pod sets the local endpoint to `http://localhost:8000/v1` and model to `Qwen/Qwen3-8B`. If using a different environment, pass `--base` and `--model` explicitly. `localhost` on your laptop is not the model server inside the pod.

### Shared gpt-oss comparison

Use the organizer-provided root URL and explicitly override the local model settings:

```bash
export GPTOSS_BASE_URL="https://ORGANIZER_PROVIDED_HOST"
python -u agent.py --all --base "$GPTOSS_BASE_URL" --path /agg/v1 --model gpt-oss-20b --rounds 8 --samples 1 --context 8192 --terse 1 --repeat 3 --log comparison-gptoss.jsonl
```

For a long run, use the same `nohup ... > run-comparison.log 2>&1 < /dev/null &` pattern.

Passing `--base` matters because `KERNEL_AGENT_BASE_URL` in a seat takes precedence over `GPTOSS_BASE_URL`. Do not append `/agg/v1` both to the base and through `--path`.

### Laptop-only NumPy track

From the repo root, install dependencies in an existing or new virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd projects/20-kernel-agent
python kernelbench.py --selftest
python kernelbench.py --list
python kernelbench.py --level 5 --show
python kernelbench.py --level 5 --check my_kernel.py --all-failures
```

`my_kernel.py` is a candidate you create; it is not shipped.

Manual shared-model driver, from this directory:

```bash
export GPTOSS_BASE_URL="https://ORGANIZER_PROVIDED_HOST"
python try_level.py 5
python try_level.py 5 "Replace np.max and np.sum with explicit column loops." --repair
```

**Use positional level `5`, not `--level 5`.** The README's flag example is incompatible with this helper's argument parsing. `try_level.py` always appends `/agg/v1/chat/completions`, so its environment variable must contain the root URL. It stores the last code and verdict in `/tmp/level5.py` and `/tmp/level5.fail`; preserve them elsewhere if needed. It is a manual helper, not the full NKI agent.

## 11. Model and hardware facts that affect your approach

The following behavior is documented/measured in this repo; recheck it on today's deployment before making new performance claims.

| Setting/property | Practical decision |
|---|---|
| Qwen3-8B uses sampling in this agent | Four candidates per round can be different |
| Shared gpt-oss is configured for greedy decoding | Use one sample; change prompts rather than retrying identical requests |
| Qwen thinking mode performed very poorly | Leave `--think` off for the baseline |
| Long constraint prompts caused reasoning-only gpt-oss output | Start terse; repair one named issue |
| Context is limited to 8192 in the documented seats/shared deployment | Reserve space for completion and inspect `finish_reason` |
| Shared native `tools=` did not work in repo probes | Use explicit text/JSON tool protocols if needed |
| Shared endpoint has `/agg/v1` and `/disagg/v1` | Pin one for comparisons; clients support both |
| Shared endpoint is network restricted | A TCP timeout can require organizer network access help |
| Shared endpoint's documented TLS mismatch | Starter clients disable verification for that endpoint |
| Shared capacity was around four requests/sec in historical tests | Avoid unnecessary fan-out/load tests during shared use |

Do not confuse **gptoss/chat.py `--think`**, which displays reasoning, with the kernel agent's `--think`, which requests Qwen thinking mode.

The workshop configuration uses TP=2 and documents two free logical cores. Both current checkers run on CPU. Adding real device timing requires ensuring core availability; the repo suggests `NEURON_RT_VISIBLE_CORES=0,1` but explicitly says this arrangement was not exercised there. Verify actual allocation on your seat before relying on it.

**Performance claims:** the current harness counts simulated DMA traffic and computes arithmetic intensity. It does not measure device latency. Lower HBM bytes can be a valid measured outcome; it is not an observed latency speedup.

For matmul:

```text
FLOPs = 2*M*K*N
minimum bytes = input lhsT bytes + input rhs bytes + output bytes
traffic multiplier = measured bytes / minimum bytes
arithmetic intensity = FLOPs / measured bytes
```

The repo's 222 FLOPs/byte ridge is a documented **bfloat16 NeuronCore-v2 reference figure**. Current test inputs are float32 and the event hardware is Trainium2, so treat this as the harness's illustrative model, not a verified dtype/hardware-specific peak for your current device. Report actual bytes/dtype and avoid deriving a claimed hardware utilization percentage from that comparison.

All stock matmul test shapes have limited ideal arithmetic intensity. Compare your kernel with the **shape's byte floor**, not only the ridge; some shapes cannot become compute-bound regardless of reuse.

## 12. A practical plan for the remaining day

Adjust this to your actual remaining time. Preserve the measurement/demo budget even if implementation takes longer.

| Time from start | Work | Exit criterion |
|---|---|---|
| 0–30 min | Get seat/model ready; run self-tests/reference checks | Known-good environment or deliberate NumPy fallback |
| 30–60 min | Run baseline; read saved failures; choose one target | Exact baseline command, logs, and failure category |
| 60–150 min | Shared checker gates, structured failure, one targeted repair route, essential logging | First measurable recovery with trace |
| 150–210 min | Repeat A/B tests and held-out evaluation | Results table with failures retained |
| 210–270 min | One extension only if core result is stable | Stronger coverage, token selection, or second level |
| Last 45–60 min | Freeze code, package artifacts, rehearse | Reproducible submission and a working demo |

If you have only three hours, prioritize: one target, reliable logging, one repair improvement, repeated comparisons, and the demo. Skip attention, RL, and device profiling.

Suggested team responsibilities:

- **Agent/repair owner:** controller and context selection.
- **Checker owner:** references, planted bugs, edge cases, shared gating.
- **Evaluation owner:** experiment IDs, repeat runs, taxonomy, charts.
- **Demo/write-up owner:** reproduction steps and clear presentation.

For three people, combine evaluation and presentation. Teammates can use separate assigned seats to run independent experiments, but record which configuration and hardware produced each result.

## 13. Evaluation protocol: how to prove your improvement

1. Freeze and record the baseline commit/configuration before editing.
2. Choose one independent variable first: for example, generic feedback versus structured surgical repair.
3. Hold model, endpoint, temperature, sample count, round budget, context, checker, and development cases constant.
4. Run at least five local-Qwen trials if the time budget permits. Report the denominator even if fewer fit. A round with four candidates is four samples, not four independent complete agent runs.
5. Record first-attempt performance as the one-shot control. If claiming iteration adds value beyond extra sampling, also compare with an equal-request-budget generation-only control.
6. Evaluate final selected kernels on separate shapes/values/seeds that were not used to tune the repair prompts.
7. Keep reference kernels and held-out answers out of the generation context if you are claiming the agent discovered the implementation. Reference code can validate the harness; disclose any examples supplied to the model.
8. Count every failed, empty, truncated, and infrastructure-interrupted attempt. Separate infrastructure failures from correctness failures without silently deleting either.
9. Report both success and cost. Distinguish “best candidate seen” from “last candidate returned.”

Suggested results table:

| Variant | Level/track | Solved runs / total | Reward min/mean/max | Attempts to verified | Input/output tokens | Time | Held-out cases passed |
|---|---|---|---|---|---|---|---|
| Baseline | NKI 4 | fill from runs | fill | fill | fill | fill | fill |
| Improved | NKI 4 | fill from runs | fill | fill | fill | fill | fill |

Also report **false verification**: how many outputs labeled verified by your agent fail the independent evaluation. Zero observed false verifications on a small suite is evidence about that suite, not proof of universal correctness.

Five runs can reveal variability but are weak evidence for small gains. Treat a clear recovery from a reproducible structural failure differently from a tiny rate change on a noisy level, and state uncertainty.

### Minimum attempt log additions

```json
{
  "run_id": "improved-qwen-l4-r03",
  "variant": "structured-repair",
  "commit": "actual-commit",
  "track": "nki",
  "level": 4,
  "round": 2,
  "sample_id": 0,
  "model": "Qwen/Qwen3-8B",
  "context_limit": 8192,
  "prompt": "exact prompt sent",
  "response": "raw model response",
  "code": "extracted source",
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 0, "completion_tokens": 0},
  "latency_seconds": 0,
  "failure_category": "tile_bounds",
  "feedback": "specific evidence and next action",
  "reward": 0.3,
  "verification_status": "FAILED_EXECUTION"
}
```

The zeros above are schema placeholders. Use measured values or `null` with an explanation when unavailable; never manufacture measurements. Add per-case outcomes, traffic data, configuration, and token-component estimates as needed. Avoid credentials and full sensitive endpoint URLs in shareable records.

## 14. Submission and demo package

Proposed folder layout; **these artifacts do not exist yet**:

```text
submission/
  REPRO.md                  one-page instructions and result summary
  METHOD.md                 checker decisions, tolerances, repair/context design
  FAILURE_TAXONOMY.md        categories, counts, representative examples
  eval-cases.json            development/held-out cases and seeds
  results.csv                aggregate results
  attempts/                  complete JSONL logs
  kernels/                   final generated candidates
  figures/                   success/cost and token-budget charts
```

`attempts.jsonl`, `runs/`, `run*.log`, and several held-out filenames are gitignored. **Explicitly package the authorized submission artifacts** or choose intentional tracked paths. Do not assume a git push includes the evidence. Do not force-add credentials or organizer-only material.

### A strong 3–5 minute demo

1. State the problem: small models produce plausible illegal or incorrect kernels.
2. Show one baseline failure and the checker's evidence.
3. Show the exact targeted repair and why that context was selected.
4. Show the next candidate passing the same checks, then the separate evaluation result.
5. Show the repeated-run comparison and cost, plus one remaining failure.
6. State the verification boundary: CPU simulation, device execution, and timing are separate claims.

Have a recorded trace ready in case the endpoint is slow. Label it as a prior run; distinguish it from any live execution. Do not make the whole presentation depend on a shared endpoint responding in time.

Suggested opening sentence:

> “Our contribution is a failure-to-repair controller: it selects the specific API and boundary information needed for the next attempt, and only reports success after all checker gates pass.”

Use actual numbers after you have them. If the change does not improve the solve rate, show what failed and what your analysis established rather than inventing a win.

## 15. Traps to avoid

- Spending the day on a polished chat UI without a checker/evaluation contribution.
- Reporting a starter result or copied tutorial kernel as an autonomous new solution.
- Assuming the larger model will outperform Qwen on every level.
- Repeating the same greedy prompt or sending duplicate greedy samples.
- Turning on thinking mode without a controlled reason to test it.
- Pasting every constraint/API document into every generation prompt.
- Reintroducing the reverted global chunked-copy example without measuring its regression risk.
- Treating a runtime exception as proof the code computed wrong numbers; read `parts` and the error.
- Interpreting partial reward as verified correctness or calibrated confidence.
- Trusting the NumPy static scan to enforce all intended rules.
- Treating the NKI CLI and agent checks as equivalent before resolving their differences.
- Claiming speedup from simulated bytes or comparing float32 measurements against a bf16 peak as if matched.
- Starting with attention, RL training, or hardware profiling before a smaller loop works.
- Reporting mock/offline reference replay as a model result.
- Running the old cluster-job deployment instructions when your participant role only permits seat access/logs.
- Losing work when a pod is replaced: `/workspace` uses `emptyDir`, not persistent storage. Back up code and evidence to your own authorized location.

## 16. What was actually validated during this analysis

| Check | Result | Scope |
|---|---|---|
| `python3 kernelbench.py --selftest` | Passed | Correct NumPy example, planted edge bug, and selected banned calls |
| `python3 nkibench.py --selftest` | Passed CPU portions | References/static checks/byte sizing/roofline/mismatch descriptions; explicitly skipped NKI simulation |
| `python3 level0_heatrod.py --selftest` | Passed | Exact generated answers and planted physics failures |
| `python3 level1_heatrod.py --selftest` | Passed | Insulated-boundary cases and parabola truncation tolerance |
| `tool_calc.py` documented integral | Returned `32/pi**3` (~1.03205) | Calculator smoke check; this helper has no `--selftest` option |
| NumPy shortcut diagnostics | Confirmed incomplete scan | `@` matmul and `.T.copy()` examples produced no static violations |
| Stubbed correct attention result through NKI agent | Reproduced `KeyError: 'M'` | Controller integration only; no real NKI execution |
| Model/Trainium/device latency | Not run | Requires workshop access and SDK/device environment |

No application source was modified as part of creating this guide. A passing self-test is a useful baseline, not exhaustive validation; the checker gaps above remain open.

## 17. Resume instructions for us or a future coding assistant

Use this prompt:

> “Read `HACKATHON_GUIDE.md` first. Continue the kernel-agent hackathon project from that context. Inspect only the files needed for the next change, preserve the baseline, and record which findings have been fixed. Do not claim simulator traffic is device latency.”

Then establish only the live facts this snapshot cannot know:

- Your assigned seat, remaining time, and confirmed event rubric.
- Whether the model is serving and at what context length.
- Which track you selected and what the baseline actually measured today.
- Which changes and runs have happened since commit `8f1ca41`.

**Recommended next action:** get the NKI reference check and a level-4 baseline running, then implement shared verification gates, complete attempt metadata, and one repair improvement targeted at the observed failure. If seat/SDK access is unavailable, switch deliberately to `kernelbench.py` and strengthen its legality checks before automating its repair loop.

After each meaningful implementation or evaluation, append a short dated update here with the commit, exact command, result files, measured result, and unresolved issue. That keeps this guide useful without another full-repo analysis.
