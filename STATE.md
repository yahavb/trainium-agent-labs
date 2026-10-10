# Where this stands, and how to resume

Written after losing an instance mid-session. Everything below is pushed; nothing lives only on a host.
Repo: https://github.com/yahavb/trainium-agent-labs (private)

---

## What this repo is for

A one-day workshop where each team gets a `trn2.3xlarge` and builds an **agent loop**: a model attempts
something, a **checker** grades it, and the grade plus the *reason* feed the next attempt.

The thing being taught is not the model. It is that **the checker decides whether the loop works.** A
checker that says "wrong, off by 341 percent" is true and useless; one that says "the term sin(pi\*x)
should not be there at all" names the change. Across both projects the fix was a better error message
rather than a better model **eight separate times**, each one measured.

---

## Resume in four commands

On a fresh `trn2.3xlarge`:

```bash
neuron-ls                                               # expect 1 device, 4 cores, 96 GB
git clone https://github.com/yahavb/trainium-agent-labs.git && cd trainium-agent-labs
docker --version || ./install-docker.sh
MAX_MODEL_LEN=8192 ./serve.sh                           # ~4 min: pulls image, downloads 16 GB, compiles
```

Then, in a second shell:

```bash
docker exec -it vllm bash                               # WAIT for the prompt before typing
git config --global --add safe.directory /workspace
export KERNEL_AGENT_BASE_URL=http://localhost:8000/v1
export KERNEL_AGENT_MODEL=Qwen/Qwen3-8B
cd /workspace/projects/02-kernel-agent
python agent.py --all --rounds 8 --samples 4 --context 8192
```

`README.md` section 1 has the same thing with every step explained.

---

## Project 1 — the heat-rod agent. **SOLVED, 6 of 6.**

`projects/01-heat-rod-pde/`. A small model solves heat-equation problems; SymPy checks the physics by
plugging the answer back in. Needs no Neuron cores at all.

Reward out of 1.0: 0.4 the equation holds, 0.2 each boundary, 0.2 the starting shape. Partial credit is
the point. Every attempt is logged as prompt/answer/reward — the file an RL trainer would consume, which
is the obvious extension.

**Levels.** 0.1–0.3 are both ends at zero with sine starts: solved on round 0. 1.1–1.2 add an insulated
end, where the allowed frequencies halve. 1.3 is a parabola start, where no finite answer exists and the
checker accepts a truncation within 0.5% — measured to need three terms.

**The four findings, in order, each one changing the code:**

1. **It answered in notation, not numbers** — `X(x)T(t)`, then a LaTeX sum over undetermined
   coefficients. A *worked example* of an acceptable answer fixed it outright. Prohibitions did not.
2. **A true number that says nothing.** Told only "off by 341 percent", it sat at 0.8 for four rounds,
   keeping the even terms that must vanish. The checker now projects onto the problem's own basis and
   names which term is wrong.
3. **Revealing the target made it stop thinking.** With the correct coefficients printed, it copied
   `1.032` and `0.03822` out of the message and derived nothing. Feedback is now **directional only**;
   `REVEAL_COEFFICIENTS = True` restores numbers for teaching.
4. **It cannot do the integral.** Given direction alone it guessed 1.6, −0.8, 0.4, −0.2, 0.1 and
   repeated itself. So it got **a calculator it aims itself** (`tool_calc.py`): it writes
   `COMPUTE: <expr>`, SymPy evaluates, the value comes back. The model chooses *which* integral — that is
   the reasoning — and SymPy does the arithmetic. Coefficients then came out exactly right.

**Then the failure moved rather than vanished:** with coefficients correct it paired `exp(-2*pi**2*t)`
with `sin(pi*x/2)`, mismatching the frequency. So the checker learned to measure each term's decay rate
separately. Level 1.3 now solves in two rounds, while `--no-tools` stalls at 0.8 — that comparison is the
project's headline.

**Worth telling participants:** the reward *fell* 0.8 → 0.6 at the moment understanding improved, because
the equation carries 0.4 and the starting shape 0.2. A single score can move the wrong way while the work
gets better.

---

## Project 2 — the kernel agent. **RUNS, NOT SOLVED.** Best 0.62 of 1.0.

`projects/02-kernel-agent/`. An agent writes NKI kernels; the harness checks them. Three levels of
checker, by cost: **static rules** (milliseconds), **`nki.simulate` on the CPU** (seconds), and
**on-device timing + profile** — the last **not built**.

Reward: 0.1 parses, 0.2 rules clean, 0.2 runs, 0.5 correct on every shape, prorated.

### Status per level — 5 runs, local Qwen3-8B, 8 rounds x 4 samples, context 8192

```
level 1: solved 0/5   all = [0.30, 0.30, 0.30, 0.30, 0.30]
level 2: solved 4/5   all = [1.00, 1.00, 1.00, 0.50, 1.00]   <- current baseline
level 3: solved 0/5   all = [0.30, 0.30, 0.30, 0.30, 0.30]
level 4: solved 0/5   all = [0.62, 0.62, 0.62, 0.62, 0.62]
```

An earlier 5-run measurement had level 2 at 2/5. The difference is not attributable: 2/5 vs 4/5 at n=5 is
within what this level's variance produces, and several feedback messages changed in between. **Treat 4/5
as the baseline and do not claim the improvement.**

**The important part is which numbers move.** Level 2 varies wildly — 0.30 to 1.00, solving 2 times in 5.
Levels 1, 3 and 4 are *identical* across all five runs, which is 20 samples each with zero spread.

So there are **two different problems**, and they need opposite treatment:

* **Level 2 is luck-limited.** Report a rate, not a verdict; it solves about 2 in 5. Single-run
  comparisons here are meaningless.
* **Levels 1, 3 and 4 are capability-limited.** Zero variance means more samples and more rounds buy
  nothing, and it also means **changes to these levels ARE cleanly attributable** — if a fix moves 0.30,
  that was the fix. An earlier claim in this file that variance swamps all changes was only true of
  level 2.

Each wall is one NKI idiom the model does not have, and it is the same idiom three times — **tiling**:

| level | the wall, every run | best |
|---|---|---|
| 1 | `SBUF and PSUM tensors must have at least 2 dimensions` — it builds a 1-D tile | 0.30 |
| 3 | `cannot reshape array of size 32768 into shape (1,64)` — it reshapes instead of slicing | 0.30 |
| 4 | `dma_copy dst partition dimension 256 exceeds maximum 128` — it allocates one tile for the whole tensor | 0.62, 1 of 4 shapes |

Level 4 is the closest: it passes the single-tile shape every time and fails every shape needing real
tiling.

**A chunked-copy example was tried in the prompt and REVERTED.** It targeted exactly these walls and made
everything worse — level 2 went from 2 of 5 solved to 0 of 5, level 4 from 0.62 to 0.30, measured over five
runs each way. Level 2's winning kernel never chunked; the example pushed it toward `dma_transpose`, which
it got wrong five ways. **Do not re-add it without measuring.** Without `--repeat` it would have shipped.

### gpt-oss-20b, 3 runs, --samples 1 --terse 1 --context 8192

```
level 1: solved 0/3   all = [0.30, 0.30, 0.30]
level 2: solved 3/3   all = [1.00, 1.00, 1.00]
level 3: solved 0/3   all = [0.30, 0.30, 0.30]
level 4: solved 0/3   all = [0.30, 0.30, 0.30]
```

**The three runs are byte-identical** -- same errors in the same order, same per-round timings to 0.1 s,
same kernel text. That is greedy decoding, and it is the cleanest demonstration in the repo of why the two
models need different loop designs.

**The 8B model beats the 20B model on level 4: 0.62 vs 0.30.** gpt-oss returned NO CODE in four of six
rounds there, after ~10,000 characters of hidden reasoning each, at 21 s per empty round. Capacity spent
reasoning is capacity not spent answering. Level 2 goes the other way: gpt-oss 3/3 against Qwen3 4/5 --
same ceiling, less spread.

Known gap this exposed: terseness escalates only when the *latest* attempt produced no code, so once a
round succeeds the prompt grows again and later rounds go empty at terseness 2 with a 2549-character repair
prompt. Escalating on every empty answer regardless would likely recover those rounds. **Untried.**

### What matters about the measurement

* **Arithmetic intensity is HBM traffic.** Flops are fixed by the problem, so intensity is just
  flops/bytes. The byte floor — each input read once, output written once — therefore sets a **ceiling**
  no kernel can beat.
* **Every matmul test shape is below the ridge**, with ceilings of 28–73 against 222 Flops/Byte. So the
  available win here is removing redundant traffic, **not** reaching compute-bound. The old message told
  the agent to find reuse that does not exist; it now separates *your kernel's* waste from *the shape's*
  limit. A square bf16 matmul needs n ≈ 667 before the ceiling clears the ridge.
* **Redundancy is invisible on small shapes.** The reference tiled kernel is at the floor (1.00×) on a
  single-tile shape and 2.00× on the largest. A kernel can look optimal on a toy and waste half its
  bandwidth on a real one.

### The two findings that generalise beyond this project

1. **Thinking mode destroys both models.** Measured on Qwen3, only `--think` changed: **~8 s → 446 s per
   round, and 0.30–0.62 → 0.00**, every sample truncated at ~9,900 characters with the budget spent
   reasoning. On gpt-oss, a 1866-character prompt gave 13,245 characters of reasoning and no answer while
   581 characters gave working code. **A bigger budget does not fix either.** Shorter prompt, less
   reasoning.
2. **A greedy model plus an unchanged prompt repeats forever.** Nineteen identical failures in a row, by
   construction. Hence the ledger of failed attempts, and stopping a level after four occurrences of one
   failure — including when it *alternates* between two, which a consecutive-streak check misses.

### Harness bugs the runs exposed, all fixed

* Byte counting was **2× low** (assumed 2 bytes on float32 inputs), which doubled every intensity — in the
  dangerous direction, making a starved kernel look compute-bound.
* `nki.simulate_kernel` does not exist; the API is `nki.simulate(kernel)(*args)`. Detects either now.
* A kernel **wrote into its input** and passed, because the reference happened to run first. Inputs are
  snapshotted and compared now.
* A "solved" level 2 transposed **one element at a time** — the exact suboptimality that level teaches —
  and the harness said nothing, because transfers were only counted for levels 3+.
* The repair prompt rebuilt from the **best** attempt, so one bad round froze it permanently. It repairs
  the latest now.
* **My own feedback caused failures twice:** "chunks of 128" with no clamp made it read past a 32-row
  tensor, and a buffer-placement message answered an error whose real fix was a different *function*
  (`dma_copy`, not `tensor_copy`, to reach HBM).

---

## Environment facts, hard-won

* **The AMI's pre-installed vLLM cannot serve Qwen3-8B.** Newest venv is vLLM-Neuron **0.21**, whose
  registry has only `LlamaForCausalLM`, `GptOssForCausalLM`, `Eagle3LlamaForCausalLM` and
  `Qwen3VLForConditionalGeneration` — the last being the **vision** model. You get
  `AttributeError: Qwen3ForCausalLM has no attribute from_configs`, which looks like a broken install and
  is a model-support gap. `serve.sh` uses the **0.24 container** instead.
* **`NEURON_SKIP_EFA_AFFINITY=1` is required**, or workers abort with `No EFA device found`.
* **`--no-enable-prefix-caching`** is required: prefix caching wants a segmented-prefill size ≥ 512.
* **`--num-gpu-blocks-override`** is required; `serve.sh` computes it from the context length.
* **A core cannot be shared, but the chip has four.** vLLM at TP 2 occupies NC 2–3; NC 0–1 stay idle at
  0 B. So nothing has to be remote, and both projects need **zero** cores today.
* **`docker exec -it vllm bash` takes a second.** Anything typed before the prompt goes to the host and
  is lost. This caught us three times.
* **A restarted container is a fresh shell** — re-export `KERNEL_AGENT_BASE_URL` and
  `KERNEL_AGENT_MODEL`.
* **`git config --global --add safe.directory /workspace`** before any git command inside the container.

---

## Next, in priority order

1. ~~`TP=4`~~ **TRIED, no benefit.** It serves fine on all four cores at 8192, but round times are 8–20 s,
   the same as TP 2. The loop is bound by how long the model takes to generate, not by compute, so the
   whole chip buys nothing. **Stay at TP 2** and keep NC 0–1 free for on-device timing later.
2. **Reference kernels for levels 5–7.** The traffic bars exist but nothing demonstrates passing them, so
   the optimization half of the ladder is unproven.
3. **Layers 2 and 3 of the checker** — real latency via `nki.baremetal`, and a profile. Until then
   **latency cannot be measured at all** and every number is throughput reasoning from the simulator.
   This is also where `NEURON_RT_VISIBLE_CORES=0,1` gets exercised.
4. **Try project 2 against gpt-oss-20b** with `--samples 1 --context 8192 --terse 1` and compare which
   model clears which level. Greedy sampling there makes extra samples pointless.
5. **Count non-`dma_copy` writes**, so an element-at-a-time kernel is flagged issue-bound instead of
   quietly passing.

## Still open for the event itself

* Student access instructions — how a team reaches its instance on the day. The README now
  starts at `neuron-ls`, assuming access already exists, so this lives only here.
* The repo is **private**; students need it public or need invites.
* `RUN-OF-DAY.md` and `LOCAL-NOTES.md` are gitignored and hold the judging answers and the endpoint URL.
