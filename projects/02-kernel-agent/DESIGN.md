# Kernel agent v2: six roles

Status: **built, being tuned**, branch `31p`. The roles are in `agents2/`, run with `agent2.py`; this
document started as the proposal (2026-10-10 ~20:40 UTC) and the sections below "Why change" are still
that plan. Numbers marked *measured* come from runs on the seat pods today.

The roles: **manager**, **planner**, **retriever**, **coder**, **debugger**, **reviewer**.

## Implementation status (2026-10-10 ~22:40 UTC)

`agent.py` is unchanged and stays the baseline.

| piece | status |
|---|---|
| manager, planner, retriever, coder (write/apply/improve), debugger, reviewer | built |
| checks: parse, lint, `check_rules`, simulate, compare; process pool with timeout | built |
| **the agents pull documentation themselves** (`LOOKUP:`, `agents2/lookup.py`) | built |
| retriever sources: checked cards (`nki_cheatsheet.md`, then `agents2/cards.md`), introspection, `third_party/` prose | built |
| debugger examples from `nki_fix_examples.md`, matched by error and level | built |
| prompt packer with the real Qwen3 tokenizer; per-role caps | built |
| `events.jsonl` (every call from every role, with prompt and reply), `attempts.jsonl` in agent.py's schema | built |
| zero-byte traffic bypass closed (levels 5–7) | built, in `agents2/checks.py` |
| thread ends judged by stage reached as well as reward | built, `manager.progress()` |
| `--classify`, `--dry-run`, `--no-lookup`, `--no-skeleton`, `--no-docs-request`, `--index names`, `--cards introspect`, `--hint` | built |
| profiler layer 2 (bytes per operand), profiling as a tool, edit mode, LLM manager, device profiling | not built |

**Pull, not push.** A first prompt is the problem statement as `agent.py` gives it (operation, entry
point, NumPy reference, test shapes, the one-line hardware limits, the imports). Of `agent.py`'s API card
the coder's first prompt keeps only the organisers' complete example kernel (`copy_kernel`: output in
HBM, a tile in SBUF, `dma_copy` in and out). It computes nothing, so it gives no algorithm away, and
without it the last level-1 runs still called `nl.mean` straight on the HBM input (`--no-skeleton` leaves
it out). The card's list of functions is gone, and levels 1–2 get no matmul text.

The planner's and coder's first prompts also carry an *index* of names (the nisa and nl functions, tile
methods, the topic `rules`). Repairs and improvements carry no index: the change names what to use, and
its documentation comes with it. Any role may answer `LOOKUP: a, b, c` to get documentation before
answering:
- planner: 2 rounds;
- coder and debugger: 1 round each.

The lookup is offered as one of two ways to reply, inside the reply format, which is always the last
thing in the prompt. Before ~21:55 UTC it was a separate paragraph just above "Reply with ONE python
code block", two competing formats; run 1 of the pull design made only 4 lookups, so that run did not
really test pulling. After the last round the format says "answer now"; a lookup asked for anyway gets
its documentation and one more call, and a reply that is still a lookup counts as no answer (the
debugger falls back, the planner re-plans).

Offered that way, the lookup was still never taken: 0 lookups in 39 calls in the level-1 run of
`5bbecbd`. So since ~22:15 UTC **the planner's first call only asks which documentation it needs**
("Before you plan, choose the documentation you need… LOOKUP: …", 60 tokens out); it then plans with
that documentation, with one optional lookup left. Nothing is suggested, so the model still chooses what
to pull. `--no-docs-request` turns it off.

**Echo handling** (since ~22:15 UTC). In the same run, 12 of 26 coder calls returned the kernel
unchanged, ending 3 of 4 threads. Every one came after a change that described the failure rather than
naming an edit (nkibench's WRONG SHAPE and NON-FINITE messages, a generic "call it with its real
signature", a poor closest name), and the hot retry of the same change echoed 6 times out of 6. Now:
- nkibench's mismatch messages (wrong shape, NaNs, zeros, partial output, hardware hazard) go to the
  debugger's model, which names a line, instead of being passed on as the change; INPUT_MODIFIED,
  whose message is an edit, keeps its rule;
- the keyword rules name the edit for each of Python's messages (an argument given twice, missing
  arguments, too many positional ones);
- an unchanged kernel is not re-sent the same change: the debugger, told the change was tried, names a
  different one, once per failure (it was once per thread). A second echo for that failure ends the
  thread.

**Described index** (since ~22:40 UTC, from another session's patch). The index in the planner,
coder-write and debugger prompts is now ~18 hand-picked entries, each with one line on what it does
(`agents2/index.py`: about 430 tokens, against 784 for every public name). `--index names` restores
the old list, and the LOOKUP topic `all` returns it. Entries follow the level: matmul and
`nc_transpose` from level 3, `t.permute` not at level 2. Measured by that session on seat-198
(Qwen3-32B, planner only, LOOKUP off, 8 plans per arm):
- names that exist, in the right module: level 1 48% -> 100%, level 4 55% -> 91%;
- plans choosing `nisa.tensor_partition_reduce` (it reduces across partitions, the wrong axis for
  pooling): level 1 5/8 -> 1/8, level 4 4/8 -> 0/8.
Not yet measured with the 8B, or on solve rate. `fix_name()` keeps a real name written under the wrong
prefix in a plan (`nl.tensor_reduce` -> `nisa.tensor_reduce`, `nl.reshape` -> `t.reshape`) instead of
dropping it and re-planning, and never maps to a name withheld at the level.

The descriptions are pushed into every first prompt. At level 1 they describe `t.reshape`,
`t.permute`, `t.ap` and `nl.sum` / `nl.mean` "over free (trailing) axes", the pieces of both known
level-1 routes. That's milder than the cards pushed earlier (no examples, no recipe), and the names
list already named all of them, but it is a push: whether it's the default is the user's call.

What the planner pulled goes to the coder with the plan. A failed check pulls what the error names (the
function's card, a checked fix example), as checker feedback does. Withheld cards are enforced in
`Retriever.shown()` and `withhold.json` in `Retriever.allowed()`; the debugger's rules pass on no
withheld name and restate no withheld card (the transpose rule defers to the model at level 2).

**Open (the user's call):** level-2 withholding is inconsistent. Every level-2 test shape has at most
128 rows, so any one transpose call does the whole job, yet only `t.permute` and `nisa.dma_transpose` are
withheld there; `nisa.nc_transpose`'s card and `nl.transpose`'s docstring are served. Either withhold
API facts for every transpose route at level 2, or withhold patterns only (the K-loop, the strided
pooling view) and drop `withhold=2` from `agents2/cards.md`.

Measured on seat-35:
- 30/30 unit tests and `tests/test_index.py` 4/4, on the laptop and on seat-35 (~22:40 UTC); the checks
  below were re-run then too.
- Lint is clean on the reference kernels and the checked fragments.
- `--offline --all` scores 4/4.
- Our 4 cards are backed by 7 simulator checks, all holding. The cheat-sheet's 37 hold, and so do the 8
  fix examples, trimmed `nomatmul1` included.
- `--classify` over 428 recorded agent.py failures:
  - all 428 fall into an error kind, and a rule names the change for each. None of them is a
    mismatch (agent.py's recorded failures are all errors before correct values), so sending the
    mismatch kinds to the debugger's model (~22:15 UTC) left this at 428/428 (re-measured on seat-35);
  - lint flags 166 before the simulator, with 0 false positives;
  - 423 get a checked fix example.
- Level 1, one run each:

  | version | best |
  |---|---|
  | first agent2 | **0.50** (kernel ran, wrong values) |
  | cards pushed into every prompt | 0.30 |
  | pull design | 0.30 (4 lookups) |
  | `5bbecbd` (example kernel, lookup in the reply format) | 0.50 (wrong shape; 0 lookups, 12 echoes) |

  No level-1 run has solved it. The baseline is 0.30 on every run, a wall, so 0.50 is a real change, but
  the others are single runs.
- **Not measured yet:** levels 2–4 with agent2. The README's warning applies (a worked example once took
  level 2 from 4/5 to 0/5), so measure `--all` before claiming anything.

## Lessons from the READMEs, and where agent2 stands

| README lesson | agent2 |
|---|---|
| Constraints belong in the verifier, not the prompt | rules stay in the checks; prompts carry no rule lists |
| A verdict is not an instruction: send one named change | the debugger always names one change; rules write it for 428/428 recorded failures |
| The same error three rounds running means fix the message | new rules came from repeats in our logs (tile_size, HBM data, transpose, NkiTensor methods) |
| Give the model a tool, not a hint | `LOOKUP`: the model aims the retriever itself |
| Do not print the answer in your feedback | answer-like material is pulled, never pushed. `nomatmul1` no longer shows the `t.ap` view. `t.permute` and `dma_transpose` cards are withheld at level 2 |
| Thinking off; capacity spent reasoning is not spent answering | thinking off, hard output caps per role |
| Expect the failure to move rather than vanish | threads end on no *progress* (stage reached), not on no reward gain |
| One run is not a result; levels 1, 3, 4 are walls | report rates over `--repeat`; a wall broken once is a signal |
| A reasonable-looking prompt change made every level worse | measure all levels before claiming a gain (not yet done) |
| Giving computed numbers is a hint, not a tool | to do: profiling for levels 5–7 as a tool the model calls, not pushed ratios |

## Why change

`agent.py` is one loop: a single prompt writes a kernel, the checker grades it, and the checker's message goes
back for a repair. Its failures, from 168 level-1 attempts on seat-35 (*measured*):

| what goes wrong | evidence | role that fixes it |
|---|---|---|
| wrong algorithm from the start | 100% of attempts call `nc_matmul` for pooling with the old API card, 0% with the matmul text removed | planner, retriever |
| invented API | `nisa.multiply` in 75–100% of attempts; made-up kwargs such as `transpose_moving=` | retriever, debugger |
| feedback says what's wrong, not what to do | `nisa.multiply` gets 25 unrelated names, alphabetically | debugger |
| repairs echo the input kernel unchanged | the `--echo-break` lever exists because of this | manager |
| the 4 samples are the same kernel | identical in 72 of 83 rounds (top-p 0.95); and different text is still the same approach | planner, manager (one approach per thread) |
| nothing ever questions the algorithm | best score 0.30 after every repair, with 17 distinct kernels in 168 | debugger can say "approach is wrong" |
| speed is taken from byte counts that can be bypassed | zero counted bytes passes the traffic bar | reviewer |
| nothing records why a run stopped | open issue 2 in `AGENTS.md` | manager |

## Constraints that shape the design

- **8K context.** The seat-35 server runs `max_model_len` 8192, and that covers prompt plus answer.
  The design reads the real limit from `/v1/models` at startup, so a 16K or 32K server later only changes
  the budget table.
- **Output tokens are the cost; prompt tokens are nearly free.** Decode is ~99% of model time (*measured*:
  7.9 s of prefill against 1,299 s of decode). Per-token time (*measured*): ~71 ms with one request,
  ~79 ms each at 2, 184 ms each at 4. So every model role gets a hard output cap, and the cheap thing to
  spend is prompt context.
- **2 concurrent requests per server is the throughput peak** (*measured*: 25.4 tok/s at 2, against 22 at 4).
  The server takes 4 at most (`--max-num-seqs 4`), and anything past that queues.
- **Checks are CPU-side and sequential within a process.** `nki.simulate` takes 0.7–1.7 s, and the
  `dma_copy` byte counter patches a module, so it isn't thread-safe; run checks in separate processes.
  The pod has 11 CPUs, and the server uses ~5 while generating.
- **Lessons already written into the code** (`first_prompt`, `repair_prompt` docstrings):
  - a prompt with a list of prohibitions makes the model audit itself and return nothing, so rules
    live in the checker;
  - verbatim checker reports reproduce the same mistake, so feedback must name one change;
  - `--think` was ~55× slower per round and scored 0.
- **Docs:**
  - the installed nki 0.6.0 is the authority: signatures and docstrings for all 62 `nki.isa` and 108
    of 112 `nki.language` callables;
  - `third_party/neuron-agentic-development/` is written for nki 0.4.0, and has known wrong claims (its
    README);
  - `withhold.json` lists the files the agent must not see at each level.

## The shape of it

**Stateless roles, stateful manager.** Every model call is one fresh, single-turn prompt, built from the
manager's ledger. No chat history ever accumulates, so the 8K limit applies per call, never to the run as
a whole.

**Model only where judgement is needed.**
- The manager and retriever are code.
- The debugger and reviewer use rules first, and call the model only when no rule covers the case.
- The planner and coder always call the model.

```
 MANAGER (code): starts threads, ends threads, ends the level, owns the ledger, logs why it stopped

 one thread = one approach:

   PLANNER ─▶ RETRIEVER ─▶ CODER ─▶ CHECKS ─┬─ fail ─▶ DEBUGGER ─┬─ one named change ─▶ CODER ─▶ CHECKS …
                                            │                    └─ "approach is wrong" ─▶ MANAGER ends the
                                            │                                              thread, asks PLANNER
                                            └─ pass ─▶ REVIEWER ─┬─ accept ─▶ MANAGER ends the level
                                                                 └─ one improvement ─▶ CODER ─▶ CHECKS …
```

**CHECKS** is code, run in this order:
1. **compile:** the file parses, and lint passes (names and signatures);
2. **text scan:** `nkibench.check_rules`;
3. **run:** `nki.simulate`;
4. **correctness:** compare against the reference on every shape.

The manager records the stage each attempt reached.

| role | model or code | output cap (tokens) | when |
|---|---|---|---|
| manager | code (an LLM manager is a later experiment) | — | always |
| planner | model | 160 | start of each thread |
| retriever | code (model lookups are a later option) | — | whenever another role needs docs |
| coder | model | 900 (L1–3), 1,400 (L4+) | after a plan, a debugger change, or a reviewer improvement |
| debugger | rules first, model second | 100 | after a failed check |
| reviewer | code measures, rules first, model second | 100 | after a kernel passes every check |

### Manager (code)

- **Threads:**
  - Each thread follows one approach, from a plan to an accepted kernel or a dead end.
  - Run 2 threads per server: 2 requests at once is the throughput peak. While one thread is being
    checked (CPU), the other is generating, so the server stays busy.
  - Threads can run on different seats' servers, through config.
- **Ledger, per level:**
  - approaches: the plan line, the functions it calls, best score, error types seen, attempts used,
    status;
  - per thread, its attempts: sha, stage reached, error type and line, the change that was asked for;
  - docs already shown; the best kernel so far; tokens and seconds by role.
- **End a thread when:**
  - the same error type comes back at the same line after a fix;
  - the coder returns the kernel it was given twice. After the first time, it gets one retry at a
    higher temperature with "you returned the same code";
  - N repairs pass with no score gain (about 6 to start);
  - the debugger says the approach is wrong;
  - the reviewer ends it (two tries without a gain).
- **Replace an ended thread:** ask the planner for a new approach, passing every failed approach with
  its reason.
- **End the level when:**
  - the reviewer accepts a kernel;
  - or the budgets run out (about 4 approaches per level, plus a token budget).

  Always log the stop reason: `solved`, `exhausted`, `cycling`, `http_errors`, `empty_answers` or
  `budget`.
- **Route** each role to an endpoint, model, temperature and top-p, from config.
  - The default is everything on the local 8B.
  - Experiment: planner and debugger on the Qwen3-32B started on seat-198 at ~20:10 UTC. Check its
    health, speed and who owns that slot first.
  - The shared gpt-oss is a poor fit: greedy, and it needs ~2,500 tokens of hidden reasoning per answer.
- **Later experiment:** a model-driven manager against these rules, compared on solve rate and tokens.

### Planner (model)

Chooses *what algorithm* to write before any code exists. That's the step the current loop skips, and the
level-1 data says it's where things go wrong.

- **Input:**
  - the operation, the entry name and arguments, the NumPy reference and every test shape;
  - the hardware numbers (partition max 128, PSUM and SBUF sizes, from `nl.tile_size`);
  - the retriever's **API map**: every nki 0.6.0 name grouped by category, names only (target: under
    1,000 tokens, to measure);
  - the failed approaches so far, one line each.
- **Output**, line-based so an 8B model can follow it and a regex can parse it:
  ```
  APPROACH: strided access-pattern view of each pooling window, nl.sum over the window axes
  CALLS: nl.ndarray, nisa.dma_copy, nl.sum, nisa.tensor_scalar
  LAYOUT: channels on partitions (tiles of 128); H*W on the free axis
  STEPS: 1 load a 128-channel tile  2 view windows  3 sum + scale  4 store
  ```
- **Checks (code):** every name in `CALLS` must exist in the installed nki. Unknown names get close
  matches from the retriever, then one re-plan.
- **Diversity comes from approaches, not tokens.** Each thread gets its own approach (temperature 1.0,
  top-p 1.0). A plan whose `CALLS` set matches an approach already tried is dropped.

### Retriever (code): "pull NKI itself"

A lookup step, not a model, so it spends no output tokens. Sources, in order of trust:

1. **Installed nki 0.6.0, introspected** (`inspect.signature` and `__doc__`). Generated at startup inside
   the pod and cached under `runs/`. It's never committed, because the docstrings are Amazon's.
2. **Verified examples:** our small example kernels (`FRAGMENTS`, `--check-fragments`). Later, the
   agent's own solved kernels from lower levels (a skill library).
3. **`third_party/` docs** (nki 0.4.0), chunked by heading and indexed by the API names each chunk
   mentions.
   - Filtered by `withhold.json` for the current level, enforced in exactly one function, with a test.
   - A code snippet is only shown if it passes `nki.simulate` on 0.6.0.
   - Chunks that repeat the README's known-wrong claims get an annotation.
   - The AWS debugging skill's error references (`skills/neuron-nki-debugging/references/ncc-*.md`)
     seed the debugger's error types. They describe compiler errors, though, not simulator ones.

Who calls it:
- the planner, for the API map;
- the coder, for cards on exactly the plan's `CALLS`;
- the debugger, for the card on the function named in an error.

Interface: `lookup(names=[...], query=None, level=n, budget_tokens=k) -> list[Card]`. A card:

```
nisa.tensor_scalar(dst, data, op0, operand0, reverse0=False, op1=None, operand1=None, reverse1=False, ...)
  <first two sentences of the 0.6.0 docstring>
  e.g. (checked on nki 0.6.0): nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=0.25)
```

Later options:
- model-picked lookups for vague questions ("how do I reduce a strided window?");
- a `LOOKUP:` line the coder may write instead of code. It costs ~10 output tokens, and the prompt is
  re-sent with the answer.

### Coder (model)

One role, three modes, always a single fresh prompt:

| mode | input |
|---|---|
| write | the plan, the cards for its `CALLS`, a skeleton (imports, decorator, entry name and arguments, the output `nl.ndarray(..., buffer=nl.shared_hbm)` and `return`) |
| apply | the current kernel, the debugger's one named change, the card for the function involved |
| improve | the best correct kernel, the reviewer's one improvement |

- It replaces the static `API_CARD`, which is what pushed level 1 toward `nc_matmul`. `API_CARD` stays
  in `agent.py`, because the k8s jobs import it.
- **Lever, off by default:** the level's own hint, `nkibench.LEVELS[n]["notes"]`. For level 1: "nl.sum
  and nl.mean over a strided access-pattern view are the intended NKI route". It's the organisers'
  text, so report results with and without it.
- Sampling: top-p 1.0. Temperature 0.7 for apply and improve, 1.0 for write.
- Output: one code block, no prose. The output cap is by level.
- **Stretch lever, edit mode:** for apply and improve, reply with `REPLACE lines a-b WITH …` instead of
  the whole kernel. That's ~80 output tokens instead of ~350, about 4× faster. It falls back to a whole
  kernel when the edit doesn't apply.

### Debugger (rules first, model second)

Runs after any failed check. Its output is always one of two things:
- **one named change**, for the coder:
  ```
  CAUSE: nisa.multiply does not exist
  LINES: 14
  CHANGE: use nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=0.25)
  ```
- or **`APPROACH WRONG: <reason>`**, for the manager.

How it gets there:
1. **Classify** the first failure into an error type:
   - before the kernel runs: `PARSE`, `NAME`, `KWARG`, `RULES`;
   - shape and memory: `DMA_SHAPE` (element-count mismatch), `PARTITION` (>128), `TILE_RANK`,
     `MEMSPACE` (PSUM/SBUF/HBM placement), `REDUCE_AXIS` (reduce axis must be last);
   - `TIMEOUT`;
   - wrong results: `WRONG_VALUES`, subdivided by `nkibench.describe_mismatch`: zeros, partial
     coverage, ragged edge, core arithmetic.

   Target: at least 90% of recorded errors land in a type.
2. **Rule path, no model call.** For `NAME`, `KWARG`, `PARTITION`, `DMA_SHAPE` and the like, a template
   fills in the change from the real signature, the retriever's close matches and the offending line.
3. **Model path** (≤100 tokens out), for `WRONG_VALUES` and anything unclassified.
   - Input: the plan line, the kernel with line numbers, the error or mismatch description (for wrong
     values, a 4×4 corner of got and expected on the smallest shape), and the card for the function
     involved.
   - It may answer `APPROACH WRONG` when the plan can't produce the result. That's the escape route the
     level-1 runs never had.
4. **Never** a list of 25 names; never the raw checker report on its own.

A repair is debugger plus coder: ~80 + ~350 tokens on the model path (~34 s), or ~350 (~28 s) when a
rule writes the change.

### Reviewer (code measures, rules first, model second)

Only sees kernels that pass every check on every shape. Every number it reports is labelled `simulated`
or `device`.

- **What "done" means depends on the level:**

  | level | accept when |
  |---|---|
  | 1–4, 8 | correct on every shape. Accept at once, no model call. The ladder has no speed bar there. |
  | 5–7 | the HBM traffic is under the level's bar (`max_waste`: 1.6×, 1.25×, 1.05× of the byte floor) |

  Optional experiment: keep improving past the bar, e.g. level 2's transfer count.
- **Measures** (code, in the simulator):
  - Layer 1, fix what exists:
    - zero counted bytes is a failure, not a pass;
    - load the kernel *inside* the counting patch, so aliases are counted too;
    - replace the roofline text that's known to mislead (`explain_roofline`, and `reuse_report`'s PSUM
      advice).
  - Layer 2, new:
    - bytes per operand, by matching each `dma_copy` source and destination to the argument tensors,
      which gives a re-read factor per operand ("rhs read 4.0×, once per m tile");
    - the number of transfers and their mean size;
    - calls per `nisa` function;
    - how full each tile is (partition use out of 128, free use out of 512).
  - Layer 3, on the device (stretch): `neuron-profile` on NC 0–1. Nothing installed obviously runs an NKI
    kernel on the device from the pod (`torch_neuronx` is missing), so ask an Annapurna engineer first.
- **One improvement**, for the coder's improve mode. A rule first (re-read factor > 1, so hoist that
  operand's load out of the loop it repeats in). The model (≤100 tokens out) only when no rule matches.
- **Ends the thread** after two improvement tries with no byte reduction. The best correct kernel is
  kept either way, and a change that breaks correctness is never accepted.

## Fitting in 8K

A **prompt packer** builds every prompt from named sections with priorities and caps:

- It counts tokens with the Qwen3 tokenizer, which is in the pod's model cache.
- It drops whole sections lowest-priority first: old ledger lines, then extra cards, then examples.
- It never cuts code. A kernel over its cap is a failure: "kernel too long".
- It asserts `prompt + max_tokens ≤ max_model_len − 192`.
- It logs every section's token count, so overflows show up in the data.

| model call | max output | prompt cap | expected prompt (estimate) | worst case |
|---|---|---|---|---|
| planner | 160 | 3,000 | ~2,200 (task 600, API map ≤1,000, failed approaches ≤300, shapes 100) | 3,160 |
| coder, write | 900 / 1,400 | 4,000 | ~1,800 (task 600, plan 120, cards ≤900, skeleton 80) | 5,400 |
| coder, apply / improve | 900 / 1,400 | 4,500 | ~1,700 (kernel ≤1,200, change 100, card 300) | 5,900 |
| debugger, model path | 100 | 4,000 | ~2,000 (kernel ≤1,200, error 300, card 300, plan 50) | 4,100 |
| reviewer, model path | 100 | 3,500 | ~1,800 (kernel ≤1,200, profile 300) | 3,600 |

The manager and retriever make no model calls. Every call fits with >2K tokens to spare. Today's prompts
are 546–602 tokens (median) and answers ~280 (*measured*). The level most likely to press on 8K is level 8
(attention), where kernels are longer.

## Speed (estimate, at ~79 ms/token with 2 threads)

| step | output tokens | time |
|---|---|---|
| plan | ~120 | ~10 s |
| code (write) | ~350 | ~28 s |
| checks | — | ~1–2 s |
| debugger, rule path | 0 | ~0 s |
| debugger, model path | ~80 | ~6 s |
| code (apply), whole kernel | ~350 | ~28 s |
| code (apply), edit mode | ~80 | ~6 s |
| reviewer, rule path / model path | 0 / ~80 | ~0 / ~6 s |

Today, a round of 4 identical samples takes ~50 s (*measured*). The new loop spends similar time per step,
but each step is a different, targeted move. Judge it by **tokens and seconds per solve**, not per round.

## Code layout: build beside `agent.py`, not inside it

`agent.py` stays the untouched baseline, which also keeps us clear of Leyang's edits to it. The new system
imports what it needs from it (`grade`, `extract_code`, the HTTP client) and moves code out later.

```
projects/02-kernel-agent/
  agent.py                 baseline, unchanged
  agent2.py                CLI for the new system; same flags where they overlap
  kernel_agent/
    config.py              roles → endpoint, model, temperatures, caps
    llm.py                 client, retries, token counting, the prompt packer
    ledger.py              approaches, threads, attempts, stop reasons
    events.py              one JSON line per model call and per check
    checks.py              lint, the rules scan, simulate, compare; a process pool
    manager.py             threads, termination, routing
    planner.py
    retriever.py           lookup() over the knowledge sources, withhold.json enforced here only
    coder.py               write / apply / improve
    debugger.py            error types, rule templates, the model path
    reviewer.py            accept bars, rule improvements, the model path
    profile.py             simulator measurements: bytes per operand, transfers, op counts
    knowledge/nkidocs.py   introspection of the installed nki, cached per pod
    knowledge/thirdparty.py  chunk index for third_party/
    knowledge/examples.py  verified snippets, later the solved-kernel library
  tests/                   unit tests; the scratchpad tests move here
```

For tests without the model:
- `gptoss/mockserver.py` (a fake OpenAI endpoint) or a replay of recorded answers from `attempts.jsonl`;
- `--offline` (the reference kernels).

## Build order

Each step goes behind a flag, off by default. Measure each step on level 1 with `--repeat 3` against the
baseline before starting the next. Stopping after any step still leaves a result to report.

| step | builds | gate before moving on |
|---|---|---|
| 0 | `llm.py` and the packer, `ledger.py`, `events.py`, `checks.py`, a manager running today's loop (code, then repair from the raw report) | level 1 reproduces the baseline's ~0.30; no prompt over budget |
| 1 | debugger (error types, rule templates, model path, approach-wrong) and lint | ≥90% of the 168 recorded errors classified; how many failures lint catches before the simulator; L1 ×3 |
| 2 | retriever (introspection, cards, API map, `third_party/` with withholding) and the coder's three modes | the withholding test passes; examples still simulate; L1 ×3, L1–4 ×1 |
| 3 | planner, and manager threads and replans | L1–4 ×3 |
| 4 | reviewer, with profile layers 1–2 and the bypass fix | L4–7 ×3; no zero-byte passes |
| 5 (stretch) | model-driven manager vs rules, 32B planner and debugger, retriever lookups, edit mode, the solved-kernel library, device profiling | each as its own A/B |

**First thing to try, before building more:** run lint and the debugger's classifier over the 168 recorded
level-1 attempts. It needs no model, and shows how many failures rules alone would have caught.

What each run reports:
- solve rate per level, with the spread over N;
- best reward;
- tokens and seconds per level, and calls per role;
- stop reasons;
- distinct approaches tried;
- how many prompts the packer had to trim.

The ablation table (baseline, then each role added in turn) is the hand-in.

**Run budget:** a level-1 run takes ~7 min on the 8K server, so ~20 min per configuration at `--repeat 3`.
Run one configuration per server at a time.
- Ours: seat-35 (8B) and seat-199 (shared; needs `./serve.sh`).
- With permission: seats 36 and 37.
- The 32B server on seat-198, for the planner experiment.

## Team split (suggested)

Step 0 fixes the interfaces first (`Plan`, `Card`, `Attempt`, `CheckResult`, `Change`, `Verdict`
dataclasses), so the three of us can then work in parallel:

| who | owns |
|---|---|
| A | manager and the plumbing: `llm.py` and the packer, `ledger.py`, `events.py`, `checks.py`, `agent2.py`, the runs |
| B | retriever and debugger: introspection, cards, `third_party/` and withholding, lint, error types and templates |
| C | planner, coder and reviewer: their prompts, `profile.py`, the bypass fix, the write-up |

## Risks

- **More roles means more calls.** The planner, debugger and reviewer add tokens. Rules come first so
  most of their decisions cost nothing; measure tokens per solve, and drop any model path that doesn't
  pay for itself.
- **An 8B model drifts off structured formats.** Use line formats, parse leniently, and fall back
  (e.g. treat the whole reply as the plan or the change).
- **Answer leakage** from the docs: one enforcement point for `withhold.json`, plus a test that scans
  every card served at every level.
- **Doc version skew:** introspection first; third-party snippets only if they simulate on 0.6.0.
- **Hint fairness:** report the level-hint and skeleton levers both on and off.
- **Simulator is not device:** label every performance number.
- **Check concurrency:** process isolation, never threads.

## Open questions

1. May prompts use nkibench's own level hint (`notes`)? We'll report both either way.
2. Is there a supported way to run an NKI kernel on NC 0–1 from a seat pod for profiling? Ask an Annapurna
   engineer.
3. Who owns the seat-198 Qwen3-32B slot, and for how long? It decides whether the planner and debugger
   can use 32B.
