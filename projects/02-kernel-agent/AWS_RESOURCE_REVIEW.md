# AWS Neuron resources: review and hackathon roadmap

Reviewed 2026-10-10. Development: `/tmp/trainium-kernel-dev`, branch
`feat/nki-diagnostic-selection`, base `8f1ca41`. Reference clone:
`/tmp/aws-neuron-agentic-reference`, AWS commit
`ee25e45b9ace4aaefd8b8913bf5fa9b68adf6674`.

This document attributes the external workflows and documentation to AWS. They
were studied as references, without installing the plugin, deploying its agents,
executing its scripts, or running kernels/inference. Phase 1 is implemented;
Phases 2 and 3 below are proposals. No improvement in correctness is claimed.

## Inspected material and evidence boundaries

All requested files were read: `agents/neuron-nki-agent.md`,
`agents/neuron-nki-writer-agent.md`, `agents/neuron-nki-debugger-agent.md`, writing,
debugging, and docs `SKILL.md`, and writing references `nki-language-constraint`,
`api-translation`, `common-patterns`, `memory-patterns`, and `indexing-patterns`.
Also inspected the artifact deployment implementation and environment hook.

The agents specify an Opus-backed, tool-using host workflow. The writer maps
operations, designs layout/tiling, allocates explicit destinations, and validates
against CPU references. The debugger captures error code/location/operation,
tries a direct fix, consults documentation/examples, then progressively
simplifies tiles, tiling, fusion, dtype, and parallelism. It caps iterations at
10, reacts to repeated errors, and reports trade-offs. The docs skill uses local
symbol/task/error indices, then reads relevant files. These are concrete workflow
instructions, not a measured Qwen3 implementation. `src/neuron_agentic_development/deploy.py`
copies/transforms artifacts for supported clients; the hook checks dependencies.
Neither is an executable counterpart of our best-of-four grading controller.

Source links, pinned to the reviewed AWS commit:
- [Unified agent](https://github.com/aws-neuron/neuron-agentic-development/blob/ee25e45b9ace4aaefd8b8913bf5fa9b68adf6674/agents/neuron-nki-agent.md)
- [Writer](https://github.com/aws-neuron/neuron-agentic-development/blob/ee25e45b9ace4aaefd8b8913bf5fa9b68adf6674/agents/neuron-nki-writer-agent.md)
- [Debugger](https://github.com/aws-neuron/neuron-agentic-development/blob/ee25e45b9ace4aaefd8b8913bf5fa9b68adf6674/agents/neuron-nki-debugger-agent.md)
- [Documentation routing](https://github.com/aws-neuron/neuron-agentic-development/blob/ee25e45b9ace4aaefd8b8913bf5fa9b68adf6674/skills/neuron-nki-docs/SKILL.md)
- [Writing guidance](https://github.com/aws-neuron/neuron-agentic-development/blob/ee25e45b9ace4aaefd8b8913bf5fa9b68adf6674/skills/neuron-nki-writing/SKILL.md)
- [Device debugging guidance](https://github.com/aws-neuron/neuron-agentic-development/blob/ee25e45b9ace4aaefd8b8913bf5fa9b68adf6674/skills/neuron-nki-debugging/SKILL.md)

## Architecture comparison

| Capability | AWS inspected workflows | Our baseline / approved Phase 1 | Useful delta; avoid duplication |
|---|---|---|---|
| Diagnosis | Compiler codes, source location, API, suggestions; obvious fix first | `grade()` and `enrich()` already supply targeted repairs; Phase 1 adds canonical categories/signatures | Route categories to relevant constraints; retain raw feedback |
| Documentation | Local indices by symbol, task, error, hardware; cross-reference details | Static `API_CARD`, installed signatures and fuzzy real-name lookup (`agent.py:172,208,231,249`) | Small semantic/shape cards; fuzzy name similarity does not establish an equivalent operation |
| Repair planning | Direct fix, examples, progressively simplify; report trade-offs | `repair_prompt()` carries complete selected code and feedback but demands keeping everything else identical (`agent.py:396`) | Docs can explain a local invariant; larger algorithm revision is a future planner, outside current scope |
| Failure memory | Debug timeline, iteration cap, repeat reaction | Per-solve ledger and exact-feedback cycle stop (`agent.py:538`); Phase 1 normalized failure history | Existing memory is sufficient for current ablations; no inspected durable failure database to adopt |
| Generation | A tool-using writer, staged translation and validation | Four parallel requests receive the same prompt (`agent.py:473`) | Deterministic, distinct prompt perspectives with the same call budget |
| Selection | No equivalent best-of-four numerical reward/tie selector found | Stable maximum reward; Phase 1 opt-in diagnostic ties (`agent.py:513`) | Measure tie selection after ensuring there are genuinely different candidates |
| Hardware | Target detection, explicit compilation/device numerics, profiling and isolation guidance | Static rules, CPU simulation, NumPy numerics, input-mutation/traffic/hazard gates; no device timing in the loop (`nkibench.py:358,421,611`; `agent.py:133`) | Preserve checker; later device-validate successes without claiming simulation latency is device latency |
| Iteration/evaluation | Up to 10 debug iterations; CPU references and optional intermediate checks | Latest selected attempt drives next repair, repeats and cycle stop; logged code/reward/feedback | Controlled ablations, run identities, actual tokens, phase timing; do not rebuild the verification loop |

AWS offers broader indexed knowledge and a stronger explicit simplification workflow.
Our original contribution can be measurable behavior under an 8B-model/8192-token
constraint: canonical diagnoses, explainable reward-preserving selection,
four-call prompt diversification, and compatibility-filtered small cards. These
are engineering hypotheses; AWS provenance or extra components alone is not a result.

## Exact official documentation links

| Resource | What to extract for our failures |
|---|---|
| [dma_copy](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.dma_copy.html) | Element count equality, memory regions, slices |
| [nc_matmul](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html) | Actual arguments, operand layout, Trainium2 dtype/buffer/tiling and accumulation constraints |
| [tensor_scalar](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_scalar.html) | op0/operand0 and scalar or per-partition vector operands |
| [tensor_reduce](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_reduce.html) | Trailing free-axis reductions and output shape |
| [tensor_copy](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_copy.html) | On-chip transfer and matching partition/free element counts |
| [ndarray](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.language.ndarray.html) | Explicit shape, dtype, buffer and allocation |
| [simulate](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.simulate.html) | `nki.simulate(kernel)(*args)` |
| [Simulator guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/nki_simulator.html) | Target selection, unsupported behavior, simulator/device distinctions |
| [Language guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/nki-language-guide.html) | Specialization versus runtime, views, supported constructs |
| [Memory hierarchy](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/memory-hierarchy-overview.html) | Explicit HBM/SBUF/PSUM movement |
| [Trainium2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html) | NeuronCore-v3 memory and compute engines |
| [Tiling](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/tiling-overview.html) | Partition/free layout and instruction limits |
| [Indexing](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/indexing-overview.html) | Slices, free versus partition axes; exclude complete pooling examples from cards |
| [NKI release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/nki.html) | Neuron 2.32.0 / NKI 0.6.0 correspondence and migration details |

These pages returned HTTP 200 during review. Older `/nki/tiling-overview.html`,
`/nki/indexing-overview.html`, and `/nki/nki-language-guide.html` paths returned
404; use the links above. Future cards should record a versioned URL, checked
date, SDK build, and reviewed AWS commit rather than depending on mutable `latest`.

## Compatibility audit: installed NKI 0.6.0

Read-only `inspect.signature`, docstrings and source inspection established:

| API | Installed parameters (annotations omitted) |
|---|---|
| `dma_copy` | `dst, src, priority=None, oob_mode=error, dge_mode=unknown, engine=unknown, name=None` |
| `nc_matmul` | `dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, accumulate=None, tile_position=(), tile_size=(), perf_mode=none, name=None` |
| `tensor_scalar` | `dst, data, op0, operand0, reverse0=False, op1=None, operand1=None, reverse1=False, engine=unknown, name=None` |
| `tensor_reduce` | `dst, op, data, axis, negate=False, keepdims=False, name=None` |
| `tensor_copy` | `dst, src, engine=unknown, name=None` |
| `ndarray` | `shape, dtype, buffer=nl.sbuf, name='', address=None` |
| `simulate` | `kernel` |

Installed version: `0.6.0+31049202112.g85070674`. These facts confirm symbol/signature
availability, not that arbitrary uses compile or run correctly. No SDK change,
new simulation, or device execution was performed during this review.

Important filters and conflicts:
- The AWS summary tables' generic `K <= 2048` conflicts with the detailed API and
  installed normal-mode limits: benchmark FP32 contraction tile K <= 128,
  stationary free M <= 128, moving free N <= 512 on Trainium2. FP8 modes are
  different and are not recommended for these FP32 experiments. Inputs are SBUF;
  matmul output is FP32 PSUM. `transpose_moving` does not exist, and `is_transpose`
  is a specialized mode, not a drop-in replacement.
- The SDK source enforces on-chip tensor rank >= 2. Full on-chip capacity is
  not the same as the maximum tile accepted by one instruction. Do not treat
  blanket SBUF F=32767 or PSUM F=512 statements as universal capacity rules.
- Some AWS migration tables ban `nl.load`, `nl.store`, and reduce `negate`.
  Installed load/store/sum have actual implementations, and reduce accepts
  negate. Explicit ISA guidance remains useful without claiming those helpers
  are absent. `nl.arange` is absent here.
- Installed range helper docstrings describe deprecated, fully unrolled aliases
  of `range`; AWS writer/common-patterns loop-performance distinctions conflict
  with that description. Do not promise a speedup merely by changing a loop name.
- `nl.fori_loop` and `nl.while_loop` exist; no frontend migration is required for
  our static-shape hackathon work. Some guides still show older dynamic-loop
  forms; introduction/replacement does not mean unconditional removal in 0.6.
- NeuronCore-v4 compute `.indirect()`, DMA priority, and larger matmul destination
  limits are Trainium3 features despite being documented in the same SDK.
  Presence of a Python symbol is insufficient hardware validation.
- Installed target resolver honors `NEURON_PLATFORM_TARGET_OVERRIDE`, then
  detects hardware, falling back to trn3. Future experiments must consistently
  specify and log trn2 for both arms without changing the active baseline.
- Some snippets contradict their own text: indexing allocates a transposed tile
  with P=512, and docs lookup demonstrates an ISA activation without explicit
  dst. Do not inject these. AWS utility conventions are not new SDK requirements;
  do not copy/import nkilib machinery for small kernels without need.

## Five actionable insights

1. Route unsupported arguments to the real installed signature AND the operation
   semantics. Removing `transpose_moving` alone cannot fix an incorrectly laid-out
   matmul. Preserve the exact failing expression and show only the relevant rule.
2. For DMA mismatch, connect the reported counts to the candidate's src/dst slice
   expressions and allocation. Level 1 recorded src=4/dst=16384; Level 2
   src=384/dst=512. A short equal-count/boundary invariant can be more useful than
   another generic whole-kernel template. Counts alone do not identify the line.
3. Pair buffer diagnosis with the full legal transfer route and rank requirement.
   Recorded `dst must be in ['psum'], got sbuf` maps to matmul placement when
   candidate source confirms matmul. Generic placement errors must not always
   retrieve matmul guidance. On-chip allocations need P and F dimensions.
4. Diversify prompts before expecting selection to rescue identical candidates.
   The immutable 144-record snapshot has 30/36 rounds with one AST-distinct kernel,
   two rounds with two, and four with three. Existing tie selection changed one
   Level 3 choice; it changed none in Levels 1-2. This is mechanism evidence only.
5. Treat simulation as a correctness filter, not hardware proof or a latency
   benchmark. Later validate successful kernels on reserved hardware. Keep the
   existing mutation, numerical, traffic and hazard gates; no replacement checker.

## Proposed local reference-card retrieval (Phase 3; not implemented)

Use a small version-controlled JSON or Python card catalog and a deterministic
`select_cards(diagnostic, source, compatibility, budget)` function. No vector
database, network request in the repair loop, or additional model call.

Each card stores: ID, covered category/API, installed signature, operation
meaning, operand shape/buffer constraints, a brief general repair principle,
hardware/SDK tags, source URL/section and revision, checked date, and evidence
status (`signature/source checked`, later `micro-simulated`, later `device checked`).
Never embed complete benchmark kernels, reference functions, pooling/transposition
tutorial solutions, or shape-specific benchmark answers.

Routing: category narrows possible cards; source AST resolves aliases and the
relevant call/operand/allocation. If the error lacks a source line, do not invent
one: report ambiguous calls. Rank exact API matches over general shape/memory
cards; UNKNOWN returns no card unless source provides an unambiguous relevant API.
Filter SDK/hardware compatibility before retrieval. Deduplicate already-present
signature advice. Return up to two cards under a proposed ~500-token total budget,
to be calibrated using actual tokenizer counts rather than len/4 guesses.

Append cards to the selected code plus original enriched feedback. Drop optional
cards first if needed to preserve the answer allocation and full code/error; do
not silently truncate context. Preserve the existing repair instruction initially
to isolate the retrieval effect. Wrong-algorithm failures may still require a
larger repair-policy change, which is explicitly beyond Phase 3's small experiment.
Log card IDs/version, inclusion reason, compatibility checks and token counts.

Tests: category+source routing, alias handling, missing/ambiguous locations,
unsupported targets, signature drift, budget exhaustion and no-solution content.
Small API exercises for future cards should run in unique files without invoking
grade's shared `/tmp/_agent_levelN.py`; schedule hardware tests only on a confirmed
available core. A card must accurately disclose its verification level.

## Proposed diversity and evaluation

Phase 2 changes `ask_parallel()` to dispatch one independently constructed prompt
per existing sample. Add a small pure prompt builder and opt-in strategy flag.
Keep sample 0 as the existing prompt; other perspectives focus on operand shapes,
buffer/dataflow, or tiling/output coverage. They must be generic constraints,
not hidden reference answers. Request one full kernel, not an extra plan response.
Do not assume a textual variation produces a different algorithm. Log prompt IDs,
AST hashes and distinct candidate count; retain duplicate grades for the initial
ablation so checker work and selection are comparable. Keep four model calls,
temperature 0.6, top_p 0.95, thinking off and current token ceilings.

Measure S=diagnostic selection, D=diverse prompts, R=retrieval independently:
full factorial `000,100,010,001,110,101,011,111`. For today's time budget, first
compare 000 versus D, then D versus D+S, then R independently and in combination.
Incomplete arms cannot establish full interaction effects; report that clearly.

Use identical benchmark cases, numerical tolerance/input seeds, 8-round ceiling,
4 candidates per active round, existing early stop and at least 5 repeats per arm
if feasible. Record rounds and candidates to first success; failures remain
censored at their actual stop rather than being excluded from average attempts.
Report solve counts/rates with uncertainty, per-shape correctness, best/average
reward with defined aggregation, failure categories, normalized recurrence,
unique candidate count, actual API usage tokens, and generation/checker/total
wall times. Retrieval naturally changes input tokens; measure that cost rather
than claiming equal token use. Timing comparisons need stable endpoint contention.

Before evaluation, add narrow shared instrumentation for response usage/finish
reason, run/repeat/candidate IDs and phase timing across all arms; these are not
currently captured completely. Default row compatibility must remain additive.
Use immutable revision and settings manifests, new unique artifact directories,
and sequential arms. Do not run while protected grading paths overlap. Use the
same explicit trn2 target setting for both new control and treatment runs; do not
retroactively equate these with an older differently configured run.

## Priority and effort

| Phase | Status / scope | Estimated effort excluding inference |
|---|---|---|
| 1 | Complete: classifier, optional selector, 28 tests, offline replay | Already validated; no application edits in this resource review |
| 2 | Generic distinct prompts, opt-in dispatch, diversity logs and mocked tests | 65-105 minutes |
| 3 | Curated API cards, category+source routing, budget/compatibility tests | 90-135 minutes |
| 4 setup | Shared actual usage/timing/run-ID instrumentation and evaluation manifests | 30-45 minutes |
| 4 runs | Repeated controlled experiments, initially Levels 1-3 then broader | Depends on measured endpoint/grade times; not executed |

Next coding task: Phase 2 prompt diversity. Keep the validated Phase 1 selector
and test whether four calls yield more than one structurally distinct kernel.
Then test whether that diversity improves solve rate. Full adaptive repair
planning, long-term failure databases, profile infrastructure, and SDK migration
are deferred. This review changed only worktree documentation.
