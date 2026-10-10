# Experiment log: kernel agent (project 2)

Goal: get Qwen3-8B unstuck on levels 1, 3 and 4 of `projects/02-kernel-agent`.
Every number here is from the CPU simulator (`nki.simulate`), not on-device timing.

**Setup, identical for every run:** seat-221, Qwen3-8B via `./serve.sh` (TP=2, context 8192),
`agent.py --level L --rounds 8 --context 8192 --repeat 2` for L in 1, 3, 4, plus per-experiment flags.
Runner: `runexp.sh NAME [flags]` in the pod writes `NAME.log` and `NAME_L{1,3,4}.jsonl`.
One change per experiment; a change is kept only if no level gets worse.

## Pre-flight

- `python nkibench.py --selftest`: **SELFTEST PASSED** (nki 0.6.0 importable, `nki.simulate`).
- `python nkibench.py --level 4 --check reference_level4.py`: reference passes all shapes.
- Model server health: 200.

## Findings before any experiment (these changed the plan)

**1. At temperature 0.6, the model's kernel output is fixed.** The level-1 first prompt, sent once
and then four times in parallel, gave five byte-identical replies (294 tokens, md5 `78d7d5`). A poem
prompt sent four times in parallel gave four different replies, so the server samples correctly;
code is simply low-entropy at 0.6. Consequences:
- `--samples 4` buys no extra attempts here and costs about 2.5x per round (53 s vs 21 s).
- The README's "zero spread on levels 1, 3, 4" is likely this determinism rather than a capability
  wall. Repeating a run at 0.6 should reproduce it exactly, so `--repeat 2` is enough to confirm.
- At temperature 1.0 the same prompt gave 4 different kernels out of 4; at 1.3, 4 out of 4.

**2. Sending a `seed` in a request crashes the server.** vLLM-Neuron 0.24 raised
`NotImplementedError: ... implement getNewGenerator ... for PrivateUse1` and the engine died
(`EngineDeadError`, every later request HTTP 500). An orphaned `VLLM::Worker_TP0` then held the
NeuronCores, so `./serve.sh` failed with `Logical Neuron Core(s) not available ... cores busy` until
that process was killed. Restart took ~2.5 min. **Never pass `seed` to this endpoint.** Crash log
kept in the pod at `/tmp/vllm.crash-seed.log`.

**3. At level 1, 0.30 does not mean "runs but wrong numbers".** The weights are parses 0.1, rules
0.2, runs 0.2, correct 0.5. Every level-1 attempt had `runs: False`: the kernels **crashed in the
simulator**, so `describe_mismatch` never fired. The model builds average pooling out of
`nc_matmul` on a 128x128 tile per 2x2 window, and the feedback walks it through one crash at a time:

| rounds | feedback the model got |
|---|---|
| 0-1 | `dma_copy requires src and dst to have the same number of elements, got src=4, dst=16384` |
| 2 | `nc_matmul() got an unexpected keyword argument 'transpose_moving'` |
| 3 | `dst must be in ['psum'], got sbuf` |
| 4-6 | `nki.isa has no attribute 'multiply'` |
| 7 | `nki.isa has no attribute 'scalar_mul'` |

Each message fixes the crash it names, and the next one appears. Nothing tells the model that the
approach itself is wrong: the reference uses a strided view and `nl.sum` over the window axes, then
multiplies by 1/(pool area). Rounds 4-7 come from an earlier, partial run.

**4. Generation is ~14 tokens/s for one request**, ~22 tokens/s total for four in parallel, so a
round is ~20 s with one sample. The README's ~8 s rounds were on a standalone trn2.3xlarge.

**Incident.** An earlier background agent kept running after it was told to stop, and ran a second
baseline against the same server for a few minutes; both partial runs were discarded (moved to
`aborted/` in the pod). It is stopped and nothing else of it is running.

**Change made to the harness:** `agent.py` gained `--temperature` (default 0.6, the old hard-coded
value), so behaviour is unchanged unless the flag is passed.

## Baseline (`base`)

Flags: `--samples 1 --temperature 0.6` (equivalent to the old `--samples 4`, since the four
samples were identical). 20:43 to 20:56 pod time, 14 minutes.

| level | run 1 | run 2 | solved |
|---|---|---|---|
| 1 average pooling | 0.30 | 0.30 | 0/2 |
| 3 matmul, one tile | 0.30 | 0.30 | 0/2 |
| 4 matmul, tiled | 0.62 | 0.30 | 0/2 |

Same numbers as the README. Level 1's two runs were byte-identical; levels 3 and 4 diverged in run 2,
so the output is near-deterministic rather than fully fixed (correcting finding 1 slightly).

**Where each level stops.** Almost every attempt **crashes in the simulator** (`runs: False`). The
only kernel that ran at all was level 4 run 1 round 3 (1 of 4 shapes correct, then a crash on the
256-row shape). So `describe_mismatch`, the "numbers are wrong" feedback, is almost never reached;
the crash messages from `enrich()` in `agent.py` are what the model actually learns from.

- **Level 1** stops on `module 'nki.isa' has no attribute 'multiply'`, 4 rounds running. The
  feedback says "nothing similar exists" and lists 25 alphabetical names (`NkiInstruction`, ...).
  The function exists as `nl.multiply`, in the other module; nothing says so.
- **Level 3** stops on `cannot reshape array of size 32768 into shape (1,64)`, answered with
  "Do not reshape". The model never called reshape: it allocated the matmul result as `(1, M)`
  instead of `(M, N)`, and the simulator failed writing 64x512 values into 64 slots.
- **Level 4** reaches 0.62 once, then regresses to `dma_copy requires HBM or SBUF tensors, got
  src=MemoryRegion.psum`, which reaches the model with no explanation at all.

## Experiment 1 (`exp1`): sample variety

Flags: `--samples 4 --temperature 1.0`. Hypothesis: four different kernels per round explore more
than one near-fixed kernel, and the agent keeps the best. 21:00 to 21:35 pod time, 34 minutes.

| level | baseline | exp1 | attempts | distinct kernels | ran at all |
|---|---|---|---|---|---|
| 1 | 0.30, 0.30 | 0.30, 0.30 | 64 | 10 | 0 |
| 3 | 0.30, 0.30 | 0.30, 0.30 | 48 | 12 | 0 |
| 4 | 0.62, 0.30 | **0.62, 0.62** | 32 | 8 | 26 |

**Result:** small gain on level 4 only (mean 0.46 to 0.62: best-of-4 reached 0.62 on round 0 in both
runs). No level solved; 2.4x the wall-clock. Variety does not get past a wall the feedback cannot
explain: at level 1, 29 of 64 attempts hit the same `nisa.multiply` error; at level 3, 32 of 48 hit
the "Do not reshape" message.

**Level 4's real wall shows up here:** 26 of 32 attempts *run* and pass the 128-row shape, then fail
on the 256-row one with `dma_copy dst partition dimension 256 exceeds maximum 128`. The model writes
one tile and never loops over several. That message already exists and is detailed, and is ignored.

**Decision:** not adopted as the default (cost too high for one level's gain); experiment 2 runs at
baseline flags so the feedback change is isolated. Worth combining with whatever wins, at the end.

## Experiment 2 (`exp2`): crash feedback that names the fix

Three changes to `enrich()` in `agent.py`, one per level, so each level's result is attributable to
one message. Flags as baseline (`--samples 1 --temperature 0.6`).

| level | old feedback | new feedback (appended) |
|---|---|---|
| 1 | "`nki.isa` has no `multiply`, and nothing similar exists. Its real names include: NkiInstruction, ..." | "`multiply` is not in `nki.isa`, it is in `nki.language`: write nl.multiply instead of nisa.multiply." |
| 3 | "Do not reshape. Work with the shapes you were given ..." | "A result of 32768 elements was written into a tile of shape (1, 64), which holds 64. The destination is the wrong shape. nc_matmul with stationary (K, M) and moving (K, N) produces an (M, N) result, so the psum tile, the sbuf tile and the output must all be (M, N)." |
| 4 | (none) | "nisa.dma_copy cannot read PSUM. First nisa.tensor_copy the PSUM tile into an SBUF tile of the same shape, then nisa.dma_copy that SBUF tile to the shared_hbm output." |

None of these gives the answer's values; each names the one change. Unit-checked in the pod against
the exact baseline error strings. Ran 21:37 to 21:49 pod time.

| level | baseline | exp2 | did the new message fire? |
|---|---|---|---|
| 1 | 0.30, 0.30 | 0.30, 0.30 | **yes, and the model followed it** |
| 3 | 0.30, 0.30 | 0.30, 0.30 | no |
| 4 | 0.62, 0.30 | 0.62, 0.62 | no |

**Level 1: the message worked and the wall moved.** Told "write nl.multiply instead of
nisa.multiply", the model moved modules, but for both names: it wrote
`nl.tensor_scalar(dst=tile, data=0.5, op0=nisa.multiply)`, the exact reverse of the correct
`nisa.tensor_scalar(..., op0=nl.multiply, ...)`. The next message moved `tensor_scalar` back, the
one after moved `multiply` back, and it alternated to round 8. It no longer stopped early (the old
message ended both runs at round 7 with "identical failure 4 rounds running"), but the score held.
Each message is correct and fixes one name, and that is why it oscillates: the fix needs both
names changed at once.

**Levels 3 and 4: not attributable.** Level 3's first kernel differed from the baseline's (near-
deterministic, not fixed), so it stuck on an earlier crash, a `dma_copy` element mismatch, and never
reached the reshape error. Level 4 stuck from round 0 on `partition dimension 256 exceeds maximum 128`;
its improved run 2 came from a different first kernel, not from this change.

**Decision:** keep all three (each message is strictly more accurate than the one it replaced, and
none fired wrongly). `agent.py` is at this state.
