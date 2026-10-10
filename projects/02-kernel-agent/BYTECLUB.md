# Team byteclub: kernel agent results

**Seat 53. Qwen3-8B on our own Trainium chip. Settings: `--rounds 8 --samples 4 --context 8192`.**

## Headline

**Level 4 (tiled matmul) went from never solved to solved every time.** The organisers' agent solved it
0 times out of 5, with a best score of 0.62. Ours solved it 3 times out of 3, each within 2 rounds.

| level | organisers' baseline (5 runs) | + guide retrieval (run A, 2 runs) | + checker fixes (run C, 1 run) | + reward fix (run D, level 3 only) |
|---|---|---|---|---|
| 1 average pooling | 0/5, always 0.30 | 0/2, 0.30 / 0.30 | 0/1, 0.30 | not run |
| 2 transpose | 4/5, 0.50–1.00 | 0/2, 0.30 / 0.62 | 0/1, 0.62 | not run |
| 3 matmul, one tile | 0/5, always 0.30 | 0/2, 0.50 / 0.50 | 0/1, 0.50 | 0/1, 0.30 (stopped on the reshape message, see Next steps) |
| **4 matmul, tiled** | **0/5, always 0.62** | **2/2, solved in 2 rounds** | **1/1, solved in 2 rounds** | not run |

**On the number of runs:** the baseline is the organisers' published 5-run measurement, made with
the same settings. One run of our own baseline matched it on level 1 (0.30). Each of our runs takes
25–40 minutes on this model, so we have 2 + 1 + 1 runs, not 5 per condition. On levels 1, 3 and 4
the baseline has **zero spread** (the README calls these capability walls), so moving off those
values is attributable to the change. Level 2 is the noisy level, and on it we are **worse**; see
below.

## What we built

### 1. A retriever that decides which 8,000 tokens the model sees (`retrieve.py`)

The mentor's reference repo, `aws-neuron/neuron-agentic-development` (skills `neuron-nki-*`), is
about 1.4 MB of guides, API pages, tutorials and examples. The model sees 8,192 tokens in total. The
challenge brief puts it this way: *"deciding what the model gets to look at is the actual problem."*

- **Chunk:** every `##`/`###` section, and every example `.py` file with its docstrings and torch
  test harness removed.
- **Score:** BM25 over identifier-aware tokens, so `nc_matmul` also matches `matmul`. The query is
  the operation on the first round. On repair rounds it is **the checker's feedback**, weighted 3x,
  so a PSUM error pulls in the PSUM section.
- **Pack:** best-first under a budget (6,000 characters on the first round, 4,000 on repairs), capped
  so an answer of at least 2,500 tokens always fits. Chunks are never truncated, because half a code
  example teaches a broken pattern.
- **Freshness check, which is a checker for the docs:** every `nl.X` / `nisa.X` used in a chunk's
  *code* is checked against the NKI actually installed in the pod (0.6.0). **31 chunks were dropped.**
  We found this by measurement. The guide's own average-pooling tutorial uses `nl.mgrid`, which has
  been removed. Retrieved verbatim, it taught the model `nl.mgrid` and then `nl.arange` for three
  rounds in a row. Prose is exempt, because the migration tables mention removed calls only to
  forbid them.

`python agent.py ... --docs <path to neuron-agentic-development/skills>` turns it on. Without
`--docs`, the agent behaves exactly as before.

### 2. Checker fixes: every message names the change, not just the symptom

Each fix came from reading a real failed attempt in the log.

| failure the model hit | old message | new message | file |
|---|---|---|---|
| level 3, M=64: `for m in range(M // 128)` ran **zero** times, so the output was all NaN | "Usually an uninitialised PSUM or SBUF tile" (sends the model to check allocations, which were fine) | **"NOTHING WAS WRITTEN… the loops ran ZERO times… 64 // 128 == 0. Use (dim+TILE-1)//TILE…"**, plus a partial version naming the unwritten block | `nkibench.py` `describe_mismatch` |
| full-size tile allocated for a partial slice (`src=8192, dst=16384`) | "allocate with exactly the shape of the slice" (doesn't say which side is wrong) | **"the destination is 2x bigger… the slice is right; the allocation is wrong… msz = min(TILE_M, M - m*TILE_M)"** | `agent.py` `enrich` |
| model copied the tutorial's `assert M % 128 == 0` | bare `AssertionError` | **"That assert is YOUR code… the kernel rejected a legal input (M=64)… delete it"** | `agent.py` `enrich` |
| `nl.div_ceil` (the guide tells you to use `div_ceil`, but defines it in-file) | "closest real names are…" | **"div_ceil is not part of nki… write it yourself: def div_ceil(n, d): …"** | `agent.py` `enrich` |

### 3. A reward fix: the checker was rewarding a kernel that does nothing

Before: a kernel that never wrote its output (all NaN) scored **0.50** for "runs". The model's
genuine next step (loop fixed, tile still too wide, out of bounds) scored **0.30**. The loop keeps
the best-scoring attempt, so **it fell back to the do-nothing kernel every round**. Run C shows this
directly: the "NOTHING WAS WRITTEN" kernel won round after round. Now "runs" requires that at least
one output element was actually written. Re-grading the logged kernels confirmed the change: the
do-nothing kernel now scores 0.30, and the correct reference kernels still score 1.00.

## The honest part

- **Level 4 was solved by adapting the guide's own matmul tutorial.** The retriever found
  `tutorials/matrix_multiplication.md`, and the winning kernel follows it closely. That is
  legitimate, since the challenge is open-book and the organisers ship the reference kernels saying
  "hiding it buys nothing". But it means the gain is **retrieval**, not the model learning to tile.
  Level 4's test shapes are all multiples of 128, so the tutorial's assumptions hold. Level 3
  (M=64) breaks them, which is why it stays hard.
- **Retrieval hurt level 2.** The organisers got 4/5; we got 0/3, with a best of 0.62. The guide's
  transpose material is about `nc_transpose`, which swaps the partition and free axes. This level
  transposes *within* the free axes. The retrieved docs pointed the model at the wrong primitive. A
  retriever needs an off switch per level, or a relevance threshold. We did not build either.
- **Level 1 did not move.** Even with the `.ap()` API page retrieved, the 8B model does not get the
  `[stride, count]` access-pattern semantics right.
- Run counts are small: 2, 1 and 3, as explained above.

## Next steps, in order of what we expect to pay off

1. **Level 3: one more message.** The current blocker is `cannot reshape array of size 32768 into
   shape (128,512)`. The existing message says "Do not reshape", but the model never called reshape.
   It is the same full-size-tile-versus-M=64 mismatch, and should say so.
2. **A relevance threshold per retrieved chunk**, so a level whose docs mislead (level 2) gets none.
3. **The ablation flag is built but not yet measured:** `--no-same-op` drops the tutorials and the
   matmul example, to show how much of the level 4 gain is the tutorial itself.

## Files

- `retrieve.py`: the retriever and the freshness check (new)
- `agent.py`: the `--docs`, `--doc-chars`, `--doc-chars-repair` and `--no-same-op` flags; the
  `enrich()` messages; the reward fix
- `nkibench.py`: the "NOTHING WAS WRITTEN" and "PART OF THE OUTPUT WAS NEVER WRITTEN" messages
- `runs/`: every attempt with its score, code, feedback and **which guide chunks it was shown**
  (`*.jsonl`), plus the console logs

**Why run D reads 0.30 where runs A and C read 0.50:** the reward fix took away the 0.20 that the
do-nothing kernel used to get for "runs". The score is lower because it is now honest, not because
the agent got worse. `smoke.*` in `runs/` is the first try with retrieval, before the freshness
check. It is the run that showed `nl.mgrid` coming in from the stale tutorial.
