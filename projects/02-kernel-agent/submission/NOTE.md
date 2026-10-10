# Kernel agent: what the checker's feedback is worth

Seat 17 · project 2 · **interim note, 10 Oct 15:05 EDT.** Ten more runs are queued on the seat; the
tables here cover the runs that have finished and will be regenerated when the rest do.

## What we ran

The repo's kernel agent, unchanged except for the checker's feedback. Model: Qwen3-8B on the seat's
Trainium chip, started with `./serve.sh`, thinking off, context 8192, 4 samples per round, 8 rounds.
The model, its settings and the task prompt were not changed. Checker: `nkibench.py` layer 1, which
simulates the kernel on the CPU. Nothing here was timed on the device.

One variable changes between runs: what the checker says when a kernel fails.

| feedback | what the model is told |
|---|---|
| `enriched` | the repo's original: the exception plus a fix instruction |
| `located` | the same, plus the model's own failing line quoted back |
| `directed` | plus the two real shapes in a failed copy, and where a misplaced name really lives |
| `directed3` | plus a message for each specific wall found in the logs below |

Every `directed` and `directed3` message was written after seeing the model hit that exact wall on
this seat. The kernel line and error that prompted each one are quoted in a comment beside it in
`agent.py`.

## Results so far

| level | `located`, 4 runs | `directed`, 1 run |
|---|---|---|
| 1 average pooling | 0.30, 0.30, 0.30, 0.30 | 0.30 |
| 2 transpose | 1.00, 0.30, 1.00, 0.30 | 1.00 |
| 3 matmul, one tile | 0.30, 0.30, 0.30, 0.30 | 0.30 |
| 4 matmul, tiled | 0.75, 0.30, 0.62 (fourth still running) | 0.75 |

`enriched` and `directed3` have no finished run on this seat yet. Per-run detail:
[`RESULTS.md`](RESULTS.md). Mistakes grouped and counted:
[`TAXONOMY-located-5runs.md`](TAXONOMY-located-5runs.md), [`TAXONOMY-directed.md`](TAXONOMY-directed.md).
Every attempt with its kernel, feedback and score: [`logs/`](logs/).

## What the logs show

1. **One run is not a result.** The same checker gave level 4 a best score of 0.75, 0.30 and 0.62 in
   three runs, and solved level 2 in two runs out of four. Our earlier draft of this note credited
   0.75 on level 4 to `directed`; `located` reached the same score in one run of three, so that
   claim is withdrawn. The repo's own figure for level 4 is 0.62 on five of five runs.
2. **Level 2 is decided in the first round.** When the first answer was right it scored 1.00 at
   round 0; when it was wrong, neither run recovered, in four and six repair rounds. The first prompt is
   identical in every feedback mode, so level 2 measures feedback only in the runs that start wrong.
3. **Two true messages can form a loop.** On level 2 the model allocated a tile of 16384 elements for
   an input of 384. "The tile holds 16384 elements, you copied 384" made it copy a 128-wide slice;
   "you indexed up to 127 on a dimension of 12" sent it back. The same loop appeared on level 3.
   Three runs alternated between the two messages for six or seven rounds. This is the largest
   single pattern in the log: 19 of 74 repair steps.
4. **A wrong diagnosis cost a level.** On level 4 at 0.62 the model deleted the copy into its output
   and returned an array of NaN. The checker said "usually an uninitialised tile". The model
   invented `nisa.psum_init`, was offered `gpsimd_engine` as the closest real name, called it, and
   finished the run at 0.30. Every output element was NaN, which the checker could see and did not
   say.
5. **Three errors had no advice at all** beyond the quoted line, and each was repeated for two to four
   rounds: an index past the end, `tuple index out of range`, and a contraction dimension mismatch.
6. **A named fix is taken; a list of names is not.** The model wrote `nisa.multiply`. Told the first
   25 names in `nki.isa`, it repeated the mistake for three rounds and then invented
   `nisa.scalar_mul`. Told "write `nl.multiply`" in the `directed` run, it fixed it in the next round.
   This is one observation in one run.
7. **Three of four samples are wasted after the first round.** In 71 of 74 repair rounds all four
   samples were the same kernel. In first rounds they differed on levels 2 to 4 (2 to 4 distinct
   answers). Level 1 produced the identical kernel in every sample of every run for the first six
   rounds, so repeating level 1 adds no information.
8. **Our own metric was wrong once.** `report.py` counted a repair step as "moved" if the mistake
   differed from the round before. That scored the loop in point 3 as six steps out of six moving.
   It now counts a return to an earlier mistake separately ("went back").

`directed3` replaces the messages behind points 3, 4 and 5 and the two behind the level 3 wall
(a "give it the shape (1, N)" instruction that led to a "cannot reshape" error in a kernel that never
calls reshape). Whether those changes raise scores is what the queued runs measure.

## Where the prompt goes (`located`, 356 attempts, mean per prompt)

| prompt | prompt tokens | answer tokens | made of (characters) |
|---|---|---|---|
| first | 717 | 333 | API card 1581, task wording 506, reference 400 |
| repair | 577 | 352 | previous kernel 1052, checker feedback 587, wording 200 |
| repair with ledger | 716 | 324 | previous kernel 998, checker feedback 528, ledger 485, wording 267 |

No answer was cut off: all 356 attempts ended with `finish_reason=stop`. A round took a median of
76 s; while a second agent shared the model, most rounds took 190 to 265 s.

## What these numbers do not show

- **No baseline on this seat yet.** No `enriched` run has finished here; the 0.62 quoted for level 4
  is the repo's figure from different hardware. Five alternating pairs of `enriched` and `directed3`
  on levels 2 to 4 are queued.
- **`directed3` is untested.** Its messages were written from failures seen in these logs and have
  unit tests on their wording, but no run has used them.
- **Three or four runs is still few.** With level 4 ranging from 0.30 to 0.75 inside one mode, a
  difference between modes needs to be larger than that range to mean anything.
- **Level 1 is not close.** The model computes a matrix product of each window with itself where a sum
  is needed. Fixing one error per round does not change that approach.
- **Some `directed3` messages describe a method**, for example how to split M and N into chunks, at
  the same level of detail as the repo's own message for the K dimension. None contains code for a
  kernel.
- **The `located` log was recovered, and its last few rounds may be missing.** The log file was
  deleted from the seat while the run was writing to it; we read it back through the process's open
  handle once a minute.
- **Simulator only.** Scores come from `nki.simulate` on the CPU.

## Reproduce

```bash
cd /workspace/projects/02-kernel-agent
python nkibench.py --selftest
python agent.py --all --rounds 8 --samples 4 --context 8192 --feedback located --repeat 5 --log located.jsonl
python report.py located.jsonl             # one row per run and level
python taxonomy.py --log located.jsonl     # mistakes grouped and counted
```

Run one agent at a time on a seat.
