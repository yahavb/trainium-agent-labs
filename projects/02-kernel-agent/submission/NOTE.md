# Kernel agent: what the checker's feedback is worth

Team 36 · project 2 · 10 Oct, final note.

## What we did

We left the model, its settings and the task prompt alone and changed one thing: **what the checker
says when a kernel fails.** Then we measured, round by round, what the model did with it.

Model: Qwen3-8B on a Trainium chip, started with `./serve.sh`, thinking off, context 8192, 4 samples
per round, up to 8 rounds. Checker: `nkibench.py` layer 1, which simulates the kernel on the CPU. The
levels, test shapes and scoring weights are the repo's, unchanged. Nothing here was timed on the device.

| feedback | what the model is told |
|---|---|
| `enriched` | the repo's original: the exception plus a fix instruction |
| `located` | the same, plus the model's own failing line quoted back |
| `directed` to `directed4` | plus facts only the checker can see (real shapes, where a name lives, which input value belongs at which output position) and replacements for messages the logs showed were misleading |
| `directed5` | plus a pre-check that reads the whole kernel and lists every fault that is certain from its text, not only the first one that crashes |

Every message was written after seeing the model hit that exact wall on a seat. The kernel line and
error that prompted each one are quoted in a comment beside it in `agent.py`.

## Result

Best score per run. Each number is one run of up to 8 rounds.

| level | `enriched` (repo) | `located` | `directed3` | `directed4` | `directed5` |
|---|---|---|---|---|---|
| 2 transpose | 0.30 | 1.00, 0.30, 1.00, 0.30, 0.30 | 0.50 | 0.50 | not run |
| 3 matmul, one tile | 0.30 | 0.30 ×5 | 0.30 | 0.30 | 0.30, 0.30 |
| 4 matmul, tiled | 0.62 | 0.75, 0.30, 0.62, 0.75, 0.75 | 0.62 | **1.00**, 0.75 | stopped from outside after one round |

Level 1 stayed at 0.30 in seven of seven runs (`located` ×6, `directed` ×1) and was not run again.
One further `located` run of level 2 was cut off after three rounds at 0.30.

**Level 4 was solved once**, correct on all four shapes with K, M and N all tiled, in one of two
complete `directed4` runs. The repo records 0.62 on five of five runs, and its checker reached 0.62 in
our one run of it here. That is one solve in two attempts, not a rate.

**Level 3 is not solved.** Under `directed5` the model repaired a different fault in each of its
first five rounds and ran out of rounds short of a complete kernel. Under the repo's checker it
repeated its first fault. Repair steps that reached a fault not seen before in the run: 5 of 7 in
each `directed5` run, 1 of 5 under `enriched`. The score is 0.30 either way, because a kernel scores
nothing more until it runs to the end.

Per-run detail: [`RESULTS.md`](RESULTS.md). Every attempt with its kernel, feedback and score:
[`logs/`](logs/) (904 attempts).

## How level 4 was solved

Both `directed4` runs took the same path for four rounds: 0.62, 0.62, 0.50, 0.75.

1. **0.62 to 0.50.** The repo's advice for K over 128 says to "accumulate". The model added
   `accumulate=True` to `nisa.nc_matmul`, and every output element became NaN. Our version of that
   advice ends "do not add an accumulate argument". The model added it anyway, in both runs.
2. **0.50 to 0.75.** The checker said: remove `accumulate=True`; repeated calls into one psum tile
   already add up. Fixed in one round, both times.
3. **0.75 to 1.00.** The simulator's error for M over 128 carries no advice. Ours says to keep the K
   loop and add loops over M and N, with one psum tile per chunk. In the first run the next kernel was
   correct. In the second the model also wrote the psum result straight to the output, spent three
   rounds on an error that carried no advice, and ended the run at 0.30 (best 0.75).

## What the logs show

1. **One run is not a result.** One unchanged checker (`located`) gave level 4 a best score between
   0.30 and 0.75 across five runs and solved level 2 in two of five. An earlier draft of this note
   credited 0.75 on level 4 to one of our messages; `located` reached it in three runs of five, and we
   withdrew the claim.
2. **A message we wrote made a run worse.** The first version of our NaN message said "the array you
   return was never written". It was written. The model searched four rounds for a missing copy and
   finished at 0.50, below the 0.62 it had. The message now states only what the checker can verify
   from the kernel text. That corrected version is step 2 above.
3. **Two true messages can form a loop.** "The tile holds 16384 elements, you copied 384" made the
   model copy a 128-wide slice; "you indexed up to 127 on a dimension of 12" sent it back. Under
   `located` this was the largest pattern in the log: 22 of 100 repair steps.
4. **One error per round is too slow.** A kernel from the level 3 probe had six separate faults in
   three lines, all readable from the text. The checker reported the first, as a Python type error,
   four rounds running. `directed5` adds a pre-check that reports such faults together. It has not
   yet been exercised on the chip: see the limits below.
5. **Where the quoted line points matters.** Told the right shape for an output allocated too small,
   with the failing copy quoted, the model edited the copy for three rounds. With the allocation line
   quoted, it fixed it in one.
6. **Level 2 is decided in the first round.** A right first answer scores 1.00 at round 0; no run that
   started wrong recovered. Our messages moved those runs from a crash (0.30) to a kernel that runs
   with values in the wrong positions (0.50), not to a solve.
7. **Three of four samples are wasted after the first round.** Under `located`, all four samples were
   the same kernel in 93 of 100 repair rounds. Level 1 produced the identical kernel in every sample of
   every run for its first six rounds.
8. **Our own metric was wrong once.** `report.py` counted a repair step as "moved" if the mistake
   differed from the round before, which scored the loop in point 3 as progress on every step. It now
   counts a return to an earlier mistake separately.

## What these numbers do not show

- **Few runs.** `enriched` has one run per level on this hardware, `directed4` two on level 4,
  `directed5` two on level 3. With level 4 ranging from 0.30 to 0.75 under a single checker,
  nothing here is a rate.
- **Development and evaluation are not separated.** Each message was written from a run and tried on
  the next. The level 4 solve came on the first run after two of its messages were written, and the
  run that followed it did not repeat it. We did not get to a set of fresh runs on a frozen checker.
- **Some messages describe a method**, such as how to split M and N into chunks, at the level of
  detail of the repo's own message for K. None contains kernel code. The line between naming a
  fault and giving the answer is a judgement, and ours is in `agent.py` to be checked.
- **The pre-check is untested on the chip.** It passes its offline tests, against a small imitation
  of the NKI library. In the `directed5` runs here it never fired (0 of 72 attempts), because none of those
  kernels had a fault of the kind it looks for. The progress in those runs comes from the other
  `directed5` messages.
- **Runs were interrupted.** The seat was shared. One `directed5` run was stopped from outside one
  round into level 4, a second-seat run after one round, and the five-run `located` log was deleted
  under the running process and recovered through its open file handle; its last rounds may be
  missing. Rounds took 47 to 110 s alone and up to 265 s when another job shared the chip.
- **Simulator only.** Scores come from `nki.simulate` on the CPU.

## Reproduce

```bash
cd /workspace/projects/02-kernel-agent
python nkibench.py --selftest
python agent.py --levels 4 --rounds 8 --samples 4 --context 8192 --feedback directed4 --log l4.jsonl
python agent.py --levels 4 --rounds 8 --samples 4 --context 8192 --feedback enriched  --log l4.jsonl
python report.py l4.jsonl                 # one row per run and level
python taxonomy.py --log l4.jsonl         # mistakes grouped and counted
```

Run one agent at a time on a seat.
