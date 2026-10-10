# Kernel agent: what the checker's feedback is worth

Seat 17 · project 2 · **interim note, 10 Oct 13:05 EDT.** More runs are in progress; the tables
here cover the runs that have finished and will be regenerated when the rest do.

## What we ran

The repo's kernel agent, unchanged except for the checker's feedback. Model: Qwen3-8B on the seat's
Trainium chip, started with `./serve.sh`, thinking off, context 8192, 4 samples per round. The model,
its settings and the task prompt were not changed. Checker: `nkibench.py` layer 1, which simulates the
kernel on the CPU. Nothing here was timed on the device.

One variable changes between runs: what the checker says when a kernel fails.

| feedback | what the model is told |
|---|---|
| `enriched` | the repo's original: the exception plus a fix instruction |
| `located` | the same, plus the model's own failing line quoted back |
| `directed` | plus the two real shapes in a failed copy, and where a misplaced name really lives |
| `directed2` | plus the real argument list of the function it misused, messages for four errors that had no usable advice, and one corrected message |

Every `directed` and `directed2` message was written after seeing the model make that exact mistake
on this seat. The log line that prompted each one is in the commit history.

## Results so far (one run per row)

| level | feedback | best score | rounds | repair steps that fixed the named mistake |
|---|---|---|---|---|
| 1 average pooling | located | 0.30 | 8 | 5 of 7 |
| 1 average pooling | directed | 0.30 | 8 | 6 of 7 |
| 2 transpose | directed | 1.00 (round 0) | 1 | n/a |
| 3 matmul, one tile | directed | 0.30 | 6 | 2 of 5 |
| 4 matmul, tiled | directed | **0.75** | 6 | 2 of 5 |

Full table: [`RESULTS.md`](RESULTS.md). Mistakes grouped and counted:
[`TAXONOMY-directed.md`](TAXONOMY-directed.md), [`TAXONOMY-located.md`](TAXONOMY-located.md).
Every attempt with its score: [`logs/`](logs/).

## What the logs show

1. **A named fix is taken; a list of names is not.** The model wrote `nisa.multiply`. Under `located`
   the checker listed the first 25 names in `nki.isa`: the model repeated the mistake for three rounds
   and then invented `nisa.scalar_mul`. Under `directed` the checker said "write `nl.multiply`": fixed
   in the next round.
2. **Level 4 reached 0.75, 2 of 4 shapes.** The repo records 0.62 on five of five runs. The model
   chunked the K dimension correctly, then hit `stationary free dimension 256 exceeds 128`, for which
   the checker had no advice, and repeated it four rounds. `directed2` adds that advice.
3. **One of our own messages caused a cycle.** On level 3 the model used one tile for two inputs of
   different sizes. Our message offered two fixes, one of them impossible in that case, and the model
   alternated between them for six rounds. `directed2` replaces it.
4. **Extra samples only help in the first round.** The four samples were identical in 28 of 31 repair
   rounds. In first rounds they differed on levels 2 to 4 (3 or 4 distinct answers) and were identical
   on level 1 in four of four runs. Level 2 was solved at round 0 in one run and not in another, with
   the same prompt.
5. **Two runs on one seat cost four times the wall-clock.** Alone, a round takes 52 to 94 s (5.4
   tokens/s per sequence). With a second agent on the same model, most rounds took 190 to 265 s.

## Where the prompt goes (`directed` run, mean per prompt)

| prompt | prompt tokens | answer tokens | made of (characters) |
|---|---|---|---|
| first | 716 | 300 | API card 1581, task wording 506, reference 399 |
| repair | 538 | 341 | previous kernel 995, checker feedback 518, wording 201 |
| repair with ledger | 823 | 455 | previous kernel 1351, ledger 473, feedback 432, wording 266 |

No answer was cut off: all 84 attempts in that run ended with `finish_reason=stop`.

## What these numbers do not show

- **One run per row.** First-round answers vary between runs on levels 2 to 4, so a single score can
  move. Three runs each of `enriched` and `directed2` on levels 3 and 4 are queued.
- **No baseline on this seat yet.** The `enriched` run is queued; the 0.62 above is the repo's figure
  from different hardware.
- **Level 1 is not close.** The model computes a matrix product of each window with itself where a sum
  is needed. Fixing one error per round does not change that approach.
- **The level 4 message in `directed2` describes the tiling method** in words, at the same level of
  detail as the repo's own message for the K dimension. It contains no code.
- **Simulator only.** Scores come from `nki.simulate` on the CPU.

## Reproduce

```bash
cd /workspace/projects/02-kernel-agent
python nkibench.py --selftest
python agent.py --all --rounds 8 --samples 4 --context 8192 --feedback directed --log directed.jsonl
python report.py directed.jsonl          # the table above
python taxonomy.py --log directed.jsonl  # mistakes grouped and counted
```

Run one agent at a time on a seat.
