# Heat-rod agent: tool-first solving and all-candidate feedback (results, 2026-10-10)

Qwen3-8B on one Trainium2 chip (seat-266, vLLM-neuron 0.24, TP=2, 4 sequences), 4 samples per round,
up to 4 rounds. Level 0 and level 1 problems at seeds 0, 1, 2 (L and k change with the seed). Every attempt is in
`results/*.jsonl`; `python analyze.py` regenerates every table here. The model was not changed.
The reward (`pdecheck.check`) was not changed; its answer-line parser now strips markdown bold.

## Headline

| arm | what changes | problems | solved | rounds to solve | tokens / solve | s / solve | solved / h |
|---|---|---|---|---|---|---|---|
| as shipped (earlier today) | repo as cloned | level 1, seed 0 | 2/3 | 1, 1, >4 | n/a | 484 | 7.4 |
| **T3d: derived eigenfunctions** | model writes the problem's structure (k, L, u(x,0), end types); SymPy derives eigenfunctions, rates and coefficients | level 1, seeds 0–2 | **9/9** | all 1 | **240** | **12** | **295** |
| T3d on level 0 | same mode | level 0, seeds 0–2 | **9/9** | all 1 (all 4 samples 1.0) | 241 | 12 | 295 |
| T3 (first version of T3d) | same, model also wrote an unused eigenfunction field, long feedback | level 1, seeds 0–2 | 9/9 | all 1 | 304 | 15 | 239 |
| T2: model chooses eigenfunction | model writes the structure plus X_n(x); SymPy does the rest | level 1, seeds 0–2 | 4/9 | 2, >4, 1, >4, >4, 1, >4, >4, 1 | 1960 | 107 | 34 |
| T2c: T2 + compact feedback | PASS/FAIL per check, one-line diagnosis, root cause; no long expression | level 1, seeds 0–2 | 5/9 | 2, 3, 1, >4, >4, 1, >4, >4, 1 | 1517 | 83 | 44 |
| T2A: T2 + four different prompts | best-ever, "families already tried", "check X_n at both ends", fresh | level 1, seeds 0–2 | 5/9 | 2, >4, 1, >4, >4, 1, 2, >4, 1 | 1452 | 81 | 44 |
| B: baseline free-form | today's loop + logging, retries, parse fixes, best-ever memory | level 1.3, seeds 0–2 | 0/3 | >4, >4, >4 | — | — | 0 |
| **F2: all-candidate feedback** | next round built from all four candidates: term table, keep/fix line, 4 prompt variants | level 1.3, seeds 0–2 | **2/3** | 2, 2, >4 | 6608 | 321 | 11 |

n = 1 run per problem and seed (9 problems per level-1 arm, 3 per 1.3 arm). Times are wall-clock with
one agent on the server.

## What the numbers say

1. **Moving the mathematics to SymPy is what made the agent fast and reliable.** With the eigenfunction
   family derived from the boundary types (left/right ∈ {dirichlet, neumann}), every level-1 problem at
   every seed is solved in round 0, at ~12 s and ~240 output tokens per problem. The as-shipped loop
   spent up to 1200 tokens per sample (5 of 8 level-1.1/1.2 samples were cut off there) and 215–314 s
   per round.
2. **The one thing the model could not do was choose the eigenfunction for an insulated end.** When it
   had to write X_n(x) (T2), it used the both-ends-fixed family sin(nπx/L) on 1.1/1.2 and never found
   sin((2n−1)πx/(2L)) without help. Level 1.3 (both ends fixed) solved in round 0 at every seed.
3. **Rounds 1–3 repeated the same prompt in T2 because the candidates were identical**, not because
   information was dropped: all four specs were byte-identical every round at every seed (temperatures
   0.6–1.0). With four different prompts (T2A) the model produced 2–3 different families per round,
   all of them still wrong for the insulated end. A quick test showed unseeded sampling does vary on this
   server; short JSON replies are simply very confident.
4. **All-candidate feedback rescues the free-form loop.** On 1.3, the baseline repeated the same 0.6
   answer (right coefficients, wrong exponents) for four rounds at every seed. Building round 2 from all
   four candidates solved seeds 0 and 2 in round 2. The solving prompts were V3 (best-ever plus "your
   starting shape already matches: keep every coefficient, only the exponents are wrong") and V2 (the
   per-term table plus "change only what the checker flagged").
5. **A splice of the candidates would have solved the same two runs in the same round.** It is logged
   (`source="splice"`) and not counted as a model solve.
6. **Compact feedback helps a little where the model must still decide** (T2c: first solve of 1.2 in
   that mode, round 3), and it no longer sends the solver's long expression back to the model.

## What we did not show

- n = 1 per problem and seed. The level-1 arms are 9 problems each; the 1.3 head-to-head is 3 per arm.
- No baseline run of 1.1/1.2 at seeds 1–2 (time); the free-form comparison is on 1.3 only.
- The "worked example anchors the exponents" hypothesis was not tested directly.
- Levels with Robin ends, non-zero ends or sources (where method selection and SciPy would matter) were
  not built.

## Incidents during the runs

- **A test request with a per-request `seed` crashed the vLLM-neuron engine** at 19:37 UTC
  (`getNewGenerator` not implemented for the Neuron device). The server was restarted with `./serve.sh`;
  runs that hit the dead server were moved to `results/aborted/` and re-run. Do not send `seed`.
- The solver's process pool could deadlock at exit; `agent.py` now stops its workers and exits
  explicitly.

## Reproduce

```bash
cd /workspace/projects/01-heat-rod-pde
python agent.py --level 1 --all --seed 0 --mode full --temps 0.6,0.8,1.0,1.0 --arm T3d        # derived eigenfunctions
python agent.py --level 1 --all --seed 0 --mode ansatz --temps 0.6,0.8,1.0,1.0 --arm T2c      # model chooses X_n
python agent.py --level 1 --sub 3 --seed 0 --arm B                                            # baseline free-form
python agent.py --level 1 --sub 3 --seed 0 --aggregate all --splice --arm F2                  # all-candidate feedback
python analyze.py
python solver_tool.py --selftest
```

Code: `solver_tool.py` (spec parser, SymPy solver, boundary-type → eigenfunction table, consistency
note), `aggregate.py` (per-term verdicts, dedupe, prompt variants, splice), `agent.py` (`--mode`,
`--aggregate`, `--splice`, `--temps`, `--arm`, retries, full logging), `analyze.py`.

## Final benchmark: Idea 2 + A + B, free-form (LLM writes the answer), seed 0 — snapshot at 21:55 UTC, run still in progress

| problem | baseline best | final best | baseline mean | final mean | baseline rounds | final rounds | final tokens | baseline time | final time |
|---|---|---|---|---|---|---|---|---|---|
| level0.1 | 1.0 | not run | 1.00 | not run | 1 | not run | not run | n/a | not run |
| level1.1 | 1.0 | 1.0 | 0.25 | 0.25 | 1 | 1 | 4624 | 215 s | 215 s |
| level1.2 | 1.0 | 0.6 | 0.50 | 0.15 | 1 | >1 | 5320 | 314 s | 252 s |
| level1.3 | 0.8 | not run | 0.45 | not run | >4 | not run | not run | 439 s | not run |

Final-run logs: `results/FINAL-*.jsonl`, `results/FINAL-L1-s0.txt`. Level 0 had not started at snapshot time.

## Best result per problem across all runs (seed 0)

| problem | LLM writes the answer (free-form) | LLM fills a form, SymPy writes the answer |
|---|---|---|
| 0.1 | solved r1 (original agent) | solved r1 (fast mode, all seeds) |
| 0.2, 0.3 | not run | solved r1 (fast mode, all seeds) |
| 1.1 | solved r1 (original; final Idea 2 A+B) | solved r1 (fast mode) |
| 1.2 | solved r1 (original agent) | solved r1 (fast mode) |
| 1.3 | solved r2 (Idea 2, seeds 0 and 2; A+B and A, B alone at seed 1) — original never solved | solved r1 (fast mode) |
