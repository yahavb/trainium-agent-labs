# Project 1 submission: where everything is

Hack the Chip, seat-266, 2026-10-10. Heat-rod agent on Qwen3-8B / Trainium2.
All paths are inside the pod unless marked "laptop".

## The three hand-in items

| item | file |
|---|---|
| 1. The checker and its reasoning | `projects/01-heat-rod-pde/pdecheck.py` + `results/CHECKER.md` |
| 2. The attempt log | `results/*.jsonl` (one file per run) + `results/*.txt` (console output per run) |
| 3. The one-page note | `results/RESULTS.md` |

## Headline results

| arm | what it is | solved | time per solve |
|---|---|---|---|
| as shipped | original agent, level 1, seed 0 | 2/3 | 484 s |
| **T3d, level 0** | fast mode: LLM fills in a form, SymPy does the maths | **9/9** | 12 s |
| **T3d, level 1** | same | **9/9** | 12 s |
| T2 / T2c / T2A | LLM also chooses the waves | 4–5/9 | 81–107 s |
| B | free-form baseline, level 1.3 | 0/3 | — |
| **F2** | free-form + all-candidate feedback, level 1.3 | **2/3** | 321 s |

Seeds 0, 1, 2; one run per problem and seed. Regenerate the table: `python analyze.py`.

## Result files, by arm

All in `projects/01-heat-rod-pde/results/`. `<run id>` = date-time-pid.

| arm | mode / flags | logs | console |
|---|---|---|---|
| as shipped | original code | `asshipped-attempts.jsonl` | `asshipped-run.log` |
| T3d (level 1) | `--mode full` (derived eigenfunctions) | `T3d-<run id>.jsonl` ×3 | `T3d-s0/1/2.txt` |
| L0-T3d (level 0) | `--mode full` | `L0-T3d-<run id>.jsonl` ×3 | `L0-T3d-s0/1/2.txt` |
| T3 | first version of T3d | `T3-<run id>.jsonl` ×3 | `T3-s0/1/2.txt` |
| T2 | `--mode ansatz` (LLM chooses waves) | `T2-<run id>.jsonl` ×3 | `T2-s0/1/2.txt` |
| T2c | T2 + compact feedback | `T2c-<run id>.jsonl` ×3 | `T2c-s0/1/2.txt` |
| T2A | T2 + `--aggregate all` (4 prompt variants) | `T2A-<run id>.jsonl` ×3 | `T2A-s0/1/2.txt` |
| B | free-form baseline, 1.3 | `B-<run id>.jsonl` ×3 | `B-13s0/1/2.txt` |
| F2 | `--aggregate all --splice`, 1.3 | `F2-<run id>.jsonl` ×3 | `F2-13s0/1/2.txt` |
| aborted | runs that hit the crashed server (19:37–19:50 UTC) | `aborted/` | `vllm-crash-1937.log` |

`queue.sh`, `queue3.sh` and their `.log` files are the scripts that ran the arms in order.

## Code

In `projects/01-heat-rod-pde/`:

| file | status | what |
|---|---|---|
| `agent.py` | changed | the loop; `--mode`, `--aggregate`, `--splice`, `--temps`, `--arm`, retries, full logging |
| `pdecheck.py` | changed (reward unchanged) | checker; bold-answer fix, `prompt_of(style=...)` |
| `tool_calc.py` | changed | COMPUTE parsing fix |
| `solver_tool.py` | new | spec parser + SymPy solver + boundary-type → eigenfunction table |
| `aggregate.py` | new | all-candidate feedback: per-term table, prompt variants, splice |
| `analyze.py` | new | results tables from `results/*.jsonl` |

Full diff against the original repo: `/workspace/handoff/project1-changes.diff`.

## Reproduce

```bash
cd /workspace/projects/01-heat-rod-pde
python agent.py --level 1 --all --seed 0 --mode full --temps 0.6,0.8,1.0,1.0 --arm T3d
python agent.py --level 1 --sub 3 --seed 0 --aggregate all --splice --arm F2
python analyze.py
```

Do not send a per-request `seed` to the model server: it crashes vLLM-neuron 0.24.

## Background and planning docs

| what | where |
|---|---|
| Findings (18 topic files) | `/workspace/findings/README.md` |
| Implementation plan (11 files + prototypes) | `/workspace/findings/plan/README.md` |
| Plan as a web page | https://claude.ai/artifact/RfCm2M8c32aVZa5UGegsxa (private) |

## Packaged copies (to take off the pod)

| file | contents |
|---|---|
| `/workspace/handoff/project1-submission.tgz` | code + `results/` (this file included) |
| `/workspace/handoff/project1-changes.diff` | code changes as a diff |
| `/workspace/handoff/findings.tgz` | findings folder (packed by another session at 19:37, before the plan was finished) |

From a laptop terminal with the workshop credentials pasted in:

```bash
kubectl cp seat-266:/workspace/handoff/project1-submission.tgz ./project1-submission.tgz
tar xzf project1-submission.tgz
```
