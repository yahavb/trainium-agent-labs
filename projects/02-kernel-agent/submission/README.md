# kernel-agent

An agent that writes tiled NumPy kernels (Stage A) for a 10-level op ladder, using a model that sees
at most 8192 tokens per call (input + output). The rules live in the verifier, not in the prompt: the
model writes freely, the harness finds what is wrong, and the agent translates that verdict into one
concrete change. A second stage makes verified kernels cheaper (FLOPs, HBM traffic, instructions)
without ever making them wrong.

Results, design and every experiment: `notes/notes.tex` (E1–E23). Team plan and status: `TEAM.md`.
Deliverables: agent `kagent/`, harness `kagent/harness.py` (tolerance = proven float32 error bound),
eval set `EVALSET.md`, failure taxonomy `TAXONOMY.md`, token instrumentation `scripts/plot_tokens.py`.

## Reproduce

**1. Locally, no model (about 2 minutes)**

```bash
uv sync
uv run pytest -q                                   # expect: 59 passed
uv run python -m kagent.harness 4 kernels/hand/l4_rmsnorm.py          # PASS, within the float32 bound
uv run python -m kagent.harness 3 kernels/broken/l3_overwrite_per_tile.py --fixes   # names the bug
uv run python -m kagent.agent --levels 1 2 --scripted tests/scripts/demo.json --out runs/demo.jsonl
uv run python scripts/plot_tokens.py runs/demo.jsonl                  # input tokens per attempt, by section
uv run python scripts/list_cases.py                                   # regenerates EVALSET.md
```

**2. With the model, on a seat pod** (Qwen3-8B on vLLM, started by `/workspace/serve.sh`)

```bash
# AWS credentials from the workshop channel in this terminal, then:
aws eks update-kubeconfig --name hack-hyd --region ap-south-2
export POD=seat-NNN
kubectl exec $POD -c app -- bash -c 'curl -s localhost:8000/health -o /dev/null -w "%{http_code}\n"'   # 200
scripts/sync.sh                                    # push tracked files to /workspace/stageA
scripts/launch.sh ours 1 2 3 4                     # background; 8 attempts, 5 runs per level
scripts/launch.sh naive 1 2 3 4                    # ablation: the checker's report sent back verbatim
kubectl exec $POD -c app -- bash -c 'tail -n 20 /workspace/stageA/runs/*.log'
scripts/fetch.sh                                   # copies the logs to results/$POD/
```

**3. Analysis**

```bash
R=results/seat-130
uv run python scripts/analyze.py ours=$R/ours-L1234-seat-130.jsonl naive=$R/naive-L1234-seat-130.jsonl --recheck
uv run python scripts/taxonomy.py ours=$R/ours-L1234-seat-130.jsonl naive=$R/naive-L1234-seat-130.jsonl --md TAXONOMY.md
uv run python scripts/plot_tokens.py $R/ours-L1234-seat-130.jsonl 4     # one run, all levels
# efficiency stage on the verified kernels (on the pod), then the before/after figure:
python3 scripts/optimize.py --from runs/ours-L1234-seat-130.jsonl --rounds 3 --out runs/opt.jsonl
uv run python scripts/opt_report.py $R/opt2-live.jsonl
```

`main` is agent v2 (version `e22ccab872`, 8 attempts), the version every benchmark arm so far ran.
Branch `v3` (version `ea9befddac`) holds the next iteration (notes E23): structural directives may
rewrite loops, one remedy per directive, crash-specific fixes, 12 attempts.

Expected (seat-130, version `e22ccab872`): `ours` L1 5/5, L2 4/5, L3 5/5, L4 1/5; the optimiser
accepts 5 of 11 rewrites, with up to 79x fewer instructions (`notes/fig-opt2.png`).

## Things that will bite you

- **Never send `seed`** in a request: the Neuron build of vLLM kills its engine (notes E5).
- The server's 8192-token context is **input + output**; the client sizes the answer from an exact
  `/tokenize` count and never parses a truncated reply.
- The server is near-deterministic at `temperature=0.6` (E10): repeats use five task wordings,
  otherwise five runs are one run five times.
- Compare runs only at the same **code version** (hash of `kagent/*.py`, logged with every result;
  `analyze.py` flags mixed versions). `scripts/` is outside the hash on purpose.
- Replies are cached on disk by exact request; a restart replays identical requests for free and
  marks them `cached` (their latency is the original call's).
- `main` is protected: changes go through pull requests.

## Layout

| path | what |
|---|---|
| `kagent/agent.py` | the repair loop: prompts, reply gate, ledger, holdout confidence, token accounting |
| `kagent/harness.py` | verify a kernel: rules, numerics against the proven bound, diagnosis, cost ledger |
| `kagent/static.py`, `kagent/guard.py` | rule checks: AST scan, and a runtime numpy proxy that records violations by line |
| `kagent/levels.py` | 10 levels: reference, bound, hostile dev and holdout cases, level-specific diagnosis |
| `kagent/translate.py`, `kagent/rules.py` | verdict -> one instruction; banned constructs and their fixes |
| `kagent/client.py` | model client: exact token counts, budget, finish_reason, cache |
| `scripts/` | sync/launch/fetch, analysis, taxonomy, token plot, efficiency optimiser and report |
| `kernels/hand/`, `kernels/broken/` | correct kernels (must pass) and deliberate or live bugs (must be named) |
| `upstream-nki/` | Stage B: the upstream NKI agent's baseline and feedback experiments (PR #1) |
| `notes/notes.tex` | design and experiment log |
