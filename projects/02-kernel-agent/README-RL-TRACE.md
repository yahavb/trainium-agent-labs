# Verifier-grounded RL trace experiment

This is an additive experiment for `projects/02-kernel-agent`. It reuses the existing
`agent.py` model client and `nkibench.py` verifier. Leave both files unchanged.

## Files to replace

Replace the entire contents of:
- `projects/02-kernel-agent/rl_trace_agent.py`
- `projects/02-kernel-agent/README-RL-TRACE.md` (this file)

Leave `agent.py`, `nkibench.py`, and `summarize_rl_trace.py` unchanged.

## Why the NKI API prompt is explicit

The NKI checker errors are evidence about the generated code, not an environment failure:

- `No module named 'nl'`: `nl` is an alias created by `import nki.language as nl`, not a module.
- `MemoryRegion object is not callable`: `nl.sbuf`, `nl.psum`, and `nl.shared_hbm` are buffer values,
  passed as `buffer=nl.sbuf`, not function calls.
- `NkiTensor.reshape() takes 2 positional arguments but 4 were given`: NKI's tensor reshape
  expects a single shape tuple if used, such as `x.reshape((P, F1, F2))`, not
  `x.reshape(P, F1, F2)`.

For level 2, the updated prompt additionally tells the model the exact index mapping for transpose
and recommends the established NKI approach using `nl.affine_range`, `nl.ds`, and `nisa.tensor_copy`.
It tells the model to avoid NKI tensor `reshape`/`transpose` for this task. The repository's known
reference kernel is `reference_level2.py`; you can validate the checker against it with:

```bash
python nkibench.py --level 2 --check reference_level2.py
```

That is a diagnostic check, not an agent result.

## Run from the correct directory

```bash
cd /workspace/My_Working_Dir/trainium-agent-labs/projects/02-kernel-agent
python nkibench.py --selftest
python -m py_compile agent.py nkibench.py rl_trace_agent.py
```

The self-test does not require the model endpoint. The agent run does. Check whether your seat
already configured the endpoint:

```bash
echo "$KERNEL_AGENT_BASE_URL"
echo "$KERNEL_AGENT_MODEL"
```

Use the actual endpoint supplied by your environment; do not assume localhost is serving the model.

## First run

Use a fresh log so it is easy to compare with earlier attempts:

```bash
python rl_trace_agent.py \
  --level 2 --rounds 8 --samples 1 --context 8192 \
  --log rl_level2_v3.jsonl
```

Then try tiled matmul:

```bash
python rl_trace_agent.py \
  --level 4 --rounds 8 --samples 1 --context 8192 \
  --log rl_level4_v3.jsonl
```

Optional background run:

```bash
nohup python rl_trace_agent.py --level 2 --rounds 8 --samples 1 --context 8192 \
  --log rl_level2_v3.jsonl > rl_level2_v3.log 2>&1 < /dev/null &
tail -f rl_level2_v3.log
```

Summarize JSONL logs with the existing companion script:

```bash
python summarize_rl_trace.py rl_level2_v3.jsonl rl_level4_v3.jsonl
```

## How the controller works

The controller uses contextual UCB over four prompt/repair strategies:
- `direct`: concise direct generation;
- `contract_trace`: explicit shape/dtype/output contract;
- `counterexample`: focus on ragged dimensions and boundaries;
- `repair_diagnosis`: use the latest checker feedback to make a minimal repair.

The context includes kernel level and the failure category returned by the verifier. The policy updates
strategy values using each attempt's reward.

`R = 0.85 * R_kernel + 0.10 * R_trace + 0.05 * R_evidence`

- `R_kernel`: original verifier reward; this dominates.
- `R_trace`: deterministic coverage score for a compact structured engineering record, not its length.
- `R_evidence`: based on actual verifier feedback, never the model's assertion that its code works.

Each JSONL row stores the action, structured trace, code, actual feedback, reward parts, failure category,
and generation latency. Traces are short auditable implementation notes, not private chain-of-thought.

## Baseline comparison

Compare the original `agent.py` and this controller using the same model, endpoint, level, context,
and generation budget. Run multiple independent trials. Report:
1. first-attempt reward and fully verified-kernel rate;
2. attempts to first fully verified kernel;
3. mean verifier reward under a fixed call budget;
4. rates of bad imports, unsupported API use, parse errors, and numerical mismatches;
5. repeated-code and repeated-failure rates;
6. trace format validity and coverage;
7. wall-clock time and token/call cost where available.

A structured trace is not proof of correctness. The verifier is authoritative.

## Limitations

- This is online policy learning over prompt strategies, not weight-level PPO/GRPO fine-tuning.
- `nkibench.py` uses the NKI simulator for functional checks. It does not establish real Trainium
  latency or downstream model speedups.
- The UCB controller needs repeated trials; do not draw conclusions from one run.
