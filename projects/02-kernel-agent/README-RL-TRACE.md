# Verifier-grounded RL trace experiment

This is an additive experiment for `projects/02-kernel-agent`. It reuses the original
`agent.py` model client and `nkibench.py` verifier. It does not modify either file.

## What changed

- `rl_trace_agent.py` adds an online contextual UCB bandit over four generation/repair strategies.
- Each attempt logs a short structured engineering trace, candidate code, the actual verifier
  feedback, and reward components to JSONL.
- The verifier remains the source of truth. A model saying a kernel is correct does not count.
- The prompt now states the valid NKI imports verbatim. In particular, **`nl` is an alias, not a
  module**: use `import nki.language as nl`, never `import nl`. Likewise `nl.sbuf`, `nl.psum`,
  and `nl.shared_hbm` are buffer values, not functions; pass them as `buffer=nl.sbuf`, etc.

This is online RL over prompt strategy selection, not weight-level RL fine-tuning. The model is
served through an inference endpoint, so its weights are not updated by this script. The traces
are short, visible, structured engineering notes; the script does not request private chain of
thought or reward verbosity.

## Run from the correct directory

The files `agent.py`, `nkibench.py`, and `rl_trace_agent.py` are inside
`projects/02-kernel-agent`, not at the repository root:

```bash
cd /workspace/My_Working_Dir/trainium-agent-labs/projects/02-kernel-agent
python nkibench.py --selftest
python -m py_compile agent.py nkibench.py rl_trace_agent.py
```

The self-test does not require the model endpoint. The agent run does. In the seat pod, the
environment normally sets `KERNEL_AGENT_BASE_URL` and `KERNEL_AGENT_MODEL`; check with:

```bash
echo "$KERNEL_AGENT_BASE_URL"
echo "$KERNEL_AGENT_MODEL"
```

If needed, set the URL to the actual endpoint used by your pod. Do not assume `localhost:8000`
unless the model server is running in the same pod and listening on that port.

## First run

Start with the transpose task, one sample per round to limit endpoint load:

```bash
python rl_trace_agent.py \
  --level 2 --rounds 8 --samples 1 --context 8192 \
  --log rl_level2.jsonl
```

Then test tiled matmul:

```bash
python rl_trace_agent.py \
  --level 4 --rounds 8 --samples 1 --context 8192 \
  --log rl_level4.jsonl
```

For a long session, use the same background pattern as the original README:

```bash
nohup python rl_trace_agent.py --level 2 --rounds 8 --samples 1 --context 8192 \
  --log rl_level2.jsonl > rl_level2.log 2>&1 < /dev/null &
tail -f rl_level2.log
```

Summarize logs with the companion script:

```bash
python summarize_rl_trace.py rl_level2.jsonl rl_level4.jsonl
```

## Strategy arms

- `direct`: short direct generation.
- `contract_trace`: explicit shape/dtype/output and partial-tile contract.
- `counterexample`: emphasis on ragged dimensions and boundary behavior.
- `repair_diagnosis`: use recent verifier output to make one minimal repair.

The policy conditions on the current level and the failure category returned by the checker.
It uses UCB exploration and updates action values from each attempt's observed reward.

## Reward and logs

`R = 0.85 * R_kernel + 0.10 * R_trace + 0.05 * R_evidence`

- `R_kernel`: original `agent.grade()` reward (correctness remains dominant).
- `R_trace`: deterministic field-coverage score for the structured trace, not its length.
- `R_evidence`: score based on the checker pass/failure category, never on self-reported success.

Each JSONL row records the prompt strategy, failure category, trace, candidate code, verifier
feedback, reward components, and elapsed generation time.

## Baseline comparison

Use the same model, endpoint, kernel level, context, and total generation budget for baseline
`agent.py` and `rl_trace_agent.py`. Run several independent trials. Report:

1. first-attempt reward and fraction reaching full verifier reward 1.0;
2. attempts until a fully verified kernel is found;
3. mean verifier reward at a fixed call budget;
4. rates of parse errors, invalid imports/API use, and numerical mismatches;
5. repeated-code / repeated-failure rate;
6. trace-format validity and coverage;
7. elapsed time and token/call cost where available.

A valid trace is not evidence that a kernel is correct. `nkibench.py` is authoritative.

## Current limitations

- This is an initial online policy-learning experiment, not PPO/GRPO fine-tuning of Qwen or gpt-oss.
- The existing verifier uses the NKI simulator for functional checks. It does not establish real
  Trainium latency or downstream generative-model speedups.
- Do not evaluate based on one run; the UCB estimates need repeated trials.
