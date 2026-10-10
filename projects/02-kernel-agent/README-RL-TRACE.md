# Verifier-grounded RL trace agent

This is an additive experiment for the existing `projects/02-kernel-agent` implementation.
It does not modify `agent.py` or `nkibench.py`.

## What it does

- Reuses the original `agent.py` model endpoint, code extractor, and `grade()` verifier.
- Requests a compact, visible engineering trace in a fixed JSON schema before the kernel code.
  It does **not** request or reward hidden/private chain-of-thought.
- Runs the unchanged `nkibench.grade`/NKI simulator path as the source of truth.
- Uses a per-level, failure-state contextual UCB bandit to select among four prompt strategies:
  `direct`, `contract_trace`, `counterexample`, and `repair_diagnosis`.
- Logs each attempt's code, structured trace, verifier result, reward components, chosen action,
  and elapsed time to JSONL for analysis.

**Important distinction:** this is agent-level online RL (policy learning over prompt/repair
strategies). It does not update Qwen/gpt-oss weights. The repo currently calls an inference
endpoint, and cannot backpropagate into that endpoint's model. This is the practical first
experiment you can run against the existing implementation. True RL fine-tuning would require
a trainable model checkpoint, policy-training code, and suitable accelerator/training setup.

## Setup

Use the same environment and endpoint as the original agent. From this directory:

```bash
python nkibench.py --selftest
python -m py_compile agent.py nkibench.py rl_trace_agent.py
```

Set the model endpoint (the original repo's seat environment may already set it):

```bash
export KERNEL_AGENT_BASE_URL=http://localhost:8000/v1
export KERNEL_AGENT_MODEL=Qwen/Qwen3-8B
```

Run a small first experiment:

```bash
python rl_trace_agent.py --level 2 --rounds 8 --samples 1 --context 8192 --log rl_level2.jsonl
```

Then test a harder level:

```bash
python rl_trace_agent.py --level 4 --rounds 8 --samples 1 --context 8192 --log rl_level4.jsonl
```

For the shared endpoint, use the path expected by your deployment:

```bash
export GPTOSS_BASE_URL=https://YOUR-ENDPOINT
python rl_trace_agent.py --level 4 --rounds 6 --samples 1 --context 8192 --path /agg/v1 --log rl_gptoss.jsonl
```

For long runs in a remote shell, follow the repo's existing `nohup ... > run.log 2>&1 < /dev/null &` advice.

## Reward

The online controller uses:

`R = 0.85 * R_kernel + 0.10 * R_trace + 0.05 * R_evidence`

- `R_kernel`: the original graded verifier reward; this dominates.
- `R_trace`: deterministic score for a valid JSON trace covering the required engineering fields,
  with explicit shape/correctness checks (and tile/memory checks for matmul levels). It penalizes
  unparseable or absent traces and does not reward verbosity.
- `R_evidence`: derived from the actual checker category and pass/fail state, never from the model's
  claim that its own kernel is correct.

The UCB controller learns action values separately by level and observed failure state. Since a
kernel can receive partial credit from the existing verifier, the controller can learn before a
level is completely solved.

## What to compare

Run the baseline `agent.py` and this controller on the same level, model, context length, round
budget, and seed/repeat schedule. Report:

1. first-attempt verifier reward and parse/rule/run/correct rates;
2. fraction of runs reaching reward 1.0;
3. attempts to first fully verified kernel;
4. mean verifier reward under a fixed generation budget;
5. duplicate-code / repeated-failure rate;
6. calls, elapsed time, and tokens if the endpoint returns usage;
7. trace validity and coverage, plus whether trace contents correspond to the actual verifier feedback;
8. generalization on held-out shapes (add these to `nkibench.py` rather than tuning only to its shipped cases).

Do not interpret one run as evidence: use multiple seeds/runs. The controller is intentionally small;
with few attempts it may not have enough data to reliably identify the best action.

## Caveats

- The extra trace instruction can increase prompt length and may reduce code quality on models that
  struggle with structured output. The four arms are designed to let the controller learn which
  framing works better, but test against the original baseline.
- The script currently uses the original checker and CPU-side NKI simulation. This is not a real
  Trainium latency benchmark. Hardware speedups require the later on-device timing layer.
- The script expects the repo's NKI SDK/simulator environment for grading generated NKI kernels.
