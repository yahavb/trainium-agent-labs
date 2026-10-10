"""Feedback v8 (E-div): v7, plus a different prompt per sample, because the seat server decodes greedily.

Measured on the seats (2026-10-10, 14:30): the vLLM-Neuron server returns byte-identical completions
for identical requests whatever the sampling settings -- 4 of 4 identical at temperature 0.7, and the
same with n=4 in one request; a `seed` field returns HTTP 500 and kills the engine. In the logs, 89 of
106 baseline rounds had four identical samples, and v7's level-4 runs were one trajectory replayed five
times. So --samples 4 bought one sample per round, and --repeat 5 one run.

The one change: sample 1 keeps the prompt byte-for-byte, so v7's own trajectory is always among the
candidates; sample k >= 2 gets one line appended to the user message, "(attempt k of n, run r)".
Everything else is v7: messages, grading, best-of-round (ties still go to the earliest sample, i.e.
sample 1), ledger, verdicts. Run it exactly like feedback_v7.py, with the exports in V7.md:

    python3 feedback_v8.py --level 4 --rounds 8 --samples 4 --context 8192 --repeat 5 \\
        --log Ediv_L4.jsonl --verdicts verdicts_div_L4.jsonl

The token split in attempts.jsonl is computed from sample 1's prompt; samples k >= 2 carry about ten
more prompt tokens than it says.
"""
import concurrent.futures as cf

import feedback_v7 as v7

agent = v7.agent


def variant(prompt, k, n, run):
    """Sample k's prompt: sample 1 unchanged, the others with one line naming the attempt."""
    return prompt if k == 1 else f"{prompt}\n\n(attempt {k} of {n}, run {run + 1})"


def ask_parallel_div(a, prompt, n):
    run = getattr(a, "run", 0)
    prompts = [variant(prompt, k, n, run) for k in range(1, n + 1)]
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(agent.ask, a, p) for p in prompts]]


agent.ask_parallel = ask_parallel_div   # solve() looks it up at call time

if __name__ == "__main__":
    print(f"feedback v8: v7 (PROMPT1={v7.PROMPT1} MESSAGES={v7.v6.MESSAGES} CARD={v7.v5.CARD} "
          f"SAMPLING={v7.v5.SAMPLING} GATE={v7.GATE or 'off'}) + one prompt line per sample k>=2")
    agent.main()
