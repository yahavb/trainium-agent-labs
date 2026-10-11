# Agentic workflow — Team 15, Syntax Error

The agentic aspect is the automated **write → check → repair → check again** loop. Beyond using Trainium for inference, we delegate kernel implementation and revision to Qwen3-8B.

## What do we delegate to the model?

| Task | What Qwen does |
|---|---|
| Write a candidate kernel | Turn the operation specification and provided template into executable NKI code: indexing, data movement, computation, and output. |
| Repair correctness errors | Read its previous code and the checker's diagnosis, then produce a revised implementation. |
| Implement efficiency improvements | Apply instructions about reusing loaded tiles, reducing memory traffic, and preserving accumulators across computation steps. |

Examples from the recorded experiments:

- **L3/L4 — matrix multiplication:** the model swapped the matrix slice dimensions. Our feedback identified the incorrect slice and explained the required axis order. In the matched comparison, each level improved from 0/5 to 5/5 using the same model, starting template, and numerical checks.
- **L5/L7 — memory reuse:** feedback identified repeated input loads. Qwen revised the implementation to cache and reuse tiles. These results concern the modified checker's traffic metric on the tested shapes; they do not establish measured device speedups.
- **L6 — accumulation:** feedback explained that the accumulator must remain alive across all K tiles. Qwen revised the loop and accumulator placement.

## What makes the repair loop autonomous?

After a run starts, the Python controller automatically:

1. Requests candidate code.
2. Invokes syntax, API/rule, simulation, numerical, mutation, and memory-traffic checks.
3. Sends the latest candidate and actionable feedback back to Qwen.
4. Adds a short ledger of prior verdicts when failures repeat.
5. Stops on verification success or the run's stopping conditions, including its attempt budget.

A person does not need to copy each error back into the model. The implementation is in [the agent loop](artifact/projects/02-kernel-agent/agent.py); [nkibench.py](artifact/projects/02-kernel-agent/nkibench.py) and [verdicts.py](artifact/projects/02-kernel-agent/verdicts.py) provide checks and feedback. Each attempt's code, score, and feedback are logged.

## What did our team contribute?

We designed diagnostic feedback, templates, checker rules, and the repair controller. Several instructions provide substantial algorithmic guidance, so the model's contribution is implementing and revising kernels within that guidance. The final runtime uses one coding model with a programmed checker and controller.

Claude assisted our team's supervised development workflow: reading failures, proposing fixes, and helping analyze experiments. It is not an autonomous outer agent that rewrites the deployed checker during a run.

The results and qualifications are indexed in [README.md](README.md), with the original experiment records under [artifact/](artifact/).

## Presentation summary

> We delegate kernel coding, iterative repair, and implementation of memory optimizations to Qwen. Our contribution is a verification loop that turns concrete failures into targeted repair instructions. That lets the system improve a candidate automatically, with every attempt checked and logged.
