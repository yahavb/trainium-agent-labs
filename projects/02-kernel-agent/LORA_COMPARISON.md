# Matched original / LoRA / planner comparison

| Level | A: original + legacy | B: original + planner | C: LoRA + legacy | D: LoRA + planner |
|---|---|---|---|---|
| 1 | Pending | Pending | Pending | Pending |
| 2 | Pending | Pending | Pending | Pending |
| 3 | Pending | Pending | Pending | Pending |
| 4 | Pending | Pending | Pending | Pending |

These cells contain only the requested matched 2x2 study. Pending cells must not be substituted with historical or different-policy results. Study queued at /tmp/trainium-kernel-dev/projects/02-kernel-agent/runs/lora-matched-2x2-_mu040_v/evaluation; it preserves the previously scheduled full-adapter evaluation, then runs all four arms sequentially. A separate held-out repair worker is queued after the preserved matched study; the newer reproduction script also supports a held-out check before the arms. Same CPU Transformers model instance, tokenizer, checker, shapes, temperature .6, top_p .95, thinking off, context 8192, four candidates, eight-round ceiling, repeat one. A/C use standard generation, reward selection, standard repair, legacy feedback, examples off, no adaptive history. B/D differ only by hardware/semantic planner. The serving process switches PEFT adapter layers off/on per request; no server restart or weight merge.

Separate available evidence (not the matched 2x2): original five-repeat baseline solved zero of five on every level; historical best rewards .30/.30/.30/.625. Untuned full-agent scores are .30/.30/1.00/.625. First planner full-agent Level 1 run: .30, zero numerical cases, 20 candidates before unchanged repetition stop. Shape-consistency refinement completed at .30 in controlled-20261010T195135-qyw82zco. A generic instruction-legalizer warm replay scored 1.00 on Level 1, 4/4 shapes; this is excluded from cold-start and LoRA results. Cold-start planner/legalizer evaluation is recorded separately in controlled-20261010T201403-vxbwe5dd.

Training effect on initial generation, mathematical operation, valid APIs, repair success and official correctness: **not yet established**. Candidate code, actual tokens and per-shape checker results will be saved for every arm; held-out repair checks use real simulator/NumPy comparisons, not loss proxies. Wall times from this CPU study cannot be compared with older Neuron vLLM runs as model speedups.

```bash
python -B training/run_matched.py --prior-evaluation runs/lora-full-evaluation-e43ejnxq/evaluation --output runs/NEW-MATCHED --rounds 8 --samples 4 --levels 1 2 3 4
python -B -m training.report
```

New studies require exclusive artifacts and no competing evaluation. The quoted command waits for the prior evaluation instead of launching competing endpoint requests. One repetition per arm is not evidence of reliable superiority.
