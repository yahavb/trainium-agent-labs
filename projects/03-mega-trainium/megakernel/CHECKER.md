# The checker, and why it accepts what it accepts

`check.py` decides whether a decode-megakernel candidate is accepted. A kernel here can be wrong in a way
nothing flags on its own: it compiles, runs, and returns numbers of the right size and range. Attempt 1 in
`ATTEMPTS.md` was 52% off on real weights and looked fine. So the checker's job is to measure correctness
against an independent reference and, when it rejects, to **name the change to make**.

```
NEURON_RT_VISIBLE_CORES=2 python check.py --resident   # final run, 2026-10-10 20:30 UTC
[PASS] layout  reference matches HF  (17s)
[PASS] tiny    max error 1.02e-02 within 2.2e-02 (3x the 7.3e-03 that bf16 rounding alone gives)  (17s)
[PASS] head    embedding exact, logits rel-L2 2.4e-03, argmax consistent  (48s)
[PASS] real    teacher-forced argmax agrees on 31/32 steps; every disagreement is a near-tie inside bf16 noise (step 6: HF margin 0.028 < 3 sigma 0.0712)  (142s)
[PASS] speed   27.74 ms/token vs 47.60 per-op (1.72x), 75% of the HBM floor  (82s)
ACCEPTED
```

Every run is appended to `results/check_log.jsonl`. The same checker on deliberately broken inputs
(`--control`):

```
python check.py --gates layout,tiny --control mask_active_zero
[FAIL] tiny    max error 6.19e-02 > 2.2e-02; the output is closer to the 'active_token_excluded' variant (rel-L2 1.01e-02) than to the correct reference (5.61e-02): the kernel implements active_token_excluded
python check.py --gates layout,tiny --control no_rope_input
[FAIL] tiny    max error 6.55e-01 > 2.2e-02; the output is closer to the 'no_rope' variant (rel-L2 9.77e-03) than to the correct reference (6.27e-01): the kernel implements no_rope
```

## The gates, cheapest first

| gate | what it compares | accepts when | on failure it says |
|---|---|---|---|
| `layout` | float32 torch reference (`qwen3_ref.py`) vs HuggingFace Qwen3, tiny config | max relative diff < 1e-5 | fix the reference (QKV packing, QK-norm, RoPE, GQA) before trusting any later gate |
| `tiny` | layer kernel vs the reference, tiny config, 2 layers | max error ≤ max(3 × bf16 floor, 2e-2), where the bf16 floor is the error the *same math* gets just from rounding inputs to bf16 | which deliberately wrong reference the output matches better than the correct one: interleaved RoPE, no QK-norm, mask counting the active token twice, no RoPE, attention off. "Your kernel implements X" |
| `head` | embedding gather and LM head + argmax, Qwen3-8B vocabulary, random weights vs float32 numpy | gathered row bit-exact; logits rel-L2 < 2e-2; returned index = argmax of the kernel's own logits | which of the three broke, and where to look (indirect DMA stride; `pack_lm_head` permutation; argmax index bookkeeping) |
| `real` | real Qwen3-8B, 36 layers, 21-token prompt + 32 greedy steps, **teacher-forced** against HF float32 | every step where the argmax differs is explained by bf16 noise: HF's own fp32 margin between its token and the kernel's < 3σ of the kernel's logit error at that step | the step, both tokens, HF's margin vs 3σ ("a real error, not rounding"), and the next tool: bisect layers with `test_qwen3_8b.py --layers N` |
| `speed` | full decode step, one launch vs per-op `torch.compile` of the same math, same core, 36 layers | faster than per-op | the two times: nothing gained |

## Why these thresholds

* **Relative to bf16 rounding, not to zero.** The model ships in bf16. The torch reference computed in bf16
  is itself 7e-3 to 1e-2 away from float32. A threshold below that would reject correct kernels; one far
  above it would have let attempt 1 through on the tiny config. 3× the measured floor is tight enough to
  catch a missing term (attempt 1 was 30–50× over) and loose enough for accumulation order.
* **Teacher-forced, not free-running.** Free-running greedy text diverges for good after one flipped
  near-tie: attempt 8 matched HF for 6 tokens, then picked HF's #2 at a margin of 0.028 logits and the
  texts parted. Positional match was 8/32, which looks like failure and isn't. Feeding both models the
  same context makes every step an independent test (31/32).
* **"Explained by noise" is measured per step, not assumed.** The kernel's logit error has a standard
  deviation σ at each step (about 1% of the logits). If HF's own margin between the two candidate tokens is
  under 3σ, rounding alone can flip the order. Otherwise the disagreement is a bug, however rare.
* **Exactness where exactness is possible.** The embedding gather moves bits; anything but bit-exact is a
  wrong row or stride, so no tolerance.
* **Speed is a gate, with the floor as context.** A one-launch kernel slower than the per-op path is a
  failed attempt whatever its correctness. The HBM floor (weights + KV once at 736 GB/s) is reported so a
  pass also says how much is left.

## Why diagnostics that name a cause

The reference the kernel is checked against had to be right first (`layout`), and the bug that cost the
most time was found by comparing against references that are wrong on purpose: the output matched
"active token excluded" to 3e-3 at every write position. That comparison is now built into the `tiny`
gate, so the next time the kernel misreads a convention, the checker says which one.

Same idea as in the workshop's other projects: "off by 52%" is true and useless; "the active token is
missing from attention" names the change.

## Scope and limits

* Batch 1, greedy, one logical core (LNC=2), Qwen3-8B. The thresholds were derived for bf16 weights.
* `real` uses one prompt; the spread across prompts and runs is in `NOTE.md`.
* `speed` uses random weights (timing does not depend on values) and syncs every call, which slightly
  favours the per-op baseline.
