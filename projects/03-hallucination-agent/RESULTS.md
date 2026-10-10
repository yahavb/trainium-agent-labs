# Project 3 — Results: the grounding agent on Qwen3-8B

**Team:** _(names)_  ·  **Hardware:** one Trainium2 chip (seat-17), vLLM, Qwen3-8B, TP=2, 4 concurrent requests  ·  **Date:** 10 Oct 2026

> **Status:** one run per configuration. Replication on two further disjoint question sets is
> in progress (section 6). Until it lands, treat every percentage as a single measurement.

## Headline

On 60 questions it was never tuned on (28 unanswerable, 32 answerable), a **label-free verifier
plus "refuse when unverified" cut hallucinations by 71% (17 → 5)** and raised the share of given
answers that were right from **34% to 64%**. The price is coverage: correct answers to answerable
questions fell **12 → 9** and refusals rose **5 → 20**, at 1.35× the model calls. It is a point
on a risk–coverage curve, not a free win. The model's weights were never changed.

---

## 1. Method, in brief

A **CRITIC-style loop with RARR-style attribution**, run at inference only:

1. **Format.** For each passage and question, Qwen3-8B must end with `ANSWER:` and a word-for-word
   `QUOTE:` from the passage, or `NOT_IN_CONTEXT`. The quote turns "is this grounded?" into a
   substring check.
2. **Grader (deterministic, no LLM judge).** `check()` knows the gold label and scores
   0.1 format + 0.3 grounded + 0.6 correct, with a failure label per reply. `selfcheck()` knows only
   the passage: every quote verbatim, and the answer inside the quote. Only `selfcheck` is
   available to the deployable agent.
3. **Label-free agent.** Answer → `selfcheck` → accept, or retry with the reason (up to 3 rounds,
   1 sample, 4 questions in parallel to fill the server's 4 slots).
4. **Checklist verifier** (`--verify checklist`; factored Chain-of-Verification). A fresh call sees
   only the question, the answer and the quotes; it lists every detail the question requires
   (who, what, to whom, when, where, how), marks each STATED or NOT STATED, and any NOT STATED
   rejects the answer. Its missing detail becomes the next round's feedback.
5. **Fail-closed** (`--fail-closed`). If no answer is verified within the rounds, answer
   `NOT_IN_CONTEXT` rather than present an unverified answer.

**Data:** FaithEval-unanswerable (passages edited so the answer is missing), SQuAD 2.0 dev
(adversarial unanswerables + answerable), HotpotQA distractor (two-hop, all answerable).
Development used the first 12 items; every result below uses later, disjoint items.

## 2. How we got here (each step measured)

| step | finding |
|---|---|
| Pilot, 7 datasets, loop with gold feedback | Qwen3-8B hallucinated **0%** on invented companies but **12–88%** of first answers on real text: the failures come from what the model already believes. In the pilot it **never** over-refused (HotpotQA, added later, did). |
| Same loop, vague feedback | Fixed **0** of the faith-unans hallucinations in 3 rounds. Failures shared one shape: **a real quote that misses one detail** of the question (time "in the 2000s", mechanism "signal transduction", direction "who swore fealty to whom", date "10th century" vs "1000s"). `selfcheck` passes all of them. |
| Basic verifier (12 dev items) | Caught 1 of 3 hallucinations, wrongly rejected 1 of 5 correct answers. Net 9/12 → 9/12. |
| Checklist verifier (same 12) | Caught **3 of 3** at the first check and kept the correct answer it doubted, but the model **re-answered until a later check passed** (2nd/3rd try). Net 9/12; verified-and-right 9/12 → 9/11. |
| Fail-closed | Turns "never verified" into an abstention instead of an unverified answer. |
| Grader fix | HotpotQA stores tokenised text (`New York 's`); natural quotes failed the verbatim check. Spacing around punctuation is now ignored (words must still match). |

## 3. Results

**Test A — 40 questions (33 unanswerable, 7 answerable; FaithEval-unanswerable + SQuAD 2.0)**

| | plain | checklist | checklist + fail-closed |
|---|---|---|---|
| hallucinated | 16 | 14 | **8** |
| faith-unans right (of 20) | 12 | 13 | **18** |
| squad2 unanswerable right (of 13) | 6 | 6 | 7 |
| squad2 answerable right (of 7) | 5 | 5 | 5 |
| verified-and-right | 23/36 | 24/33 | 24/33 |
| calls / question | 1.2 | 2.2 | 2.4 |

This mix is unanswerable-heavy: always answering `NOT_IN_CONTEXT` would score 33/40. Test B
balances it.

**Test B — 60 questions, balanced (28 unanswerable, 32 answerable; + HotpotQA)**

| dataset | n | plain right / halluc / refused | fail-closed right / halluc / refused |
|---|---|---|---|
| faith-unans (unanswerable) | 20 | 15 / 5 / 0 | **20 / 0 / 0** |
| squad2 unanswerable | 8 | 5 / 3 / 0 | 6 / 2 / 0 |
| squad2 answerable | 12 | 6 / 2 / 0 | 5 / 1 / 6 |
| hotpot (answerable, 2-hop) | 20 | 6 / 7 / 5 | 4 / 2 / 14 |
| **total** | **60** | **32 / 17 / 5** | **35 / 5 / 20** |

| | plain | fail-closed |
|---|---|---|
| answers given (not abstained) | 35 | 14 |
| …of which right | 12 (34%) | **9 (64%)** |
| answerable questions answered right | **12/32** | 9/32 |
| calls / question | 1.7 | 2.3 |

## 4. Failure taxonomy

| failure | where | what happens |
|---|---|---|
| **Detail omission** | faith-unans, squad2 | Real quote, right topic, one detail of the question unstated (time, mechanism, direction, date). Passes `selfcheck`. The dominant hallucination. |
| **Parametric override** | hotpot, popqa | Answers from memory (photographer Jürgen Vollmer → "American"; gold: German). |
| **Fabricated quote under pressure** | faith-unans | After feedback, invents a sentence that fits the question exactly. |
| **Verifier wear-down** | all | Rejected answers are re-submitted with new quotes until a check passes. |
| **Over-strict verifier** | hotpot, squad2 answerable | Two-hop evidence spans sentences; the checklist marks bridged details NOT STATED. In the first Test B run (before the grader fix) it wrongly rejected 13 of 20 correct candidates. |
| **Adversarial unanswerables** | squad2 | Role swaps and look-alike entities mostly pass the verifier. |
| Identical samples | all | In every case we inspected, the 4 samples at temperature 0.6 were identical, so sample agreement (`--agree`) carried no signal on this setup. |

## 5. Limitations

- **Single run per configuration;** replication pending (section 6).
- **Small n,** especially answerable questions in Test A (7).
- **The checklist prompt was written after reading the 12 dev failures**; Tests A and B are the
  unbiased measurements.
- **SQuAD 2.0 is likely in Qwen3-8B's training data;** FaithEval and the invented-company set are
  the contamination controls.
- **The quote format itself is hard for two-hop answers:** plain HotpotQA scores 6/20 under it.
- Results with the gold-label loop (pilot) are upper bounds, not deployable numbers.

## 6. Replication (pending)

Same configurations on two further disjoint sets of 60 (`--offset 112`, `--offset 172`).

| question set | plain right / halluc / refused | fail-closed right / halluc / refused | hallucination change |
|---|---|---|---|
| offset 52 (Test B) | 32 / 17 / 5 | 35 / 5 / 20 | −71% |
| offset 112 | _pending_ | _pending_ | |
| offset 172 | _pending_ | _pending_ | |

## 7. Reproduce

```bash
cd projects/03-hallucination-agent
python halcheck.py --selftest && python halsets.py --selftest && python agent.py --selftest
for d in squad2 hotpot faith-unans; do python halsets.py --fetch $d --n 300; done
C="--data faith-unans+squad2+hotpot --n 60 --offset 52 --label-free --samples 1 --rounds 3 --workers 4 -q"
python agent.py $C                                  --log b_plain.jsonl
python agent.py $C --verify checklist --fail-closed --log b_fc.jsonl
python compare.py b_plain b_fc
```

**References.** CRITIC (Gou et al., 2023) · RARR (Gao et al., 2022) · Chain-of-Verification
(Dhuliawala et al., 2023) · Huang et al., 2023; Kamoi et al., 2024 (self-correction needs reliable
feedback) · FaithEval (Ming et al., ICLR 2025) · AbstentionBench (Kirichenko et al., 2025) ·
Why Language Models Hallucinate (Kalai et al., 2025) · SQuAD 2.0 · HotpotQA.
