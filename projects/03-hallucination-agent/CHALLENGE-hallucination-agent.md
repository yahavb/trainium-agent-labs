# Challenge — The grounding agent

**Build an agent that stops a small model from making things up, using nothing but prompts and
checks. The weights are frozen. The visible problem is hallucination. The real problem is telling
"I don't know" apart from "I didn't look."**

> **STATUS: UNMEASURED.** The checker and the loop run, and the checker passes its own selftest on
> 700 generated items. **Nothing here has been run against Qwen3-8B yet.** The first team to run
> the zero-shot baseline with `--repeat 5` sets the number everyone else is measured against.

---

## Why this is hard, and why it isn't the reason you think

A hallucination is a fluent, confident answer that nothing supports. It does not crash. It reads
exactly like a right answer. That makes it the same kind of problem as project 2's wrong softmax:
**it fails silently**, and only a checker can see it.

The obvious fix is to tell the model "only answer if the passage says so." Try it. You will get one
of two failures, and fixing one causes the other:

> **Make the model careful and it answers "not in the passage" to questions the passage answers.
> Leave it alone and it invents a founder, a year, a town, in perfect grammar.**

That is why every level here **mixes** answerable and unanswerable questions, and why a checker that
only rewards abstaining gets gamed on the first round. An agent that always abstains scores 0.1 on
half of every level. One that always answers scores at most 0.4 on the other half.

**You cannot change the model.** Inference only: the endpoint on your chip, your prompts, your
checks, and as many calls as you can afford. Everything below is something an agent does *around*
a frozen model, which is the situation most people deploying a model are actually in.

## The setup: fictional worlds, so memory cannot help

Every item is a short passage about **invented companies**: founders, years, towns, headcounts.
Qwen3-8B has never seen them, so it cannot answer from memory. Whatever it says either comes from
the passage or is made up, and the checker can tell which by string matching. **No LLM judge
anywhere.** A judge model hallucinates too, and on a greedy endpoint you cannot even average it out.

The model must end with two lines:

```
ANSWER: <short answer>      or NOT_IN_CONTEXT      or FALSE_PREMISE
QUOTE: "<sentence copied word for word from the passage>"
```

The quote turns "is this grounded?" into a substring check.

## The ladder

Each level has four items; seeds 0–999 are yours, seeds ≥ 1000 and three unseen question templates
are held back for judging.

| # | level | the trap |
|---|---|---|
| 1 | lookup: one company, every fact present | none: the "the loop works" checkpoint |
| 2 | absent: the answering sentence is deleted in half the items | a plausible fact is missing, and the model supplies one |
| 3 | distractor: two companies share a name stem | it quotes the look-alike. Item 4: **only** the look-alike states the fact |
| 4 | false premise: the question assumes the wrong founder or town | it answers the question as asked. Item 3 has a **true** premise, to catch over-correction |
| 5 | counter-parametric: a "fictional briefing" alters famous facts | it answers from memory ("Paris"). Item 4 states a fact **truthfully**, so "distrust everything" fails too |
| 6 | multi-hop: find the company by founder, then read its fact | the first hop resolves and the second is missing |
| 7 | arithmetic: total or difference of two stated figures | the answer is not in the passage; the figures are. Item 4: one figure is missing |

### Then on real data

The invented companies prove the loop is not riding on memorised answers. Real datasets show it
matters. `halsets.py` loads SQuAD 2.0, HotpotQA, MuSiQue, the three FaithEval sets and PopQA into
the same item format, so the checker and the loop run unchanged (README, "Real datasets").

**Choose the demo dataset by measuring, with `pilot.py`, not from a paper.** You need a dataset
where Qwen3-8B hallucinates a lot in round 0 **and** the passage holds the fix, so the loop can
remove it. FaithEval is the likely candidate (built to make models contradict their context;
larger models were not more faithful), and MuSiQue the backup. PopQA is closed book: the loop
cannot supply a fact the model lacks, only turn a wrong answer into `NOT_SURE`. That is still a
result, reported as hallucination *and* coverage by popularity bucket.

Two rules for real data. **Never run an all-unanswerable set alone** (`faith-unans`,
`faith-incon`): always abstaining would score perfectly. Mix it with an answerable set:
`faith-unans+squad2`. And treat SQuAD as a **contamination control**: if the loop helps far more
there than on FaithEval or `halworld`, suspect memory.

Levels you could add (the most valuable contribution a team can make): needle-in-a-long-passage near
the 8192-token limit, two sentences that contradict each other, a question whose answer is a list,
and multi-turn, where the false premise comes from an earlier turn.

---

## Non-negotiable: two checkers, and know which one you are using

`halcheck.py` has two functions, and confusing them is the easiest way to fool yourself.

| | knows | use it for | available in deployment? |
|---|---|---|---|
| `check(item, reply)` | the gold label | scoring, the dev loop, building the example bank | **no** |
| `selfcheck(passage, reply)` | only the passage | the agent's own verification | **yes** |

`selfcheck` asks only what can be verified without the answer: *is every quote really in the
passage, and does it contain the answer?* That catches invented quotes and wrong-sentence quotes. It
**cannot** catch an unnecessary abstention, because "NOT_IN_CONTEXT" claims nothing. **That blind
spot is the problem to solve.** In `--label-free` mode, an agent that learns to abstain looks
"verified" on every item it gave up on.

The oracle loop is your lab. The label-free agent is your product. **Report both, and never report
an oracle-loop number as what the agent can do.** This split matches what the self-correction
literature found: models repair well with reliable external feedback and badly without it
(Huang et al. 2023; Kamoi et al. 2024).

Reward, out of 1.0: **0.1** format, **0.3** grounded (real quote, holds the claim), **0.6** correct.
`python halcheck.py --selftest` proves every gold reply scores 1.0 and every planted mistake lands in
the right bucket. **Run it before trusting any score, and again after every change you make to the
checker.**

## The feedback is where you win

Project 1 found it eight times: *the quality of the error message is the quality of the agent.*

- ❌ *"Hallucination detected."* That's true, and the model apologizes and does it again.
- ❌ *"The answer is 1922."* Project 1 measured this: shown the target, the model copies it and
  learns nothing. Worse, your example bank fills up with copied answers.
- ✅ *"Your quote is about Veldane Works, but the question asks about Veldane Mills. Find the
  sentence about Veldane Mills."*
- ✅ *"The text you quoted does not appear in the passage. Copy a sentence word for word, or answer
  NOT_IN_CONTEXT."*
- ✅ *"Before answering, check that the passage agrees with what the question assumes about who and
  where."*

The starter feedback is **directional**: it says where to look and what kind of mistake it was,
never the value. Improving these messages, and **measuring** each improvement with `--repeat`, is
the core of the challenge.

---

## Stage B — make round 0 better, without touching the weights

The loop fixes answers after the fact, at the cost of extra calls. Stage B asks how much of that
can be had **up front, or without the answer key**. Four levers are built in, each off by default
so you can measure what it buys. The baseline for all of them is `--rounds 1`, the model alone.

| lever | flag | what it does | the risk to measure |
|---|---|---|---|
| **worked examples** | `--examples bank.jsonl --k 3` | the oracle loop's successes on seeds 0–199 become in-context examples on unseen items | examples that are mostly abstentions teach abstaining |
| **sample agreement** | `--label-free --agree 3` | accept only when 3 of 4 samples pass `selfcheck` *and* agree (after SelfCheckGPT) | an easy question the model is unsure about never gets accepted |
| **abstention challenge** | `--label-free --challenge-abstain` | before accepting NOT_IN_CONTEXT, make the model copy every sentence about the subject | it talks itself into an answer that isn't there |
| **the repair loop** | `--label-free --rounds 4` | `selfcheck`'s reason feeds the next attempt | cost: up to 4× the calls |

The example bank is the bridge from the oracle to deployment. `build_examples.py` takes replies the
oracle loop reached, **in the model's own wording**, and `agent.py --examples` puts them in front of
items they were never written for. `--hard` keeps only the ones the model got wrong at first, which
are the traps it actually falls into. `--gold` builds a bank from templated ideal answers instead.
**That is the baseline to beat:** do examples the loop *discovered* help more than gold ones?

```
agent.py (oracle, seeds 0-199)  →  attempts.jsonl  →  build_examples.py  →  bank.jsonl
                                                              ↓
          agent.py --rounds 1 --seed 500 --seeds 5 --repeat 5 --examples bank.jsonl   (unseen seeds)
```

**What to measure:** round-0 hallucination rate **and** over-abstention rate, before vs. after,
on seeds the bank never saw, and the **calls per item** each lever costs. If hallucination falls and
over-abstention rises by the same amount, you taught it to refuse, not to read. Say so.

Three worked examples add about 1,800 characters, comfortably inside the 8192-token window. But
project 2 found that a longer prompt can make a model reason instead of answer, so check
`finish_reason` (the agent warns on `length`) and measure `--k 1` against `--k 3`.

---

## Facts from the other projects that will save you the morning

All measured in this repo; see the top-level README.

- **Keep the prompt short.** A prompt full of prohibitions made gpt-oss audit itself until it ran
  out of tokens and returned nothing. The starter prompt states the format once; the checker holds
  the rules. If you add "never invent, never guess, always verify…", measure it.
- **Thinking mode is off for a reason.** It cost 446 s per round and scored 0.00 on project 2.
- **One run is not a result.** Level 2 of project 2 swung 0.50 → 1.00 on luck. Use `--repeat 5`
  and report a rate. A plausible prompt "improvement" there made two levels strictly worse and would
  have shipped without it.
- **A greedy model plus an unchanged prompt repeats forever.** The loop keeps a ledger of failed
  replies for that reason. On gpt-oss (greedy), `--agree` is meaningless: every sample is identical.

## Scoring

The same rubric as every problem.

| weight | | |
|---|---|---|
| **30%** | Correctness | The **label-free** agent's hallucination **and** over-abstention on held-back seeds and templates. |
| **25%** | Delivered result | Levels cleared, and calls per item. Fewer calls for the same result is better agent design. |
| **25%** | Method & honesty | Does "verified" mean verified? Report how often `--label-free` claimed success and the oracle disagreed. Failure taxonomy with counts. |
| **20%** | Demo & write-up | Show a hallucination, the feedback, and the recovery. Then show a lever that prevents it in round 0. |

**A team that cuts hallucination by a measured, modest amount without raising over-abstention beats
one claiming zero hallucinations that cannot show the abstention rate.**

## What to hand in

1. **The checker**, with any levels you added and the reasoning for what it accepts.
2. **The attempt log**, both modes.
3. **The failure taxonomy**: counts per label, per level, before and after your best lever.
4. **The lever table**: round-0 hallucination, over-abstention and calls per item, for the baseline
   and each lever, `--repeat 5`, unseen seeds.
5. **A one-page reproduction note.**

## Where to start, in order

1. `python halcheck.py --selftest`, `python halsets.py --selftest`, then
   `python agent.py --offline --level 1`. No model needed.
2. Read five items with `python halworld.py --show 4 --sub 1 --answer`. Answer them yourself.
3. Baseline: `python agent.py --all --rounds 1 --repeat 5 -q`. **This is your "before".**
4. Fetch the datasets and run `pilot.py` on 50 items each. Pick the demo dataset from the table.
5. The oracle loop on seeds 0–199 (or the first part of the chosen dataset), to see what feedback
   fixes and to fill the example bank.
6. The label-free agent: plain, then with each lever. One change at a time, `--repeat 5` each.
7. Everything final on seeds 500+, or on `--offset` items of the dataset nothing was tuned on.

## Reading

The method is a CRITIC-style loop with RARR-style attribution. Refining answers after the fact,
at inference: **Self-Refine**, **CRITIC** (tool-checked critiques,
the closest to this design), **Chain-of-Verification**, **RARR** (revise to match retrieved evidence),
**Reflexion** (a memory of past failures, like the ledger). Why external feedback matters: **Huang et
al. 2023**, **Kamoi et al. 2024**. Label-free signals: **SelfCheckGPT**, **self-consistency**,
**semantic entropy**, **Kadavath et al. 2022**. On why grading schemes reward guessing: **Kalai et al.
2025, "Why language models hallucinate"**. Datasets: **FaithEval** (ICLR 2025), **AbstentionBench**
(reasoning tuning cut abstention ~24%, which makes Qwen3's thinking on/off a natural experiment),
**PopQA**, **MuSiQue**, **HotpotQA**, **SQuAD 2.0**. Links are in the README.
