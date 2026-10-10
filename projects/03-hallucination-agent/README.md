# Project 3 — The grounding agent

**A small model answers questions about a passage. A checker catches every answer the passage does
not support, and the loop repairs it. The weights never change: everything the agent does, it does
with prompts and checks.**

> ## STATUS: STARTER, UNMEASURED
>
> Checker selftest passes on 700 generated items; the dataset loaders pass theirs on fixture rows.
> The loop runs offline. **No run against Qwen3-8B
> yet**, so there is no baseline and no transcript. Producing them is step one.
> The full challenge, with the ladder, scoring and the inference-time levers:
> [`CHALLENGE-hallucination-agent.md`](CHALLENGE-hallucination-agent.md).

## The loop

```
controller   poses an item about invented companies          halworld.py
generator    the model proposes N replies, sampling ON       Qwen3-8B on your chip
checker      grades each one: quote real? answer right?      halcheck.py
loop         the best reply's reason becomes the next prompt agent.py
examples     the loop's successes, reused as prompt examples build_examples.py
datasets     SQuAD 2.0, HotpotQA, MuSiQue, FaithEval, PopQA  halsets.py
pilot        which dataset shows the update rule working     pilot.py
```

The method is a CRITIC-style loop (an external, exact checker drives each revision) with
RARR-style attribution (every answer must quote its evidence). See the reading list.

Same architecture as project 1: the same server (`serve.sh`) and endpoint, the checker on the CPU
with **no NeuronCores**, and the same JSONL attempt log. What is new is the label-free mode, an agent
that has to judge itself without the answer key, and the levers that help it.

## Run it

Inside your pod, with `./serve.sh` running in another terminal (top-level README, steps 1–7):

```bash
cd /workspace/projects/03-hallucination-agent

python halcheck.py --selftest                 # prove the checker before trusting a score
python halworld.py --list                     # the ladder
python halworld.py --show 4 --sub 1 --answer  # one item and its gold label
python agent.py --offline --level 1           # the plumbing, no model

python agent.py --level 2                     # one level, oracle feedback
python agent.py --level 2 --label-free        # the deployable version

# the zero-shot baseline: your "before" number
nohup python agent.py --all --rounds 1 --repeat 5 -q > baseline.log 2>&1 < /dev/null &
tail -f baseline.log

# the oracle loop on seeds 0-199, then a bank of worked examples from it
nohup python agent.py --all --seeds 200 -q --log lab.jsonl > lab.log 2>&1 < /dev/null &
python build_examples.py lab.jsonl --out bank.jsonl
python build_examples.py --gold 1-7 --out gold_bank.jsonl          # the baseline to beat

# levers, on seeds nothing was tuned on, one at a time
python agent.py --all --rounds 1 --seed 500 --seeds 5 --repeat 5 -q --examples bank.jsonl
python agent.py --all --seed 500 --seeds 5 --repeat 5 -q --label-free --agree 3
python agent.py --all --seed 500 --seeds 5 --repeat 5 -q --label-free --challenge-abstain
```

The pod already sets `HEATROD_BASE_URL`; `agent.py` uses it unless `HALLU_BASE_URL` is set.

## Real datasets

`halworld` is the contamination-free control. To show the update rule on data people recognise,
`halsets.py` turns public datasets into the same items, so the checker and loop run unchanged:

| name | what it is | gold kinds | note |
|---|---|---|---|
| `squad2` | SQuAD 2.0 dev: one passage, ~1/3 unanswerable | answer / not_in_context | probably in Qwen3's training data: a control, not the headline |
| `hotpot` | HotpotQA distractor: 10 paragraphs, 2 supporting sentences | answer | yes/no answers skip the answer-in-quote check |
| `musique` | MuSiQue full: 20 paragraphs, answerable/unanswerable twins | answer / not_in_context | not on the Hub; import the official JSONL |
| `faith-unans` | FaithEval unanswerable | not_in_context (all) | **mix with an answerable set**: `faith-unans+squad2` |
| `faith-incon` | FaithEval inconsistent: the passage contradicts itself | conflict (all) | adds a `CONFLICT` answer; mix it too |
| `faith-cf` | FaithEval counterfactual: multiple choice, passage contradicts world knowledge | answer | the quote needs to be real, not to contain the answer |
| `popqa` | PopQA, **closed book**, bucketed by entity popularity | answer | no passage: `NOT_SURE` is the honest abstention, and only `--agree` can help |

```bash
python halsets.py --selftest                       # loaders vs. fixture rows, no network
python halsets.py --fetch squad2 --n 300           # downloads into data/ (gitignored)
python halsets.py --fetch faith-unans --n 200
python halsets.py --inspect faith-unans            # CHECK THE MAPPING before trusting a number
python halsets.py --import musique musique_full_v1.0_dev.jsonl --n 300   # from StonyBrookNLP/musique

python agent.py --data faith-unans+squad2 --n 100 --rounds 1 -q          # round 0 only
nohup python pilot.py --data squad2 faith-unans+squad2 faith-incon+squad2 faith-cf hotpot popqa halworld \
      --n 50 > pilot.log 2>&1 < /dev/null &
```

**`pilot.py` is how you choose the demo dataset.** It runs the same items with no help and then with
the oracle loop, and prints round-0 hallucination next to post-loop hallucination for each dataset.
You want both: lots of hallucination, and a loop that removes it. Lots of hallucination with no
drop means the fix isn't in the context, and no update rule will show anything there.

Caveats, all unmeasured:

- **FaithEval and PopQA field names were written from documentation**, not a live download, which
  this environment could not reach. `--inspect` shows a raw row beside the item made from it; the
  loaders accept several field-name spellings, but check before reporting.
- **Long contexts.** HotpotQA and especially MuSiQue prompts can pass 3,000 tokens. Items over
  `--max-chars` (12,000, about 3.4k tokens) are skipped and counted. To keep them, serve with
  `MAX_MODEL_LEN=8192 ./serve.sh` and pass `--max-chars 26000`.
- **Downloads** use the Hugging Face dataset viewer API, falling back to the `datasets` package.
  The pod downloads model weights from the Hub, so it should reach it; set `HF_TOKEN` if asked.

## Reading the summary

```
level  solved  round-0 halluc  round-0 over-abstain  final halluc  mean rounds
```

**Round-0 halluc** and **round-0 over-abstain** are the two numbers that matter, and they trade
against each other. Report them together, always. In `--label-free` mode the last column becomes
`verified-and-right`: of the items the agent claimed to have verified, how many the oracle agrees
with. That is your calibration number.

## A team of four

| who | owns | first hour |
|---|---|---|
| 1 | **checker & levels**: new levels, held-out templates, the selftest | answer 20 items by hand; find a case the checker grades wrong |
| 2 | **loop & feedback**: turning verdicts into instructions | run the baseline with `--repeat 5` |
| 3 | **the label-free agent**: agreement, the abstention challenge, new label-free checks | run `--label-free` on level 2 and read every item it "verified" wrongly |
| 4 | **examples, analysis & demo**: the example bank, taxonomy, lever table, write-up | read `attempts.jsonl` and classify what the labels miss |

## Reading list

*Refining answers after the fact, zero-shot, inference only:*
- [Self-Refine](https://arxiv.org/abs/2303.17651) (Madaan et al. 2023): generate, critique, refine with one model.
- [CRITIC](https://arxiv.org/abs/2305.11738) (Gou et al. 2023): critiques checked with tools. Closest to this design.
- [Chain-of-Verification](https://arxiv.org/abs/2309.11495) (Dhuliawala et al. 2023): plan verification questions, answer them independently, revise. The abstention challenge is a small version of this.
- [RARR](https://arxiv.org/abs/2210.08726) (Gao et al. 2022): post-hoc attribution and revision against evidence; the quote check is a simple version of this.
- [Reflexion](https://arxiv.org/abs/2303.11366) (Shinn et al. 2023): a written memory of failures across attempts, like the ledger here.

*Whether self-correction works, and when:*
- [LLMs Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798) (Huang et al. 2023).
- [When Can LLMs Actually Correct Their Own Mistakes?](https://arxiv.org/abs/2406.01297) (Kamoi et al., TACL 2024). The reason for the oracle vs. label-free split.

*Signals without labels:*
- [SelfCheckGPT](https://arxiv.org/abs/2303.08896) (Manakul et al. 2023): disagreement between samples. What `--agree` uses.
- [Self-Consistency](https://arxiv.org/abs/2203.11171) (Wang et al. 2022): majority vote over sampled answers.
- [Semantic entropy](https://www.nature.com/articles/s41586-024-07421-0) (Farquhar et al., Nature 2024).
- [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221) (Kadavath et al. 2022).

*Why the reward is shaped this way:*
- [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) (Kalai et al. 2025): grading that gives no credit for "I don't know" rewards guessing.

*Datasets:*
- [FaithEval](https://arxiv.org/abs/2410.03727) (Ming et al., ICLR 2025): unanswerable, inconsistent and counterfactual contexts.
- [AbstentionBench](https://arxiv.org/abs/2506.09038) (Kirichenko et al., NeurIPS 2025): 20 datasets of questions that should not be answered outright.
- [PopQA / When Not to Trust Language Models](https://aclanthology.org/2023.acl-long.546/) (Mallen et al., ACL 2023).
- [MuSiQue](https://github.com/StonyBrookNLP/musique) (Trivedi et al., TACL 2022), [HotpotQA](https://hotpotqa.github.io/) (Yang et al. 2018), [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) (Rajpurkar et al. 2018).
