# Guide: Hack the Chip · seat 73

> A plain-language guide to this repository. `README.md` is the event's original upstream README, kept as it was;
> `README_zh.md` is this guide in Chinese.
> Every number comes from our own run logs. **[sim]** = result on the CPU simulator, **[device]** = result on a
> real Trainium chip.

---

## 1. What we did, in one sentence

**We never changed the model (Qwen3-8B, on our own Trainium chip). We only changed what the checker says when the
model is wrong, and wrapped it in a second loop where Claude acts as an auditor. That took an 8B model from solving
none of 4 kernel levels to solving 7 of 8, in 5 runs out of 5.**

---

## 2. The event, and what "upstream" means

- **The event**: "Hack the Chip", a one-day hackathon run by NYU and Annapurna Labs (the AWS team that designs its
  own chips).
- **"Upstream"**: the starter repository the organisers gave us, `yahavb/trainium-agent-labs`. It contains:
  - a **Trainium chip** with a small model, **Qwen3-8B**, running on it;
  - an **agent loop**: the model writes code → a **checker** scores it and says what is wrong → that message goes
    back to the model → the model tries again;
  - an **8-level ladder** (we call it Stage B): the model writes **NKI kernels**, the low-level programs that run
    directly on the chip;
  - a **10-level NumPy ladder** (Stage A): NumPy written under "kernel rules", checked on the CPU.
- **The event's thesis**: success depends on the **quality of the checker's feedback**, not on the model. "Wrong,
  off by 341 percent" is true and useless; "this term should not be there at all" tells the model what to change.

### What the 8 levels test

| level | task | what it tests |
|---|---|---|
| L1 | average pooling | can the model write legal chip code at all |
| L2 | transpose | data layout on the chip |
| L3 | single-tile matmul | the hardware rules of matrix multiply |
| L4 | tiled matmul | splitting a big matrix into tiles |
| L5 | matmul, loads hoisted | **beyond correctness**: HBM traffic ≤ 1.90× the theoretical minimum |
| L6 | matmul, M/N blocked | traffic ≤ 1.30× |
| L7 | matmul, fully blocked | traffic ≤ 1.05× (almost no repeated transfers allowed) |
| L8 | single-head attention | two matmuls with a numerically stable softmax in between |

L5–L7 compute exactly the same thing as L4. **They grade how few bytes you move**, because a chip spends most of
its time moving data, not computing.

---

## 3. Where the problems were

We read the model's nearly 2,000 attempts and found three kinds of problem:

1. **The checker's message was true, but not an instruction**
   - Example: the model wrote `lhsT[m, k]` instead of `lhsT[k, m]` (axes swapped). On square data this **does not
     crash; it silently computes the wrong answer**.
   - The upstream checker only said "most elements are wrong, check the layout". The model could not tell what to
     change, so it repeated the same mistake.
2. **A misleading message is worse than a vague one**
   - The model had simply forgotten to copy its data in, but the checker said "probably an uninitialised tile".
   - The model then invented functions like `nisa.psum_zero`, which do not exist, chasing a problem that wasn't
     there.
3. **Upstream's own prompt was the cause**
   - **L6**: upstream's level-6 hint ("block so tiles are reused") steered the model into the wrong loop order.
   - **L8**: the API card listed only `nl.sum`, and all 115 attempts used the wrong reduction operator.

---

## 4. Our approach: two loops (the core innovation)

```
┌──────────── Inner loop (from the event kit; runs every round) ────────────┐
│ code → agent writes a kernel → checker scores it → "what's wrong" becomes   │
│ "what to change" → back to the agent                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                 ↓ stalled (stuck at 0 / same error again / got worse)
┌──────────── Outer loop (our innovation) ───────────────────────────────────┐
│ Claude audits from outside the code: how the agent is repairing vs what the │
│ level actually requires → proposes one new instruction → we check it first →│
│ A/B test on the chips → ship it as a new flag only if it works               │
└─────────────────────────────────────────────────────────────────────────────┘
                 ↓ back to the inner loop
```

**The key points:**
- **Every new instruction is checked before use**: either we hand-write a kernel that passes, proving the path is
  completable, or we confirm it never fires on any correct code.
- **Every change is A/B tested**: same settings, one arm with the old message (control) and one with the new,
  5 runs each, judged by the solve rate.
- **Changes that don't help are dropped**: doc retrieval (E2), AST auto-fixes (E3) and the v10 instructions for L8
  were all dropped, and the reasons are in LOG.md.

---

## 5. What we changed, and what each change did (5 runs per level, [sim])

| change | what it does | effect |
|---|---|---|
| skeleton `--skeleton` | gives the model a code frame with TODO slots (not the answer) | L2: 0 → 5/5 |
| **v2** | the checker **reads the model's code**; for swapped axes it says "write `lhsT[k0:…, m0:…]`" | **L3, L4: 0 → 5/5** |
| **v3** | L1: "reduce each pooling window into a (C,1) tile" | **L1: 0 → 5/5** |
| **v4** | L5: "the lhsT load doesn't depend on n; move it out of the n loop" | **L5: 0 → 5/5** (control 0/5) |
| **v5** | L5–L7: "load every tile from memory exactly once" | **L7: 0 → 5/5** (control 0/5) |
| v7 / v8 / prompt fix | name "data never copied in" and "k loop in the wrong place"; replace L6's misleading hint | **L6: 0 → 5/5** |
| v9 | check every call against the real function signatures before simulating, and **list all errors at once** | saves several rounds |
| device rule | ban list comprehensions (the simulator accepts them, the chip's compiler doesn't) | L7 kernels are chip-compatible |
| upstream tool fixes | re-derive the L5/L6 bars from AWS's own kernels; fix the L8 crash | the ladder itself becomes trustworthy |

---

## 6. Final results

### Stage B (NKI, 8 levels): final agent "C10"

| | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
|---|---|---|---|---|---|---|---|---|
| upstream authors | 0/5 | 4/5 | 0/5 | 0/5 | none | none | none | none |
| our baseline | 0/5 | 0/5 | 0/5 | 0/5 | | | | |
| **final agent** | **5/5** | **5/5** | **5/5** | **5/5** | **5/5** | **5/5** | **5/5** | 0/5 |
| round solved | 2 | 1 | 2 | 2 | 4 | 3–4 | 3 | — |

### Other results
- **Stage A (NumPy, 10 levels): 9 solved**; only A5 (softmax) remains.
- **On the real chip [device]**: agent-written L3, L4 and L5 kernels run correctly on Trainium2 (error 1e-6).
- **Hostile tests**: all **31/31** kernels the agent claimed "solved" at 0.95 confidence pass tests they never saw;
  **AWS's own tutorial kernels fail them**, and one returns silently wrong numbers at K=131.
- **Confidence calibration**: Brier score **0.030** (closer to 0 is better; n=38).
- **Re-grading**: the 354 attempts behind the main decisions, re-scored from their own code: **0 mismatches**.

### Honest limits
- **L8 (attention) is unsolved**: after 4 rounds of audit the model now reaches the softmax, but still loops there.
- **No speed numbers**: the kernels compute correctly on the chip, but we did not reach on-chip timing.
- **"5 runs" are not fully independent**: the server decodes deterministically. Stage A repeats are identical, so
  they count as effectively 1 run.
- **L1 is the least stable**: pooled over every final-configuration run it is 17/20.
- **L2's skeleton is a strong hint**: it nearly states the transpose's index rule, so L2 counts as "solved with a
  strong hint".

---

## 7. Where everything is

### Deliverables for the judges
| file | what it is |
|---|---|
| `SUBMISSION.md` | the index: where every deliverable is |
| `CHECKER.md` | what the checker accepts and rejects, and why; the effect of every instruction version |
| `ATTEMPTS.md` + `attempts_all.csv` | **attempt log**: every attempt with its score (516 runs, 1,941 attempts) |
| `NOTE.md` | **one-page note**: what we ran, on what, how many runs, the spread |
| `RESULTS.md` | every experiment's result (ADOPT / REVERT) |
| `FAILURES.md` | failure taxonomy, counted by root cause |
| `CALIBRATION.md` | confidence vs hostile tests |
| `TOKENS.md` + `token_budget.png` | where every input token went |
| `REPRODUCE.md` | full commands to reproduce from scratch |
| `LOG.md` | **the day's log**: every decision, dead end and bug, timestamped |

### Demo and script
| file | what it is |
|---|---|
| slide deck (web) | https://claude.ai/artifact/QLxaNdnMKGBsm4EGbpQHQV (9 English slides; slide 2 is the demo video) |
| `SCRIPT.md` | **3-minute script**: English + Chinese, slide by slide |
| `DEMO.md` | demo flow and likely judge questions |
| `demo/demo.mp4` | the demo video (43 s) |

### Code (all in `projects/02-kernel-agent/`)
| file | what it is |
|---|---|
| `agent.py` | the agent loop; every improvement is a flag (`--v2-verdicts` and so on) |
| `verdicts.py` | **the core**: turns the checker's verdict into a concrete instruction (v2–v10 live here) |
| `levers.py` | skeletons, doc retrieval, mechanical fixes |
| `nkibench.py` | the Stage B checker (bars corrected, device rule added) |
| `heldout.py` | hostile tests + confidence |
| `kernelbench.py` / `stagea.py` | the Stage A checker and agent |
| `regrade.py` | the re-grading tool (checks results were not contaminated) |
| `jit_device.py` | runs kernels on the real chip |
| `agent_solutions/` | kernels **the model wrote**, saved as is, never hand-edited |
| `handwritten/` | kernels **we wrote by hand** to prove each instruction is completable |

### Raw data
- `runs/<time>_<name>/`: each experiment's raw logs (jsonl), with the full prompts, code and feedback.

---

## 8. How to run it

```bash
# 1. The demo (runs on a laptop, no chip needed; replays real logs; Enter to page)
python scripts/demo.py

# 2. Re-render the demo video
python scripts/make_demo_video.py demo

# 3. Refresh every deliverable (needs access to the pods)
scripts/refresh_deliverables.sh <label>
```

The full command for running the final agent on a chip pod is in section "4b" of `REPRODUCE.md`.

---

## 9. Glossary

| term | meaning |
|---|---|
| checker | scores the model's code and says what is wrong |
| verdict / instruction | "it's wrong" / "change this line to this" |
| kernel | a low-level program that runs directly on the chip |
| NKI | the language for writing kernels for AWS Trainium |
| Trainium | AWS's own AI chip |
| skeleton | a code frame with TODO slots |
| A/B test, control | a controlled comparison; control = the arm without the change |
| memory traffic / bytes / floor | data moved from memory / the theoretical minimum |
| held-out tests | tests the model never sees, so it can't just fit the public ones |
| calibration / Brier | whether stated confidence is accurate; Brier closer to 0 is better |
| regrade | re-scoring from code to check results weren't contaminated |
| ADOPT / REVERT | keep / roll back |
| seat | a pod with its own chip (we used seats 70–74, five in total) |
| [sim] / [device] | simulator result / real-chip result |
