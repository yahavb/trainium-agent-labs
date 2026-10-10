# Pitch (3 minutes)

Every number here is in `docs/note.md` and comes from the real chip. Do not say a number that is
not there.

## 0:00 — The opening

> "The question for today was whether a small model, in a loop, on your chip, can tune a kernel.
> We built that loop with two 8-billion-parameter models and measured every answer on a real
> Trainium2 core. What we found is that the model is not what decides the outcome. The checker is.
> Same model, same chip, same budget: by redesigning only what the checker says, we took the loop
> from 3 good runs out of 30 to 29 out of 30."

## 0:25 — The method

Three roles, as our mentor asked for.

1. **The workload.** AWS's blocked matmul kernel on a Trainium2 core. Three settings control how much data it loads per transfer and how much it keeps on the chip.
2. **The checker.** Code runs the kernel on the chip and asks first: is the answer still right? Then a small model reads the measurements and writes **one instruction**.
3. **The optimiser.** A second small model reads that instruction and sets the three values. Then we measure again.

Say this plainly: the models choose three numbers; they do not write kernel code. And: we measured
every legal setting on the chip at two shapes, 75 and 96 of them, so we know the right answer for
every run.

## 0:50 — The story: five checkers, one model

Show `results/curve_s30_K256_M4096_N12288.png`.

The shape is a rank-256 adapter-style layer sized to Qwen3-8B's own layer widths. Only 4 of 96 settings are good,
and a bad one is 7 times slower. AWS's default settings do not even run here.

> "Version one showed the model the raw profile. It lost badly to random guessing. We read the log.
> The checker said 'increase N to 24, that reduces data movement', the chip came back two and a
> half times slower, and it said it again. Two thirds of its attempts never produced a
> measurement."

> "So we changed the message, not the model. Version two showed it comparisons worked out in code:
> 'raising N from 2 to 24: 2.6 times slower'. Version three gave it a menu of moves with the
> measured record for each. Version four caught the second model editing a line it was not asked
> to touch. Version five had no wasted attempts at all."

| | Runs within 10% of the best | Attempts with no measurement |
|---|---|---|
| Random guessing | 73% | none |
| Checker v1 | 3 of 30 | 217 of 300 |
| Checker v5, same model | 29 of 30 | 0 of 300 |

> "That is the workshop's own lesson, measured: fix the feedback, not the model."

## 1:40 — The limit we found

> "Here is the part we did not expect. Take the menu our checker builds and replace the model with
> a one-line rule: pick the move with the best measured record. No model at all. It finds the best
> setting in 71% of runs. The 8B model manages 50%, and random 38%. So the value is in the
> measured menu. The model choosing from it adds nothing we could measure."

Do not say the model beats random at finding the best: 50% against 38% is not conclusive at 30 runs.
"More reliable than random" is supported (29 of 30 within 10%, against 73%).

On the easy shape, 2048 square, everything works including guessing: 39 of 75 settings are within
2% of the best and AWS's default is one of them. One sentence, then move on.

## 2:05 — Does it change real inference time?

> "We put the kernel inside a model built from Qwen3-8B's feed-forward block and timed inference
> on the chip, against plain PyTorch at the same precision."

| Two feed-forward blocks, 512 tokens | 32-bit | 16-bit |
|---|---|---|
| Our kernel, untuned | 20.12 ms | 16.53 ms |
| Our kernel, tuned | 17.40 ms | 5.02 ms |
| Plain PyTorch | 17.26 ms | 4.77 ms |

> "With the settings our loop chose, inference time dropped 13.5%, level with PyTorch's own. At
> 16-bit the 32-bit answer no longer holds, so we ran the loop again at 16-bit, scored on the
> block's own inference time. Five runs, five different starts, as slow as 8.6 milliseconds. All
> five ended between 2.86 and 2.89, about three times faster, within 5% of PyTorch at 2.75."

| One feed-forward block, 16-bit, loop scored on inference time | |
|---|---|
| Slowest start | 8.63 ms |
| Where the five loop runs ended | 2.86 to 2.89 ms |
| Best of all 72 settings | 2.86 ms |
| Plain PyTorch | 2.75 ms |

Say that random choice also gets within 2% of the best in 83% of runs on this problem, so this is
not evidence that the model beats random. Say that these are small models with random weights, not
Qwen3-8B itself. The 16-bit column in the two-block table above is a hand-found setting.

## 2:30 — What the checker caught

Pick two, by time.

- **The simulator and the chip disagree.** On two of four small shapes the simulator counts wasted re-reads where the chip moved the minimum. The hackathon's optimisation levels are scored on the simulator.
- **The pod's default core setting doubles every profile count.** Our first profile was wrong for this reason.
- **AWS's tutorial default settings run on none of Qwen3-8B's four layer shapes at 512 tokens.**
- **An untuned 16-bit kernel came out 2.4% wrong with no error.** The checker's numerics stage is what catches it.

## 2:45 — The honest claim, and close

- We did not beat AWS's compiler. Tuned, a hand-written kernel ties it at best.
- The small model is not better than a one-line rule.
- What we are handing over: an on-chip checker, full ground truth at two shapes, and measured evidence of what makes a small-model loop work and what does not.

> "We came to show that a small model could tune your chip. What we can show instead is exactly
> what the model needs to be told, and that once you can say it that clearly, you may not need the
> model. The next step is to point this at an operation the compiler does not already handle
> well, where a hand-written kernel is actually needed."

## Likely questions

| Question | Answer |
|---|---|
| Does your tuned kernel beat AWS's compiler? | No. It ties it at 32-bit (17.40 against 17.26 ms) and is about 5% behind at 16-bit (2.86 against 2.75 ms for one block). Where the compiler can fuse the matmul with the next operation it is 28% ahead. |
| Then why use a hand-written kernel at all? | Only where the compiler has no good answer: a new operation, or a pattern it does not fuse. We did not test such a case; matmul was the kernel we could measure completely in a day. |
| Why a model at all, if a rule does better? | On this problem, we would not. That is our finding. The model was the way we discovered what the rule needed to know. |
| Does the model beat random? | It is more reliable: 29 of 30 runs within 10% of the best, against 73% for random. At finding the exact best it is 50% against 38%, which is not conclusive at 30 runs. |
| Did you make Qwen3-8B itself faster? | No. Its matmuls come from AWS's compiler, so our settings have nothing to plug into. We timed small models built from its layer sizes. One loop run per layer shape improved all four by 1.14x to 1.31x as standalone kernels. |
| Why 16-bit? Is that a fair comparison? | Only like for like: 32-bit against 32-bit, 16-bit against 16-bit. Both tables are in the note. |
| How is this different from `neuron-explorer recommend`? | We ran it in print-only mode. It is Bedrock-powered, sends a 155,604-character prompt, and is told not to give code. It advises a person. Ours is an 8B model with a prompt of a few thousand characters, in a loop that changes the kernel and measures again. |
| Do the models write the kernel? | No. They choose three block-size settings; code applies them. |
| Is 1.34x at size 2048 a gain over what AWS ships? | No. It is from the slowest setting. AWS's default there measures 905.5 µs against a best of 904.7. |
| How noisy are the timings? | Under 1% across three captures. |
| Why do runs start from different settings? | The served model answers the same prompt almost identically each time, so repeats from one start are not independent. |
| Did the model just remember the AWS tutorial? | Possibly; it is public. Every model arm uses the same model, so differences between checker designs are not explained by memory. |
| Energy? | Not measured. The pod exposes no power reading. |
| Was a large model involved? | Only to write the harness. Nothing larger than 8B is in the loop. |
