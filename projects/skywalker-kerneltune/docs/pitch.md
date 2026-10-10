# KernelTune: the pitch (3 minutes)

Team Skywalker, seats 205–209. Every number here is in `docs/note.md` or the README and comes from
the real chip. Do not say a number that is not there.

## 0:00 — The opening

> "A hand-written chip kernel has settings that decide how fast it runs. Pick wrong and it is up to
> seven times slower, and the right answer changes with every shape and precision. Today an engineer
> finds it by trial and error. We built KernelTune: a loop where two 8-billion-parameter models tune
> those settings, with every attempt measured on a real Trainium2 core. And we found that the model
> is not what decides whether it works. The checker is. Same model, same chip, same budget: by
> redesigning only what the checker says, we went from 3 good runs out of 30 to 29 out of 30."

## 0:30 — What we built

Three roles, as our mentor asked for.

1. **The workload.** AWS's blocked matmul kernel. Three settings say how much data it loads per trip and how much it keeps on the chip. There are 75 to 96 legal combinations depending on the shape.
2. **The checker.** Code runs the kernel on the chip and first asks: is the answer still right? Then a small model reads the measurements and says what to change.
3. **The optimiser.** A second small model edits the setting. Then we measure again. Ten attempts, keeping the fastest.

Say plainly: the models choose three numbers; they do not write kernel code. And: we measured every
legal setting on the chip, so we know the right answer for every run.

## 0:50 — Five checkers, one model

Show `results/curve_s30_K256_M4096_N12288.png`.

The shape is an adapter-style layer sized to Qwen3-8B's layer widths. Only 4 of 96 settings are
good, the worst is 7.1 times slower than the best, and AWS's suggested settings do not run here.

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

## 1:35 — The limit we found

> "Here is the part we did not expect. Take the menu our checker builds and replace the model with
> a one-line rule: pick the move with the best measured record. No model at all. It finds the best
> setting in 71% of runs. The 8B model manages 50%, and random 38%. So the value is in the
> measured menu, not in the model choosing from it."

Do not say the model beats random at finding the best: 50% against 38% is not conclusive at 30 runs.
"More reliable than random" is supported (29 of 30 within 10%, against 73%).

## 1:55 — Does it reach a model?

> "We ran the loop on every linear-layer shape in Qwen3-8B and compared it with AWS's own compiler
> doing the same multiply with no hand-written kernel."

| Qwen3-8B, all 252 projections, 512 tokens, 32-bit | Time |
|---|---|
| Our kernel, untuned | 434.1 ms |
| Our kernel, tuned by the loop | 370.9 ms |
| AWS compiler | 372.6 ms |

> "Untuned, a hand-written kernel is 12 to 30 percent slower than the compiler. Tuned by the loop
> it is level with it on every layer. Then we ran the loop at 16-bit, the precision real models
> use, scored on real inference time. From as slow as 8.6 milliseconds, all five runs ended at
> about 2.9, within 5% of the compiler at 2.75."

| One Qwen3-8B feed-forward block, 16-bit | Inference time |
|---|---|
| Our kernel, untuned | 8.63 ms |
| Our kernel, tuned by the loop (five runs) | 2.86 to 2.89 ms |
| Best of all 72 settings | 2.86 ms |
| AWS compiler | 2.75 ms |

Say that the 434 / 371 / 373 figures are sums of standalone kernel times, not the model running,
and that the 16-bit test is a small model with random weights at Qwen3-8B's sizes. Say "level with
the compiler", never "faster than": the differences per layer are 1.5% or less and were measured on
different seats.

## 2:30 — What the checker caught

Pick two, by time.

- **The simulator and the chip disagree.** On two of four small shapes the simulator counts wasted re-reads where the chip moved the minimum. The hackathon's optimisation levels are scored on the simulator.
- **The pod's default core setting doubles every profile count.** Our first profile was wrong for this reason.
- **AWS's suggested settings run on none of Qwen3-8B's four layer shapes at 512 tokens.**
- **An untuned 16-bit kernel came out 2.4% wrong with no error.** The checker's accuracy stage is what catches it.

## 2:45 — The close

> "So what are we handing over? An on-chip checker this track did not have. Full ground truth at
> two shapes. And measured evidence of what a small-model loop needs: tell it what the chip
> measured, not what to think. We do not beat your compiler, and for matmul nobody needs to. The
> next step is to point KernelTune at the kernels people do write by hand, fused attention and
> quantised operations, where there is no compiler version to fall back on."

## Likely questions

| Question | Answer |
|---|---|
| What exactly are the settings? | Three block sizes, M, N and K: how many tiles of each matrix are loaded per transfer and kept on the chip. Bigger blocks mean fewer trips and more reuse, but need more of a small on-chip buffer. |
| What does "untuned" mean? | Our kernel with all three settings at 1: one tile at a time, the most trips, no reuse. |
| How does the loop get M, N and K from the hardware profile? | It does not calculate them. It tries a change, times it on the chip, and keeps what is faster. The profile is the scoreboard. The designs that worked show the model which changes measurably helped, instead of asking it to reason from the raw profile. |
| How is there a compiler number if the kernel is hand-written? | They are two ways to do the same multiply. The compiler number is plain PyTorch compiled by `neuronx-cc`, with no hand-written kernel and no settings. It is our yardstick. |
| Does your tuned kernel beat AWS's compiler? | No. It is level with it at 32-bit (370.9 against 372.6 ms across Qwen3-8B's layers) and 4% to 5% behind at 16-bit (2.86 against 2.75 ms for one block). Where the compiler can merge the multiply with the next operation it is 28% ahead. |
| Then why would anyone hand-write a kernel? | For a matmul, they would not. Hand-written kernels are for what the compiler does not cover or covers badly: fused attention, custom normalisation, quantised multiplies. We used matmul because it has a public reference kernel, a compiler version to compare against, and few enough settings to measure all of them. |
| How would this help someone running, say, YOLOv8? | Not at all if they use the stock model through the compiler. It helps where part of the model is a hand-written kernel: give the loop the kernel's settings, a reference answer and the legal values, and it tunes that kernel for every shape in the model. We have not built a convolution kernel. |
| Why a model at all, if a rule does better? | On this problem, we would not use one. That is our finding. The model was the way we discovered what the rule needed to know. |
| Does the model beat random? | It is more reliable: 29 of 30 runs within 10% of the best, against 73% for random. At finding the exact best it is 50% against 38%, which is not conclusive at 30 runs. |
| Did you make Qwen3-8B itself faster? | No. Its multiplies come from AWS's compiler, so our settings have nothing to plug into. We tuned standalone kernels at its four layer shapes (1.14x to 1.31x over untuned) and timed small models built from its layer sizes. |
| Inside a model, what did tuning change? | Two feed-forward blocks at Qwen3-8B's sizes, 32-bit: 20.12 ms untuned, 17.40 ms with the loop's settings (13.5% less), 17.26 ms for the compiler. |
| Why is the adapter shape so different from 2048? | It has a small shared dimension, so there is little arithmetic per byte and the chip spends its time moving data. Our settings control how data is moved, so they matter 7x there and barely at all at 2048. We picked that shape on purpose as the hard case. |
| How is this different from `neuron-explorer recommend`? | We ran it in print-only mode. It is Bedrock-powered, sends a 155,604-character prompt, and is told not to give code. It advises a person. Ours is an 8B model with a prompt of a few thousand characters, in a loop that changes the kernel and measures again. |
| Do the models write the kernel? | No. They choose three block-size settings; code applies them. |
| Is 1.34x at size 2048 a gain over what AWS ships? | No. It is from the slowest setting. AWS's suggested settings there measure 905.5 µs against a best of 904.7. |
| At 16-bit, did the loop beat random? | No evidence of it. All five runs ended within 2% of the best, but random does that in 83% of runs on that problem. It shows the loop working on real inference time. |
| How noisy are the timings? | Under 1% across three captures. |
| Why do runs start from different settings? | The served model answers the same prompt almost identically each time, so repeats from one start are not independent. |
| Did the model just remember the AWS tutorial? | Possibly; it is public. Every model arm uses the same model, so differences between checker designs are not explained by memory. |
| Energy? | Not measured. The pods expose no power reading. |
| Was a large model involved? | Only to write the harness. Nothing larger than 8B is in the loop. |
