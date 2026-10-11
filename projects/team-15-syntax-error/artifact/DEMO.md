# DEMO + PRESENTATION — 3 minutes total

> The current, timed 3-minute script (English + 中文) is **`SCRIPT.md`**; the slides' speaker notes match it.

**Part A: live demo, under 1 minute.** Show our code and what it changed.
**Part B: presentation, 2 minutes.** The idea, the evidence, the honesty.

Before you start: open a terminal in the repo root, and open `projects/02-kernel-agent/verdicts.py` in the editor
at `def code_hint` (search for it). Font size large. Every number is from our own logged runs.

---

## Part A — Demo (≈ 55 s)

| time | do this | say this (English) | 中文意思 |
|---|---|---|---|
| 0:00 | run `python scripts/demo.py`, screen 1 appears | "Level 4, tiled matmul. The model swapped the axes of one slice: `lhsT[m, k]` instead of `lhsT[k, m]`. On square tiles that doesn't crash. It silently computes the wrong matrix." | 模型把一个切片的两个轴写反了，方块形状下不会报错，只会悄悄算错 |
| 0:12 | point at the yellow text | "With the same skeleton, upstream's message says: most elements are wrong, check the operand layout. True, and useless: the model never fixed it. Zero of five." | 原检查器说"大部分元素错了"，对但没用，5 次 0 解出 |
| 0:20 | point at the green text | "Ours reads the code and names the line and the fix. Next round: solved. Five of five." | 我们的检查器读代码，点名哪一行、怎么改，下一轮就解出，5/5 |
| 0:27 | switch to the editor, `def code_hint` in `verdicts.py` | "This is all it is: about fifteen lines that find the swapped slice in the model's own code." | 实现就这么十几行：在模型自己的代码里找出写反的切片 |
| 0:35 | back to terminal, Enter → screen 2 | "Level 5 is about bytes, not correctness. Same model, same seat, alternating runs. Upstream's hint: zero of five. Our one sentence, 'this load doesn't depend on n, move it out': five of five." | 第 5 级比的是搬运量。同一张卡交替跑：上游提示 0/5，我们一句话 5/5 |
| 0:47 | Enter → screen 3 | "Baseline: nothing solved. Final agent: seven of eight levels, five out of five." | 基线一级都没解出；最终版 8 级解出 7 级，每级 5/5 |

If the terminal fails, `python scripts/demo.py --fast` prints everything at once, or show `ATTEMPTS.md` (search `c1_L4`).

---

## Part B — Presentation (≈ 2 min, ~270 words)

### 0:00–0:25 · The question and the answer *（问题和答案）*
> "The challenge asked: can a small model on a chip we control, driven by an agent, solve what it can't alone?
> We kept one 8-billion-parameter model all day, Qwen3-8B on our own Trainium chip, and never changed it. We
> only changed what the checker *says* when the model is wrong. That took the NKI kernel ladder from zero
> levels to seven of eight, including all three levels that are graded on memory traffic, not correctness."

*中文：题目问小模型加 agent 能不能做到模型自己做不到的事。我们一整天没换模型，只改检查器说的话，结果从 0 级到 8 级里的 7 级，包括三个按数据搬运量评分的性能级。*

### 0:25–1:05 · What we actually changed *（我们具体改了什么）*
> "Three kinds of change.
> **One:** the checker reads the model's code, not just the error, and names the one line to change. Swapped
> axes, a load inside the wrong loop, tiles that were allocated but never filled.
> **Two:** twice, the problem was the upstream prompt itself. Level 6's hint pushed the model into the wrong
> loop order; the API card offered only `nl.sum`. Changing one sentence fixed level 6.
> **Three:** we closed the loop to the real chip. The model's kernels for levels 3, 4 and 5 run correctly on
> Trainium. Its first level-7 kernel didn't compile on the device, so that became a checker rule."

*中文：三类改动。一，检查器读代码，点名要改的那一行。二，有两次问题出在上游自己的提示上，改一句话就修好了第 6 级。三，打通真芯片：第 3、4、5 级的 kernel 在 Trainium 上运行正确；第 7 级最初的版本在芯片上编译不过，我们把这条变成了检查规则。*

### 1:05–1:40 · How we know it's real *（为什么可信）*
> "Every number is a rate over five runs, and every change was A/B tested. Three changes were dropped: two
> measured useless or worse, one shown never to fire. We also built held-out tests the agent never sees: ragged shapes, prime sizes,
> huge values. Every kernel our agent claimed at 0.95 confidence passed them. AWS's own tutorial matmul kernels
> failed them, and one returns silently wrong numbers."

*中文：每个数字都是 5 次的成功率，每个改动都做了对照实验，有两个我们自己的改动让结果变差，已经撤回。我们还做了 agent 从没见过的刁钻测试：agent 以 0.95 置信度声明解出的全部通过；AWS 官方教程的 kernel 全部失败，其中一个会静默算错。*

### 1:40–2:00 · Honest limits and close *（局限和结尾）*
> "What we didn't do: level 8, attention, is still unsolved, and we can tell you exactly where it loops. We
> report no latency numbers, because we didn't reach device-side timing. And the server decodes
> deterministically, so our repeats are less independent than five sounds. It's all in the log.
> The checker is the product: one sentence of feedback was worth more than any model change."

*中文：没做到的：第 8 级注意力还没解出，但我们知道它卡在哪里；没有报延迟数字，因为没测到芯片上的计时；服务器解码是确定性的，所以 5 次重复没有听起来那么独立。全部写在日志里。结论：检查器才是产品，一句好的反馈比换模型更有用。*

---

### Likely questions *（评委可能的提问）*
| question | answer |
|---|---|
| Did you give the model the answer? | No target values ever appear in feedback. For softmax (Stage A) we *refused* to paste a full kernel even though it would have "solved" it. Skeletons have TODO slots, and each was proved completable by hand first. |
| Is 5/5 meaningful if decoding is deterministic? | Stage B repeats did vary (batching): level 1 is 17/20 pooled. Stage A repeats are identical, so we call it n=1. |
| Why not a bigger model? | The upstream authors measured that gpt-oss-20b did *worse* than the 8B on level 4. Feedback quality mattered more. |
| What would you do next? | A static check for the remaining level-8 softmax loop, and device-side timing (`SpikeModel.benchmark`, API already found). |
