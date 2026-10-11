# SCRIPT — 3 minutes, slide by slide (English to say · 中文帮你理解)

Total ≈ 390 English words ≈ 2 min 50 s at a calm pace. The demo video (slide 2) plays by itself; you talk over it.
The same English text is in each slide's speaker notes.

---

### Slide 1 · Cover · 10 s
**Say:** "Same model all day — Qwen3-8B on our own Trainium chip. We never changed it. We only changed what the checker says when it's wrong. Here's what that did."

**中文：** 我们一整天都用同一个模型（Qwen3-8B，跑在我们自己的 Trainium 芯片上），从没换过。我们只改了"检查器在模型出错时说的话"。下面看看这带来了什么。

---

### Slide 2 · Demo video · 50 s (talk over the video)
**Say:** "Level 4. The model swapped the axes of one slice — on square tiles that doesn't crash, it silently computes the wrong matrix. With the same skeleton, upstream's message says 'most elements are wrong, check the layout'. True, and useless: zero of five. Ours reads the code and names the line and the fix — next round, solved, five of five. *(code appears)* That's all it is: about fifteen lines. *(A/B appears)* Level 5 is about memory traffic. Same model, same chip, alternating runs: upstream's hint, zero of five; our one sentence, five of five. *(scoreboard)* Baseline: nothing. Final agent: seven of eight levels."

**中文：**
- 第 4 级：模型把一个切片的两个轴写反了。方块形状下它**不会报错，只会悄悄算错**。
- 在同样的骨架下，上游原版的提示说"大部分元素错了，检查一下布局"，说得对但没用，5 次 0 解出。
- 我们的检查器**读模型的代码**，直接点名哪一行、怎么改，下一轮就解出，5 次全中。
- （出现代码时）实现就这么十几行。
- （出现对照实验时）第 5 级比的是从内存搬了多少数据。同一张卡交替跑：上游的提示 0/5，我们的一句话 5/5。
- （出现记分板时）基线一级都没解出，最终版 8 级解出 7 级。

---

### Slide 3 · Where we started · 12 s
**Say:** "The upstream checker always told the truth, but never what to change. The real bug was one swapped line. Baseline, same model: zero of four levels."

**中文：** 比赛方原来的检查器说的都是真话，但从不说该改什么。真正的错误往往只是一行写反了。用同一个模型跑原版，4 级一级都没过。

---

### Slide 4 · Two loops (the flowchart) · 25 s — **the key slide**
**Say:** "So we built two loops. The inner loop: the agent writes a kernel on Trainium, the checker scores it and turns the verdict into one instruction, and the agent tries again. The outer loop is our innovation: when the inner loop stalls, Claude audits from outside the code — the agent's repair logic against what the level actually requires — proposes a new instruction. We check it before use, A/B test it on our Trainium seats, and ship it as one flag."

**中文（指着图讲）：**
- **蓝色的内循环**：agent 写 kernel → 检查器打分 → 把"错在哪"翻译成"改哪里" → agent 再试。这是比赛本来就有的。
- **橙色的外循环是我们的创新**：内循环卡住时（一直 0 分、反复同样的错、或者变差了），让 **Claude 站在代码之外当审核员**，对照"agent 是怎么修的"和"这一级真正要什么"，找出 agent 的修复思路哪里偏了，提出新的指令。
- 我们先**验证**这个指令可行（手写一个能通过的 kernel，或确认它不会误伤任何正确代码），再在**芯片上做对照实验**，有效才作为一个新开关上线，然后回到内循环。

---

### Slide 5 · Claude as auditor · 20 s
**Say:** "Every stall became one flag. Swapped axes: zero to five of five. Twice the cause was the upstream prompt itself — changing one sentence fixed level 6. The chip rejected code the simulator accepts, so that became a checker rule. And three changes were dropped: two measured useless or worse, one shown never to fire."

**中文：** 每一次卡住都变成了一个新开关：
- 轴写反了 → 从 0/5 到 5/5；
- 有两次病根是**比赛方自己的提示词**，改一句话就修好了第 6 级；
- 真芯片拒绝了模拟器能通过的写法，我们把这条变成了检查规则；
- 也有 3 个改动被放弃：两个实测没用或更差，一个经核查根本不会被触发。

---

### Slide 6 · Results · 12 s
**Say:** "Same model: from zero of four to seven of eight levels, five of five each — including all three levels graded on memory traffic. Nine of ten on the NumPy ladder."

**中文：** 模型没换，从 4 级 0 过，到 8 级过 7 级，每级 5 次全中，包括 3 个考"数据搬运量"的性能级。另一套 NumPy 阶梯 10 级过 9 级。

---

### Slide 7 · Why trust it · 12 s
**Say:** "Every solve also passes hostile tests it never saw — AWS's own tutorial kernels fail them. The agent's confidence is calibrated, and levels 3 to 5 run correctly on the real chip."

**中文：** 每个"解出"的 kernel 都通过了它从没见过的刁钻测试，而 AWS 官方教程的 kernel 反而过不了。agent 自己给的置信度是准的（Brier 0.03，越接近 0 越准）。第 3–5 级的 kernel 在真芯片上也运行正确。

---

### Slide 8 · Honest limits · 10 s
**Say:** "What we didn't do: attention is still zero of five, we make no latency claims, and our repeats are less independent than five sounds. It's all in the log."

**中文：** 没做到的：第 8 级（注意力）还是 0/5；没有报运行速度的数字；服务器的输出是确定性的，所以"5 次"没有听起来那么独立。这些全部写在日志里。

---

### Slide 9 · Close · 5 s
**Say:** "The checker is the product — and Claude kept finding the one sentence that mattered. Thank you."

**中文：** 检查器才是真正的产品，而 Claude 一次次找到了那句最关键的话。谢谢。

---

## Words worth knowing · 几个关键词
| English | 意思 |
|---|---|
| checker | 检查器，给模型的代码打分并说明错在哪 |
| verdict vs instruction | "判定（错了）"对比"指令（改这一行）" |
| kernel | 直接在芯片上运行的底层计算程序 |
| Trainium | AWS 自研的 AI 芯片 |
| A/B test, control | 对照实验：一组用旧提示（对照组），一组用新提示 |
| memory traffic / bytes | 从内存搬运的数据量，性能级比的是这个 |
| held-out tests | 模型从没见过的测试，用来防止"刚好过公开测试" |
| calibrated / Brier | 置信度是否准确；Brier 越接近 0 越准 |
| [sim] / [device] | 模拟器上的结果 / 真芯片上的结果 |
