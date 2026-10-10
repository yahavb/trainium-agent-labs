# 项目 2 — Kernel Agent

**Agent 编写直接在芯片上运行的小程序，并在过程中持续检查自身输出。**

> ## 状态：可以运行，但尚未完全解决
>
> **当前可以运行，且已在 trn2 节点验证：**关卡阶梯、检查器（`nkibench.py`）、四个可通过检查的参考 Kernel，以及通过实时模型编写 Kernel 的 Agent 循环（`agent.py`）。
>
> **尚未解决：** Agent 还没有解决任何一个关卡。最高分只有 0.30/1.0：代码能解析、符合规则、能执行，但计算结果不对。下方记录准确展示了它卡在哪里，这就是留给你的问题。
>
> **仍缺失：** Level 5–7 的参考 Kernel，因此关卡阶梯的优化后半部分尚无标准答案；检查器第 2、3 层也没有实现，所以**现在完全无法测量延迟（latency）**。本文所有相关数字都来自模拟器的吞吐量分析。
>
> 这是一个真正尚待解决的问题，不是藏着标准答案的整理版练习。如果你能将某一关稳定做到 1.0，就取得了团队此前未能实现的成果。

---

## 核心思路

线性代数是 AI 芯片主要的耗时部分。矩阵乘法、归一化、Softmax 在 PyTorch 中可能只需一行，但底层都有人编写了 **Kernel**：明确将哪些数据 Tile 搬进片上内存、让哪个计算单元执行运算，以及如何重复利用数据，避免为同一字节反复付出传输成本。

写 Kernel 是专业工作，微妙的错误甚至比明显错误更危险，因为**程序可能完全不会崩溃**，只返回看似合理的错误数值。

因此要构建这样的 Agent：编写 Kernel、运行、与参考结果对比、阅读失败反馈、再次尝试。循环和 Project 1 一样，但检查器不同，而且最终必须能在硬件上执行验证。

## 使用哪个模型、哪些核心

**一个 NeuronCore 不能由两个进程共享。** 但一颗芯片有四个逻辑核心，模型服务仅使用两个，因此还剩两个。模型运行时可通过 `neuron-top` 检查：NC 2、NC 3 用于模型，NC 0、NC 1 空闲。

**当前框架的 Layer 1 不需要 NeuronCore。** 它通过 CPU 上的 `nki.simulate` 在数秒内检查 Kernel。Agent 的数百次尝试应在这里进行，而不是每次等待编译。因此 Agent 可以直接使用**本 Pod 中 `./serve.sh` 启动的模型**，不需要远程模型。

在一个终端内启动模型（先执行 `kubectl exec -it seat-42 -- bash`，然后 `cd /workspace && ./serve.sh`，等待 `READY`），再在第二个终端通过相同的 `kubectl exec` 进入同一个 Pod；请将 42 换成自己的座位号。随后执行：

```bash
git config --global --add safe.directory /workspace    # 在此运行任何 git 命令前必须设置
cd /workspace/projects/02-kernel-agent

python nkibench.py --selftest                       # 先验证测试框架本身
python nkibench.py --level 4 --check reference_level4.py

nohup python agent.py --all --rounds 6 --samples 2 --context 4096 > run.log 2>&1 < /dev/null &
tail -f run.log
```

> **只有这样启动，长任务才可在断线后继续运行。** `kubectl exec` 会话可能因 Wi-Fi、笔记本休眠或 `connection reset by peer` 断开，Shell 中的前台进程也会被终止。所以让任务后台运行、把输出写入日志，再实时查看：
>
> ```bash
> nohup python agent.py ARGS > run.log 2>&1 < /dev/null &
> tail -f run.log        # Ctrl+C 只停止跟踪日志，后台任务继续
> ```
>
> 断线后重新运行 `kubectl exec -it seat-42 -- bash`，再用 `tail -f /workspace/projects/02-kernel-agent/run.log` 继续看日志。`pgrep -af agent.py` 可检查任务是否仍在运行。

本页后面出现的每条 `python agent.py ...` 运行命令也都耗时较长，应以同样方式后台执行。

Pod 已设置 `KERNEL_AGENT_BASE_URL` 和 `KERNEL_AGENT_MODEL`，无需自行导出。`--context 4096` 在服务启动时设置的 8192 token 上下文范围内，Agent 会限制答案预算使其不超出。

### 如何发挥本地模型的潜力

默认配置较保守。下面四种调整按推荐优先级排列。

**1. 使用完整上下文。** Pod 中的服务器上下文长度是 8192，因此 Agent 应传入 `--context 8192`。修复提示词需要包含上一个 Kernel、检查器的建议与失败历史；若只给 4096，答案的可用空间会被挤占。

**2. 将每轮样本数从两个提高到四个。** 服务端可同时运行四条序列，因此几乎不额外增加实际耗时，就能把每轮尝试数翻倍。这对本地模型有用，但对共享 gpt-oss 端点没有意义——后者采用贪心解码，四次采样会得到四份相同答案。

**3. 运行。** 保持 Thinking **关闭**：

```bash
python agent.py --all --rounds 8 --samples 4 --context 8192
```

**4. 不要开启 Thinking。实测效果差很多。** `--think` 只是为了复现实验而提供，并非建议使用。同一批关卡、同一服务器（8192 上下文），唯一变化是 `--think`：

| 模式 | 每轮耗时 | 答案 |
|---|---|---|
| 关闭 Thinking | **约 8 秒** | 产生代码，评分 0.30–0.62 |
| 开启 Thinking | **446 秒** | 约 9,900 字符后截断，得分 **0.00** |

每个样本都以 `finish_reason=length` 结束，产生约 9,900 字符，却没有可用代码：模型把全部预算用于推理，没能完成答案。速度慢了 55 倍，却没有收获。

**这是第二个出现类似现象的模型**，也是本项目最有迁移价值的经验。共享 gpt-oss 端点收到 1,866 字符提示词时，生成了 13,245 字符的隐藏推理却没有答案；581 字符的提示词反而能得到可运行代码。两个模型失败模式相同，且**增加 token 预算并不能解决**：应当缩短提示词、减少推理，而不是增加空间。自己的 Agent 如果不返回东西，首先排查这里。

**不必折腾的配置：** 将张量并行从 2 增至 4 可能改善模型服务表现，却不会让模型变聪明，还会占用后面真实芯片计时需要的两个空闲核心。

### 小模型卡住时，尝试大模型

在本地 Qwen3-8B 上观察到：Level 2 有时解决，其余卡在 0.30–0.62。这时值得试试更大的模型：

```bash
export KERNEL_AGENT_BASE_URL="«组织者提供的 URL»/agg/v1"
export KERNEL_AGENT_MODEL=gpt-oss-20b
python agent.py --all --rounds 6 --samples 1 --context 8192 --terse 1
```

使用 `--samples 1`，因为该端点贪心采样，多样本会完全相同；使用 `--terse 1`，因为提示词太长会导致模型只推理不回答。记录各模型分别完成哪些关卡、用了几轮，这个对比本身就是实验结果。

只有加入**真实设备计时**（尚未实现的 Layer 2 和 3）时，才需要分配核心，并固定到模型未使用的两个核心：

```bash
export NEURON_RT_VISIBLE_CORES=0,1
```

这是文档化的机制，但项目尚未实测，可能需要调试。如果想让自己的 Kernel 使用整颗芯片，可以停止本地模型服务，改用独立硬件上的共享 `gpt-oss-20b`（[`../../gptoss/`](../../gptoss/)）。在使用前阅读 [`../../gptoss/README.md`](../../gptoss/README.md)：贪心解码意味着重复相同提示词无用；`tools=` 参数无效；输入上限为 8192 token。

## 已实现的 Layer 1：`nkibench.py`

其中所有操作都在 CPU 上完成，**不需要 Trainium 设备**；这正是 Agent 应当大量迭代的地方。

```bash
python nkibench.py --selftest                              # 首先验证检查框架
python nkibench.py --list                                  # 列出全部关卡
python nkibench.py --level 4 --show                          # 查看关卡要求
python nkibench.py --level 4 --check reference_level4.py      # 验证 Kernel
python nkibench.py --roofline 4096 4096 4096                # 写代码前先做性能判断
```

它依次回答三个问题，在第一次失败时停止：

1. **是否违反规则？** 静态扫描检查将整个运算交给框架的调用（`np.mean`、`torch.matmul`、`@` 运算符、输入参数上的 `.T`）、超过 128 的 partition 维度、缺少 `@nki.jit`、入口函数名错误等。**不会错误禁止 NKI 自身的原语**：例如对跨步视图使用 `nl.sum` 正是池化教程的方法，而 `nisa.nc_matmul` 是矩阵乘法关卡的核心。错拒正确 Kernel 比漏掉作弊代码更糟糕。
2. **数值是否正确？** 用 `nki.simulate_kernel` 对照 NumPy 参考实现，测试包含无法被 Tile 大小整除的刁钻 Shape。失败信息指出错误元素、输出中错误比例，以及错误是否发生在**不完整的边缘 Tile**，还是核心运算中——两者成因不同。
3. **受内存还是计算限制？** 在模拟期间封装 `nisa.dma_copy` 统计 HBM 传输字节数，依据 Shape 计算 FLOPs，再与 Roofline 拐点比较。这样**无需芯片也能估计算术强度**，因此 Level 4 能在内循环中完成诊断，不必等编译之后。

`reference_level1.py` 到 `reference_level4.py` 是教程的参考 Kernel，特意直接提供。教程本身公开，隐藏它们没有价值。检查框架的参考实现如果不能被阅读，也不值得信任。先用它们验证框架，再写自己的实现。

### 运行 Agent 时会看到什么

以下是 Qwen3-8B 上 `agent.py` 的真实输出，每轮两次尝试。**建议在开始运行前先阅读。** 项目还没有全部解决，但每次修复都让瓶颈转移；识别自己正卡在哪个瓶颈上，才是关键技能。

**Run 1 — 没有返回可加载的代码。**

```text
round 0: rewards [0.0, 0.0]  best 0.00  (54.8s)
  The code does not parse: invalid decimal literal on line 2.
round 3: rewards [0.1, 0.1]  best 0.10  (13.3s)
  Rule violations: `tensor_avgpool_kernel` is not decorated with `@nki.jit`
```

两个问题都在测试框架而非模型本身。Thinking 开启后，模型把 token 预算用在思考，最终仅返回片段——注意每轮 54.8 秒的耗时。此外代码提取器将自然语言文字提交给编译器，导致编号列表被解析成 `invalid decimal literal`。

**Run 2 — 代码能运行，但所有 NKI 函数名都是编造的。**

```text
round 0: rewards [0.3, 0.3]  best 0.30  (9.0s)
  raised AttributeError: module 'nki.language' has no attribute 'dot'
round 1: raised AttributeError: module 'nki.language' has no attribute 'value'
round 2: raised AttributeError: module 'nki.language' has no attribute 'sbuf_scalar'
```

每轮时间降至 5–12 秒，表明截断问题修复。但 `nl.dot`、`nl.value`、`nl.sbuf_scalar`、`tile.mean` 都不存在。**模型每轮猜另一个不存在的名字**，因为反馈只指出哪里错，却不告诉它如何修复。这是仓库中第四次遇到同一教训。

**Run 3 — 函数名是真的，但还存在一个具体误用。**

```text
round 0: rewards [0.3, 0.3]  best 0.30  (9.0s)
  raised TypeError: 'MemoryRegion' object is not callable
round 3: rewards [0.3, 0.3]  best 0.30  (7.8s)
  raised TypeError: 'MemoryRegion' object is not callable
```

API 速查卡解决了虚构名称问题。现在模型写出 `nl.sbuf(shape, dtype)`，把内存区域当函数调用，而正确写法是 `nl.ndarray(..., buffer=nl.sbuf)`。提示词虽描述了正确调用方式，模型却忽略了，于是改为提供**完整 Kernel 示例**；这一办法曾修复 Project 1 中同类型的问题。

**如何解读自己的运行日志：**

| 现象 | 含义 |
|---|---|
| `0.0`，每轮约 55 秒 | 答案截断，检查 `TRUNCATED` 提示 |
| `0.1` | 能解析但违反规则，查看是哪一条 |
| `0.3` | 规则没问题且能运行，但数值错误；这才开始解决真正的任务 |
| `1.0` | 所有 Shape 都正确，并打印 Roofline 结论 |
| **连续三轮同样错误** | 反馈只是判定，不是可执行指令；应改错误信息，而不是一味修改生成提示词 |
| 同一轮所有奖励相同 | 可能未开启有效采样，没有可以挑选的差异 |

### 一次运行不等于结论

**下面是实测，不是推测。** Level 2 进行了三次运行，代码、设置、模型都完全相同：

```text
run 1:  1.00  SOLVED on round 0
run 2:  1.00  SOLVED on round 3
run 3:  0.50  not solved in 6 rounds
```

本地 Qwen3 开启采样后，不总是选择概率最大的下一个 token，而是在合理候选中随机采样。这是有意为之：每轮四个样本就能得到四种**不同**的尝试，而不是四份复制。代价是**任何单次运行都带有运气成分**。

由此产生两个结论：

- **不能简单宣称 Level 2“已解决”。** 它大约三次中成功两次；应报告**成功率**，比简单的成功/失败更诚实也更有用。
- **纯粹运气带来的波动，比大多数改进效果还大。** 如果修改反馈后下一次得分下降，无从判断原因是修改还是随机采样。因此单次运行对比没有信息量，包括开发本检查框架时曾做过的那些对比。

因此要测量成功率：

```bash
python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5
```

该命令按关卡输出成功次数、最好、最差及平均结果。**报告成功率，不要只报告最好的一次。** 仅凭一次幸运运行就宣称“我们解决了它”，测到的是运气，不是 Agent 能力；参考实现本身在若干次提交中就犯过这种错误。

#### 测量的实际收益：撤销了一项“改进”

刚加入 `--repeat` 后，团队尝试了一个看似合理的修改：在提示词中添加第二个完整示例，演示如何以 128 为块遍历某一维度，针对三个关卡共同的瓶颈。修改前后各做五次运行，其他条件不变：

| 关卡 | 修改前 | 修改后 |
|---|---|---|
| 1 | 0.30 | 0.30 |
| 2 | **5 次解决 2 次** | 5 次 0 次成功，每次 0.30 |
| 3 | 0.30 | 0.30 |
| 4 | **0.62**，4 种 Shape 中通过 1 种 | 0.30，全部未通过 |

**结果全面变差，修改被撤销。** 原因颇有启发：Level 2 成功的 Kernel 根本没做分块，只使用三个普通嵌套循环；新示例反而引导模型使用 `dma_transpose`，又以五种不同方式用错它。Level 4 甚至无法再通过过去每次都能通过的那个 Shape。

需要吸取两点：

- **针对性的、看起来合理的提示词改进，让所有关卡变差。** 表面上看不出明显问题，只有测量能够揭示。
- **没有 `--repeat`，这项改动很可能会被保留下来。** 修改后的单次运行会被当作正常随机波动，退化被 Level 2 原有的噪声掩盖。

因此应该报告成功率，而不是某一次运行；上述问题实际发生在参考实现里，并非假设的团队错误。

### 亲自感受差别：不是所有 token 都相同——实测

在共享 `gpt-oss-20b` 上运行完全相同的关卡，服务端使用**贪心解码**：

```bash
export KERNEL_AGENT_BASE_URL="«组织者提供的 URL»/agg/v1"
export KERNEL_AGENT_MODEL=gpt-oss-20b
python agent.py --all --rounds 8 --samples 1 --context 8192 --terse 1 --repeat 3
```

设置 `--samples 1`，因为在贪心解码下，四个样本会给出四份**完全相同**的答案。

**两种模型、同一测试框架、同一关卡。** Qwen3-8B 跑五次，gpt-oss-20b 跑三次：

| 关卡 | Qwen3-8B（8B，采样） | gpt-oss-20b（20B，贪心） |
|---|---|---|
| 1 平均池化 | 0/5，始终 0.30 | 0/3，始终 0.30 |
| 2 转置 | **4/5**，得分 0.5–1.0 | **3/3**，始终 1.00 |
| 3 单 Tile 矩阵乘法 | 0/5，始终 0.30 | 0/3，始终 0.30 |
| 4 分块矩阵乘法 | **0.62** | **0.30** |

**有三个发现，没有一个是“模型越大越好”。**

**1. 贪心模型的结果可以精确复现到每一轮。** 三次 gpt-oss 运行不仅相似，而是完全一致：同样的错误按同样顺序发生，每轮耗时精确到 0.1 秒都一样，最终 Kernel 逐字符相同。Qwen3 在同一关卡上得到 1.0、1.0、1.0、0.5、1.0。一种模型适合逐项定位回归，另一种必须多次平均。

**2. 贪心解码在能解决的问题上更可靠。** Level 2 中 gpt-oss 3/3 成功，Qwen3 4/5 成功。最高能力相同，但波动更小。

**3. 8B 模型在 Level 4 上胜过 20B 模型**：0.62 对 0.30。原因是整个项目的重要教训：gpt-oss 把预算用在推理上，最后没有输出。该关卡六轮中四轮返回空结果，每轮在生成约 10,000 字符隐藏推理后结束，各浪费约 21 秒。Qwen3 推理较少，反而真正输出代码并走得更远。

**所以“更强的模型”不等于“更好的 Agent”。** 用于隐藏推理的生成预算就无法用于最终答案，而 Agent 循环必须获得答案。这是模型解码行为的特性，而非知识多少。因此这个端点需要 `--terse 1`，另一模型默认关闭 Thinking。

**报告什么：** 两种模型的结果及其波动。只跑大模型的团队可能断言题目太难；两种都跑的团队却可能发现小模型更适合，并能解释原因。

### 性能不佳如何体现在日志中

下面是一段真实 trn2 节点运行时 `--check` 的输出：

```text
level 4: matmul, tiled
  rules      clean
  numerics   4/4 shapes passed

  K=128 M=128 N=512: 294,912 HBM bytes in 3 transfers
    MEMORY BOUND: 56.9 Flops/Byte against a ridge of 222, so 26% of what the engine
    could sustain. The engine is idle waiting for data. Find reuse -- make the same
    bytes do more work -- rather than tuning the arithmetic. 3.9x more reuse needed.

  K=256 M=256 N=1024: 1,835,008 HBM bytes in 20 transfers
    MEMORY BOUND: 73.1 Flops/Byte against a ridge of 222, so 33% of what the engine
    could sustain. ... 3.0x more reuse needed.
```

**对每个 Shape，只需四类数字就足以诊断，而且它们都不是计时结果：**

- **传输字节数与传输次数**：例如 `294,912 HBM bytes in 3 transfers`。字节数能发现同一数据被反复读取的 Kernel；次数可识别相反的问题——大量极小的数据传输，即使字节数很少，单次发起传输的成本仍可能占主导。Level 2 的目的就是教会大家区分这两类问题。传输次数也用于检查测量代码：对于 `K=256 M=256 N=1024`，20 次传输恰好是 8 次内层迭代 × 2 次加载 + 4 次结果写回。若这个数字错误，后续分析都不可信。
- **算术强度与 Roofline 拐点的比较**：`56.9 Flops/Byte against a ridge of 222`，用量化比例而非主观印象描述未达到理想性能的程度。
- **达到理论上限的比例**：`26% of what the engine could sustain`，意味着约四分之三的机器性能没有被利用，可以直接作为评分依据。
- **距离目标还差多少倍**：`3.9x more reuse needed`，让 Agent 获得明确目标，并知道**什么时候停止优化**，而不是无休止地调参。

**真正有价值的是跨关卡的指标变化，而不是某一行。** 在该次运行中，Level 3 达到上限的 18%，Level 4 的不同 Shape 达到 26%、33%、38%。Tiling 明显增加了复用，但所有关卡仍为 Memory Bound，因此优化空间依然存在。**这一进步过程才是应提交的测量结果**：提交完整表格，而非单个数字。

该运行还暴露两个值得团队注意的问题：

- **传输字节数悄悄差了两倍，比完全不测量更危险。** 最初误以为 float32 输入每个元素占 2 字节，导致字节数低估整整一半、算术强度虚高两倍。例如 Level 3 曾被报告为 39.4 Flops/Byte，但修正后的 float32 数值是 14.2。这个错误会让严重受内存限制的 Kernel 看起来更接近 Compute Bound，导致优化方向错误。信任测量前，应先与一个手工算出的结果核对。
- **数据类型必须与 Roofline 上限匹配。** 222 Flops/Byte 是公开的 **bfloat16** 对应指标，而这里的 Shape 使用 float32。因此工具明确声明该结论只是指示性判断，而非严格同条件对比。不能拿一种 dtype 的测试数据与另一种 dtype 的峰值进行直接等价比较。

**刻意缺失的部分：Latency。** 上面的内容全是吞吐量推理，不需要设备。实际执行时间以及 DMA 加载是否与计算重叠，属于 Layer 2、3 的范围；再精确的算术强度也看不到这些问题。

### 在集群上运行

Layer 1 **不占用 Neuron 设备**，因此可以与模型服务一起运行，互不争抢设备：

```bash
kubectl create configmap nkibench-code \
  --from-file=nkibench.py --from-file=reference_level1.py --from-file=reference_level2.py \
  --from-file=reference_level3.py --from-file=reference_level4.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f ../../k8s/nkibench-job.yaml
kubectl logs -f job/nkibench
```

这个 Job 将输出 SDK 本身的 Tile 大小常量，运行自检，并让全部四个参考 Kernel 通过测试框架。

### 哪些已验证，哪些还没有

规则检查器、参考实现、Shape 生成器、失败消息和 Roofline 算术，都能通过任意机器上的 `--selftest` 运行。**Roofline 模型精确复现了教程公布的数据**：每 16.8 MFlops 传输 160 KB、算术强度 102.4、拐点 222。这项核对尤其重要，因为整个诊断都建立在它之上。

尚未在普通环境验证的是：`simulate_and_count()` 会导入 `nki`，因此此前从未在 Neuron 环境以外运行。它通过修改 `nisa.dma_copy` 统计字节数，这依赖于 Kernel 使用它的方式。**上面的集群 Job 就是用来验证这一点的**，自检也会明确说明。

## Agent 遇到的低效问题及其顺序

每类问题都有测试框架打印的**信号**、相应**修复措施**，以及修复后暴露的**下一个瓶颈**。最后一列尤其重要：解决一个问题并非任务结束，而是瓶颈转移。

> **实测与预期的区别：** 第 1–4 项在 trn2 节点上测量过。第 5 项及后续内容是根据教程结构**推测的**，尚无 Agent 真正跑通这些关卡，仓库也缺少 Level 5–7 的参考 Kernel。应把它们当成路线图，而不是已确认的结果。

### 1. 根本无法编译或运行——*已实测*

**信号：** 规则扫描报错，或模拟器抛出异常。例如 `line 5: partition dimension 256 exceeds the maximum of 128`，或者 `no function named nki_matmul_tiled_ is defined`。

**修复：** 将 partition 分成不超过 128 的 Tile，使用正确入口函数名，添加 `@nki.jit`。

**暴露出的下一个问题：** 代码是否计算正确，这是独立的问题。

### 2. 中间正确、边缘错误——*预期会出现，测试框架已支持检测*

**信号：** `NUMERICAL MISMATCH ... this is in the final partial PARTITION tile — the ragged edge is the likely cause, not the core arithmetic`。

**修复：** 对最后一个不完整 Tile 单独处理。

**暴露：** 代码数值正确后，才能真正讨论性能。

### 3. 只能处理一个 Tile，无法扩展——*已实测，Level 3*

**信号：** `K=128 M=64 N=512` 正确，但更大 Shape 都失败，算术强度只有 **19.7 Flops/Byte，约为拐点的 9%**。

**修复：** 在三个维度上都循环遍历 Tile。

**暴露：** 分块之后产生的冗余加载。

### 4. 相同数据反复穿越总线——*已实测，Level 4*

**信号：** 算术强度随 Shape 增长，但始终明显低于拐点。实测为 **28.4、36.6、42.7、28.4 Flops/Byte，仅为拐点的 13%–19%**；所有 Shape 仍为 `MEMORY BOUND`，并显示 `3.0x to 3.9x more reuse needed`。

**修复：** 把数据加载移出最内层循环，避免每一轮都重新读取相同 Tile。

**暴露：** 这样的提升只实现了一行 Tile 的复用，但 SBUF 实际能存放更多。

### 5. SBUF 利用不足，只复用一行 Tile——*预期，Level 5 → 6*

**信号：** 算术强度高于第 4 项，但仍 `MEMORY BOUND`；读取字节数依然远超操作数本身大小。

**修复：** 对 M、N 维进一步做 Blocking，让更多 Tile 常驻缓存。

**暴露：** 此时不是单纯的数学推导问题，而是**搜索空间问题**：块大小受 SBUF 容量约束，需要 Agent 探索。日志中的 `Nx more reuse needed` 可作为停止搜索的判断依据。

### 6. K 维仍持续流式加载——*预期，Level 6 → 7*

**信号：** 算术强度接近但仍小于拐点，且与 Shape 有关。

**修复：** 对 K 维也进行 Blocking。

**暴露：** 最终逼近 Roofline 拐点，吞吐量分析到此告一段落。

### 7. 达到拐点，但单次调用依然慢——*预期，Layer 1 无法观察*

**信号：** 当前检查框架完全看不到。算术强度已达到拐点，但每次执行 Kernel 仍慢。**只有 Profiling 才能发现**：DMA 忙碌时 Tensor Engine 闲置，说明加载与计算轮流进行，没有重叠。

**修复：** 使用 Double Buffering（双缓冲），当前 Tile 正在计算时就开始加载下一个 Tile。

**暴露：** Blocking 后的 Kernel 在小 Shape 上可能反而比 Level 5 慢，因为它付出了复用的额外成本，却没获得实际复用收益。这正体现 **Latency 与 Throughput 的矛盾**；应作为发现报告，而不是掩盖成“性能回归”。

### 8. 跑得快，但答案错——*每一关都必须通过的门槛*

**信号：** 数值校验失败，但算术强度看起来很好。

**修复：** 没有捷径——计为零分。必须在考虑任何候选方案的计时之前，自动重新校验正确性。

---

### Agent 本身的低效问题——真正耗掉一天的地方

以下问题都在 [Project 1](../01-heat-rod-pde/) 或仓库早期 Kernel 工作中实际观察到，并非假设。

| Agent 的失败方式 | 日志表现 | 经验证有效的修复 |
|---|---|---|
| 输出符号表达式而不是代码 | `Could not read the expression 'X(x)T(t)'` | 提供**完整合格答案示例**，而非禁止事项清单 |
| 按规则反复自我审查，最后不输出 | 返回内容为空；token 预算越大，推理越多 | 将限制条件从**生成提示词移到验证器**中 |
| 完全相同地重试，却期望不同结果 | 每次答案逐字节相同 | 修改**提示词**而非寄希望于随机性；采样可能是贪心的 |
| 猜答案，不做计算 | 合理但不收敛的模式；连续两次相同答案 | 提供可由模型自主调用的**工具** |
| 直接抄反馈中的标准答案 | 看似“解决”但没有推导 | **只给方向性信息**，不要泄露目标值 |
| 对失败 Kernel 声称“已验证” | — | 这种行为的评分应低于诚实报告失败 |
| 修好一个地方又弄坏另一个 | 理解加深但**分数反而下降** | 预期会发生；设计奖励权重时考虑这种现象 |

## 添加自己的运算

**现有关卡只是起点，不是整个任务。** 矩阵乘法关卡存在，是因为教程提供了参考答案，方便验证测试框架。一旦确认框架可信，就可以将它用于真正感兴趣的运算。

添加一种运算只需要：**一个参考实现、一个输入构造器、一次 `level(...)` 调用。** 框架的其他部分无需了解新运算。`nkibench.py` 的 Level 8 就是完整示例——单头 Attention，仅需编写以下三个部分：

```python
def ref_attention(q, k, v):                    # 1. NumPy 标准答案
    d = q.shape[1]
    scores = (q @ k.T) / np.sqrt(d)
    scores = scores - scores.max(axis=-1, keepdims=True)   # 否则 exp() 可能溢出
    e = np.exp(scores)
    return (e / e.sum(axis=-1, keepdims=True)) @ v

def _args_attention(spec, r):                  # 2. 构造输入
    n, d = spec["seq"], spec["dim"]
    return tuple(r.standard_normal((n, d)).astype(np.float32) for _ in range(3))

level(8, "single-head attention", "nki_attention_",       # 3. 注册关卡
      "what it teaches", "what there is to optimize",
      ref_attention,
      [dict(seq=128, dim=64), dict(seq=64, dim=128), dict(seq=96, dim=32)],
      {"softmax", "attention", "matmul", "einsum"},        # 禁止整步作弊的框架调用
      make_args=_args_attention,
      label=lambda sp: f"seq={sp['seq']} dim={sp['dim']}")
```

这样无需额外开发，就能自动获得：静态规则扫描、与参考实现的 CPU 模拟比较、刁钻 Shape 测试、输入是否被意外修改的检查、HBM 字节数和传输次数、与理论最小传输量的比较、算术强度上限分析，以及 Agent 循环。`python agent.py --level 8` 可以直接运行。

**添加新运算时必须做好的三件事：**

- **禁止一行就完成整个操作的框架调用**，否则模型会直接调用。例如禁止 `softmax` 和 `matmul`，但不要禁止 NKI 原语；禁掉 `nl.sum` 会错误排除有效 Kernel。
- **加入至少一种无法被 128 整除的 Shape。** 大多数模型生成的 Kernel 在中间 Tile 正确，但最后一个不完整 Tile 出错。
- **保证参考实现显然正确，并说明其中陷阱。** Attention 的陷阱是直接计算 `exp()` 可能上溢。省略减去最大值步骤的 Kernel，在较小测试数据上似乎没问题，真实数据上却可能产生 NaN。正是这种“看似正确但实际错误”的特性，才让一个操作值得加入测试框架。

**为什么 Attention 是自然的下一步：**它由一次矩阵乘法、数值稳定的 Softmax、第二次矩阵乘法组成，中间分数矩阵大小为 `seq × seq`。将中间矩阵写入 HBM 再读出会带来主要性能损失，这也是 Fused Attention Kernel 存在的原因。测试框架已有的流量统计可以展示这一点：比较实际传输量与理论下界，融合版本将立刻与朴素版本拉开差距。

## 按照最能节省时间的顺序给出的建议

1. **把模拟器放进 Agent 的内循环。** 用 Layer 1 检查正确性，Layer 2、3 检查性能。每次尝试都编译的 Agent，一整天只能尝试几次；每次都模拟的 Agent 可以做几百次迭代。这是最重要的架构决策。
2. **运行 `--selftest`，先理解它证明了什么，再信任评分。** 它发现过错误标注的参考实现、错误拒绝正确 Kernel 的规则，以及两倍的字节统计误差。这些全发生在当前框架自己身上，发生在评测任何参赛代码之前。
3. **修改代码前，先计算算术强度。** Memory Bound 就要提高数据复用；Compute Bound 说明内存读取已跟得上，继续调加载没意义。盲目猜瓶颈是浪费一天的常见方式。
4. **分别检查传输次数与传输字节数。** 它们对应不同问题，解决方案不同。Level 2 主要受传输次数影响。
5. **限制条件应放在验证器，而非生成提示词中。** 本仓库两次实测：给模型一长串规则，它会不断审查自己然后不输出；完全不提供约束，它会自信地生成不合法代码。有效办法是让模型自由生成，由**验证器**发现违规，再只返回一条明确可执行的修改建议。
6. **判定不等于指令。** “它很慢”和“误差 341%”都可能正确，但都无用。“`sin(pi*x)` 这一项根本不应出现”才指出了修复方式。Project 1 三次遇到类似问题，每次都靠改进错误信息，而不是换更好的模型。
7. **不要在反馈中打印答案。** Project 1 实测：一旦告诉模型正确系数，它会原样复制，完全没有推导。只指出**什么东西错了**、**方向上怎么错了**，不要提供目标值。
8. **给模型工具，而非单纯提示。** 模型算不出积分，仅靠方向反馈只会猜。让它自己决定如何调用计算器后解决了问题：模型负责判断**计算什么**，SymPy 负责具体算术；前者才是应衡量的推理能力。
9. **预计错误会转移，而不是直接消失。** 系数正确后，衰减率又出错。若新出错部分权重高于新修复部分，即便理解更深，分数也可能**下降**。
10. **先检查边缘残缺 Tile。** 大多数生成 Kernel 在内部正确，最后一个不完整 Tile 错误。测试框架会告诉你问题属于哪种情况，应相信它的诊断。
11. **一次运行不是实验结论。** 每轮四个样本时，结果随运行而波动。报告运行次数以及结果分布范围。
12. **区分模拟器结果和真实设备结果。** 评委会问；“我们还没让它在芯片上跑起来”本身可以接受，前提是你主动明确说明。

## 尚待实现的内容

- 测试框架：参考实现、三层检查器，以及能够帮助 Agent 实际采取行动的错误信息，而非没有指导价值的简单结论。
- 哪些关卡会被保留用于最终评审。
- 算术强度计算是由系统提供给 Agent，还是要求 Agent 自行完成。直接提供计算值与仅提供线索存在明显区别；Project 1 测量过这一权衡，发现直接给出算好的值会让模型停止自行推导。

## 目前可用的版本：笔记本电脑版

[`CHALLENGE-kernel-agent.md`](CHALLENGE-kernel-agent.md) 是完整任务说明，[`kernelbench.py`](kernelbench.py) 是可用的测试框架。它要求在**类似 Kernel 的规则约束下使用 NumPy 编写 Kernel**：固定 Tile、显式循环、不允许使用全数组快捷操作、必须处理最后一个残缺 Tile。这些规则是 Kernel 编程的难点，全部可以在 CPU 上以毫秒级速度检查，因此经验能迁移到真实硬件。

```bash
pip install numpy
python kernelbench.py --selftest        # 证明能捕获预埋错误
python kernelbench.py --list            # 十个关卡
python kernelbench.py --level 1 --show
python kernelbench.py --level 1 --check my_kernel.py
python try_level.py --level 5             # 驱动模型尝试一个关卡
```

**即使已经有设备版测试框架，也建议从这里开始。** Agent 架构完全相同，反馈循环却快上千倍。

任务文档还记录了一个最有帮助的经验：在提示词中列出约束条件，模型往往什么也不输出；完全不列，模型就自信地产生违反规则的程序。有效做法是先朴素生成，让**验证器**发现违规，并只返回一个精确指出修改内容的指令。**限制条件应该放在验证器中，而不是生成提示词中。** Project 1 又三次遇到了同样的瓶颈。
