# 黑客松笔记（Trainium agent labs）

**座位**：每人一个 pod，命令里的 `<N>` 换成**你自己的座位号**（队长 teoguo = seat-116）。不要进别人的座位。

> 来源：`README.md`、`STATE.md`、`projects/02-kernel-agent/README.md`、`projects/02-kernel-agent/CHALLENGE-kernel-agent.md`。
> 一句话核心：**决定 agent loop 好坏的是 checker，不是模型。** 「错了，偏差 341%」是真话但没用；「sin(pi·x) 这一项根本不该出现」才是能执行的指令。

---

## 0. 当前状态（2026-10-10 16:40，随时更新）

**人员**：teoguo（seat-116）+ liuyq（有经验）两人主力；另外三位新手做辅助，不计入关键路径。
**题目**：做项目 2（`projects/02-kernel-agent`，NKI kernel agent，在芯片上跑）。CHALLENGE（Stage A，`kernelbench.py`）有时间再做，同一套 agent 搬过去。评分按 30/25/25/20 那一套（CHALLENGE 第 239 行写着 "Same rubric as every problem"）。

**仓库和 remote**（本地 checkout：`trainium-agent-labs/`，工作分支 `team`）
| remote | 仓库 | 用途 |
|---|---|---|
| `team` | github.com/liuyq123/trainium-agent-labs | **共用仓库**，本地 `team` 分支跟踪 `team/master`，`git pull --rebase` / `git push` 直接用 |
| `origin` | github.com/teoguo/trainium-agent-labs | teoguo 的 fork，以后从这里给原仓库提 PR（从 `master` 开分支） |
| `upstream` | github.com/yahavb/trainium-agent-labs | 原仓库，只读 |

**Baseline**（seat-116，`--all --rounds 8 --samples 4 --context 8192 --repeat 5`，10:50–12:43，已完成，424 次尝试）
```
          solved   5 次分数                      STATE.md 参考
level 1   0/5      [0.30, 0.30, 0.30, 0.30, 0.30]   0/5，全部 0.30
level 2   3/5      [1.00, 0.30, 1.00, 0.30, 1.00]   4/5
level 3   0/5      [0.30, 0.30, 0.30, 0.30, 0.30]   0/5，全部 0.30
level 4   0/5      [0.62, 0.62, 0.50, 0.62, 0.62]   0/5，全部 0.62
```
**截止时间 18:30**（问过主办方）。座位 115/117/118/119 是组员的，可以并行跑；操作手册见 Claude 文档「Trainium 组员操作手册」。
日志拉回本机：`scripts/sync.sh 116 pull` → `runs/seat-116/latest/`（gitignore，不提交）。
重新生成分类表：`.venv/bin/python scripts/attempts_to_csv.py runs/seat-116/latest/projects/02-kernel-agent/attempts.jsonl -o analysis/baseline_seat116`

**实验结果（执行 1 维护；全部 `--rounds 8 --samples 4 --context 8192 --repeat 5`，vLLM TP2/8192/seqs 4，模拟目标 trn2/gen3）**

> ⚠ **服务端是 greedy 解码**（14:30 实测）：同一请求在 temperature 0.7 下 4/4 输出逐字相同，`n=4` 也一样，带 `seed` 返回 HTTP 500 并把引擎弄崩（117 因此重启过一次）。批次组成不同时输出会变，所以不是完全确定。结论：`--samples 4` 往往只是 1 个样本，`--repeat 5` 往往是同一条轨迹重放 5 次，所以下表多一列「不同轨迹数」（整条轨迹的代码序列不同的 run 数）。baseline 116：106 轮里 89 轮 4 个样本逐字相同。

| 实验 | commit | level | 座位 | solved | 5 次分数（解出的轮次） | 不同轨迹数 | reaudit | 决定 |
|---|---|---|---|---|---|---|---|---|
| baseline | 8f1ca41 前 | L1/L2/L3/L4 | 116 | 0/5、3/5、0/5、0/5 | 见上表 | L1 第 0 轮 5 次相同；L2–L4 第 0 轮各不相同 | L2 2 个 kernel PASS | 参照 |
| baseline 复现 | upstream 8f1ca41 | L1/L2/L3/L4 | 119 | 0/5、2/5、0/5、0/5 | L2 [1,.3,.3,1,.3]，L4 [.62×4,.5] | — | — | 和 116 一致 |
| E-A | 5ed7ec2（进程里是 26c43ed） | L1 | 116 | 0/5 | 全 0.30 | 1–2（日志没有 run 字段） | — | 作为铺路改动采用 |
| E-v3 | c39c0ce | L4 | 117 | 1/2（停在第 3 次） | 1.0（第 6 轮）、0.75 | 2 | PASS | 被 v7 取代 |
| E-F | ac258d2 | L1 | 116 | 0/5 | 全 0.30 | **1** | — | **不采用，已 revert（d97b5e4）** |
| **E-v7** | ce0403c | L3 | 117 | **5/5** | 全 1.0（都在第 1 轮） | 4 | PASS（13 次解出，2 个不同 kernel） | **采用（主线）** |
| **E-v7** | ce0403c | L4 | 118 | **5/5** | 全 1.0（都在第 3 轮） | **1** | PASS（20 次解出，1 个 kernel） | **采用（主线）** |
| E-v7 | ce0403c | L1 | 119 | 0/1（第 2 次时为了 E-div 停掉） | 0.50 | — | — | 进行中止 |
| E-v7 | ce0403c（按 SUBMISSION 的 clone 步骤部署） | L2 | 116 | 进行中（14:50 起） | | | | 回归检查 |
| E-div | 81549a0（v7 + feedback_v8.py） | L3 / L4 | 119 / 118 | L3 5/5，L4 5/5 | L3 都在第 1 轮，L4 都在第 3 轮 | L3 3，L4 5 | PASS（每级只有 1 个 kernel） | 样本变化让轨迹变多，但收敛到同一个解 |
| E-div | 81549a0 | L1 | 117 | 0/1（第 2 次时停掉） | 0.50 | — | — | — |
| 消融 A1–A5 | ce0403c（v7 底座，--repeat 1） | L3 / L4 | 119 / 118 | 只有 A2（CARD=theirs）在 L3 上没解出 | A1 L4 第 2 轮；A2、A3 L4 第 5 轮；A4、A5 L4 第 3 轮 | — | — | L3 靠的是 CARD=category；L4 不依赖单个开关 |
| v8（SKELETON 开） | 9a19bb0 | L4 / L3 / L1 | 118 / 119 / 117、119 | L4 0/2，L3 1/1（第 3 轮），L1 0/2 | L4 0.62 ×2 | — | — | **SKELETON 不采用**（模型把 <…> 填错，L4 越界卡死） |
| v8（81c37cf 之后） | 6f11031 | L1 | 116 | 1/1（第 1 轮） | 1.0 | — | PASS，能 lower 到 trn2 | 第一次 L1 模拟器解出 |
| L2 第 0 轮诊断 | 9a19bb0 / 原版 agent.py | L2 r0 ×20 | 116 / 119 | v8 0/20，v8p（原版首轮 prompt）0/20，**v8ps（原版 prompt + 原版采样）2/20，原版 agent.py 2/20** | — | — | — | **L2 退步的原因是 SAMPLING=qwen** |
| **v8.2（最终候选）** | **96a9fc9**，md5 c2f81a；V7.md 的 export + SKELETON=0 TRUNCFIX=1 L1FIX=1 MIXSAMP=1 L2HINT=1 MIX=1 | **L1** | 117 ×3，118 ×2 | **5/5** | 解出轮次 2、2、0、4、2 | 5 个 kernel | PASS ×5，都能 lower 到 trn2 | **采用** |
| **v8.2** | 96a9fc9 | **L2** | 116 ×3，119 ×2 | **3/5** | [.5, 1, .5, 1, 1]，解出轮次 4、2、1 | 3 个 kernel | PASS | 采用（≥ baseline，v7 是 0/3） |
| **v8.2** | 96a9fc9 | **L3** | 118 ×3，119，116 | **5/5** | 解出轮次 0、2、0、0、0 | 3 个 kernel | PASS | 采用 |
| **v8.2** | 96a9fc9 | **L4** | 119 ×5 | **5/5** | 都在第 2 轮 | 1 个 kernel（和 v7 相同） | PASS | 采用 |

日志：`runs/seat-<N>/<实验>/`（不进 git）。1.0 一律用 `scripts/reaudit.py` 复审。

**已确认的发现**
1. **每轮约 50 秒，慢在模型生成，不在评分；没有便宜的提速办法。** 12:50 在 seat-116 实测：1 条并发 13.9 tok/s，4 条并发总共 22.1 tok/s（每条 5.5）。模型确实在 Trainium 上跑（`neuron-ls` 有进程；`PJRT_DEVICE=CPU` 是 vllm_neuron 自己设的）。CPU 配额 11 核只用了约 5 核，节流约 1%，所以不是 CPU 配额卡住；`vllm._C` 缺失在 Neuron 上是正常的。可调的只有 `--optimization-level 3`（默认 O1）或 TP=4，都要重新编译、耗时未知、还会让 baseline 失效，**决定不改**。对策：多座位并行，一次只测一个 level；输出 token 贵、输入 token 便宜（prefill 约 270 tok/s）。
2. **失败分类**（5 次 run，424 次尝试，`analysis/baseline_seat116_summary.csv`）：
   - 拷贝两边大小不一致 96 次（L1 40、L2 30、L3 26）
   - 编造不存在的函数 80 次（全在 L1：`nisa.multiply`、`nisa.scalar_mul`）
   - 下标越界 64 次（L2/L3/L4）
   - tile 超过 128 行 55 次（L4 主要卡点）
   - 乱用 reshape 50 次（L3 48）、tile 只有一维 22 次（L3）
3. **harness 的反馈里已经附带了修改建议**（`agent.py` 的 `enrich()`），但同样的错误还是反复出现，说明现有建议没起作用。改反馈要从这里入手。
4. **port-forward 没有权限**，agent 只能在 pod 里跑；`kubectl cp` / exec 可以用。

**参考仓库：aws-neuron/neuron-agentic-development**（AWS 工程师推荐，已 clone 到 `../neuron-agentic-development/`，只读，不放进我们的仓库）
- 最有用的是 `skills/neuron-nki-docs/references/`：`indices/symbol-lookup.md`（全部 NKI 符号和所在模块）、`programming/api/*.md`（API 签名）、`debugging/error-codes/`
- 例子：`multiply` 存在，但它是 `nl.multiply`，是一个**操作类型**，要传给 `nisa.tensor_tensor(..., op=nl.multiply)` / `nisa.tensor_scalar(...)`，不是 `nki.isa` 里能直接调用的函数。现在 harness 按字母相似度推荐「scalar_engine…」，模型看了没用
- ⚠️ 文档对应 NKI 0.4.0，pod 里是 **0.6.0**。写进反馈的名字要先在 pod 里确认存在
- ⚠️ `references/downloads/*_nki_kernels.py`（average_pool2d、matmul、transpose2d）基本就是 level 1–4 的标准答案，**不能放进 prompt**（等于泄题），只给人看

**正在做 / 下一步**
1. [实验 A 已接进 `enrich()` 的 `KNOWN_FIXES`，commit 5ed7ec2，待在 seat-116 上验证 L1] 编造的名字只有 4 种（nisa.multiply 36、transpose_moving 13、nisa.scalar_mul 12、tile_size() 1）；0.6.0 里 `nl.multiply` 可直接调用，`nc_matmul` 的转置参数叫 `is_transpose`
2. 精简 API 卡片（level 1–4 用到的十几个函数，约 300 token）放进 prompt
3. liuyq 的 `feedback_v2.py` / `feedback_v3.py`（commit fc137e7）已经覆盖 L4「tile 超过 128 行」和「拷贝大小不一致」；她在 4090 上 L4 解出 5/10（原版 0/15），待在 pod 上确认
4. [硬件合法性检查：**已用真 nki 0.6.0 验证通过**（本机 Docker，按 `SETUP_PYTHON.md`）] 模拟器分配 tile 时不查上限，现在模拟时记录每个 `nl.ndarray/zeros/ones/full`，分区 >128、PSUM 每分区 >16 KiB、SBUF 每分区 >192 KiB 判「ILLEGAL ON HARDWARE」，优先于数值比较；跨多个 PSUM bank 合法，不判。验证：selftest 通过；4 个参考 kernel 全过（审计到 12/12/5/90 次分配）；「分配 (256,128) SBUF 但每次 DMA 只搬 128 行」的 L4 kernel，关审计真模拟器给 4/4（漏洞属实），开审计判 ILLEGAL；跨 2 个 bank 的 PSUM 不误伤。**修了一个坑（13:55 更正：比原来说的严重）**：nki 按文件路径缓存，同一进程里同一路径、**字节数相同**的后一个候选会被当成第一个来模拟——**数值和分配都是旧的**。实测：参考 L4 把 nc_matmul 两个操作数对调（长度不变，单独跑会报错），紧跟参考写到同一路径就判 4/4。所以 c39c0ce 之前同一进程连续评分的 run（baseline、4090 的 L4 5/10、117/116 早先的 run）都可能有**假解**，**任何 1.0 都要先用 `scripts/reaudit.py` 重审**。已重审：baseline L2（2 个 kernel）PASS；117 的 L4 v3 解 PASS。现在 `agent.py` / `feedback_v2.py` 每个候选写到唯一路径（`nkibench.candidate_path()`），v2 的 `locate()` 照常能指出出错行。**26c43ed 有这个坑，别用它跑实验**（或设 `NKIBENCH_NO_ALLOC_AUDIT=1`）。4090 上的 L4 5/10 是在没有审计时测的，复现时要重新审。
   - **13:35 座位占用**：115 跑 `daykit/.../feedback_v7.py`（L2、L9）；116 跑实验 A（L1，`attempts_expA_L1.jsonl`）；117 跑 L4 v3 `--repeat 5`（`attempts_v3_L4.jsonl`，13:13 启动，`agent.py`/`feedback_v2.py` 是旧版，**解出的 kernel 要重审**）；118 在重启 vLLM（新配置，编译中）；119 跑 `agent.py --all --repeat 5`。
   - **重审**：`python scripts/reaudit.py <attempts.jsonl ...>` 把每个 1.0 的 kernel 单独开新进程跑 `--check`（需要 nki：pod 或 SETUP_PYTHON.md 的 Docker）。已重审 baseline：4 次 L2 解出（2 个不同 kernel）全部 PASS，baseline 数字成立。
5. agent 加 token 分段统计（prompt 里规则说明 / 上次代码 / 报错 / 失败记录各多少）和置信度输出
6. 每个改动用 `--level X --repeat 5` 验证，变差就回滚；约 16:30 冻结，多座位并行跑最终对比，18:15 前提交

**文件索引**：`上手指南.md`（连座位的命令）· `scripts/connect.sh` · `scripts/sync.sh` · `scripts/attempts_to_csv.py` · `analysis/`（分类表）· `projects/02-kernel-agent/{agent.py,nkibench.py}`（要改的代码）

---

## 1. 评分标准（所有题目同一套）

| 权重 | 项目 | 要点 |
|---|---|---|
| **30%** | 正确性 | 在**评委保留的 shape 和恶意数值**上通过几级，不是你自己的测试。**违反规则 = 0 分**，不是扣分。 |
| **25%** | 交付成果 | 通过了几级 + 每级用了几次尝试。尝试次数少 = agent 设计好，而不是运气好。 |
| **25%** | 方法与诚实 | **agent 知不知道自己失败了？** 报 "verified" 但实际没过的，比诚实报失败**更差**。外加 token 预算的统计、失败分类（taxonomy）。 |
| **20%** | 演示与写作 | 要展示**一次失败和恢复**，不只是成功。别人能不能复现。 |

**拉开差距的两件事：**
- **校准（Calibration）**：agent 必须输出置信度，而且要准。"我验证不了 level 8" 比 "done" 然后撒谎更值钱。
- **失败分类**：跑完整个 ladder，收集所有错误 kernel，归成几类有名字的失败模式并给出计数。不需要加速器，是评委最想留下的东西。

> 「过 4 级 + 对另外 6 级做严谨失败分类」的队，会赢「声称过了 level 9 但拿不出验证」的队——这就是评分标准本身。

---

## 2. 必交物

**README Part 4（任何项目都要）：**
1. **你的 checker**，以及它接受/拒绝什么的理由（这是主办方要留下的东西）。
2. **attempt log**：每次尝试 + 分数（agent 已经写到 `attempts.jsonl`；`scripts/sync.sh <N> pull` 拉回本地 `runs/`）。
3. **一页说明**：跑了什么、在什么上跑、结果如何——**包括跑了几次、分布（spread）是多少**。

**CHALLENGE-kernel-agent.md（选 kernel agent 题时）：**
1. agent 本身
2. 验证 harness——**写明容差和理由**（"rtol=1e-5" 是答案，"看起来差不多" 不是）
3. **eval set**：测过的 shape 和数值，含恶意值。**强制**。
4. 失败分类（分组 + 计数）
5. **token 统计**：每次尝试的输入 token 数和花在哪（docs / 错误上下文 / 代码）。一张「每次尝试 token 分配」图是 demo 里最有价值的东西 → 本机 venv 已装 matplotlib。
6. 一页复现说明

---

## 3. 已知的坑（全部是实测出来的）

### 模型 / prompt
- **不要开 thinking（`--think`）。** Qwen3 实测：每轮 ~8s → **446s**，分数 0.30–0.62 → **0.00**，每个样本都 `finish_reason=length` 截断在 ~9,900 字符、没有代码。**加大 token 预算也没用，解法是缩短 prompt。** agent 什么都没返回时先查这里。
- **prompt 里不要堆规则。** 给一串约束 → 模型逐条自我审查、陷入循环、什么都不输出；不给约束 → 输出自信但违规的代码。**正确做法：先放开生成，让 verifier 抓违规，再回传「只改这一处」的单条指令。约束属于 verifier，不属于生成 prompt。**
- **verdict ≠ instruction。** 把 `line 16: calls banned max` 原样回传 → 同样的违规又出现；改写成「用显式循环替换 np.max/np.sum，其余不变」→ 一轮修好。
- **不要在反馈里给出目标值。** 给了正确系数，模型直接抄、什么都不推导。只说**哪里错、往哪个方向错**。
- **给工具，不给提示。** 模型算不了积分 → 给它一个自己调用的计算器（project 1 的 `tool_calc.py`）。
- **看似合理的 prompt 改进可能全面变差。** 加了一个「按 128 分块拷贝」的示例：level 2 从 2/5 → 0/5，level 4 从 0.62 → 0.30，已回滚。**没测量前不要加回去。**
- **失败会转移而不是消失**；分数可能在理解变好时反而下降（权重分配导致）。

### 测量
- **用 `--repeat N` 报成功率，不报最好的一次。** Level 2 同样设置 5 次：`[1.00, 1.00, 1.00, 0.50, 1.00]`，纯运气就能 0.5↔1.0。
- Level 1/3/4 五次结果完全一样（0 方差）= **能力墙**，改动效果可以干净归因；level 2 是**运气限制**，单次对比无意义。
- 三面墙其实是同一个 NKI 习惯用法——**tiling**：
  - L1：`SBUF and PSUM tensors must have at least 2 dimensions`（建了 1-D tile）
  - L3：`cannot reshape array of size 32768 into shape (1,64)`（reshape 而不是切片）
  - L4：`dma_copy dst partition dimension 256 exceeds maximum 128`（整个张量一个 tile）→ 0.62，4 个 shape 只过 1 个
- **说清楚哪些数字来自模拟器、哪些来自设备**；layer 2/3（真实延迟）还没做，目前所有数字都是 `nki.simulate` 的吞吐推算。
- 222 Flops/Byte 是 bf16 的 ridge，测试 shape 是 float32 → 结论只是参考值。

### 参数
- **`--context 8192`**（seat pod 里 server 就是 8192）。repair prompt 要装上一版 kernel + checker 指令 + 失败 ledger，4096 会挤掉回答空间。
- 本地 Qwen3：`--samples 4`（server 同时跑 4 条，几乎免费）。
- 共享 gpt-oss-20b：**greedy 解码** → `--samples 1`（多采样=完全相同的答案），`--terse 1`（长 prompt 会让它只推理不回答），重试原 prompt 毫无意义，`tools=` 参数无效，输入上限 8192 token，接近上限会**静默截断**——**每次调用都检查 `finish_reason`**，`max_tokens` 至少 2500。~4 req/s 全场共享。
- 8B 在 level 4 **赢了** 20B（0.62 vs 0.30）：gpt-oss 6 轮里 4 轮在 ~10,000 字符隐藏推理后返回空。

### 环境 / 集群
- **凭证只在当前终端有效，而且会过期。** 新开终端/标签要重新粘贴；`ExpiredToken` 时粘贴频道里最新的块。**不要写进任何文件，不要提交。**
- **没有 `kubectl port-forward` 权限**（RBAC 只给了 pods get/list/watch、pods/log、pods/exec）。本机调不到 pod 里的 `localhost:8000`，**agent 只能在 pod 里跑**。已在 seat-116 实测：`can-i create pods/portforward` = no，port-forward 报 `cannot create resource "pods/portforward"`。`kubectl cp` 双向可用（走 exec）。
- `kubectl exec -it` 后**等 `root@seat-N:/workspace#` 提示符出现再打字**，否则输入进的是本机 shell。
- 长任务一律 `nohup python agent.py ARGS > run.log 2>&1 < /dev/null &` + `tail -f run.log`，否则断线就没了。`pgrep -af agent.py` 看是否还在跑。
- **pod 被替换，`/workspace` 就没了** → 及时 push 到自己的 git 或 `scripts/sync.sh pull` 拉回本地。
- pod 里任何 git 命令前：`git config --global --add safe.directory /workspace`。
- NeuronCore 不能两个进程共享：vLLM 占 NC 2–3，NC 0–1 空闲；以后做设备上计时用 `NEURON_RT_VISIBLE_CORES=0,1`（未验证）。
- `nki` 只在 Trainium pod 里有，本机 import 失败是正常的。本机能跑的：`kernelbench.py`（Stage A，纯 NumPy）和 `nkibench.py --selftest`（跳过模拟部分）。

---

## 4. 接下来该跑的命令

### 本机（每个新终端）
```bash
# 1) 粘贴频道里的 macOS/Linux 凭证块（export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN）
# 2) 连上自己的座位
scripts/connect.sh <N>
kubectl exec -it seat-<N> -- bash
```

### pod 里，终端 1：起模型
```bash
neuron-ls
cd /workspace && ./serve.sh                      # ~4 分钟，等 READY；不要关
```

### pod 里，终端 2：跑 agent
```bash
git config --global --add safe.directory /workspace
cd /workspace/projects/02-kernel-agent
python nkibench.py --selftest                     # 先证明 harness 可信
python nkibench.py --level 4 --check reference_level4.py
nohup python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5 > run.log 2>&1 < /dev/null &
tail -f run.log                                   # 这就是 baseline，记下每级 solved x/5 和 spread
```

### 本机：同步
```bash
DRY_RUN=1 scripts/sync.sh <N> push             # 先看会推哪些文件
scripts/sync.sh <N> push                       # 本地改的 projects/ 文件 → pod /workspace
scripts/sync.sh <N> pull                       # pod 的 run*.log / *.jsonl → runs/seat-<N>/<时间>/
```

### 本机：不连集群也能做的 Stage A
```bash
source .venv/bin/activate
cd projects/02-kernel-agent
python kernelbench.py --selftest
python kernelbench.py --list
python kernelbench.py --level 1 --show
python kernelbench.py --level 1 --check my_kernel.py
```

### 建议顺序
1. 先跑 `--repeat 5` 的 baseline，记录数字（这是之后所有对比的参照）。
2. 只改 **checker 的反馈文案**（把 verdict 改成 instruction），一次只改一处，每次都 `--repeat 5`。
3. 从第一天就记 token 分配（docs / 错误 / 代码 / ledger），最后画图。
4. 持续收集错误 kernel，按失败模式归类计数。
5. 给 agent 输出加置信度，并检验它准不准。
