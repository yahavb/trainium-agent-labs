# 黑客松笔记（Trainium agent labs）

**座位**：每人一个 pod，命令里的 `<N>` 换成**你自己的座位号**（队长 teoguo = seat-116）。不要进别人的座位。

> 来源：`README.md`、`STATE.md`、`projects/02-kernel-agent/README.md`、`projects/02-kernel-agent/CHALLENGE-kernel-agent.md`。
> 一句话核心：**决定 agent loop 好坏的是 checker，不是模型。** 「错了，偏差 341%」是真话但没用；「sin(pi·x) 这一项根本不该出现」才是能执行的指令。

---

## 0. 当前状态（2026-10-10 12:00，随时更新）

**人员**：teoguo（seat-116）+ liuyq（有经验）两人主力；另外三位新手做辅助，不计入关键路径。
**题目**：做项目 2（`projects/02-kernel-agent`，NKI kernel agent，在芯片上跑）。CHALLENGE（Stage A，`kernelbench.py`）有时间再做，同一套 agent 搬过去。评分按 30/25/25/20 那一套（CHALLENGE 第 239 行写着 "Same rubric as every problem"）。

**仓库和 remote**（本地 checkout：`trainium-agent-labs/`，工作分支 `team`）
| remote | 仓库 | 用途 |
|---|---|---|
| `team` | github.com/liuyq123/trainium-agent-labs | **共用仓库**，本地 `team` 分支跟踪 `team/master`，`git pull --rebase` / `git push` 直接用 |
| `origin` | github.com/teoguo/trainium-agent-labs | teoguo 的 fork，以后从这里给原仓库提 PR（从 `master` 开分支） |
| `upstream` | github.com/yahavb/trainium-agent-labs | 原仓库，只读 |

**Baseline**（seat-116，`--all --rounds 8 --samples 4 --context 8192 --repeat 5`，约 12:45 跑完）
```
          run1   run2   STATE.md 参考
level 1   0.30   0.30   0/5，全部 0.30
level 2   1.00✅ 0.30   4/5
level 3   0.30   0.30   0/5，全部 0.30
level 4   0.62   0.62   0/5，全部 0.62
```
日志拉回本机：`scripts/sync.sh 116 pull` → `runs/seat-116/latest/`（gitignore，不提交）。
重新生成分类表：`.venv/bin/python scripts/attempts_to_csv.py runs/seat-116/latest/projects/02-kernel-agent/attempts.jsonl -o analysis/baseline_seat116`

**已确认的发现**
1. **每轮约 50 秒，慢在模型生成，不在评分。** vLLM 总生成速度约 22 tok/s（4 条并发，每条约 5.6 tok/s），worker 一直占满 CPU，日志里有 `vllm._C` 缺失。这是服务端配置，我们不改。对策：一次只测一个 level（`--level X --repeat 5`，15–25 分钟），多个座位并行；输出 token 贵、输入 token 便宜（prefill 约 240 tok/s）。
2. **失败分类**（前 3 次 run，224 次尝试，`analysis/baseline_seat116_summary.csv`）：
   - 拷贝两边大小不一致 48 次（L1/L2/L3 都有）
   - 编造不存在的函数 48 次（全在 L1：`nisa.multiply`、`nisa.scalar_mul`）
   - 下标越界 32 次（L2/L3）
   - tile 超过 128 行 27 次（L4 唯一卡住的地方）
   - 乱用 reshape 21 次、tile 只有一维 16 次（L3）
3. **harness 的反馈里已经附带了修改建议**（`agent.py` 的 `enrich()`），但同样的错误还是反复出现，说明现有建议没起作用。改反馈要从这里入手。
4. **port-forward 没有权限**，agent 只能在 pod 里跑；`kubectl cp` / exec 可以用。

**参考仓库：aws-neuron/neuron-agentic-development**（AWS 工程师推荐，已 clone 到 `../neuron-agentic-development/`，只读，不放进我们的仓库）
- 最有用的是 `skills/neuron-nki-docs/references/`：`indices/symbol-lookup.md`（全部 NKI 符号和所在模块）、`programming/api/*.md`（API 签名）、`debugging/error-codes/`
- 例子：`multiply` 存在，但它是 `nl.multiply`，是一个**操作类型**，要传给 `nisa.tensor_tensor(..., op=nl.multiply)` / `nisa.tensor_scalar(...)`，不是 `nki.isa` 里能直接调用的函数。现在 harness 按字母相似度推荐「scalar_engine…」，模型看了没用
- ⚠️ 文档对应 NKI 0.4.0，pod 里是 **0.6.0**。写进反馈的名字要先在 pod 里确认存在
- ⚠️ `references/downloads/*_nki_kernels.py`（average_pool2d、matmul、transpose2d）基本就是 level 1–4 的标准答案，**不能放进 prompt**（等于泄题），只给人看

**正在做 / 下一步**
1. [进行中] 做「编造的名字 → 正确写法」对照表（从 baseline 收集 + 查官方文档 + pod 验证），接进 checker 的反馈 → `analysis/api_name_fixes.md`
2. 精简 API 卡片（level 1–4 用到的十几个函数，约 300 token）放进 prompt
3. 改反馈的优先级：L4「tile 超过 128 行」（离解出来最近）→「拷贝大小不一致」（影响三关）→ L1 编造函数
4. agent 加 token 分段统计（prompt 里规则说明 / 上次代码 / 报错 / 失败记录各多少）和置信度输出
5. 每个改动用 `--level X --repeat 5` 验证，变差就回滚；下午 3 点冻结，跑最终对比

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
