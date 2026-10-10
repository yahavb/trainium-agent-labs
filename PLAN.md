# PLAN：项目 2（NKI kernel agent）

> **v1.8，14:46。teoguo 已确认 Q1/Q4/Q5/Q6（见 §8）。** 这个文件只有总规划写。实验数字写在 NOTES §0，只由执行 1 写；这里只放任务、负责人、验收标准和决策。
> 时间锚点：16:30 开始写文档 · **17:00 冻结代码** · 17:00–17:45 最终跑 · **18:15 提交（teoguo 提 PR）** · 18:30 截止。

## 0. 目标

| | 内容 | 对应分数 |
|---|---|---|
| **底线**（解出几级都要交） | checker 和理由、eval set（含恶意值）、attempt log、失败分类、token 分配图、校准表、一次失败加恢复的完整过程、一页复现说明 | 方法与诚实 25% + 演示 20% |
| **冲分** | L1/L3/L4 里至少再解出一级，并且过保留测试集 | 正确性 30% + 交付 25% |

**参照（baseline）**：seat-116，10:50–12:43，`--rounds 8 --samples 4 --context 8192 --repeat 5`
L1 0/5（全 0.30）· L2 3/5（1,.3,1,.3,1）· L3 0/5（全 0.30）· L4 0/5（.62×4，.50）
复现：seat-119 用 upstream 8f1ca41 又跑了一遍（已跑完）：L1 0/5（全 0.30）· L2 2/5（1,.3,.3,1,.3）· L3 0/5（全 0.30）· L4 0/5（.62×4，.50），和上面一致。全量重新打分（设 trn2）420 次全部和日志一致，L2 两个解都过保留集 16/16（889bea9）。
路径缓存 bug（ded1ef2）：c39c0ce 之前的 run 可能有假解。baseline 不受影响：424 次尝试换新路径全部重新打分，结果和日志一致（b39989f）。
baseline 回溯校准（b39989f）：3 次解出置信度 0.90，保留集 16/16 VERIFIED；17 次没解出置信度 0；Brier 0.001（只算 3 次解出是 0.010）；「置信度 ≥0.5 但没过」0 次。
⚠ team 分支已经包含实验 A（5ed7ec2），所以 baseline 代码是 5ed7ec2 之前的版本。

## 1. 分工

| 谁 | 负责 | 不做 |
|---|---|---|
| 总规划（本会话） | PLAN.md、挑实验、决定采用还是回滚、16:30 起写 SUBMISSION.md | 不改代码、不碰 pod |
| 执行 1 | 唯一操作 seat-115~119 的会话：部署、跑实验、reaudit、拉日志、写 NOTES §0 的数字；实验需要的代码改动也由它提交 | 不改执行 2 的文件；不碰 liuyq 在 115 上的进程 |
| 执行 2（../tal-deliv） | 已完成：eval set eeb13e9、token 统计 a681b90、置信度/校准 7868c08→e7663a3、分类脚本 f41b84e、baseline 回溯校准 b39989f、CHECKER.md 草稿 ab075c1。接下来：E-A 重新打分、结果汇总脚本、v7 兼容性检查、失败加恢复素材、最终的表和图 | 不碰 pod；不再改 agent.py |
| teoguo | 拍板决策点、和 liuyq 对接、刷新凭证、18:15 提 PR | |
| liuyq | feedback_v7（**主线反馈**，取代 v3），已推进来（7417cf9）。14:22 起在 115 上**专攻 L1** | 新版本不覆盖 feedback_v7.py |
| 新手 ×3 | 17:45 后照一页说明从头复现一遍，报告卡在哪一步 | 不在关键路径上 |

## 2. 座位（13:44 执行 1 回报，14:00 更新）

| 座位 | 现在 | 代码 | vLLM | 空出来 |
|---|---|---|---|---|
| 115 | liuyq：daykit feedback_v7，L2 和 L9 各 ×5，两个进程共用一个服务（13:18 起）。L9 2/2 VERIFIED。另据 183b384 的提交说明：v7 写出的 L2/L3/L4/L9/L11 kernel 已经在**芯片上**跑对 | daykit 自己的代码，projects/ 是 upstream（没有审计） | READY，**max-num-seqs 8，和 baseline 不同** | 14:30 后，liuyq 的，不碰 |
| 116 | f41b84e 验证通过（13:53），改部署 735fa48；**14:15 起跑 E-F（L1 ×5）** | → 735fa48 + E-F | READY，同 baseline | ~15:00 |
| 117 | (a) E-v3 L4（run 1 第 6 轮 SOLVED）和 (b) 组员的 `agent.py --all` 都停掉（teoguo 14:00/14:05），先拉日志 | → 243864f（= ce0403c 的代码，不含 E-F） | READY，同 baseline | **v7 L3**，14:07 开跑 |
| 118 | O3 放弃，13:57 默认配置 READY（21.9 tok/s，和 baseline 一样）。13:59 组员又开了一个 `agent.py --all`，**组员同意停掉**（teoguo 14:10），先拉日志 | → 243864f/ce0403c | READY，同 baseline | **v7 L4** |
| 119 | 组员的 baseline 复现，4/5，跑完拉日志 | upstream 8f1ca41 → ce0403c | READY，同 baseline | ~14:05 后 **v7 L1** |

## 3. 时间表

| 时间 | 任务 | 负责 | 验收 |
|---|---|---|---|
| 14:00–14:05 | 停掉 117 的 (b)（先拉日志，按 PID 只停它） | 执行 1 | 只剩 v3 L4 一个进程 |
| 14:05–14:20 | 116/118/119 部署 735fa48 并验证 | 执行 1 | 每个座位一行：selftest、4 个参考 `--eval`、offline 8 个 VERIFIED + Brier、只有一个 vLLM 且配置同 baseline、没有别的 agent 进程 |
| ✅ 13:59 | v7 进 team/master（7417cf9）：feedback_v2–v7、ops07/08（L9–14）、V7.md（配置、命令、每处改动） | liuyq | |
| 14:00–14:20 | E-A 日志全量重新打分（E-A 进程加载的是有缓存 bug 的 26c43ed） | 执行 2 | 不一致条数；analysis/calibration_expA_L1.md |
| ✅ 14:00 | 结果汇总脚本 scripts/summarize.py（ce0403c）：吃多个座位的 attempts + verdicts，出每个 level 的 solved x/n、5 次分数、解出用几次尝试、token、保留集结论（markdown 表）。17:45 后要在 15 分钟内出最终数字，所以现在先用 baseline 和 E-A 测好 | 执行 2 | 对 baseline 的输出和 NOTES §0 的数字一致 |
| 14:02–14:30 | v7 兼容性检查：在 ce0403c 上离线跑通，有 prompt_split 和 verdicts；和不带 e7663a3 时的 prompt 逐字节相同 | 执行 2 | 回报提交号和证据 |
| **14:25** | **D1** v7 主线开跑 | 总规划 | |
| 14:15–15:00 | **第 2 轮 a：E-F 在 L1（116）** | 执行 1 | 按 §4 格式回报；第 1 个 run 结束先确认 prompt_split 和 verdicts.jsonl 都有 |
| 14:07–15:15 | **第 2 轮 b：v7 在 L3（117，14:07）、L4（118，14:12）、L1（119，14:14）各 ×5**，三个座位开跑前检查全过，命令照 V7.md，开跑前过 test_v7.py 并查模拟目标 | 执行 1 | 按 §4 格式回报；第 1 个 run 结束先确认 prompt_split 和 verdicts.jsonl 都有 |
| 14:30– | 115 的 v7 L2/L9 跑完后：拉日志，用我们的 nkibench 对 1.0 的 kernel 做 reaudit 和保留集检查 | 执行 1 | 这几次解出在审计后还算不算数 |
| ✅ 14:20 | 一键出报告 scripts/report.py（2f30c2e）：先打检查表（四种日志齐不齐、run 数、有没有混进别的调用），再出 summary、taxonomy、token 图。D2 和 17:45 都用它 | 执行 2 | |
| ✅ 14:26 | attempt log 收进仓库 analysis/logs/（9ea5407，2.0 MB，含 README 和凭证扫描，0 命中） | 执行 2 | |
| 14:27–15:10 | 照文档从头复现：全新 clone，只按 SUBMISSION §1 和 SETUP_PYTHON.md 做，记下卡住的地方和建议的改法 | 执行 2 | analysis/repro_dryrun.md |
| ✅ 14:24 | prompt 泄题检查（a9f52d8）：0 命中，阳性对照 95 处 | 执行 2 | |
| ~~14:21–15:10~~ | prompt 泄题检查：抓下 v7 发出的全部请求体，和教程 kernel、reference_level1-4.py、answers/ 逐行比对 | 执行 2 | analysis/prompt_leak_check.md，一句结论 |
| 14:30– | 115 上 liuyq 的 v7 L2/L9 跑完后：执行 1 拉日志，执行 2 用我们的 harness 全量重新打分（L9 由 ops07 注册） | 执行 1 → 执行 2 | 审计后还剩几次解出 |
| ✅ 14:37 | **D2a**：E-F 不进最终版（v7 下 L1 的主要失败不是参数错；解码是确定性的，反馈一改就可能改动 v7 已解出的轨迹）。revert ac258d2 | 总规划 → 执行 1 | HEAD 的 agent.py md5 = df0288 |
| ~14:40–14:55 | 116 的 E-F 一跑完就接着跑 **v7 L2 ×5**（ce0403c，不含 E-F），不等 D2a。部署照 SUBMISSION §1 从公开仓库 clone 到 /workspace/team，顺便在真 pod 上验证复现步骤 | 执行 1 | clone 成功没有、md5、按 §4 格式回报 |
| ✅ 14:37 | **D2（提前）：v7 采用为主线**。L3 5/5（都在第 1 轮解出）、L4 4/4（都在第 3 轮解出），全部 VERIFIED；L1 没变少。L2 的回归检查由 116 在跑 | 总规划 | |
| ⚠ 14:35 | **发现：服务实际上是 greedy 解码**（执行 1 实测）。并发 4 个、n=4，输出都逐字相同；带 seed 返回 500，117 的 vLLM 因此崩了一次，已重启。所以 `--samples 4` 和 `--repeat 5` 大多不是独立样本，v7 L4 的 4 次 run 轨迹逐字相同。**之后报告一律加「不同轨迹数」**；把 run 拆到多个座位跑，得到的也只是复制品，§4 的「提速」对 v7 作废 | 执行 1 | |
| 14:45–15:35 | **第 3 轮：E-div（v8）**：v7 上加一层 feedback_v8.py，只改一处：样本 1 的 prompt 不动，样本 k≥2 在末尾加 `(attempt k of n, run r)`，让 greedy 也能出不同的样本。4 个座位各跑一级 ×5：117 L1、118 L4、119 L3、116 L2 | 执行 1 | 按 §4 格式，多一列「不同轨迹数」 |
| **15:40** | **D3a** E-div 采用还是退回 v7：L3、L4 各 ≥4/5 且 L2 ≥2/5 就采用；L1 有解出算额外收获 | 总规划 | |
| 15:05–15:50 | **v7 逐项消融**：greedy 下每个变体跑 1 次就是那条轨迹。A1 PROMPT1=theirs、A2 CARD=theirs、A3 MESSAGES=v4、A4 REPAIR_PROMPT=theirs、A5 SAMPLING=theirs，各跑 L3 和 L4。回答「v7 里哪一项让 L3/L4 解出来」，方法分靠它；别的队在 L4 上做了 6 种反馈形式的对照 | 执行 1 | 一张表：变体 × L3/L4 的分数、第几轮解出、从第几轮和 v7 分叉 |
| 15:50–16:10 | 只在 liuyq 的 L1 版本到了的时候测它（L1 ×5）；没到就不加实验 | 执行 1 | 16:10 前出结果 |
| ✅ 14:05 | token 统计加 `--usage`：从 v7 的 USAGE_LOG 取服务器给的准确 token（494d9e9） | 执行 2 | |
| ✅ 14:09 | 模拟目标检查（530ab9a）：baseline 和 E-A 共 584 次在 trn2/trn3 下重新打分，和日志完全一致。但一般来说会有影响：宽 tile 在 trn3 下会成为假解。**规则：没设 trn2 的 run 里出现的 1.0，引用前要设 trn2 重新打分** | 执行 2 | |
| 14:10–14:40 | 失败加恢复素材，先做 E-v3 L4 run 1（第 6 轮解出）：每轮的 kernel 改动、checker 原话、指令原文、token、分数；最后那个 kernel 设 trn2 跑 reaudit 和保留集 | 执行 2 | analysis/recovery_v3_L4_run1.md |
| 15:00–16:00 | 失败加恢复素材（v7 的）：从日志里再挑 1–2 段完整过程（失败的 kernel → checker 原话 → 回传的指令 → 修好的那一轮），附每轮 token | 执行 2 | 每段能定位到 文件/run/level/round |
| 15:30 | 用第一份新格式日志出 token 分配图草稿 | 执行 2 | analysis/ 下 .png + .csv |
| **16:10** | **D3** 定最终版：包含哪些 commit、最终命令行 | 总规划 + teoguo | 写进 §5 |
| 16:15–16:35 | 最终版冒烟：1 个座位 `--all --repeat 1 --rounds 2` | 执行 1 | 4 个 level 跑完不崩，verdicts 4 条，token 字段齐 |
| 16:15–16:40 | 每级挑一个上交 kernel：对目前所有 v7 日志跑 V7.md「Before handing in」（compile_solves7.py 编译，pick_nki.py 挑选），check/device_check.py 上芯片验证（要先停掉那个座位的 vLLM） | **teoguo**（座位自定，不占 116–119） | nki_kernels/ 每级一个，写明是模拟器验证、编译通过还是芯片上验证 |
| ✅ 14:10 起 | 写 SUBMISSION.md（仓库根目录）。**14:46 起就集中写，不等 16:30**：据 teoguo 的情报，我们总分落后最多的就是写作 | 总规划 | 15:30 前能写的章节都写完；18:05 前 [[TBD]] 清零 |
| 16:40–17:00 | 冻结前检查，5 个座位逐个过 §5 清单 | 执行 1 | 5 行全绿 |
| **17:00** | **冻结**：打 tag `final`，之后只改文档 | 执行 1 打 tag | |
| 17:00–17:45 | 最终跑（分配见 §5） | 执行 1 | 每个 level 5 次。没跑完的按实际完成次数报 |
| 17:45–18:00 | 拉日志（attempts、verdicts、nki_verdicts、usage）；reaudit 所有 1.0；日志拷进 analysis/logs/final/（runs/ 不进 git）；最终日志里有更好的解就重跑 pick_nki.py（只编译，teoguo 跑） | 执行 1 | |
| 17:50–18:00 | 最终版失败分类、token 图、校准表 | 执行 2 | analysis/*_final.* |
| 17:50–18:10 | 数字填进 SUBMISSION.md，每个数字都要能指到文件 | 总规划 | |
| 18:05–18:10 | 提交前检查：全仓库扫凭证（AKIA、aws_secret、SESSION_TOKEN 等）；文档里的 [[TBD]] 全部填完；README 顶部加一行链接到 SUBMISSION.md；**在全新 clone 里确认 analysis/logs/final/ 的日志都在**（.gitignore 会悄悄吞掉 attempts.jsonl 和 *.log，只有 analysis/logs/ 下面例外，见 a0af434） | 总规划 | 扫描结果为空，全新 clone 里文件齐 |
| 18:10–18:15 | commit、push team/master；PR 提到 github.com/liuyq123/trainium-agent-labs，由 liuyq 提 | liuyq（teoguo 对接） | |
| 18:15–18:30 | 缓冲 | | |

## 4. 实验

**提速（14:26 起；14:40 更正）**：原计划把一个实验的 5 次 run 拆到多个座位同时跑。但 14:35 发现解码是 greedy，v7 的 5 次 run 大多是同一条轨迹，拆开跑只得到复制品，所以**对 v7 作废**。E-div 让样本真正不同以后，拆分才重新有意义。
不做的提速：O3（编译 33 分钟没好）、改 TP 或 max-num-seqs（都要重新编译，baseline 也会失效）、改 samples 或 rounds（会改变 agent 本身）。实测 batch 2 每步 79 ms、batch 4 每步 183 ms，总吞吐 batch 2 反而略高，但改 samples 就是改 agent，今天不动。

**规程**
1. 只改一处，一个 commit。v7 整体替换 v3 算一处。
2. `--level X --rounds 8 --samples 4 --context 8192 --repeat 5 --log E编号_LX.jsonl`。vLLM 配置和 baseline 相同，**每个座位只有一个 agent 进程**，代码 ≥ 735fa48。
3. 反馈里新出现的 API 名，先在 pod 上用 `inspect` 核对，核对输出贴进回报。
4. 1.0 的 kernel 全部跑 `scripts/reaudit.py`，再看 verdicts.jsonl 里保留测试集的结果。**c39c0ce 之前的代码跑出的日志，引用前要全部重新打分（不只是 1.0 的），因为路径缓存既会造成假解也会造成假失败。**
5. 回报一行：`E-编号 | commit | level | 座位 | solved x/5 | 5 次分数 | 解出时用了几轮 | 前 3 种失败（次数，和 baseline 同 level 比）| reaudit | 保留集 | 建议`

**判定规则**（事先定好，看到结果后不改）
- 目标 level solved 比参照多，并且全部过 reaudit：**采用**
- solved 相同，但目标失败类别的次数降了一半以上、没冒出新类别、平均分没降：**作为铺路改动采用**（写明理由）
- solved 变少，或平均分低于参照，或任意一次低于参照的最低分：**回滚**
- 其他情况：不采用，保持代码简单
- 改到共用部分（prompt 模板、ledger、repair prompt、停止规则）的，另在 L2 跑 ×5 做回归检查，solved ≥ 2/5 才算没变差
- 参照指该 level 当前采用的版本，一开始就是 baseline
- **底座固定**：我们的改动都叠在 v7 @ ce0403c 上。liuyq 之后的新版本必须作为新文件推进来（不覆盖 feedback_v7.py），15:15 前到，才能在第 3 轮测；否则不进最终版
- **v7 是一整套改动，按整体判**：L1/L3/L4 合计 solved 比 baseline 多，并且没有任何一级 solved 变少，就整体采用。有级别 solved 不变、平均分下降的，写进文档，不阻止采用。不按 level 拆着用（不搞「L3 用 v7、L1 用 agent.py」）

**结果和队列**

| 编号 | 改了什么 | level | 结果 | 决定 |
|---|---|---|---|---|
| E-A | `enrich()` 的 KNOWN_FIXES：把编造的名字换成 0.6.0 里真实的写法（5ed7ec2） | L1 | 0/5，全 0.30。编造函数 80→20；新出现 tensor_scalar() 缺参数 40 次（属于「函数参数用错」，不算新类别）；拷贝大小不一致 40→60。模型照反馈改了，然后卡在下一层 | **作为铺路改动采用** |
| E-v3 | 换成 feedback_v3（MESSAGES=v3 REPAIR_PROMPT=restructure） | L4 | **1/2**：run 1 第 6 轮解出（reaudit PASS，ded1ef2），run 2 跑满 8 轮最好 0.75；第 3 次刚开始就停了，座位让给 v7。日志 runs/seat-117/Ev3_L4/（100 条） | 被 v7 取代，只当证据和失败加恢复素材 |
| **E-v7** | 换成 liuyq 的 feedback_v7，配置和命令照 V7.md（PROMPT1=v2、CARD=category、MESSAGES=v5、REPAIR_PROMPT=restructure、SAMPLING=qwen、GATE=static、模拟目标 trn2） | L3（117）、L4（118）、L1（119），然后 L2 | L3 **5/5**（第 1 轮解出，第 0 轮有 3 种不同输出）；L4 **4/4**（第 3 轮解出，4 次轨迹逐字相同，run 5 在跑）；全部 VERIFIED。L1 进行中 0 解；L2 在 116 上跑 | **采用为主线**（14:37）；1.0 统一 reaudit |
| **E-div（v8）** | 样本 1 不动，样本 k≥2 的 prompt 末尾加 `(attempt k of n, run r)`；新文件 feedback_v8.py，叠在 v7 上 | L1–L4 各 ×5 | 第 3 轮，14:45 起 | D3a 15:40 |
| E-C | 「cannot reshape」改成一条指令：出错那一行用切片取 tile | L3 | 候选，看 v7 之后 L3 还剩不剩这一类 | |
| E-F | 函数缺参数、多参数或参数名不对时，反馈给出：出错那一行、运行时 inspect.signature 取到的真实签名、一句「按签名改这一处」 | L1 | 3 次跑完都是 0.30，三次轨迹相同（greedy）；wrong-signature 从 E-A 的每次 8 降到每次 3，主要失败变成 dma_copy 元素数不一致（每次 14） | **不进最终版**，revert ac258d2（14:37） |
| E-E | 「拷贝两边大小不一致」改成指出两边各自的 shape 和出错行 | 看情况 | 候选 | |

可选：liuyq 自己加的保留题 L9–L11 不在官方 ladder 里，但可以作为泛化的证据。前提是在我们的 harness 上 reaudit 过；只在有空座位时跑。

## 5. 最终跑（D3 填入口和命令）

入口：v7，export 照 V7.md（D3 确认）。公共参数：`--rounds 8 --samples 4 --context 8192`，不加 `--think`，`nohup … &`。

| 座位 | 任务 | 最坏耗时 |
|---|---|---|
| 116 | L2 ×5 → final_L2.jsonl | ~15 分钟（跑完就当备用座位） |
| 117 | L3 ×5 → final_L3.jsonl | ~37 分钟 |
| 118 | L4 ×5 → final_L4.jsonl | ~37 分钟 |
| 119 | L1 ×5 → final_L1.jsonl | ~37 分钟 |

**115 不能用**（teoguo 14:12），只有 4 个座位，没有现成的备用。L1/L3/L4 最坏都要 37 分钟左右（每轮约 55 秒，跑满 8 轮 × 5 次），17:00 开跑最晚 17:40 结束。L2 通常 15 分钟跑完，116 就当备用。某个座位出问题时，备用座位按剩下的次数补跑；17:45 还没跑完的，按实际完成次数如实报。座位和 level 的对应沿用第 2 轮，座位本身有问题容易看出来。

**冻结前检查**（每个座位一行）：只有一个 vLLM，READY，启动参数和 baseline 相同（TP2/8192/seqs 4）· 没有别的 agent 进程 · `git rev-parse HEAD` 等于 tag `final` · `--selftest` PASS · `--level 4 --eval reference_level4.py` 16/16 · V7.md 的 export 都设了（尤其 USAGE_LOG，最终 token 图靠它）· 旧日志已拉回 · 凭证新鲜。

## 6. 决策点

| 时间 | 决策 | 默认做法 |
|---|---|---|
| D1 14:25 | v7 开跑 | v7 没按时进仓库，就请 liuyq 同意后由执行 1 从 115 拷过来 |
| D2a ✅14:37 | E-F 不进最终版，revert | |
| D2 15:15 | v7 在每个 level 的结果；第 3 轮排什么 | 按 §4 的 v7 整体判定规则。liuyq 如果有比 v7 更新的版本，**15:15 前进 team/master** 才能在第 3 轮测 ×5；没测过的版本不进最终版 |
| D3a 15:40 | E-div 采用还是退回 v7 | L3、L4 各 ≥4/5 且 L2 ≥2/5 就采用 |
| D3 16:10 | 最终版包含哪些 commit | 只放过了判定规则的改动 |
| D4 16:35 | 冒烟不过怎么办 | 退回上一个冒烟通过的 commit |
| D5 17:00 | 冻结；哪些座位能用 | 不全绿的座位不用，它的 level 等 116 跑完 L2 后补跑 |
| D6 17:45 | 没跑完的怎么报 | 按实际完成次数报 x/n，不延长到 17:50 以后 |
| D7 18:10 | 提 PR | |

## 7. 风险

| 风险 | 对策 |
|---|---|
| v7 不在仓库里，配置只有 liuyq 知道 | 14:20 前由 liuyq 推进来；推迟就按 D1 的默认做法 |
| v7 的结果是在 upstream nkibench 上跑出来的（没有分配审计、没有保留集，还可能有路径缓存造成的假解，见 ded1ef2） | 一律在我们的 harness 上 reaudit 和跑保留集之后才计数 |
| 多个进程共用一个服务（117 上每轮慢到 3 倍）；组员自己在座位上开 run（117、118 都出现过） | 每个座位只跑一个 agent 进程，写进冻结前检查；teoguo 通知组员 115~119 先别自己开 run |
| 最终跑 45 分钟跑不完 | 只有 4 个座位，没法拆 L1；116 跑完 L2 当备用；没完成的如实报 |
| 凭证过期 | agent 是 nohup 在 pod 里跑的，不受影响，只影响拉日志。teoguo 16:45 和 17:40 各刷新一次 |
| pod 被替换，/workspace 丢失 | 每个实验结束马上 `sync.sh pull`；最终跑期间每 15 分钟拉一次 |
| agent.py 改动冲突 | 执行 2 不再改 agent.py；实验改动只由执行 1 提交 |
| 组合起来从没一起跑过 | 16:15 冒烟；最终数字以最终跑为准，和单个实验对不上就如实写 |
| 模拟目标不一致：v7 设了 trn2，baseline、E-A、E-F 没设 | **已解决**：执行 1 在 pod 上用探测 kernel 确认，不设变量时 nki 按硬件选的就是 trn2（gen3）；执行 2 在本机也查过，两种目标下 baseline 和 E-A 的分数相同（530ab9a）。「不设就是 trn3」只适用于没有 Neuron 设备的机器 |
| v7 的 `ask` 替换了 agent.py 的，attempts 里的 token 只是估算 | 精确数字以 USAGE_LOG 为准；token 图要用它（在 v7 兼容性检查里确认能对上每次尝试） |
| 两套置信度：agent.py 的 `--verdicts`（先给置信度，再跑我们的保留集）和 v7 的 NKI_VERDICTS（它自己的额外用例和编译器） | 默认两套都报，都拿我们的保留集结果来校准；D3 定哪一套写在正文 |

## 8. 规则和已做的决定

- 不开 `--think`
- prompt 里不堆规则，约束放进 checker
- 反馈只说改哪一处，不给答案。**例外（Q1）**：按 liuyq 的做法，允许反馈把出错那一处的循环直接写成代码，代码用模型自己的变量名和 shape 表达式。文档里要写明这一点
- neuron-agentic-development 里的教程 kernel 不进 prompt
- 写进反馈的 API 名先在 pod 上用 inspect 核对
- 凭证不写进任何文件
- 报成功率和分布（`--repeat 5`），不报最好的一次；写清楚哪些数字来自模拟器、哪些来自设备

| | 决定 |
|---|---|
| Q1 | 按 liuyq 的做法（teoguo，14:00） |
| Q2 Stage A | 不做（默认） |
| Q3 API 卡片放进 prompt | 我们自己不加；v7 里如果有，按 liuyq 的配置 |
| v7 首轮 prompt 里的例子 | 已核对（14:05）：CARD_REDUCE3D（channel mean）和 v4 的 row mean 都不是教程 kernel。和教程 avgpool 的共同部分只是 dma_copy → nl.sum → tensor_scalar 的骨架，L1 的关键写法 `.ap()` 窗口视图不在里面。文档里写明 prompt 里有这个例子 |
| 117 的 (b) | 停掉（teoguo 问过组员，14:00） |
| SBUF 上限 | 保留 192 KiB（NeuronCore-v2 的值；trn2 是 224 KiB）。比硬件严只会误拒，不会放过违规；CHECKER.md 写明。日志里出现落在 192–224 KiB 的拒绝时再改 |
| Q4 | PR 提到 github.com/liuyq123/trainium-agent-labs，由 liuyq 提（teoguo 14:52 更新） |
| Q5 | 仓库根目录放 SUBMISSION.md，README 顶部加一行链接 |
| Q6 | v7 取代 v3 |
| 最终跑座位 | 115 不能用，只用 116–119（teoguo 14:12） |
| 上交前三步（编译、挑 kernel、上芯片验证） | teoguo 跑（14:12） |
| liuyq 的分工 | 她在 115 上专攻 L1；她的 L1 版本 15:15 前用新文件名推进来，第 3 轮测；我们第 3 轮的座位优先给 L3/L4（14:22） |
