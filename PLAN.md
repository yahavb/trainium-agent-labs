# PLAN：项目 2（NKI kernel agent）

> **v1.3，14:10。teoguo 已确认 Q1/Q4/Q5/Q6（见 §8）。** 这个文件只有总规划写。实验数字写在 NOTES §0，只由执行 1 写；这里只放任务、负责人、验收标准和决策。
> 时间锚点：16:30 开始写文档 · **17:00 冻结代码** · 17:00–17:45 最终跑 · **18:15 提交（teoguo 提 PR）** · 18:30 截止。

## 0. 目标

| | 内容 | 对应分数 |
|---|---|---|
| **底线**（解出几级都要交） | checker 和理由、eval set（含恶意值）、attempt log、失败分类、token 分配图、校准表、一次失败加恢复的完整过程、一页复现说明 | 方法与诚实 25% + 演示 20% |
| **冲分** | L1/L3/L4 里至少再解出一级，并且过保留测试集 | 正确性 30% + 交付 25% |

**参照（baseline）**：seat-116，10:50–12:43，`--rounds 8 --samples 4 --context 8192 --repeat 5`
L1 0/5（全 0.30）· L2 3/5（1,.3,1,.3,1）· L3 0/5（全 0.30）· L4 0/5（.62×4，.50）
复现：seat-119 用 upstream 8f1ca41 又跑了一遍，13:44 时 4/5 完成：L2 2/4，L1/L3/L4 和上面完全一样。
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
| liuyq | feedback_v7（**主线反馈**，取代 v3）：推进 team/master，给出最终配置 | |
| 新手 ×3 | 17:45 后照一页说明从头复现一遍，报告卡在哪一步 | 不在关键路径上 |

## 2. 座位（13:44 执行 1 回报，14:00 更新）

| 座位 | 现在 | 代码 | vLLM | 空出来 |
|---|---|---|---|---|
| 115 | liuyq：daykit feedback_v7，L2 和 L9 各 ×5，两个进程共用一个服务（13:18 起）。L9 2/2 VERIFIED。另据 183b384 的提交说明：v7 写出的 L2/L3/L4/L9/L11 kernel 已经在**芯片上**跑对 | daykit 自己的代码，projects/ 是 upstream（没有审计） | READY，**max-num-seqs 8，和 baseline 不同** | 14:30 后，liuyq 的，不碰 |
| 116 | f41b84e 验证通过（13:53），改部署 735fa48；**14:15 起跑 E-F（L1 ×5）** | → 735fa48 + E-F | READY，同 baseline | ~15:00 |
| 117 | (a) E-v3 L4（run 1 第 6 轮 SOLVED）和 (b) 组员的 `agent.py --all` 都停掉（teoguo 14:00/14:05），先拉日志 | → ce0403c | READY，同 baseline | **v7 L4** |
| 118 | O3 放弃，13:57 默认配置 READY（21.9 tok/s，和 baseline 一样）。13:59 组员又开了一个 `agent.py --all`，**组员同意停掉**（teoguo 14:10），先拉日志 | → ce0403c | READY，同 baseline | **v7 L3** |
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
| 14:10–15:15 | **第 2 轮 b：v7 在 L4（117）、L1（119）、L3（118）各 ×5**，命令照 V7.md，开跑前过 test_v7.py 并查模拟目标 | 执行 1 | 按 §4 格式回报；第 1 个 run 结束先确认 prompt_split 和 verdicts.jsonl 都有 |
| 14:30– | 115 的 v7 L2/L9 跑完后：拉日志，用我们的 nkibench 对 1.0 的 kernel 做 reaudit 和保留集检查 | 执行 1 | 这几次解出在审计后还算不算数 |
| **15:05** | **D2** v7 每个 level 对比 baseline，采用还是回滚；排第 3 轮 | 总规划 | |
| 15:10–16:10 | **第 3 轮（最后一轮）**：v7 L2 ×5（我们的 harness）+ 针对最大剩余卡点的一处改动。16:10 没出结果的不进最终版 | 执行 1 | 按 §4 格式回报 |
| ✅ 14:05 | token 统计加 `--usage`：从 v7 的 USAGE_LOG 取服务器给的准确 token（494d9e9） | 执行 2 | |
| 14:08–14:30 | 模拟目标检查：baseline 和 E-A 在设 trn2、不设 trn2 两种情况下各重新打分一遍，看结果变不变 | 执行 2 | analysis/sim_target_check.md，一句结论 |
| 15:00–16:00 | 失败加恢复素材：从日志里挑 2–3 段完整过程（失败的 kernel → checker 原话 → 回传的指令 → 修好的那一轮），附每轮 token | 执行 2 | 每段能定位到 文件/run/level/round |
| 15:30 | 用第一份新格式日志出 token 分配图草稿 | 执行 2 | analysis/ 下 .png + .csv |
| **16:10** | **D3** 定最终版：包含哪些 commit、最终命令行 | 总规划 + teoguo | 写进 §5 |
| 16:15–16:35 | 最终版冒烟：1 个座位 `--all --repeat 1 --rounds 2` | 执行 1 | 4 个 level 跑完不崩，verdicts 4 条，token 字段齐 |
| 16:30– | 写 SUBMISSION.md（仓库根目录，README 顶部加一行链接过去），最终数字先空着 | 总规划 | 骨架和已有数字 |
| 16:40–17:00 | 冻结前检查，5 个座位逐个过 §5 清单 | 执行 1 | 5 行全绿 |
| **17:00** | **冻结**：打 tag `final`，之后只改文档 | 执行 1 打 tag | |
| 17:00–17:45 | 最终跑（分配见 §5） | 执行 1 | 每个 level 5 次。没跑完的按实际完成次数报 |
| 17:45–18:00 | 拉日志；reaudit 所有 1.0；日志拷进 analysis/final/（runs/ 不进 git） | 执行 1 | |
| 17:50–18:00 | 最终版失败分类、token 图、校准表 | 执行 2 | analysis/*_final.* |
| 17:50–18:10 | 数字填进 SUBMISSION.md，每个数字都要能指到文件 | 总规划 | |
| 18:10–18:15 | commit、push team/master、提 PR | teoguo | |
| 18:15–18:30 | 缓冲 | | |

## 4. 实验

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

**结果和队列**

| 编号 | 改了什么 | level | 结果 | 决定 |
|---|---|---|---|---|
| E-A | `enrich()` 的 KNOWN_FIXES：把编造的名字换成 0.6.0 里真实的写法（5ed7ec2） | L1 | 0/5，全 0.30。编造函数 80→20；新出现 tensor_scalar() 缺参数 40 次（属于「函数参数用错」，不算新类别）；拷贝大小不一致 40→60。模型照反馈改了，然后卡在下一层 | **作为铺路改动采用** |
| E-v3 | 换成 feedback_v3（MESSAGES=v3 REPAIR_PROMPT=restructure） | L4 | 跑完 1 次，1/1，第 6 轮解出，reaudit PASS（ded1ef2）。14:05 停掉，让座位给 v7 | 被 v7 取代，只当证据和失败加恢复素材 |
| **E-v7** | 换成 liuyq 的 feedback_v7，配置和命令照 V7.md（PROMPT1=v2、CARD=category、MESSAGES=v5、REPAIR_PROMPT=restructure、SAMPLING=qwen、GATE=static、模拟目标 trn2） | L4（117）、L1（119）、L3（118），然后 L2 | 14:10 起，**commit ce0403c，在 E-F 之前**（v7 会调用 agent.enrich()，用 ≥ac258d2 的版本就会把 E-F 一起带进去） | D2 决定。v7 和 E-F 都采用的话，第 3 轮在 L1 上测 v7 + E-F |
| E-C | 「cannot reshape」改成一条指令：出错那一行用切片取 tile | L3 | 候选，看 v7 之后 L3 还剩不剩这一类 | |
| E-F | 函数缺参数、多参数或参数名不对时，反馈给出：出错那一行、运行时 inspect.signature 取到的真实签名、一句「按签名改这一处」 | L1 | ac258d2，116 上 14:04 开跑，约 14:45 出结果。参照 E-A（重新打分后）：wrong_signature 60、拷贝大小不一致 60 | D2 决定 |
| E-E | 「拷贝两边大小不一致」改成指出两边各自的 shape 和出错行 | 看情况 | 候选 | |

可选：liuyq 自己加的保留题 L9–L11 不在官方 ladder 里，但可以作为泛化的证据。前提是在我们的 harness 上 reaudit 过；只在有空座位时跑。

## 5. 最终跑（D3 填入口和命令）

入口：v7，export 照 V7.md（D3 确认）。公共参数：`--rounds 8 --samples 4 --context 8192`，不加 `--think`，`nohup … &`。

| 座位 | 任务 | 最坏耗时 |
|---|---|---|
| a | L1 ×3 → final_L1a.jsonl | ~22 分钟 |
| b | L1 ×2 → final_L1b.jsonl | ~15 分钟 |
| c | L2 ×5 → final_L2.jsonl | ~15 分钟（跑完就当备用座位） |
| d | L3 ×5 → final_L3.jsonl | ~37 分钟 |
| e | L4 ×5 → final_L4.jsonl | ~37 分钟 |

L1 每次都跑满 8 轮，最慢，所以拆到两个座位。最坏耗时按每轮约 55 秒、一个服务只有一个 agent 进程估算。

**冻结前检查**（每个座位一行）：只有一个 vLLM，READY，启动参数和 baseline 相同（TP2/8192/seqs 4）· 没有别的 agent 进程 · `git rev-parse HEAD` 等于 tag `final` · `--selftest` PASS · `--level 4 --eval reference_level4.py` 16/16 · V7.md 的 export 都设了（尤其 USAGE_LOG，最终 token 图靠它）· 旧日志已拉回 · 凭证新鲜。

## 6. 决策点

| 时间 | 决策 | 默认做法 |
|---|---|---|
| D1 14:25 | v7 开跑 | v7 没按时进仓库，就请 liuyq 同意后由执行 1 从 115 拷过来 |
| D2 15:05 | v7 在每个 level 的结果；第 3 轮排什么 | 按判定规则。v7 没有任何一级变差就作为主线 |
| D3 16:10 | 最终版包含哪些 commit | 只放过了判定规则的改动 |
| D4 16:35 | 冒烟不过怎么办 | 退回上一个冒烟通过的 commit |
| D5 17:00 | 冻结；哪些座位能用 | 不全绿的座位不用，任务挪到 c |
| D6 17:45 | 没跑完的怎么报 | 按实际完成次数报 x/n，不延长到 17:50 以后 |
| D7 18:10 | 提 PR | |

## 7. 风险

| 风险 | 对策 |
|---|---|
| v7 不在仓库里，配置只有 liuyq 知道 | 14:20 前由 liuyq 推进来；推迟就按 D1 的默认做法 |
| v7 的结果是在 upstream nkibench 上跑出来的（没有分配审计、没有保留集，还可能有路径缓存造成的假解，见 ded1ef2） | 一律在我们的 harness 上 reaudit 和跑保留集之后才计数 |
| 多个进程共用一个服务（117 上每轮慢到 3 倍）；组员自己在座位上开 run（117、118 都出现过） | 每个座位只跑一个 agent 进程，写进冻结前检查；teoguo 通知组员 115~119 先别自己开 run |
| 最终跑 45 分钟跑不完 | L1 拆两个座位；c 先跑完当备用；没完成的如实报 |
| 凭证过期 | agent 是 nohup 在 pod 里跑的，不受影响，只影响拉日志。teoguo 16:45 和 17:40 各刷新一次 |
| pod 被替换，/workspace 丢失 | 每个实验结束马上 `sync.sh pull`；最终跑期间每 15 分钟拉一次 |
| agent.py 改动冲突 | 执行 2 不再改 agent.py；实验改动只由执行 1 提交 |
| 组合起来从没一起跑过 | 16:15 冒烟；最终数字以最终跑为准，和单个实验对不上就如实写 |
| 模拟目标不一致：v7 设了 NEURON_PLATFORM_TARGET_OVERRIDE=trn2，baseline、E-A、E-F 没设（V7.md 说不设就模拟 trn3） | 执行 1 开跑前在 pod 上查清；如果默认不是 trn2，就让执行 2 设 trn2 重新给 baseline 打分，看结果变不变。文档里写明每个数字的模拟目标 |
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
| Q4 | teoguo 提 PR |
| Q5 | 仓库根目录放 SUBMISSION.md，README 顶部加一行链接 |
| Q6 | v7 取代 v3 |
