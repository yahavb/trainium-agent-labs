# 小组 Project 1：从运行示例到可复现实验

我们的起点是活动方的 heat-rod agent。研究问题是：**把修复反馈按检查项组织，能否让小模型更稳定、用更少轮数解出 level 1.3？** `structured` 是待验证的方案，不是已经证明有效的优化。

## 代码在哪里，在哪里运行？

| 位置 | 用途 |
|---|---|
| GitHub `ChenYujunjks/trainium-agent-labs` | 小组代码仓库 |
| Mac `/Users/linqianyang/Documents/trainium-agent-labs-project1` | 编辑、提交代码 |
| seat-87 `/workspace/trainium-team87` | 下载小组代码、运行真实实验 |

三个位置不会自动同步。推荐流程是：本地修改 → commit/push → 服务器 pull → 运行 → 下载结果。GitHub 保存代码；Trainium 服务器运行模型。终端出现 `root@seat-87` 时，你在服务器；否则检查自己是否还在 Mac。

## 先理解四个文件

- `level1_heatrod.py`：出题。`u(x,t)` 表示位置 x、时间 t 的温度。
- `agent.py`：请求模型生成候选，选择得分最高的答案，用反馈安排下一轮。
- `pdecheck.py`：检查方程、两端边界、初始温度，提供分数和诊断。
- `tool_calc.py`：计算器。模型决定计算哪个积分，SymPy 负责计算。

当前优化的是提示和反馈，不更新模型权重。Agent 和 checker 在 CPU 上运行，模型服务使用 Trainium；不需要为了运行本项目而手动启动剩余芯片核。

## 第一次在服务器准备小组版本

在 **seat-87 终端**执行（第一次 clone 即可；已有同名目录时先检查内容）：

```bash
git clone --branch project1-experiment-workflow https://github.com/ChenYujunjks/trainium-agent-labs.git /workspace/trainium-team87
cd /workspace/trainium-team87/projects/01-heat-rod-pde
python level0_heatrod.py --selftest
python level1_heatrod.py --selftest
```

沿用已经启动的模型服务。seat 的 shell 通常已设置 `HEATROD_BASE_URL` 和 `HEATROD_MODEL`。如果新入口提示 URL 未设置，请核对活动环境；本地 `localhost` 与服务器 `localhost` 不是同一台机器。

## 参数改哪里？

编辑 `configs/baseline.json` 或复制一份配置。标准配置如下：

| 配置项 | 意思 |
|---|---|
| `level: 1, sub: 3` | 固定研究 level 1.3 |
| `seed: 0` | 固定题目生成和 checker 采样，不固定模型生成 |
| `samples: 4` | 每轮并发生成 4 个候选，与芯片核数无关 |
| `rounds: 4` | 最多尝试 4 轮，成功会提前结束 |
| `max_tokens: 1200` | 每次模型回复的 token 上限，不是上下文总长度 |
| `tool_steps: 1` | 每个候选允许 1 轮计算器交互；一轮可包含多个表达式 |
| `no_tools: false` | 允许模型使用计算器；改成 true 可做消融实验 |
| `think: false` | 沿用不启用额外思考的设置 |
| `feedback_style` | baseline 原版修复提示；structured 分项修复提示 |
| `tool_prompt_style` | baseline 原版工具说明；concise 请求只输出计算表达式和最终答案 |
| `repeats: 5` | 完整实验重复 5 次 |

同一个对照实验只改一个因素。baseline 与 structured 只有 `feedback_style` 不同；它们沿用相同题目、采样数、轮数、工具设置和评分标准。

`configs/concise_tools.json` 是另外一个独立实验，与 baseline 只差 `tool_prompt_style`。
它测试能否减少工具请求阶段的输出，保留相同 1200 token 上限、采样数、修复反馈和计算器。
输出更短可能影响正确率，所以不能只比较速度。不要在同一次比较里同时启用 structured 和 concise。

## 先做短实验，再做正式比较

在 **服务器项目目录**先运行一次原版，确认环境：

```bash
python -u run_experiments.py --config configs/baseline.json --repeats 1
```

正式运行原版（5 次）：

```bash
nohup python -u run_experiments.py --config configs/baseline.json > /tmp/heatrod-baseline.log 2>&1 < /dev/null &
tail -f /tmp/heatrod-baseline.log
```

`Ctrl+C` 只退出 tail。等基线实验完成后，再运行实验方案，避免同时请求模型影响时间比较：

```bash
nohup python -u run_experiments.py --config configs/structured.json > /tmp/heatrod-structured.log 2>&1 < /dev/null &
tail -f /tmp/heatrod-structured.log
```

若要结束后台实验，先用 `pgrep -af '[r]un_experiments.py'` 找到本次 runner 的 PID，
再 `kill -TERM 实际PID`。Runner 会结束它启动的 agent 子进程，并把本次批次标记为未完成。
模型服务保持运行。

每次启动都会建立独立结果目录，开头打印 `Results directory`。不会覆盖前一次结果。先用 5 次做探索，样本很少，不要把小幅变化说成确定提升。如果原版总在首轮成功，修复提示根本没有使用，两种方案不能据此比较；应另外选取双方相同、确实需要重试的题目设置，并单独报告。

### 优先排查生成耗时的短实验

已观察的一次新版本基线中，生成和工具耗时 105.5 秒，checker 仅 0.2 秒；
正确候选首次请求输出 959 tokens，用时 97.8 秒。所有回复正常 stop，没有截断。
这是用户提供的单次日志，不是稳定性能结论，完整记录见 [观察记录](OBSERVATIONS.zh-CN.md)。

简洁工具请求的第一轮实验已完成：每轮约 43 秒，但四轮都未解出，完整运行 173.7 秒、
总输出 3632 tokens。原版单次为 106.3 秒、2313 tokens，首轮成功。
这不足以证明总体差异，但不支持采用简洁方案作为默认优化。
两个方案的单次结果和局限见 [英文实验记录](EXPERIMENT_RESULTS.md)。

下一步先保留原版工具说明，单独测试 structured 修复反馈：

```bash
python -u run_experiments.py --config configs/structured.json --repeats 1
```

若首轮失败，检查 structured 是否在后续轮修正了指数；若首轮成功，说明这次没有用到修复反馈。
随后对可比较方案串行重复实验。检查成功率、总耗时、总输出 tokens 和截断次数，保留失败运行。
输出长可能包含有效推导，不能预先认定全部是冗余内容。

## 结果保存在哪里？

```text
runs/live-baseline-时间-随机后缀/
  config.json          本次实际配置
  environment.json     Python、依赖版本、模型名称、endpoint 主机名
  source/              本次执行的 Python 代码快照
  run-01/
    command.json       实际命令
    console.log        人可读的输出
    attempts.jsonl     每个候选一行的完整尝试记录
  run-02/ ...
  summary.json         成功率、成功时轮数中位数、所有运行耗时中位数
```

实际执行的是 `source/` 快照，所以运行期间更新工作目录不会改变已经开始的实验。

`attempts.jsonl` 记录答案、得分、各检查项、反馈、每次 API 请求的实际提示和回答、计算器表达式与结果、API token 用量（服务若返回）及截断状态。每轮还拆分 `generation_and_tools_seconds` 与 `checker_seconds`。这两个是**整轮时间**，在本轮候选记录里重复保存，分析时按 round 去重；多个候选的 API 时间并行发生，不能求和当作整轮延迟。

`summary.json` 只对完成的真实批次给出 solve_rate；进程失败或批次未完成时为 null，不能把失败运行丢掉再计算好看的成功率。`failed_checks_by_candidate` 统计失败候选的检查项，适合定位问题，不是独立实验样本数。

每次运行还汇总 `model_requests`、`completion_tokens`（API 未提供用量时为 null）、
`truncated_replies` 以及按 round 去重后的生成/工具与 checker 时间。
输出 tokens 是所有候选、所有模型请求的总和；并发请求耗时不能相加当作用户等待时间。

下载结果，在 **Mac 终端**执行：

```bash
kubectl cp seat-87:/workspace/trainium-team87/projects/01-heat-rod-pde/runs /Users/linqianyang/Documents/heat-rod-results-01
```

下一批结果换一个本地目录名称。`runs/` 默认被 Git 忽略，先保存在本地，再按活动要求提交结果；不要只保留在临时 pod 中。

## 本地修改后怎么更新服务器？

我们在本地编辑并推送小组分支后，在 **服务器仓库目录**执行：

```bash
cd /workspace/trainium-team87
git pull --ff-only
cd projects/01-heat-rod-pde
```

服务器主要用于运行，代码尽量在本地编辑。若 pull 提示本地修改冲突，先保存修改、检查差异，再处理；不要直接丢弃。

## 没有服务器时怎样学习？

在 **Mac 仓库根目录**安装到独立虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python projects/01-heat-rod-pde/run_experiments.py --offline --repeats 1
```

离线模式使用程序预设的假回答，只用于学习循环、验证日志流程。结果目录标记为 offline，正式成功率留空，不能用来声称模型有效。

## 小组接下来做什么？

1. 跑一轮真实基线，找出一次失败和下一轮修复。
2. 对照源码理解：错误来自系数、衰减速度、边界，还是答案格式？
3. 串行运行 baseline 与 structured，各重复 5 次。
4. 分析成功率、轮数、生成与检查耗时，并阅读失败案例。提升不了也要如实报告。
5. 用 `EXPERIMENT_NOTE.zh-CN.md` 整理一页说明，连同 checker 和完整尝试日志提交。

先掌握 Python 函数、字典、JSON、命令行参数和日志，再按具体失败补热方程知识。当前不需要研究 NKI kernel 或多核并行。
