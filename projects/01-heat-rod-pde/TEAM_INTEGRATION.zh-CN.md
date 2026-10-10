# 小组共同开发入口

整合分支：`project1-team-integration`，目标主干：小组仓库的 `master`。

这个分支以 `challenge1-improvements` 的 `c3c16f6` 为基础，合入
`project1-experiment-workflow` 的 `200f6f7`。原成员分支保留，未直接更新 `master`。
Project 2 的 `bruce/instrumentation` 未纳入；新增的 `project1-repair-loop-results`
也暂未纳入，其中的反馈记忆与当前 Agent 有重叠，需按功能审查后取用。

## 两部分各负责什么

| 来源 | 保留的功能 | 入口 |
|---|---|---|
| challenge1-improvements | 保留最佳候选、错误记忆、服务错误重试、独立物理验证 | improved_agent.py |
| project1-experiment-workflow | 原版行为的详细日志、固定配置、重复运行、源码快照、失败分析 | run_experiments.py |

目前运行器仍启动 `agent.py`，不会因为合并了文件就自动测试 `improved_agent.py`。
两者的日志格式和重复语义也不同：运行器的 `--repeats` 重复同一个 problem seed；
改进 Agent 的 `--repeat` 会增加 problem seed。下一步统一对照入口时必须明确这点。

## 为什么额外验证单独记录

`validation.py` 来自 `challenge1-improvements`。原检查器评分后，只有满分候选再接受
独立检查：更多空间/时间点、接近 t=0 的时间点、更加密集的初始形状网格。

整合后的新日志增加：

- `grade.original_reward`、`grade.original_parts`：额外验证之前的原始评分；语法保护拒绝时为 null/空。
- `grade.validation_status`：passed、failed、not_run 或 syntax_rejected。
- `grade.reward`：改进 Agent 实际用于选择候选和判定完成的分数。

原始满分不等于额外验证通过。回归测试保留了一个原始评分 1.0、额外验证失败的高频反例。
旧日志没有这些新字段，不能假装旧实验已经记录了它们；可用原始答案做明确标注的事后复评。

## 统一的开发约定

主 PR 对应本整合分支。后续每个人从最新整合分支创建功能分支，小 PR 的目标设为
`project1-team-integration`。经另一位组员检查后合入；最终共同版本再经主 PR 合入 `master`。
这样成员的独立成果会在同一个可审查版本汇合。

PR 标题和正文统一英语，写明问题、改动、验证和局限。每个功能明确负责人，避免四人同时
改同一个 Agent 循环。正式实验在共享服务上排队；每人使用自己的代码目录。

## 当前验证与下一步

本次整合的 10 项验证/Agent 测试、11 项工作流测试，以及原始两个 checker 的 selftest 均通过。
`configs/guided.json` 是待测的可选提示词实验，默认关闭；其 pilot 在 seat-87 独立目录运行，
尚不能据此声称优化成功。完整过程见 `TUNING_WALKTHROUGH.zh-CN.md`。

后续统一 baseline/改进 Agent 的实验预算、源码快照和双层评分，保留失败与超时。取得真实
对照结果后更新主 PR，而不是为每次试验建立一份互不关联的最终项目。
