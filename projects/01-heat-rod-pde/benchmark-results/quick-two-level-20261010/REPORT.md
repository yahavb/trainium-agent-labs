# 每级一次真实 Qwen3-8B 对比

测试状态：完成。每个版本在 Level 0.1、Level 1.1 各运行一次，seed=0；总计四次求解。

比较 master 的 improved_agent.py 与 resilience 的 algorithm_agent.py（E 配置），不是默认运行方式的分支速度对比。

| 题目 | 版本 | 原始 Checker 最佳得分 | 完成状态 | 秒 | 模型请求 | 计算执行 | 缓存命中 | 输出 tokens |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Level 0.1 | master | 1.0 | solved | 31.1 | 3 | 1 | 0 | 734 |
| Level 0.1 | resilience | 1.0 | solved | 6.6 | 4 | 1 | 1 | 124 |
| Level 1.1 | master | 0.0 | budget_timeout | 180.0 | 10 | 26 | 0 | None |
| Level 1.1 | resilience | 1.0 | solved | 7.9 | 4 | 2 | 0 | 144 |

## 配置与评分

- 模型：真实 Qwen/Qwen3-8B；seat-85 环境配置的模型服务。
- samples=2，rounds=3，workers=2，每次请求 max_tokens=512，tool_steps=1。
- 每次求解外部限时 180 秒；优化 Agent 内部限时 177 秒，预留保存结果时间。
- temperature=0.6，top_p=0.95，thinking=false；transport retries=0。
- 模型随机种子未发送；题目 seed 固定。两级轮换先后顺序，运行之间不重叠。
- 按相同原始 Checker 评分；候选出现 1.0 即可被 Agent 判为 solved。
- 未启用额外验证，不将原始 1.0 宣称为通过增强验证。
- 原始 Agent/Checker/Calculator/题目文件在两个快照中的内容均相同；优化 Agent 是可选新增入口。

## 限制

这是固定两题、每版本一次的快速诊断。没有独立重复，不代表完整 Level 成功率或统计显著提升。
模型输出长度、请求顺序、服务负载和随机性均会影响耗时。失败或 timeout 不得作为完成同等任务的加速样本。
此次没有执行回归测试、官方 selftest 或新增增强验证模块，因此不能宣布整个项目 READY。

## 复现与证据

- master commit: b94759c18e91139a128ed260b6a7dd18ba59148d
- resilience commit: a4b58a5b0ec6f4cc96ab7d9517cb35c00b0cbb68
- 快照、配置、文件哈希：comparison.json；每次实际命令：各 case 目录 invocation.json。
- 每次 prompt、真实模型回答、tokens、工具调用及延迟：telemetry.jsonl 与 attempts.jsonl。
- 原始 Checker 候选评分：各 case 目录 original-checker-audit.json。
- 动态测试的脚本：quick_level_compare.py。
- 原始 Pod 路径：/tmp/heatrod-quick-20261010-api-aligned。
- 固定顺序：master-L0、resilience-L0、resilience-L1、master-L1。

## 本次 Level 1 失败现象

master 执行 26 次 Calculator 计算，但只有 3 种完全相同字符串的请求，分别重复 10、8、8 次。请求保留未绑定的 n，有的附加 Python for 语句；使用的 sin(n*pi*x) 也没有正确表达右端 Neumann 的半整数模式。完成的 4 个候选均属格式失败，代表输出为 X(x)T(t)。真实响应中 6 次达到 token 上限，超时时还有 1 个请求未结束，因此其完整 token 总量记为 null，已观测 completion tokens 为 3680。

优化 Agent 在两题均于第一轮得到 1.0；Level 0 有一次 Calculator 缓存命中。此小样本不能独立归因哪项优化有效。
