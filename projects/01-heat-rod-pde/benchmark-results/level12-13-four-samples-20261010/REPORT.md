# 最新 Agent：Level 1.2、1.3，各轮四个 sample

真实 Qwen/Qwen3-8B。每题 seed=0，samples=4，最多 3 轮，workers=2，max_tokens=512，tool_steps=1；每题限时 180 秒。
按原始 Checker 评分，任一候选达到 1.0 即可判为 solved；额外验证未运行。
Agent 为 algorithm_agent.py，features=cache,final,feedback,adaptive。

| 题目 | 状态 | 最佳原始得分 | 秒 | 请求数 | 计算执行 | 缓存命中 | 已记录候选 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Level 1.2 | solved | 1.0 | 110.4 | 14 | 0 | 0 | 8 |
| Level 1.3 | solved | 1.0 | 166.9 | 16 | 11 | 6 | 8 |

## Level 1.2 各候选

| 轮（从 0 开始） | sample | 原始得分 | 错误 |
| ---: | ---: | ---: | --- |
| 0 | 1 | 0.0 |  |
| 0 | 0 | 0.0 |  |
| 0 | 2 | 0.0 |  |
| 0 | 3 | 0.0 |  |
| 1 | 0 | 1.0 |  |
| 1 | 1 | 1.0 |  |
| 1 | 3 | 0.0 |  |
| 1 | 2 | 0.0 |  |

最终采用的最佳表达式：`sin(pi*x/6)*exp(-3/2*(pi/6)**2*t) + 2*sin(5*pi*x/6)*exp(-3/2*(5*pi/6)**2*t)`

Tokens：prompt=9341，completion=2665；未知用量请求=0。

## Level 1.3 各候选

| 轮（从 0 开始） | sample | 原始得分 | 错误 |
| ---: | ---: | ---: | --- |
| 0 | 0 | 0.6 |  |
| 0 | 1 | 0.6 |  |
| 0 | 2 | 0.4 |  |
| 0 | 3 | 0.0 |  |
| 1 | 0 | 1.0 |  |
| 1 | 1 | 1.0 |  |
| 1 | 2 | 1.0 |  |
| 1 | 3 | 1.0 |  |

最终采用的最佳表达式：`(32/pi**3)*sin(pi*x/2)*exp(- (pi**2/2)*t ) + (32/(27*pi**3))*sin(3*pi*x/2)*exp(- (9*pi**2/2)*t ) + (32/(125*pi**3))*sin(5*pi*x/2)*exp(- (25*pi**2/2)*t )`

Tokens：prompt=13852，completion=3566；未知用量请求=0。

## 复现与限制

代码版本：cd16f7b57a878661bb371f4c1ef09bd6864ff356。与前次 a4b58a5 的 Python 源码一致，新增提交只有证据文件。
只有每题一次运行，没有独立重复，不应宣称统计显著或覆盖整个 Level。本次只测试优化 Agent，没有同时跑 master 对照。
原始 1.0 不代表通过可选增强验证。没有执行全套回归测试或官方 selftest。
comparison.json 保存配置、环境、文件哈希、各题 summary 和各候选得分；invocation.json 保存实际命令。
evidence.zip 保存 prompt、真实回答、Checker 反馈、Calculator trace、请求延迟、token 用量、源代码与运行脚本。
原始 Pod 日志路径：/tmp/heatrod-level12-13-four-samples-20261010。
