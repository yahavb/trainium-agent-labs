# 成员 B 工作交付记录

## 范围与源提交
- 分支：`grace/prompt-nki`。
- 提示词提交：`5dd32df9b946e24bb8452fe5142b1610174961a2`（Improve NKI prompts for levels 1 and 3）。
- 仅整理该提交和本次 L1/L3 验证；未修改 checker、评分规则、任务定义或队友文件，未运行全量测试、未合并 main。

## 提示词提交真实 diff
根据 `git show --numstat 5dd32df9b946e24bb8452fe5142b1610174961a2`：
| 文件 | 新增 | 删除 |
|---|---:|---:|
| `projects/02-kernel-agent/agent.py` | 11 | 1 |
| `projects/02-kernel-agent/kernel_guides.py` | 27 | 0 |
| `projects/02-kernel-agent/prompts.py` | 28 | 0 |
| **合计** | **66** | **1** |

- `prompts.py` 按 level 选择相关 NKI API 卡片，加入 Pmax 和 matmul free-dimension 限制；按 level 载入知识，不把所有 API 堆进每个提示。
- `kernel_guides.py` 的 L1 指引具体说明输入/输出形状、pool 窗口两轴归约及除以窗口面积、DMA 源/目标元素数匹配、partition 限制和边界自查。
- L3 指引具体说明 lhsT=(K,M)、rhs=(K,N)、输出=(M,N)，二维 SBUF/PSUM tile、HBM→SBUF→PSUM→SBUF→HBM 数据流；本关 K=128、M=64、N=512 可单 tile 完成。
- `agent.py` 在原接口 `first_prompt(level, terse=0)` 中接入 level context，普通和两种 terse 提示路径都保留 context。输出仍要求单个 Python 代码块。删除的一行是通用 API 卡片插入点，由分级 context 替换。

## 本次 L1/L3 验证
seat-22 app 容器，HEAD 为 `5dd32df9b946e24bb8452fe5142b1610174961a2`；Qwen3-8B 服务响应正常。每级单次运行参数：8 rounds、4 samples、context 8192、repeat 1。

| Level | rounds | 尝试记录 | Reward / 解决 | 正确性与执行 |
|---|---:|---:|---|---|
| L1 | 8/8 | 32 | 0.30；未解决（0/1） | 0/4 shapes 通过；解析和静态规则得分，NKI 模拟执行失败，正确性未通过。 |
| L3 | 5/8（重复错误提前停止） | 20 | 0.30；未解决（0/1） | 0/1 shape 通过；解析和静态规则得分，NKI 模拟执行失败，正确性未通过。 |

L1 代表性错误：`nl.Layout`、`nl.OpCode` 不存在；`ndarray(layout=...)` 不支持；`tensor_scalar` 的 `dtype=` 不支持且调用缺少必需参数。L3 代表性错误：`dma_copy(src_layout=...)` 不支持；之后反复把 SBUF 传作 `nc_matmul` 输出，但输出必须位于 PSUM；同一错误连续四轮后 agent 停止。

该项目当前通过 CPU 上的 `nki.simulate` 做检查，硬件延迟与 profile 测量层尚未实现。本次没有 kernel 性能数据；生成耗时不作为 kernel 性能。

## Baseline 与结论限制
仓库 `STATE.md` / 项目说明记录的 Qwen3-8B baseline 为 5 次重复运行，每次 8 rounds × 4 samples、context 8192。L1 和 L3 均为 0/5 解决、每次 Reward 0.30。本次每级只运行 1 次，单次配置相同但重复次数少于 baseline。观测 Reward 与 baseline 记录一致；不能据此证明提分或稳定性变化。**当前尚未证明提分。**

## 附带验证日志
- `projects/02-kernel-agent/validation-runs/5dd32df-20261010/l1-console.log`
- `projects/02-kernel-agent/validation-runs/5dd32df-20261010/l1-attempts.jsonl`
- `projects/02-kernel-agent/validation-runs/5dd32df-20261010/l3-console.log`
- `projects/02-kernel-agent/validation-runs/5dd32df-20261010/l3-attempts.jsonl`
