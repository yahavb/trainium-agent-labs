# 本地 NKI 文档接入

`skills/neuron-nki-docs/` 来自下载的 AWS Neuron 文档 skill，保留了完整
`SKILL.md`、`references/` 和许可证。无需额外 pip 包，也无需访问网络。

`nki_docs.py` 读取 symbol-lookup 索引，按 level 选取 API；修复时优先检索反馈中的
API 名和 `NCC_*` 错误码，再使用候选代码中的 API 与 level 文档补充。
`agent.py` 在每轮模型请求前注入最多三个带来源路径的文档片段。
这是控制器自动检索，不是模型自主调用文件工具；不会把 SKILL.md 当成生成任务发送给模型。

在包含 Neuron SDK 和原有项目依赖的环境中，正常运行就会启用：

```bash
python agent.py --level 3 --context 8192 --rounds 8 --samples 4
```

默认文档上限 1600 字符，`terse=1/2` 时进一步缩为 800/400 字符。
注入时使用原 agent 的字符数/4 token 估算，预留 `--max-tokens` 的输出预算；
预算不足或文档不存在时不注入。此估算并非模型 tokenizer 的精确计数。
实际 SDK 签名、checker 限制和项目 reference 优先于下载文档。

自定义位置与大小：

```bash
python agent.py --level 3 --nki-docs /path/to/neuron-nki-docs --docs-chars 1200
```

也可以设置 `NKI_DOCS_DIR`，值指向含 `SKILL.md` 的目录。
部署到 `/workspace` 或 Kubernetes 时，把 `skills/`、`nki_docs.py` 和修改后的
`agent.py` 一起复制过去；仅复制 agent.py 无法加载检索模块。

关闭检索用于对照实验：

```bash
python agent.py --level 3 --no-nki-docs
```

每次尝试的 JSONL 增加 `docs_chars`，记录文档及附加格式的字符数；
`prompt_chars` 记录注入后的基础请求长度，不含各 sample 的策略后缀。

本地检索测试（无需 Neuron SDK）：

```bash
python -m unittest test_nki_docs
```
