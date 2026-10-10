# 编造的 API → 正确写法（对照表）

> 来源：seat-116 baseline 前 3 次 run 的 224 次尝试（`analysis/baseline_seat116_attempts.csv`）。
> 每一条都在 pod 里用 `inspect.signature` 对 **nki 0.6.0** 核对过；文档出处指 `../neuron-agentic-development/skills/neuron-nki-docs/references/`（文档版本 NKI 0.4.0）。
> 用途：替换 `projects/02-kernel-agent/agent.py` 里 `enrich()` 的兜底逻辑。现在遇到不存在的名字时，它用 `difflib` 按字母相似度推荐（例如 `scalar_engine, tensor_scalar_cumulative`），模型看了还是不知道该怎么写。

## 1. baseline 里实际出现的错误

| 模型写的 | 次数 | level | 0.6.0 的真实情况 | 建议的反馈（verdict → instruction） |
|---|---|---|---|---|
| `nisa.multiply(...)` | 36 | 1 | `nki.isa` 里**没有** `multiply`。`multiply` 在 `nki.language` 里：`nl.multiply` 可以作为操作类型传给 nisa 指令，也可以直接调用 `nl.multiply(x, y)` | `multiply` is not in nki.isa. To scale a tile by a constant write `nisa.tensor_scalar(dst=out, data=t, op0=nl.multiply, operand0=c)`; to multiply two tiles write `nisa.tensor_tensor(dst=out, data1=a, data2=b, op=nl.multiply)`. Note `nl.`, not `nisa.`, in front of `multiply`. |
| `nisa.scalar_mul(...)` | 12 | 1 | 不存在，`nl` 里也没有。意图是「乘以常数」 | 同上第一句：there is no scalar_mul. To multiply a tile by a constant use `nisa.tensor_scalar(dst=out, data=t, op0=nl.multiply, operand0=c)`. |
| `nisa.nc_matmul(..., transpose_moving=...)` | 13 | 1, 2 | 没有这个参数。真实签名：`nc_matmul(dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, accumulate=None, tile_position=(), tile_size=(), perf_mode=..., name=None)` | Remove `transpose_moving=`; nc_matmul has no such argument. （L1 额外加一句：average pooling needs no matmul at all — reduce each pool window with `nl.sum(view, axis=[...])` over a strided `tile.ap([...])` view.）（L2 额外加一句：a transpose does not need nc_matmul; `nisa.nc_transpose(dst=, data=)` exists in 0.6.0 — 用法未验证，先在 pod 里试过再写进反馈。） |
| `nl.tile_size(...)` 当函数调用 | 1 | 4 | `nl.tile_size` 是一个**常量集合**，不能调用（属性如 `nl.tile_size.pmax`） | `nl.tile_size` is a set of constants, not a function: read `nl.tile_size.pmax` (=128) instead of calling it. |

**现在 harness 给的反馈（对比）**
- `nisa.multiply` → *"`nki.isa` has no `multiply`, and nothing similar exists. Its real names include: NkiInstruction, …"*（列了 25 个按字母排序的名字）
- `nisa.scalar_mul` → *"The closest real names are: scalar_engine, tensor_scalar_cumulative, …"*
- `transpose_moving` → *"Remove the `transpose_moving=` argument. The real signature is …"*（这一条已经不错，只缺 L1 的「根本不需要 matmul」）

## 2. 可能的根因

prompt 里的 `API_CARD`（`agent.py:171`）写的是 `nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=0.5)`。模型看到 `multiply`，很可能把它当成了 `nisa` 下的函数。可以考虑在卡片里明确写一句：*"ops like nl.multiply / nl.add are passed as op= arguments; they are not nisa functions"*。改了以后要用 `--level 1 --repeat 5` 测，因为改 prompt 有可能让别的 level 变差（STATE.md 里记录过一次这样的回滚）。

## 3. 0.6.0 核对过的签名（level 1–4 用得到的）

```
nl.ndarray(shape, dtype, buffer=nl.sbuf, name='', address=None)
nl.affine_range(start, stop=None, step=1)
nl.sum(x, axis, dtype=None, keepdims=False)
nl.multiply(x, y=None, dtype=None)          # 也可作为 op= 传入
nisa.dma_copy(dst, src, ...)
nisa.tensor_copy(dst, src, ...)
nisa.tensor_scalar(dst, data, op0, operand0, reverse0=False, op1=None, operand1=None, ...)
nisa.tensor_tensor(dst, data1, data2, op, ...)
nisa.tensor_reduce(dst, op, data, axis, negate=False, keepdims=False)
nisa.nc_matmul(dst, stationary, moving, ..., is_transpose=False, accumulate=None, ...)
nisa.nc_transpose(dst, data, ...)
```

## 4. 接进 harness 的方式（草案）

在 `enrich()` 里，`module ... has no attribute` 那个分支之前，先查一个手写的映射；查不到再走 `available_names()` 的兜底：

```python
KNOWN_FIXES = {
    "nki.isa.multiply": "...",     # 第 1 节表格里的建议文案
    "nki.isa.scalar_mul": "...",
}
```

验证方法：`python agent.py --level 1 --rounds 8 --samples 4 --context 8192 --repeat 5 --log attempts_fix_api.jsonl`，和 baseline 的 level 1（每次都是 0.30）比较。level 1 历来零方差，所以只要分数有变化，就能归因到这个改动。

## 5. 注意

- 不要把 `references/downloads/average_pool2d_nki_kernels.py` 等教程 kernel 放进 prompt，那基本就是标准答案。上面的建议只给出 API 用法，不给完整解法。
- L1 修好编造函数以后，下一面墙大概率是「拷贝两边大小不一致」（L1 也有 24 次），要做好卡点会转移的准备。
