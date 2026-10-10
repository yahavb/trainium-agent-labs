# StreamDiffusionV2 causal Wan2.1-1.3B: shapes and non-portable ops

Read from the local clone `repos/StreamDiffusionV2` at commit `6961a5c` (not executed). Paths below are relative to
that clone. `cm` = `models/wan/causal_model.py`, `base` = `models/wan/wan_base/modules/model.py`,
`attn` = `models/wan/wan_base/modules/attention.py`, `pipe` = `models/wan/causal_stream_inference.py`,
`inf` = `streamv2v/inference.py`.

## Classes

| What | Class | File |
|---|---|---|
| Causal model | `CausalWanModel` | `cm:579` |
| Block | `CausalWanAttentionBlock` | `cm:448` |
| Self-attention with KV cache | `CausalWanSelfAttention` | `cm:175` |
| Cross-attention | `WanT2VCrossAttention` | `base:159` |
| Norms | `WanRMSNorm` `base:70`, `WanLayerNorm` `base:89` | |
| Wrapper that loads the checkpoint | `CausalWanDiffusionWrapper` | `models/wan/wan_wrapper.py:343` |
| Streaming pipeline | `CausalStreamInferencePipeline` | `pipe:15` |
| Demo driver | `SingleGPUInferencePipeline` (`inf:59`), used by `demo/vid2vid.py:32` | |

Checkpoint: `jerryfeng/StreamDiffusionV2 : wan_causal_dmd_v2v/model.pt` (5.68 GB), a dict `{"generator": state_dict}`
with keys `model.blocks.<i>.*` in fp32. Block 0 has 27 tensors.

## The v2v config the demo uses

`demo/config.py:73-85` defaults: `configs/wan_causal_dmd_v2v.yaml`, `--step 2`, `--noise_scale 0.8`,
`--model_type T2V-1.3B`.

| Setting | Value | Where |
|---|---|---|
| Resolution (demo) | 512 x 512 | `demo/vid2vid.py:304-308` (fixed, hidden fields) |
| Resolution (offline `run_v2v.sh`) | 480 x 832 | `run_v2v.sh:40-41` |
| `num_frame_per_block` | 1 latent frame per chunk | yaml line 53 |
| Pixel frames per chunk | 4 (`base_chunk_size` 4 x `num_frame_per_block`) | `inf:99`, `inf:192` |
| `denoising_step_list` in yaml | [700, 500, 400, 200, 0] | yaml lines 14-19 |
| Steps actually run | **[700, 500]**: `--step 2` keeps the first 2 non-zero steps, v2v drops the trailing 0 | `streamv2v/inference_common.py:86-93`, `pipe:77-79` |
| First step is adaptive | `current_step = int(1000 * noise_scale) - 100` overwrites step 0 every call | `inf:51-57`, `pipe:278-279` |
| KV cache window | `num_kv_cache: 6` latent frames | yaml line 54, `pipe:52-53` |
| Sink | `num_sink_tokens: 3` (it counts frames: first 3 frames are never evicted) | yaml line 55, `pipe:96`, `cm:318` |
| Adaptive sink refresh | `adapt_sink_threshold: 0.2` | yaml line 56, `cm:332-347` |
| RoPE position rewind | `t_refresh = 50` frames, rewinds to frame 5 | `inf:100`, `inf:234-236` |
| Batched denoising | batch = number of steps = 2; row 0 is the newest frame at step 0, row 1 the previous frame at step 1; one KV cache per row | `pipe:206`, `pipe:224-229`, `pipe:269-289` |
| Working dtype | bf16 | `inf:92`, `inf:129` |

### Ring buffer implementation (`cm:304-398`)

The cache is a preallocated `[B, 6144, 12, 128]` tensor per block, per K and V (`pipe:100-101`), plus
`global_end_index`, `local_end_index`, per-slot `pos`, and Python ints `current_step` / `total_steps`.

- **Fill phase** (`cm:378-393`): chunk `i` is written in place at slot `i`. While filling, slot ends past the sink
  region are appended to the Python list `self.evict_idx` (`cm:381-384`), giving `[4096, 5120, 6144]` for the demo.
- **Steady state** (`cm:350-376`): the chunk overwrites `[target_end - L, target_end)` with
  `target_end = evict_idx[0]`, then the list is rotated (`cm:361-363`). So slots 3, 4, 5 are a ring and slots
  0, 1, 2 (the sink) are permanent.
- In no-batch mode (`cache_bs == 1`) the same slot is overwritten for every denoising step of a chunk and the ring
  advances only on the last one (`cm:358-364`). In batch mode each row has its own cache and advances every call.
- The chunk's K/V are written **before** attention, and attention runs over `cache[:, :local_end_index]`
  (`cm:400-414`), so the chunk attends to itself plus the cached frames. Keys are stored **after** RoPE (`cm:369`).
- **Adaptive sink** (`cm:332-347`): if the new K/V means have cosine < 0.2 against a sink frame, that sink slot is
  queued for eviction first.
- **Position rewind** (`cm:323-330`, `cm:204-223`): after `t_refresh` the caller rewinds `current_start`; cached
  non-sink keys are then re-rotated along the temporal RoPE band (`_shift_temporal_rope`, `cm:114`).

## Computed numbers

| Quantity | Demo 512 x 512 | Offline 480 x 832 |
|---|---|---|
| Latent per latent frame (VAE 8x spatial, 16 ch) | 16 x 64 x 64 | 16 x 60 x 104 |
| Temporal | 1 latent frame = 4 pixel frames (VAE 4x) | same |
| Patch size | (1, 2, 2) (`cm:593`) | same |
| Token grid per latent frame | 32 x 32 | 30 x 52 |
| Tokens per latent frame | **1024** | 1560 |
| Tokens per chunk (`L_chunk`, 1 frame per block) | **1024** | 1560 |
| KV cache length (`L_cache`, 6 frames) | **6144** | 9360 |
| of which sink (3 frames) | 3072 | 4680 |
| dim / heads / head_dim | 1536 / 12 / 128 | same |
| RoPE bands inside head_dim (t, h, w) | 44 / 42 / 42 dims = 22 / 21 / 21 complex pairs (`cm:689-694`) | same |
| FFN dim | 8960 | same |
| Text context length / raw text dim | 512 / 4096 (zero padded, `cm:838-843`) | same |
| Layers | 30 | same |
| `e` (time modulation) into a block | `[B, F, 6, 1536]`, F = 1 (`cm:832-833`) | same |
| Self-attention score matrix per call | 12 x 1024 x 6144 | 12 x 1560 x 9360 |
| One K or V cache, bf16 | 18.9 MB | 28.8 MB |
| All KV caches (30 layers, K+V, 2 denoising rows) | 2.26 GB | 3.45 GB |

Block-level tensor shapes at the demo config: `x [B, 1024, 1536]`, `e [B, 1, 6, 1536]`, `context [B, 512, 1536]`
(after `text_embedding`), `kv_cache["k"/"v"] [B, 6144, 12, 128]`, with B = 2 in the demo's batched mode.

One-off shape: the session's first call (`pipe.prepare`, `inf:188-202`) processes 5 pixel frames = **2 latent
frames (2048 tokens)** in one chunk, sequentially over the 2 steps with batch 1. Every later call is 1 frame.

## Non-portable ops inside one block

Everything a fixed-shape, tensors-only Neuron graph cannot express, on the KV-cache inference path of
`CausalWanAttentionBlock.forward`.

### flash_attn

| Where | What |
|---|---|
| `cm:24-33` | optional `flash_attn_interface` import |
| `cm:406-414` | self-attention: `flash_attn_interface.flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens)` inside `torch.cuda.device(...)`, wrapped in `try/except RuntimeError` (`cm:415-428`) |
| `base:189` -> `attn:138` | cross-attention: `flash_attention(q, k, v, k_lens=context_lens)` |
| `attn:187` | `assert q.device.type == 'cuda'` |
| `attn:229`, `attn:244` | `flash_attn_varlen_func` (FA3 / FA2) with `cu_seqlens` built by `cumsum` (`attn:233-236`) |
| `attn:197-213` | varlen packing: `torch.cat([u[:v] for u, v in zip(q, q_lens)])`, data-dependent lengths |
| `cm:12`, `cm:38-39`, `cm:280-285` | `torch.compile(flex_attention)` with a `BlockMask`; training path only (`kv_cache is None`), not used when streaming |

The repo already ships SDPA fallbacks used when flash_attn is missing: `attention_with_kvcache_fallback`
(`cm:126-172`) and `_sdpa_attention_fallback` (`attn:80-135`). They are still not traceable as written: `cm:138`
branches on `torch.all(cache_seqlens == max_seq_len)`, `cm:155` loops over `cache_seqlens.tolist()`, and
`attn:41-65` builds a boolean mask from runtime lengths.

### Complex-number RoPE

| Where | What |
|---|---|
| `base:35` | `torch.polar(...)`: `freqs` is a complex128 table `[1024, 64]` |
| `cm:103-106` | `torch.view_as_complex(x.to(torch.float64).reshape(...))`, complex multiply, `torch.view_as_real` (fp64) |
| `cm:57-73` | frequency rows sliced by runtime `start_frame` (`cm:65`), memoised in a Python `OrderedDict` |
| `cm:76-91` | `grid_sizes.tolist()` (`cm:86`), `start_frame.tolist()` (`cm:81`), Python loop over the batch |
| `cm:114-123` | `_shift_temporal_rope`: `.conj()`, `view_as_complex` / `view_as_real`, row picked by `abs(delta)` |
| `base:52`, `base:62` | same complex ops in `rope_apply` (non-cache path) |

### In-place cache writes

| Where | What |
|---|---|
| `cm:106` | `output[i, :seq_len] = ...` into a clone |
| `cm:369-370` | `kv_cache["k"/"v"][i:i+1, target_end-num_new_tokens:target_end] = ...` (steady state) |
| `cm:388-389` | `kv_cache["k"/"v"][i:i+1, local_start_index:local_end_index] = ...` (fill phase) |
| `cm:373-374`, `cm:392-393` | `slot_pos[i, slot0:slot0+n] = torch.arange(...)` |
| `cm:351-352`, `cm:397-398` | `.fill_()` on `global_end_index` / `local_end_index` |
| `cm:222-223` | re-rotated keys written back per slot |
| `base:176-180` | `crossattn_cache["k"/"v"/"is_init"]` mutated on first use |

### Data-dependent indexing

| Where | What |
|---|---|
| `cm:369-370`, `cm:388-389` | write offsets come from `evict_idx` / `local_end_index` |
| `cm:402-404` | `kv_cache["k"][:, :max_seq_len]`: the attended length changes while the cache fills |
| `cm:65` | `temporal[start_frame:start_frame + f]` |
| `cm:118` | `temporal[abs(delta)]` |
| `cm:334`, `cm:338` | sink slices sized by runtime `sink_tokens` |
| `cm:157-158` | per-sample `k_cache[b, :seq_len]` in the fallback |

### Python control flow and state driven by tensor values

| Where | What |
|---|---|
| `cm:287` | `math.prod(grid_sizes[0][1:]).item()` |
| `cm:309-313` | `self.evict_idx`: Python list state stored on the module |
| `cm:315` | `for i, c_start in enumerate(current_start)` over a tensor |
| `cm:328-330` | `if filled.any() and slot_pos[i][filled].max().item() - ... > ...` |
| `cm:332`, `cm:344-347` | adaptive sink: `if avg_cos_sim.min() < thr`, `torch.argmin(...).item()`, `list.insert` |
| `cm:350` | `if current_end > kv_cache_size or kv_cache["local_end_index"][i] >= kv_cache_size` |
| `cm:358-364` | `kv_cache['current_step']` Python counter decides whether the ring advances |
| `cm:361-363` | `evict_idx.pop(0)` / `.append` |
| `cm:376`, `cm:379`, `cm:381` | `.item()` on index tensors |
| `cm:382-384` | `if rolling_end > ... and ...` |
| `cm:400-402` | `torch.tensor(seq_lens)`, `seq_lens.max().item()` |
| `cm:138`, `cm:155` | fallback attention branches / loops on `cache_seqlens` |
| `cm:213-223` | `_realign_kv_positions`: `.tolist()`, list comprehensions, per-slot loop |
| `base:174-183` | `if crossattn_cache is not None: if not crossattn_cache["is_init"]` |
| `attn:196`, `attn:205` | `if q_lens is None` / `if k_lens is None` |

### Fine as is (static once shapes are fixed)

`cm:512` (`num_frames, frame_seqlen` from shapes), `cm:515` (`chunk(6)`), `cm:520-539` (`unflatten` / `flatten`
with shape-derived sizes), `base:83` (RMSNorm computed in fp32 then cast back), `WanLayerNorm`, the FFN with
`GELU(approximate='tanh')`.

## What `block_port.py` does about each

| Original | Port |
|---|---|
| `flash_attn_with_kvcache`, `flash_attention` | `F.scaled_dot_product_attention` (a matmul + softmax variant is kept behind `manual_attn=True`) |
| complex RoPE in fp64 | host precomputes `rope_cos` / `rope_sin` `[1, L, 1, 64]`; graph does `(a cos - b sin, a sin + b cos)` |
| in-place ring buffer with `evict_idx` | `new = cat(cache[:, :S], cache[:, S+L:], new_kv)`, S = 3072 sink tokens; returned as outputs |
| variable attended length while filling | fixed length plus host-supplied additive `attn_bias [1, 1, 1, 6144]` |
| sink slots filled by the first 3 chunks | host copies the tail slice into the sink slot for the first 3 chunks (`CacheState.commit`) |
| adaptive sink refresh | not in graph; host decision (disabled in the parity reference, as in `wan_causal_dmd_v2v_fast.yaml`) |
| `t_refresh` key re-rotation | not in graph; host-side operation on the cache every ~45 chunks |
| `crossattn_cache` | recomputed from `context` each call (could be hoisted to the host later) |
| batch of 2 denoising rows | batch 1 per call; one cache per denoising step, two calls per chunk |

With S = 0 the update is exactly `cat(cache[:, L_chunk:], new_kv)`. The repo does not do that: it keeps the first
3 frames forever, so the default here is S = 3 frames, which makes the port match the repo chunk for chunk
(`check_cpu.py`).
