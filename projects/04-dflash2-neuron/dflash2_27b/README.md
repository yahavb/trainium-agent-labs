# DFlash2 for Qwen3.8-27B on Neuron — components + blocker

This directory holds the DFlash2-specific pieces for the paper's target pair
(`Qwen/Qwen3.8-27B` + `z-lab/Qwen3.8-27B-DFlash2`) and documents the one
remaining blocker to an end-to-end Neuron serve.

## What's here (implemented + verified against the real downloaded weights)

- **`dflash2_components.py`** — the two DFlash2-specific mechanisms the plain
  DFlash draft lacks, in raw PyTorch (traceable on Neuron, no NKI):
  - `GroupedDynamicCausalConv` — two-tap grouped **dynamic causal convolution**
    (`conv_kernel_size=2`, `conv_group_size=16`) wrapping attention and MLP per
    draft layer; `prepare`/`finish` split matches the reference.
  - `CandidateSelector` — top-16 unary candidates per position, then a rank-256
    **predecessor/successor transition score**
    `unary(c) + <predecessor_codebook[prev] * hidden_projection(h), successor_codebook[c]>`
    and a greedy self-conditioned path walk from the anchor token.
  - Faithful port of z-lab/dflash `dflash/model.py`
    (`GroupedDynamicCausalConv`, `CandidateSelector`, `DFlash2DraftModel.propose`).
  - Verification (loads the real `z-lab/Qwen3.8-27B-DFlash2` safetensors):
    `DFLASH2_COMPONENTS: PASS` — selector path ⊆ top-k, deterministic at
    temperature 0, conv is causal (no future leakage).

- **`nki_rejection_sampler.py`** — an NKI greedy rejection-sampler kernel
  (scan-as-matmul). Correct in the NKI CPU simulator (405/405 cases) but
  mismatches on the baremetal device; kept for reference. Profiling showed the
  rejection sampler is **low ROI** (a tiny `[batch,k]` op); the real kernel
  targets are draft attention / the lm_head argmax.

## The blocker to end-to-end serving on Neuron

`Qwen/Qwen3.8-27B` is **not** a plain Qwen3. Its text backbone is
`model_type: qwen3_5_text`: a **hybrid linear-attention (Mamba-2 / SSM) +
full-attention** model (48 of 64 layers are `linear_attention` with
`A_log` / `dt_bias` / `conv1d` / `in_proj_*` SSM weights; every 4th layer is
full attention), plus `attn_output_gate`, partial RoPE (0.25), mRoPE, an MTP
head, and a vision tower. The Neuron backend (`vllm_neuron`) only implements
plain `Qwen3ForCausalLM`, so it cannot build this target's graph.

Unblocking requires implementing the Qwen3.5 hybrid text model on Neuron — the
hard piece being the linear-attention/SSM selective scan (a new kernel). The
DFlash2 draft components above are ready to wire into a proposer once the target
loads. See the task list in the project tracker.
