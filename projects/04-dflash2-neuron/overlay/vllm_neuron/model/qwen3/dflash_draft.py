# SPDX-License-Identifier: Apache-2.0
"""
DFlash2 draft model for the Neuron backend (Qwen3 target).

Checkpoint (z-lab/Qwen3-8B-DFlash-b16): 5 Qwen3 decoder layers + fc + hidden_norm
+ norm. embed_tokens / lm_head are shared with the target (loaded by the proposer).

One propose step:
  1. Context:  ctx = hidden_norm(fc(concat of 5 target hidden states)).
               For every draft layer, K/V of ctx (k_norm + RoPE on K) are written
               straight into that layer's KV cache at the target tokens' slots.
  2. Query:    [next_token, MASK x num_spec] per request, at positions p+1 .. p+1+num_spec,
               run through the 5 layers using the normal Neuron decode attention, which
               reads the context K/V from the cache.
  3. Draft:    greedy tokens at the MASK positions -> num_spec draft tokens.

Simplification vs. the GPU reference: attention inside the query block is causal
(the Neuron decode kernel), not bidirectional. The target still verifies every draft
token, so output is unchanged; only the acceptance rate can be lower.
"""

import torch
from torch import nn

import vllm_neuron.functional as NF

from .config import Qwen3Config
from .model import Qwen3ForCausalLM, Qwen3RMSNorm, apply_rotary_pos_emb


def _extract_accepted_tokens(
    input_ids: torch.Tensor,
    sampling_positions: torch.Tensor,
    raw_sampled_token_ids: torch.Tensor,
    vocab_size: int,
    num_speculative_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """On-device: bonus token per request + sampling positions adjusted for rejections.

    raw_sampled_token_ids is the rejection sampler output, [bs, num_spec+1] with -1 for
    rejected tokens (or [bs, 1] after prefill). Returns
    (input_ids patched with the bonus token, adjusted sampling_positions, bonus [bs]).
    """
    valid_mask = (raw_sampled_token_ids != -1) & (raw_sampled_token_ids < vocab_size)
    valid_count = valid_mask.sum(dim=1)
    last_valid_idx = torch.clamp(valid_count - 1, min=0)
    next_token_ids = (
        raw_sampled_token_ids.gather(1, last_valid_idx.unsqueeze(1).to(torch.long))
        .squeeze(1)
        .to(torch.int32)
    )
    next_token_ids = torch.where(
        valid_count > 0, next_token_ids, torch.zeros_like(next_token_ids)
    )
    if raw_sampled_token_ids.shape[1] > 1:
        num_rejected = torch.clamp(num_speculative_tokens + 1 - valid_count, min=0)
        sampling_positions = torch.clamp(
            sampling_positions - num_rejected.to(sampling_positions.dtype), min=0
        )
    input_ids = input_ids.scatter(0, sampling_positions.to(torch.long), next_token_ids)
    return input_ids, sampling_positions, next_token_ids


class DFlashQwen3NeuronDraft(Qwen3ForCausalLM):
    """Neuron DFlash2 draft: Qwen3 backbone (5 layers) + fc + hidden_norm."""

    def __init__(self, config: Qwen3Config, start_layer_idx: int = 0, dflash_config=None):
        super().__init__(config)
        dfc = dflash_config or {}
        self.dflash_config = dfc
        self.mask_token_id = dfc.get("mask_token_id")
        n_target = len(dfc.get("target_layer_ids") or []) or 1
        dtype = self.model.norm.weight.dtype
        hidden = config.hidden_size

        self.fc = nn.Linear(hidden * n_target, hidden, bias=False, dtype=dtype)
        self.hidden_norm = Qwen3RMSNorm(hidden, config.rms_norm_eps, dtype)
        self._dflash_draft = True  # load_weights: bare checkpoint keys, no embed/lm_head
        self.num_speculative_tokens = 1  # set by the proposer before compile

        # Draft layers are named layers.{start_layer_idx + i} (36..40 for Qwen3-8B) so
        # their KV cache / attention metadata keys don't collide with the target's.
        for layer in self.model.layers:
            layer.layer_idx += start_layer_idx
            if hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx += start_layer_idx

    @classmethod
    def from_configs(cls, config, start_layer_idx: int = 0, neuron_config=None):
        neuron_cfg = Qwen3Config.from_configs(hf_config=config, neuron_config=neuron_config)
        return cls(
            neuron_cfg,
            start_layer_idx=start_layer_idx,
            dflash_config=getattr(config, "dflash_config", None) or {},
        )

    @staticmethod
    def _write_context_kv(attn, x, position_embeddings, slot_mapping, block_size):
        """Project ctx through this layer's K/V weights and write into its KV cache."""
        tokens = x.shape[0]
        qkv = NF.qkv_proj(
            hidden=x.unsqueeze(0), qkv_weights=attn.qkv_proj_weight, bias=None
        ).squeeze(0)
        _, k, v = torch.tensor_split(qkv, attn.qkv_split_indices, dim=-1)
        nkh, hd = attn.num_key_value_heads_per_rank, attn.head_dim
        k = k.view(tokens, nkh, hd).transpose(0, 1)
        v = v.view(tokens, nkh, hd).transpose(0, 1)
        k = attn.k_norm(k)
        cos, sin = position_embeddings
        k, _ = apply_rotary_pos_emb(k, k, cos, sin)

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        head_idx = torch.arange(nkh, dtype=torch.long, device=x.device).repeat_interleave(
            slot_mapping.shape[0]
        )
        blk_idx = block_indices.repeat(nkh)
        pos_idx = position_indices.repeat(nkh)
        attn.k_cache.index_put_(
            (blk_idx, head_idx, pos_idx), k.reshape(-1, hd).to(attn.k_cache.dtype)
        )
        attn.v_cache.index_put_(
            (blk_idx, head_idx, pos_idx), v.reshape(-1, hd).to(attn.v_cache.dtype)
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        initial_target_hidden_states: torch.Tensor,
        attn_metadata=None,
        sampling_positions: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        raw_sampled_token_ids: torch.Tensor | None = None,
        prev_sampled_token_ids: torch.Tensor | None = None,
        prev_num_draft_tokens: torch.Tensor | None = None,
        req_indices_per_token: torch.Tensor | None = None,
        **kwargs,
    ):
        # Async scheduling: correct positions/slots for last step's rejected drafts.
        if (
            prev_sampled_token_ids is not None
            and prev_num_draft_tokens is not None
            and req_indices_per_token is not None
        ):
            positions, attn_metadata = NF.correct_spec_decode_positions_and_slot_mapping(
                positions,
                attn_metadata,
                prev_sampled_token_ids,
                prev_num_draft_tokens,
                req_indices_per_token,
                self.config.vocab_size,
            )

        if raw_sampled_token_ids is not None:
            input_ids, sampling_positions, bonus = _extract_accepted_tokens(
                input_ids,
                sampling_positions,
                raw_sampled_token_ids,
                self.config.vocab_size,
                self.num_speculative_tokens,
            )
        else:
            bonus = None

        layers = self.model.layers
        meta0 = attn_metadata[f"layers.{layers[0].layer_idx}.self_attn"]
        block_table = meta0["block_table_tensor"]
        block_size = meta0["block_size"]
        max_blocks = meta0["max_blocks_per_seq"]

        # 1. Context K/V from the target hidden states.
        ctx = self.hidden_norm(
            self.fc(initial_target_hidden_states.to(self.fc.weight.dtype))
        )
        ctx_pe = self.model.rotary_emb(
            positions.to(torch.int32), device=ctx.device, dtype=ctx.dtype
        )
        for layer in layers:
            self._write_context_kv(
                layer.self_attn,
                ctx,
                ctx_pe,
                attn_metadata[f"layers.{layer.layer_idx}.self_attn"]["slot_mapping"],
                block_size,
            )

        # 2. Query block: [next_token, MASK x num_spec] at positions p+1 .. p+1+num_spec.
        ns = self.num_speculative_tokens
        q_len = ns + 1
        bs = sampling_positions.shape[0]
        sp = sampling_positions.to(torch.long)
        next_tok = input_ids[sp].to(torch.int32)
        masks = torch.full(
            (bs, ns), self.mask_token_id, dtype=torch.int32, device=input_ids.device
        )
        q_ids = torch.cat([next_tok.view(bs, 1), masks], dim=1).view(-1)

        base = positions[sp].to(torch.long) + 1
        offsets = torch.arange(q_len, dtype=torch.long, device=base.device)
        q_pos = (base.view(bs, 1) + offsets.view(1, q_len)).view(-1)
        block_no = torch.clamp(q_pos // block_size, max=max_blocks - 1)
        block_ids = (
            block_table.repeat_interleave(q_len, dim=0)
            .gather(1, block_no.view(-1, 1))
            .view(-1)
        )
        q_slots = block_ids.to(torch.long) * block_size + (q_pos % block_size)

        q_meta = {}
        for layer in layers:
            q_meta[f"layers.{layer.layer_idx}.self_attn"] = {
                "block_table_tensor": block_table,
                "slot_mapping": q_slots,
                "max_query_len": q_len,
                "block_size": block_size,
                "max_blocks_per_seq": max_blocks,
                "decode_token_threshold": q_len,  # always the decode attention path
            }

        q_pos32 = q_pos.to(torch.int32)
        hidden = self.model.embed_tokens(q_ids, scatter_tokens=False, rank=rank)
        q_pe = self.model.rotary_emb(q_pos32, device=hidden.device, dtype=hidden.dtype)
        for layer in layers:
            hidden = layer(
                hidden,
                positions=q_pos32,
                position_embeddings=q_pe,
                attn_metadata=q_meta,
            )
        hidden = self.model.norm(hidden)

        # 3. Greedy draft tokens at the MASK positions.
        mask_rows = (
            torch.arange(bs * q_len, device=hidden.device).view(bs, q_len)[:, 1:].reshape(-1)
        )
        logits = self.lm_head(torch.index_select(hidden, 0, mask_rows))
        if self.on_device_sampling_config is not None:
            tokens = self.sampler(logits).to(torch.int32)
        else:
            tokens = torch.argmax(logits, dim=-1).to(torch.int32)
        drafts = tokens.view(bs, ns)

        # Same output contract the Neuron runner expects from a draft model:
        # (bonus + drafts [bs, 1+ns] or drafts [bs, ns], drafts only [bs, ns], logits|None)
        if bonus is not None:
            stacked = torch.cat([bonus.view(bs, 1).to(torch.int32), drafts], dim=1)
        else:
            stacked = drafts
        return stacked, drafts, None
