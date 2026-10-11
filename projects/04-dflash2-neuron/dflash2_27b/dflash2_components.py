#!/usr/bin/env python3
"""dflash2_components.py — raw-PyTorch DFlash2 draft components (Neuron-ready),
faithful to the z-lab/dflash reference (dflash/model.py).

Implements the two DFlash2-specific mechanisms missing from the Neuron backend:
  * GroupedDynamicCausalConv  — the two-tap local dynamic convolution that wraps
    attention and MLP in each draft layer (task 9)
  * CandidateSelector         — top-k unary candidates + predecessor/successor
    low-rank transition scoring + greedy self-conditioned path walk (task 10)

Verified by loading the real z-lab/Qwen3.8-27B-DFlash2 weights and running forward
passes (shapes + numeric). No NKI; pure torch so it can be traced on Neuron.
"""
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_GLOB = "/root/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/*/"


# ---- Task 9: two-tap grouped dynamic causal convolution ----------------------
def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    """Faithful port of the reference _grouped_dynamic_convolve.
    hidden  [B, L, H]; base [kernel_size, H]; dynamic [B, L, kernel_size, groups].
    """
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.view(batch, length, groups, group_size)
    dynamic = dynamic.view(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        # causal shift by `offset` along length
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].view(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.view_as(hidden)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size, kernel_size, group_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * groups, bias=False)

    def prepare(self, hidden):
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        return (
            _grouped_dynamic_convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden, dynamic):
        return _grouped_dynamic_convolve(hidden, dynamic, self.base_kernel[1], self.group_size)


# ---- Task 10: candidate selector (top-k + predecessor/successor path walk) ----
class CandidateSelector(nn.Module):
    def __init__(self, vocab_size, hidden_size, rank, top_k):
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Embedding(vocab_size, rank)
        self.successor_codebook = nn.Embedding(vocab_size, rank)
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)

    @torch.no_grad()
    def select(self, hidden, logits, anchor_ids, temperature=0.0):
        """hidden [B,L,H]; logits [B,L,V]; anchor_ids [B]. Returns path [B,L]."""
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        hidden = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path = []
        for position in range(hidden.shape[1]):
            # score = unary(cand) + <predecessor_codebook[prev] * hidden_proj(h), successor_codebook[cand]>
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk",
                self.predecessor_codebook(predecessor) * hidden[:, position],
                self.successor_codebook(candidates[:, position]),
            )
            index = torch.argmax(scores, dim=-1)  # greedy (temperature 0)
            predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
            path.append(predecessor)
        return torch.stack(path, dim=1), candidates


# ---- verification harness ----------------------------------------------------
def _load_weights():
    ck = glob.glob(CKPT_GLOB)[0]
    f = glob.glob(os.path.join(ck, "*.safetensors"))[0]
    from safetensors.torch import load_file
    return load_file(f)


def main():
    import json
    ck = glob.glob(CKPT_GLOB)[0]
    cfg = json.load(open(os.path.join(ck, "config.json")))
    dfc = cfg["dflash_config"]
    H = cfg["hidden_size"]; V = cfg["vocab_size"]
    rank = dfc["selector_rank"]; top_k = dfc["selector_top_k"]
    ks = dfc["conv_kernel_size"]; gs = dfc["conv_group_size"]
    print(f"config: H={H} V={V} rank={rank} top_k={top_k} conv_kernel={ks} group={gs}")

    sd = _load_weights()
    torch.manual_seed(0)

    # --- selector: load real codebooks + projection, run a forward ---
    sel = CandidateSelector(V, H, rank, top_k)
    sel.predecessor_codebook.weight.data = sd["candidate_selector.predecessor_codebook"].float()
    sel.successor_codebook.weight.data = sd["candidate_selector.successor_codebook"].float()
    sel.hidden_projection.weight.data = sd["candidate_selector.hidden_projection.weight"].float()
    B, L = 1, dfc["block_size"]
    hidden = torch.randn(B, L, H)
    logits = torch.randn(B, L, V)
    anchor = torch.randint(0, V, (B,))
    path, cand = sel.select(hidden, logits, anchor, temperature=0.0)
    assert path.shape == (B, L), path.shape
    assert cand.shape == (B, L, top_k), cand.shape
    # every chosen token must be one of its position's top-k candidates
    in_cand = (path.unsqueeze(-1) == cand).any(-1).all().item()
    print(f"SELECTOR ok: path={path.shape} candidates={cand.shape} path_in_topk={in_cand}")
    # determinism check (greedy must be repeatable)
    path2, _ = sel.select(hidden, logits, anchor, temperature=0.0)
    print("SELECTOR deterministic:", torch.equal(path, path2))

    # --- conv: load real layer-0 kernels, run prepare/finish ---
    conv = GroupedDynamicCausalConv(H, ks, gs)
    conv.base_kernel.data = sd["layers.0.attention_conv.base_kernel"].float()
    conv.kernel_projection.weight.data = sd["layers.0.attention_conv.kernel_projection.weight"].float()
    x = torch.randn(B, L, H)
    pre, dyn = conv.prepare(x)
    fin = conv.finish(pre, dyn)
    assert pre.shape == x.shape and fin.shape == x.shape
    # causality: output at position t must not depend on t+1 (2-tap causal)
    x2 = x.clone(); x2[:, -1] += 100.0  # perturb last position
    pre_b, dyn_b = conv.prepare(x2)
    changed = (pre_b - pre).abs().sum(-1)[0]  # per-position change
    causal_ok = changed[:-1].sum().item() == 0.0  # only last position should change
    print(f"CONV ok: prepare={pre.shape} finish={fin.shape} causal(no future leak)={causal_ok}")

    print("\nINPUT_IN_CANDIDATES:", in_cand, "| CONV_CAUSAL:", causal_ok,
          "| DETERMINISTIC:", torch.equal(path, path2))
    if in_cand and causal_ok and torch.equal(path, path2):
        print("DFLASH2_COMPONENTS: PASS")
    else:
        print("DFLASH2_COMPONENTS: FAIL")


if __name__ == "__main__":
    main()
