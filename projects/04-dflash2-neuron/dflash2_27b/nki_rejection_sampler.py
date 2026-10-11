#!/usr/bin/env python3
"""nki_rejection_bm.py — NKI greedy rejection-sampler kernel validated via
nki.baremetal (runs on the NeuronCore with numpy I/O; no torch_neuronx needed).

Kernel computes the per-position accepted target token for greedy speculative
decoding: accept valid positions until (and including) the first draft/target
mismatch; reject the rest. Output -1 for rejected/padding. Bonus-token handling
is done in the thin numpy wrapper (same split as the stock torch version).
"""
import argparse
import time
import numpy as np

PLACEHOLDER = -1

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

# Execution mode set in main(): "device" uses the @nki.jit baremetal path,
# "sim" routes through nki.simulate_kernel (CPU simulator, numpy I/O).
_MODE = "device"


def _run_kernel(*args):
    if _MODE == "sim":
        return nki.simulate_kernel(reject_scan_nki, *args)
    return reject_scan_nki(*args)


@nki.jit
def reject_scan_nki(drafts_2d, targets_2d, valid_2d, utri):
    """Fused greedy accept scan.

    Inputs (all [B, L] int32; utri is [L, L] float32 strictly-upper-triangular,
    utri[i, j] = 1.0 if i < j else 0.0):
      drafts_2d  : draft token ids (padding = -1 where not valid)
      targets_2d : target-sampled token ids aligned to same positions
      valid_2d   : 1 where position is a real draft, else 0
    Output [B, L] int32: accepted target token per position, -1 for reject/pad.

    accept[b,j] = valid[b,j] AND (no earlier valid mismatch in row b).
    The exclusive prefix-count of mismatches is computed as a matmul with a
    strictly-upper-triangular matrix (scan-as-matmul on the tensor engine):
        prior[b,j] = sum_{i<j} mism[b,i] = (mism @ utri)[b,j]
    """
    B, L = drafts_2d.shape
    result = nl.ndarray((B, L), dtype=nl.int32, buffer=nl.shared_hbm)

    d = nl.load(drafts_2d)
    t = nl.load(targets_2d)
    v = nl.load(valid_2d)
    U = nl.load(utri)                                 # [L, L] f32

    is_valid = nl.greater(v, 0)                       # v == 1
    match = nl.equal(d, t)                            # draft == target
    mism = nl.logical_and(nl.logical_not(match), is_valid)
    mism_f = mism.astype(nl.float32)                  # [B, L]

    # prior[b,j] = sum_{i<j} mism[b,i]  == mism_f @ utri   -> [B, L]
    prior_psum = nl.matmul(mism_f, U, transpose_x=False)
    prior = nl.copy(prior_psum, dtype=nl.float32)

    no_prior = nl.less(prior, 0.5)                    # prior == 0
    accept = nl.logical_and(is_valid, no_prior)
    ph = nl.full((B, L), PLACEHOLDER, dtype=nl.int32)
    res = nl.where(accept, t, ph)

    res_sb = nl.copy(res, dtype=result.dtype)
    nl.store(result, value=res_sb)
    return result


def _build_tiles(draft_ids, cu, max_spec_len, target_token_ids):
    """numpy version of the reference's 2D tile construction."""
    batch = cu.shape[0]
    batch_idx = np.arange(batch)
    bonus = target_token_ids[cu + batch_idx]
    pos = np.arange(max_spec_len)[None, :]
    num_drafts = np.diff(cu, prepend=0)
    valid = pos < num_drafts[:, None]
    cu_start = np.concatenate([[0], cu[:-1]])
    draft_idx = np.clip(cu_start[:, None] + pos, None, draft_ids.shape[0] - 1)
    drafts_2d = np.where(valid, draft_ids[draft_idx], PLACEHOLDER)
    target_idx = np.clip(cu_start[:, None] + batch_idx[:, None] + pos, None, target_token_ids.shape[0] - 1)
    targets_2d = np.where(valid, target_token_ids[target_idx], PLACEHOLDER)
    return drafts_2d, targets_2d, valid, num_drafts, bonus, batch_idx


def nki_rejection_sampler(draft_ids, cu, max_spec_len, target_token_ids):
    drafts_2d, targets_2d, valid, num_drafts, bonus, batch_idx = _build_tiles(
        draft_ids, cu, max_spec_len, target_token_ids)
    L = max_spec_len
    # strictly-upper-triangular: utri[i,j] = 1 if i < j else 0
    utri = np.triu(np.ones((L, L), np.float32), k=1)
    out = _run_kernel(drafts_2d.astype(np.int32),
                      targets_2d.astype(np.int32),
                      valid.astype(np.int32),
                      utri)
    out = np.asarray(out)
    batch = cu.shape[0]
    result = np.concatenate([out, np.full((batch, 1), PLACEHOLDER, np.int32)], axis=1)
    all_accepted = ((drafts_2d == targets_2d) & valid).all(axis=1)
    result[batch_idx, num_drafts] = np.where(all_accepted, bonus, PLACEHOLDER).astype(np.int32)
    return result


def reference_numpy(draft_ids, cu, max_spec_len, target_token_ids):
    drafts_2d, targets_2d, valid, num_drafts, bonus, batch_idx = _build_tiles(
        draft_ids, cu, max_spec_len, target_token_ids)
    matches = (drafts_2d == targets_2d) & valid
    mism = ~matches
    sentinel = np.concatenate([mism, np.ones((cu.shape[0], 1), bool)], axis=1)
    first_mismatch = sentinel.astype(np.float32).argmax(axis=1)
    pos = np.arange(max_spec_len)[None, :]
    accept = (pos <= first_mismatch[:, None]) & valid
    output = np.where(accept, targets_2d, PLACEHOLDER)
    all_accepted = (matches & valid).all(axis=1)
    batch = cu.shape[0]
    result = np.concatenate([output, np.full((batch, 1), PLACEHOLDER, np.int32)], axis=1).astype(np.int32)
    result[batch_idx, num_drafts] = np.where(all_accepted, bonus, PLACEHOLDER).astype(np.int32)
    return result


def _make_case(batch, k, accept_frac, seed):
    rng = np.random.default_rng(seed)
    num_drafts = rng.integers(1, k + 1, size=batch)
    cu = np.cumsum(num_drafts).astype(np.int64)
    total = int(cu[-1])
    draft_ids = rng.integers(0, 1000, size=total).astype(np.int32)
    targets = []
    gi = 0
    for b in range(batch):
        nd = int(num_drafts[b])
        for _ in range(nd):
            if rng.random() < accept_frac:
                targets.append(int(draft_ids[gi]))
            else:
                targets.append(99999)
            gi += 1
        targets.append(42)
    return draft_ids, cu, k, np.array(targets, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--mode", choices=["device", "sim"], default="device")
    args = ap.parse_args()
    global _MODE
    _MODE = args.mode
    print(f"mode={_MODE}")

    # NKI baremetal needs fixed shapes per compile; pad all cases to L=k, fixed batch.
    ok = True
    checked = 0
    for seed in range(15):
        for batch in (1, 2, 4):
            for k in (2, 4, 8):
                for af in (0.0, 0.5, 1.0):
                    d, cu, L, tgt = _make_case(batch, k, af, seed)
                    ref = reference_numpy(d, cu, L, tgt)
                    got = nki_rejection_sampler(d, cu, L, tgt)
                    checked += 1
                    if not np.array_equal(ref, got):
                        ok = False
                        print(f"MISMATCH seed={seed} batch={batch} k={k} af={af}")
                        print(" ref:", ref.tolist())
                        print(" got:", got.tolist())
                        break
                if not ok: break
            if not ok: break
        if not ok: break
    print(f"CORRECTNESS: {'PASS' if ok else 'FAIL'} ({checked} cases)")

    # Timing: NKI kernel (baremetal, device) vs numpy reference, fixed shape.
    d, cu, L, tgt = _make_case(4, 8, 0.6, 999)
    drafts_2d, targets_2d, valid, *_ = _build_tiles(d, cu, L, tgt)
    a = drafts_2d.astype(np.int32); b = targets_2d.astype(np.int32); c = valid.astype(np.int32)
    utri = np.triu(np.ones((L, L), np.float32), k=1)
    # warm (triggers compile)
    _run_kernel(a, b, c, utri)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        _run_kernel(a, b, c, utri)
    t1 = time.perf_counter()
    for _ in range(args.iters):
        reference_numpy(d, cu, L, tgt)
    t2 = time.perf_counter()
    nki_us = (t1 - t0)/args.iters*1e6
    ref_us = (t2 - t1)/args.iters*1e6
    print(f"PERF iters={args.iters}: nki_device={nki_us:.1f}us numpy_ref={ref_us:.1f}us")
    print("(device call includes host->device dispatch overhead; kernel-only time is smaller)")


if __name__ == "__main__":
    main()
