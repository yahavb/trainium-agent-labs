# Worked example: one piece through all three tools

The Qwen3-8B LM head (4096 → 151,936 vocabulary projection + argmax, 1.24 GB of weights per token), taken
from the library kernel to our own, following the skill's loop:
profile (`neuron-explorer` capture via `megakernel/kernels/profile_mega.py`) → bounds (`analyze_profile.py`) → ask the rules corpus → one change → prove the piece → device gate → profile again.

**What this is, exactly.** A replay of the LM-head work from 2026-10-10 (`megakernel/ATTEMPTS.md` #9, #10,
#17, #18). The **heuristics queries and nkiequiv proofs are run live** by `replay_cpu.py` (output:
`results/cpu_replay.txt`, CPU only). The **device numbers are from the profiles and tests recorded during the
original work** (`megakernel/results/profiles/{head,head_hwdge,head_tiled}/bounds.json`, ATTEMPTS.md).
`run_device.sh` re-measured v0, v1 and v2 at 21:41–21:47 UTC the same evening and matched them within 1.5%
(below the table); v2a and v3 can't be re-run as they stood (v2a's bug is fixed in the code, v3 needs the full step). On the day, the changes came
from reading profiles and nkilib source; the corpus queries were added afterwards and show what an agent
following the skill is handed at each step. It starts from the library head, as the skill prescribes, not
from a hand-written naive kernel; the first hand-written version (v2a) is the one that was wrong.

## The steps

| | kernel | device (time per core, DMA rate) | what the profile / test says | the corpus returns (live) | nkiequiv (live, CPU) |
|---|---|---|---|---|---|
| v0 | nkilib `output_projection_tkg` | **2.71 ms**, 230 GB/s; ideal 1.43 ms | DMA-bound; `W_lm` streamed by software DGE: 4,129 GpSimd `DMA_DIRECT2D` issued one at a time | `dge-swdge-consumes-gpsimd`: software DGE costs GpSimd cycles; use hardware DGE where the pattern allows | not run (library block, 128-wide tiles) |
| v1 | v0 with weight DMAs on hardware DGE, two queues | **2.53 ms** (−7%), 246 GB/s | still DMA-bound; each transfer moves ~1.2 KB per partition | `dma-4kib-per-partition-saturates`: ≥ 4 KiB per partition to approach peak | not run (data movement only) |
| v2a | **our own head**: weights pre-tiled on the host so each DMA moves 32 KB contiguous per partition, weights stationary, 4-deep ring; 4 vocab tiles accumulate in one `[P, 4]` PSUM tile | — | `test_head.py --tiled`: **14.5% rel-L2**, tiles 0–2 of every 4-tile block lost one k-contribution | `cp-psum-accumulate-semantics`, `psum-bank-size-2kib-and-bank-cycling`: give each accumulation group its own PSUM tile or bank | **EQUIVALENT** (0.35 s): correct in NKI's semantics, so the proof can't see this bug |
| v2 | v2a with one PSUM tile per vocab tile | **2.23 ms** (−18% vs v0), 280 GB/s | logits rel-L2 2.38e-3, identical to nkilib's head | — | **EQUIVALENT**: 0.33 s at small tiles, 21.6 s at full Trainium2 tiles |
| control | v2 with one k-tile skipped | — | — | — | **NOT_EQUIVALENT**, counterexample at output [0, 0] naming the missing `x[*, 3]` terms (0.35 s) |
| v3 | v2 inside the full one-launch step | 2 layers, fixed position: 3.71 vs 4.28 ms | generate loop: **out-of-bounds indirect DMA** (scalar DGE, nrta 1006) on the second token | `for_error`: `dma-dge-mode-selection` (choose the DGE mode per DMA deliberately), `op-gather-scatter-is-the-weak-spot` | can't model indirect DMA |

Re-measured with `run_device.sh` on 2026-10-10, 21:41–21:47 UTC (`results/*.log`, `results/bounds.log`, profiles in
`megakernel/results/profiles/ex_head*`): **2.71 / 2.53 / 2.20 ms** per core for v0 / v1 / v2 (230 / 246 / 285 GB/s);
logits rel-L2 2.377e-3 for both v0 and v2. The profile shows the mechanism: v0 and v1 move the 622 MB per core in
~4,130 DMAs of ~150 KB each (~1.2 KB per partition), v2 in ~155 DMAs of ~4 MB (32 KB per partition).

v3 is unresolved, so the final megakernel keeps v0 and leaves ~0.48 ms per token here. The corpus's lead
(the embedding and RoPE-row gathers use software DGE while v2 put the head's weight DMAs on hardware DGE in
the same kernel) is untested.

## What each tool did

* **The skill** set the order: no change without a profile, one change per step, the cheap gate before the
  expensive one, and a profile after every change. v1 was kept as a measurement, not adopted, because −7%
  wasn't worth monkeypatching nkilib.
* **The heuristics corpus** turned each profile symptom into a named rule with a fix: software DGE → hardware
  DGE (v1), ~1.2 KB per partition → ≥ 4 KiB contiguous (v2). For the PSUM bug it returned the right
  neighbourhood (one PSUM tile per accumulation group) but not the exact behaviour, which is now in
  `skill/nki-megakernel-optimizer/references/pitfalls.md`.
* **nkiequiv** checks a rewrite in a third of a second and names a dropped term (the control). It proved v2a,
  which was wrong on the device: the shared-PSUM fault is outside its model.
* **The device gate** caught v2a and v3, which neither of the other two could see.

## Reproduce

```bash
git clone https://github.com/forthoney/symnki && git -C symnki checkout b9b6b2a
NKIEQUIV=$PWD/symnki python examples/lm-head-walkthrough/replay_cpu.py         # ~25 s, CPU only
NEURON_RT_VISIBLE_CORES=2 bash examples/lm-head-walkthrough/run_device.sh       # device half: head tests + 3 profiles
```
