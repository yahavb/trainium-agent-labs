# nkiequiv (symnki) on our megakernel: proof of concept

[nkiequiv](https://github.com/forthoney/symnki), from our teammate, proves an NKI kernel equal to a
reference over the reals. It runs the kernel on a symbolic NKI interpreter, compares canonical forms,
falls back to Z3, then tests numerics in emulated hardware precision. It needs no Trainium device. We
pointed it at the matmul core of our own LM head (`kernels/qwen3_decode_step.py::lm_head_tiled_argmax`),
copied into `lm_head_poc.py`, against a 6-line NumPy spec in the kernel's SBUF output layout.

Verified with nkiequiv at commit `b9b6b2a` (forthoney/symnki main, 2026-10-10), which adds a fast path in
`algebra.dot` and fixes 0-d reads and writes. The verdicts are the same as at `d45e4c0`, where this proof of
concept was first run; the full-tile proof is ~9% faster.

```bash
git clone https://github.com/forthoney/symnki && git -C symnki checkout b9b6b2a
export PYTHONPATH=$PWD/symnki            # deps: numpy, ml_dtypes, z3-solver (already on the seat image)
python -m nkiequiv check numpy:lm_head_poc.py::ref_lm_head lm_head_poc.py::lm_head_core --spec lm_head_poc.py
```

| kernel | nkiequiv | on the Trainium2 device (`kernels/test_head.py`, `--tiled` for our head) |
|---|---|---|
| `lm_head_core`: the structure in the final code | **EQUIVALENT, accept**: 32/32 outputs proved at the small profile in 0.03 s of solver time; numerics pass at full Trainium2 tiles (rel err 1.2e-7) | correct (logits rel-L2 2.4e-3, bf16) |
| `lm_head_core_skip_k`: deliberate bug, one k-tile never accumulated | **NOT_EQUIVALENT**: counterexample at output [0, 0], with both symbolic expressions showing the missing `x[*, 3]` terms | — |
| `lm_head_core_shared_psum`: our real attempt-17 bug, 4 accumulation groups in one `[P, 4]` PSUM tile | **EQUIVALENT** | **wrong: 14.5% rel-L2**, tiles 0–2 of every block lost one k-contribution |

Full output: `results.txt`.

## Numbers: what the check costs, and what it verified

**Cost of a correctness verdict** (wall clock including Python start-up, CPU only, no device):

| check | time | covers |
|---|---|---|
| nkiequiv, profile `small` (pmax 4) | **0.30 s** | 32 outputs proved over the reals |
| nkiequiv, profile `medium` (pmax 8) | **0.37 s** | 64 outputs proved |
| nkiequiv, profile `full` (Trainium2 tiles, pmax 128) | **22 s** (24–25 s at `d45e4c0`) | 1,024 outputs proved |
| our device head gate (`check.py --gates head`: compile + run on Trainium2) | 41–48 s | one random input, 151,936 logits |
| our device real-weights gate (36 layers, 32 teacher-forced steps) | 142 s | real model, one prompt |
| one 36-layer compile | ~11 min | — |

The wrong variant (`skip_k`) is rejected in the same time as the correct one is accepted, at every profile.
At the small profile a proof costs about 1/150 of a device check of the head, so an agent can try many
rewrites per compile and send only the survivors to the device.

**Performance of the kernel it verified** (device profiles, `results/profiles/*/bounds.json`; per physical core):

| LM head (1.24 GB of weights, Qwen3-8B vocabulary) | time | DMA rate |
|---|---|---|
| nkilib `output_projection_tkg` (software DGE on GpSimd) | 2.71 ms | ~230 GB/s |
| same, hardware DGE on two queues | 2.53 ms | ~246 GB/s |
| **ours, `lm_head_tiled_argmax`** (the structure proved here) | **2.23 ms (−18%)** | ~280 GB/s, ~78% of the shared HBM-stack ceiling |

In the full one-launch step at 2 layers (torch path, fixed position) it gives 4.28 → 3.71 ms. It isn't in
the final kernel yet: inside the generate loop it hits an out-of-bounds indirect DMA (ATTEMPTS #18) in a part
of the step nkiequiv can't model (indirect addressing).

## What this says

* **It fits the loop.** A proof in milliseconds, at shapes derived from `nl.tile_size`, with a counterexample
  that names the missing term. A cheap first gate before a 1–10 minute compile and a device run.
* **The two checkers catch different bugs.** The shared-PSUM kernel is correct in the language's semantics,
  and nkiequiv rightly proves it. On the device it is wrong, so the fault is in the hardware or compiler
  behaviour for interleaved accumulation groups, which the interpreter doesn't model. A conformance case for
  symnki: `scripts/conformance.py` against `nki.simulate`, then the device.
* **Coverage limits on the rest of our kernel**, per symnki's README: indirect/dynamic DMA (`.ap(vector_offset=...)`:
  our embedding gather, RoPE-row gather, nkilib's KV-cache scatter) is not modelled, nor are collectives.
  nkilib blocks hard-code 128-wide tiles, so they need `--profile full`, which is slow. Gate pieces, not the
  whole 36-layer step.
