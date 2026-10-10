# Trainium results from seat 212 (own CPU reference, not the frozen fixture)

Status: measured on 2026-10-10. These results do **not** count as Trainium v0. They use our own
`bench.py` check, not the team's frozen CPU fixture. Their inputs are synthetic: random fields
with a fixed 30% land mask. They show that the port computes the same function as the CPU and
how fast it runs. Trainium v0 still needs `candidate_adapter.py` run against
`fixtures/cpu_reference.npz` and a pass from `checker.py` once the tolerances are frozen.

Setup: one Trainium2 logical NeuronCore (core 2). torch 2.11, torch-xla 2.11 + libneuronxla
3.0.5356, neuronx-cc 2.27.5334, Neuron runtime 2.34. Batch 1 unless noted. Each value is the
median of 20 timed passes after 3 warm-up passes. The CPU baseline is the same model in fp32 on
the pod's 11 vCPUs. Check: relative RMS error over ocean cells must be 3% or less, and every
land cell must be exactly zero.

| Grid | Weights | Setting | ms per sample | Rel. RMS error vs CPU fp32 |
| --- | --- | --- | ---: | ---: |
| 2 deg (90x180) | random | CPU, 11 threads, fp32 | 364.2 | reference |
| 2 deg | random | Trainium fp32 | 58.2 | 1.1e-6 |
| 2 deg | random | Trainium bf16 (`--dtype bf16`) | 15.5 | 1.0e-2 |
| 2 deg | random | Trainium bf16, batch 2 | 9.9 | 1.0e-2 |
| 2 deg | random | Trainium bf16, cores 2 and 3 in parallel (2,000 passes each) | 15.6 per core, about 128 samples/s combined | 1.0e-2 |
| 1 deg (180x360) | Samudra2 `onedeg` | CPU, 11 threads, fp32 (1 pass) | 1,648.7 | reference |
| 1 deg | Samudra2 `onedeg` | Trainium fp32 | 230.5 | 1.2e-6 |
| 1 deg | Samudra2 `onedeg` | Trainium fp32 model + compiler auto-cast to bf16 | **52.6** | 1.0e-2 |
| 1 deg | Samudra2 `onedeg` | Trainium `--dtype bf16` | compile fails (`NCC_IBIR229`) | (none) |
| 1/2 deg | Samudra2 `halfdeg` | Trainium fp32 | does not fit: needs 28.5 GB, a core has 24 GB | (none) |

No effect: `--model-type=unet-inference` (229.9 ms on 1 deg fp32; 15.7 ms on 2 deg bf16), and
`--optlevel=1` for the bf16 crash.

Commands: see `trainium/README.md`. For example, the 52.6 ms run is
`NEURON_CC_FLAGS="--auto-cast=matmult --auto-cast-type=bf16" python bench.py --device neuron --grid 1deg --hf onedeg`.
