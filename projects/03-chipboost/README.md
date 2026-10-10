# CHIPBOOST

Can Qwen3-8B, served on one Trainium2 chip, make the matmul it is built from faster on that same chip,
under a referee strict enough that no speedup can be faked?

Built at Hack the Chip (NYU × Annapurna Labs), 10 October 2026. Numbers are measured on the chip unless
marked [sim]. [NOTE.md](NOTE.md) is the one-page note, with runs and spread.

## Setup

- **Shapes:** Qwen3-8B's per-core matmuls at tensor parallelism 2 and 256 tokens: gate_up (4096×256×6144)
  plus q_proj (4096×256×2048). The score is their summed time.
- **Start kernel:** the NKI tutorial's tiled matmul, 960 µs.
- **Expert kernel:** AWS's published "fully optimised" matmul with two fixes (fp32 accumulation, and block
  sizes that fit Qwen3's shapes), 385 µs (2.49×).
- **Hardware:** one Trainium2 chip per seat, seats 100 to 102. Neuron SDK 2.32, NKI 0.6.0.

## Results

| | |
|---|---|
| **Referee** | Caught 34 of 36 planted cheats and accepted 9 of 9 honest kernels. Neither cheat it missed gained any speed. |
| **AWS tutorial bug** | AWS's published matmul rounds its running sum to bf16 once per K-block. It fails its own correctness check at K=8192 [sim] and a held-out Qwen3 shape. An fp32 accumulator fixes it, at no measurable speed cost. |
| **Block-size tuning** | Random search, 3 runs × 24 tries: 1.325–1.372× over AWS's defaults. Measuring all 62 settings puts AWS's default at #29 and the ceiling at 1.375×. The best setting does not transfer: +84% to −61% on unseen shapes. |
| **The model** | First version, alone or with the referee's feedback: 0 faster kernels in 96 attempts. With feedback that names the change to make: in its first run, a correct kernel 1.517× faster on the third attempt, and correct on 6 of 6 unseen shapes (1.52× geometric mean). Repeats and the success rate are in NOTE.md. |

- **Also measured, not a referee verdict:** splitting the expert across both physical cores of an LNC=2
  NeuronCore gives 1.50× more (5.0× the start kernel).
- **Not claimed:** any end-to-end Qwen3 speedup. No kernel was plugged into the served model.

## How the referee decides

1. **Rules:** banned calls and imports. The candidate runs in a sandboxed child process.
2. **Correctness:** first in the CPU simulator, then on the chip with hostile inputs.
3. **Timing:** interleaved A/B against the start kernel, on the device clock. Under 1% counts as no gain.
4. **Held-out shapes:** a kernel that would be "faster" must also pass three random shapes it has never seen.
5. **Feedback:** one instruction goes back to the model.

## Where things are

| | |
|---|---|
| **Checker and its reasoning** | [`speedcheck.py`](speedcheck.py), [`timing.py`](timing.py), [`REFEREE.md`](REFEREE.md) |
| **Red team** | [`redteam/`](redteam/): cheating kernels and their results |
| **Attempt logs** | [`logs/seat-101/`](logs/seat-101/) (model), [`logs/seat-102/`](logs/seat-102/) (random search, sweep), [`experiments/`](experiments/) (seat 100) |
| **Dashboard** | [`dashboard/index.html`](dashboard/index.html), built from the logs by `dashboard/build.py` |
| **Kernels** | [`kernels/`](kernels/): start, expert, AWS as published, the model's 1.517× kernel |
| **Per-owner detail** | [`STATUS.md`](STATUS.md) (P1), [`P2_STATUS.md`](P2_STATUS.md), [`P3_STATUS.md`](P3_STATUS.md) |
| **Original plan** | [`PLAN.md`](PLAN.md) |

## Run it (in a seat pod)

```bash
cd projects/03-chipboost && export CHIPBOOST_SEAT=102
python speedcheck.py --op matmul --check kernels/matmul_expert.py   # one referee verdict
python agent.py --arm referee --budget 8                            # the model loop (needs the vLLM server)
python search.py --budget 24 --seed 0                               # random search over block sizes
python heldout_grid.py --op matmul                                  # every arm's best on unseen shapes
python tools/aws_matmul_bf16_repro.py --sim                         # the AWS bug, CPU only
python dashboard/build.py                                           # rebuild the dashboard from the logs
```

## Team

| Role | |
|---|---|
| P1: referee and timing | [likhith2366](https://github.com/likhith2366) |
| P2: kernels, search, held-out grid | Jithendra Puppala ([jithendra1798](https://github.com/jithendra1798)) |
| P3: model loop and red team | Siva Balan ([Sivabalan21](https://github.com/Sivabalan21)) |
| P4: dashboard and note | Bala Sai Manikanta Sandeep Puppala ([manikanta-sandeep](https://github.com/manikanta-sandeep)) |
| Review and integration | Nihal Ajayakumar ([anihal](https://github.com/anihal)) |

Resumes are in [`resumes/`](resumes/).
