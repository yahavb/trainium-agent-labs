# nki-sched

A scheduling language on top of NKI: write the algorithm in torch, transform it with a short schedule
(tile, stage into SBUF/PSUM, tensorize, hoist, ...), emit a verified NKI kernel.

**Status:** the tiled matmul works end to end (torch spec -> loop IR -> schedule -> NKI source). Multibuffer /
pipeline / overlap, symbolic blocking and attention are still plan. See `PLAN.md` section 13 for exactly
what was verified and what was not.

* [`PLAN.md`](PLAN.md): architecture, IR, lowering, correctness strategy, MLIR decision, roadmap, status
* [`SYNTAX.md`](SYNTAX.md): schedule-language draft and the worked matmul schedules
* [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md): Halide / TVM / Exo digest

## Try it

```
pip install -r requirements.txt
python -m pytest                          # unit tests; no Neuron SDK needed
python examples/loops.py                  # split / reorder / compute_at: prints the IR after each step
python examples/memory.py                 # staging, set_memory, tensorize: prints the IR and the NKI kernel
python examples/matmul_ladder.py l4 out.py   # emit the tiled-matmul kernel (l5-l7 are hoisting examples)
```

On a Trainium node (integration; not part of the unit tests):

```
ENV_FILE=../.env scripts/device_check.sh l4        # nkibench + simulator f32/bf16 + a real NeuronCore run
python scripts/nkibench_rules.py out.py 4 --nkibench /path/to/nkibench.py   # just the static rule scan
```
