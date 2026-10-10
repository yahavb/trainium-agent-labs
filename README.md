# nki-sched

A scheduling language on top of NKI: write the algorithm in torch, transform it with a short schedule
(tile, stage into SBUF/PSUM, tensorize, multi-buffer, pipeline, ...), emit a verified NKI kernel.

**Status: plan/design only, no code yet.**

* [`PLAN.md`](PLAN.md): architecture, IR, lowering, correctness strategy, MLIR decision, roadmap
* [`SYNTAX.md`](SYNTAX.md): language draft; schedules reproducing the nkibench level 4->7 matmul ladder
* [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md): Halide / TVM / Exo digest and what we take from each
