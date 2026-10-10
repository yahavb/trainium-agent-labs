# Contact physics: first development milestone

The separate frictional gripping task, checker reasoning, development cases and
reproduction commands are documented in `GRIPPING.md`. It extends the physics
to two-pad pyramidal-friction snapshots without modifying the frozen frictionless
suite. No Qwen or gripping-device result is claimed yet.

`THROUGHPUT.md` describes the public-only execution harness, trusted CPU grading,
accepted-worlds-per-second metric and natural-language next-iteration feedback.
The checker is deterministic code, not a second model running on Neuron cores.

`QWEN_EXPERIMENT.md` records the failed simulator baseline, compact feedback,
the first-generation helper, and evidence required for later correctness and
throughput claims. Candidate generation is not verification or model training.

`NKI_FEEDBACK.md` documents official-source API guidance and installed-SDK
signature capture added to new Qwen requests. Deployment now also needs
`nki_guidance.py`; hardware bottlenecks remain hypotheses until profiled.

Engine-backed validation and the dataset/model-evaluation contract are described
in `BENCHMARK_PROTOCOL.md`. Install optional `requirements-engine.txt` to run:

```bash
python engine_validation.py
python engine_validation.py --replay data/engine-validation-<run-id>
```

These cases compare extracted MuJoCo force QPs with an independent solver. They
are distinct from the synthetic impulse fixtures below. No Qwen generation or
training has run. A frozen public/private split can now be built and verified:

```bash
python build_engine_sets.py --out data/engine-sets-v1
python build_engine_sets.py --verify data/engine-sets-v1
```

Existing output directories are never overwritten. Only deploy the `public/`
subdirectory to a candidate environment. Keep `evaluator-only/` outside that
environment; it contains private cases and all oracle outputs. See the protocol
for frozen candidate gates and remaining execution-isolation requirements.

This folder implements analytical frictionless sphere-contact snapshots, an FP64
NNLS reference for the constrained QP, a CPU FP32 projected-gradient baseline,
planted-bug checks, a seat resource probe, and an initial fixed-step NKI update.
The NKI update passed the initial 24-case device validation. It does not yet implement a
converged NKI contact solver, MuJoCo extraction, rollouts, the agent, or training.

The snapshots describe instantaneous contacts: independent spheres against planes,
pairs of spheres and vertical stacks. Sphere normal impulses pass through their
centers, so translational dynamics suffice here. Regularization is fixed to 1e-3;
restitution and position stabilization are zero. These are normalized synthetic
physical inputs, not captured trajectories. Do not claim engine equivalence.

From this folder, using the repository's local environment:

```bash
../../.venv-physics-current/bin/python -m unittest -v
../../.venv-physics-current/bin/python generate.py --workers 4 --runs 5
```

Generated fixtures are in `data/development/`; the manifest records oracle checks,
conditioning, FP32 failures and iterations. Failed baseline cases remain recorded.
The eigenvalue computation and oracle are offline preparation, not timed solves.

`--workers` parallelizes independent fixtures with one numerical-library thread
per worker. Automatic selection respects the cgroup CPU quota and caps at eight.
On seat-260, start with eight workers against the measured 11-CPU quota; compare
generation wall time before increasing it. This is not a proven optimal setting.
`--runs 5` repeats each candidate five times. Every score is appended to
`data/cpu-attempts.jsonl`; existing attempts are retained. Each invocation has a
unique run ID and writes `data/development/run-note.md` with counts and spread.
Checker reasoning is in `CHECKER.md`. These are the initial submission artifacts;
the actual model attempt loop and hardware results still need implementation.

Inside the pod, after copying updated files:

```bash
python -m unittest -v
python generate.py --workers 8 --runs 5
```

The generator defaults to reference-free `--stopping velocity-bound`. It checks
the original problem in FP64 and requires a conservative velocity-error bound as
well as the residual gate. The 4096-update limit and checker thresholds remain
fixed. To preserve the previous development run, use a new output directory:

```bash
python generate.py --workers 8 --runs 5 --out data/velocity-bound
```

`--stopping residual` retains a residual-only stopping option (now independently
evaluated in FP64). Historical runs keep their original source hash and results.

From the local repository root, with workshop credentials in that terminal:

```bash
kubectl cp projects/03-contact-physics seat-260:/workspace/projects/03-contact-physics
kubectl exec -it seat-260 -- bash
```

Inside the seat:

```bash
cd /workspace/projects/03-contact-physics
python probe_seat.py
```

Check resource limits and packages before installing anything. To verify hardware
compilation, inspect `neuron-top` and identify cores the model is not using. Only
if 0 and 1 are confirmed free, run:

```bash
NEURON_RT_VISIBLE_CORES=0,1 python probe_seat.py --compile-smoke
```

This runs the existing reference matmul on hardware; it is a compatibility test,
not a physics or latency benchmark. Read the resulting exit status and report.
Copy the JSON to your laptop from the local terminal:

```bash
kubectl cp seat-260:/workspace/projects/03-contact-physics/seat-capabilities.json ./seat-capabilities.json
```

Compute allocation: CPU generates fixtures and verifies results; Trainium serves
Qwen and, on independently verified available cores, executes candidate kernels.
Reported host CPU count and filesystem capacity are not dedicated pod allocations.
The cgroup probe helps establish actual limits. LLM fine-tuning feasibility remains
unverified; it is not needed for this first milestone.

## Fixed-step NKI validation

`nki_contact.py` implements one world's projected-gradient updates, keeping its
matrix and impulse vector on chip between updates. It is a correctness baseline,
not an optimized batched implementation. Contact counts are padded to powers of
two up to 128; padded rows and columns have zero coupling. `check_update.py`
checks agreement with fixed CPU updates, zero padding and unchanged inputs, while
recording full-physics correctness separately. Update score 1 does not certify
convergence. Every test case appends a scored record to `data/update-attempts.jsonl`.

Inside seat-260, first simulate a small suite, then run on confirmed free cores:

```bash
python -u check_update.py --backend simulate --contacts 8 --steps 8 --seeds 1
NEURON_RT_VISIBLE_CORES=0,1 python -u check_update.py --backend device --contacts 8 --steps 8 --seeds 1
```

These runs print progress before each call. If simulation fails, diagnose it before
compiling. Only after both small runs pass, use the default 24-case suite including
33-contact ragged inputs. First-call elapsed time includes possible compilation.
Post-first-call host timings are smoke diagnostics, not a speedup benchmark.

## Timing diagnostic

With the model stopped and the selected cores confirmed available:

```bash
NEURON_RT_VISIBLE_CORES=0,1 python -u benchmark_update.py
```

This records five independent benchmark invocations, each with 20 warmups and
200 measured iterations per timing mode. A JIT call preserves the compiled NEFF;
it is then loaded and validated with `nrtpy` using the same inputs.
Device timing measures NeuronCore execution including device DMA, while host
timing includes communication overhead. Compilation and loading are outside these
timings. Full physics scores remain separate from fixed-update equivalence.
See the [AWS nrtpy benchmark tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrtpy/tutorial-validate-benchmark.html).
Unknown compiled tensor names or unavailable APIs produce an error report instead
of an automatic SDK change. Results, spread and a run note are saved in
`data/benchmark-update-<run-id>/`; every attempt is appended to
`data/benchmark-attempts.jsonl`. Hardware timing still needs to run on the seat.

## Independent-world launch baseline

`nki_contact_batch.py` packs distinct matrices and step sizes for up to 16 worlds
into one launch. The worlds execute serially; this is a launch-amortization
baseline, not a multi-core or parallel-world performance claim. Equal active
contact counts are currently required. Fixture generation and packing are outside
timing; every world is checked against its own CPU updates and physics oracle.

```bash
python -u check_batch.py --backend simulate --batch 2
NEURON_RT_VISIBLE_CORES=0,1 python -u check_batch.py --backend device --batch 2
```

Run the device smoke only after simulation succeeds. A successful smoke can be
followed with `--runs 5 --warmup 20 --iters 200`, then repeated at batch sizes
1, 2, 4 and 8. Compare both ms/batch and worlds/second, preserving per-world physics
scores. Do not treat update equivalence as solver convergence. Results, timing
spread and a note are in `data/batch-<backend>-<run-id>/`; append-only attempts
are in `data/batch-attempts.jsonl`. Stack tests are available with `--scene stack`.
