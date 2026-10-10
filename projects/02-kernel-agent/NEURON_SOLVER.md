# NeuronSolver

NeuronSolver generates and repairs NKI kernels using the existing OpenAI-compatible Qwen3 endpoint and `nkibench` checks. Levels 1–4 are the default evaluation suite; explicit levels 5–8 are supported. No level has been verified solved by NeuronSolver yet.

## Run locally on macOS

Use your existing Python with numpy and httpx, or a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s projects/02-kernel-agent/tests -v
python projects/02-kernel-agent/nkibench.py --selftest
python projects/02-kernel-agent/neuron_solver.py --doctor
```

Do not install Linux Neuron packages on the Mac. `--doctor` reports installed package versions and available simulation API. Other live commands fail early if NKI is missing. The simulator itself runs on CPU, but needs the Linux Neuron SDK; the model endpoint needs serving hardware.

## Deploy from your laptop

Use your own assigned seat. Replace `YOUR_SEAT` below; no seat or credentials were available during local development.

```bash
export SEAT=seat-YOUR_SEAT
kubectl exec "$SEAT" -- git -C /workspace fetch origin feature/neuron-solver
kubectl exec "$SEAT" -- git -C /workspace switch feature/neuron-solver
kubectl exec -it "$SEAT" -- bash
```

For an existing checkout with uncommitted work, preserve that work before switching. If the branch is not available remotely, copy the implementation without touching the baseline or reference files:

```bash
kubectl cp projects/02-kernel-agent/neuron_solver.py "$SEAT":/workspace/projects/02-kernel-agent/neuron_solver.py
kubectl cp projects/02-kernel-agent/repair_engine.py "$SEAT":/workspace/projects/02-kernel-agent/repair_engine.py
kubectl cp projects/02-kernel-agent/nki_knowledge.py "$SEAT":/workspace/projects/02-kernel-agent/nki_knowledge.py
kubectl cp projects/02-kernel-agent/candidate_store.py "$SEAT":/workspace/projects/02-kernel-agent/candidate_store.py
kubectl cp projects/02-kernel-agent/solver_checker.py "$SEAT":/workspace/projects/02-kernel-agent/solver_checker.py
kubectl cp projects/02-kernel-agent/evaluate.py "$SEAT":/workspace/projects/02-kernel-agent/evaluate.py
```

Inside the pod, use the existing Neuron environment. Start `/workspace/serve.sh` in a separate shell if the model is not already running; leave that shell open.

```bash
cd /workspace/projects/02-kernel-agent
export KERNEL_AGENT_BASE_URL=http://localhost:8000/v1
export KERNEL_AGENT_MODEL=Qwen/Qwen3-8B
python neuron_solver.py --doctor
python nkibench.py --selftest
python neuron_solver.py --level 4 --rounds 8 --samples 4 --context 8192
python neuron_solver.py --levels 1 2 3 4 --rounds 8 --samples 4 --context 8192 --probes
nohup python evaluate.py --repeat 5 --levels 1 2 3 4 --rounds 8 --samples 4 --context 8192 > neuron-evaluation.log 2>&1 < /dev/null &
tail -f neuron-evaluation.log
```

Alternative endpoints: pass `--base https://HOST/v1 --model MODEL`. Append any gateway path to `--base` yourself. Inference reuses the baseline's `agent.ask_parallel` and `agent.ask`: temperature 0.6, top_p 0.95, thinking disabled unless `--think`, 900-second request timeout, and its existing TLS behavior (`verify=False`). Context accounting uses the baseline's character/4 estimate; exact token counts and server-side decoding can differ. Use the same context ceiling in `serve.sh` and CLI. No API-key CLI has been added.

## Controller and grading

`neuron_solver.py` owns the bounded generation/repair loop. `agent.py` provides extraction and inference unchanged. A mathematical specification names the actual signature, tensor layouts, shapes, and traffic ceiling; it never calls `inspect.getsource` or reads reference kernels.

`solver_checker.py` executes one candidate in a temporary subprocess. It reuses `nkibench` input builders, rule checks, simulator/traffic counting, numerical mismatch descriptions, input mutation checks, and traffic gates. All official cases run, and hardware-correctness warnings reject a candidate. A subprocess timeout defaults to 120 seconds (`--timeout`); killed checks are marked incomplete and their simulator call counts are unknown rather than measured zero. There is no claim of device compilation or timing from a simulator pass.

Generation has no tools or reference access. Only the grading side calls the mathematical references. Imports of grading/oracle modules and common file-read/dynamic execution functions are rejected. These checks and subprocess isolation are not an adversarial security sandbox: run generated kernels on the dedicated pod, not beside sensitive credentials. The existing checker is also not an exhaustive anti-cheating verifier.

`repair_engine.py` distinguishes dimensions, partitions, DMA shapes, bounds, API names, matmul layout, allocation, reshape/slicing, output shape, missing writes, numerics, traffic, syntax, and partial-shape failures. Diagnostics include checker evidence, possible source locations, repair advice, constraints, and validation steps. Root causes are explicitly hypotheses unless the checker names a location. It uses the baseline's installed-name enrichment for API failures.

`candidate_store.py` hashes normalized Python ASTs, so formatting/comments do not bypass duplicate detection. Every attempted candidate and diagnostic is saved. Repairs use the best candidate that preserves its previously passing case set; regressions are logged and do not replace it. Repeated hashes skip simulation. Repeated categories/hashes trigger a prompt requesting a simpler data flow or different tiling strategy with a failure ledger. If the repair prompt exceeds the estimated context allocation, generation restarts from the concise specification and recent failures. It stops on an official full pass or exhausted rounds. No cross-run candidate cache/resume is implemented.

`--probes` adds index-encoded, impulse and identity/constant inputs on the first official shape. The probes use the real argument signature and operation reference, including nonlinear attention. Probe feedback is included in repairs but does not change official success. Official shapes remain authoritative. No additional irregular-shape generalization or complete write-coverage proof is claimed; future boundary probes can extend this suite.

## Artifacts

The default `solver_runs/` directory is gitignored. Each new invocation gets a unique subdirectory, with sanitized configuration and per-level:

- `attempts.jsonl`: source hash, round, reward, full per-case results, diagnostic evidence, prompt, duplicate and regression flags.
- `<hash>.py`: each extracted candidate; `best.py` and `best.json`: best retained candidate.
- `solved.py`: only after every official check passes.
- `result.json`: solve status, mean/best reward, rounds to success, model/simulator calls, error counts, repair improvements, duplicate count, elapsed time, output hashes, environment.
- `interrupted.json`: model transport failures, when generation raises.

Use `--output /path/outside/repo` for long experiments. Timeout call counts are lower bounds; exact counts are unavailable for the interrupted subprocess. `repair_success_rate` means a nonduplicate repair improves reward without losing previously passing cases. It is not the probability of solving a level. Probe calls count toward simulator calls. No secrets or endpoint URLs are persisted in configuration.

## Evaluation and integrity

`evaluate.py --repeat 5` alternates baseline/solver order, runs both with shared model, endpoint, context, sampling, round budget, samples, and hardware environment, and writes `config.json`, `trials.jsonl`, and `summary.json`. Summary reports per-level solve rate, mean candidate reward, best reward, rounds to success, call counts, category frequencies, and runtime. Individual solver trials report repair rates; baseline repair rates are unavailable. Early stopping and duplicate skipping intentionally affect consumed budgets. Original baseline checks have no subprocess timeout.

**The unchanged original baseline embeds NumPy reference source in its generation prompts.** Its results are labeled `original-baseline-reference-in-prompts`. This is a comparison of implementations with a known contamination asymmetry, not a clean autonomous capability comparison. NeuronSolver never puts reference implementations into generation or repair prompts. Repeated identical normalized output sequences are flagged; requested temperature cannot prove stochastic server decoding, so repeated greedy trials must not be interpreted as independent evidence.

Developer oracle checking is explicit and excluded from evaluation:

```bash
python neuron_solver.py --level 4 --oracle-debug reference_level4.py
```

No oracle mode is available in `evaluate.py`. Baseline `--offline` replay remains unchanged and is not an experiment.

## AWS knowledge and API compatibility

Technical references inspected during implementation:

- [AWS toolkit overview and agent roles](https://github.com/aws-neuron/neuron-agentic-development).
- [NKI writing](https://github.com/aws-neuron/neuron-agentic-development/blob/main/skills/neuron-nki-writing/SKILL.md): memory placement, tiled data flow, and utility patterns. Its Beta 3 examples are not assumed compatible with this pod.
- [NKI debugging](https://github.com/aws-neuron/neuron-agentic-development/blob/main/skills/neuron-nki-debugging/SKILL.md): isolate compiler errors, validate against CPU references, match platform flags, and pin free cores for device validation.
- [NKI docs](https://github.com/aws-neuron/neuron-agentic-development/blob/main/skills/neuron-nki-docs/SKILL.md): selective API/error documentation lookup.
- [NKI profiling](https://github.com/aws-neuron/neuron-agentic-development/blob/main/skills/neuron-nki-profiling/SKILL.md): real hardware profiles use compiled NEFF and execution NTFF artifacts; simulator byte counts are not latency profiles.

`nki_knowledge.py` provides small category-specific programming cards and retrieves actual installed ISA signatures. It records SDK/package versions and detects `nki.simulate` versus `nki.simulate_kernel`; `nkibench` handles that API distinction. It does not import the toolkit's Kiro/Claude agent integrations or retrieve full toolkit files into prompts. There is no automatic web retriever or Beta-version migration. If the installed SDK differs substantially, inspect `--doctor` and live signatures before changing knowledge cards. Unknown package versions remain explicitly unknown.

## Verified results and outstanding work

Local macOS development: 12 unit tests pass; existing `nkibench.py --selftest` passes its CPU-only checks. Tests cover classification, prompt integrity, candidate hashing/storage, duplicate skipping/fallback, full-pass stopping, regression retention, timeout handling, subprocess syntax rejection, grading import boundaries, traffic gates, nonlinear attention probe signatures, and repeated-trial reporting. Mocked simulator tests verify controller behavior, not NKI correctness.

NKI is not installed locally. No remote seat or inference endpoint was supplied. **No measured NeuronSolver solve rate or verified solved level is available.** Outstanding: run the above evaluation on the pod; inspect real NKI failure diagnostics; test SDK compatibility and simulator counting; compile/run accepted kernels on Trainium2; profile only after correctness is established. Levels 5–7 additionally require traffic improvements and level 8 is experimental. Historical baseline measurements in the repository are not new measurements.
