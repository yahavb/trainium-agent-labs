# Gripping Throughput and Feedback

## Roles

Qwen generates NKI source (not implemented in this harness yet).
NKI simulation checks kernel execution on CPU; device execution measures Trainium.
A deterministic CPU evaluator checks returned values using trusted references.
A deterministic feedback formatter explains failed gates and builds next-prompt.txt.
The checker is not an LLM. No second model or additional Neuron cores are needed
for verification or feedback. Suggested failure causes are hypotheses, not proofs.
Scheduling Qwen serving and kernel execution still requires confirmed free cores.

## Metric

Primary target: accepted gripping problems per second. Batch sizes 1/2/4/8/16
each process ALL the same 16 public development cases, without dropped remainders.
Every repeated group must pass pre/post-timing physics checks, input preservation
and zero output padding. Otherwise the entire workload is ineligible for a
throughput claim; raw timings may still be retained as diagnostic evidence.
Different fixed update budgets are permitted only if the same accuracy gates pass.
This measures cost to reach accepted accuracy at that budget, not automatic
convergence or a guarantee that the smallest acceptable budget was found.

Throughput = total problems / total measured time, not an arithmetic mean of
individual throughput rates. Report batch-latency median, min, max and p99 too.
Compile/first call is separate. Device timing excludes compilation and transfers.
Host runtime timing uses preallocated tensors, excludes packing, initial tensor
transfers and final readback. Packing time is logged separately; neither metric
is an end-to-end simulator or full application benchmark.
The existing baseline processes worlds serially within a launch; batching is
not proof of parallel execution or use of multiple cores.

## Public Execution, Trusted Grading

Only public/ goes to the execution environment, not the suite root containing
evaluator-only references. Execution needs grip_throughput.py, batch_reference.py,
update_reference.py, nki_contact_batch.py, benchmark_update.py and contact.py.
Do not copy the entire project directory to a model environment with reference data.
The seat needs its existing NumPy/SciPy/NKI/nrtpy installations, not MuJoCo.

On the seat, from the project directory, after copying these source files and the
public directory to gripping-public, perform a small simulator smoke test:

```bash
python grip_throughput.py run --public gripping-public --backend simulate --batch 2 --steps 32 --runs 1 --out data/grip-sim-smoke
```

Copy data/grip-sim-smoke back to the local project. Grade locally:

```bash
../../.venv-physics-current/bin/python grip_throughput.py grade --suite data/gripping-development-v1 --results data/grip-sim-smoke
```

32 updates are a smoke test, NOT expected to solve every gripping problem. A
nonzero grading exit status means the candidate failed physics, not corrupted
reference data. The report still saves every score and detailed next-prompt.txt.

After smoke testing, and ONLY with confirmed free device cores, execute a device
run with the same public inputs and an explicitly chosen update budget:

```bash
NEURON_RT_VISIBLE_CORES=0,1 python grip_throughput.py run --public gripping-public --backend device --batch 2 --steps 32 --runs 5 --warmup 20 --iters 200 --out data/grip-device-b2
```

The listed core IDs are the earlier allocation, not an assertion they remain free.
Copy results back and grade before claiming any performance. Existing output
directories cannot be reused. Increase batches only after the basic device path
works; do not spend a large compilation budget on an unverified solver.

## Feedback and Audit

execution.json binds public input hashes, contract, source snapshots, output
hashes, workload coverage and raw timing samples. execution-attempts.jsonl records
every execution, including errors. Local grading creates a unique grading-*/
directory containing results.json, scored attempts.jsonl, next-prompt.txt and a
one-page run-note.md. Repeated failure text is deduplicated in the next prompt;
the complete scored log still retains all repetitions. Reference solution arrays
and private cases are not included in feedback.

Physics feedback names the measured residual, force error, acceleration error,
objective error, friction-cone violation or wrench error and its unchanged limit.
It suggests what to inspect without claiming to know the exact implementation bug.
Successful outputs proceed to throughput optimization rather than relaxed gates.

This is a trusted baseline harness, not a sandbox for arbitrary generated code,
nor proof against fabricated remote logs. An isolated generated-code worker and
the Qwen HTTP generation/revision controller remain to be wired. CPU smoke tests
validate grading plumbing; no gripping Trainium result or Qwen improvement is
claimed from them. The frozen frictionless split remains unchanged.
