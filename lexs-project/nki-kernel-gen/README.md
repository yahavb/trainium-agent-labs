# NKI kernel search

## NKIEvolve endpoint (run inside seat-56)

Use the same Python environment that already runs `nki_evolve.py` on Trainium.
Install the API/tunnel dependencies once, then start the server:

```bash
cd /path/on/the/chip/nki-kernel-gen
python -m pip install 'fastapi>=0.115,<1' 'pydantic>=2.9,<3' 'uvicorn>=0.30,<1' 'ngrok>=1.4,<2' 'psutil>=6,<8' 'openai>=1.60,<3'
# Or, with the existing OpenEvolve dependency installed: pip install -e '.[server,tracking]'

# Set these in your shell using your own credentials; do not commit them.
export NGROK_AUTHTOKEN="<your-ngrok-token>"
export OPENAI_KEY="<your-openai-key>"
export WANDB_KEY="<your-wandb-key>"
export WANDB_PROJECT="trainium-kernels"
export WANDB_ENTITY="ahackett-santa-clara-university"

python backend.py
```

The script prints a public HTTPS endpoint URL and a generated API access token.
The ngrok SDK opens the tunnel directly; no separate ngrok binary or `kubectl`
command is needed. `OPENAI_API_KEY` and `WANDB_API_KEY` are also accepted by the
existing runner. Set `NKIEVOLVE_API_TOKEN` (at least 24 characters) before starting
to reuse your own access token. The API token is separate from provider keys.
Provider keys stay in the worker environment and are not included in job configs.

Keep the process alive, for example in `tmux`. Ctrl+C closes the tunnel and
cancels the active job. `python backend.py --local-only` skips ngrok; `--port 8001`
changes the local port. The default bind address is loopback. Outbound network
access to ngrok, OpenAI and W&B must be available from the pod.

Keep the sibling `frontend/` directory alongside `nki-kernel-gen/` on the worker.
Open `<endpoint>/` for the NKIEvolve UI: Launcher, Logs and Kernel Code. Paste
the generated API access token into its connection bar. Description input makes
a paid GPT-6 Luna call and returns editable PyTorch for review before launch.
The UI and API share the same origin, so no CORS configuration is needed.

Open `<endpoint>/docs` for interactive API documentation. Click **Authorize**
and paste the generated API token. All API routes require
`Authorization: Bearer <token>`. For a separately hosted frontend, set
`NKIEVOLVE_ALLOWED_ORIGINS` to a comma-separated list of exact frontend origins
before startup. CORS is disabled by default.

The API runs the existing `nki_evolve.py` in hardware mode with the requested
model. It reuses `context_files` from the repository's `config.json`; it writes
new references, configs and outputs under `outputs/api/<job-id>/`.
Only one run can use this API worker at a time; another launch returns HTTP 409.
Use a single server process. Completed jobs survive server restarts; an unfinished
job is marked interrupted after a restart and is not automatically resumed.

| Route | Result |
| --- | --- |
| `GET /health` | API status, supported models, active job ID |
| `POST /references/generate` | Description → PyTorch source for review; does not execute it |
| `POST /runs` | Launch a job; returns HTTP 202 with its `id` immediately |
| `GET /runs` | List saved jobs |
| `GET /runs/{id}` | Job status, live metrics, W&B URL, kernel availability |
| `GET /runs/{id}/logs` | Last 128 KB of console output |
| `GET /runs/{id}/kernel` | Best correct, measured NKI kernel as Python text |
| `POST /runs/{id}/cancel` | Stop an active job and its evaluator processes |

Example request body for `POST /runs`:

```json
{
  "reference_code": "import torch\n\ndef reference(a, b):\n    return a + b\n\ndef cases(seed):\n    g = torch.Generator().manual_seed(seed)\n    for shape in [(128, 512), (129, 513)]:\n        yield (torch.randn(shape, generator=g), torch.randn(shape, generator=g))\n\nRTOL = 1e-5\nATOL = 1e-6\n",
  "verified": true,
  "module_name": "tensor-add",
  "model": "gpt-6-luna",
  "iterations": 10,
  "timeout": 600,
  "wandb": true
}
```

`reference_code` must define both `reference(*inputs)` and `cases(seed)`.
`verified: true` records the caller's approval before executing the reference;
the API checks syntax and entry points, not semantic correctness. Models are
restricted to `gpt-6-luna`, `gpt-6-sol` and `gpt-6-astra`.
Optionally send `initial_kernel_code`, with a plain `kernel(*inputs)` function
and evolution markers. Otherwise the existing runner generates the initial NKI
kernel. Set `wandb: false` to disable uploads; local metrics are still collected.
The status endpoint returns the same rows as W&B, including `kernel/speedup`
and `kernel/latency_us`; poll it every 2–3 seconds for frontend charts.
Speedup is relative to the starting NKI kernel, not PyTorch. The kernel endpoint
returns 404 until a correct hardware-measured baseline/candidate has been saved.

This is an authenticated API for trusted users. Submitted references and generated
kernels execute Python on the worker; the existing harness is not a security
sandbox. Anyone holding the API token can generate references and start paid searches.

API-only tests (no Trainium, W&B or model calls):

```bash
python -m pip install pytest httpx
python -m pytest tests/test_backend.py -q
```

This project lives at `lexs-project/nki-kernel-gen/` in the enclosing repository.
It shares that repository's Git history. OpenEvolve is a sibling dependency
registered as a submodule; there is no separate Git repository inside this folder.

## JSON entry point

```bash
python nki_evolve.py --config config.json
```

The included `config.json` searches a row-wise softmax implementation:

```json
{
  "reference": "examples/softmax/reference_pytorch_softmax.py",
  "module_name": "fused-softmax",
  "initial_kernel": "examples/softmax/initial_kernel.py",
  "mode": "hardware",
  "model": "gpt-6-luna",
  "iterations": 10,
  "timeout": 600,
  "output": "outputs/softmax"
}
```

Paths resolve relative to the JSON file. Set `OPENAI_API_KEY` or add `--prompt-key`
to enter it without echo. `--check` validates the starting kernel without model
calls. Hardware commands must run where `neuron-ls` can access the device, outside
the Codex execution sandbox in this environment.

`OPENAI_KEY` is also accepted as an alias for `OPENAI_API_KEY`, and `WANDB_KEY`
as an alias for `WANDB_API_KEY`. Canonical credential variables take precedence
if both are set. `WANDB_PROJECT` overrides the W&B project in the config and
accepts `entity/project`, which is normalized into separate environment fields
before W&B initialization. Prefer `WANDB_ENTITY=ahackett-santa-clara-university`
and `WANDB_PROJECT=trainium-kernels`. Export these variables in your shell, then run
`python nki_evolve.py --config config.json` without prompt flags. Plain shell
assignments must be exported for the Python process to see them.

Only `reference` is required for a search. If `initial_kernel` is omitted, the
runner asks the model to write one, tests it in the simulator, and feeds errors
back for up to `bootstrap_attempts` attempts (default 3). This makes paid model
calls and saves each generated candidate and evaluation report under `output`.
A passing starting kernel is then validated on hardware before evolution starts.
Softmax passes all 15 simulator and Trainium2 correctness cases. Its initial
measured mean of per-shape p50 device runtimes was approximately 25 microseconds.
The reference Python module must define `reference(*inputs)` and `cases(seed)`
as described below; a bare function without input cases is insufficient.
Other optional fields are `checkpoint`, `command` (`search` or `check`), and
`prompt_key` (boolean). Unknown fields are rejected.

## NKI authoring context and evolution history

The supplied config sets `context_files` to an empty list so it works without
additional repository checkouts. Optionally add paths to NKI authoring Markdown
documents; paths resolve relative to the config file and missing documents are
errors. The runner embeds their bodies in the system prompt and strips agent
metadata. The harness contract and installed API signatures take precedence.

OpenEvolve still performs all parent selection, island scheduling, population
updates and history selection. Its normal prompt includes the selected parent,
parent metrics and error artifacts, top programs from that island, and sampled
inspirations. It does not give every request the entire history or all earlier
runs. Use `checkpoint` to resume a population from a previous run.

The generation client preserves OpenEvolve's system/user messages and records
them in `output/prompts/request_*.json`. Its GPT-6 adapter sends
`max_completion_tokens` and omits sampling controls while reasoning is enabled.
These files let you verify exactly which source, previous candidates and compiler
feedback were sent to GPT; client configuration and API keys are not recorded.

A separate project using OpenEvolve to evolve NKI implementations of a trusted
PyTorch operation. OpenEvolve generates changes, evaluates candidates, maintains
its population and saves the best programs and checkpoints. Its repository is
used as a dependency and does not need source changes.

## Install

From this directory:

```bash
python -m pip install -e '.[server,tracking]'
```

OpenEvolve is installed as a Python dependency; no Git submodule is needed.

Install the AWS Neuron SDK separately in a supported environment. This project
targets the current `nki` API, with explicit ISA destination tensors (tested with
compiler 2.27.5334.0). Hardware evaluation requires a visible Trainium device and
Neuron runtime. The standalone compile adapter uses version-specific SDK internals
in `hardware.py`; SDK upgrades may require updating that adapter. Logical core
degree follows `NEURON_LOGICAL_NC_CONFIG` (2 on this Trainium2 instance).

## Check and search

```bash
# CPU simulator: verifies correctness, supplies no performance estimate.
nki-search check --task examples/add/task.py \
  --initial examples/add/initial_kernel.py --mode simulate

# Run on a Trainium host. Securely enter a key when prompted.
nki-search search --task examples/add/task.py \
  --initial examples/add/initial_kernel.py --mode hardware \
  --model gpt-6-luna --iterations 10 --prompt-key
```

Alternatively supply `OPENAI_API_KEY` through your environment. Keys are not saved
in project files and provider credential variables are removed from evaluation
workers. Each search iteration makes paid model calls. `--model` selects the
OpenAI model; access depends on your account. `--output` chooses the OpenEvolve
artifact directory, and `--checkpoint /path/to/checkpoint` resumes a run.
Simulation searches are useful to debug generation, but every correct candidate
receives the same score, so they cannot discover the fastest kernel.

## Your own PyTorch operation

Create a trusted task module containing:

```python
import torch

def reference(a, b):
    return a + b

def cases(seed):
    g = torch.Generator().manual_seed(seed)
    yield (torch.randn(128, 512, generator=g),
           torch.randn(128, 512, generator=g))

RTOL = 1e-5
ATOL = 1e-6
```

Provide an initial NKI source file with a plain `kernel(*inputs)` function and
`# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers around editable code. The
harness applies NKI execution decorators. The initial kernel must pass validation
before search starts. Reference source is included in the optimization prompt.
Cases are positional tuples of CPU PyTorch tensors and scalar/static arguments;
outputs must be tensors or tuples/lists of tensors. Float32/float16 are supported
through NumPy conversion; PyTorch bfloat16 needs a dedicated conversion adapter
and is not supported in this first version.

The reference runs on CPU; vanilla PyTorch code does not automatically execute on
Trainium. Neuron PyTorch/XLA integration is needed for that. Here PyTorch supplies
ground truth, while NKI supplies the Trainium implementation.

## Evaluation and score

Three seeds (0, 17, 42) exercise every task case. Exact output shapes/dtypes,
finite values and numerical tolerance must all pass. Hardware mode checks both
simulator and real standalone NKI outputs, then benchmarks with 10 warmup and 100
iterations using the Neuron runtime device profiler. Correctness is checked before
timing. Each distinct input shape/dtype/static-argument combination is timed once,
while correctness runs across all seeds and edge cases.
The score is the geometric mean of initial-kernel / candidate p50 NeuronCore
latency across cases; a score above 1 means faster than the initial NKI kernel.
`latency_us` is the arithmetic mean of those per-shape medians. Speedup and minimum
mean latency can select different winners, so both are tracked separately.
This is not speedup over compiled PyTorch. Compilation time is not scored.
Failures and timeouts receive zero and return error artifacts to OpenEvolve.

Evaluation uses a fresh subprocess, a temporary working directory, a host lock
to serialize device use, and a process-group timeout. Generated Python is still
executable code: this is process isolation, not a security sandbox. Use a dedicated
worker without credentials or valuable writable files for untrusted candidates.
Correctness is limited to your supplied cases; include varied shapes, tails,
extreme inputs and operation-specific edge cases. A search finds the best measured
candidate in its budget; it does not prove global optimality.

## Tracking

```bash
python -m pip install -e '.[tracking]'
python nki_evolve.py --config config.json --prompt-key --prompt-wandb-key
```

Add this JSON block to enable W&B:

```json
"wandb": {
  "enabled": true,
  "entity": "ahackett-santa-clara-university",
  "project": "trainium-kernels",
  "mode": "online"
}
```

W&B expects `WANDB_API_KEY`; `WANDB_KEY` is accepted as an alias. You can use
`--prompt-wandb-key` to supply it without echo or saving it to project files.
Online mode uploads numeric metrics to the configured project. Offline mode
needs no W&B credentials; synchronize it later with `wandb sync`.

Run names default to `<module_name>-<model>`, so the supplied config creates
`fused-softmax-gpt-6-luna`. Set `module_name` in the JSON config or pass
`--module-name` to `nki-search`. When omitted, the reference file's stem supplies
the module name; underscores become hyphens. An explicit `wandb.name` overrides
the generated name.

Every search also saves `metrics.jsonl`, `runtime_history.png`, a candidate source
for each evaluated iteration, and `best_kernel.py` (best geometric speedup).
Baseline is iteration 0. W&B charts use `iteration` on the x-axis and record
`kernel/latency_us`, `kernel/best_latency_us`, `kernel/speedup`,
`kernel/best_speedup`, and `kernel/correctness`. Failed candidates have no latency
point. These are evolutionary iterations, not training epochs.

## Tests

```bash
python -m pytest -q
```

Tests cover real simulator correctness, rejection of incorrect output, output
contracts, subprocess timeouts and the OpenEvolve configuration. Device benchmarks
need a Trainium host and are not tested by simulator tests.

References: [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve),
[AWS NKI benchmark API](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/nki/api/generated/nki.benchmark.html),
[OpenAI Chat Completions](https://developers.openai.com/api/reference/chat-completions/overview).
