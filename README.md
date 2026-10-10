# Hack the Chip — NYU × Annapurna Labs

**Annapurna Labs** is the Amazon team that designs **AWS Trainium**, the custom silicon behind a large
share of AI training and inference on AWS. For one day we are handing you that hardware, a language model
already running on it, and a question we care about.

> **A small model on a chip you control, driven by an agent, can solve problems the model cannot solve on
> its own.** Build the loop that does it, and prove it.

The engineers who design these chips will be on the floor all day.

**The idea in one line:** a model attempts something, a **checker** grades it, and the grade *plus the
reason* feed the next attempt. What decides whether that loop works is not the model — it is the checker.
*"Wrong, off by 341 percent"* is true and useless. *"The term sin(pi\*x) should not be there at all"* names
the change. Everything here was built by learning that the hard way, eight separate times.

Resuming or picking this up cold? [`STATE.md`](STATE.md) has exactly where things stand and what to do next.

---

# Part 1 — Get onto your chip

Every participant has a **seat number**, handed out at check-in. Your seat is a pod on our Trainium
cluster, called `seat-<your number>`. It has its own Trainium chip and this repo at `/workspace`; you
start the model in it yourself, in step 6. You reach it from your own laptop. **Your seat is yours. Don't go into anyone else's.**

Nine steps. Steps 1–5 run on **your laptop**; steps 6–9 run **inside your pod**.

## On your laptop

### 1. Install the AWS CLI and kubectl

**macOS** (with [Homebrew](https://brew.sh)):

```bash
brew install awscli kubectl
```

No Homebrew? Use the [AWS CLI installer](https://awscli.amazonaws.com/AWSCLIV2.pkg), then
`curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/darwin/arm64/kubectl" && chmod +x kubectl && sudo mv kubectl /usr/local/bin/`
(Intel Mac: replace `arm64` with `amd64`).

**Linux** (x86_64):

```bash
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip -q awscliv2.zip && sudo ./aws/install
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/
```

**Windows** (PowerShell):

```powershell
winget install -e --id Amazon.AWSCLI
winget install -e --id Kubernetes.kubectl
```

**Then close the terminal and open a new one**, or it will not find the commands. Check both:

```bash
aws --version
kubectl version --client
```

### 2. Paste the workshop credentials

The organisers post a block in the workshop channel with three versions — **macOS / Linux**, **Windows
Command Prompt** and **Windows PowerShell**. Paste only the one for your terminal, into the terminal you
will use. They look like:

```bash
# macOS / Linux
export AWS_ACCESS_KEY_ID="ASIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
```

```bat
:: Windows Command Prompt (no quotes)
SET AWS_ACCESS_KEY_ID=ASIA...
SET AWS_SECRET_ACCESS_KEY=...
SET AWS_SESSION_TOKEN=...
```

```powershell
# Windows PowerShell
$env:AWS_ACCESS_KEY_ID="ASIA..."
$env:AWS_SECRET_ACCESS_KEY="..."
$env:AWS_SESSION_TOKEN="..."
```

> **These live only in that terminal.** A new terminal, or a new tab, needs them pasted again. They also
> **expire** — when kubectl suddenly says `ExpiredToken` or asks for credentials, paste the newest block
> from the channel. Your pod and your work are untouched. Don't commit them, don't post them anywhere else.

### 3. Point kubectl at the cluster

Same command on every system:

```bash
aws eks update-kubeconfig --name hack-hyd --region ap-south-2
```

It prints `Added new context ... to ...kube/config`. If it fails with an access or credentials error,
the credentials from step 2 are missing from this terminal — paste them again.

### 4. Find your pod

Use **your** seat number; 42 here is an example.

```bash
kubectl get pod seat-42
```

You want `READY 1/1` and `STATUS Running`. `Init` or `0/1` means it is still starting — wait a minute and
ask again. Anything else (`Pending`, `CrashLoopBackOff`, `Error`), find an organiser; they will move you to
a spare seat.

### 5. Get a shell inside it

```bash
kubectl exec -it seat-42 -- bash
```

> **Wait for the new prompt (`root@seat-42:/workspace#`) before typing anything else.** Anything typed
> before it appears goes to your laptop's shell. Want a second terminal? Open one, paste the credentials
> again (step 2), and run the same `kubectl exec` — both land in the same pod.

## Inside your pod

### 6. Start the model

```bash
neuron-ls
cd /workspace
./serve.sh
```

Expect a `NEURON DEVICE` from `neuron-ls`. `serve.sh` then starts the model on your chip and prints
`READY` when it answers on `http://localhost:8000` — about 4 minutes the first time. If it stops with an
error instead, find an organiser.

**Leave this terminal open.** Open a **second terminal**, paste the credentials again (step 2), and get a
second shell in the same pod — `kubectl exec -it seat-42 -- bash`. Steps 7–9 run in that second one.
`./serve.sh --logs` (in `/workspace`) shows the model's log.

### 7. One line of setup

```bash
git config --global --add safe.directory /workspace
```

Needed before **any** git command in here. Your work lives in `/workspace` for as long as the pod does —
**if the pod is replaced, it is gone**, so push anything you want to keep to your own git repo.

### 8. Run project 1 — the heat-rod agent

```bash
cd /workspace/projects/01-heat-rod-pde

python level0_heatrod.py --selftest     # prove the checker BEFORE trusting a score
python agent.py --level 0 --all         # warm-up: solved on round 0

nohup python agent.py --level 1 --all > run.log 2>&1 < /dev/null &    # the real one
tail -f run.log                         # Ctrl+C stops watching; the run keeps going
```

> **Long runs survive a dropped connection only if you start them like this.** A `kubectl exec` session
> can drop (Wi-Fi, laptop sleep, `connection reset by peer`), and anything running in the foreground of
> that shell dies with it. So start the run in the background, writing to a log, and watch the log:
>
> ```bash
> nohup python agent.py ARGS > run.log 2>&1 < /dev/null &
> tail -f run.log        # Ctrl+C stops watching; the run keeps going
> ```
>
> Disconnected? Reconnect with `kubectl exec -it seat-42 -- bash` and pick up where you were with
> `tail -f /workspace/projects/01-heat-rod-pde/run.log`. `pgrep -af agent.py` shows whether it is still running.

`sympy` and `numpy` are already in the pod. **This project is solved, 6 of 6.** Write-up:
[`projects/01-heat-rod-pde/`](projects/01-heat-rod-pde/).

### 9. Run project 2 — the kernel agent

```bash
cd /workspace/projects/20-kernel-agent

python nkibench.py --selftest

nohup python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5 > run.log 2>&1 < /dev/null &
tail -f run.log                         # reconnect: tail -f /workspace/projects/20-kernel-agent/run.log
```

**This project is unsolved — that is the point.** A real open problem, not an exercise with a hidden answer.
Write-up: [`projects/20-kernel-agent/`](projects/20-kernel-agent/).

> The pod sets `HEATROD_BASE_URL`, `KERNEL_AGENT_BASE_URL` and the model names for you, in every shell —
> `env | grep -E 'HEATROD|KERNEL_AGENT'` shows them, so there is nothing to
> export.

**Organisers:** setting up the cluster, sharing the credentials and assigning seats is in
[`workshop/FACILITATOR.md`](workshop/FACILITATOR.md). `serve.sh` runs inside a seat pod; it no longer starts a docker
container, so the standalone-instance notes in [`STATE.md`](STATE.md) are history.

---

# Part 2 — What we already measured, so you do not re-learn it

Every number below came from runs in this repo, on this hardware.

## One run is not a result

Five runs of project 2, identical settings, nothing changed between them:

```
level 1: solved 0/5   all = [0.30, 0.30, 0.30, 0.30, 0.30]
level 2: solved 4/5   all = [1.00, 1.00, 1.00, 0.50, 1.00]
level 3: solved 0/5   all = [0.30, 0.30, 0.30, 0.30, 0.30]
level 4: solved 0/5   all = [0.62, 0.62, 0.62, 0.62, 0.62]
```

Level 2 swings from 0.50 to 1.00 on luck alone. **Report a rate, not your best run** — `--repeat N` does it.
And notice levels 1, 3 and 4 have *zero* spread: those are capability walls, not dice, so changes to them
are cleanly attributable while changes to level 2 are not.

**What that bought, concretely.** A plausible prompt improvement went in — a worked tiling example, aimed at
the exact wall three levels were stuck on. Five runs before, five after: level 2 fell from 4/5 to **0/5**,
level 4 from 0.62 to **0.30**. Strictly worse, and reverted. **Without `--repeat` it would have shipped**,
because one run would have read as ordinary variance.

## Not all tokens are equal

The same ladder against the shared `gpt-oss-20b`, where sampling is **greedy server-side**:

| level | Qwen3-8B (8B, sampling) | gpt-oss-20b (20B, greedy) |
|---|---|---|
| 1 average pooling | 0/5, always 0.30 | 0/3, always 0.30 |
| 2 transpose | **4/5**, 0.5–1.0 | **3/3**, always 1.00 |
| 3 matmul single tile | 0/5, always 0.30 | 0/3, always 0.30 |
| 4 matmul tiled | **0.62** | **0.30** |

* **The greedy runs are byte-identical** — same errors in the same order, same per-round timings to a tenth
  of a second, the same kernel character for character. One model you can bisect; the other you must
  average.
* **Greedy is more reliable where it works:** level 2 solved 3/3 against 4/5. Same ceiling, less spread.
* **The 8B model beats the 20B model on level 4**, 0.62 to 0.30. gpt-oss returned *no code* in four of six
  rounds there, after ~10,000 characters of hidden reasoning each, 21 s per empty round. **Capacity spent
  reasoning is capacity not spent answering**, and an agent loop needs answers.

So *more capable model* did not mean *better agent*. On a greedy model the prompt is your only lever — a
retry that resends the same text is a no-op, which is why the agent keeps a ledger of failed attempts.

## Reasoning modes cost more than they return

Measured on Qwen3, only `--think` changed:

| | round time | result |
|---|---|---|
| thinking off | **~8 s** | code, 0.30–0.62 |
| thinking on | **446 s** | truncated at ~9,900 chars, **0.00** |

Every sample hit `finish_reason=length` with no usable code. **A bigger token budget does not fix this on
either model** — the fix is a shorter prompt. If your agent returns nothing, look there first.

## Settings that are not preferences

| setting | why |
|---|---|
| `TP=2`, not 4 | TP=4 serves fine but round times are identical — the loop is bound by generation length, not compute. TP 2 leaves NC 0–1 free for on-device timing later. |
| `--samples 4` locally, `--samples 1` on gpt-oss | Qwen3's samples differ and are nearly free (the server runs 4 at once). gpt-oss is greedy, so 4 samples are 4 identical answers. |
| `--context 8192` | Repair prompts carry the previous kernel, the checker's instruction and the ledger. At 4096 that squeezes out the answer. |
| `--terse 1` on gpt-oss | A long prompt makes it reason *instead of* answering: 1866 chars → 13,245 chars of reasoning and no code; 581 chars → working code. |

---

# Part 3 — Which model your project talks to

**A NeuronCore cannot be shared by two processes.** But your chip has **four** logical cores and the server
uses two, so two are free. Check with `neuron-top` while it runs: NC 2–3 hold the model, NC 0–1 sit at 0 B.

| your project needs | model from | cores |
|---|---|---|
| no device — the checker runs on CPU | **local Qwen3-8B** | none |
| to compile and run kernels on the chip | **local Qwen3-8B** | pin to the free cores: `NEURON_RT_VISIBLE_CORES=0,1` |
| the whole chip for serving experiments | **shared `gpt-oss-20b`** ([`gptoss/`](gptoss/)), separate hardware | all four |

**Both sample projects are in the first row today**, so nothing has to be remote. Project 1's checker is
SymPy; project 2's simulates kernels on the **CPU** with `nki.simulate`, deliberately — seconds per
iteration instead of a compile, which is where an agent should spend its attempts. Cores are needed only for
real on-device timing, which is not built yet. `NEURON_RT_VISIBLE_CORES` is the documented mechanism and we
have not exercised it, so expect to debug it.

## Why a container, when the instance already has vLLM

The AMI ships Neuron virtualenvs under `/opt`, the newest being vLLM-Neuron **0.21**. That version **cannot
serve Qwen3-8B** — its registry holds only `LlamaForCausalLM`, `GptOssForCausalLM`, `Eagle3LlamaForCausalLM`
and `Qwen3VLForConditionalGeneration`, the last being the **vision** model. Point it at Qwen3-8B and you
get:

```
AttributeError: type object 'Qwen3ForCausalLM' has no attribute 'from_configs'
```

which looks like a broken install and is a model-support gap. `serve.sh` uses the **0.24** container, which
supports it. To use a pre-installed venv instead, `openai/gpt-oss-20b` works there with
`--hf-overrides '{"quantization_config": {}}'` to load in bf16.

Three flags in `serve.sh` are not optional, each found by reading a failure:

* `NEURON_SKIP_EFA_AFFINITY=1` — without it the workers abort with `No EFA device found`. It skips a
  CPU-placement optimization that assumes networking hardware a single chip lacks.
* `--no-enable-prefix-caching` — prefix caching demands a segmented-prefill size of 512 or more.
* `--num-gpu-blocks-override` — required, or you get out-of-bounds errors. `serve.sh` computes it from the
  context length so it stays correct when you change that.

`--tensor-parallel-size 2` matches the hardware: one chip at LNC=2 is two logical NeuronCores. `k8s/` holds
the same configuration as manifests, which is how this was developed; you do not need them on the instance.

---

# Part 4 — The challenge

Build an **agent** around a model: it attempts a problem, something **checks** the attempt, and the check
plus the reason feed the next attempt — until it solves the problem, or can say honestly that it could not.

**What to hand in**, whichever project you pick:

1. **Your checker**, and the reasoning behind what it accepts and rejects. This is the artifact we keep.
2. **Your attempt log** — every attempt with its score, so someone else can see the loop working.
3. **A one-page note**: what you ran, on what, what came out — including how many runs, and the spread.

**You do not need to be a programmer to matter here.** Deciding what counts as correct, and turning a
verdict into an instruction, are the hard parts and neither is a coding task. Teams of 3–5; the teams that
stack five programmers tend to lose these.

## Sample projects

| # | project | status |
|---|---|---|
| **1** | [**The heat-rod agent**](projects/01-heat-rod-pde/) — a small model solves heat-equation problems under a checker that grades the physics, with a calculator it aims itself. | **solved 6/6** |
| **2** | [**The kernel agent**](projects/20-kernel-agent/) — an agent writes NKI kernels that run on the chip and keeps verifying its own output. | **11 of 11 levels** (team 20, see [RESULTS](projects/20-kernel-agent/RESULTS.md)); level 8 with an explicit stage plan |

Both project READMEs open with a real transcript of what the loop prints, so you can judge a project — and
tell progress from flailing — before starting.

Or **propose your own**: it must run on the hardware we give you and produce the three deliverables above.
Find an Annapurna engineer before 11:30 and we will tell you honestly whether it fits in a day.

Adding a project is a folder under `projects/` with a README and a checker. Adding an *operation* to project
2 is one NumPy reference, one input builder, and one registration call — level 8 there is the worked example.

---

# Part 5 — The shared `gpt-oss-20b` endpoint

A 20-billion-parameter open-weight model on Trainium2, behind an OpenAI-compatible API, on hardware separate
from your instance. Use it to compare against your local model, or when you want the whole chip for your own
code.

```bash
export GPTOSS_BASE_URL="https://..."     # the organisers will give you this
pip install httpx
python gptoss/chat.py
```

It behaves in several ways that look like bugs and are not — it thinks before answering, sampling is greedy
so retrying is pointless, and the `tools=` parameter does nothing. **Read
[`gptoss/README.md`](gptoss/README.md) before building against it.** Every number there was measured.

---

## Setup

```bash
pip install -r requirements.txt
```

## License / use

Sample code, provided as-is for the event, free to reuse.
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) and
[`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) are under their own licenses.
