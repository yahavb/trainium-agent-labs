# Hack the Chip — NYU × Annapurna Labs

**Annapurna Labs** is the Amazon team that designs **AWS Trainium**, the custom silicon behind a
large share of AI training and inference on AWS. For one day we are handing you that hardware, a
language model already running on it, and a question we care about.

> **A small model on a chip you control, driven by an agent, can solve problems the model cannot
> solve on its own.** Build the loop that does it, and prove it.

The engineers who design these chips will be on the floor all day.

---

## 1. Get your instance running

You get a **`trn2.3xlarge`** for the day: one Trainium2 chip, 4 NeuronCores at LNC=2, ~96 GB of
on-chip memory, root access. Yours. If you break it we hand you another one.

**Access:** «TODO — organisers: instance id, SSM command, per-team card»

Everything below is the exact sequence, in order, verified on a real instance. **Use two terminals:**
one for the model, one for your work.

### Terminal 1 — the model

**1. Confirm you have a chip.**

```bash
neuron-ls
```

You should see one NEURON DEVICE with 4 cores and 96 GB. If not, stop and find an organiser.

**2. Clone the repo.**

```bash
cd ~ && git clone https://github.com/yahavb/trainium-agent-labs.git
cd trainium-agent-labs
```

**3. Check Docker, and install it only if missing.**

```bash
docker --version || ./install-docker.sh
```

If it installs, it may tell you to run `newgrp docker` before continuing.

**4. Start the model.**

```bash
./serve.sh
```

That serves **Qwen3-8B** on your chip behind an OpenAI-compatible API on port 8000, and waits until it
answers. **The first run takes about 5 minutes** — it pulls a container image, downloads ~16 GB of
weights, then compiles. It prints `still starting, N minutes elapsed` while it works, and `READY` when
it is done. Later starts reuse the cache and are quick.

**5. Confirm it is really answering.**

```bash
curl -s localhost:8000/v1/models
```

You want to see `"id":"Qwen/Qwen3-8B"`. Leave this terminal alone from here on.

### Terminal 2 — your work

**6. Get a shell inside the container**, where the model is on `localhost` and this repo is mounted at
`/workspace`:

```bash
docker exec -it vllm bash
```

> **Wait for the new prompt before typing anything else.** It takes a second or two, and anything you
> type in the meantime goes to your laptop's shell instead and is lost. This catches everyone once.

**7. Now, inside the container**, run the first project:

```bash
cd /workspace/projects/01-heat-rod-pde
export HEATROD_BASE_URL=http://localhost:8000/v1

python level0_heatrod.py --selftest     # prove the checker BEFORE you trust a score
python agent.py --level 0 --all         # the warm-up: solved on round 0
python agent.py --level 1 --all         # the real one
```

`sympy` and `numpy` are already in the container. If you are running outside it, `pip install sympy
numpy httpx` first.

### Useful while it runs

```bash
./serve.sh --logs     # follow the model's log
./serve.sh --stop     # stop and remove it
```

### Why a container, when the instance already has vLLM

The AMI ships several Neuron virtualenvs under `/opt`, and the newest is vLLM-Neuron **0.21**. That
version **cannot serve Qwen3-8B** — its model registry has only `LlamaForCausalLM`,
`GptOssForCausalLM`, `Eagle3LlamaForCausalLM` and `Qwen3VLForConditionalGeneration`, the last being the
*vision* model. Point it at Qwen3-8B and you get:

```
AttributeError: type object 'Qwen3ForCausalLM' has no attribute 'from_configs'
```

which is vLLM's generic class being used where the Neuron loader expected a Neuron one. `serve.sh` uses
the **0.24** container image instead, which does support it. If you would rather use a pre-installed
venv, `openai/gpt-oss-20b` works there and needs `--hf-overrides '{"quantization_config": {}}'` to load
in bf16.

Three settings in `serve.sh` are not optional, and all three were found by reading a failure:

* `NEURON_SKIP_EFA_AFFINITY=1` — without it the workers abort with `No EFA device found`. It skips a
  CPU-placement optimization that assumes networking hardware a single chip does not have.
* `--no-enable-prefix-caching` — prefix caching demands a segmented-prefill size of 512 or more.
* `--num-gpu-blocks-override` — required, or you get out-of-bounds errors. `serve.sh` computes it from
  the context length so it stays correct if you change that.

`--tensor-parallel-size 2` matches the hardware: one chip at LNC=2 is two logical NeuronCores.

`k8s/` holds the same configuration as Kubernetes manifests, which is how this was developed. You do
not need them on the instance.

## 2. Which model your project talks to

**A NeuronCore cannot be shared by two processes.** But a `trn2.3xlarge` has **four** logical cores,
and the model server only needs two — so the other two are yours. Confirm it with `neuron-top` while
the server is up:

```
[-] ND 0                      20.3GB
      NC 0                     0.0B      <- free
      NC 1                     0.0B      <- free
      [+] NC 2                10.0GB     <- vLLM
      [+] NC 3                10.3GB     <- vLLM
```

So nothing has to be remote. Three cases, in increasing order of what they need:

| your project needs | where the model comes from | cores it needs |
|---|---|---|
| no device at all — the checker runs on CPU | **the local Qwen3-8B** on `localhost:8000` | none |
| to compile and run kernels on the chip | **the local Qwen3-8B** | pin to the free cores with `NEURON_RT_VISIBLE_CORES=0,1` |
| the whole chip for serving experiments | **the shared `gpt-oss-20b` endpoint** ([`gptoss/`](gptoss/)), which lives on separate hardware | all four |

**Both sample projects fall in the first row today**, so both run against the model on your own
instance with nothing else to arrange:

* Project 1's checker is SymPy — pure CPU.
* Project 2's checker simulates kernels on the **CPU** (`nki.simulate`), which is deliberate: it is
  seconds per iteration instead of a compile, and it is where an agent should spend its attempts.

Only when you add real on-device timing — layers 2 and 3 of project 2's checker, which are not built
yet — do you need cores, and then you pin to the two the server is not using. `NEURON_RT_VISIBLE_CORES`
is the documented mechanism for that; we have not exercised it here, so expect to debug it.

## 3. The challenge

Build an **agent** around a model: it attempts a problem, something **checks** the attempt, and the
check plus the reason feed the next attempt. Repeat until it solves the problem, or until it can say
honestly that it could not.

The ingredient that makes or breaks this is the **checker**. A checker that says *"wrong, off by 341
percent"* is true and useless. A checker that says *"the term sin(pi\*x) should not be there at
all"* names the change to make. Project 1 records three separate occasions where the fix was not a
better model but a better error message — that is the lesson of the day.

**What to hand in**, whichever project you pick:

1. **Your checker**, and the reasoning behind what it accepts and rejects. This is the artifact we
   keep.
2. **Your attempt log** — every attempt with its score, so someone else can see the loop working.
3. **A one-page note**: what you ran, on what, what came out.

**You do not need to be a programmer to matter here.** Deciding what counts as correct, and turning
a verdict into an instruction, are the hard parts and they are not coding tasks. Teams of 3–5. Teams
that stack five programmers tend to lose these.

---

## 4. Sample projects

| # | project | status |
|---|---|---|
| **1** | [**The heat-rod agent**](projects/01-heat-rod-pde/) — a small model solves heat-equation problems under a checker that grades the physics, with a calculator it aims itself. **Worked end to end; the solution is included.** | **solved** |
| **2** | [**The kernel agent**](projects/02-kernel-agent/) — an agent writes small kernels for linear-algebra operations that run directly on the chip, and keeps verifying its own output as it goes. | runs; **unsolved** |

**Both project READMEs open with a real transcript** of what the loop prints when it runs, so you
can judge whether a project suits you — and tell progress from flailing — before you start anything.

Or **propose your own**. Two requirements: it runs on the hardware we give you, and it produces the
three deliverables above. Find an Annapurna engineer before 11:30 and we will tell you honestly
whether it fits in a day.

Adding a project is a folder under `projects/` with a README and a checker. That is the whole
pattern.

---

## 5. The shared `gpt-oss-20b` endpoint

A 20-billion-parameter open-weight model on Trainium2, behind an OpenAI-compatible API, on hardware
separate from your instance. Use it when your own chip is busy with your own code, or just to talk to
a model in the first five minutes.

```bash
export GPTOSS_BASE_URL="https://..."     # the organisers will give you this
pip install httpx
python gptoss/chat.py
```

It is shared by everyone in the room and it behaves in several ways that look like bugs and are not —
it thinks before it answers, sampling is greedy so retrying is pointless, and the `tools=` parameter
does nothing. **Read [`gptoss/README.md`](gptoss/README.md) before you build against it.** Every
number in that document was measured, and it will save you hours.

---

## Setup

```bash
pip install -r requirements.txt
```

## License / use

Sample code, provided as-is for the event, free to reuse.
[`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) and
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) are under their own licenses.
