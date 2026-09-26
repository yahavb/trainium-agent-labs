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

Three commands from a fresh instance to a model answering on `localhost`:

```bash
git clone https://github.com/yahavb/nyu-gptoss-kit.git && cd nyu-gptoss-kit
./install-docker.sh
./serve.sh
```

`serve.sh` starts **Qwen3-8B** on your chip in a container, behind an OpenAI-compatible API on port
8000, and waits until it answers. The first run downloads the weights and compiles the model —
measured at **102 seconds of compile, answering about 4 minutes after start**. Later runs reuse the
cache and are quick.

```bash
./serve.sh --logs      # follow it
./serve.sh --stop      # stop it
curl -s localhost:8000/v1/models
```

### Then run a project inside that container

The model is on `localhost` in there, and this repo is mounted at `/workspace`:

```bash
docker exec -it vllm bash
cd /workspace/projects/01-heat-rod-pde
pip install sympy
export HEATROD_BASE_URL=http://localhost:8000/v1
python agent.py --level 1 --all
```

### Three settings that are not optional

All three were found by reading a failure, and two of them stop the server dead:

* `NEURON_SKIP_EFA_AFFINITY=1` — without it the workers abort with `No EFA device found`. It skips a
  CPU-placement optimization that assumes networking hardware a single chip does not have.
* `--no-enable-prefix-caching` — prefix caching demands a segmented-prefill size of 512 or more.
* `--num-gpu-blocks-override` — required, or you get out-of-bounds errors. `serve.sh` computes it
  from the context length, so it stays correct if you change that.

`--tensor-parallel-size 2` matches the hardware: one chip at LNC=2 is two logical NeuronCores.

`k8s/` holds the same configuration as Kubernetes manifests, which is how this was developed. You do
not need them on the instance.

---

## 2. One chip, one process — the constraint that shapes your project

**A Neuron device cannot be shared by two processes.** If vLLM is holding the chip and serving a
model, nothing else can run on it; if your own code needs the chip, vLLM cannot be running.

That single fact decides where your model comes from:

| your project needs the chip for | the model you talk to |
|---|---|
| nothing — the chip only serves the model | **Qwen3-8B on your own instance** (`k8s/qwen3-8b-vllm.yaml`) |
| running your own kernels or measuring hardware | **the shared `gpt-oss-20b` endpoint** ([`gptoss/`](gptoss/)) — it lives on separate hardware, so your chip stays free |

Decide this on day one. It is not a detail, and discovering it at 3 PM costs an afternoon.

---

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
| **1** | [**The heat-rod agent**](projects/01-heat-rod-pde/) — a small model solves heat-equation problems under a checker that grades the physics, with a calculator it aims itself. **Worked end to end; the solution is included.** | ready |
| **2** | [**The kernel agent**](projects/02-kernel-agent/) — an agent writes small kernels for linear-algebra operations that run directly on the chip, and keeps verifying its own output as it goes. | **not finalised** |

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
