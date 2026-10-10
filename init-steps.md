# init-steps.md: get a loop running

Two paths. **A** needs nothing and proves the loop on your laptop in two minutes. **B** is the real
thing, on your seat's chip. Do A first if you are waiting for your seat number or credentials.

Every command below comes from `README.md` Part 1 and the project READMEs.

---

## Path A: no seat, no chip (laptop only)

Runs the whole loop (controller, checker, feedback, rounds) against a **canned** model. Good for
reading the code and seeing what a round prints. **Never report a number from it.**

```bash
git clone https://github.com/buddhsen-tripathi/trainium-agent-labs.git
cd trainium-agent-labs
pip install -r requirements.txt            # httpx, numpy, sympy

cd projects/01-heat-rod-pde
python level0_heatrod.py --selftest        # prove the checker before trusting a score
python agent.py --offline --level 1 --all  # the loop, canned generator
```

For the kernel side, the NumPy harness also runs on a laptop with no accelerator:

```bash
cd ../02-kernel-agent
python kernelbench.py --selftest           # proves it catches planted bugs
python kernelbench.py --list               # the ten-level ladder
python kernelbench.py --level 1 --show
```

---

## Path B: the real loop on your seat

### You need first

| have | where from |
|---|---|
| a **seat number** (`seat-N`) | check-in sheet / sticky note |
| the **workshop credentials** (3 AWS values) | the pinned post in the workshop channel |
| `aws` and `kubectl` installed | step 1 below |

### On your laptop

```bash
# 1. install (macOS shown; Linux/Windows are in README Part 1)
brew install awscli kubectl
aws --version && kubectl version --client      # open a NEW terminal first if "command not found"

# 2. paste the credentials block for YOUR shell, into THIS terminal
export AWS_ACCESS_KEY_ID="ASIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."

# 3. point kubectl at the cluster
aws eks update-kubeconfig --name hack-hyd --region ap-south-2

# 4. find your pod: want READY 1/1 and STATUS Running
kubectl get pod seat-N

# 5. shell inside it, and WAIT for the prompt root@seat-N:/workspace# before typing
kubectl exec -it seat-N -- bash
```

### Inside the pod, terminal 1: start the model

```bash
neuron-ls              # expect a NEURON DEVICE
cd /workspace
./serve.sh             # ~4 min the first time; prints READY
```

Leave this terminal open. (`./serve.sh --logs` follows the log, `./serve.sh --stop` stops it.)

### Terminal 2: run the loop

Open a second terminal, **paste the credentials again** (step 2), then:

```bash
kubectl exec -it seat-N -- bash         # same pod, wait for the prompt
git config --global --add safe.directory /workspace

cd /workspace/projects/01-heat-rod-pde
python level0_heatrod.py --selftest             # prove the checker first
python agent.py --level 0 --all                 # warm-up: solved on round 0

# the real one, in the background so a dropped connection can't kill it
nohup python agent.py --level 1 --all > run.log 2>&1 < /dev/null &
tail -f run.log                                  # Ctrl+C stops watching; the run keeps going
```

The pod already sets `HEATROD_BASE_URL` and the model name, so there is nothing to export.

### Project 2 (the unsolved one)

```bash
cd /workspace/projects/02-kernel-agent
python nkibench.py --selftest
nohup python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5 > run.log 2>&1 < /dev/null &
tail -f run.log
```

---

## How to know it is working

| you see | it means |
|---|---|
| `SOLVED: u(x, t) = ...` | heat-rod level done; the loop works |
| rewards differ within a round, e.g. `[0.0, 1.0, 0.4, 0.0]` | sampling is on, so there is something to choose between |
| `0.0` and rounds near 55 s | answer truncated; check for the TRUNCATED notice |
| `0.1` / `0.3` (project 2) | parses but breaks a rule / runs but numbers are wrong |
| the **same** error three rounds running | your feedback is a verdict, not an instruction. Fix the message, not the prompt. |

Every attempt is appended to `attempts.jsonl` (prompt/answer/reward). That file is deliverable 2.

---

## When something goes wrong

| symptom | fix |
|---|---|
| `ExpiredToken` / asks for credentials | paste the newest credentials block; your pod and work are untouched |
| `command not found: aws` | close the terminal and open a new one |
| pod `Init` or `0/1` | still starting, wait a minute |
| pod `Pending` / `CrashLoopBackOff` / `Error` | find an organiser; they move you to a spare seat |
| typed something and it ran on your laptop | you typed before the `root@seat-N` prompt appeared |
| `kubectl exec` dropped mid-run | reconnect; `tail -f run.log`; `pgrep -af agent.py` shows if it is alive |
| git says "dubious ownership" | `git config --global --add safe.directory /workspace` |
| no code comes back, rounds very slow | thinking mode is on or the prompt is too long; shorten the prompt, don't raise `max_tokens` |

**Rules of the room:** your seat is yours, so don't exec into anyone else's. If the pod is replaced,
`/workspace` is gone, so `git push` anything you want to keep to your own repo.
