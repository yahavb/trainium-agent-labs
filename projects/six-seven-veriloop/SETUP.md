# Setup: get on a seat, start the model, install the simulator

Every participant got a **seat number** at check-in. Your seat is a small server (a Kubernetes "pod")
called `seat-<your number>` with **its own Trainium chip** and the Qwen3-8B model. You reach it from your
own laptop. **Use only your own seat** — the organisers' one rule is: don't go into anyone else's.

You need a seat to run the model. The checker, the self-test and the analysis scripts run on any laptop
with Icarus Verilog.

## First time — about 10 minutes

### 1. Install the two tools (on your laptop, once)

**Mac:**
```bash
brew install awscli kubectl
```
**Windows (PowerShell):**
```powershell
winget install -e --id Amazon.AWSCLI
winget install -e --id Kubernetes.kubectl
```
Then **close the terminal and open a new one**. Check: `aws --version` and `kubectl version --client`.

### 2. Paste the workshop credentials

The organisers posted a block in the **workshop channel** with three versions. Paste the one for your
terminal (Mac/Linux: the `export AWS_...` lines; Windows: the `$env:AWS_...` lines for PowerShell) into
the terminal you will use, and press Enter.

> These work **only in that terminal window**. A new window or tab needs them pasted again. They also
> **expire**: if you later see `ExpiredToken`, paste the newest block from the channel.
> **Never put them in the repo, a file, or a chat message.** They are shared by the whole event.

### 3. Connect to the cluster
```bash
aws eks update-kubeconfig --name hack-hyd --region ap-south-2
```
It should print `Added new context ...`. An access error means step 2 is missing in this window.

### 4. Check your seat
```bash
kubectl get pod seat-<N>
```
You want `READY 1/1` and `STATUS Running`. `Pending` / `CrashLoopBackOff` / `Error` → find an organiser
for a spare seat.

### 5. Open a shell on your seat
```bash
kubectl exec -it seat-<N> -c app -- bash
```
**Wait for the prompt `root@seat-<N>:/workspace#` before typing anything.** Anything typed earlier goes to
your laptop instead.

### 6. Start the model (once per seat)
```bash
cd /workspace && ./serve.sh
```
About 4 minutes; it prints `READY`. **Leave this window open.** Open a **second terminal**, paste the
credentials again (step 2), and run step 5 again — that second shell is where you work.

Check the model answers (in the second shell):
```bash
curl -s localhost:8000/v1/models | grep -o 'Qwen/Qwen3-8B'
```

---

## Every time after that

New terminal → paste credentials (step 2) → `kubectl exec -it seat-<N> -c app -- bash` (step 5).
If the model was stopped, run step 6 again.

## Install the simulator (once per seat, needed for VeriLoop)

Inside your seat:
```bash
apt-get update && apt-get install -y iverilog
iverilog -V | head -1
```

## Getting our code onto your seat, and results back

Inside a seat this repo is already at `/workspace`: run from `/workspace/projects/six-seven-veriloop`.
Or work on your laptop and copy files over.
Run these **on your laptop**, from the repo folder:

```bash
# laptop → seat: copy the whole veriloop folder
kubectl cp projects/six-seven-veriloop/veriloop seat-<N>:/workspace/veriloop -c app

# seat → laptop: bring a result file back into results/
kubectl cp seat-<N>:/workspace/veriloop/results/run.jsonl results/2026-10-10_<you>_<what>.jsonl -c app
```
Then commit and push the result from your laptop as usual.

## Running something long (an experiment)

A `kubectl exec` window can drop (Wi-Fi, laptop sleep). Anything running in the foreground dies with it.
Start long runs in the background, writing to a log:

```bash
cd /workspace/veriloop
nohup python run_experiment.py ARGS > run.log 2>&1 < /dev/null &
tail -f run.log          # Ctrl+C stops watching; the run keeps going
```
Reconnected later? `tail -f /workspace/veriloop/run.log`. Still running? `pgrep -af run_experiment`.

## Talking to the model from code

Inside your seat the model is at **`http://localhost:8000/v1`** (OpenAI-compatible API), model name
**`Qwen/Qwen3-8B`**. Keep its "thinking" mode **off** and prompts short — the organisers measured that
thinking made it 55× slower and score zero.

**Two things measured on our seats:**
- **Never send `seed` in a request.** On this server it crashed the whole model (seat 7). If the model
  stops answering, restart it: `cd /workspace && ./serve.sh`.
- At `temperature` 0.7 the model gave the same answer 4 times out of 4; at 1.0 (top_p 0.95) the
  answers differ. `veriloop/agent.py` already uses 1.0.

## When something goes wrong

| you see | do this |
|---|---|
| `ExpiredToken`, or it asks for credentials | paste the newest credentials block from the channel |
| `error: You must be logged in` / access denied | step 2 is missing in this window — paste the credentials |
| `connection reset by peer`, window froze | reconnect with step 5; background runs keep going |
| `Defaulted container "app"` | harmless |
| the model does not answer on `localhost:8000` | run `./serve.sh` again in `/workspace`; `./serve.sh --logs` shows its log |
| the pod was replaced / your files are gone | everything in the seat is lost when a pod is replaced — that is why the code lives in the repo; copy it over again |
| anything else | find an organiser |
