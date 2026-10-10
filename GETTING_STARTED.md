# Getting started for teammates (one page)

> Just copy and paste. Replace every `<N>` with **your own seat number** (the one you got at check-in, e.g. `42`).
> If something goes wrong, check the "Common errors" table at the end first, then ask in the group chat.

**Three hard rules**
1. **Paste the AWS credentials only into the terminal.** Never write them into a file, post them in the group chat, or commit them to git (the repo is public).
2. **Only enter your own seat.** Do not enter anyone else's `seat-xx`.
3. When a pod is replaced, everything inside it is lost. Pull back anything you want to keep to your own machine promptly (step 7).

---

## 1. Install the tools (once)

**Mac (with Homebrew):**
```bash
brew install awscli kubectl git
```

**Windows (PowerShell):**
```powershell
winget install -e --id Amazon.AWSCLI
winget install -e --id Kubernetes.kubectl
winget install -e --id Git.Git
```

When they are installed, **close the terminal and open a new one**, then check:
```bash
aws --version
kubectl version --client
```

## 2. Get the code (once)

```bash
git clone https://github.com/liuyq123/trainium-agent-labs.git
cd trainium-agent-labs
```

> If you only read logs, classify failures and write docs, you don't need Python. To run `kernelbench.py` on your own machine:
> `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt matplotlib`

## 3. Every new terminal: paste the credentials first

Copy **the block for your system** from the channel and paste it straight into the terminal, then press Enter. It looks like this (use the latest values from the channel):

```bash
# Mac / Linux
export AWS_ACCESS_KEY_ID="ASIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."
```
```powershell
# Windows PowerShell
$env:AWS_ACCESS_KEY_ID="ASIA..."
$env:AWS_SECRET_ACCESS_KEY="..."
$env:AWS_SESSION_TOKEN="..."
```

> **Every new window or tab needs the credentials pasted again.** They expire: when you see `ExpiredToken`, paste the latest block from the channel.

## 4. Connect to your seat

**Mac** (inside the `trainium-agent-labs` folder):
```bash
scripts/connect.sh <N>
```

**Windows:**
```powershell
aws eks update-kubeconfig --name hack-hyd --region ap-south-2
kubectl get pod seat-<N>
```

`READY 1/1` and `STATUS Running` mean it's fine. If it shows `Init` or `0/1`, wait a minute and try again; any other status, ask the organizers for another seat.

## 5. Enter the pod and start the model (terminal 1, keep it open)

```bash
kubectl exec -it seat-<N> -- bash
```

**Wait for the `root@seat-<N>:/workspace#` prompt before typing**, otherwise what you type goes to your local terminal. Then:

```bash
cd /workspace && ./serve.sh
```

After about 4 minutes `READY` appears and it's done. **Keep this terminal open and don't type in it again.**

## 6. Run an experiment (terminal 2)

Open a new terminal, **paste the credentials first (step 3)**, then:

```bash
kubectl exec -it seat-<N> -- bash
```

Inside:

```bash
git config --global --add safe.directory /workspace
cd /workspace/projects/02-kernel-agent
python nkibench.py --selftest
```

When the last line says `SELFTEST PASSED`, run **the experiment command the team lead sent you**. It usually looks like this (`<name>` is the name of this experiment):

```bash
nohup python agent.py --level 3 --rounds 8 --samples 4 --context 8192 --repeat 5 --log attempts_<name>.jsonl > run_<name>.log 2>&1 < /dev/null &
tail -f run_<name>.log
```

- `Ctrl+C` only **stops watching the log**; the experiment keeps running in the background.
- After a disconnect, enter the pod again and keep watching the log: `tail -f /workspace/projects/02-kernel-agent/run_<name>.log`
- To check whether the experiment is still running: `pgrep -af agent.py`
- A round takes about 50 seconds; 5 runs of one level take about 15–25 minutes.

## 7. Pull the results back to your machine

**Mac** (local terminal, inside the `trainium-agent-labs` folder):
```bash
scripts/sync.sh <N> pull
```
The results land in `runs/seat-<N>/<time>/`. Send that folder to the team lead, or tell them where to get it.

**Windows:**
```powershell
kubectl cp seat-<N>:/workspace/projects/02-kernel-agent/run_<name>.log run_<name>.log
kubectl cp seat-<N>:/workspace/projects/02-kernel-agent/attempts_<name>.jsonl attempts_<name>.jsonl
```

---

## Common errors

| what you see | cause | what to do |
|---|---|---|
| `Unable to locate credentials` / `NoCredentials` | the credentials weren't pasted in this terminal | do step 3 |
| `ExpiredToken` | the credentials expired | paste the **latest** block from the channel |
| `forbidden` | wrong seat number, or you entered someone else's seat | check `<N>` |
| `localhost:8080 ... refused` | kubectl isn't connected to the cluster yet | do step 4 |
| `connection reset by peer` / terminal frozen | the network dropped | redo step 3, then 5 or 6. Experiments started with `nohup` are not affected |
| `import nki` fails on your machine | `nki` is installed only in the pod | expected; run it in the pod |
| `port-forward ... forbidden` | we don't have that permission | expected; the agent can only run in the pod |

More background (scoring, what to hand in, known pitfalls): [`NOTES.md`](NOTES.md).
