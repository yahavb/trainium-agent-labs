# REPRODUCE — from a fresh seat pod

Everything here is **[sim]** (nki 0.6 CPU simulator) unless a line says **[device]**. Numbers come from
`--repeat`/multi-run rates, never single runs.

## 0. Get the code into the pod

```bash
kubectl exec -it seat-NN -- bash          # wait for the root@seat-NN prompt
cd /workspace && git config --global --add safe.directory /workspace
git clone https://github.com/qz2930-crypto/hack-the-chip-seat73.git team && cd team/projects/02-kernel-agent
```

(From a laptop without pod git access: `scripts/pod.sh sync` tars `projects/02-kernel-agent/` into the pod.)

## 1. Start the model (Qwen3-8B, vLLM-Neuron 0.24, TP=2, context 8192)

```bash
cd /workspace && ./serve.sh                # prints READY after ~4 min
```

**Never send `seed` in a request to this server.** It kills the engine (`getNewGenerator … PrivateUse1`,
LOG.md 11:51). The server decodes deterministically: `temperature` does not vary the output, so
`--samples 1` costs a quarter of the time and loses nothing.

## 2. Prove the harnesses before trusting a score (no model calls)

```bash
python nkibench.py --selftest              # SELFTEST PASSED
python verdicts.py                         # VERDICTS SELFTEST PASSED  (verdict -> instruction table)
python levers.py                           # LEVERS SELFTEST PASSED    (mech fixes, skeletons, doc slices)
python kernelbench.py --selftest           # Stage-A NumPy harness: every planted bug caught
for lv in 4 5 6 7; do python nkibench.py --level $lv --check reference_level$lv.py; done   # 4/4 each
python heldout.py --refs                   # held-out hostile cases + calibration on the references
```

## 3. Baseline (upstream agent behaviour)

```bash
for L in 1 2 3 4; do
  nohup python -u agent.py --tag base_L$L --log runs/base_L$L.jsonl --level $L \
        --rounds 8 --samples 1 --context 8192 --repeat 5 > runs/base_L$L.log 2>&1 < /dev/null &
done
```

## 4. The lever experiments (each vs baseline), via the queue runner

```bash
# one line per single run: <tag> <agent args>; runq keeps 4 agent processes alive
for e in "skel --skeleton" "retr --retrieve" "mech --mech"; do set -- $e
  for r in 0 1 2 3 4; do for L in 1 3 4 2; do
    echo "${1}_L${L}_r${r} --level $L --rounds 8 --samples 1 --context 8192 --repeat 1 $2"
  done; done; done > runs/queue.txt
setsid nohup python runq.py > runs/runq.log 2>&1 < /dev/null &
```

## 4b. The final agent (C10) and Stage A, as reported

```bash
F="--samples 1 --context 8192 --repeat 1 --skeleton --v2-verdicts --v3-verdicts --v4-verdicts --v5-verdicts    --v6-verdicts --v7-verdicts --v8-verdicts --v9-verdicts --prompt-fixes --device-rules --directional"
for L in 1 2 3 4 5 6 7; do python -u agent.py --level $L --rounds 8 --tag c10_L$L --log runs/c10_L$L.jsonl $F; done
python -u agent.py --level 8 --rounds 12 --tag c10_L8 --log runs/c10_L8.jsonl $F
python -u stagea.py --endpoint url --base http://localhost:8000/v1 --model Qwen/Qwen3-8B --levels 1-10        --rounds 6 --no-cache --max-tokens 2500 --v2 --v3 --v4-smart --tag A6_qwen
python regrade.py --v2 --v3 --v4 --v5 --v6 --v7 --v8 --v9 --directional --device-rules --prompt-fixes 'runs/c10_*.jsonl'
NEURON_RT_VISIBLE_CORES=2 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python jit_device.py agent_solutions/c1_L4_r0.py:nki_matmul_tiled_   # [device]
```

## 5. Turn logs into the deliverables (laptop or pod)

```bash
scripts/refresh_deliverables.sh <label>     # all of the below in one go, snapshot + commit
python scripts/results.py  runs/live/*.jsonl                      # RESULTS.md cells: solved k/n, mean, all=[...]
python scripts/taxonomy.py "runs/live/*.jsonl" > FAILURES.md
python scripts/attempt_log.py "runs/live/*.jsonl"                 # ATTEMPTS.md + attempts_all.csv          # failure taxonomy with counts
python scripts/token_report.py runs/live/*.jsonl > TOKENS.md        # input tokens per section
python scripts/token_report.py --png token_budget.png --tag <tag> runs/live/*.jsonl
cd projects/02-kernel-agent && python heldout.py --refs --logs '../../runs/live/*.jsonl' > ../../CALIBRATION.md
```

`scripts/pod.sh pullall` mirrors the pod's `runs/` into `runs/live/`; snapshots are copied to
`runs/<HHMM>_<label>/` at each phase boundary.
