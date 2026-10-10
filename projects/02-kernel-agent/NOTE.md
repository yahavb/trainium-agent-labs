# &lt;Team&gt; — Kernel agent on Trainium2: one page

**What we ran:** Qwen3-8B (vLLM-Neuron 0.24, TP=2, ctx 8192) on one Trainium2 chip per seat; NKI
0.6.0 simulator checker (+ compile gate / on-device for N kernels). Also gpt-oss-20b (shared,
greedy) for comparison.

**Baseline (our pods, n=…):** L1 0/n, L2 k/n [CI], L3 0/n, L4 0/n (0.62).  
*(fill from `analyze.py logs/base_*.jsonl`)*

**What we changed (each measured, n=5, all levels, so a regression elsewhere is visible):**

| change (flag) | L1 | L2 | L3 | L4 | kept? |
|---|---|---|---|---|---|
| level-3 reshape feedback | | | | | |
| `--tools doc` | | | | | |
| `--tools probe` | | | | | |
| `--tiles` | | | | | |
| `--skills` | | | | | |
| `--beam 2` | | | | | |

**Final config (n=5):** `<command>`  
**Held-out ops (never tuned on, n=3):** L10 ragged matmul, L11 relu-affine, L12 softmax (hostile),
L13 RMSNorm (hostile). `<rates>`

**Spread:** best / worst / mean per level.  
**Simulator vs device:** which numbers are which (calibration labels from `--calibration`).

**Failure taxonomy:** top 5 classes with counts (from `analyze.py` → `taxonomy.csv`); which
interventions killed which class.

**What didn't work (and why we think so):** `<e.g. the reverted tiling example pattern, if seen>`

**Reproduce:**
```bash
git clone … && cd projects/02-kernel-agent
python nkibench.py --selftest
python mutants.py --rules-only
./run_exp.sh final --all --rounds 8 --samples 4 --context 8192 --repeat 5
python analyze.py logs/final.jsonl
```
