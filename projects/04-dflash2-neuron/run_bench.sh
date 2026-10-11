#!/usr/bin/env bash
# run_bench.sh <num_draft_tokens|0 for baseline> — run bench_dflash2.py and save results/<file>.txt
set -euo pipefail
K=${1:?usage: run_bench.sh <num_draft_tokens, 0 = baseline>}
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/results"
if [ "$K" = 0 ]; then
    OUT="$HERE/results/bench_baseline.txt"
    SERVER="# server: bash serve_baseline.sh  (Qwen/Qwen3-8B, no speculative decoding, same config as serve_dflash2.sh)"
else
    OUT="$HERE/results/bench_dflash2_k${K}.txt"
    SERVER="# server: DFLASH_NUM_SPEC=${K} bash serve_dflash2.sh  (Qwen/Qwen3-8B + z-lab/Qwen3-8B-DFlash-b16, ${K} draft tokens)"
fi
{
    echo "# python3 bench_dflash2.py --max-tokens 128 --runs 2"
    echo "$SERVER"
    echo "# $(hostname), trn2 (1 chip, TP=2), $(date -u '+%Y-%m-%d %H:%M') UTC"
    python3 "$HERE/bench_dflash2.py" --max-tokens 128 --runs 2
} | tee "$OUT"
echo "saved: $OUT"
