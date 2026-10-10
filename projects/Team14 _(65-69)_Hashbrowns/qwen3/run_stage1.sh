#!/usr/bin/env bash
# run_stage1.sh -- stage 1 of the Qwen3-8B work, driven entirely from the laptop.
#
# Run it on the LAPTOP, in the terminal that has the AWS credentials -- not inside the pod:
#
#     bash qwen3/run_stage1.sh
#
# It brings RESULTS.md and the V0-V4 comparison back, pushes the two stage-1 scripts to the pod, makes
# sure vLLM is answering, runs the stack probe and the serving baseline inside the pod, and copies every
# result back into qwen3/. Every command it sends to the pod uses absolute paths, so nothing depends on
# which directory a shell happens to be in.
set -euo pipefail

POD=${POD:-seat-65}
P=/workspace/projects/03-attention-kernel
K=(kubectl exec -c app "$POD" --)

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is not here. This runs on your laptop, not inside the pod." >&2
  exit 1
fi
cd "$(dirname "$0")/.."                                 # the laptop's 03-attention-kernel folder
if ! "${K[@]}" true 2>/dev/null; then
  echo "Cannot reach $POD. Paste the newest workshop credentials into this terminal and retry." >&2
  exit 1
fi

echo "== 1/7  RESULTS.md and the V0-V4 comparison -> laptop"
kubectl cp -c app "$POD:$P/RESULTS.md" ./RESULTS.md >/dev/null 2>&1 && echo "  RESULTS.md" || echo "  no RESULTS.md on the pod"
mkdir -p results_pod
"${K[@]}" bash -c "cd $P/results && tar cf - --exclude='*.ntff' --exclude='*.neff' compare_*" \
  | tar xf - -C results_pod && echo "  results_pod/" || echo "  comparison not copied"

echo "== 2/7  stage-1 scripts -> pod"
"${K[@]}" mkdir -p "$P/qwen3"
for f in qwen3/bench_serving.py qwen3/probe_stack.py; do
  if kubectl cp -c app "$f" "$POD:$P/$f" >/dev/null 2>&1; then echo "  $f"
  else echo "  could not copy $f to the pod; stopping." >&2; exit 1; fi
done

echo "== 3/7  is vLLM answering?"
if "${K[@]}" python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" 2>/dev/null; then
  echo "  yes"
else
  echo "  no -- starting it with serve.sh (about 4 minutes)"
  "${K[@]}" bash -c /workspace/serve.sh
fi

echo "== 4/7  where Qwen3-8B's attention runs"
"${K[@]}" bash -c "python3 $P/qwen3/probe_stack.py > $P/qwen3/probe_stack.log 2>&1; tail -3 $P/qwen3/probe_stack.log"

echo "== 5/7  serving baseline, concurrency 1 (a few minutes)"
"${K[@]}" bash -c "python3 $P/qwen3/bench_serving.py --label stock > $P/qwen3/serving_stock_c1.log 2>&1; tail -12 $P/qwen3/serving_stock_c1.log"

echo "== 6/7  serving baseline, concurrency 4"
"${K[@]}" bash -c "python3 $P/qwen3/bench_serving.py --label stock --concurrency 4 --repeats 3 > $P/qwen3/serving_stock_c4.log 2>&1; tail -10 $P/qwen3/serving_stock_c4.log"

echo "== 7/7  results -> laptop"
for f in stack_src.tar.gz probe_stack.log serving_stock_c1.log serving_stock_c4.log; do
  kubectl cp -c app "$POD:$P/qwen3/$f" "qwen3/$f" >/dev/null 2>&1 && echo "  qwen3/$f" || echo "  qwen3/$f MISSING"
done
"${K[@]}" bash -c "cd $P/qwen3 && tar cf - results" | tar xf - -C qwen3 && echo "  qwen3/results/"

echo
echo "Done. Tell Claude the stage-1 results are in qwen3/."
