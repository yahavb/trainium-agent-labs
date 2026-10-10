#!/usr/bin/env bash
# Runs inside the seat pod. Starts detached HF downloads into the default (shared,
# persistent) HF cache and clones StreamDiffusionV2 source. Returns immediately.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/logs"
mkdir -p "$LOGS" "$ROOT/repos"
HF_CACHE="${HF_HOME:-/root/.cache/huggingface}"
mkdir -p "$HF_CACHE/hub"
MIN_FREE_GB=60
MODELS=(stabilityai/sd-turbo Wan-AI/Wan2.1-T2V-1.3B jerryfeng/StreamDiffusionV2 madebyollin/taesd)

free_gb="$(df -BG --output=avail "$HF_CACHE" | tail -1 | tr -dc '0-9')"
echo "free on $HF_CACHE: ${free_gb} GB"
if (( free_gb < MIN_FREE_GB )); then
  echo "error: under ${MIN_FREE_GB} GB free on $HF_CACHE; not starting downloads" >&2
  exit 1
fi

if ! python3 -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "huggingface_hub missing; installing"
  pip install -q huggingface_hub
fi
if command -v hf >/dev/null 2>&1; then
  DL=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  DL=(huggingface-cli download)
else
  echo "error: neither hf nor huggingface-cli on PATH" >&2
  exit 1
fi
echo "downloader: ${DL[*]}"

is_cached() {
  local dir="$HF_CACHE/hub/models--${1//\//--}"
  [[ -d "$dir/snapshots" ]] && [[ -n "$(ls -A "$dir/snapshots" 2>/dev/null)" ]] \
    && ! find "$dir/blobs" -name '*.incomplete' -print -quit 2>/dev/null | grep -q .
}

: > "$LOGS/download_pids.txt"
for repo in "${MODELS[@]}"; do
  name="$(basename "$repo" | tr '[:upper:]' '[:lower:]')"
  if is_cached "$repo"; then
    echo "skip $repo (already complete in cache)"
    continue
  fi
  nohup setsid "${DL[@]}" "$repo" > "$LOGS/$name.log" 2>&1 < /dev/null &
  echo "$repo $!" | tee -a "$LOGS/download_pids.txt"
done

if [[ -d "$ROOT/repos/StreamDiffusionV2/.git" ]]; then
  echo "skip clone (repos/StreamDiffusionV2 exists)"
else
  nohup setsid git clone https://github.com/chenfengxu714/StreamDiffusionV2 "$ROOT/repos/StreamDiffusionV2" \
    > "$LOGS/clone_streamdiffusionv2.log" 2>&1 < /dev/null &
  echo "git-clone $!" | tee -a "$LOGS/download_pids.txt"
fi
