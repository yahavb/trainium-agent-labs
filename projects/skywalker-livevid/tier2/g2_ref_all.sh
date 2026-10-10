#!/usr/bin/env bash
# Runs inside the seat-234 pod from tier2/: setup, T5 once (skipped if the embedding exists), then the CPU reference.
set -euo pipefail
./g2_setup.sh
[[ -s /workspace/livevid/artifacts/tier2/g2/prompt_anime.pt ]] || python -u g2_ref.py t5
python -u g2_ref.py run
