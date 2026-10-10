#!/usr/bin/env bash
# Runs inside a seat pod. Idempotently builds the classic torch-neuronx venv at
# /workspace/venvs/tnx without touching system python. Safe to re-run after a pod replacement.
set -euo pipefail

VENV="${VENV:-/workspace/venvs/tnx}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/logs" "$(dirname "$VENV")"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv-cache}"

IDX=(--extra-index-url https://pip.repos.neuron.amazonaws.com --index-strategy unsafe-best-match)
PKGS=("neuronx-cc==2.*" torch-neuronx torchvision diffusers transformers accelerate safetensors
      pillow opencv-python-headless numpy
      # neuronx-cc 2.27 fails with NCC_ISMP902 "is_subset(): incompatible function arguments" on islpy 2026.2.x
      "islpy==2026.1")

if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo ">> uv $(uv --version)"

venv_py_version() { "$VENV/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true; }

install_core() {
  uv pip install --python "$VENV/bin/python" "${IDX[@]}" "${PKGS[@]}"
}

PY_ORDER=(3.11 3.10)
[[ "$(venv_py_version)" == "3.10" ]] && PY_ORDER=(3.10)

ok=0
for py in "${PY_ORDER[@]}"; do
  if [[ "$(venv_py_version)" != "$py" ]]; then
    echo ">> creating $VENV with python $py"
    uv venv --clear --python "$py" "$VENV"
  fi
  echo ">> installing core packages into python $py venv"
  if install_core; then ok=1; break; fi
  echo ">> install failed with python $py"
done
[[ $ok -eq 1 ]] || { echo "ERROR: core install failed for python 3.11 and 3.10" >&2; exit 1; }

# torch_xla's _XLAC links libpython dynamically, which uv's standalone python keeps outside the loader path.
PY_LIBDIR="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
cat > "$VENV/neuron_env.sh" <<EOF
export PATH="$VENV/bin:\$PATH"
export LD_LIBRARY_PATH="$PY_LIBDIR:/opt/aws/neuron/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
grep -q neuron_env.sh "$VENV/bin/activate" || echo "source $VENV/neuron_env.sh" >> "$VENV/bin/activate"
source "$VENV/neuron_env.sh"
echo ">> wrote $VENV/neuron_env.sh (source it before using $VENV/bin/python)"

echo ">> optimum-neuron[neuronx] dry run"
DRY="$ROOT/logs/optimum_neuron_dryrun.txt"
set +e
uv pip install --python "$VENV/bin/python" "${IDX[@]}" --dry-run "optimum-neuron[neuronx]" > "$DRY" 2>&1
dry_rc=$?
set -e
cat "$DRY"
conflicts="$(grep -E '^\s*[-+~] (torch|torch-neuronx|neuronx-cc|torchvision)==' "$DRY" || true)"
if [[ $dry_rc -ne 0 ]]; then
  echo ">> optimum-neuron NOT installed: dry run failed (rc=$dry_rc), see $DRY"
elif [[ -n "$conflicts" ]]; then
  echo ">> optimum-neuron NOT installed: it would change pinned packages:"
  echo "$conflicts"
else
  echo ">> optimum-neuron leaves torch/torch-neuronx/neuronx-cc unchanged; installing"
  uv pip install --python "$VENV/bin/python" "${IDX[@]}" "optimum-neuron[neuronx]"
fi

echo ">> versions"
"$VENV/bin/python" - <<'EOF'
import sys
from importlib.metadata import version, PackageNotFoundError
print("python", sys.version.split()[0])
for p in ["torch", "torch-neuronx", "neuronx-cc", "torchvision", "diffusers", "transformers",
          "accelerate", "optimum-neuron", "numpy"]:
    try:
        print(f"{p} {version(p)}")
    except PackageNotFoundError:
        print(f"{p} <not installed>")
import torch, torch_neuronx
print("import torch_neuronx ok:", torch_neuronx.__version__)
EOF
echo ">> setup_venv done"
