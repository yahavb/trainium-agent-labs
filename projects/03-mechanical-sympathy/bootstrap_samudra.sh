#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pin_file="$project_root/samudra-source.json"
destination="${1:-$project_root/.scratch/Samudra}"

if [[ ! -f "$pin_file" ]]; then
  echo "Source pin not found: $pin_file" >&2
  exit 2
fi

repository="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repository"])' "$pin_file")"
upstream_repository="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["upstream_repository"])' "$pin_file")"
commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$pin_file")"

if [[ -d "$destination/.git" ]]; then
  if [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
    echo "Refusing to change a Samudra checkout with local changes: $destination" >&2
    exit 2
  fi
else
  mkdir -p "$(dirname "$destination")"
  git clone --no-checkout "$repository" "$destination"
fi

git -C "$destination" remote set-url origin "$repository"
if git -C "$destination" remote get-url upstream >/dev/null 2>&1; then
  git -C "$destination" remote set-url upstream "$upstream_repository"
else
  git -C "$destination" remote add upstream "$upstream_repository"
fi

git -C "$destination" fetch --depth=1 origin "$commit"
git -C "$destination" checkout --detach "$commit"

actual_commit="$(git -C "$destination" rev-parse HEAD)"
if [[ "$actual_commit" != "$commit" ]]; then
  echo "Samudra checkout mismatch: expected $commit, found $actual_commit" >&2
  exit 1
fi

echo "Samudra is ready at $destination"
echo "Pinned commit: $actual_commit"
