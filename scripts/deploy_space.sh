#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
space_url="$(git -C "$repo_root" remote get-url space)"
source_commit="$(git -C "$repo_root" rev-parse --short main)"
temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/fake-news-space.XXXXXX")"
space_checkout="$temp_dir/repo"

cleanup() {
  rm -rf "$temp_dir"
}
trap cleanup EXIT

if ! git xet --version >/dev/null 2>&1; then
  echo "git-xet is required for the tracked checkpoint binaries." >&2
  exit 1
fi

# Clone the existing Space history, then replace its tree with tracked files from main.
git clone --quiet "$space_url" "$space_checkout"
git -C "$space_checkout" remote rename origin space
git -C "$repo_root" archive main | tar -x -C "$space_checkout"
cp "$repo_root/space/README.md" "$space_checkout/README.md"

git -C "$space_checkout" add -A

# These files must be Xet/LFS pointers in the Space commit, not raw binary blobs.
for checkpoint in \
  models/roberta-trained-welfake/adapter_model.safetensors \
  models/roberta-trained-welfake/classifier_head.pt
do
  index_size="$(git -C "$space_checkout" cat-file -s ":$checkpoint")"
  if (( index_size > 1024 )); then
    echo "Checkpoint pointer conversion failed: $checkpoint" >&2
    exit 1
  fi
done

if git -C "$space_checkout" diff --cached --quiet; then
  echo "Space already matches source commit $source_commit."
  exit 0
fi

git -C "$space_checkout" commit --quiet \
  -m "Deploy source commit $source_commit"
git -C "$space_checkout" push space main

echo "Deployed source commit $source_commit to the Space."
