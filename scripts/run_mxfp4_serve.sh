#!/usr/bin/env bash
# Serve GPT-OSS 20B with in-repo MXFP4 kernels using the local Python code.
# Usage: ./scripts/run_mxfp4_serve.sh [additional serve args...]

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Enter repo root
cd "$repo_root"

# Activate env for Mojo imports and Python path.
source scripts/mxfp4_env.sh

# Ensure we use the repo's Python packages.
export PYTHONPATH="$repo_root/max/python:$repo_root/bazel-bin/max/python"

# Allow MXFP4 even if upstream guards are present.
export MAX_ALLOW_UNSUPPORTED_ENCODING=1
export MAX_SKIP_MEMORY_CHECK="${MAX_SKIP_MEMORY_CHECK:-1}"

if [[ -z "${MAX_CUSTOM_EXTENSIONS:-}" ]]; then
  export MAX_CUSTOM_EXTENSIONS="$repo_root/max/kernels/src/Mogg/MXFP4Extension"
fi

python -m max.entrypoints.pipelines serve \
  --model openai/gpt-oss-20b \
  --quantization-encoding mxfp4 \
  --devices gpu \
  --force \
  "$@"
