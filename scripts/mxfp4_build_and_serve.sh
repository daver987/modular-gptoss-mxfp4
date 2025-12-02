#!/usr/bin/env bash
# Build the MXFP4 kernel package and serve GPT-OSS with it loaded automatically.
# Usage: pixi run ./scripts/mxfp4_build_and_serve.sh [extra serve args...]

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$repo_root"

echo "Building MXFP4 package (opt)..."
./bazelw build -c opt //max/kernels/src/Mogg/MXFP4Extension:MXFP4Extension

# Default to the source package so the runtime can build/load it automatically.
if [[ -z "${MAX_CUSTOM_EXTENSIONS:-}" ]]; then
  export MAX_CUSTOM_EXTENSIONS="$repo_root/max/kernels/src/Mogg/MXFP4Extension"
fi

# Prepare environment (Mojo paths + PYTHONPATH).
source scripts/mxfp4_env.sh
export PYTHONPATH="$repo_root/max/python:$repo_root/bazel-bin/max/python"
export MAX_ALLOW_UNSUPPORTED_ENCODING=1
export MAX_SKIP_MEMORY_CHECK="${MAX_SKIP_MEMORY_CHECK:-1}"

echo "Serving with MAX_CUSTOM_EXTENSIONS=$MAX_CUSTOM_EXTENSIONS"
python -m max.entrypoints.pipelines serve \
  --model openai/gpt-oss-20b \
  --quantization-encoding mxfp4 \
  --devices gpu \
  --force \
  "$@"
