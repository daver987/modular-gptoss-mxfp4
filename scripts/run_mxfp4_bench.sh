#!/usr/bin/env bash
# Run the MXFP4 grouped matmul microbenchmark with sane defaults.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
CUSTOM_EXT="${MAX_CUSTOM_EXTENSIONS:-${MXFP4_KERNEL_PACKAGE:-}}"
if [[ -z "${CUSTOM_EXT}" ]]; then
  # Try MOGGKernelAPI first (built-in MXFP4 kernels)
  DEFAULT_EXT="${REPO_ROOT}/bazel-bin/max/kernels/src/Mogg/MOGGKernelAPI/MOGGKernelAPI.mojopkg"
  if [[ -f "${DEFAULT_EXT}" ]]; then
    CUSTOM_EXT="${DEFAULT_EXT}"
  else
    echo "No custom extension provided. Set MAX_CUSTOM_EXTENSIONS or MXFP4_KERNEL_PACKAGE." >&2
    echo "Build the kernels first with: ./bazelw build //max/kernels/src/Mogg/MOGGKernelAPI:MOGGKernelAPI" >&2
    exit 1
  fi
fi

export MAX_CUSTOM_EXTENSIONS="${CUSTOM_EXT}"
export PYTHONPATH="${REPO_ROOT}/max/python:${REPO_ROOT}/bazel-bin/max/python:${PYTHONPATH:-}"

ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
  ARGS=(
    --iters 10
    --warmup 2
    --num-experts 32
    --tokens-per-expert 4
    --in-features 2880
    --out-features 5760
    --device gpu
  )
fi

echo "Using MAX_CUSTOM_EXTENSIONS=${MAX_CUSTOM_EXTENSIONS}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "Args: ${ARGS[*]}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/bench_mxfp4_matmul.py" "${ARGS[@]}"
