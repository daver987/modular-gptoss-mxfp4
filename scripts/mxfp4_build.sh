#!/usr/bin/env bash
# Builds the MXFP4 Mojo kernels (built-in via MOGGKernelAPI).
#
# The MXFP4 kernels are now registered directly in MOGGKernelAPI.mojo, so we
# only need to build that package. No separate custom_ops package is required.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "Building MOGGKernelAPI (includes MXFP4 kernels)..."
./bazelw build //max/kernels/src/Mogg/MOGGKernelAPI:MOGGKernelAPI

builtin_pkg="$repo_root/bazel-bin/max/kernels/src/Mogg/MOGGKernelAPI/MOGGKernelAPI.mojopkg"
if [[ -f "$builtin_pkg" ]]; then
  echo
  echo "MOGGKernelAPI package: $builtin_pkg"
  echo
  echo "The MXFP4 kernels (mo.moe.mx4.matmul, mo.moe.mx4.matmul_swiglu) are"
  echo "registered inside MOGGKernelAPI. Use MAX_CUSTOM_EXTENSIONS if needed."
else
  echo "error: expected package at $builtin_pkg but it was not produced" >&2
  exit 1
fi
