#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${LAB_PYTHON:-}" ]; then
  lab_env="${LAB_ENV:-/tmp/gemm-lab-qualification-venv}"
  if [ ! -x "$lab_env/bin/python" ]; then uv venv --python 3.13 "$lab_env"; fi
  uv pip sync --python "$lab_env/bin/python" requirements.lock
  LAB_PYTHON="$lab_env/bin/python"
fi
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/kernels-fa2-cache}"
export MAX_JOBS="${MAX_JOBS:-4}"
"$LAB_PYTHON" scripts/run.py "$@"
