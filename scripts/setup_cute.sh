#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .deps
if [ ! -d .deps/cutlass-v4.6.1/.git ]; then
  git clone --depth 1 --branch v4.6.1 https://github.com/NVIDIA/cutlass.git .deps/cutlass-v4.6.1
fi
test "$(git -C .deps/cutlass-v4.6.1 rev-parse HEAD)" = e05f953a5b3d38adc240df2ff928e0421c2abba3
