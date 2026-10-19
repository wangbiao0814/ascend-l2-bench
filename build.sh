#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${ASCEND_HOME_PATH:?Load your CANN set_env.sh first}"
soc="${1:?Usage: bash build.sh EXACT_SOC_VERSION}"
python3 -c 'import torch, torch_npu, pybind11; assert torch.npu.is_available(), "NPU unavailable"'
cmake -S . -B build -DSOC_VERSION="$soc" \
    -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel "${BUILD_JOBS:-8}"
cp config.ini python/ascend_kernel/ascend_kernel/config.ini
(cd python/ascend_kernel && python3 setup.py bdist_wheel)
mkdir -p output
cp python/ascend_kernel/dist/*.whl output/
echo 'Build complete. Run python3 bench.py (no installation needed), or install output/*.whl.'
