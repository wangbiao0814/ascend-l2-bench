#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${ASCEND_HOME_PATH:?Load your CANN set_env.sh first}"
soc="${1:?Usage: bash build.sh EXACT_SOC_VERSION}"
cmake -S . -B build-acl -DSOC_VERSION="$soc" \
    -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DCMAKE_BUILD_TYPE=Release
cmake --build build-acl --parallel "${BUILD_JOBS:-8}"
echo 'Build complete: output/libl2_bench.so. Run python3 bench.py.'
