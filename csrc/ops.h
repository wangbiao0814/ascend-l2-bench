#pragma once
#include <ATen/ATen.h>
namespace ascend_kernel {
void l2_load_bench(const at::Tensor &self, at::Tensor output, int64_t repeats,
    int64_t tileElements, int64_t groupTiles, int64_t cache, int64_t requestedCores);
}
