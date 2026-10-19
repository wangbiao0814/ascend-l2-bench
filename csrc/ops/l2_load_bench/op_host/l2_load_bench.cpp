#include <algorithm>
#include "torch_kernel_helper.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_l2_load_bench.h"

namespace ascend_kernel {
void l2_load_bench(const at::Tensor &self, at::Tensor output, int64_t repeats,
    int64_t tileElements, int64_t groupTiles, int64_t cache, int64_t requestedCores)
{
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
        "input must be an NPU tensor");
    TORCH_CHECK(self.device() == output.device(), "device mismatch");
    TORCH_CHECK(self.scalar_type() == at::kFloat && output.scalar_type() == at::kFloat,
        "only float32 is supported");
    TORCH_CHECK(self.is_contiguous() && output.is_contiguous(), "ND contiguous tensors required");
    TORCH_CHECK(self.sizes() == output.sizes(), "output shape mismatch");
    TORCH_CHECK(!self.is_alias_of(output), "input and output must not share storage");
    TORCH_CHECK(tileElements >= 128 && tileElements <= 16384 && tileElements % 128 == 0,
        "tile_elements must be a multiple of 128 in [128, 16384]");
    TORCH_CHECK(self.numel() > 0 && self.numel() % tileElements == 0,
        "input must contain a positive whole number of tiles");
    TORCH_CHECK(repeats > 0 && repeats <= 1024, "repeats must be in [1, 1024]");
    TORCH_CHECK(groupTiles >= 0, "group_tiles must be >= 0 (0 means full scan)");
    TORCH_CHECK(cache == 0 || cache == 1, "cache must be 0 or 1");
    TORCH_CHECK(requestedCores >= 0, "cores must be >= 0 (0 means automatic)");
    c10_npu::NPUGuard deviceGuard(self.device());
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "could not initialize Ascend platform");
    int64_t available = platform->GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    TORCH_CHECK(available > 0 && ubSize > 0, "invalid platform capabilities");
    // 2 input buffers + 2 output buffers, all float32. Reserve 8 KiB.
    TORCH_CHECK(static_cast<uint64_t>(tileElements) * 16 + 8192 <= ubSize,
        "tile is too large for this product's UB; reduce --tile-kib");
    TORCH_CHECK(requestedCores <= available, "requested cores exceed available AIV cores");
    int64_t tiles = self.numel() / tileElements;
    int64_t cores = requestedCores == 0 ? available : requestedCores;
    uint32_t blockDim = static_cast<uint32_t>(std::min(cores, tiles));
    EXEC_KERNEL_CMD(l2_load_bench, blockDim, self, output, tiles, tileElements,
        repeats, groupTiles, cache);
}
}
