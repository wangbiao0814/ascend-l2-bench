#include "kernel_operator.h"

// FP32 byte-matched probes, not INT8 expert arithmetic. Every input lane
// contributes to a per-core checksum; only 32 bytes/core are written to GM.
class MemoryProbe {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR indices, GM_ADDR output,
        int64_t requests, int64_t chunkElements, int64_t cache,
        int64_t indexed, int64_t window)
    {
        const int64_t block = AscendC::GetBlockIdx();
        const int64_t blocks = AscendC::GetBlockNum();
        const int64_t local = (requests + blocks - 1 - block) / blocks;
        AscendC::GlobalTensor<float> source, result;
        AscendC::GlobalTensor<int64_t> offsets;
        source.SetGlobalBuffer((__gm__ float *)x, requests * chunkElements);
        source.SetL2CacheHint(cache ? AscendC::CacheMode::CACHE_MODE_NORMAL
                                  : AscendC::CacheMode::CACHE_MODE_DISABLE);
        offsets.SetGlobalBuffer((__gm__ int64_t *)indices, requests);
        result.SetGlobalBuffer((__gm__ float *)output + block * 8, 8);
        result.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        pipe.InitBuffer(inputQueue, 2, chunkElements * sizeof(float));
        pipe.InitBuffer(accumBuffer, chunkElements * sizeof(float));
        // Level-2 ReduceSum scratch: one partial per 64 FP32 lanes, padded to 32 B.
        const int64_t partials = ((chunkElements + 63) / 64 + 7) / 8 * 8;
        pipe.InitBuffer(scratchBuffer, partials * sizeof(float));
        pipe.InitBuffer(outQueue, 1, 32);
        auto accum = accumBuffer.Get<float>();
        auto scratch = scratchBuffer.Get<float>();
        AscendC::Duplicate(accum, 0.0f, static_cast<int32_t>(chunkElements));
        // Window is 1 or 2. Two copies can be enqueued before consumption.
        for (int64_t begin = 0; begin < local; begin += window) {
            int64_t width = local - begin;
            width = width < window ? width : window;
            for (int64_t j = 0; j < width; ++j) {
                const int64_t request = block + (begin + j) * blocks;
                const int64_t chunk = indexed ? offsets.GetValue(request) : request;
                auto input = inputQueue.AllocTensor<float>();
                AscendC::DataCopyExtParams copy{1,
                    static_cast<uint32_t>(chunkElements * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
                // Set the hint on the actual slice used by DataCopyPad as well.
                if (indexed == 2) {
                    // Index/control negative control: same index reads and UB
                    // accumulation, no KV payload request to HBM.
                    float v = static_cast<float>(chunk % 5 - 2);
                    AscendC::Duplicate(input, v, static_cast<int32_t>(chunkElements));
                } else if (cache) {
                    auto slice = source[chunk * chunkElements];
                    slice.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
                    AscendC::DataCopyPad(input, slice, copy, pad);
                } else {
                    auto slice = source[chunk * chunkElements];
                    slice.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                    AscendC::DataCopyPad(input, slice, copy, pad);
                }
                inputQueue.EnQue(input);
            }
            for (int64_t j = 0; j < width; ++j) {
                auto input = inputQueue.DeQue<float>();
                AscendC::Add(accum, accum, input, static_cast<int32_t>(chunkElements));
                inputQueue.FreeTensor(input);
            }
        }
        auto sum = outQueue.AllocTensor<float>();
        AscendC::Duplicate(sum, 0.0f, 8);
        AscendC::ReduceSum(sum, accum, scratch, static_cast<int32_t>(chunkElements));
        outQueue.EnQue(sum);
        sum = outQueue.DeQue<float>();
        AscendC::DataCopyExtParams store{1, 32, 0, 0, 0};
        AscendC::DataCopyPad(result, sum, store);
        outQueue.FreeTensor(sum);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuffer, scratchBuffer;
};

extern "C" __global__ __aicore__ void memory_probe(GM_ADDR x, GM_ADDR indices,
    GM_ADDR checksum, int64_t requests, int64_t chunkElements, int64_t cache,
    int64_t indexed, int64_t window)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    MemoryProbe probe;
    probe.Run(x, indices, checksum, requests, chunkElements, cache, indexed, window);
}

