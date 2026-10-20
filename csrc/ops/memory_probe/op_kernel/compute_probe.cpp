#include "kernel_operator.h"

extern "C" __global__ __aicore__ void compute_probe(GM_ADDR output,
    int64_t elements, int64_t rounds)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> queue;
    pipe.InitBuffer(queue, 1, elements * sizeof(float));
    auto value = queue.AllocTensor<float>();
    AscendC::Duplicate(value, 1.0f, static_cast<int32_t>(elements));
    for (int64_t r = 0; r < rounds; ++r) {
        AscendC::Adds(value, value, 1.0f, static_cast<int32_t>(elements));
    }
    queue.EnQue(value);
    value = queue.DeQue<float>();
    AscendC::GlobalTensor<float> target;
    target.SetGlobalBuffer((__gm__ float *)output + AscendC::GetBlockIdx() * elements, elements);
    AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(elements * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPad(target, value, copy);
    queue.FreeTensor(value);
}
