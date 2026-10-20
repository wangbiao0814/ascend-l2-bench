#pragma once
#include "acl/acl.h"
#include "kernel_operator.h"
#define ACLRT_LAUNCH_KERNEL(name) stub_launch_##name
extern "C" void memory_probe(void *, void *, void *, int64_t, int64_t, int64_t, int64_t, int64_t);
inline int stub_launch_memory_probe(uint32_t blocks, aclrtStream stream, void *x,
    void *indices, void *output, int64_t requests, int64_t elements,
    int64_t cache, int64_t indexed, int64_t window) {
    if (std::getenv("PROBE_STUB_FAIL")) return 123;
    AscendC::stubBlocks = blocks;
    for (uint32_t b = 0; b < blocks; ++b) {
        AscendC::stubBlock = b;
        memory_probe(x, indices, output, requests, elements, cache, indexed, window);
    }
    if (std::getenv("PROBE_STUB_CORRUPT")) static_cast<float *>(output)[0] += 999;
    stream->clock += .01f + static_cast<float>(requests * elements) / 10000000.0f;
    return 0;
}
