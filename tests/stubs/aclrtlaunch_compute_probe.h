#pragma once
#include "aclrtlaunch_memory_probe.h"
extern "C" void compute_probe(void *, int64_t, int64_t);
inline int stub_launch_compute_probe(uint32_t blocks, aclrtStream stream, void *output,
    int64_t elements, int64_t rounds) {
    AscendC::stubBlocks = blocks;
    for (uint32_t b = 0; b < blocks; ++b) {
        AscendC::stubBlock = b; compute_probe(output, elements, rounds);
    }
    stream->clock += .01f + static_cast<float>(rounds) / 100000;
    return 0;
}
