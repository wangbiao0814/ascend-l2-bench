#include "aclnnop/aclnn_matmul.h"
extern "C" int probe_stub_resources() {
    return static_cast<int>(stubAllocations.size()) + stubEvents + stubStreams + stubExecutors;
}
