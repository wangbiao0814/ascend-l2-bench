#include <torch/library.h>
#include "ops.h"
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("l2_load_bench(Tensor x, Tensor(a!) out, int repeats, int tile_elements, "
          "int group_tiles, int cache, int cores=0) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("l2_load_bench", TORCH_FN(ascend_kernel::l2_load_bench));
}
