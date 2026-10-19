set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/ascend_kernel/ascend_kernel/lib)
ascendc_library(no_workspace_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/ops/l2_load_bench/op_kernel/l2_load_bench.cpp)
add_library(ascend_kernel SHARED
    ${PROJECT_OP_SRC_BASE}/register.cpp
    ${PROJECT_OP_SRC_BASE}/ops/l2_load_bench/op_host/l2_load_bench.cpp)
target_link_libraries(ascend_kernel PRIVATE no_workspace_kernel torch_npu
    ascendcl tiling_api nnopbase opapi register platform ascendalog dl)
target_link_directories(ascend_kernel PRIVATE ${TORCH_DIR}/lib ${TORCH_NPU_DIR}/lib)
target_include_directories(ascend_kernel PRIVATE
    ${PROJECT_OP_SRC_BASE} ${PROJECT_OP_SRC_BASE}/utils
    ${TORCH_DIR}/include ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include ${TORCH_NPU_DIR}/include/third_party/acl/inc
    ${TORCH_NPU_DIR}/include/third_party/hccl/inc ${PYTHON_INCLUDE_DIR}
    ${ASCEND_INCLUDE_DIR} ${ASCEND_INCLUDE_DIR}/external
    ${ASCEND_INCLUDE_DIR}/experiment/platform ${ASCEND_INCLUDE_DIR}/experiment/runtime)
