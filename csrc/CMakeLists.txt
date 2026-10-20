ascendc_library(l2_device_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/ops/l2_load_bench/op_kernel/l2_load_bench.cpp
    ${PROJECT_OP_SRC_BASE}/ops/memory_probe/op_kernel/memory_probe.cpp
    ${PROJECT_OP_SRC_BASE}/ops/memory_probe/op_kernel/compute_probe.cpp)
add_library(l2_bench SHARED
    ${PROJECT_OP_SRC_BASE}/ops/l2_load_bench/op_host/l2_load_bench.cpp
    ${PROJECT_OP_SRC_BASE}/ops/memory_probe/op_host/memory_probe.cpp)
target_link_libraries(l2_bench PRIVATE l2_device_kernel ascendcl tiling_api platform opapi nnopbase dl)
target_link_directories(l2_bench PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_include_directories(l2_bench PRIVATE
    ${PROJECT_OP_SRC_BASE}
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/include/external
    ${ASCEND_HOME_PATH}/include/experiment/platform
    ${ASCEND_HOME_PATH}/include/experiment/runtime)
set_target_properties(l2_bench PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output
    BUILD_RPATH "${ASCEND_HOME_PATH}/lib64")
