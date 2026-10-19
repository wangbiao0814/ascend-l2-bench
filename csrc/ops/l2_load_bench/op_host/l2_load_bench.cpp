#include "ops.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_l2_load_bench.h"

namespace {
thread_local std::string lastError;
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void check(aclError code, const char *operation) {
    if (code != ACL_SUCCESS) {
        const char *detail = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(operation) + " failed (" +
            std::to_string(code) + ")" + (detail ? std::string(": ") + detail : ""));
    }
}
#define ACL_CHECK(call) check((call), #call)
template<class F> int guarded(F operation) noexcept {
    try { lastError.clear(); operation(); return 0; }
    catch (const std::exception &e) { lastError = e.what(); return -1; }
    catch (...) { lastError = "unknown native exception"; return -1; }
}
struct Session {
    int device = 0;
    bool initialized = false, selected = false;
    aclrtStream stream = nullptr;
    aclrtEvent start = nullptr, end = nullptr;
    int64_t cores = 0;
    uint64_t ub = 0;
    ~Session() {
        if (stream) (void)aclrtSynchronizeStream(stream);
        if (end) (void)aclrtDestroyEvent(end);
        if (start) (void)aclrtDestroyEvent(start);
        if (stream) (void)aclrtDestroyStream(stream);
        if (selected) (void)aclrtResetDevice(device);
        if (initialized) (void)aclFinalize();
    }
};
struct Buffer {
    Session *owner = nullptr;
    void *x = nullptr, *y = nullptr;
    size_t bytes = 0;
    uint32_t seed = 0;
    ~Buffer() {
        if (y) (void)aclrtFree(y);
        if (x) (void)aclrtFree(x);
    }
};
Session &sessionOf(void *handle) {
    require(handle != nullptr, "null session");
    return *static_cast<Session *>(handle);
}
Buffer &bufferOf(Session &s, void *handle) {
    require(handle != nullptr, "null buffer");
    auto &b = *static_cast<Buffer *>(handle);
    require(b.owner == &s, "buffer belongs to another session");
    return b;
}
// Exactly representable FP32 fractions, with varied values across tiles.
float inputAt(size_t i, uint32_t seed) {
    uint32_t v = static_cast<uint32_t>(i) ^ seed;
    v ^= v >> 16; v *= 0x7feb352dU; v ^= v >> 15;
    return (static_cast<int32_t>(v & 0xffffU) - 32768) / 1024.0f;
}
constexpr size_t chunkElements = 262144;
}

extern "C" const char *l2_last_error() { return lastError.c_str(); }
extern "C" int l2_open(int device, void **result) {
    return guarded([&] {
        require(result != nullptr && device >= 0, "invalid session output or device");
        *result = nullptr;
        auto s = std::make_unique<Session>();
        s->device = device;
        ACL_CHECK(aclInit(nullptr)); s->initialized = true;
        ACL_CHECK(aclrtSetDevice(device)); s->selected = true;
        ACL_CHECK(aclrtCreateStream(&s->stream));
        ACL_CHECK(aclrtCreateEventExWithFlag(&s->start, ACL_EVENT_TIME_LINE));
        ACL_CHECK(aclrtCreateEventExWithFlag(&s->end, ACL_EVENT_TIME_LINE));
        auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        require(platform != nullptr, "could not initialize Ascend platform");
        s->cores = platform->GetCoreNumAiv();
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, s->ub);
        require(s->cores > 0 && s->ub > 0, "invalid platform capabilities");
        *result = s.release();
    });
}
extern "C" int l2_info(void *handle, int64_t *cores, uint64_t *ub) {
    return guarded([&] {
        auto &s = sessionOf(handle);
        require(cores && ub, "null info output"); *cores = s.cores; *ub = s.ub;
    });
}
extern "C" int l2_alloc(void *handle, size_t bytes, uint32_t seed, void **result) {
    return guarded([&] {
        auto &s = sessionOf(handle);
        require(result && bytes > 0 && bytes % sizeof(float) == 0 &&
            bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()), "invalid allocation size");
        *result = nullptr;
        auto b = std::make_unique<Buffer>();
        b->owner = &s; b->bytes = bytes; b->seed = seed;
        ACL_CHECK(aclrtMalloc(&b->x, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(&b->y, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        std::vector<float> host(std::min(bytes / sizeof(float), chunkElements));
        auto dst = static_cast<char *>(b->x);
        for (size_t begin = 0; begin < bytes / sizeof(float); begin += host.size()) {
            size_t count = std::min(host.size(), bytes / sizeof(float) - begin);
            for (size_t j = 0; j < count; ++j) host[j] = inputAt(begin + j, seed);
            ACL_CHECK(aclrtMemcpy(dst + begin * sizeof(float), count * sizeof(float),
                host.data(), count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        }
        ACL_CHECK(aclrtMemset(b->y, bytes, 0xff, bytes));
        *result = b.release();
    });
}
extern "C" int l2_poison(void *handle, void *buffer) {
    return guarded([&] {
        auto &s = sessionOf(handle); auto &b = bufferOf(s, buffer);
        ACL_CHECK(aclrtSynchronizeStream(s.stream));
        ACL_CHECK(aclrtMemset(b.y, b.bytes, 0xff, b.bytes));
    });
}
extern "C" int l2_run(void *handle, void *buffer, int64_t repeats,
    int64_t tileElements, int64_t group, int64_t cache, int64_t cores,
    int timed, float *elapsed) {
    return guarded([&] {
        auto &s = sessionOf(handle); auto &b = bufferOf(s, buffer);
        require(tileElements >= 128 && tileElements <= 16384 && tileElements % 128 == 0,
            "tile_elements must be a multiple of 128 in [128, 16384]");
        require(b.bytes % (tileElements * sizeof(float)) == 0, "input must contain whole tiles");
        require(repeats >= 1 && repeats <= 1024, "repeats must be in [1, 1024]");
        require(group >= 0, "group_tiles must be >= 0");
        require(cache == 0 || cache == 1, "cache must be 0 or 1");
        require(cores >= 0 && cores <= s.cores, "cores must be in [0, available AIV cores]");
        require(static_cast<uint64_t>(tileElements) * 16 + 8192 <= s.ub,
            "tile exceeds device UB capacity; reduce --tile-kib");
        require(!timed || elapsed, "timing output is required");
        int64_t tiles = b.bytes / (tileElements * sizeof(float));
        uint32_t blocks = static_cast<uint32_t>(std::min(cores ? cores : s.cores, tiles));
        if (timed) ACL_CHECK(aclrtRecordEvent(s.start, s.stream));
        ACL_CHECK(ACLRT_LAUNCH_KERNEL(l2_load_bench)(blocks, s.stream, b.x, b.y,
            tiles, tileElements, repeats, group, cache));
        if (timed) ACL_CHECK(aclrtRecordEvent(s.end, s.stream));
        ACL_CHECK(aclrtSynchronizeStream(s.stream));
        if (timed) ACL_CHECK(aclrtEventElapsedTime(elapsed, s.start, s.end));
    });
}
extern "C" int l2_verify(void *handle, void *buffer, int full) {
    return guarded([&] {
        auto &s = sessionOf(handle); auto &b = bufferOf(s, buffer);
        ACL_CHECK(aclrtSynchronizeStream(s.stream));
        size_t n = b.bytes / sizeof(float);
        std::vector<float> host(std::min(n, full ? chunkElements : size_t(1024)));
        // Sample four windows including both ends; full mode covers every element.
        size_t windows = full ? (n + host.size() - 1) / host.size() : 4;
        for (size_t w = 0; w < windows; ++w) {
            size_t begin = full ? w * host.size() : (n - host.size()) * w / 3;
            size_t count = std::min(host.size(), n - begin);
            ACL_CHECK(aclrtMemcpy(host.data(), count * sizeof(float),
                static_cast<char *>(b.y) + begin * sizeof(float), count * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST));
            for (size_t j = 0; j < count; ++j) {
                require(host[j] == inputAt(begin + j, b.seed) + 1.0f,
                    "correctness check failed: output != input + 1");
            }
        }
    });
}
extern "C" int l2_free(void *handle, void *buffer) {
    return guarded([&] {
        auto &s = sessionOf(handle); auto &b = bufferOf(s, buffer);
        ACL_CHECK(aclrtSynchronizeStream(s.stream)); delete &b;
    });
}
extern "C" int l2_close(void *handle) {
    return guarded([&] { delete &sessionOf(handle); });
}
