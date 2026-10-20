#pragma once
// CPU test double only. Does not validate CANN ABI, pipeline synchronization,
// cache hints, physical concurrency, or hardware performance.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_set>
using aclError = int;
struct StubStream { float clock = 0; };
struct StubEvent { float time = 0; };
using aclrtStream = StubStream *;
using aclrtEvent = StubEvent *;
inline std::unordered_set<void *> stubAllocations;
inline int stubEvents = 0, stubStreams = 0;
constexpr int ACL_SUCCESS = 0, ACL_EVENT_TIME_LINE = 8, ACL_EVENT_SYNC = 1;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0, ACL_MEMCPY_HOST_TO_DEVICE = 1, ACL_MEMCPY_DEVICE_TO_HOST = 2;
constexpr int ACL_FLOAT = 0, ACL_FORMAT_ND = 0;
inline const char *aclGetRecentErrMsg() { return "CPU test runtime"; }
inline int aclInit(const char *) { return 0; }
inline int aclFinalize() { return 0; }
inline int aclrtSetDevice(int) { return 0; }
inline int aclrtResetDevice(int) { return 0; }
inline int aclrtCreateStream(aclrtStream *s) { *s = new StubStream(); ++stubStreams; return 0; }
inline int aclrtDestroyStream(aclrtStream s) { delete s; --stubStreams; return 0; }
inline int aclrtCreateEventExWithFlag(aclrtEvent *e, uint32_t) { *e = new StubEvent(); ++stubEvents; return 0; }
inline int aclrtDestroyEvent(aclrtEvent e) { delete e; --stubEvents; return 0; }
inline int aclrtRecordEvent(aclrtEvent e, aclrtStream s) { e->time = s->clock; return 0; }
inline int aclrtStreamWaitEvent(aclrtStream s, aclrtEvent e) { s->clock = std::max(s->clock, e->time); return 0; }
inline int aclrtSynchronizeStream(aclrtStream) { return 0; }
inline int aclrtEventElapsedTime(float *out, aclrtEvent a, aclrtEvent b) { *out = b->time - a->time; return 0; }
inline int aclrtMalloc(void **p, size_t bytes, int) {
    *p = std::malloc(bytes); if (!*p) return 1; stubAllocations.insert(*p); return 0;
}
inline int aclrtFree(void *p) {
    if (!stubAllocations.erase(p)) return 1;
    std::free(p); return 0;
}
inline int aclrtMemcpy(void *out, size_t cap, const void *in, size_t bytes, int) {
    if (bytes > cap) return 1;
    std::memcpy(out, in, bytes); return 0;
}
inline int aclrtMemset(void *out, size_t cap, int value, size_t bytes) {
    if (bytes > cap) return 1;
    std::memset(out, value, bytes); return 0;
}
struct aclTensor { float *data; int64_t rows, cols; };
inline aclTensor *aclCreateTensor(const int64_t *shape, uint64_t, int, const int64_t *, int64_t,
    int, const int64_t *, uint64_t, void *data) {
    return new aclTensor{static_cast<float *>(data), shape[0], shape[1]};
}
inline int aclDestroyTensor(aclTensor *t) { delete t; return 0; }
