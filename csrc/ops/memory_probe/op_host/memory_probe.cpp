#include "probe_api.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_matmul.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_memory_probe.h"
#include "aclrtlaunch_compute_probe.h"

namespace {
thread_local std::string error;
void need(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void check(int64_t status, const char *call) {
    if (status != 0) {
        const char *detail = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(call) + " failed: " + std::to_string(status) +
            (detail ? std::string(" ") + detail : ""));
    }
}
#define CHECK(call) check((call), #call)
template<class F> int guard(F op) noexcept {
    try { error.clear(); op(); return 0; }
    catch (const std::exception &e) { error = e.what(); return -1; }
    catch (...) { error = "unknown probe error"; return -1; }
}
struct DeviceMemory {
    void *ptr = nullptr;
    ~DeviceMemory() { if (ptr) (void)aclrtFree(ptr); }
    void allocate(uint64_t bytes) {
        need(!ptr, "memory already allocated");
        if (bytes) CHECK(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
};
struct Event {
    aclrtEvent value = nullptr;
    ~Event() { if (value) (void)aclrtDestroyEvent(value); }
    void create() { CHECK(aclrtCreateEventExWithFlag(&value, ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC)); }
};
int cellValue(uint64_t cell, uint32_t seed) {
    uint32_t x = static_cast<uint32_t>(cell) ^ seed;
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    return static_cast<int>(x % 5) - 2;
}
struct Buffer {
    DeviceMemory data;
    uint64_t bytes = 0;
    uint32_t seed = 0;
    // Exact prefix sum in units of 256-byte cells, 64 equal FP32 values/cell.
    std::vector<int64_t> prefix;
};
struct Plan {
    Buffer *buffer = nullptr;
    DeviceMemory indices;
    std::vector<int64_t> offsets;
    uint64_t usefulBytes = 0;
    int64_t chunkBytes = 0, order = 0;
};
struct Gemm {
    Buffer *weight = nullptr;
    DeviceMemory input, output, workspace;
    aclTensor *a = nullptr, *b = nullptr, *c = nullptr;
    int64_t m = 0, k = 0, n = 0, math = 1;
    uint64_t workspaceBytes = 0;
    aclOpExecutor *executor = nullptr;
    std::vector<float> reference;
    ~Gemm() {
        if (executor) (void)aclDestroyAclOpExecutor(executor);
        if (c) (void)aclDestroyTensor(c);
        if (b) (void)aclDestroyTensor(b);
        if (a) (void)aclDestroyTensor(a);
    }
};
struct Job {
    Event start, end;
    DeviceMemory output;
    std::vector<double> expected;
    Gemm *gemm = nullptr;
    bool submitted = false;
    int kind = 0; // 0=read, 1=compute, 2=GEMM
    int64_t stream = 0, dep0 = -1, dep1 = -1, blocks = 0;
    int64_t cache = 1, window = 2, rounds = 0;
    Plan *plan = nullptr;
    aclOpExecutor *executor = nullptr;
};
struct Session {
    int device = 0;
    bool init = false, selected = false, active = false, finished = true, failed = false;
    aclrtStream streams[2] = {nullptr, nullptr};
    Event epoch[2], trialEnd;
    int64_t cores = 0;
    uint64_t ub = 0;
    std::vector<std::unique_ptr<Buffer>> buffers;
    std::vector<std::unique_ptr<Plan>> plans;
    std::vector<std::unique_ptr<Gemm>> gemms;
    std::vector<std::unique_ptr<Job>> jobs;
    void sync() { for (auto stream : streams) if (stream) CHECK(aclrtSynchronizeStream(stream)); }
    ~Session() {
        for (auto stream : streams) if (stream) (void)aclrtSynchronizeStream(stream);
        jobs.clear(); gemms.clear(); plans.clear(); buffers.clear();
        for (auto &event : epoch) if (event.value) {
            (void)aclrtDestroyEvent(event.value); event.value = nullptr;
        }
        if (trialEnd.value) { (void)aclrtDestroyEvent(trialEnd.value); trialEnd.value = nullptr; }
        for (auto stream : streams) if (stream) (void)aclrtDestroyStream(stream);
        if (selected) (void)aclrtResetDevice(device);
        if (init) (void)aclFinalize();
    }
};
Session &get(void *p) { need(p, "null session"); return *static_cast<Session *>(p); }
template<class T> T &item(std::vector<std::unique_ptr<T>> &v, int64_t i) {
    need(i >= 0 && static_cast<size_t>(i) < v.size(), "invalid resource/job id"); return *v[i];
}
int64_t blockCount(Session &s, int64_t requested, int64_t tasks) {
    need(requested >= 0 && requested <= s.cores, "cores exceed available AIV cores");
    return std::min(requested ? requested : s.cores, tasks);
}
void initialize(Buffer &b, uint64_t bytes, uint32_t seed) {
    need(bytes > 0 && bytes % 256 == 0 && bytes <= (uint64_t(1) << 34),
        "buffer size must be a positive multiple of 256 bytes, at most 16 GiB");
    b.bytes = bytes; b.seed = seed; b.data.allocate(bytes);
    b.prefix.resize(bytes / 256 + 1);
    std::vector<float> host(std::min<uint64_t>(bytes / 4, 262144));
    for (uint64_t begin = 0; begin < bytes / 4; begin += host.size()) {
        uint64_t count = std::min<uint64_t>(host.size(), bytes / 4 - begin);
        for (uint64_t j = 0; j < count; ++j) host[j] = cellValue((begin + j) / 64, seed);
        CHECK(aclrtMemcpy(static_cast<char *>(b.data.ptr) + begin * 4, count * 4,
            host.data(), count * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    for (uint64_t c = 0; c < bytes / 256; ++c) b.prefix[c + 1] = b.prefix[c] + cellValue(c, seed) * 64;
}
aclTensor *matrix(void *data, int64_t rows, int64_t cols) {
    int64_t shape[2] = {rows, cols}, stride[2] = {cols, 1};
    auto tensor = aclCreateTensor(shape, 2, ACL_FLOAT, stride, 0, ACL_FORMAT_ND, shape, 2, data);
    need(tensor != nullptr, "aclCreateTensor failed"); return tensor;
}
void waitDeps(Session &s, int64_t stream, int64_t d0, int64_t d1) {
    need(s.active && !s.finished && !s.failed && stream >= 0 && stream < 2, "trial not active or invalid stream");
    for (auto dep : {d0, d1}) if (dep != -1) {
        auto &job = item(s.jobs, dep);
        need(job.submitted, "dependency was not submitted");
        CHECK(aclrtStreamWaitEvent(s.streams[stream], job.end.value));
    }
}
Job &newJob(Session &s, int64_t stream, int64_t d0, int64_t d1, int64_t *id) {
    need(!s.active && !s.failed && id && stream >= 0 && stream < 2, "prepare jobs outside active trials");
    for (auto dep : {d0, d1}) if (dep != -1) (void)item(s.jobs, dep);
    auto job = std::make_unique<Job>();
    job->start.create(); job->end.create();
    job->stream = stream; job->dep0 = d0; job->dep1 = d1;
    *id = static_cast<int64_t>(s.jobs.size());
    s.jobs.push_back(std::move(job));
    return *s.jobs.back();
}
void finishJob(Session &s, Job &job, int64_t stream) {
    CHECK(aclrtRecordEvent(job.end.value, s.streams[stream])); job.submitted = true;
}
}
extern "C" const char *pm_error() { return error.c_str(); }
extern "C" int pm_open(int device, void **out) {
    return guard([&] {
        need(out && device >= 0, "invalid session output/device"); *out = nullptr;
        auto s = std::make_unique<Session>(); s->device = device;
        CHECK(aclInit(nullptr)); s->init = true;
        CHECK(aclrtSetDevice(device)); s->selected = true;
        for (auto &stream : s->streams) CHECK(aclrtCreateStream(&stream));
        for (auto &event : s->epoch) event.create();
        s->trialEnd.create();
        auto p = platform_ascendc::PlatformAscendCManager::GetInstance();
        need(p, "platform unavailable"); s->cores = p->GetCoreNumAiv();
        p->GetCoreMemSize(platform_ascendc::CoreMemType::UB, s->ub);
        need(s->cores > 0 && s->ub > 0, "invalid platform capabilities"); *out = s.release();
    });
}
extern "C" int pm_info(void *p, int64_t *cores, uint64_t *ub) {
    return guard([&] { auto &s = get(p); need(cores && ub, "null info output"); *cores = s.cores; *ub = s.ub; });
}
extern "C" int pm_buffer(void *p, uint64_t bytes, uint32_t seed, int64_t *id) {
    return guard([&] {
        auto &s = get(p); need(!s.active && id, "allocate only outside trials");
        auto b = std::make_unique<Buffer>(); initialize(*b, bytes, seed);
        *id = s.buffers.size(); s.buffers.push_back(std::move(b));
    });
}
extern "C" int pm_plan(void *p, int64_t buffer, uint64_t bytes, int64_t chunk,
    int64_t order, uint32_t seed, int64_t *id) {
    return guard([&] {
        auto &s = get(p); auto &b = item(s.buffers, buffer);
        need(!s.active && id, "prepare plan only outside trials");
        need(chunk >= 256 && chunk <= 65536 && chunk % 256 == 0, "chunk must be a multiple of 256 in [256,65536]");
        need(bytes > 0 && bytes <= b.bytes && bytes % chunk == 0, "useful bytes must fit pool and contain whole chunks");
        need(order >= 0 && order <= 3, "order must be direct, indexed-linear, indexed-shuffle, or index-control");
        auto plan = std::make_unique<Plan>(); plan->buffer = &b;
        plan->usefulBytes = bytes; plan->chunkBytes = chunk; plan->order = order;
        plan->offsets.resize(bytes / chunk);
        // All orders touch exactly the same first usefulBytes address multiset.
        std::iota(plan->offsets.begin(), plan->offsets.end(), int64_t(0));
        if (order >= 2) { std::mt19937 rng(seed); std::shuffle(plan->offsets.begin(), plan->offsets.end(), rng); }
        plan->indices.allocate(plan->offsets.size() * sizeof(int64_t));
        CHECK(aclrtMemcpy(plan->indices.ptr, plan->offsets.size() * 8,
            plan->offsets.data(), plan->offsets.size() * 8, ACL_MEMCPY_HOST_TO_DEVICE));
        *id = s.plans.size(); s.plans.push_back(std::move(plan));
    });
}
extern "C" int pm_gemm(void *p, int64_t weight, int64_t m, int64_t k,
    int64_t n, int64_t math, int64_t *id) {
    return guard([&] {
        auto &s = get(p); auto &w = item(s.buffers, weight);
        need(!s.active && id, "prepare GEMM only outside trials");
        need(m > 0 && m <= 4096 && k > 0 && n > 0 && k <= 65536 && n <= 65536 &&
             static_cast<uint64_t>(k) * n * 4 == w.bytes, "GEMM weight bytes must equal k*n*4");
        need(math >= 0 && math <= 3, "math type must be in [0,3]");
        auto g = std::make_unique<Gemm>(); g->weight = &w; g->m = m; g->k = k; g->n = n; g->math = math;
        g->input.allocate(m * k * 4); g->output.allocate(m * n * 4);
        std::vector<float> ones(m * k, 1.0f);
        CHECK(aclrtMemcpy(g->input.ptr, m * k * 4, ones.data(), m * k * 4, ACL_MEMCPY_HOST_TO_DEVICE));
        g->a = matrix(g->input.ptr, m, k); g->b = matrix(w.data.ptr, k, n); g->c = matrix(g->output.ptr, m, n);
        g->reference.resize(n, 0.0f);
        for (int64_t row = 0; row < k; ++row) for (int64_t col = 0; col < n; ++col)
            g->reference[col] += cellValue((row * n + col) / 64, w.seed);
        *id = s.gemms.size(); s.gemms.push_back(std::move(g));
    });
}
extern "C" int pm_prepare_gemm(void *p, int64_t id, int64_t count) {
    return guard([&] {
        auto &s = get(p); auto &g = item(s.gemms, id);
        need(!s.active && count > 0 && count <= 1024, "prepare GEMM outside trials, count in [1,1024]");
        if (!g.executor) {
            uint64_t bytes = 0;
            CHECK(aclnnMatmulGetWorkspaceSize(g.a, g.b, g.c, static_cast<int8_t>(g.math), &bytes, &g.executor));
            need(g.executor, "null GEMM executor");
            CHECK(aclSetAclOpExecutorRepeatable(g.executor));
            g.workspace.allocate(bytes); g.workspaceBytes = bytes;
        }
    });
}
extern "C" int pm_reset(void *p) {
    return guard([&] {
        auto &s = get(p); need(!s.active && !s.failed, "finish previous trial before reset");
        s.sync(); s.jobs.clear();
        for (auto &g : s.gemms)
            CHECK(aclrtMemset(g->output.ptr, g->m * g->n * 4, 0xff, g->m * g->n * 4));
    });
}
extern "C" int pm_begin(void *p) {
    return guard([&] {
        auto &s = get(p); need(!s.active && !s.failed && !s.jobs.empty(), "prepare a nonempty graph before begin");
        for (auto &job : s.jobs) need(!job->submitted, "reset graph between executions");
        CHECK(aclrtRecordEvent(s.epoch[0].value, s.streams[0]));
        CHECK(aclrtStreamWaitEvent(s.streams[1], s.epoch[0].value));
        CHECK(aclrtRecordEvent(s.epoch[1].value, s.streams[1]));
        s.active = true; s.finished = false;
        try {
            for (auto &entry : s.jobs) {
                auto &job = *entry;
                waitDeps(s, job.stream, job.dep0, job.dep1);
                auto stream = s.streams[job.stream];
                CHECK(aclrtRecordEvent(job.start.value, stream));
                if (job.kind == 0) {
                    auto &plan = *job.plan;
                    int64_t requests = plan.offsets.size(), elements = plan.chunkBytes / 4;
                    int64_t indexed = plan.order == 3 ? 2 : (plan.order != 0);
                    CHECK(ACLRT_LAUNCH_KERNEL(memory_probe)(static_cast<uint32_t>(job.blocks), stream,
                        plan.buffer->data.ptr, plan.indices.ptr, job.output.ptr, requests, elements,
                        job.cache, indexed, job.window));
                } else if (job.kind == 1) {
                    int64_t elements = 1024;
                    CHECK(ACLRT_LAUNCH_KERNEL(compute_probe)(static_cast<uint32_t>(job.blocks), stream,
                        job.output.ptr, elements, job.rounds));
                } else {
                    auto &g = *job.gemm;
                    CHECK(aclnnMatmul(g.workspace.ptr, g.workspaceBytes, job.executor, stream));
                }
                finishJob(s, job, job.stream);
            }
            // Join both streams on stream 0; elapsed-time pairs are always
            // recorded on the same stream, as required by the ACL workflow.
            for (auto &job : s.jobs) if (job->stream == 1)
                CHECK(aclrtStreamWaitEvent(s.streams[0], job->end.value));
            CHECK(aclrtRecordEvent(s.trialEnd.value, s.streams[0]));
        } catch (...) { s.failed = true; throw; }
    });
}
extern "C" int pm_read(void *p, int64_t planId, int64_t stream, int64_t cores,
    int64_t cache, int64_t window, int64_t d0, int64_t d1, int64_t *id) {
    return guard([&] {
        auto &s = get(p); auto &plan = item(s.plans, planId);
        need(cache == 0 || cache == 1, "cache must be 0 or 1");
        need(window == 1 || window == 2, "software window must be 1 or 2");
        const int64_t requests = plan.offsets.size(), elements = plan.chunkBytes / 4;
        const int64_t blocks = blockCount(s, cores, requests);
        uint64_t scratch = ((elements + 63) / 64 + 7) / 8 * 32;
        need(3 * plan.chunkBytes + scratch + 32 + 8192 <= s.ub, "probe tile exceeds UB; reduce chunk bytes");
        auto &job = newJob(s, stream, d0, d1, id);
        job.output.allocate(blocks * 32);
        job.expected.resize(blocks, 0.0);
        const int64_t cells = plan.chunkBytes / 256;
        for (int64_t r = 0; r < requests; ++r) {
            int64_t cell = plan.offsets[r] * cells;
            job.expected[r % blocks] += plan.order == 3
                ? (plan.offsets[r] % 5 - 2) * elements
                : plan.buffer->prefix[cell + cells] - plan.buffer->prefix[cell];
        }
        CHECK(aclrtMemset(job.output.ptr, blocks * 32, 0xff, blocks * 32));
        job.kind = 0; job.plan = &plan; job.blocks = blocks;
        job.cache = cache; job.window = window;
    });
}
extern "C" int pm_compute(void *p, int64_t stream, int64_t cores, int64_t rounds,
    int64_t d0, int64_t d1, int64_t *id) {
    return guard([&] {
        auto &s = get(p); need(rounds >= 0 && rounds <= 1000000, "compute rounds must be in [0,1000000]");
        int64_t blocks = blockCount(s, cores, s.cores), elements = 1024;
        auto &job = newJob(s, stream, d0, d1, id);
        job.output.allocate(blocks * elements * 4);
        job.expected.assign(blocks * elements, static_cast<double>(rounds + 1));
        CHECK(aclrtMemset(job.output.ptr, blocks * elements * 4, 0xff, blocks * elements * 4));
        job.kind = 1; job.blocks = blocks; job.rounds = rounds;
    });
}
extern "C" int pm_matmul(void *p, int64_t gemm, int64_t stream,
    int64_t d0, int64_t d1, int64_t *id) {
    return guard([&] {
        auto &s = get(p); auto &g = item(s.gemms, gemm);
        need(g.executor, "prepare GEMM executor before begin");
        for (auto &other : s.jobs) if (other->gemm == &g)
            need(other->stream == stream, "a GEMM workspace/output cannot be used concurrently on different streams");
        auto &job = newJob(s, stream, d0, d1, id); job.gemm = &g;
        job.kind = 2; job.executor = g.executor;
    });
}
extern "C" int pm_finish(void *p) {
    return guard([&] { auto &s = get(p); need(s.active, "no active trial"); s.sync(); s.finished = true; s.active = false; });
}
extern "C" int pm_times(void *p, int64_t id, float *start, float *end) {
    return guard([&] {
        auto &s = get(p); need(s.finished && start && end, "finish trial before reading times");
        auto &job = item(s.jobs, id); need(job.submitted, "incomplete job");
        CHECK(aclrtEventElapsedTime(start, s.epoch[job.stream].value, job.start.value));
        CHECK(aclrtEventElapsedTime(end, s.epoch[job.stream].value, job.end.value));
    });
}
extern "C" int pm_total(void *p, float *elapsed) {
    return guard([&] {
        auto &s = get(p); need(s.finished && elapsed && !s.jobs.empty(), "finish before total time");
        CHECK(aclrtEventElapsedTime(elapsed, s.epoch[0].value, s.trialEnd.value));
    });
}
extern "C" int pm_verify(void *p, int64_t id) {
    return guard([&] {
        auto &s = get(p); need(s.finished, "finish before verify"); auto &job = item(s.jobs, id);
        need(job.submitted, "incomplete job");
        if (job.gemm) {
            auto &g = *job.gemm;
            std::vector<float> host(g.m * g.n);
            CHECK(aclrtMemcpy(host.data(), host.size() * 4, g.output.ptr, host.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST));
            for (int64_t r = 0; r < g.m; ++r) for (int64_t c = 0; c < g.n; ++c)
                need(std::isfinite(host[r * g.n + c]) && std::abs(host[r * g.n + c] - g.reference[c]) <= 0.01f,
                    "GEMM correctness failed");
        } else {
            // Read outputs have 8 FP32 slots/core; compute outputs are dense.
            // compute jobs use 1024 floats/core; read jobs use one value per 8 slots.
            bool read = job.kind == 0;
            size_t count = job.expected.size() * (read ? 8 : 1);
            std::vector<float> host(count);
            CHECK(aclrtMemcpy(host.data(), count * 4, job.output.ptr, count * 4, ACL_MEMCPY_DEVICE_TO_HOST));
            for (size_t i = 0; i < job.expected.size(); ++i) {
                double actual = host[i * (read ? 8 : 1)];
                need(std::isfinite(actual) && std::abs(actual - job.expected[i]) <= 0.01,
                    "probe checksum incorrect (missing/duplicate read or invalid compute)");
            }
        }
    });
}
extern "C" int pm_close(void *p) { return guard([&] { delete &get(p); }); }
