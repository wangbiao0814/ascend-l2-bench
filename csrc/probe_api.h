#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// All handles belong to their session, which owns their lifetime. One host thread.
// pm_read/compute/matmul prepare a graph outside the timed region.
// pm_begin enqueues it without synchronizing; pm_finish is the explicit join.
const char *pm_error(void);
int pm_open(int device, void **session);
int pm_info(void *session, int64_t *cores, uint64_t *ub);
int pm_buffer(void *session, uint64_t bytes, uint32_t seed, int64_t *id);
// order=0: direct, 1: indexed ascending, 2: same index multiset shuffled,
// 3: shuffled index/control instructions only, no KV payload reads.
int pm_plan(void *session, int64_t buffer, uint64_t usefulBytes,
    int64_t chunkBytes, int64_t order, uint32_t seed, int64_t *id);
int pm_gemm(void *session, int64_t weight, int64_t m, int64_t k,
    int64_t n, int64_t mathType, int64_t *id);
int pm_prepare_gemm(void *session, int64_t gemm, int64_t count);
int pm_reset(void *session);
int pm_begin(void *session);
int pm_read(void *session, int64_t plan, int64_t stream, int64_t cores,
    int64_t cache, int64_t window, int64_t dep0, int64_t dep1, int64_t *job);
int pm_compute(void *session, int64_t stream, int64_t cores, int64_t rounds,
    int64_t dep0, int64_t dep1, int64_t *job);
int pm_matmul(void *session, int64_t gemm, int64_t stream,
    int64_t dep0, int64_t dep1, int64_t *job);
int pm_finish(void *session);
// Stage event envelopes use separate stream-local epochs; not a common timeline.
// pm_total uses a same-stream epoch/join pair covering both streams.
int pm_times(void *session, int64_t job, float *startMs, float *endMs);
int pm_total(void *session, float *elapsedMs);
int pm_verify(void *session, int64_t job);
int pm_close(void *session);
#ifdef __cplusplus
}
#endif
