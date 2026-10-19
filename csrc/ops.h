#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// Opaque handles belong to one process and must be used on the creating thread.
// Open one session at a time. Close buffers before closing their session.
// All status functions return 0 on success, -1 on error; see l2_last_error().
const char *l2_last_error(void);
int l2_open(int device, void **session);
int l2_info(void *session, int64_t *cores, uint64_t *ub_bytes);
int l2_alloc(void *session, size_t bytes, uint32_t seed, void **buffer);
int l2_poison(void *session, void *buffer);
int l2_run(void *session, void *buffer, int64_t repeats, int64_t tile_elements,
    int64_t group_tiles, int64_t cache, int64_t cores, int timed, float *elapsed_ms);
int l2_verify(void *session, void *buffer, int full);
int l2_free(void *session, void *buffer);
int l2_close(void *session);
#ifdef __cplusplus
}
#endif
