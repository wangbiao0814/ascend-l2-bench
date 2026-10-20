#pragma once
#include "acl/acl.h"
inline int stubExecutors = 0;
struct aclOpExecutor { const aclTensor *a, *b; aclTensor *c; };
inline int aclnnMatmulGetWorkspaceSize(const aclTensor *a, const aclTensor *b,
    aclTensor *c, int8_t, uint64_t *bytes, aclOpExecutor **executor) {
    if (a->cols != b->rows) return 1;
    *bytes = 128; *executor = new aclOpExecutor{a, b, c}; ++stubExecutors; return 0;
}
inline int aclnnMatmul(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    for (int64_t i = 0; i < e->a->rows; ++i) for (int64_t j = 0; j < e->b->cols; ++j) {
        float value = 0;
        for (int64_t k = 0; k < e->a->cols; ++k)
            value += e->a->data[i * e->a->cols + k] * e->b->data[k * e->b->cols + j];
        e->c->data[i * e->c->cols + j] = value;
    }
    s->clock += .05f; return 0;
}

inline int aclSetAclOpExecutorRepeatable(aclOpExecutor*) { return 0; }
inline int aclDestroyAclOpExecutor(aclOpExecutor *e) { delete e; --stubExecutors; return 0; }
