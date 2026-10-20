#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <stdexcept>
#include <vector>
#define __aicore__
#define __global__
#define __gm__
#define KERNEL_TASK_TYPE_DEFAULT(x)
#define KERNEL_TYPE_AIV_ONLY 0
using GM_ADDR = void *;
namespace AscendC {
inline uint32_t stubBlock = 0, stubBlocks = 1;
inline uint32_t GetBlockIdx() { return stubBlock; }
inline uint32_t GetBlockNum() { return stubBlocks; }
enum class TPosition { VECIN, VECOUT, VECCALC };
enum class CacheMode { CACHE_MODE_DISABLE, CACHE_MODE_NORMAL };
template<class T> struct LocalTensor {
    std::shared_ptr<std::vector<uint8_t>> storage;
    size_t offset = 0;
    T *ptr() const { return reinterpret_cast<T *>(storage->data() + offset); }
    LocalTensor operator[](int64_t i) const { return {storage, offset + static_cast<size_t>(i) * sizeof(T)}; }
};
template<class T> struct GlobalTensor {
    T *data = nullptr;
    void SetGlobalBuffer(T *p, int64_t = 0) { data = p; }
    void SetL2CacheHint(CacheMode) {}
    T GetValue(int64_t i) { return data[i]; }
    GlobalTensor operator[](int64_t i) const { return {data + i}; }
};
struct TensorMemory {
    size_t bytes = 0;
    std::shared_ptr<std::vector<uint8_t>> data;
};
template<TPosition, int> struct TQue : TensorMemory {
    std::deque<std::shared_ptr<std::vector<uint8_t>>> tensors;
    template<class T> LocalTensor<T> AllocTensor() { return {std::make_shared<std::vector<uint8_t>>(bytes), 0}; }
    template<class T> void EnQue(LocalTensor<T> t) { tensors.push_back(t.storage); }
    template<class T> LocalTensor<T> DeQue() {
        if (tensors.empty()) throw std::runtime_error("empty CPU queue");
        auto t = tensors.front(); tensors.pop_front(); return {t, 0};
    }
    template<class T> void FreeTensor(LocalTensor<T>) {}
};
template<TPosition> struct TBuf : TensorMemory {
    template<class T> LocalTensor<T> Get() { return {data, 0}; }
};
struct TPipe {
    template<class Q> void InitBuffer(Q &q, int, int64_t bytes) { q.bytes = bytes; }
    template<class B> void InitBuffer(B &b, int64_t bytes) {
        b.bytes = bytes; b.data = std::make_shared<std::vector<uint8_t>>(bytes);
    }
};
struct DataCopyExtParams { uint16_t blockCount; uint32_t blockLen; uint32_t srcStride, dstStride; uint32_t reserved; };
template<class T> struct DataCopyPadExtParams { bool pad; uint8_t left, right; T value; };
template<class T> void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src,
    DataCopyExtParams p, DataCopyPadExtParams<T>) { std::memcpy(dst.ptr(), src.data, p.blockLen); }
template<class T> void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src,
    DataCopyExtParams p) { std::memcpy(dst.data, src.ptr(), p.blockLen); }
template<class T> void Duplicate(LocalTensor<T> out, T value, int32_t n) { std::fill(out.ptr(), out.ptr() + n, value); }
template<class T> void Add(LocalTensor<T> out, LocalTensor<T> a, LocalTensor<T> b, int32_t n) {
    for (int32_t i = 0; i < n; ++i) out.ptr()[i] = a.ptr()[i] + b.ptr()[i];
}
template<class T> void Adds(LocalTensor<T> out, LocalTensor<T> a, T value, int32_t n) {
    for (int32_t i = 0; i < n; ++i) out.ptr()[i] = a.ptr()[i] + value;
}
template<class T> void ReduceSum(LocalTensor<T> out, LocalTensor<T> a, LocalTensor<T>, int32_t n) {
    T value = 0; for (int32_t i = 0; i < n; ++i) value += a.ptr()[i]; out.ptr()[0] = value;
}
}
