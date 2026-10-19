#include "kernel_operator.h"

// Every visit reloads GM into UB, executes Adds, and stores to GM.
// No UB reuse across visits: changing order only changes reuse distance in L2.
class L2LoadBench {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int64_t tiles,
        int64_t tileElements, int64_t repeats, int64_t groupTiles, int64_t cache)
    {
        tileElements_ = tileElements;
        repeats_ = repeats;
        groupTiles_ = groupTiles;
        block_ = AscendC::GetBlockIdx();
        blocks_ = AscendC::GetBlockNum();
        localTiles_ = (tiles + blocks_ - 1 - block_) / blocks_;
        x_.SetGlobalBuffer((__gm__ float *)x, tiles * tileElements);
        y_.SetGlobalBuffer((__gm__ float *)y, tiles * tileElements);
        x_.SetL2CacheHint(cache ? AscendC::CacheMode::CACHE_MODE_NORMAL
                              : AscendC::CacheMode::CACHE_MODE_DISABLE);
        // Keep output writes from competing for cache capacity in every case.
        y_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        pipe_.InitBuffer(in_, 2, tileElements * sizeof(float));
        pipe_.InitBuffer(out_, 2, tileElements * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // groupTiles=0 means scan this core's entire shard before repeating.
        int64_t group = groupTiles_ == 0 ? localTiles_ : groupTiles_;
        for (int64_t begin = 0; begin < localTiles_; begin += group) {
            int64_t end = begin + group;
            end = end < localTiles_ ? end : localTiles_;
            for (int64_t r = 0; r < repeats_; ++r) {
                for (int64_t t = begin; t < end; ++t) {
                    Visit((t * blocks_ + block_) * tileElements_);
                }
            }
        }
    }
private:
    __aicore__ inline void Visit(int64_t offset)
    {
        AscendC::DataCopyExtParams copy{1,
            static_cast<uint32_t>(tileElements_ * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        auto input = in_.AllocTensor<float>();
        AscendC::DataCopyPad(input, x_[offset], copy, pad);
        in_.EnQue(input);
        input = in_.DeQue<float>();
        auto output = out_.AllocTensor<float>();
        AscendC::Adds(output, input, 1.0f, static_cast<int32_t>(tileElements_));
        out_.EnQue(output);
        in_.FreeTensor(input);
        output = out_.DeQue<float>();
        AscendC::DataCopyPad(y_[offset], output, copy);
        out_.FreeTensor(output);
    }
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> in_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> out_;
    AscendC::GlobalTensor<float> x_, y_;
    int64_t tileElements_, repeats_, groupTiles_, block_, blocks_, localTiles_;
};

extern "C" __global__ __aicore__ void l2_load_bench(
    GM_ADDR x, GM_ADDR y, int64_t tiles, int64_t tileElements,
    int64_t repeats, int64_t groupTiles, int64_t cache)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    L2LoadBench op;
    op.Init(x, y, tiles, tileElements, repeats, groupTiles, cache);
    op.Process();
}
