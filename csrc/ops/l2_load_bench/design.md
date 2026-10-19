# l2_load_bench 设计

## 1. 接口与语义

```cpp
void l2_load_bench(const at::Tensor& x, at::Tensor out,
    int64_t repeats, int64_t tile_elements, int64_t group_tiles,
    int64_t cache, int64_t cores);
```

FP32、连续 ND、NPU 上的独立输入/输出，元素数量相等且为 tile_elements 整数倍。
没有对应的 PyTorch 同名算子；参考表达式为 `out = x + 1`。
每个元素重复执行 R 次相同表达式，每次均重新加载原始 x 并写 out，
不是 `out += 1`，最终 out 始终为 x+1。无需 workspace。

## 2. 两级切分与计算

设总 tile 数 T，实际核数 B=min(请求核数或全部 AIV 核数,T)。
核 b 持有 tile `b, b+B, b+2B, ...`；每个 tile 只属于一个核，写入无跨核竞争。
每个核将自己持有的 tile 分成 G 个 tile 一组（最后一组可不足 G）。
G=0 时一组包含该核全部 tile。

```text
for group in local_groups:
    for r in range(R):
        for tile in group:
            DataCopyPad(input_UB, x_GM[tile])
            inQueue.EnQue / DeQue   # 等待 MTE2 -> Vector
            Adds(output_UB, input_UB, 1.0f)
            outQueue.EnQue / DeQue  # 等待 Vector -> MTE3
            DataCopyPad(out_GM[tile], output_UB)
            FreeTensor             # 管理缓冲复用依赖
```

每次 Visit 都是相同指令路径，不缓存输入 UB 内容供后续 Visit 使用。
输入 NORMAL/DISABLE 两种 L2 策略，输出始终 DISABLE。

## 3. UB 与参数

| 缓冲 | 数量 | 总字节数 |
|---|---:|---:|
| FP32 输入队列 | 2 | 8 * tile_elements |
| FP32 输出队列 | 2 | 8 * tile_elements |
| 合计 | 4 | 16 * tile_elements |

bufferCoefficient=16，默认 tile_elements=4096（16 KiB 输入 tile），
队列分配总计 64 KiB。Host 查询 UB 容量并留 8 KiB 余量，不硬编码容量和核数。
限定 tile_elements 为 128 元素（512 字节）的倍数，范围 [128,16384]。
不支持尾 tile；测试端向上取整总元素数并报告实际字节数。
FP32 不需要 Cast。所有 EXEC_KERNEL_CMD 参数使用左值。
标量 tiling 参数由直调 kernel 参数传入，无额外 tiling GM Buffer。

## 4. 对照设计

N 为输入实际字节数。所有策略：GM->UB 逻辑读 N*R 字节，
UB->GM 逻辑写 N*R 字节，FP32 加法 (N/4)*R 次。
G=1 是立即重复读取，G 增大拉长重用距离，G=0 是分片完整扫描后重复。
近似同时活跃输入窗口 `B*G*tile_bytes`，上限 N；只用于规划，
不是缓存驻留保证。不同核无组间全局同步，其进度偏移会影响实际工作集。

首轮全冷、当前活跃数据完全容纳于 L2、无其他干扰时，理想读命中率约
(R-1)/R；实际命中率必须用硬件计数器验证。
性能同时受输出 HBM 写、MTE、队列同步、向量执行和调度影响。
计时包含所有阶段，不能将有效读带宽解释为纯 MTE2/L2/HBM 带宽。

## 5. 检查清单

- 已更新 ops.h、register.cpp、csrc/CMakeLists.txt。
- 已提供 CPU 调度覆盖/归属/次数验证及 NPU 全输出精度测试。
- 已提供事件计时、CSV 和单次 profiler 入口。
- Mac 本地没有 CANN/NPU，尚未编译和上板验证。
