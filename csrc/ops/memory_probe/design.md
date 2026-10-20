# memory_probe 实现设计

实验假设与对照见 `docs/moe_memory_experiments.md`；实际命令和限制见 `docs/moe_experiments_runbook.md`。

## 接口与语义

`csrc/probe_api.h` 定义纯 C ABI。`pm_buffer/plan/gemm` 准备资源；`pm_reset` 清除上一轮任务；`pm_read/compute/matmul` 在计时外准备任务图节点；`pm_begin` 异步提交整个图；`pm_finish` 显式同步；`pm_times/total/verify` 读取时间及验证。Session 独占资源，同一 host 线程使用，关闭时先同步再释放。

- memory_probe：FP32 只读探针。一次调用读取有用字节集合一次，每个元素参与校验；不是权重元素加一或写回。
- order=0 直接顺序；order=1 索引顺序；order=2 对同一索引集合做固定种子排列；order=3 保留索引与 UB 计算、不读取 KV 载荷的负对照。
- compute_probe：1024 个 FP32 元素常驻 UB，初值 1，每轮 Adds(+1)，仅用于时间间隔控制，不作为 GEMM 替代。
- E1 的 C0/G1 使用 ACLNN Matmul，FP32 输入，显式 cubeMathType；不是完整 MoE 或真实路由。

## 两级切分

块大小为 T 字节，256 的整数倍；总有用字节数必须整除 T，当前不支持尾块。

B=min(请求核数或查询到的可用 AIV 核数, 请求块数)。核 b 处理请求 `b,b+B,b+2B,...`。索引模式先读请求对应的索引，再搬入同样大小的有效块。每核只有自己的一份 checksum 输出，无写入别名。

输入队列使用两块 UB，软件窗口仅允许 1/2：窗口 2 时先提交最多两次搬入，再消费两份输入。软件窗口不等于硬件请求槽数量。

## 访存探针流程

```text
Duplicate(accumulator, 0)
for each window of local requests:
    for request in window:
        chunk = direct_request_id or indices[request_id]
        input = inputQueue.AllocTensor<float>()
        DataCopyPad(input, source[chunk])
        inputQueue.EnQue(input)
    for request in window:
        input = inputQueue.DeQue<float>()
        Add(accumulator, accumulator, input)
        inputQueue.FreeTensor(input)
ReduceSum(checksum, accumulator, scratch)
DataCopyPad(GM_checksum_per_core, checksum, 32 bytes)
```

NORMAL/DISABLE 在实际 DataCopyPad 使用的 GlobalTensor 切片上分别设置常量；仍需检查生成指令与实际计数器，不能把源码分支当作已验证的硬件缓存行为。

归约使用基本 Level-2 ReduceSum，不使用 Pattern 归约。accumulator 被最终归约消费，此后不复用；输入 GM 不会被修改。

## UB 分配

| 缓冲 | 字节数 |
|---|---:|
| inputQueue 双缓冲 | 2T |
| accumulator | T |
| 基本 ReduceSum 的 partial scratch | align32(ceil(T/256) × 4) |
| checksum VECOUT | 32 |

默认 T=16384：输入 32768 + accumulator 16384 + scratch 256 + 输出 32 = **49440 B**。Host 另留 8192 B，并与硬件查询到的 UB 容量比较。索引当前逐请求从 GM 读取，没有额外 UB 索引缓存；其成本通过 indexed-linear 和 index-control 对照单独观察。

compute_probe：一个 4096 B VECOUT buffer。原生访存探针无固定系统 workspace；只申请每任务 checksum。GEMM workspace 来自 ACLNN 查询，复用 executor 和 workspace 只可在同一 stream 顺序使用。

## 数据与正确性

输入按每 256 B 一组生成 [-2,2] 范围的合成整数 FP32，CPU 保存精确的 cell prefix sum。Host 根据请求索引和分核方式计算每核期望校验和；每轮输出先填 NaN，防止未写出的旧值通过。校验和不是无碰撞散列，不能代替静态覆盖检查；CPU 测试另验证完整分配和非整除核数情况。

Matmul 输入 X 为全 1，W 使用相同输入生成器，CPU 计算每列之和，逐元素验证输出。其结果只用于受控计算依赖，不模拟实际 token 内容。

## 计时与验证边界

事件、输出、CPU 参考和 GEMM 资源均在图提交前准备。每阶段 event elapsed time 只使用同一 stream 的起止事件；每个 stream 有独立局部起点，不把两条局部时间线叠加来计算并发。总时间由 stream 0 的起点和等待所有背景结束后的 join 事件测量。

CPU 模拟测试可检验控制逻辑、引用、校验、依赖与清理，不能检验实际 CANN ABI、指令、流水同步、缓存、并发或性能。必须在目标设备运行 `tests/test_probe_npu.py` 并结合系统 profiler 复核。
