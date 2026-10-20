# MoE 访存实验：实现、参数与 tile 划分详解

本文对应当前仓库的 `moe_bench.py`、`probe_backend.py` 和 `csrc/ops/memory_probe/` 实现。运行入口和环境准备见 [运行说明](moe_experiments_runbook.md)，研究假设见 [实验设计](moe_memory_experiments.md)。本文以**代码实际执行的行为**为准，区分已实现的对照与仍需设备计数器验证的解释。

以下例子假定设备报告 **48 个可用 AIV 核**，数据已经在 NPU HBM 中。实验不包含 CPU→NPU 权重搬运，不依赖 PyTorch。

## 1. 四组实验分别回答什么

| 实验 | 对应问题 | 核心对照 | 主要观测 |
|---|---|---|---|
| E0 | 重用前的访存干扰是否让第二次读变慢？ | A→A；A→D→A；A→计算间隔→A | 第二次读 A 的延迟，以及间隔是否匹配 |
| E1 | 提前读权重能否隐藏后续 GEMM 的等待？ | 按需 GEMM；串行预读；另一个 stream 提前预读 | 整图耗时、G1 耗时、预读成本、实际并发 |
| E2 | 把同一专家的两次访问靠近，是否减少重读代价？ | A→B→C→A 与 A→A→B→C | A2 延迟和等工作量图的整图耗时 |
| E3 | KV 小块随机读取是否拖慢关键读取？限制背景强度是否有效？ | 前景独立；大块顺序、小块顺序、随机背景；核数/窗口限制 | 前景 F 延迟，以及背景付出的时间代价 |

E0 是基础诊断，E2 是重排对照，E1 是提前加载对照，E3 是并发干扰对照。一次 `--experiment all` 按 E0→E2→E1→E3 的顺序运行。

这些实验可以发现“在这个工作负载下出现了延迟差异”。仅有延迟不能直接证明 L2 淘汰、物理请求槽占满或实际模型端到端收益。

## 2. 共同的数据与读取实现

### 2.1 MiB、tile、chunk 与旧版 group

- `1 MiB = 1,048,576 B`，`1 KiB = 1,024 B`。本文所有 MiB 都是二进制单位。
- `tile`：普通读取探针一次从 GM/HBM 地址空间搬入 UB 的块，默认 `--tile-kib 16`，即 16,384 B。
- `chunk`：E3 背景请求的块。大块模式使用 tile 大小，小块模式使用 `--kv-chunk-bytes`，可以小至 256 B。
- `block`：一次读取 kernel 启动的逻辑工作单元，由 AIV 执行。指定 block 数不等于物理核绑定或核预留。
- `window`：每核先入队多少个输入块再消费；当前为 1 或 2，是软件队列策略。

**tile 大小由参数决定，不随总数据规模自动增大。** 在默认配置下，16、42、84、256、512 MiB 的普通读取都使用 16 KiB tile；E3 的小块背景是例外。

旧版 `bench.py` 的 `group` 和内部重复访问属于另一套 benchmark。新入口没有 `--groups` 或读取 `--repeats`；每个 read 节点只扫描一次。A1 和 A2 是两次独立 kernel 对同一个缓冲区的访问。`--compute-repeats` 只重复 E1 的 C0 GEMM。

### 2.2 每次 read 实际做什么

读取 kernel 使用 FP32 合成数据。每 256 B 内填充相同的小整数，取值范围为 −2 到 2；不同区域由种子决定。数据在计时前生成并放到设备上。

每个 block 执行：

1. 按核间划分公式确定自己负责的请求。
2. 从每个请求的 GM 地址读取一个完整块到 UB。
3. 将块内所有 FP32 元素逐位置累加到 UB 累加向量。
4. 所有请求处理后，做一次 ReduceSum，得到该 block 的校验和。
5. 每 block 仅向 GM 写 32 B 的对齐结果，包括校验值及填充。

所以它是“读取并消费全部输入的探针”，包含 Add、ReduceSum 和队列开销，不能当成没有计算开销的纯 DMA 测速。48 个 block 的结果总量仅为 `48×32 B = 1,536 B`，不会写回一份与权重等大的数组。

默认 window=2 的逻辑如下：

```text
读取第 1 个负责的块 → 入队
读取第 2 个负责的块 → 入队
出队第 1 块并累加 → 释放
出队第 2 块并累加 → 释放
继续下一批……
最后 ReduceSum → 写校验结果
```

两个输入缓冲允许排队，但这段逻辑本身不证明硬件始终有两个在途 HBM 请求。

### 2.3 UB 空间与 tile 大小限制

设块大小为 T 字节，当前读取实现分配：

| UB 内容 | 字节数 |
|---|---:|
| 输入队列两个缓冲 | 2T |
| 累加向量 | T |
| ReduceSum scratch | `align32(ceil(T/256)×4)` |
| 输出队列 | 32 |

16 KiB 时总计 `49,440 B`。Host 校验还保留 8,192 B 余量，要求设备查询的 UB 至少容纳 `57,632 B`。即使 window=1，仍分配两个输入缓冲。

CLI 允许 `--tile-kib 1..64`，但较大 tile 仍可能被实际 UB 容量检查拒绝。**允许的参数范围不等于设备一定能运行其上限。** 输入必须包含完整块，当前没有尾块路径。

## 3. 48 核如何分 tile

设输入大小为 S，tile 大小为 T，请求总数 `N=S/T`。实际 block 数：

```text
B = min(指定核数或设备可用 AIV 核数, N)
```

逻辑 block b（从 0 开始）负责：

```text
请求编号：b, b+B, b+2B, ...，直到编号 >= N
请求数量：floor((N+B-1-b)/B)
```

这是按步长 B 轮转分配，不是每核一段连续大区间。多个 block 可由设备并行执行，具体调度与并发程度需要设备时间线确认。

### 3.1 默认 16 KiB、48 block 的实例

| 输入大小 | tile 总数 | 每 block 的分配 | 每 block 的输入量 |
|---|---:|---|---|
| 16 MiB | 1,024 | block 0–15 各 22；其余各 21 | 352 / 336 KiB |
| 42 MiB | 2,688 | 全部各 56 | 896 KiB |
| 64 MiB | 4,096 | block 0–15 各 86；其余各 85 | 1,376 / 1,360 KiB |
| 84 MiB | 5,376 | 全部各 112 | 1,792 KiB |
| 256 MiB | 16,384 | block 0–15 各 342；其余各 341 | 5,472 / 5,456 KiB |
| 512 MiB | 32,768 | block 0–31 各 683；其余各 682 | 10,928 / 10,912 KiB |

以 42 MiB 为例：

```text
block 0 : tile 0, 48, 96, ... , 2640   共 56 个
block 1 : tile 1, 49, 97, ... , 2641   共 56 个
...
block 47: tile 47,95,143, ... , 2687   共 56 个
```

tile i 的字节地址是 `buffer_base + i×16,384`。同一 block 相邻两次负责的 tile 起点相隔 `48×16 KiB = 768 KiB`。

A1 和 A2 都访问 A 的同一组地址。B、C、D 是独立缓冲区；它们的 tile 编号各自从 0 开始，B 的 tile 0 和 A 的 tile 0 不是同一个地址。

### 3.2 改成 4 个 block 会怎样

42 MiB 仍为 2,688 个 tile，但每 block 负责 `2,688/4=672` 个，即 10.5 MiB。block 0 负责 0、4、8……2684，block 3 负责 3、7、11……2687。

因此 `--cores` 改变并行工作单元数量和每个 block 的工作量，`--tile-kib` 改变单次处理粒度，两者不能混为一个变量。

## 4. 公共执行流程与计时边界

### 4.1 一个参数组合如何执行

每个参数组合创建一个独立后端 session，提前分配缓冲区、生成数据、建立读取计划或 GEMM 资源。一个组合内各 variant 和 trial 复用这些缓冲区；不同组合之间不保证使用相同设备地址。

E0/E2 首先校准计算间隔，然后执行：

```text
每轮随机打乱 variant 顺序
  每个 variant：
    可选 condition 扫描，完成并校验
    准备图资源、事件、输出与参考值
    提交本次实验图
    等待两条 stream 完成
    校验所有输出
    正式 trial 写入文件；warmup 不进入统计
```

默认 warmup 为 2 轮，每轮都执行每个 variant；随后正式采样 30 轮。每轮 variant 随机顺序由 `--seed` 控制，目的是减少固定运行顺序带来的偏差。

### 4.2 condition 前缀不是硬件清缓存

默认每个 variant 前扫描独立的 512 MiB 缓冲区，设置 `--condition-mib 0` 可关闭。

- 前缀使用 `--cores`，缓存策略固定为 on，即使目标实验 `--cache off` 也是如此。
- 前缀在实验图之前单独完成，不计入 `target_us` 或 `makespan_us`。
- 它是尽力统一起始状态的访存干扰，不保证所有缓存行都被清除。
- E0 的 A1 在前缀之后重新读取 A，因此“有 512 MiB 前缀”不意味着 A2 必然冷读。
- E0/E2 的孤立间隔校准不逐次加此前缀，校准时的状态可能不同于正式图。

### 4.3 event 测量的边界

Host 先准备好图的节点和资源，再提交事件和 kernel。准备阶段不纳入图计时；执行时没有逐节点同步回 Python。

```text
stream 0: epoch0 ── 实验节点 ── 等待 stream 1 的结束事件 ── trialEnd
stream 1: 等待 epoch0 ── epoch1 ── 实验节点
```

| 字段 | 含义 | 使用注意 |
|---|---|---|
| `target_us` | A2、G1、F 或独立背景 K 的 start/end event 包络 | 包含该事件区间的执行与等待影响 |
| `makespan_us` | stream 0 从 epoch0 到 trialEnd | 包含图内所有节点、提交间隙和最终 join；不包含 condition、初始化与 CPU 校验 |
| `stage_span_us` | 单 stream 图的最后 end 减最早 start | 双 stream 图为空；背景独立基线只有一个 stream，仍有值 |
| `background_us` | stream 1 各阶段 duration 之和 | 当前 E1/E3 通常只有一个背景节点 |
| `background_end_us` | 背景相对 stream 1 局部 epoch 的结束时刻 | 不能直接减 stream 0 的时间戳 |

两条 stream 的阶段时间各自使用局部 epoch。**不能将 stages 中两个 stream 的 start/end 直接叠在同一横轴上证明重叠。** `makespan_us` 则使用 stream 0 同一起点和终点，覆盖整个图。

这些图通过 Host 按顺序提交，不是捕获好的硬件 graph；多 stream 也不保证真正同时执行。证明物理并发要另看系统 profiler 时间线。

## 5. E0：重用前的干扰与时间间隔

### 5.1 三个 variant 的实现

全部在 stream 0 顺序执行，目标均为 A2。

```text
immediate   : [A1]──────────────[A2]
interference: [A1]──[读取 D]────[A2]
time_gap    : [A1]──[UB 计算]───[A2]
```

| variant | 输入访问量 | 用途 |
|---|---|---|
| `immediate` | 2A | 立即复用基线 |
| `interference` | 2A+D | 访存干扰后的复用 |
| `time_gap` | 2A，加少量计算结果写回 | 时间间隔控制 |

A1 和 A2 是同一个 A 缓冲区，D 是独立数据。每个 read 内扫描一次；A 被读两次来自任务图中有两个节点。

### 5.2 计算间隔如何校准

先独立测 D 的 event 耗时，取 3 次中位数。再用计算 kernel 拟合该耗时，最多尝试 5 轮，每轮也取 3 次中位数。接受误差为 `max(目标耗时×15%, 2 μs)`。

`--gap-rounds` 是校准初始迭代数，不是保证最终使用的值。最终轮数限制在 0 到 1,000,000，记录在输出中。

计算 kernel 每核初始化 1,024 个 FP32 值，在 UB 中反复加 1，最后写回 4 KiB/核。48 核合计写回 192 KiB；它没有大块 GM 输入，但并非完全没有访存。校准匹配标志仅代表孤立测量匹配，正式图中仍应检查 gap 与 D 的实际阶段时间。

### 5.3 实例：A=42 MiB，D=64 MiB

```bash
python3 moe_bench.py --experiment e0 --sizes-mib 42 \
  --interference-mib 64 --tile-kib 16 --cores 48 \
  --cache on --condition-mib 512 --warmup 2 --trials 30 \
  --output results-e0-42m-64m
```

- A：2,688 个 tile，每核 56 个；A1 和 A2 都如此。
- D：4,096 个 tile，前 16 核各 86 个，其余各 85 个。
- `immediate` 逻辑输入读取量为 84 MiB。
- `interference` 为 148 MiB；其整图时间增加本来就包含读取 D，不能全部归因于 A2 重载。
- 本命令产出 `3 variants×30 trials=90` 行正式记录。

### 5.4 如何判读

优先比较同一个 trial 内 `interference.A2 − immediate.A2`，再比较 `interference.A2 − time_gap.A2`。

若只有访存干扰组的 A2 稳定变慢，而计算间隔组不变，且两者实际间隔接近，则支持“额外访存影响复用”的解释。若两种间隔组都变慢，要考虑时间、调度或其他状态变化；若差异很小，只能说在此参数下未观察到明显延迟惩罚。

要进一步称为 L2 淘汰，需要定位 A2 的缓存命中和物理读取计数。`--cache off` 是读取指令的缓存策略请求，也需要确认目标平台的实际效果。

## 6. E2：相同专家的访问重排

### 6.1 三个 variant

```text
interleaved        : [A1]──[B]──[C]──[A2]
clustered          : [A1]──[A2]──[B]──[C]
compute_gap_control: [A1]──[UB 计算]──[A2]
```

均在 stream 0。前两个图读相同的 A/B/C 缓冲区，逻辑总读取量都是 `2A+B+C`，只改变顺序。`clustered` 的 B/C 仍在图内，其耗时仍计入 makespan，即使目标 A2 已经结束。

计算间隔控制不包含 B/C 的等量访存，其用途是辅助解释 A2，不应作为等工作量整图加速基线。

### 6.2 B/C 的大小如何确定

`--interference-mib` 是 **B+C 的总和**，不是每个专家的大小。设总量为 D、tile 大小为 T：

```text
BBytes = floor(D / (2T)) × T
CBytes = D − BBytes
```

所以 interference=64 MiB 时，B=C=32 MiB。模拟三个等大专家 A=B=C=42 MiB，应设置 sizes=42、interference=84。

### 6.3 实例：三个 42 MiB 专家

```bash
python3 moe_bench.py --experiment e2 --sizes-mib 42 \
  --interference-mib 84 --tile-kib 16 --cores 48 \
  --cache on --warmup 2 --trials 30 \
  --output results-e2-equal42m
```

每个专家各有 2,688 个 tile，每核 56 个。对于 block 0：

```text
interleaved:
  读 A 的 0,48,...2640 → 读 B 的 0,48,...2640
  → 读 C 的 0,48,...2640 → 再读 A 的 0,48,...2640

clustered:
  读 A 的 0,48,...2640 → 再读 A 的 0,48,...2640
  → 读 B 的 0,48,...2640 → 读 C 的 0,48,...2640
```

以上是单 block 的逻辑序列；每个方框实际是一个覆盖全体 block 的独立 kernel，下一 kernel 在同一 stream 中等待上一 kernel 完成。不是某个 block 提前完成 A 后自行跨进 B。

两个等工作量图均读取 168 MiB。若 clustered 的 A2 和整图耗时都稳定下降，说明该轨迹下邻近复用有收益；如果只有 A2 降低，需要继续检查其他阶段是否抵消收益。

此处重排是合成轨迹。真实模型是否允许把两次专家任务靠近，要另检查 token、层间依赖和调度约束。

## 7. E1：提前读取与真实 GEMM

### 7.1 C0、P1、G1 各是什么

| 名称 | 实际操作 | 数据与控制关系 |
|---|---|---|
| C0 | ACLNN Matmul，代表前序计算窗口 | 使用独立权重与输入 |
| P1 | 普通读取探针扫描 W | 与 G1 使用同一个 W 缓冲区 |
| G1 | ACLNN Matmul，代表后续专家的一次投影 | 等待 C0；有 P1 时也等待 P1 |

G1 输入不是 C0 的输出，二者是事件依赖链。权重地址提前已知，未实现动态路由预测。P1 只是提前读取，不能跨 kernel 保留 UB 数据，也没有锁定 L2 缓存行。

### 7.2 三个 variant 与时间线

```text
demand:
  stream 0: [C0_0] [C0_1 ...] [G1]

serial_prefetch:
  stream 0: [C0_0] [C0_1 ...] [P1] [G1]

overlap_prefetch（期望的关系，不代表已经观测到物理重叠）:
  stream 0: [C0_0] [C0_1 ...] ──等待 P1── [G1]
  stream 1:    [P1 读取 W] ────────────────┘
```

实际提交顺序为：先提交 C0_0，再向 stream 1 提交 P1，然后提交剩余 C0，最后提交等待 C0/P1 的 G1。`--compute-repeats 1` 时只有 C0_0，没有 C0_1。

**当前实现有一个需要控制的变量：** serial_prefetch 的 P1 使用 `--cores`，overlap_prefetch 的 P1 使用 `--kv-cores`。若分别设为 48 和 4，比较同时改变了预读核数和时机。要先隔离调度差异，可把两者都设为 4；随后单独扫描预读核数。

### 7.3 默认矩阵形状与权重大小

当前使用 FP32 合成矩阵，X 全 1，输出与 CPU 计算的权重列和参考比较。

| 阶段 | 矩阵形状 | X 大小 | W 大小 | Y 大小 |
|---|---|---:|---:|---:|
| C0 | `[512,1024] × [1024,1024]` | 2 MiB | 4 MiB | 2 MiB |
| G1 | `[8,7168] × [7168,2048]` | 224 KiB | 56 MiB | 64 KiB |

G1 权重大小由 `matrix_k×matrix_n×4` 决定，**不使用 `--sizes-mib`**。这是单块投影矩阵，不是完整的 gate/up/down 三矩阵专家，也不是 INT8 权重读取。

`--math-type 1` 允许 FP32 输入采用降精度计算路径；它不能把合成数据大小改成 BF16/INT8。其他值为 0/2/3，需在目标平台确认支持和精度。

### 7.4 P1 的 tile 划分与 GEMM tiling 的区别

56 MiB / 16 KiB = 3,584 个预读 tile：

- 4 block：每 block 896 个 tile，读取 14 MiB。
- 48 block：前 32 个各 75 个 tile，后 16 个各 74 个。

这里将 W 按线性地址分 tile。**C0/G1 的计算 tiling、Cube 使用量由 ACLNN 决定**，不是上述 16 KiB 划分，也不由 `--cores` 直接指定。

### 7.5 建议先运行的公平对照

```bash
python3 moe_bench.py --experiment e1 \
  --matrix-k 7168 --matrix-n 2048 --tokens 8 \
  --compute-m 512 --compute-k 1024 --compute-n 1024 \
  --compute-repeats 1 --tile-kib 16 --cores 4 --kv-cores 4 \
  --cache on --math-type 1 --warmup 2 --trials 30 \
  --output results-e1-p1cores4
```

然后保持其他参数相同，分别测试 compute-repeats=2/4/8，观察更长计算窗口是否允许预读产生净收益；核数扫描时同步修改 cores 和 kv-cores，避免串行/并行预读强度不同。

主要看：

1. `demand.makespan − overlap_prefetch.makespan`：引入提前预读是否有净收益。
2. `serial_prefetch.makespan − overlap_prefetch.makespan`：移动预读时机是否有收益。
3. G1、P1、C0 的阶段耗时：G1 变快是否被预读成本或计算干扰抵消。

executor 配置为 repeatable，workspace 在计时前准备；同一 GEMM 资源按同一 stream 顺序复用。需要系统时间线证明 C0/P1 物理并发，不能仅从图结构推断。

## 8. E3：KV 请求粒度、随机性与限制策略

### 8.1 前景与背景的关系

- F：读取 sizes-mib 大小的 A，使用 tile-kib 和 cores，运行在 stream 0。
- K：扫描 kv-mib 大小的 KV 池，通常运行在 stream 1。
- 同时运行时先提交 K，再提交 F；二者没有数据依赖。
- 所有载荷读取模式都扫描同一个 KV 池的相同地址集合一次，不做有放回随机采样。

```text
stream 1: [背景 K：不同粒度、顺序、block 数、软件窗口]
stream 0:    [前景 F：固定大小、固定 block 数]
```

图示表达可并发关系，实际是否重叠仍需验证。即使设置 F=4 block、K=12 block，也没有物理核隔离或固定核心映射。

### 8.2 十个 variant

| variant | 背景模式 | 背景 block 数 | window | target |
|---|---|---|---:|---|
| `alone` | 无背景 | — | — | F |
| `background_only_large` | 大块直接顺序读，无 F | kv-cores | 2 | K |
| `background_only_random` | 小块随机索引读，无 F | kv-cores | 2 | K |
| `large_linear` | tile 大小，直接顺序读 | kv-cores | 2 | F |
| `small_indexed_linear` | chunk 大小，顺序索引读 | kv-cores | 2 | F |
| `small_random` | chunk 大小，随机索引读 | kv-cores | 2 | F |
| `limited_cores` | 同 small_random | limit-cores | 2 | F |
| `limited_window` | 同 small_random | kv-cores | 1 | F |
| `limited_random` | 同 small_random | limit-cores | 1 | F |
| `index_control` | 随机索引，但不读 KV 载荷 | kv-cores | 2 | F |

`index_control` 读取同一随机索引序列，用索引派生的小数值填充 UB，再执行累加。它用于判断索引/UB 计算等控制路径的影响，不是同工作量的 KV 优化方案。

### 8.3 256 MiB KV、12 block 的请求数量

每个小块索引为 int64，占 8 B。N 个 chunk 对应 N 个索引，分块仍采用 `r=b+t×B`。

| chunk 大小 | 请求总数 N | 12 block 的分配 | 逻辑索引字节数 |
|---|---:|---|---:|
| 256 B | 1,048,576 | 前 4 个各 87,382；其余各 87,381 | 8 MiB |
| 1,024 B | 262,144 | 前 4 个各 21,846；其余各 21,845 | 2 MiB |
| 4,096 B | 65,536 | 前 4 个各 5,462；其余各 5,461 | 0.5 MiB |
| 16,384 B | 16,384 | 前 4 个各 1,366；其余各 1,365 | 0.125 MiB |

大块直接模式不执行索引 GetValue；上述索引开销表针对 indexed 模式。索引逻辑字节数不等于实际 HBM 流量，索引本身也可能命中缓存。

减小 chunk 会增加请求数、循环次数、索引访问次数和队列操作次数。因此小块模式变慢可能来自多种因素，不能直接称为硬件请求槽耗尽。

### 8.4 顺序索引与随机索引如何分核

设 8 个 chunk、2 block，为便于理解，假设随机排列为 `[6,1,4,0,7,3,2,5]`。此排列只是示意，不代表某个 seed 的真实输出。

```text
直接顺序 / 顺序索引：
  block 0 请求 r=0,2,4,6 → chunk 0,2,4,6
  block 1 请求 r=1,3,5,7 → chunk 1,3,5,7

随机索引：
  block 0 请求 r=0,2,4,6 → chunk 6,4,7,2
  block 1 请求 r=1,3,5,7 → chunk 1,0,3,5
```

实际 C++ 使用 mt19937(seed) 和 shuffle，把 0..N−1 排列一次。顺序与随机访问的集合相同，每个载荷 chunk 都只读一次。Python 用同一个 seed 控制 variant 顺序，但随机引擎与 C++ 不同。

### 8.5 实例：42 MiB 前景、256 MiB 背景

```bash
python3 moe_bench.py --experiment e3 --sizes-mib 42 \
  --kv-mib 256 --kv-chunk-bytes 1024 --tile-kib 16 \
  --cores 4 --kv-cores 12 --limit-cores 1 \
  --cache on --warmup 2 --trials 30 \
  --output results-e3-42m-kv256m-chunk1k
```

- F：2,688 个 16 KiB tile，4 block，每 block 672 个。
- K_large：16,384 个 16 KiB 请求，12 block。
- K_random：262,144 个 1 KiB 请求，12 block。
- limited_cores：仍是同样 262,144 个请求，但全部交给 1 block，window=2。
- limited_window：仍是 12 block，只将每批先入队的块数从 2 改为 1。
- limited_random：1 block、window=1，每个请求仍访问一次。
- 本命令有 10 个 variant，产出 300 行正式记录。

### 8.6 如何判断干扰来源

| 对照 | 可帮助判断的变量 | 仍需注意 |
|---|---|---|
| small_random vs alone | 背景整体干扰 | 缓存、带宽、计算与调度都可能参与 |
| small_indexed_linear vs large_linear | 粒度与索引路径 | 同时改变粒度及有无索引 |
| small_random vs small_indexed_linear | 同粒度下的地址顺序 | 实际缓存行为仍需计数器 |
| limited_cores vs small_random | 背景 block 数 | 背景持续时间也可能增加 |
| limited_window vs small_random | 软件批次窗口 | 不是物理请求槽限额 |
| index_control vs small_random | KV 载荷请求是否为主要因素 | 两者不具有相同物理流量 |

必须同时查看 F 延迟、K 延迟/结束时间和 makespan，避免将“背景被拖得更久”解释为免费提升。当前未实现等物理 HBM 带宽匹配，也未读取请求槽占用计数，因此不能单凭 E3 确认槽位饱和。

## 9. 完整参数表

### 9.1 公共控制与读取参数

| 参数 | 默认 | 含义及适用范围 |
|---|---|---|
| `--experiment` | e0 | e0/e1/e2/e3/all |
| `--sizes-mib` | 16 42 84 | E0/E2 的 A，E3 的 F；支持多个整数值 |
| `--interference-mib` | 16 64 256 | E0 的 D；E2 的 B+C；支持多个整数值 |
| `--condition-mib` | 512 | 每 variant 前的独立干扰前缀，0 关闭 |
| `--tile-kib` | 16 | 普通 read、E1 P1、E3 大块背景的块大小 |
| `--cores` | 0 | 普通 read/计算间隔/condition 的 block 数；0 使用可用 AIV 数，再受请求数量限制 |
| `--cache` | on | on/off/both；仅控制读取探针，不控制 ACLNN GEMM |
| `--gap-rounds` | 4096 | E0/E2 校准计算间隔的初始轮数 |
| `--device` | 0 | ACL 设备编号 |
| `--warmup` | 2 | 每个 variant 的预热轮数，不写正式结果 |
| `--trials` | 30 | 每个 variant 的正式轮数 |
| `--seed` | 2026 | 合成数据、随机索引及 variant 顺序的种子来源 |
| `--library` | 仓库下 output/libl2_bench.so | 加载的原生动态库路径 |
| `--output` | 仓库下 results-moe | 结果目录，拒绝覆盖已有结果文件 |
| `--variants` | 全部 | 仅运行列出的 variant，适合单轨迹 profiling |
| `--dry-run` | 关闭 | 输出组合和图，不加载 CANN 库、不执行设备工作 |

### 9.2 E1 矩阵与预读参数

| 参数 | 默认 | 实际作用 |
|---|---:|---|
| `--tokens` | 8 | G1 的 M |
| `--matrix-k` | 7168 | G1 的 K |
| `--matrix-n` | 2048 | G1 的 N |
| `--compute-m` | 512 | C0 的 M |
| `--compute-k` | 1024 | C0 的 K |
| `--compute-n` | 1024 | C0 的 N |
| `--compute-repeats` | 1 | C0 调用次数，范围 1..1024 |
| `--math-type` | 1 | ACLNN Matmul 数学类型，CLI 允许 0/1/2/3 |
| `--kv-cores` | 12 | E1 overlap_prefetch 中 P1 的 block 数 |

注意：serial_prefetch 中 P1 使用 cores。sizes/interference/KV chunk 不参与 E1 参数组合；E1 权重大小仅由 K/N/FP32 字节数决定。

### 9.3 E3 背景参数

| 参数 | 默认 | 实际作用 |
|---|---|---|
| `--kv-mib` | 256 | KV 池大小，等于每次背景扫描的有用载荷量 |
| `--kv-chunk-bytes` | 256 1024 4096 16384 | 小块背景粒度，每个值独立形成一组实验 |
| `--kv-cores` | 12 | 普通背景的 block 数 |
| `--limit-cores` | 1 | limited_cores / limited_random 的背景 block 数 |

`limit-cores <= kv-cores`。chunk 必须在 256..65,536 B 之间、为 256 B 倍数，并整除 KV 池大小。CLI 校验所有参数，即使某参数不参与当前实验，也可能因为无效而拒绝启动。

### 9.4 对齐、核数和容量检查

- sizes/interference/condition 的字节数必须被 tile 字节数整除。
- G1 的 `K×N×4` 也必须被 tile 字节数整除。
- 所有显式请求的核数不能超过设备报告的可用 AIV 核数；`--cores 0` 为自动值。
- native 缓冲区大小必须是正的 256 B 倍数，单缓冲区上限为 16 GiB。
- GEMM 的 M 范围为 1..4096，K/N 为 1..65,536，权重大小必须恰好为 `K×N×4`。
- 实际申请能否成功还受设备可用内存和 UB 容量限制。dry-run 不验证真实设备容量或原生库可用性。

## 10. 参数组合数量与输出读取

### 10.1 多值参数是笛卡尔积

E0/E2：sizes × interference × cache。E1：一组矩阵形状 × cache。E3：sizes × KV chunk × cache。

例如：

```bash
python3 moe_bench.py --experiment e0 --sizes-mib 16 42 84 \
  --interference-mib 64 --cores 48 --trials 30 \
  --output results-e0-three-sizes
```

这是 3 个参数组合，每组合 3 个 variant，各采样 30 次，共 270 行正式记录，不包含 warmup 与校准调用。cache=both 再将组合数翻倍。

E3 的 alone、大块基线也会在每个小块 chunk 参数组合中重新测量；不要以为不同 chunk 下它们是同一次采样。

### 10.2 四个输出文件

| 文件 | 适合查看的内容 |
|---|---|
| meta.json | 完整命令参数、计时口径、实现限制 |
| trials.csv | 逐 trial 的目标耗时、整图耗时与校验标志 |
| stages.jsonl | 每个阶段耗时、stream、图依赖、校准结果 |
| summary.json | 各 variant 的 p50/p95、配对差值与 bootstrap CI |

控制台只打印 target 和 total，缺少 A1/D/gap 等阶段，分析时应查看 stages。

可在结果目录的上级执行以下只读示例，列出每个阶段：

```bash
python3 - <<'PY'
import json
from pathlib import Path
path = Path('results-e0-three-sizes/stages.jsonl')
for line in path.open():
    record = json.loads(line)
    detail = ', '.join(
        f"{s['step']}@stream{s['stream']}={s['duration_us']:.3f}us"
        for s in record['stages'])
    print(record['weight_bytes'] // 2**20, 'MiB',
          record['variant'], record['trial'], detail,
          'calibration=', record['calibration'])
PY
```

`correctness_passed` 来自实际输出与 CPU 参考比较；异常时程序报错，不能把未完成的目录当成完整实验。读取校验覆盖所有输入元素的贡献，但校验和不是无碰撞的逐元素轨迹证明。

### 10.3 统计口径

- p50/p95 采用排序后 `(n−1)×p` 位置的线性插值。
- summary 中配对差值为同一 trial 的 `first−second`，然后取这些差值的中位数；正数代表 first 更慢。
- 配对差值的中位数不一定等于两个 variant 各自中位数之差。
- 95% CI 使用 2,000 次 bootstrap，固定种子 2026；少于两个 trial 时 CI 为 null。
- 没有统一的“超过 1% 就成立”门槛，应结合配对分布、CI、重复运行与设备状态。

`hbm_read_bytes`、`l2_read_hit_rate`、`concurrency_verified` 和 `event_envelope_overlap_us` 当前默认留空。空白表示未观测，不能当成 0。

## 11. 结合你已运行的 E0 结果

你此前运行的是 sizes=16/42/84 MiB、interference=64 MiB、cores=48、trials=30。以下为该次粘贴结果中 target=A2 的中位数（单位 μs），不是本机新测数据：

| A 大小 | immediate | interference | time_gap |
|---|---:|---:|---:|
| 16 MiB | 12.45 | 10.31 | 10.43 |
| 42 MiB | 16.66 | 16.83 | 16.73 |
| 84 MiB | 25.55 | 25.74 | 25.72 |

42/84 MiB 下三组接近，16 MiB 的 immediate 反而更慢且波动较大。它们没有呈现明确的“D 让 A2 大幅变慢，而等时计算不影响 A2”的模式。

因此这次结果支持的结论是：**在这组规模和 64 MiB 干扰下，尚未观察到明显的 A2 重读延迟惩罚。** 不能据此宣布 L2 没有淘汰，也不能反推准确缓存容量。下一步应看 stages 的 A1/D/gap/A2、配对差值和更大的干扰量。

这组日志只覆盖 E0，不能代替 E1 的提前加载收益或 E3 的 KV 并发干扰结论。

## 12. 源码入口与验证范围

| 文件 | 负责内容 |
|---|---|
| [moe_bench.py](../moe_bench.py) | CLI、实验图、参数组合、间隔校准、随机试验、统计输出 |
| [probe_backend.py](../probe_backend.py) | ctypes 接口、图准备与执行、资源管理 |
| [probe_api.h](../csrc/probe_api.h) | C ABI 定义 |
| [memory_probe.cpp（Host）](../csrc/ops/memory_probe/op_host/memory_probe.cpp) | 缓冲/索引/参考值、事件、stream、GEMM executor、启动与校验 |
| [memory_probe.cpp（kernel）](../csrc/ops/memory_probe/op_kernel/memory_probe.cpp) | 请求分核、输入队列、累加与校验输出 |
| [compute_probe.cpp](../csrc/ops/memory_probe/op_kernel/compute_probe.cpp) | E0/E2 的 UB 计算间隔 |

本地 CPU 测试可验证图构造、分核公式、索引/窗口逻辑、错误路径和资源管理。它们用替代实现模拟 ACL/AscendC，不验证真实 CANN ABI、设备指令、队列流水、缓存、并发或性能。

目标设备还需要完成 CANN 编译、NPU 正确性测试与系统 profiling。尤其 E1/E3，必须确认时间线上实际发生重叠后，才适合将性能变化解释为并发执行下的收益或干扰。实际模型的路由、INT8/BF16 三矩阵专家、缓存驻留/释放策略与物理请求资源预留，当前尚未实现。
