# MoE 访存实验运行说明

## 实现范围

新增 `moe_bench.py`，使用现有 CANN/ACL 原生库，不依赖 torch 或额外 Python 包。原有 `bench.py` 和其 kernel 保持原义。

| 入口 | 已实现的对照 |
|---|---|
| e0 | A→A、A→D→A、A→纯计算→A；自动校准纯计算间隔 |
| e2 | A→B→C→A、A→A→B→C、纯计算间隔控制；相同 A 地址 |
| e1 | 按需 GEMM、串行预读后 GEMM、双 stream 提前预读后 GEMM |
| e3 | 前景/背景独立基线、大块顺序、小块索引顺序、小块随机、分别限制核数/窗口及同时限制、仅索引控制 |

读取探针逐 tile 消费所有 FP32 输入，每核写 32 B 校验结果，不写回一份等大的权重。所有读调用内部只扫描一遍；跨调用复用由任务图表达。

输入为 FP32 合成整数数据（每 256 B 一组固定小整数），用于全访问校验；不是 INT8 专家执行。E1 通过 ACLNN Matmul 调用真实矩阵乘法，固定输入 X 为全 1，输出与 CPU 参考比较。默认 G1 是 `[8,7168] × [7168,2048]`，对应一块 56 MiB FP32 矩阵；不是一个完整的三矩阵 MoE FFN。

E1 的 C0/G1 是事件依赖链；G1 的输入数据不由 C0 的输出生成。权重地址为 oracle 已知，不能把这个结果当作实际动态路由可提前获得的证明。`--cache` 只控制读探针，不能修改 ACLNN GEMM 的缓存策略。

## 编译与设备验证

加载你机器实际位置的 CANN 9.0.0 `set_env.sh`，确认 `ASCEND_HOME_PATH`，然后使用之前查询得到的完整 SoC 编译标识：

```bash
bash build.sh "$SOC_VERSION"
python3 -m unittest discover -s tests -p test_probe_npu.py -v
```

这次库新增 ACLNN Matmul，因此需要完整 CANN 开发环境及 `aclnnop/aclnn_matmul.h`、opapi、nnopbase。无需安装 Python 包。

本地 Mac 无 CANN/NPU，无法确认目标编译和设备并发；CPU 模拟测试不替代上述验证。

## 建议先运行的命令

先只看方案，不加载原生库：

```bash
python3 moe_bench.py --experiment all --sizes-mib 16 \
    --interference-mib 64 --kv-chunk-bytes 1024 --dry-run
```

第一步，基础校准。初次先用 5 次 trial 检查流程，正式采样使用 30 次：

```bash
python3 moe_bench.py --experiment e0 --sizes-mib 16 42 84 \
    --interference-mib 16 64 256 --cores 48 --cache both \
    --trials 30 --output results-moe-e0
```

第二步，跨调用专家复用：

```bash
python3 moe_bench.py --experiment e2 --sizes-mib 16 42 84 \
    --interference-mib 16 64 256 --cores 48 \
    --trials 30 --output results-moe-e2
```

`interference-mib` 是 B+C 的合计输入量；对齐到 tile 后拆成两份。间隔校准针对实际 B+C 图，不额外插入 D。同一模型专家等大时可设置 interference-mib=2×size-mib，例如 A=42 MiB、B=C=42 MiB 时使用 84。

第三步，提前读取与实际 GEMM：

```bash
python3 moe_bench.py --experiment e1 --matrix-k 7168 --matrix-n 2048 \
    --tokens 8 --compute-m 512 --compute-k 1024 --compute-n 1024 \
    --compute-repeats 1 --cores 48 --kv-cores 4 \
    --trials 30 --output results-moe-e1
```

- `--compute-*` 定义 C0 矩阵形状；`--compute-repeats` 增大前序计算窗口。
- `--kv-cores` 在 E1 中表示 P1 的预读核数；建议分别用 1、4、8、12 核。
- `--matrix-k/n` 和 `--tokens` 定义 G1；E1 不使用 `--sizes-mib` 决定权重大小。
- `--math-type 1` 允许 CANN 对 FP32 输入使用降精度的 Cube 路径；保存于 meta.json，其他值需确认目标支持。
- C0/G1 的 ACLNN executor 设为 repeatable，workspace 在计时前分配，复用资源只能在同一 stream 顺序调用，关闭 session 时显式销毁。

第四步，KV 干扰：

```bash
python3 moe_bench.py --experiment e3 --sizes-mib 28 42 \
    --kv-mib 256 --kv-chunk-bytes 256 1024 4096 16384 \
    --cores 4 --kv-cores 12 --limit-cores 1 \
    --trials 30 --output results-moe-e3
```

前景固定 4 核是一个便于观察的起点，并非物理核预留。需另扫前景核数 1/4/12/24/48。

- KV 池大小与一次有用扫描量相等；各模式访问同一地址集合。小块顺序与随机索引只改变同一索引集合的顺序。
- `large_linear` 使用 `--tile-kib` 的块大小；其余模式使用 `--kv-chunk-bytes`。
- `background_only_large/random` 是背景独立基线，target_step 为 K；其他 E3 对照的目标为 F。
- `limited_cores` 仅减少背景核数；`limited_window` 仅把软件窗口改为 1。
- `limited_random` 同时将背景 block 数设为 `--limit-cores`，输入队列软件窗口由 2 改为 1。这不是硬件请求槽预留。
- `index_control` 保留随机索引读取和每块的 UB 计算，但不读 KV 载荷。其结果与其他模式的校验参考不同，是明确的负对照，不是等工作量优化方案。
- 本轮未做物理 HBM 带宽自动匹配；不能仅凭这些结果确认请求槽饱和。

## 起始状态、正确性与时间口径

默认每个 variant 前扫描独立的 512 MiB 缓冲（`--condition-mib 512`），不计入目标图。它只是统一的尽力干扰前缀，不保证完全冷缓存。`--condition-mib 0` 关闭。

每次图执行前先完成事件、校验结果、GEMM executor/workspace 的准备；`pm_begin` 只提交 event wait、kernel/ACLNN 调用和 event record，不在提交中申请内存、传输初始化数据或做校验和参考计算。整图提交完之后才同步两个 stream，随后校验所有探针结果和 GEMM 输出。

每个 trial 区组随机打乱 variant 顺序，warmup 不进入结果统计。E0/E2 的纯计算间隔自动向 D 扫描（E0）或 B+C 合计（E2）的孤立耗时校准；`gap_calibration_matched=False` 时不能把该控制当成已经等时长。校准基于孤立测量，实际图中的间隔还要用 stages.jsonl 复核。

**各阶段时间是设备 stream event 的包络，不是 profiler 的精确 kernel 占用区间。** 每个 stream 使用自己的局部起点，只在同一个 stream 内做 event elapsed time；两个 stream 的 start/end 不可直接叠加判断并发。整段时间则由 stream 0 的公共起点和等待两 stream 后的 join 事件测量。事件包络仍可能包括主机提交间隙、内部算子及调度等待。

## 输出文件

每次运行使用不同的 `--output` 目录；程序拒绝覆盖已存在的结果文件。

| 文件 | 内容 |
|---|---|
| meta.json | 参数、合成精度、计时边界及解释限制 |
| trials.csv | 每个原始 trial：目标延迟、整图延迟、背景成本、核数、校验结果 |
| stages.jsonl | 每阶段事件时间、stream、完整任务图、依赖、间隔校准结果 |
| summary.json | p50/p95、配对中位数差值、bootstrap 95% CI |

- `target_step`：本行目标为 A2/G1/F 或背景独立基线 K。
- `target_us`：该目标阶段的事件包络延迟。
- `makespan_us`：从公共 epoch 到两 stream 中最后一个阶段完成，包含图提交等待。
- `stage_span_us`：仅对单 stream 图记录首尾阶段跨度；双 stream 图为空。
- `event_envelope_overlap_us`：保留为空，局部事件起点无法建立公共并发时间线。
- `background_end_us`：背景相对 stream 1 局部起点的结束时刻，配对比较背景代价；不与 stream 0 时间戳直接对齐。
- `hbm_read_bytes`、`l2_read_hit_rate`、`concurrency_verified`：默认空白，**没有采集就不填写**。

summary 的配对差值定义为 first−second，正数表示 first 更慢。少于两个 trial 时 CI 为 null；小样本仅检查流程，不用于稳定结论。脚本不自动宣告“L2 淘汰”或“请求槽满”。

## Profiling

先运行本机 `msprof --help` 与 `msprof op --help`，确认 CANN 9.0.0 支持的参数。系统级时间线用来验证实际并发；L2Cache/Memory/PipeUtilization 用来解释访存。

可先单独运行一个轨迹，再把它作为本机 profiling 工具的 application：

```bash
python3 moe_bench.py --experiment e2 --variants interleaved \
    --sizes-mib 42 --interference-mib 64 --warmup 0 --trials 1 \
    --output results-moe-profile-e2
```

不要把会清理 L2 的单 kernel/range 重放用于证明原始跨 kernel 复用；若本机工具支持 application replay，重放完整前缀与轨迹并检查目标 launch。E1/E3 的并发不能被采集过程串行化；不能确认时把并发状态保留为未知。stages.jsonl 的 step 是逻辑标签，不等于 profiler 的系统 task/launch ID；需结合 kernel 名称、顺序和形状定位目标。不能选出第二次 A 时，不要将聚合计数归属于 A2。

## 本地测试

```bash
python3 -m unittest discover -s tests -p test_schedule.py -v
python3 -m unittest discover -s tests -p test_probe_cpu.py -v
```

CPU 测试编译并执行真正的新增 Host/kernel C++ 控制代码，替换底层 ACL/AscendC 算子为测试实现。覆盖校验、非整除核分配、索引/窗口、事件依赖、矩阵乘法资源复用、错误和清理。模拟时钟/性能数据只写临时目录，不会写入真实实验结果。

尚需在目标设备确认：CANN 声明与编译、AI Core 队列/同步语义、缓存策略、实际并发、实际性能；真实 INT8/BF16 专家三矩阵和实际路由 trace 也尚未接入。
