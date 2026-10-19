# Ascend C：L2 重用距离与带宽对照实验

工程状态：独立 Ascend C kernel + ACL Host，提供构建脚本、计时脚本与精度测试。
构建和运行均不依赖 PyTorch、torch_npu、pybind11 或任何第三方 Python 包。
本地为 Mac，没有 CANN/NPU；**尚未完成目标设备编译和上板实测，没有预填性能数据**。
面向支持 `SetL2CacheHint` 的 Atlas A2/A3；当前目标为 910C、CANN 9.0.0。
其他产品及不同 CANN 版本需确认接口、编译目标和 profiling 支持情况。

## 实验做了什么

固定同一份 FP32 输入 x 和独立输出 y。每个元素均执行 R 次：

```text
从 x 的 GM 地址加载到 UB -> 加 1 -> 写入 y 的 GM 地址
```

每次都重新搬入输入。重复次数只改变工作量，最终输出始终 `y=x+1`，不是 `x+R`。
所有策略的输入元素、读写请求字节数、加法次数完全一致；只改变访问顺序与输入缓存策略。
输出统一关闭 L2，尽量减少输出缓存分配的干扰，仍然保留真实的输出写流量。

| group_tiles | 核内执行顺序（R=2） | 预期 |
|---|---|---|
| 1 | A,A,B,B,C,C… | 重用距离短，开 L2 时容易命中 |
| 64 | 先扫描 64 个 tile，再重复扫描这组 | 适中的活跃工作集 |
| 512 | 同样方式，一组 512 个 tile | 可在多核下使工作集跨过 L2 容量 |
| 0 | A,B,C…全部扫完，再从 A 开始 | 大输入时重用距离长，容易被替换 |

每种顺序都分别测试输入 L2 NORMAL 和 DISABLE。并不是提前给 L2 指定一个命中率。
相同 GM 地址可能从 L2 或 HBM 提供数据；源代码不会直接分配 L2 地址。

## 构建与运行

在有昇腾 NPU 的 Linux 主机上，加载与驱动配套的 CANN 开发环境。
构建需要 CMake >= 3.20、C++17 编译器和 CANN 开发工具；运行脚本只需 Python 3 标准库。
Host 直接调用 ACL，Python 通过 `ctypes` 调用 `output/libl2_bench.so`。

```bash
cd ascend-l2-bench
# 按你的实际安装位置加载 CANN set_env.sh。
# 确认 ASCEND_HOME_PATH 已设置。
# 910C/A3 先查询：npu-smi info -t board -i 0 -c 0
# 按实际输出拼接 Chip Name_NPU Name；例如输出 Ascend910 和 9391 时：
bash build.sh Ascend910_9391

# 直接加载编译出的 output/libl2_bench.so。
python3 -m unittest discover -s tests -p test_npu.py -v
python3 bench.py
```

`build.sh` 的第一个参数是完整 SoC 编译标识（传给 CMake 的 `SOC_VERSION`）。
上面的型号只是查询结果示例，请以本机输出为准。
构建目录为 `build-acl/`，与旧构建目录隔离；切换 SoC/CANN 时先移走 `build-acl/`。
`build.sh` 使用已加载的 `ASCEND_HOME_PATH`，不安装依赖。
构建产物为 `output/libl2_bench.so`，无需安装。
若库位于其他位置，使用 `--library /absolute/path/to/libl2_bench.so`。

建议先只跑一个 512 MiB 的四组顺序对照：

```bash
python3 bench.py --sizes-mib 512 --repeats 8 \
    --groups 1 64 512 0 --cache both --trials 7
```

完整默认扫描输入 16、64、256、512 MiB。默认 tile=16 KiB、R=8、
自动使用可用 AIV 核；可用 `--cores` 固定参与核数。
每个输入规模只分配一次 x/y，全部顺序使用同一份输入；以固定随机顺序运行各对照。
较大的输入需要足够 NPU 内存，默认最大输入和输出合计约 1 GiB。
输入和输出是分别分配的 FP32 连续设备缓冲区，由 ACL 管理。

## 怎样读结果

输入在 CPU 上由固定种子的确定性函数生成，再通过 ACL 拷入设备。
每种配置先把输出填为 NaN，再做一次计时外的采样正确性检查、预热及重复计时；
独立的 `test_npu.py` 会对小规模案例做全输出检查。
CSV 保存中位数、最短/最长耗时、实际核数和逻辑字节数，旁边的 meta.json 保存设备编号、AIV 核数、UB 容量与参数。
结果由本机运行产生，本工程没有提供模拟的性能数字。

设 N 为实际输入字节数，R 为重复次数，t 为算子耗时（秒）：

```text
逻辑读字节 = N*R
逻辑写字节 = N*R
有效读带宽 = N*R/t/1e9 GB/s
有效读写带宽 = 2*N*R/t/1e9 GB/s
```

这些是完整算子的有效吞吐，不是独立测得的 L2、HBM 或 MTE2 峰值带宽。
计时同时包含加载、加法、写回、同步及控制开销。
开启 L2 后，部分逻辑读取不再产生 HBM 读取，所以有效读带宽可能超过 HBM 读带宽。
输出仍反复写回，可能成为瓶颈；因此高命中率不保证耗时同比下降。
在简化的读写共用 HBM 带宽模型中，R=8 时，即使读侧理想复用，
总 HBM 流量也只是从 16N 降到约 9N；不能预期整个算子快 8 倍。

ACL NPU Event 包围一次启动，短 kernel 的结果可能混入主机提交空隙。
优先使用足够大的 N/R，并用 profiler 的 kernel duration 复核。
这些队列使用双缓冲，但本实验没有声称达到硬件极限带宽。

预期关注：

1. 输入较小时，scan 也可能命中得很好；这是容量效应的正常现象。
2. 输入显著大于 L2 时，L2 开启的 group=1 与 scan 更容易拉开差距。
3. 比较相同顺序下 L2 on/off，帮助隔离缓存的影响。
4. L2 off 时如果不同顺序仍明显不同，说明还有调度、HBM 访问局部性、写入等因素。
5. 想看容量拐点，可扫输入尺寸，或改变 group。近似组工作集为
   `min(N, 活跃核数 * group_tiles * tile_bytes)`；跨核无全局屏障，实际进度会偏离估算。

默认测试预热后的连续运行；不保证首轮读取是冷缓存。
`--evict-mib 768` 可在每次计时前读取一块独立大输入作为尽力驱逐，
这不是硬件 cache flush，也不保证完全清空 L2。驱逐过程不计入目标时间。
会额外占用两倍该值的设备内存。profiling 时应遵循工具自身的 L2 清理规则。

## 实测 L2 命中率和流量

**不要从耗时反推真实命中率。** 使用设备支持的 msprof op / msopprof 指标。
以下为官方 `msprof op` 入口示例；若所装版本使用 `msopprof`，请先看该版本 `--help`。

```bash
# 每次只指定一个策略，profile-only 跳过预热/验证/事件计时，目标算子只启动一次。
msprof op --aic-metrics=L2Cache,Memory \
    --output=./prof_group1_on \
    --application="python3 bench.py --sizes-mib 512 --repeats 8 --groups 1 --cache on --profile-only"

msprof op --aic-metrics=L2Cache,Memory \
    --output=./prof_scan_on \
    --application="python3 bench.py --sizes-mib 512 --repeats 8 --groups 0 --cache on --profile-only"

msprof op --aic-metrics=L2Cache,Memory \
    --output=./prof_group1_off \
    --application="python3 bench.py --sizes-mib 512 --repeats 8 --groups 1 --cache off --profile-only"
```

profiling 时只筛选 `l2_load_bench`，不要把初始化、内存传输等任务算入目标耗时。
查看 L2Cache 和 Memory 报告，结合目标 kernel duration 比较。
如工具支持分开的读/写命中率，优先读命中率；不要把聚合命中率直接当输入命中率。
检查 MTE2/GM->UB 流量、HBM/main-memory read 与 write 流量的区别。
部分产品不提供全部指标，部分指标会通过重放多轮采集；不要把 profiling 耗时
与无 profiler 的事件耗时混为一谈。现代 msopprof 的 kernel/range 模式会清理 L2，
application 模式不清理；本实验重复读取放在一个 kernel 内，因此内部复用仍可观察。
DISABLE 情况下计数器可能显示零请求或不适用，不能机械解读为 0% 命中。

理想化估算：若首轮全冷且所有组内数据都能留在 L2，R=8 时输入读命中率约 7/8。
该值只是机制说明，不是设备实测，也不保证 profiler 会报告这个数。

## 本地检查与文件

```bash
# 无 NPU 也能执行：
python3 -m unittest discover -s tests -p test_schedule.py -v
python3 bench.py --dry-run --sizes-mib 512 --groups 1 0
```

- `csrc/ops/l2_load_bench/op_kernel/l2_load_bench.cpp`：GM/UB 搬运、计算与分组调度。
- `csrc/ops/l2_load_bench/op_host/l2_load_bench.cpp`：ACL 初始化、内存管理、参数校验、硬件查询、启动、事件计时与正确性检查。
- `acl_backend.py`：标准库 ctypes 封装。
- `csrc/ops/l2_load_bench/design.md`：接口、UB 布局、工作量与实验边界。
- `tests/test_schedule.py`：调度覆盖、每 tile 访问次数、写入核唯一性、字节统计。
- `tests/test_npu.py`：目标设备全输出正确性、参数校验、输出 NaN 检测与事件计时检查。

## 官方依据

- [DMA 单元与默认 L2 路径](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)
- [SetL2CacheHint](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1alpha001/API/ascendcopapi/atlasascendc_api_07_00033.html)
- [msprof op 采集接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/devaids/optool/atlasopdev_16_0090.html)
- [Ascend/msopprof 用户指南与 cache 清理行为](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)
- [A3 SoC 编译型号查询](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/atctool/atlasatcparam_16_0036.html)
- [ACL 事件计时示例](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/acldevg/runtime_doc_dev_0023.html)
