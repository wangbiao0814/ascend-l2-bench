#!/usr/bin/env python3
"""Compare identical FP32 work with different GM reload reuse distances."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parent
MIB = 1024 ** 2


def plan(size_mib, tile_kib, repeats, group, cache):
    tile_bytes = tile_kib * 1024
    ntiles = math.ceil(size_mib * MIB / tile_bytes)
    size = ntiles * tile_bytes
    return dict(requested_mib=size_mib, input_bytes=size, tile_elements=tile_bytes // 4,
                tiles=ntiles, repeats=repeats, group_tiles=group, cache=cache,
                logical_read_bytes=size * repeats, logical_write_bytes=size * repeats,
                fp32_adds=(size // 4) * repeats)


def schedule(tiles, cores, repeats, group):
    """CPU oracle for kernel indexing; yields (core, global tile)."""
    cores = min(cores, tiles)
    for core in range(cores):
        local = (tiles + cores - 1 - core) // cores
        width = group or local
        for begin in range(0, local, width):
            for _ in range(repeats):
                for t in range(begin, min(begin + width, local)):
                    yield core, t * cores + core


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes-mib', nargs='+', type=int, default=[16, 64, 256, 512])
    p.add_argument('--tile-kib', type=int, default=16)
    p.add_argument('--repeats', type=int, default=8)
    p.add_argument('--groups', nargs='+', type=int, default=[1, 64, 512, 0],
                   help='tiles per core per group; 0 = full per-core scan')
    p.add_argument('--cache', choices=['on', 'off', 'both'], default='both')
    p.add_argument('--cores', type=int, default=0, help='0 = all available AIV cores')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--trials', type=int, default=7)
    p.add_argument('--evict-mib', type=int, default=0,
                   help='best-effort L2 eviction read sweep before each timed launch')
    p.add_argument('--library', type=Path,
                   default=ROOT/'python/ascend_kernel/ascend_kernel/lib/libascend_kernel.so')
    p.add_argument('--csv', type=Path, default=ROOT/'results.csv')
    p.add_argument('--dry-run', action='store_true', help='print plans without torch/NPU')
    p.add_argument('--profile-only', action='store_true',
                   help='one case, one custom kernel launch; no warmup/check/timing')
    return p


def main():
    p = parser()
    a = p.parse_args()
    if min(a.sizes_mib) <= 0 or a.tile_kib < 1 or a.tile_kib > 64:
        p.error('sizes must be positive; tile-kib must be in [1, 64]')
    if not 1 <= a.repeats <= 1024 or min(a.groups) < 0 or a.cores < 0:
        p.error('invalid repeats, groups, or cores')
    if a.trials < 1 or a.warmup < 0 or a.evict_mib < 0:
        p.error('trials >= 1; warmup and evict-mib >= 0 required')
    caches = [1, 0] if a.cache == 'both' else [int(a.cache == 'on')]
    cases = [plan(s, a.tile_kib, a.repeats, g, c)
             for s in a.sizes_mib for g in a.groups for c in caches]
    if a.profile_only and len(cases) != 1:
        p.error('--profile-only requires one size, one group, and --cache on or off')
    if a.dry_run:
        print(json.dumps(cases, indent=2))
        return

    import torch
    import torch_npu  # registers NPU backend
    if not torch.npu.is_available():
        raise RuntimeError('Ascend NPU is required')
    torch.npu.set_device(a.device)
    torch.ops.load_library(str(a.library.resolve()))
    op = torch.ops.npu.l2_load_bench
    device = f'npu:{a.device}'
    meta = dict(torch=torch.__version__, torch_npu=getattr(torch_npu, '__version__', ''),
                device=torch.npu.get_device_name(a.device), requested_cores=a.cores,
                timing='NPU event around one launch', evict_mib=a.evict_mib,
                note='GB/s is logical bytes / elapsed seconds / 1e9, not measured HBM traffic',
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    a.csv.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, ensure_ascii=False))

    ex = ey = None
    if a.evict_mib and not a.profile_only:
        e = plan(a.evict_mib, a.tile_kib, 1, 0, 1)
        ex = torch.randn(e['input_bytes'] // 4, device=device)
        ey = torch.empty_like(ex)

    def launch(x, y, case):
        op(x, y, case['repeats'], case['tile_elements'], case['group_tiles'],
           case['cache'], a.cores)

    rows = []
    # Allocate each size once; all strategies at that size use the same x and y.
    for size in a.sizes_mib:
        subset = [c for c in cases if c['requested_mib'] == size]
        n = subset[0]['input_bytes'] // 4
        x = torch.randn(n, dtype=torch.float32, device=device)
        y = torch.empty_like(x)
        torch.npu.synchronize()
        random.Random(2026 + size).shuffle(subset)
        for case in subset:
            if a.profile_only:
                launch(x, y, case)
                torch.npu.synchronize()
                print('Profile target:', json.dumps(case))
                return
            # Sample correctness before warmup, outside all timed regions.
            launch(x, y, case)
            torch.npu.synchronize()
            idx = torch.linspace(0, n - 1, min(n, 4096), device=device).long()
            torch.testing.assert_close(y[idx].cpu(), (x[idx] + 1).cpu(), rtol=1e-6, atol=1e-6)
            del idx
            for _ in range(a.warmup):
                launch(x, y, case)
            torch.npu.synchronize()
            times = []
            for _ in range(a.trials):
                if ex is not None:
                    # Read a disjoint region through L2. This is not a hardware flush.
                    op(ex, ey, 1, a.tile_kib * 256, 0, 1, a.cores)
                    torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                launch(x, y, case)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            median = statistics.median(times)
            row = dict(case, cores_requested=a.cores, median_ms=median,
                       min_ms=min(times), max_ms=max(times),
                       read_effective_GBs=case['logical_read_bytes'] / (median * 1e6),
                       rw_effective_GBs=(case['logical_read_bytes'] + case['logical_write_bytes']) / (median * 1e6))
            rows.append(row)
            label = 'scan' if case['group_tiles'] == 0 else f"group={case['group_tiles']}"
            print(f"{case['input_bytes']/MIB:7.1f} MiB {label:12s} L2={case['cache']} "
                  f"{median:9.3f} ms  read-effective={row['read_effective_GBs']:8.1f} GB/s "
                  f"rw-effective={row['rw_effective_GBs']:8.1f} GB/s")
            with a.csv.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        del x, y
    print(f'Results: {a.csv.resolve()}')


if __name__ == '__main__':
    main()
