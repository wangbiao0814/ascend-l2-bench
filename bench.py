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
                   default=ROOT/'output/libl2_bench.so')
    p.add_argument('--csv', type=Path, default=ROOT/'results.csv')
    p.add_argument('--dry-run', action='store_true', help='print plans without CANN/NPU')
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

    from acl_backend import Backend
    with Backend(a.library, a.device) as backend:
        meta = dict(device_id=a.device, available_aiv_cores=backend.cores,
                    ub_bytes=backend.ub_bytes, requested_cores=a.cores,
                    timing='ACL NPU events around one launch', evict_mib=a.evict_mib,
                    note='GB/s is logical bytes / elapsed seconds / 1e9, not measured HBM traffic',
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        a.csv.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, ensure_ascii=False))

        eviction = None
        if a.evict_mib and not a.profile_only:
            eviction_case = plan(a.evict_mib, a.tile_kib, 1, 0, 1)
            eviction = backend.allocate(eviction_case['input_bytes'], seed=17)

        rows = []
        # Allocate each size once; all strategies use identical input/output addresses.
        for size in a.sizes_mib:
            subset = [c for c in cases if c['requested_mib'] == size]
            buffer = backend.allocate(subset[0]['input_bytes'])
            random.Random(2026 + size).shuffle(subset)
            for case in subset:
                if a.profile_only:
                    backend.launch(buffer, case, a.cores)
                    print('Profile target:', json.dumps(case))
                    return
                # Poison output so a missing write cannot pass using a previous result.
                backend.poison(buffer)
                backend.launch(buffer, case, a.cores)
                backend.verify(buffer)
                for _ in range(a.warmup):
                    backend.launch(buffer, case, a.cores)
                times = []
                for _ in range(a.trials):
                    if eviction is not None:
                        backend.launch(eviction, eviction_case, a.cores)
                    elapsed = backend.launch(buffer, case, a.cores, timed=True)
                    if not math.isfinite(elapsed) or elapsed <= 0:
                        raise RuntimeError('invalid ACL event duration; use a larger input/repeat count')
                    times.append(elapsed)
                median = statistics.median(times)
                row = dict(case, cores_requested=a.cores,
                           cores_used=min(a.cores or backend.cores, case['tiles']), median_ms=median,
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
            backend.release(buffer)
    print(f'Results: {a.csv.resolve()}')


if __name__ == '__main__':
    main()
