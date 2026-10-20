#!/usr/bin/env python3
"""Controlled HBM/cache experiments on Ascend, without third-party Python packages."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parent
MIB = 1024 ** 2


def read(name, resource, cores=0, stream=0, deps=(), cache=1, window=2):
    return dict(name=name, kind='read', resource=resource, cores=cores,
                stream=stream, deps=list(deps), cache=cache, window=window)


def compute(name, rounds, cores=0, stream=0, deps=()):
    return dict(name=name, kind='compute', rounds=rounds, cores=cores,
                stream=stream, deps=list(deps))


def gemm(name, resource, stream=0, deps=()):
    return dict(name=name, kind='gemm', resource=resource, stream=stream, deps=list(deps))


def cases(experiment, cache=1, cores=0, kv_cores=12, limit_cores=1,
          gap_rounds=4096, compute_repeats=1):
    """Abstract graph specifications, also used by dry-run and host-only tests."""
    r = lambda name, resource: read(name, resource, cores, cache=cache)
    if experiment == 'e0':
        return [
            dict(variant='immediate', target='A2', steps=[r('A1', 'A'), r('A2', 'A')]),
            dict(variant='interference', target='A2', steps=[r('A1', 'A'), r('D', 'D'), r('A2', 'A')]),
            dict(variant='time_gap', target='A2', steps=[r('A1', 'A'),
                 compute('gap', gap_rounds, cores), r('A2', 'A')]),
        ]
    if experiment == 'e2':
        return [
            dict(variant='interleaved', target='A2', steps=[r('A1', 'A'), r('B', 'B'), r('C', 'C'), r('A2', 'A')]),
            dict(variant='clustered', target='A2', steps=[r('A1', 'A'), r('A2', 'A'), r('B', 'B'), r('C', 'C')]),
            dict(variant='compute_gap_control', target='A2', steps=[r('A1', 'A'),
                 compute('gap', gap_rounds, cores), r('A2', 'A')]),
        ]
    if experiment == 'e1':
        def prefix():
            return [gemm('C0_' + str(i), 'C0') for i in range(compute_repeats)]
        last = 'C0_' + str(compute_repeats - 1)
        return [
            dict(variant='demand', target='G1', steps=prefix() + [gemm('G1', 'G1', deps=[last])]),
            dict(variant='serial_prefetch', target='G1', steps=prefix() +
                 [r('P1', 'W'), gemm('G1', 'G1', deps=[last, 'P1'])]),
            dict(variant='overlap_prefetch', target='G1', steps=prefix()[:1] +
                 [read('P1', 'W', kv_cores, stream=1, cache=cache)] + prefix()[1:] +
                 [gemm('G1', 'G1', deps=[last, 'P1'])]),
        ]
    if experiment == 'e3':
        variants = [dict(variant='alone', target='F', steps=[r('F', 'A')]),
                    dict(variant='background_only_large', target='K', steps=[
                         read('K', 'K_large', kv_cores, stream=1, cache=cache)]),
                    dict(variant='background_only_random', target='K', steps=[
                         read('K', 'K_random', kv_cores, stream=1, cache=cache)])]
        for name, plan, bg_cores, window in [
                ('large_linear', 'K_large', kv_cores, 2),
                ('small_indexed_linear', 'K_linear', kv_cores, 2),
                ('small_random', 'K_random', kv_cores, 2),
                ('limited_cores', 'K_random', limit_cores, 2),
                ('limited_window', 'K_random', kv_cores, 1),
                ('limited_random', 'K_random', limit_cores, 1),
                ('index_control', 'K_control', kv_cores, 2)]:
            variants.append(dict(variant=name, target='F', steps=[
                read('K', plan, bg_cores, stream=1, cache=cache, window=window), r('F', 'A')]))
        return variants
    raise ValueError('unknown experiment')


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def paired_interval(differences, seed=2026):
    """Bootstrap paired median difference; positive means first variant is slower."""
    if len(differences) < 2:
        return [None, None]
    rng = random.Random(seed)
    samples = [statistics.median(rng.choices(differences, k=len(differences))) for _ in range(2000)]
    return [percentile(samples, .025), percentile(samples, .975)]


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row['experiment'], row['weight_bytes'], row['interference_bytes'],
               row['kv_chunk_bytes'], row['cache_policy'])
        groups.setdefault(key, {}).setdefault(row['variant'], []).append(row)
    result = []
    for key, variants in groups.items():
        summaries = {}
        for name, observations in variants.items():
            target = [r['target_us'] for r in observations]
            total = [r['makespan_us'] for r in observations]
            summaries[name] = dict(trials=len(target), target_p50_us=statistics.median(target),
                                   target_p95_us=percentile(target, .95),
                                   makespan_p50_us=statistics.median(total),
                                   correctness_passed=all(r['correctness_passed'] for r in observations))
        pairs = []
        # Report effects, without turning unobserved counters into diagnoses.
        comparisons = {'e0': [('interference', 'immediate', 'target_us')],
                       'e2': [('interleaved', 'clustered', 'target_us'),
                              ('interleaved', 'clustered', 'makespan_us')],
                       'e1': [('demand', 'overlap_prefetch', 'makespan_us'),
                              ('serial_prefetch', 'overlap_prefetch', 'makespan_us')],
                       'e3': [('small_random', 'alone', 'target_us'),
                              ('small_random', 'small_indexed_linear', 'target_us'),
                              ('small_random', 'limited_cores', 'target_us'),
                              ('small_random', 'limited_window', 'target_us'),
                              ('small_random', 'limited_random', 'target_us')]}
        for first, second, metric in comparisons[key[0]]:
            if first not in variants or second not in variants:
                continue
            left = {r['trial']: r for r in variants[first]}
            right = {r['trial']: r for r in variants[second]}
            diffs = [left[t][metric] - right[t][metric] for t in sorted(left.keys() & right.keys())]
            pairs.append(dict(first=first, second=second, metric=metric,
                              paired_median_difference_us=statistics.median(diffs),
                              bootstrap_95_ci_us=paired_interval(diffs)))
        result.append(dict(experiment=key[0], weight_bytes=key[1], interference_bytes=key[2],
                           kv_chunk_bytes=key[3], cache_policy=key[4], variants=summaries,
                           paired_comparisons=pairs))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', choices=['e0', 'e1', 'e2', 'e3', 'all'], default='e0')
    p.add_argument('--sizes-mib', nargs='+', type=int, default=[16, 42, 84])
    p.add_argument('--interference-mib', nargs='+', type=int, default=[16, 64, 256])
    p.add_argument('--condition-mib', type=int, default=512,
                   help='common best-effort interference sweep before every variant, 0 disables')
    p.add_argument('--tile-kib', type=int, default=16)
    p.add_argument('--cores', type=int, default=0)
    p.add_argument('--cache', choices=['on', 'off', 'both'], default='on')
    p.add_argument('--gap-rounds', type=int, default=4096,
                   help='initial vector rounds for time-gap calibration')
    p.add_argument('--kv-mib', type=int, default=256,
                   help='KV pool and useful bytes are equal, so all variants touch the same address set')
    p.add_argument('--kv-chunk-bytes', nargs='+', type=int, default=[256, 1024, 4096, 16384])
    p.add_argument('--kv-cores', type=int, default=12)
    p.add_argument('--limit-cores', type=int, default=1)
    p.add_argument('--matrix-k', type=int, default=7168)
    p.add_argument('--matrix-n', type=int, default=2048)
    p.add_argument('--tokens', type=int, default=8)
    p.add_argument('--compute-m', type=int, default=512)
    p.add_argument('--compute-k', type=int, default=1024)
    p.add_argument('--compute-n', type=int, default=1024)
    p.add_argument('--compute-repeats', type=int, default=1)
    p.add_argument('--math-type', type=int, choices=[0, 1, 2, 3], default=1)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--trials', type=int, default=30)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--library', type=Path, default=ROOT / 'output/libl2_bench.so')
    p.add_argument('--output', type=Path, default=ROOT / 'results-moe')
    p.add_argument('--variants', nargs='+', help='run selected variant names, useful for profiling one trace')
    p.add_argument('--dry-run', action='store_true')
    return p


def validate(a, p):
    positive = a.sizes_mib + a.interference_mib + a.kv_chunk_bytes + [a.kv_mib, a.trials,
        a.kv_cores, a.limit_cores, a.tokens, a.matrix_k, a.matrix_n, a.compute_m,
        a.compute_k, a.compute_n, a.compute_repeats]
    if min(positive) <= 0 or a.condition_mib < 0 or a.cores < 0 or a.warmup < 0 or a.device < 0:
        p.error('sizes, counts and trials must be positive; condition/cores/warmup/device >= 0')
    if a.tile_kib < 1 or a.tile_kib > 64 or not 0 <= a.gap_rounds <= 1000000:
        p.error('tile-kib in [1,64], gap-rounds in [0,1000000] required')
    if a.limit_cores > a.kv_cores or a.compute_repeats > 1024:
        p.error('limit-cores <= kv-cores and compute-repeats <= 1024 required')
    for chunk in a.kv_chunk_bytes:
        if not 256 <= chunk <= 65536 or chunk % 256 or a.kv_mib * MIB % chunk:
            p.error('KV chunks must be multiples of 256 in [256,65536] dividing KV bytes')
    if any(size * MIB % (a.tile_kib * 1024) for size in a.sizes_mib + a.interference_mib):
        p.error('sizes and interference bytes must contain whole tiles')
    if a.condition_mib * MIB % (a.tile_kib * 1024):
        p.error('conditioning bytes must contain whole tiles')
    if a.matrix_k * a.matrix_n * 4 % (a.tile_kib * 1024):
        p.error('G1 matrix bytes must contain whole read-probe tiles')


def parameters(a):
    experiments = ['e0', 'e2', 'e1', 'e3'] if a.experiment == 'all' else [a.experiment]
    caches = [1, 0] if a.cache == 'both' else [int(a.cache == 'on')]
    for experiment in experiments:
        sizes = [a.matrix_k * a.matrix_n * 4] if experiment == 'e1' else [v * MIB for v in a.sizes_mib]
        interference = [v * MIB for v in a.interference_mib] if experiment in ['e0', 'e2'] else [0]
        chunks = a.kv_chunk_bytes if experiment == 'e3' else [0]
        for size, other, chunk, cache in itertools.product(sizes, interference, chunks, caches):
            yield dict(experiment=experiment, weight_bytes=size, interference_bytes=other,
                       kv_chunk_bytes=chunk, cache_policy=cache)


def calibrate_gap(backend, target_steps, a):
    def measure(steps):
        samples = [backend.execute(steps) for _ in range(3)]
        return statistics.median(max(s['end_us'] for s in sample) -
                                 min(s['start_us'] for s in sample) for sample in samples)
    target = measure(target_steps)
    rounds = max(a.gap_rounds, 1)
    observed = 0.0
    for attempt in range(5):
        observed = measure([compute('gap', rounds, a.cores)])
        if observed <= 0:
            raise RuntimeError('invalid calibration event duration')
        if abs(observed - target) <= max(target * .15, 2):
            break
        adjusted = max(0, min(1000000, round(rounds * target / observed)))
        if adjusted == rounds or attempt == 4:
            break
        rounds = adjusted
    return rounds, dict(target_us=target, observed_us=observed,
                        matched=abs(observed - target) <= max(target * .15, 2))


def execute_suite(a, params, writer, stage_file, rows):
    from probe_backend import ProbeBackend
    with ProbeBackend(a.library, a.device) as backend:
        if max(a.cores, a.kv_cores, a.limit_cores) > backend.cores:
            raise ValueError('requested cores exceed available AIV cores')
        chunk = a.tile_kib * 1024
        cache = params['cache_policy']
        resources = {}
        def weight(name, byte_count, seed, order=0, request_chunk=chunk):
            buffer = backend.buffer(byte_count, seed)
            resources[name] = backend.plan(buffer, byte_count, request_chunk, order, a.seed)
            return buffer
        if params['experiment'] != 'e1':
            weight('A', params['weight_bytes'], a.seed)
        condition = None
        if a.condition_mib:
            weight('condition', a.condition_mib * MIB, a.seed + 99)
            condition = resources['condition']
        experiment = params['experiment']
        calibration = None
        gap_rounds = a.gap_rounds
        if experiment in ['e0', 'e2']:
            if experiment == 'e0':
                weight('D', params['interference_bytes'], a.seed + 1)
            if experiment == 'e2':
                # Split exactly the same interference byte budget across B/C.
                half = params['interference_bytes'] // (2 * chunk) * chunk
                if half == 0:
                    raise ValueError('interference budget too small for B/C')
                weight('B', half, a.seed + 2)
                weight('C', params['interference_bytes'] - half, a.seed + 3)
            target_steps = ([read('B', resources['B'], a.cores, cache=cache),
                             read('C', resources['C'], a.cores, cache=cache)] if experiment == 'e2'
                            else [read('D', resources['D'], a.cores, cache=cache)])
            gap_rounds, calibration = calibrate_gap(backend, target_steps, a)
        if experiment == 'e1':
            w = weight('W', params['weight_bytes'], a.seed + 4)
            c0w = backend.buffer(a.compute_k * a.compute_n * 4, a.seed + 5)
            resources['C0'] = backend.gemm(c0w, a.compute_m, a.compute_k, a.compute_n, a.math_type)
            resources['G1'] = backend.gemm(w, a.tokens, a.matrix_k, a.matrix_n, a.math_type)
        if experiment == 'e3':
            kv = backend.buffer(a.kv_mib * MIB, a.seed + 6)
            for name, size, order in [('K_large', chunk, 0),
                                      ('K_linear', params['kv_chunk_bytes'], 1),
                                      ('K_random', params['kv_chunk_bytes'], 2),
                                      ('K_control', params['kv_chunk_bytes'], 3)]:
                resources[name] = backend.plan(kv, a.kv_mib * MIB, size, order, a.seed)
        specs = cases(experiment, cache, a.cores, a.kv_cores, a.limit_cores,
                      gap_rounds, a.compute_repeats)
        if a.variants:
            specs = [s for s in specs if s['variant'] in a.variants]
            if not specs:
                raise ValueError('no matching variants for ' + experiment)
        # Translate only read/GEMM resources; compute nodes have no resource.
        for spec in specs:
            for step in spec['steps']:
                if 'resource' in step:
                    step['resource'] = resources[step['resource']]
        rng = random.Random(a.seed)
        for trial in range(-a.warmup, a.trials):
            order = list(specs)
            rng.shuffle(order)
            for spec in order:
                if condition is not None:
                    backend.execute([read('condition', condition, a.cores, cache=1)])
                stages = backend.execute(spec['steps'])
                if trial < 0:
                    continue
                target = next(s for s in stages if s['step'] == spec['target'])
                if any(not math.isfinite(s['duration_us']) or s['duration_us'] <= 0 for s in stages):
                    raise RuntimeError('invalid device event duration')
                start = min(s['start_us'] for s in stages)
                end = max(s['end_us'] for s in stages)
                foreground = [s for s in stages if s['stream'] == 0]
                background = [s for s in stages if s['stream'] == 1]
                row = dict(params, variant=spec['variant'], trial=trial, target_step=spec['target'],
                           target_us=target['duration_us'], makespan_us=backend.last_makespan_us,
                           stage_span_us=(end - start) if not background or not foreground else '',
                           background_us=sum(s['duration_us'] for s in background),
                           background_end_us=max((s['end_us'] for s in background), default=0),
                           event_envelope_overlap_us='',
                           available_aiv_cores=backend.cores, ub_bytes=backend.ub_bytes,
                           cores_requested=a.cores, kv_cores=a.kv_cores,
                           gap_rounds=gap_rounds,
                           gap_calibration_matched=calibration['matched'] if calibration else '',
                           index_bytes=(a.kv_mib * MIB // params['kv_chunk_bytes'] * 8
                                        if experiment == 'e3' and spec['variant'] not in ['alone', 'large_linear', 'background_only_large'] else 0),
                           hbm_read_bytes='', l2_read_hit_rate='', concurrency_verified='',
                           correctness_passed=all(s['correctness_passed'] for s in stages))
                rows.append(row)
                writer.writerow(row)
                stage_file.write(json.dumps(dict(params, variant=spec['variant'], trial=trial,
                    stages=stages, graph=spec['steps'], calibration=calibration,
                    time_basis='separate stream-local epochs; do not overlay streams',
                    concurrency_verified=None)) + '\n')
                print(f"{experiment} {params['weight_bytes']/MIB:g} MiB {spec['variant']:23s} "
                      f"trial={trial:02d} target={row['target_us']:.3f} us total={row['makespan_us']:.3f} us",
                      flush=True)


def main():
    p = parser()
    a = p.parse_args()
    validate(a, p)
    suites = list(parameters(a))
    if a.dry_run:
        preview = []
        for suite in suites:
            specs = cases(suite['experiment'], suite['cache_policy'], a.cores,
                          a.kv_cores, a.limit_cores, a.gap_rounds, a.compute_repeats)
            if a.variants:
                specs = [s for s in specs if s['variant'] in a.variants]
                if not specs:
                    p.error('no matching variants for ' + suite['experiment'])
            preview.append(dict(suite, variants=specs))
        print(json.dumps(dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                              suites=preview), indent=2))
        return
    if any((a.output / name).exists() for name in ['meta.json', 'trials.csv', 'stages.jsonl', 'summary.json']):
        p.error('output already contains results; choose a new --output directory')
    a.output.mkdir(parents=True, exist_ok=True)
    meta = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                weight_dtype='FP32 synthetic values; GEMM math_type is explicit',
                cache_policy_scope='read probes only; ACLNN GEMM uses its own cache policy',
                repeats_per_expert_call=1, precision='events saved in microseconds',
                cache_state='best-effort interference prefix; not guaranteed cold',
                e1='oracle-known weights; event dependency, not complete MoE routing',
                counters='not collected by this runner; missing counters remain empty',
                concurrency='stream-local timestamps cannot establish concurrency; verify device timeline')
    (a.output / 'meta.json').write_text(json.dumps(meta, indent=2))
    rows = []
    columns = ['experiment', 'weight_bytes', 'interference_bytes', 'kv_chunk_bytes', 'cache_policy',
               'variant', 'trial', 'target_step', 'target_us', 'makespan_us', 'stage_span_us',
               'background_us', 'background_end_us', 'event_envelope_overlap_us',
               'available_aiv_cores', 'ub_bytes', 'cores_requested', 'kv_cores', 'gap_rounds',
               'gap_calibration_matched', 'index_bytes', 'hbm_read_bytes', 'l2_read_hit_rate',
               'concurrency_verified', 'correctness_passed']
    try:
        with (a.output / 'trials.csv').open('w', newline='') as raw, (a.output / 'stages.jsonl').open('w') as stages:
            writer = csv.DictWriter(raw, fieldnames=columns)
            writer.writeheader()
            for suite in suites:
                execute_suite(a, suite, writer, stages, rows)
                raw.flush(); stages.flush()
                (a.output / 'summary.json').write_text(json.dumps(summarize(rows), indent=2))
    finally:
        if rows:
            (a.output / 'summary.json').write_text(json.dumps(summarize(rows), indent=2))
    print(f'Results: {a.output.resolve()}')


if __name__ == '__main__':
    main()
