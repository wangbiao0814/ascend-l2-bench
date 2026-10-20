"""Execute real Host/kernel control code against a CPU CANN test double.

These tests do not validate CANN headers, AI Core instructions, cache policy,
physical overlap, or performance. Run test_probe_npu.py on the target as well.
"""
import ctypes as C
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from probe_backend import ProbeBackend
from moe_bench import cases, paired_interval, read, compute, gemm

ROOT = Path(__file__).resolve().parents[1]


class ProbeCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++')
        if not compiler:
            raise unittest.SkipTest('C++ compiler not installed')
        cls.temp = tempfile.TemporaryDirectory(prefix='probe-cpu-test-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.library = Path(cls.temp.name) / 'libprobe.so'
        subprocess.run([compiler, '-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                        '-I' + str(ROOT / 'tests/stubs'), '-I' + str(ROOT / 'csrc'),
                        str(ROOT / 'csrc/ops/memory_probe/op_host/memory_probe.cpp'),
                        str(ROOT / 'csrc/ops/memory_probe/op_kernel/memory_probe.cpp'),
                        str(ROOT / 'csrc/ops/memory_probe/op_kernel/compute_probe.cpp'),
                        str(ROOT / 'tests/stubs/inspect.cpp'), '-o', str(cls.library)], check=True)
        cls.inspector = C.CDLL(str(cls.library))
        cls.inspector.probe_stub_resources.argtypes = []
        cls.inspector.probe_stub_resources.restype = C.c_int

    def tearDown(self):
        self.assertEqual(self.inspector.probe_stub_resources(), 0, 'native resource leak')

    def test_read_all_lanes_uneven_ownership_and_windows(self):
        with ProbeBackend(self.library) as backend:
            for chunk in [256, 1024, 16384]:
                buffer = backend.buffer(chunk * 97, seed=123)
                for order in range(4):
                    plan = backend.plan(buffer, chunk * 97, chunk, order, 42)
                    for cores in [1, 7, 48]:
                        for window in [1, 2]:
                            records = backend.execute([read('A', plan, cores, window=window)])
                            self.assertTrue(records[0]['correctness_passed'])

    def test_gemm_repeated_trials_and_dependencies(self):
        with ProbeBackend(self.library) as backend:
            buffer = backend.buffer(16 * 64 * 4)
            matrix = backend.gemm(buffer, 4, 16, 64)
            plan = backend.plan(buffer, 16 * 64 * 4, 256)
            for _ in range(3):
                records = backend.execute([gemm('C0', matrix),
                    read('P1', plan, 1, stream=1), gemm('G1', matrix, deps=['C0', 'P1'])])
                times = {r['step']: r for r in records}
                self.assertGreaterEqual(times['G1']['start_us'], times['C0']['end_us'] - .001)
                self.assertGreaterEqual(times['G1']['start_us'], times['P1']['end_us'] - .001)

    def test_compute_uses_all_output_and_rounds(self):
        with ProbeBackend(self.library) as backend:
            for rounds in [0, 3, 33]:
                backend.execute([compute('C', rounds, cores=3)])

    def test_corruption_is_detected(self):
        with ProbeBackend(self.library) as backend:
            buffer = backend.buffer(16384)
            plan = backend.plan(buffer, 16384)
            os.environ['PROBE_STUB_CORRUPT'] = '1'
            try:
                with self.assertRaisesRegex(RuntimeError, 'checksum'):
                    backend.execute([read('bad', plan)])
            finally:
                del os.environ['PROBE_STUB_CORRUPT']

    def test_launch_failure_and_cleanup(self):
        with ProbeBackend(self.library) as backend:
            buffer = backend.buffer(16384)
            plan = backend.plan(buffer, 16384)
            os.environ['PROBE_STUB_FAIL'] = '1'
            try:
                with self.assertRaisesRegex(RuntimeError, '123'):
                    backend.execute([read('failed', plan)])
            finally:
                del os.environ['PROBE_STUB_FAIL']

    def test_dependency_validation_does_not_poison_next_graph(self):
        with ProbeBackend(self.library) as backend:
            weight = backend.buffer(4096)
            matrix = backend.gemm(weight, 2, 16, 64)
            with self.assertRaisesRegex(ValueError, 'earlier'):
                backend.execute([gemm('G', matrix, deps=['future'])])
            backend.execute([gemm('G', matrix)])

    def test_prepared_gemm_cleanup_after_other_launch_fails(self):
        with ProbeBackend(self.library) as backend:
            weight = backend.buffer(4096)
            matrix = backend.gemm(weight, 2, 16, 64)
            plan = backend.plan(weight, 4096, chunk=256)
            os.environ['PROBE_STUB_FAIL'] = '1'
            try:
                with self.assertRaisesRegex(RuntimeError, '123'):
                    backend.execute([read('failed', plan), gemm('not_started', matrix)])
            finally:
                del os.environ['PROBE_STUB_FAIL']

    def test_native_argument_validation(self):
        with ProbeBackend(self.library) as backend:
            buffer = backend.buffer(16384)
            with self.assertRaisesRegex(RuntimeError, 'chunk'):
                backend.plan(buffer, 16384, chunk=129)
            with self.assertRaisesRegex(RuntimeError, 'whole chunks'):
                backend.plan(buffer, 16383)
            plan = backend.plan(buffer, 16384)
            with self.assertRaisesRegex(RuntimeError, 'cores'):
                backend.execute([read('too_many', plan, cores=49)])
            with self.assertRaisesRegex(RuntimeError, 'window'):
                backend.execute([read('too_deep', plan, window=3)])

    def test_end_to_end_all_experiments(self):
        output = Path(self.temp.name) / 'runner'
        subprocess.run([sys.executable, str(ROOT / 'moe_bench.py'), '--experiment', 'all',
                        '--library', str(self.library), '--sizes-mib', '1', '--interference-mib', '1',
                        '--condition-mib', '1', '--kv-mib', '1', '--kv-chunk-bytes', '256',
                        '--tile-kib', '1', '--matrix-k', '16', '--matrix-n', '64',
                        '--compute-m', '4', '--compute-k', '16', '--compute-n', '64',
                        '--tokens', '2', '--compute-repeats', '2', '--cores', '2',
                        '--kv-cores', '1', '--warmup', '0', '--trials', '2',
                        '--output', str(output)], check=True, capture_output=True, text=True)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual({s['experiment'] for s in summary}, {'e0', 'e1', 'e2', 'e3'})
        stages = [json.loads(line) for line in (output / 'stages.jsonl').read_text().splitlines()]
        self.assertTrue(all(s['concurrency_verified'] is None for s in stages))
        self.assertTrue(all(v['correctness_passed'] for s in summary for v in s['variants'].values()))
        # Two stream envelopes can overlap in the stub, but are never marked as
        # verified hardware concurrency in machine-readable results.
        self.assertTrue((output / 'trials.csv').exists())


class GraphTests(unittest.TestCase):
    def test_e2_same_work_multiset(self):
        specs = cases('e2')
        self.assertEqual(sorted(s['resource'] for s in specs[0]['steps']),
                         sorted(s['resource'] for s in specs[1]['steps']))
        self.assertEqual([s['name'] for s in specs[0]['steps']], ['A1', 'B', 'C', 'A2'])
        self.assertEqual([s['name'] for s in specs[1]['steps']], ['A1', 'A2', 'B', 'C'])

    def test_e1_no_accidental_shared_steps(self):
        specs = cases('e1', compute_repeats=2)
        specs[0]['steps'][0]['resource'] = 99
        self.assertEqual(specs[1]['steps'][0]['resource'], 'C0')
        self.assertEqual(specs[2]['steps'][-1]['deps'], ['C0_1', 'P1'])
        self.assertEqual([s['name'] for s in specs[2]['steps']], ['C0_0', 'P1', 'C0_1', 'G1'])

    def test_dry_run_variant_filter(self):
        result = subprocess.run([sys.executable, str(ROOT / 'moe_bench.py'), '--experiment', 'e2',
                                 '--variants', 'interleaved', '--sizes-mib', '16',
                                 '--interference-mib', '64', '--dry-run'],
                                check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        self.assertEqual([v['variant'] for v in data['suites'][0]['variants']], ['interleaved'])

    def test_paired_interval_direction(self):
        lo, hi = paired_interval([2, 2, 2])
        self.assertEqual((lo, hi), (2, 2))
        self.assertEqual(paired_interval([2]), [None, None])


if __name__ == '__main__':
    unittest.main()
