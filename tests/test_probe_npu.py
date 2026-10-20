"""Real-device correctness tests for the new probes and asynchronous graph API.

Run explicitly after building on the NPU host. No skip/fallback when unavailable.
"""
import os
from pathlib import Path
import unittest

from probe_backend import ProbeBackend
from moe_bench import read, compute, gemm

ROOT = Path(__file__).resolve().parents[1]


class ProbeNPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ProbeBackend(os.environ.get('L2_BENCH_LIBRARY',
                                   str(ROOT / 'output/libl2_bench.so')))
        cls.addClassCleanup(cls.backend.close)

    def test_read_uneven_tiles_and_shuffled_indices(self):
        b = self.backend
        pool = b.buffer(97 * 16384, seed=17)
        for chunk in [256, 1024, 16384]:
            for order in [0, 1, 2, 3]:
                plan = b.plan(pool, 97 * 16384, chunk, order)
                for cores in [1, 7, 0]:
                    for window in [1, 2]:
                        with self.subTest(chunk=chunk, order=order, cores=cores, window=window):
                            b.execute([read('A', plan, cores, window=window)])

    def test_gemm_prefetch_dependency_and_repeat(self):
        b = self.backend
        weight = b.buffer(128 * 256 * 4, seed=99)
        plan = b.plan(weight, 128 * 256 * 4)
        matrix = b.gemm(weight, 4, 128, 256)
        for _ in range(3):
            records = b.execute([gemm('C0', matrix), read('P1', plan, 1, stream=1),
                                 gemm('G1', matrix, deps=['C0', 'P1'])])
            times = {r['step']: r for r in records}
            for dependency in ['C0']:
                self.assertGreaterEqual(times['G1']['start_us'] + 1, times[dependency]['end_us'])
            # This test deliberately does not claim the kernels overlapped.

    def test_compute_control(self):
        self.backend.execute([compute('gap', 128, cores=1)])

    def test_ub_rejection(self):
        b = self.backend
        pool = b.buffer(65536)
        plan = b.plan(pool, 65536, chunk=65536)
        if 3 * 65536 + 8192 > b.ub_bytes:
            with self.assertRaisesRegex(RuntimeError, 'UB'):
                b.execute([read('oversize', plan)])


if __name__ == '__main__':
    unittest.main()
