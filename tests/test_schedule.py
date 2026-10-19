"""Check that scheduling changes neither work nor tile ownership (no NPU needed)."""
import collections
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('bench', Path(__file__).parents[1]/'bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

class ScheduleTests(unittest.TestCase):
    def test_equal_work_and_no_cross_core_writers(self):
        for tiles in [1, 3, 17, 64, 129]:
            for cores in [1, 2, 8, 40]:
                for repeats in [1, 3]:
                    baseline = None
                    for group in [1, 2, 8, 0, 1000]:
                        visits = list(bench.schedule(tiles, cores, repeats, group))
                        counts = collections.Counter(tile for _, tile in visits)
                        self.assertEqual(counts, {i: repeats for i in range(tiles)})
                        owners = collections.defaultdict(set)
                        for core, tile in visits:
                            owners[tile].add(core)
                        self.assertTrue(all(len(o) == 1 for o in owners.values()))
                        work = collections.Counter(visits)
                        if baseline is None:
                            baseline = work
                        self.assertEqual(work, baseline)

    def test_orders_differ(self):
        immediate = list(bench.schedule(4, 1, 2, 1))
        scan = list(bench.schedule(4, 1, 2, 0))
        self.assertEqual([t for _, t in immediate], [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual([t for _, t in scan], [0, 1, 2, 3, 0, 1, 2, 3])

    def test_byte_accounting(self):
        p = bench.plan(1, 16, 8, 1, 1)
        self.assertEqual(p['logical_read_bytes'], 8 * 1024**2)
        self.assertEqual(p['logical_read_bytes'], p['logical_write_bytes'])
        self.assertEqual(p['fp32_adds'], 8 * 1024**2 // 4)

if __name__ == '__main__':
    unittest.main()
