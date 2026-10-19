"""Full-output correctness and argument validation on an Ascend NPU."""
from pathlib import Path
import unittest
import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[1]

class NPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest('NPU unavailable')
        torch.npu.set_device(0)
        torch.ops.load_library(str(ROOT/'python/ascend_kernel/ascend_kernel/lib/libascend_kernel.so'))

    def test_all_strategies(self):
        # Odd tile counts exercise uneven per-core partitioning.
        for tiles in [1, 3, 97]:
            x = torch.randn(tiles * 4096, device='npu:0')
            reference = x.cpu() + 1
            for cores in [1, 0]:
                for repeats in [1, 3]:
                    for group in [1, 8, 0]:
                        for cache in [0, 1]:
                            y = torch.empty_like(x)
                            torch.ops.npu.l2_load_bench(x, y, repeats, 4096, group, cache, cores)
                            torch.testing.assert_close(y.cpu(), reference, rtol=1e-6, atol=1e-6)

    def test_alias_rejected(self):
        x = torch.randn(4096, device='npu:0')
        with self.assertRaises(RuntimeError):
            torch.ops.npu.l2_load_bench(x, x, 2, 4096, 1, 1, 1)

if __name__ == '__main__':
    unittest.main()
