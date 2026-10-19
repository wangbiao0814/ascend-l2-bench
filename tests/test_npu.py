"""Full-output correctness through ACL; requires a built library and a real NPU."""
import unittest
from acl_backend import Backend


class NPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fail visibly if CANN, the native library, or the device is unavailable.
        cls.backend = Backend()
        cls.addClassCleanup(cls.backend.close)

    def test_all_strategies(self):
        for tiles in [1, 3, 97]:
            buffer = self.backend.allocate(tiles * 4096 * 4)
            try:
                for cores in [1, 0]:
                    for repeats in [1, 3]:
                        for group in [1, 8, 0]:
                            for cache in [0, 1]:
                                with self.subTest(tiles=tiles, cores=cores, repeats=repeats,
                                                  group=group, cache=cache):
                                    case = dict(repeats=repeats, tile_elements=4096,
                                                group_tiles=group, cache=cache)
                                    self.backend.poison(buffer)
                                    self.backend.launch(buffer, case, cores)
                                    self.backend.verify(buffer, full=True)
            finally:
                self.backend.release(buffer)

    def test_invalid_arguments(self):
        buffer = self.backend.allocate(4096 * 4)
        try:
            base = dict(repeats=2, tile_elements=4096, group_tiles=1, cache=1)
            for key, value in [('repeats', 0), ('tile_elements', 129),
                               ('tile_elements', 384), ('group_tiles', -1), ('cache', 2)]:
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    self.backend.launch(buffer, dict(base, **{key: value}), cores=1)
            with self.assertRaises(RuntimeError):
                self.backend.launch(buffer, base, cores=self.backend.cores + 1)
        finally:
            self.backend.release(buffer)

    def test_events_and_poison(self):
        buffer = self.backend.allocate(1024 * 1024)
        try:
            with self.assertRaises(RuntimeError):
                self.backend.verify(buffer, full=True)
            case = dict(repeats=8, tile_elements=4096, group_tiles=1, cache=1)
            self.assertGreater(self.backend.launch(buffer, case, timed=True), 0)
            self.backend.verify(buffer, full=True)
        finally:
            self.backend.release(buffer)


if __name__ == '__main__':
    unittest.main()
