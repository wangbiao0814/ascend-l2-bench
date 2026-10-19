"""Standard-library bindings for the standalone ACL benchmark library."""
import ctypes as C
from pathlib import Path

DEFAULT_LIBRARY = Path(__file__).resolve().parent / 'output/libl2_bench.so'


class Backend:
    """Own one ACL session and its buffers; use as a context manager."""
    def __init__(self, library=DEFAULT_LIBRARY, device=0):
        self.lib = C.CDLL(str(Path(library).resolve()))
        signatures = {
            'l2_last_error': ([], C.c_char_p),
            'l2_open': ([C.c_int, C.POINTER(C.c_void_p)], C.c_int),
            'l2_info': ([C.c_void_p, C.POINTER(C.c_int64), C.POINTER(C.c_uint64)], C.c_int),
            'l2_alloc': ([C.c_void_p, C.c_size_t, C.c_uint32, C.POINTER(C.c_void_p)], C.c_int),
            'l2_poison': ([C.c_void_p, C.c_void_p], C.c_int),
            'l2_run': ([C.c_void_p, C.c_void_p] + [C.c_int64] * 5 +
                       [C.c_int, C.POINTER(C.c_float)], C.c_int),
            'l2_verify': ([C.c_void_p, C.c_void_p, C.c_int], C.c_int),
            'l2_free': ([C.c_void_p, C.c_void_p], C.c_int),
            'l2_close': ([C.c_void_p], C.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        self.handle = C.c_void_p()
        self.buffers = []
        self.check(self.lib.l2_open(device, C.byref(self.handle)))
        try:
            cores, ub = C.c_int64(), C.c_uint64()
            self.check(self.lib.l2_info(self.handle, C.byref(cores), C.byref(ub)))
            self.cores, self.ub_bytes = cores.value, ub.value
        except BaseException:
            self.close()
            raise

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.l2_last_error().decode('utf-8', errors='replace'))

    def allocate(self, size, seed=2026):
        if size <= 0:
            raise ValueError('buffer size must be positive')
        buffer = C.c_void_p()
        self.check(self.lib.l2_alloc(self.handle, size, seed, C.byref(buffer)))
        self.buffers.append(buffer)
        return buffer

    def release(self, buffer):
        self.check(self.lib.l2_free(self.handle, buffer))
        self.buffers.remove(buffer)

    def poison(self, buffer):
        self.check(self.lib.l2_poison(self.handle, buffer))

    def launch(self, buffer, case, cores=0, timed=False):
        elapsed = C.c_float()
        self.check(self.lib.l2_run(self.handle, buffer, case['repeats'],
                   case['tile_elements'], case['group_tiles'], case['cache'],
                   cores, int(timed), C.byref(elapsed)))
        return elapsed.value

    def verify(self, buffer, full=False):
        self.check(self.lib.l2_verify(self.handle, buffer, int(full)))

    def close(self):
        if self.handle.value:
            # Attempt every cleanup even if a previous operation failed.
            for buffer in reversed(self.buffers):
                self.lib.l2_free(self.handle, buffer)
            self.buffers.clear()
            self.lib.l2_close(self.handle)
            self.handle = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
