"""CANN memory-probe graph bindings; Python standard library only."""
import ctypes as C
from pathlib import Path
from acl_backend import DEFAULT_LIBRARY


class ProbeBackend:
    """Prepare every allocation before executing a graph on two ACL streams."""
    def __init__(self, library=DEFAULT_LIBRARY, device=0):
        self.lib = C.CDLL(str(Path(library).resolve()))
        ptr, i, u = C.c_void_p, C.c_int64, C.c_uint64
        definitions = {
            'pm_open': [C.c_int, C.POINTER(ptr)],
            'pm_info': [ptr, C.POINTER(i), C.POINTER(u)],
            'pm_buffer': [ptr, u, C.c_uint32, C.POINTER(i)],
            'pm_plan': [ptr, i, u, i, i, C.c_uint32, C.POINTER(i)],
            'pm_gemm': [ptr] + [i] * 5 + [C.POINTER(i)],
            'pm_prepare_gemm': [ptr, i, i],
            'pm_reset': [ptr], 'pm_begin': [ptr], 'pm_finish': [ptr],
            'pm_read': [ptr] + [i] * 7 + [C.POINTER(i)],
            'pm_compute': [ptr] + [i] * 5 + [C.POINTER(i)],
            'pm_matmul': [ptr] + [i] * 4 + [C.POINTER(i)],
            'pm_times': [ptr, i, C.POINTER(C.c_float), C.POINTER(C.c_float)],
            'pm_total': [ptr, C.POINTER(C.c_float)],
            'pm_verify': [ptr, i], 'pm_close': [ptr],
        }
        self.lib.pm_error.argtypes = []
        self.lib.pm_error.restype = C.c_char_p
        for name, args in definitions.items():
            method = getattr(self.lib, name)
            method.argtypes, method.restype = args, C.c_int
        self.handle = ptr()
        self.check(self.lib.pm_open(device, C.byref(self.handle)))
        try:
            cores, ub = i(), u()
            self.check(self.lib.pm_info(self.handle, C.byref(cores), C.byref(ub)))
            self.cores, self.ub_bytes = cores.value, ub.value
        except BaseException:
            self.close()
            raise

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.pm_error().decode('utf-8', errors='replace'))

    def resource(self, method, *args):
        result = C.c_int64()
        self.check(getattr(self.lib, method)(self.handle, *args, C.byref(result)))
        return result.value

    def buffer(self, byte_count, seed=2026):
        if byte_count <= 0 or byte_count % 256:
            raise ValueError('buffer bytes must be positive and divisible by 256')
        return self.resource('pm_buffer', byte_count, seed)

    def plan(self, buffer, byte_count, chunk=16384, order=0, seed=2026):
        return self.resource('pm_plan', buffer, byte_count, chunk, order, seed)

    def gemm(self, weight, m, k, n, math_type=1):
        return self.resource('pm_gemm', weight, m, k, n, math_type)

    def execute(self, steps):
        """Execute a topologically ordered graph and verify it outside timing.

        A step has name/kind/resource/stream/deps plus read/compute parameters.
        Stage timestamps have stream-local origins and must NOT be overlaid.
        Hardware overlap MUST be confirmed separately with a device timeline.
        """
        if not steps or len({s['name'] for s in steps}) != len(steps):
            raise ValueError('graph must be nonempty with unique step names')
        seen = set()
        for step in steps:
            if step['kind'] not in ['read', 'compute', 'gemm'] or step.get('stream', 0) not in [0, 1]:
                raise ValueError('invalid step kind or stream')
            deps = step.get('deps', [])
            if len(deps) > 2 or any(d not in seen for d in deps):
                raise ValueError('dependencies must reference at most two earlier steps')
            seen.add(step['name'])
        self.check(self.lib.pm_reset(self.handle))
        counts = {}
        for step in steps:
            if step['kind'] == 'gemm':
                resource = step['resource']
                counts[resource] = counts.get(resource, 0) + 1
        for resource, count in counts.items():
            self.check(self.lib.pm_prepare_gemm(self.handle, resource, count))
        ids = {}
        for step in steps:
            dependencies = step.get('deps', [])
            if len(dependencies) > 2 or any(d not in ids for d in dependencies):
                raise ValueError('dependencies must reference at most two earlier steps')
            deps = [ids[d] for d in dependencies] + [-1] * (2 - len(dependencies))
            stream = step.get('stream', 0)
            if step['kind'] == 'read':
                job = self.resource('pm_read', step['resource'], stream,
                                    step.get('cores', 0), step.get('cache', 1),
                                    step.get('window', 2), *deps)
            elif step['kind'] == 'compute':
                job = self.resource('pm_compute', stream, step.get('cores', 0),
                                    step['rounds'], *deps)
            elif step['kind'] == 'gemm':
                job = self.resource('pm_matmul', step['resource'], stream, *deps)
            else:
                raise ValueError('unknown step kind')
            ids[step['name']] = job
        # Every output, event, expected checksum, GEMM workspace and executor is
        # now prepared. This call only enqueues event waits, launches and records.
        self.check(self.lib.pm_begin(self.handle))
        self.check(self.lib.pm_finish(self.handle))
        total = C.c_float()
        self.check(self.lib.pm_total(self.handle, C.byref(total)))
        self.last_makespan_us = total.value * 1000
        records = []
        for step in steps:
            start, end = C.c_float(), C.c_float()
            job = ids[step['name']]
            self.check(self.lib.pm_times(self.handle, job, C.byref(start), C.byref(end)))
            self.check(self.lib.pm_verify(self.handle, job))
            records.append(dict(step=step['name'], kind=step['kind'],
                                stream=step.get('stream', 0),
                                start_us=start.value * 1000, end_us=end.value * 1000,
                                duration_us=(end.value - start.value) * 1000,
                                clock_origin='stream_local_epoch',
                                correctness_passed=True))
        return records

    def close(self):
        if self.handle.value:
            self.lib.pm_close(self.handle)
            self.handle = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
