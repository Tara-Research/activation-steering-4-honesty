import torch


class CudaTimer:
    """Context manager that times the enclosed GPU work using cuda.Event.

    Synchronizes on enter and exit so the measured interval reflects actual GPU work,
    not the asynchronous kernel-launch return. Falls back to wall-clock timing on CPU.
    """

    def __init__(self):
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            import time
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = float(self._start.elapsed_time(self._end))
        else:
            import time
            self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        return False


def reset_peak_memory_all_devices() -> None:
    if not torch.cuda.is_available():
        return
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)


def peak_memory_gb_all_devices() -> float:
    if not torch.cuda.is_available():
        return 0.0
    total_bytes = sum(
        torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())
    )
    return total_bytes / 1e9
