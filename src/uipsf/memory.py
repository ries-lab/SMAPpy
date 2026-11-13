"""The peak memory of a piece of work, measured: what a chunk of emitters costs, so that a `Problem` can size its
chunks to the memory the system has free (`solve.problem`).

A rule in bytes per pixel value was off by a factor of four between models: 16 kB per value on 4 x86 cores
for the field-dependent model of 27² regions (November 2026), 50 to 60 kB on Apple's GPU for M6's default
model of 100 slices of 27², whose pupil grid of 160² per slice is what costs (19 beads took 32 GB and swapped).
What one emitter costs depends on the pupil grid, the slices, the vectorial fields and the device, so it is
measured on the model itself: one emitter's work, its peak above the start.

The peak is the system's own count where there is one -- no sampling thread, which misses short peaks:

- CUDA: torch's allocator, `max_memory_allocated` after `reset_peak_memory_stats`;
- Apple's GPU: what Metal's driver holds after the work, its cache emptied before, less what was in use
  before (`measure`);
- macOS' CPU: the process's physical footprint since `proc_reset_footprint_interval` (libproc);
- Linux: ``VmHWM`` of /proc/self/status after writing 5 to /proc/self/clear_refs (the high-water mark reset).

On the CPU the allocator's free pages are handed back to the system first (`malloc_zone_pressure_relief`,
glibc's `malloc_trim`): pages it kept from earlier work are reused without the footprint growing, and a second
measurement of the same work read 0 where the first read 83 MB.  Released first, one emitter of a small model
read 214 MB three times in three, and chunks of 2, 3 and 6 held 0.94 to 0.97 times that per emitter.

Where the system keeps no peak that can be reset -- Windows, and a Linux container that forbids writing to
clear_refs -- the process's current use is sampled every millisecond on a thread while the work runs
(`_SamplingMeter`: Windows' private bytes, Linux's ``VmRSS``).  A sample can miss a peak shorter than that,
which `solve.problem.MEASURED_MARGIN` covers; torch's operations release the GIL, so the thread samples
during them.  Before it, Windows measured nothing and a rule of 16 kB per value chunked M2's NA 1.43 beads
about right and NA 1.50's up to ten times too large.

Where even that fails `measure` returns None, and the caller falls back to a rule.
"""
from __future__ import annotations

import ctypes
import os
import sys

import torch


def measure(fn, device) -> tuple[object, int | None]:
    """(``fn()``, the most memory it held above what was in use when it started, bytes), the bytes None where
    this system cannot say (module docstring).  ``device`` is where ``fn`` computes."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        out = fn()
        torch.cuda.synchronize(device)
        return out, max(0, torch.cuda.max_memory_allocated(device) - start)
    if device.type == "mps":
        # Metal's memory is not always counted when it is taken (the footprint missed whole GBs that the driver
        # had held before), so the allocator's own count: emptied of its cache, it keeps every block the work
        # freed, and what the driver holds after it, less what was in use before, bounds what the work held at
        # once.  Not less what the driver held before: emptying keeps a heap that still holds a live tensor,
        # and the work fits partly into its free space -- after the window's estimates one M6 bead measured
        # 1.0 GB that way where a chunk took 2.1 GB a bead, and 11 beads at once took the Mac past 35 GB.
        torch.mps.synchronize()
        torch.mps.empty_cache()
        start = torch.mps.current_allocated_memory()
        out = fn()
        torch.mps.synchronize()
        peak = torch.mps.driver_allocated_memory() - start
        torch.mps.empty_cache()
        return out, max(0, peak)
    meter = _METER
    if meter is None:
        return fn(), None
    meter.release()
    start = meter.reset()
    if start is None:
        return fn(), None
    out = fn()
    peak = meter.peak()
    return out, None if peak is None else max(0, peak - start)


class _MacMeter:
    """The process's physical footprint and its peak since the last reset (rusage_info_v4, libproc)."""
    FLAVOR = 4                     # RUSAGE_INFO_V4
    FOOTPRINT, INTERVAL_MAX = 7, 33  # the uint64 fields after the 16-byte uuid

    class _Info(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16), ("v", ctypes.c_uint64 * 64)]   # v4 has 35; room to spare

    def __init__(self):
        self.lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        self.lib.proc_reset_footprint_interval        # noqa: B018 -- AttributeError where it is missing
        self.libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        self.libc.malloc_zone_pressure_relief.restype = ctypes.c_size_t

    def release(self):
        self.libc.malloc_zone_pressure_relief(None, 0)      # every zone, as much as it can

    def _fields(self):
        info = self._Info()
        if self.lib.proc_pid_rusage(os.getpid(), self.FLAVOR, ctypes.byref(info)) != 0:
            return None
        return info.v

    def reset(self) -> int | None:
        if self.lib.proc_reset_footprint_interval(os.getpid()) != 0:
            return None
        v = self._fields()
        return None if v is None else int(v[self.FOOTPRINT])

    def peak(self) -> int | None:
        v = self._fields()
        return None if v is None else int(v[self.INTERVAL_MAX])


class _LinuxMeter:
    """The resident set and its high-water mark since writing 5 to clear_refs (``VmRSS``, ``VmHWM``)."""

    def __init__(self):
        try:
            self.trim = ctypes.CDLL(None).malloc_trim       # glibc's; musl has none
        except (OSError, AttributeError):
            self.trim = None

    def release(self):
        if self.trim is not None:
            self.trim(0)

    def _status(self, key):
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
        return None

    def reset(self) -> int | None:
        try:
            with open("/proc/self/clear_refs", "w") as f:
                f.write("5")
            return self._status("VmRSS")
        except OSError:              # a container that forbids it
            return None

    def peak(self) -> int | None:
        return self._status("VmHWM")


class _SamplingMeter:
    """The process's current use (``current``, bytes or None), sampled every `SAMPLE_SECONDS` on a thread
    between `reset` and `peak`: for a system whose peak cannot be reset (module docstring)."""
    SAMPLE_SECONDS = 0.001

    def __init__(self, current, release=None):
        self.current, self._release = current, release
        self._thread, self._max = None, None

    def release(self):
        if self._release is not None:
            self._release()

    def reset(self) -> int | None:
        import threading
        start = self.current()
        if start is None:
            return None
        self._max, self._stop = start, threading.Event()

        def sample():
            while not self._stop.is_set():
                now = self.current()
                if now is not None and now > self._max:
                    self._max = now
                self._stop.wait(self.SAMPLE_SECONDS)
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return start

    def peak(self) -> int | None:
        if self._thread is None:
            return None
        now = self.current()            # the end too: work shorter than one sample
        self._stop.set()
        self._thread.join()
        self._thread = None
        return max(self._max, now) if now is not None else self._max


class _CountersEx(ctypes.Structure):
    """Windows' PROCESS_MEMORY_COUNTERS_EX, for `K32GetProcessMemoryInfo`."""
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
        (name, ctypes.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
            "PrivateUsage")]


def _windows_private_bytes():
    """The process's private bytes now (what it has committed, swapped or not), or None."""
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    counters = _CountersEx()
    counters.cb = ctypes.sizeof(_CountersEx)
    if not kernel32.K32GetProcessMemoryInfo(ctypes.c_void_p(kernel32.GetCurrentProcess()),
                                            ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PrivateUsage)


def _windows_heapmin():
    """Hand the C runtime's freed heap pages back (``_heapmin``), so that work reusing them is counted."""
    try:
        ctypes.cdll.ucrtbase._heapmin()
    except (OSError, AttributeError):
        pass


def _linux_rss():
    return _LinuxMeter()._status("VmRSS")


def _meter():
    try:
        if sys.platform == "darwin":
            return _MacMeter()
        if sys.platform == "win32":
            return _SamplingMeter(_windows_private_bytes, _windows_heapmin) if _windows_private_bytes() else None
        if sys.platform.startswith("linux"):
            linux = _LinuxMeter()
            if linux.trim is None:              # musl: unreleased pages would be under-counted
                return None
            if os.path.exists("/proc/self/clear_refs") and linux.reset() is not None:
                return linux
            return _SamplingMeter(_linux_rss, linux.release)    # a container that forbids clear_refs
    except (OSError, AttributeError):
        return None
    return None


_METER = _meter()
