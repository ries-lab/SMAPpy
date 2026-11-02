"""Where the computation runs, and in which precision.

One code path serves three systems: PyTorch chooses the backend.  The
forward model runs in float32 (complex64) on the device; Apple's GPU (MPS)
has no float64, so the small linear algebra of the solver is moved to the
CPU in float64 (`to_cpu64`).  Gradient checks run in float64 on the CPU.
"""
from __future__ import annotations

import ctypes
import os
import warnings

import torch

# torch's own use of the deprecated `torch.jit.script` (torch 2.14.1), which needs nothing from the user: its
# forward-mode decompositions call it when the first JVP imports them (`params` derivatives), and the compiler
# again at its first call.  Here, at import, because the first JVP comes before any compile.
warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)


def default_device() -> torch.device:
    """CUDA, then Apple MPS, then the CPU.  ``UIPSF_DEVICE=cpu`` (or ``cuda:1`` …) overrides."""
    requested = os.environ.get("UIPSF_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve(device=None, dtype: torch.dtype = torch.float32) -> tuple[torch.device, torch.dtype]:
    """The device to use, refusing float64 where the hardware has none (MPS)."""
    device = default_device() if device is None else torch.device(device)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"real dtype expected (float32 or float64), got {dtype}")
    if device.type == "mps" and dtype == torch.float64:
        raise ValueError("Apple's GPU (MPS) has no float64; use float32 there, or the CPU")
    return device, dtype


def complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """The complex type matching a real one."""
    return {torch.float32: torch.complex64, torch.float64: torch.complex128}[dtype]


def to_cpu64(t: torch.Tensor) -> torch.Tensor:
    """A copy on the CPU in double precision, for the solver's small systems.

    Two steps on purpose: from MPS, ``.to("cpu", torch.float64)`` in one call returns wrong values
    (torch 2.14.1, M1 Pro: every element of a [6, 1, 41, 42, 42] tensor, off by up to its range;
    tests/uipsf_tests/test_gpu.py)."""
    t = t.detach().to("cpu")
    return t.to(torch.complex128) if t.is_complex() else t.to(torch.float64)


_COMPILED: dict = {}


def _quiet_compiler() -> None:
    """Silence a line the compiler prints on Apple's GPU and that needs nothing from the user: inductor's
    "Not enough SMs to use max_autotune_gemm mode" (a CUDA count it applies to every GPU).  It comes at the
    first call, not here, so the level stays.  torch's `torch.jit.script` warning is filtered at import."""
    import logging
    logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)


def compiled(fn, device):
    """``fn`` through `torch.compile` on Apple's GPU, ``fn`` itself elsewhere; ``UIPSF_COMPILE=0`` turns
    it off, ``=1`` on for every device.

    Measured on an M1 Pro (the pupil's Jacobian columns, 6 beads): 1.7× faster on MPS after 4 s of
    compiling (1.7 s from the warm cache); on the CPU 1.2× after 12 s, and it needs a C++ compiler,
    so not there by default.  CUDA is unmeasured."""
    flag = os.environ.get("UIPSF_COMPILE")
    if flag == "0" or (flag != "1" and torch.device(device).type != "mps"):
        return fn
    if fn not in _COMPILED:
        _quiet_compiler()
        _COMPILED[fn] = torch.compile(fn, dynamic=False)
    return _COMPILED[fn]


def on_model_device(batch, model):
    """``batch`` on the device ``model`` was built on: the fastest one by default (`default_device`).

    Readers cut the stacks on the CPU and the solver works where the batch is, so without this a fit
    from `extract_rois` ran on the CPU beside a model on the GPU (45 s against 113 s for Zenodo's
    84 beads on an M1 Pro, plan §9 M7).  A no-op when they already agree."""
    device = model.psf.grid.rho.device
    return batch if batch.data.device == device else batch.to(device)



_AVAILABLE: list = [None, -1e9]          # (bytes, when): one answer per few seconds, `available_memory_bytes`
AVAILABLE_SECONDS = 5.0


def available_memory_bytes() -> int | None:
    """`_available_memory`, asked at most once per `AVAILABLE_SECONDS`: a `Problem` is made per fit and per
    localisation, and on macOS asking runs ``vm_stat``."""
    import time
    now = time.monotonic()
    if now - _AVAILABLE[1] > AVAILABLE_SECONDS:
        _AVAILABLE[:] = [_available_memory(), now]
    return _AVAILABLE[0]


def _available_memory() -> int | None:
    """The memory a new allocation can have without swapping, bytes; None if the system does not say.

    psutil's ``available`` where it is installed (it is no dependency); else the system's own count:
    ``MemAvailable`` of /proc/meminfo on Linux (free plus what the page cache gives back -- ``MemFree`` alone
    is near zero on a machine that has been reading files), ``GlobalMemoryStatusEx``'s ``ullAvailPhys`` on
    Windows, and on macOS, whose free count is small by design (the rest is cache it reclaims), the pages
    free, inactive, speculative and purgeable that ``vm_stat`` counts (half the physical memory if it cannot
    be run)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:      # noqa: BLE001 -- not installed, or a sandbox that hides the counters
        pass
    try:
        if os.name == "nt":
            return _windows_available()
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return _macos_available() or int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) // 2
    except Exception:      # noqa: BLE001 -- any failure means "unknown", never an error in a fit
        return None
    return None


class _MemoryStatusEx(ctypes.Structure):
    """Windows' MEMORYSTATUSEX, 64 bytes, for `GlobalMemoryStatusEx`."""
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),   # DWORDs: 4 bytes on every system
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _windows_available() -> int | None:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPhys)


def _macos_available() -> int | None:
    """The reclaimable pages ``vm_stat`` counts, bytes (macOS only); None elsewhere."""
    import subprocess
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return _vm_stat_available(out, int(os.sysconf("SC_PAGE_SIZE")))


def _vm_stat_available(out: str, page: int) -> int | None:
    """Bytes in the pages ``vm_stat``'s output ``out`` counts as free, inactive, speculative or purgeable."""
    pages = {}
    for line in out.splitlines()[1:]:
        name, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if value.isdigit():
            pages[name.strip()] = int(value)
    if "Pages free" not in pages:
        return None
    kinds = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    return sum(pages.get(k, 0) for k in kinds) * page
