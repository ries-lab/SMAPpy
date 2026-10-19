"""Where the computation runs, and in which precision.

One code path serves three systems: PyTorch chooses the backend.  The
forward model runs in float32 (complex64) on the device; Apple's GPU (MPS)
has no float64, so the small linear algebra of the solver is moved to the
CPU in float64 (`to_cpu64`).  Gradient checks run in float64 on the CPU.
"""
from __future__ import annotations

import os

import torch


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
        _COMPILED[fn] = torch.compile(fn, dynamic=False)
    return _COMPILED[fn]


def on_model_device(batch, model):
    """``batch`` on the device ``model`` was built on: the fastest one by default (`default_device`).

    Readers cut the stacks on the CPU and the solver works where the batch is, so without this a fit
    from `extract_rois` ran on the CPU beside a model on the GPU (45 s against 113 s for Zenodo's
    84 beads on an M1 Pro, plan §9 M7).  A no-op when they already agree."""
    device = model.psf.grid.rho.device
    return batch if batch.data.device == device else batch.to(device)

