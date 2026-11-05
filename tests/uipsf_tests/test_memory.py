"""The solver's chunks are capped by the memory the system has available, on every platform, at what one
emitter is measured to hold (`device.available_memory_bytes`, `uipsf.memory`, `solve.problem._measured_bytes`;
docs/uipsf/field-dependence.md §9)."""
import ctypes
import sys

import pytest
import torch

import uipsf
from uipsf import device
from uipsf.noise import Poisson
from uipsf.solve import Problem
from uipsf_tests.conftest import small_beads

OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.518, n_med=1.33, pixel_size_nm=(110, 110))


def test_this_system_says_how_much_memory_is_available():
    device._AVAILABLE[:] = [None, -1e9]
    free = device.available_memory_bytes()
    assert free is not None and 1e8 < free < 1e14


def test_the_windows_structure_has_the_size_globalmemorystatusex_expects():
    assert ctypes.sizeof(device._MemoryStatusEx) == 64


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_windows_reports_its_available_physical_memory():
    assert device._windows_available() > 1e8


def test_vm_stat_counts_free_inactive_speculative_and_purgeable_pages():
    out = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
           "Pages free:                               10.\nPages active:                            100.\n"
           "Pages inactive:                            20.\nPages speculative:                          3.\n"
           "Pages wired down:                          50.\nPages purgeable:                            7.\n")
    assert device._vm_stat_available(out, 16384) == 40 * 16384
    assert device._vm_stat_available("nothing here\n", 16384) is None


def test_little_memory_splits_the_beads_into_chunks_and_changes_no_sum(monkeypatch):
    model, truth, batch = small_beads(OPTICS, n=6)
    whole = Problem(model, Poisson(), batch)
    assert whole.chunk_size == 6
    from uipsf.solve import problem
    # one bead measured at the rough rule's worst, and room for two and a half of them
    each = batch.data[0].numel() * problem.HOST_BYTES_ROUGH
    monkeypatch.setattr(problem, "_measured_bytes", lambda *a: each / problem.MEASURED_MARGIN)
    monkeypatch.setattr(device, "available_memory_bytes", lambda: int(2.5 * each / problem.HOST_SHARE))
    small = Problem(model, Poisson(), batch)
    assert small.chunk_size == 2 and small.peak_bytes == 2 * each
    assert Problem(model, Poisson(), batch, chunk_size=6).chunk_size == 2     # a caller's chunk too
    a, b = whole.normal_equations(truth), small.normal_equations(truth)
    assert torch.allclose(a.H_ss, b.H_ss, rtol=1e-9) and torch.allclose(a.g_s, b.g_s, rtol=1e-9)


def test_where_nothing_can_be_measured_the_chunk_follows_the_bytes_per_value(monkeypatch):
    model, _, batch = small_beads(OPTICS, n=6)
    from uipsf.solve import problem
    per_bead = batch.data[0].numel()
    monkeypatch.setattr(problem, "_measured_bytes", lambda *a: None)
    monkeypatch.setattr(device, "available_memory_bytes",
                        lambda: int(2.5 * per_bead * problem.HOST_BYTES_PER_VALUE / problem.HOST_SHARE))
    small = Problem(model, Poisson(), batch)
    assert small.chunk_size == 2 and small.peak_bytes is None


def test_the_estimated_peak_of_a_chunk_is_what_it_holds_measured(monkeypatch):
    """The estimate a chunk is sized by -- one emitter measured, times the chunk and `MEASURED_MARGIN` --
    against a chunk's normal equations measured: never below it, and not far above (an M1 Pro's CPU held 0.87
    to 0.97 of it)."""
    from uipsf.memory import measure
    from uipsf.solve import problem
    model, truth, batch = small_beads(OPTICS, n=6, P=48, roi=(15, 15), K=21, dtype=torch.float32)
    Problem(model, Poisson(), batch.subset(slice(0, 1)), measure=False).normal_equations(truth.subset(slice(0, 1)))
    each = problem._measured_bytes(model, Poisson(), batch, torch.device("cpu"), 16)
    if not each:
        pytest.skip("this system cannot measure a peak (uipsf.memory)")
    each = int(problem.MEASURED_MARGIN * each)
    monkeypatch.setattr(device, "available_memory_bytes", lambda: int(3.5 * each / problem.HOST_SHARE))
    chunked = Problem(model, Poisson(), batch, tangents=16)
    assert chunked.chunk_size == 3 and chunked.peak_bytes == 3 * each
    _, held = measure(lambda: chunked.normal_equations(truth), "cpu")
    assert 0.6 * chunked.peak_bytes < held <= chunked.peak_bytes


def test_an_unknown_amount_of_memory_leaves_the_chunks_as_before(monkeypatch):
    model, _, batch = small_beads(OPTICS, n=6)
    monkeypatch.setattr(device, "available_memory_bytes", lambda: None)
    assert Problem(model, Poisson(), batch).chunk_size == 6
