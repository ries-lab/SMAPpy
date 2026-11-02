"""The solver's chunks are capped by the memory the system has available, on every platform
(`device.available_memory_bytes`, `solve.problem._host_chunk_values`; docs/uipsf/field-dependence.md §9)."""
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
    per_bead = batch.data[0].numel()
    # room for two beads' worth of values at the solver's bytes per value, times the share
    from uipsf.solve import problem
    room = int(2.5 * per_bead * problem.HOST_BYTES_PER_VALUE / problem.HOST_SHARE)
    monkeypatch.setattr(device, "available_memory_bytes", lambda: room)
    small = Problem(model, Poisson(), batch)
    assert small.chunk_size == 2
    assert Problem(model, Poisson(), batch, chunk_size=6).chunk_size == 2     # a caller's chunk too
    a, b = whole.normal_equations(truth), small.normal_equations(truth)
    assert torch.allclose(a.H_ss, b.H_ss, rtol=1e-9) and torch.allclose(a.g_s, b.g_s, rtol=1e-9)


def test_an_unknown_amount_of_memory_leaves_the_chunks_as_before(monkeypatch):
    model, _, batch = small_beads(OPTICS, n=6)
    monkeypatch.setattr(device, "available_memory_bytes", lambda: None)
    assert Problem(model, Poisson(), batch).chunk_size == 6
