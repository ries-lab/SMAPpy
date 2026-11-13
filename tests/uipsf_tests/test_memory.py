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


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple's GPU only")
def test_work_that_fits_into_a_half_used_heap_is_counted_on_apples_gpu():
    """Emptying Metal's cache keeps a heap that still holds a live tensor, and work fits into its free space
    without the driver growing: measured against the driver's start, 32 MB read 0 (and an M6 bead 1.0 GB
    instead of 2.1, so 11 beads went at once and took the Mac past 35 GB)."""
    from uipsf.memory import measure
    keep = [torch.ones(256 * 1024, device="mps") for _ in range(64)]          # 1 MB each, sharing heaps
    del keep[::2]

    def work():
        made = [torch.ones(256 * 1024, device="mps") for _ in range(32)]     # 32 MB
        torch.mps.synchronize()
        return len(made)
    _, held = measure(work, "mps")
    assert held >= 32 * 2 ** 20


def test_what_the_allocators_keep_after_a_run_is_handed_back():
    """After a calibration Apple's GPU kept 3-4 GB of freed blocks, counted as the program's, until the next
    run swapped (`device.release_memory`, called when the PSF calibration window's work ends)."""
    if not torch.backends.mps.is_available():
        device.release_memory()                  # nothing to hand back, and no error
        return
    x = torch.ones(64 * 2 ** 20 // 4, device="mps")
    del x
    torch.mps.synchronize()
    assert torch.mps.driver_allocated_memory() >= 64 * 2 ** 20
    device.release_memory()
    assert torch.mps.driver_allocated_memory() < 64 * 2 ** 20


def test_on_cuda_the_chunk_is_what_one_emitter_measured_on_the_card_leaves_room_for(monkeypatch):
    """The bytes-per-value rule was a small model's: one NA 1.50 bead costs up to 25 times it, and an NVIDIA
    card on Windows, full, moved the rest into system memory.  One emitter measured caps the chunk."""
    from uipsf.solve import problem
    model, _, batch = small_beads(OPTICS, n=6)
    each = 10 * 2 ** 20
    free = int(2.5 * problem.MEASURED_MARGIN * each / problem.CUDA_SHARE)    # room for two and a half
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (free, 24 * 2 ** 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(problem, "_measured_bytes", lambda *a: each)
    small = Problem(model, Poisson(), batch, device="cuda")
    assert small.chunk_size == 2 and small.peak_bytes == 2 * int(problem.MEASURED_MARGIN * each)
    assert Problem(model, Poisson(), batch, chunk_size=256, device="cuda").chunk_size == 2   # in situ's too


def _sampling_meter():
    """A sampling meter on this process's memory now, by the system's own count, with its release."""
    from uipsf import memory
    if sys.platform == "win32":
        return memory._SamplingMeter(memory._windows_private_bytes, memory._windows_heapmin)
    if sys.platform == "darwin":
        mac = memory._MacMeter()
        return memory._SamplingMeter(lambda: int(mac._fields()[mac.FOOTPRINT]), mac.release)
    return memory._SamplingMeter(memory._linux_rss, memory._LinuxMeter().release)


def test_the_sampling_meter_sees_a_peak_the_work_held_and_freed(monkeypatch):
    """Windows keeps no peak that can be reset, so the process's use is sampled while the work runs: 200 MB
    held for 20 ms and freed again is the peak, not the end."""
    import time

    from uipsf import memory
    use = [10 * 2 ** 20]
    monkeypatch.setattr(memory, "_METER", memory._SamplingMeter(lambda: use[0]))

    def work():
        use[0] += 200 * 2 ** 20
        time.sleep(0.02)
        use[0] -= 200 * 2 ** 20
        return "done"
    assert memory.measure(work, "cpu") == ("done", 200 * 2 ** 20)


def test_this_systems_counter_of_current_use_answers_and_counts_new_pages(monkeypatch):
    """The counter a sampling meter reads here (Windows' private bytes, Linux's VmRSS, macOS' footprint): an
    answer, and pages written counted -- at least some: the allocator may hand back pages it kept."""
    from uipsf import memory
    monkeypatch.setattr(memory, "_METER", _sampling_meter())
    out, held = memory.measure(lambda: float(torch.ones(50 * 2 ** 20)[-1]), "cpu")
    assert out == 1.0 and held is not None and 0 < held < 400 * 2 ** 20
