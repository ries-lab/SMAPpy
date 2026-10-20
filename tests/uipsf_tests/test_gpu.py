"""Run by hand on a Mac (MPS) and on a CUDA machine: ``python -m pytest tests/uipsf_tests/test_gpu.py --gpu``."""
import numpy as np
import pytest
import torch

from uipsf import default_device
from uipsf.device import to_cpu64
from uipsf_tests.conftest import make_psf, stack

pytestmark = pytest.mark.gpu


@pytest.fixture
def device():
    d = default_device()
    if d.type == "cpu":
        pytest.skip("no CUDA or MPS device")
    return d


def test_the_gpu_and_the_cpu_compute_the_same_psf_and_gradient(optics, device):
    out = []
    for dev in ("cpu", device):
        psf, p = make_psf(optics, dtype=torch.float32, device=dev, bead_radius_nm=50.0)
        phase = torch.zeros_like(p["pupil.phase"])
        phase[:5] = torch.tensor([0.6, -0.3, 0.25, -0.2, 0.4])
        phase.requires_grad_(True)
        I = psf({**p, "pupil.phase": phase, "psf.sigma": torch.tensor([40.0, 30.0], device=dev)},
                stack(np.linspace(-800, 800, 17), x=25.0, dtype=torch.float32).to(dev), z_step_nm=100.0)
        (I * torch.linspace(0, 1, I.numel(), device=dev).reshape(I.shape)).sum().backward()
        out.append((I.detach().cpu().double(), phase.grad.cpu().double()))
    (I0, g0), (I1, g1) = out
    assert (I0 - I1).abs().max() / I0.max() < 1e-5
    assert (g0 - g1).abs().max() / g0.abs().max() < 1e-4


def test_to_cpu64_copies_a_gpu_tensor_exactly_where_one_combined_copy_and_cast_need_not(device):
    x = torch.rand(6, 1, 41, 42, 42, device=device) * 100 + 50
    assert torch.equal(to_cpu64(x), x.cpu().double())


def test_a_bead_fit_from_a_cpu_batch_runs_on_the_models_device(optics, device):
    # Readers cut the stacks on the CPU; the fit goes where the model is, the fastest device by default.
    from uipsf.workflows import learn_beads
    from uipsf_tests.conftest import small_beads
    model, _, batch = small_beads(optics, n=6, dtype=torch.float32)
    import uipsf
    grid = uipsf.PupilGrid.build(optics, 24, device=device)
    model = uipsf.ForwardModel(uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=4), (9, 9),
                                               sigma_nm=(30.0, 30.0)))
    assert batch.data.device.type == "cpu"
    cal = learn_beads(model, batch, first_order=None)
    assert cal.result.params.values["pupil.phase"].device.type == device.type


def test_held_out_beads_with_masked_pixels_are_reported_on_the_gpu(optics, device):
    # The deviance comes back on the CPU, the saturation mask lives with the data: they must meet.
    from dataclasses import replace

    import uipsf
    from uipsf.workflows import learn_beads
    from uipsf.workflows.rejection import RejectionSettings
    from uipsf_tests.conftest import small_beads
    _, _, batch = small_beads(optics, n=6, dtype=torch.float32)
    mask = torch.ones_like(batch.data, dtype=torch.bool)
    mask[:, :, :, 4, 4] = False
    grid = uipsf.PupilGrid.build(optics, 24, device=device)
    model = uipsf.ForwardModel(uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=4), (9, 9),
                                               sigma_nm=(30.0, 30.0)))
    cal = learn_beads(model, replace(batch, mask=mask), first_order=None, heldout=0.34,
                      rejection=RejectionSettings(enabled=False))
    assert cal.heldout_report is not None and np.isfinite(cal.heldout_report.deviance)
