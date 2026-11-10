"""The port against the original: milestone (b) of docs/decode/poc-plan.md.

The method oracle (``fixtures/v0102_method.npz``) was computed by DECODE v0.10.2 itself, torch 1.12
on a CPU (``scripts/decode/reference_v0102.py oracles``).  Its inputs are stored with it, so these
tests run the port on the very same tensors: the targets, the loss and its gradient, and the
post-processing must come out equal to 1e-6 relative, on the CPU and on the GPU there is.
"""
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from decode_smlm.generic.emitter import EmitterSet
from decode_smlm.neuralfitter.coord_transform import Offset2Coordinate
from decode_smlm.neuralfitter.loss import GaussianMMLoss
from decode_smlm.neuralfitter.post_processing import SpatialIntegration
from decode_smlm.neuralfitter.scale_transform import InverseParamListRescale, ParameterListRescale
from decode_smlm.neuralfitter.target_generator import ParameterListTarget
from decode_smlm.utils.hardware import default_device

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
DEVICES = ["cpu"] + ([default_device()] if default_device() != "cpu" else [])
REL = 1e-6


@pytest.fixture(scope="module")
def oracle():
    return dict(np.load(FIXTURES / "v0102_method.npz"))


def _close(port, original, rel=REL):
    """Equal to `rel` of the original's largest magnitude: the measure the plan states."""
    port, original = np.asarray(port, dtype=np.float64), np.asarray(original, dtype=np.float64)
    assert port.shape == original.shape
    scale = max(np.abs(original).max(), 1e-30)
    assert np.abs(port - original).max() <= rel * scale, np.abs(port - original).max() / scale


def _setup(o):
    n_batch, _, h, w = o["output"].shape
    xext, yext = (-0.5, h - 0.5), (-0.5, w - 0.5)
    em = EmitterSet(xyz=torch.from_numpy(o["em_xyz"]).float(), phot=torch.from_numpy(o["em_phot"]).float(),
                    frame_ix=torch.from_numpy(o["em_frame_ix"]).long(), xy_unit="px", px_size=(100.0, 100.0))
    return n_batch, (h, w), xext, yext, em


def test_the_targets_are_the_originals(oracle):
    n_batch, _, xext, yext, em = _setup(oracle)
    tar = ParameterListTarget(n_max=int(oracle["n_max"]), xextent=xext, yextent=yext, ix_low=0,
                              ix_high=n_batch - 1, squeeze_batch_dim=False)
    param, mask, _ = tar.forward(em, torch.from_numpy(oracle["bg_target_in"]))
    assert np.array_equal(mask.numpy(), oracle["tar_mask"])
    assert np.array_equal(param.numpy(), oracle["tar_param"])
    scaled, _, bg = ParameterListRescale(phot_max=float(oracle["phot_max"]), z_max=float(oracle["z_max"]),
                                         bg_max=float(oracle["bg_max"])).forward(
        param.clone(), mask, torch.from_numpy(oracle["bg_target_in"]))
    _close(scaled, oracle["tar_param_scaled"])
    _close(bg, oracle["tar_bg_scaled"])


@pytest.mark.parametrize("device", DEVICES)
def test_the_loss_and_its_gradient_are_the_originals(oracle, device):
    _, shape, xext, yext, _ = _setup(oracle)
    loss_fn = GaussianMMLoss(xextent=xext, yextent=yext, img_shape=shape, device=device, chweight_stat=(1.0, 1.0))
    x = torch.from_numpy(oracle["output"]).to(device).requires_grad_(True)
    target = tuple(torch.from_numpy(oracle[k]).to(device) for k in ("tar_param_scaled", "tar_mask", "tar_bg_scaled"))
    loss = loss_fn.forward(x, target, None)
    loss.sum().backward()
    _close(loss.detach().cpu(), oracle["loss"])
    _close(x.grad.cpu(), oracle["loss_grad"])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("raw_th", [0.1, 0.5])
def test_the_post_processing_finds_the_originals_localizations(oracle, device, raw_th):
    _, shape, xext, yext, _ = _setup(oracle)
    y = torch.from_numpy(oracle["output"]).to(device)
    y = InverseParamListRescale(phot_max=float(oracle["phot_max"]), z_max=float(oracle["z_max"]),
                                bg_max=float(oracle["bg_max"])).forward(y)
    y = Offset2Coordinate(xextent=xext, yextent=yext, img_shape=shape).forward(y)
    es = SpatialIntegration(raw_th=raw_th, xy_unit="px", px_size=(100.0, 100.0)).forward(y)
    key = f"pp_th{raw_th:g}".replace(".", "")
    assert np.array_equal(es.frame_ix.numpy(), oracle[f"{key}_frame_ix"])
    for field in ("xyz", "phot", "prob", "bg", "xyz_sig", "phot_sig"):
        _close(getattr(es, field).numpy(), oracle[f"{key}_{field}"])


def test_importing_smappy_and_discovering_its_plugins_leaves_torch_unloaded():
    """decode_smlm is reached only where a DECODE model is used, as uipsf is: nothing else loads torch."""
    code = ("import sys, smappy; from smappy import plugins; plugins.discover(force=True); "
            "print('torch' in sys.modules, 'decode_smlm' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split()[-2:] == ["False", "False"], out.stdout + out.stderr


def test_the_lazy_split_gives_split_sliceables_elements_in_its_order():
    """The training's dataset cuts each frame's emitters out when it is read (``LazySplit``)."""
    import torch

    from decode_smlm.generic.emitter import RandomEmitterSet
    from decode_smlm.generic.slicing import LazySplit, split_sliceable
    torch.manual_seed(0)
    em = RandomEmitterSet(500)
    em.frame_ix = torch.randint(-1, 12, (500,))
    eager = split_sliceable(em, em.frame_ix, 0, 10)
    lazy = LazySplit(em, em.frame_ix, 0, 10)
    assert len(lazy) == len(eager) == 11
    for a, b in zip(lazy, eager):
        assert torch.equal(a.xyz, b.xyz) and torch.equal(a.frame_ix, b.frame_ix) and torch.equal(a.phot, b.phot)
    assert torch.equal(lazy[-1].xyz, eager[-1].xyz) and len(lazy[2:5]) == 3
