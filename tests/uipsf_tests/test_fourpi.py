"""4Pi from beads: the coherent two-objective model, its start, its fit and its export (docs/4pi.md, M5)."""
import math

import numpy as np
import pytest
import torch

import uipsf
from uipsf.model.fourpi import FourPiModel, swapped
from uipsf.model.fourpi_start import rough_start
from uipsf.params import Params
from uipsf.simulate import simulate

E, C, M, J, R = 6, 4, 3, 9, 13
#: the real beads' model: Zernikes to n = 6 and a free photon count per observation, as v1 (docs/4pi.md §8)
N_MAX_REAL = 6
STEPS = [0.0, -2.1, -4.0]                # the mirrors' phase steps from the first, as v1 found them


def _optics():
    return uipsf.Optics(NA=1.35, wavelength_nm=600, n_imm=1.406, n_med=1.406, pixel_size_nm=(129, 129))


@pytest.fixture(scope="module")
def grid():
    return uipsf.PupilGrid.build(_optics(), 64, device="cpu")


def _truth(model):
    up = model.pupils[0]
    rng = np.random.default_rng(1)
    ph_up, ph_lo = torch.zeros(len(up.phase_modes)), torch.zeros(len(up.phase_modes))
    for j, a, b in ((6, 0.4, -0.45), (8, 0.1, 0.0), (7, 0.0, -0.08), (11, 0.0, 0.1)):
        ph_up[up.phase_modes.index(j)], ph_lo[up.phase_modes.index(j)] = a, b
    return {"pupil.phase": ph_up, "lo.pupil.phase": ph_lo, "psf.sigma": torch.tensor([50.0, 50.0]),
            "fourpi.offset": torch.tensor([20.0, -15.0, 60.0]), "fourpi.steps": torch.tensor(STEPS[1:]),
            "fourpi.delta": torch.tensor([1.7, 3.0, 4.6]), "fourpi.share": torch.tensor([0.26, 0.24, 0.25]),
            "fourpi.balance": torch.tensor([0.5, 0.45, 0.55, 0.5]),
            "fourpi.coherence": torch.tensor([0.8, 0.75, 0.82, 0.7]),
            "emitter.xyz": torch.tensor(np.c_[rng.uniform(-60, 60, (E, 2)), rng.uniform(-100, 100, E)],
                                        dtype=torch.float32),
            "emitter.phase": torch.tensor(rng.uniform(-math.pi, math.pi, E), dtype=torch.float32),
            "emitter.photons": torch.full((E,), 8000.0), "emitter.background": torch.full((E, C), 10.0)}


@pytest.fixture(scope="module")
def beads(grid):
    """Six simulated 4Pi bead stacks: astigmatism of opposite sign in the two objectives (as v1 found), a
    little coma and spherical, misaligned arms, unequal channels; 11 slices 80 nm apart × 3 mirror steps."""
    model = FourPiModel(grid, (R, R), n_max=4, magnitude_n_max=0)
    z = -np.arange(-(J // 2), J // 2 + 1) * 100.0
    roi_xy = np.random.default_rng(0).uniform(20, 180, (E, 1, 2)).repeat(C, 1)
    batch = uipsf.Batch(data=torch.zeros(E, C, J * M, R, R), z_nm=torch.tensor(np.repeat(z, M), dtype=torch.float32),
                        roi_xy=torch.tensor(roi_xy, dtype=torch.float32))
    truth = _truth(model)
    start = {"xyz": truth["emitter.xyz"], "photons": truth["emitter.photons"],
             "background": truth["emitter.background"]}
    p = Params(model.specs(batch, start), E, values=truth)
    return model, simulate(model, p, batch, seed=2), {k: v for k, v in p.physical().items()}


def test_swapping_the_arms_labels_leaves_every_image_unchanged(beads):
    model, batch, truth = beads
    with torch.no_grad():
        mu = model(truth, batch)
        other = model(swapped(model, truth), batch)
    assert torch.allclose(mu, other, rtol=1e-4, atol=1e-4 * float(mu.max()))
    assert not torch.allclose(swapped(model, truth)["pupil.phase"], truth["pupil.phase"])


def test_the_start_finds_the_channels_offsets_and_turns_reversed_steps_round(beads):
    """Told the mirror steps the other way round from how they enter the model's phase ([−2, 0, 2]), the
    start turns them round from the fringe's direction and finds the channels' offsets from the light."""
    model, batch, truth = beads
    told = FourPiModel(model.grid, (R, R), n_max=4, magnitude_n_max=0, steps_rad=[-2, 0, 2])
    start = rough_start(batch, told)
    assert start["steps_flipped"]
    assert np.allclose(start["steps"].numpy(), [-2, -4])
    d = np.angle(np.exp(1j * (start["deltas"].numpy() - truth["fourpi.delta"].numpy())))
    assert np.max(np.abs(d)) < 0.15
    assert 230 < start["period_nm"] < 320


@pytest.mark.long
def test_a_4pi_bead_fit_recovers_both_pupils_the_misalignment_the_phases_and_the_steps(beads):
    from uipsf.workflows.fourpi import learn_fourpi
    model, batch, truth = beads
    fresh = FourPiModel(model.grid, (R, R), n_max=4, magnitude_n_max=0, steps_rad=[2, 0, -2])
    fit = learn_fourpi(fresh, batch, first_order=None)
    got, std = fit.shared(), fit.result.std
    # the arms' labels are a gauge: compare with the truth or its swap, whichever the fit took
    candidates = [truth, swapped(model, truth)]
    want = min(candidates, key=lambda t: float((t["pupil.phase"] - got["pupil.phase"]).abs().sum()))
    for name in ("pupil.phase", "lo.pupil.phase", "psf.sigma", "fourpi.offset", "fourpi.steps", "fourpi.delta",
                 "fourpi.share", "fourpi.balance", "fourpi.coherence"):
        d = got[name] - want[name]
        if name in ("fourpi.steps", "fourpi.delta"):
            d = torch.atan2(torch.sin(d), torch.cos(d))
        assert torch.all(d.abs() < 4 * std[name] + 1e-3), (name, got[name], want[name], std[name])
    ph = fit.result.params.physical()
    d = np.angle(np.exp(1j * (ph["emitter.phase"] - truth["emitter.phase"]).numpy()))
    assert np.all(np.abs(d) < 4 * std["emitter.phase"].numpy())
    assert torch.all((ph["emitter.xyz"] - truth["emitter.xyz"]).abs() < 4 * std["emitter.xyz"])


@pytest.fixture(scope="module")
def exported(beads):
    pytest.importorskip("smappy")
    from uipsf.workflows.fourpi import calibration
    model, _, truth = beads
    shared = {k: v for k, v in truth.items() if not k.startswith("emitter.")}
    return model, shared, calibration((model, shared), z_range_nm=500, dz_nm=20, roi_px=17)


def test_the_export_is_the_models_own_image_at_every_phase(beads):
    """I + Re(Q e^{iφ}) from three phases is the model's image at any other phase, exactly."""
    from uipsf.workflows.fourpi import sample_iab
    model, _, truth = beads
    shared = {k: v for k, v in truth.items() if not k.startswith("emitter.")}
    z = np.array([-300.0, 0.0, 170.0])
    I, Q = sample_iab(model, shared, z)
    for phi in (0.4, 2.5, -1.9):
        with torch.no_grad():
            image = model.channel_images(shared, torch.tensor(np.c_[np.zeros((3, 2)), z], dtype=torch.float32),
                                         torch.full((3,), phi)).double().numpy()
        assert np.allclose(I + np.real(Q * np.exp(1j * phi)), image, atol=1e-6 * image.max())


def test_the_fringe_period_comes_from_the_models_own_phase_and_the_fitter_follows_it(beads, exported):
    """Spots drawn from the learned model, fitted by SMAPpy's 4Pi fitter with the exported calibration:
    z comes back, and the fitted phase moves with z by the period the export measured, not a typed one."""
    from smappy.psf import Global4PiPSF
    model, shared, cal = exported
    assert cal.phase_per_nm < 0 and 230 < cal.period_nm < 320
    n, roi = 41, 13
    z = np.linspace(-300, 300, n)
    cavity = 0.7
    xyz = torch.tensor(np.c_[np.zeros((n, 2)), z], dtype=torch.float32)
    with torch.no_grad():
        # the bead's phase is the cavity's: the fitter's phase is k z + that, k the export's
        images = model.channel_images(shared, xyz, torch.full((n,), cavity)).double().numpy()
    o = (images.shape[-1] - roi) // 2
    spots = 20000 * np.moveaxis(images[..., o:o + roi, o:o + roi], 0, 1) + 5.0
    fit = Global4PiPSF(cal)
    v = fit.unpack(fit.fit(spots.astype(np.float32)))
    assert np.max(np.abs(v["z_nm"] - z)) < 5
    slope = np.polyfit(z, np.unwrap(v["phase_rad"]), 1)[0]
    assert slope == pytest.approx(cal.phase_per_nm, rel=0.01)
    offset = np.angle(np.exp(1j * (v["phase_rad"] - cal.phase_per_nm * z - cavity)))
    assert np.max(np.abs(offset)) < 0.05


# ---------------------------------------------------------------- the example beads (zenodo 10823308)

def _data():
    import os
    from pathlib import Path
    root = Path(os.environ.get("UIPSF_DATA", Path.home() / ".cache/uipsf/zenodo"))
    if not (root / "4pi_40nm_bead").is_dir():
        pytest.skip("the 4Pi example beads are not downloaded (scripts/uipsf/fetch_zenodo.py 4pi_40nm_bead/)")
    return root


def test_the_4pi_scans_are_cut_into_one_roi_per_channel_on_the_same_bead():
    pytest.importorskip("smappy")
    from smappy.calibrate.fourpi_beads import find_beads
    paths = sorted((_data() / "4pi_40nm_bead").glob("*/*.mat"))[:2]
    beads = find_beads(paths, conversion=0.44, offset=100, roi_size=21)
    E, C, J, M, Y, X = beads.data.shape
    assert (C, J, M, Y, X) == (4, 21, 3, 21, 21) and E >= 4
    # 50 nm steps (the stage's first reading repeated is not a step), z falling along the scan
    assert np.allclose(np.diff(beads.z_nm), -50, atol=1.5) and beads.z_nm[10] == 0
    # the quadrants are registered to a pixel or two, and each channel's ROI is on its bead
    assert np.max(np.abs(beads.roi_xy - beads.roi_xy[:, :1])) <= 2
    light = beads.data.sum((2, 3))                                            # (E, C, Y, X)
    light = np.clip(light - np.median(light, axis=(-2, -1), keepdims=True), 0, None)
    centre = [(light * g).sum((-2, -1)) / light.sum((-2, -1)) for g in np.mgrid[:Y, :X]]
    assert np.all(np.abs(np.array(centre) - 10) < 1.5)


def _tracks(cal, data, crop=15, step_nm=50.0):
    """Each bead and mirror step fitted slice by slice with SMAPpy's 4Pi fitter: the scatter of the phase
    (rad) and of z (nm) about a line through the stage, over the middle 15 slices, the median of both."""
    from smappy.psf import Global4PiPSF
    C, E, M, J, Y, X = data.shape
    o = (Y - crop) // 2
    rois = np.moveaxis(data[..., o:o + crop, o:o + crop], 0, 3).reshape(-1, C, crop, crop)
    fit = Global4PiPSF(cal)
    v = fit.unpack(fit.fit(rois.astype(np.float32)))
    phase, z = v["phase_rad"].reshape(E, M, J), v["z_nm"].reshape(E, M, J)
    stage = np.arange(J) * step_nm
    core = slice(3, J - 3)

    def scatter(u):
        return np.sqrt(np.mean((u - np.polyval(np.polyfit(stage[core], u, 1), stage[core])) ** 2))
    return (np.median([scatter(np.unwrap(phase[e, m, core])) for e in range(E) for m in range(M)]),
            np.median([scatter(z[e, m, core]) for e in range(E) for m in range(M)]),
            np.median(np.angle(np.exp(1j * (phase[:, 1:] - phase[:, :1]))), axis=(0, 2)))


@pytest.mark.long
def test_the_learned_model_measures_v1s_beads_as_straight_as_v1s_model():
    """uiPSF v1's 18 training beads (its own ROIs, so both models see the same pixels), learnt here and
    fitted back by SMAPpy's 4Pi fitter with each model: the phase and z follow the stage as closely with
    this model as with v1's, to 10 % (measured: 7 % more phase scatter, 5 % more z; docs/4pi.md §8), and
    the mirror steps come out as v1's do.  Its period is measured from the model, not typed in."""
    pytest.importorskip("smappy")
    import h5py
    from smappy import fourpi

    from uipsf.model import PerObservation
    from uipsf.workflows.fourpi import calibration, learn_fourpi
    v1_path = _data() / "results/4pi_40nm_bead/psfmodel_zernike_4pi.h5"
    if not v1_path.exists():
        pytest.skip("v1's 4Pi bead model is not downloaded (results/4pi_40nm_bead/)")
    with h5py.File(v1_path, "r") as f:
        data, cor = f["rois/psf_data"][...], f["rois/cor"][...]        # (C, bead, step, slice, y, x)
    C, Eb, Mb, Jb, Y, X = data.shape
    z = -(np.arange(Jb) - (Jb - 1) / 2) * 50.0
    batch = uipsf.Batch(data=torch.as_tensor(np.transpose(data, (1, 0, 3, 2, 4, 5)).reshape(Eb, C, Jb * Mb, Y, X)),
                        z_nm=torch.as_tensor(np.repeat(z, Mb), dtype=torch.float32), s=1.0,
                        roi_xy=torch.as_tensor(np.transpose(cor, (1, 0, 2))[..., ::-1].astype(np.float32).copy()))
    grid = uipsf.PupilGrid.build(_optics(), 64)
    model = FourPiModel(grid, (Y, X), n_max=N_MAX_REAL, steps_rad=[2, 0, -2], intensity=PerObservation())
    fit = learn_fourpi(model, batch)
    cal = calibration(fit, z_range_nm=600, dz_nm=25, roi_px=21)
    assert 255 < cal.period_nm < 280
    phase, z_rms, steps = _tracks(cal, data)
    phase_v1, z_rms_v1, steps_v1 = _tracks(fourpi.load_uipsf_v1(v1_path, transpose=False), data)
    assert phase <= 1.1 * phase_v1
    assert z_rms <= 1.1 * z_rms_v1
    assert np.allclose(steps, steps_v1, atol=0.03)
