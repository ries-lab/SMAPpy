"""4Pi in situ: a model learnt from single molecules in single frames, started from a bead model
(`uipsf.workflows.fourpi_insitu`; docs/4pi.md §27)."""
import math

import numpy as np
import pytest
import torch

import uipsf
from uipsf.model.fourpi import FourPiModel
from uipsf.simulate import simulate
from uipsf.workflows import fourpi_insitu as fi
from uipsf.workflows.fourpi import calibration

C, R = 4, 11
PERIOD_NM = 270.0


@pytest.fixture(scope="module")
def bead_model():
    optics = uipsf.Optics(NA=1.35, wavelength_nm=600, n_imm=1.406, n_med=1.406, pixel_size_nm=(129, 129))
    grid = uipsf.PupilGrid.build(optics, 64, device="cpu")
    return FourPiModel(grid, (R, R), n_max=4, magnitude_n_max=0, kind="scalar", n_phases=3)


def _phase(model, pairs):
    modes = model.pupils[0].phase_modes
    up, lo = torch.zeros(len(modes)), torch.zeros(len(modes))
    for j, a, b in pairs:
        up[modes.index(j)], lo[modes.index(j)] = a, b
    return up, lo


def _shared(model, pairs, coherence, delta, dz, shift, sigma):
    up, lo = _phase(model, pairs)
    H = torch.tensor([1.0, 0.0, shift[0], 0.0, 1.0, shift[1], 0.0, 0.0])
    return {"pupil.phase": up, "lo.pupil.phase": lo, "pupil.magnitude": torch.zeros(0),
            "lo.pupil.magnitude": torch.zeros(0), "psf.sigma": torch.tensor([sigma, sigma]),
            "fourpi.offset": torch.tensor([20.0, -15.0, dz]), "fourpi.steps": torch.tensor([-2.0, -4.0]),
            "fourpi.delta": torch.tensor(delta), "fourpi.share": torch.tensor([0.26, 0.24, 0.25]),
            "fourpi.balance": torch.tensor([0.5, 0.45, 0.55, 0.5]), "fourpi.coherence": torch.tensor(coherence),
            "ch1.transform": H, "ch2.transform": H.clone(), "ch3.transform": H.clone()}


#: the sample's truth, and the bead model a day before: astigmatism and coma moved, the coherence, the
#: channels' offsets, Δz, the maps and the blur all off
TRUTH = dict(pairs=[(6, 0.4, -0.45), (8, 0.1, 0.0), (7, 0.0, -0.08), (11, 0.0, 0.1)],
             coherence=[0.8, 0.75, 0.82, 0.7], delta=[1.7, 3.0, 4.6], dz=60.0, shift=(0.3, -0.2), sigma=50.0)
START = dict(pairs=[(6, 0.25, -0.30), (8, 0.0, 0.05), (7, 0.05, 0.0), (11, 0.05, 0.05)],
             coherence=[0.7] * 4, delta=[1.9, 3.2, 4.4], dz=20.0, shift=(-0.2, 0.3), sigma=40.0)


def _molecules(model, truth, E, seed=0, cavity=False):
    """E molecules in single frames, each with its own z over ±450 nm, cavity phase and brightness, drawn
    with ``truth``; and the fitter's start: positions off by 15 nm laterally and 40 nm in z, a tenth of them
    a fringe off, the phase unknown.  ``cavity``: the phase is the cavity's, drifting over 3000 frames and
    tilted over the field, plus 0.1 rad of jitter, as a movie's."""
    rng = np.random.default_rng(seed)
    xyz = np.c_[rng.uniform(-60, 60, (E, 2)), rng.uniform(-450, 450, E)]
    roi_xy = np.repeat(rng.uniform(20, 180, (E, 1, 2)).round(), C, 1)
    other = np.random.default_rng(seed + 100)            # its own draws: the untied test's stay as they were
    frame = other.integers(0, 3000, E)
    drawn = {"xyz": xyz, "photons": rng.lognormal(math.log(4000), 0.4, E), "background": np.full((E, C), 10.0),
             "phase": rng.uniform(-math.pi, math.pi, E)}
    if cavity:
        drawn["phase"] = 2.5 + 0.6 * frame / 3000 + 0.004 * (roi_xy[:, 0, 0] - 100) + other.normal(0, 0.1, E)
    batch = fi.molecule_batch(np.zeros((E, C, R, R)), roi_xy, frame=frame)
    batch = simulate(model, fi.params_for(model, batch, truth, drawn), batch, seed=seed + 1)
    start = {"xyz": xyz + np.c_[rng.normal(0, 15, (E, 2)), rng.normal(0, 40, E)],
             "photons": drawn["photons"] * 0.9, "background": drawn["background"], "phase": np.zeros(E)}
    wrong = rng.random(E) < 0.1
    start["xyz"][wrong, 2] += PERIOD_NM * np.sign(rng.normal(size=wrong.sum()))
    return batch, start, xyz


def test_a_molecule_model_sees_one_frame_of_point_emitters_with_one_photon_count(bead_model):
    model = fi.molecule_model(bead_model, (R, R))
    batch = fi.molecule_batch(np.zeros((3, C, R, R)), np.zeros((3, C, 2)))
    names = {s.name: s.shape for s in model.specs(batch, {"xyz": torch.zeros(3, 3), "photons": torch.ones(3),
                                                          "background": torch.ones(3, C)})}
    assert names["fourpi.steps"] == (0,) and names["emitter.photons"] == ()
    assert "emitter.photons_per_slice" not in names and model.psf_up.bead_radius_nm == 0
    assert batch.data.shape == (3, C, 1, R, R)


def test_poor_fits_are_left_out_and_good_ones_never_for_fitting_too_well():
    d = np.r_[np.random.default_rng(0).normal(1.0, 0.02, 200), 0.5, 2.0]
    out = fi.outliers(d)
    assert out[-1] and not out[-2] and out[:200].sum() <= 2


def test_the_cavity_phase_is_found_from_the_molecules_with_its_drift_and_tilt_through_noise_and_wraps():
    rng = np.random.default_rng(0)
    E = 3000
    frame, xy = rng.integers(0, 6000, E), rng.uniform(0, 168, (E, 2))
    true = 3.0 + 0.4 * frame / 6000 + 0.01 * (xy[:, 1] - 84)            # drifting, tilted in y, near +-pi
    stray = (rng.random(E) < 0.1) * rng.uniform(-3, 3, E)                 # a tenth far off: wrong fits
    phase = fi._wrap(true + rng.normal(0, 0.3, E) + stray)
    cavity = fi.cavity_phases(phase, frame, xy, 1000)
    assert np.median(np.abs(fi._wrap(cavity - true))) < 0.03 and np.max(np.abs(fi._wrap(cavity - true))) < 0.12
    assert np.all(np.abs(cavity - phase) <= math.pi + 1e-9)               # each the turn nearest its own phase


def test_a_prior_with_a_mean_per_emitter_follows_its_emitters_into_a_solvers_chunk():
    from uipsf.params import Gaussian, Params, ParamSpec
    mean = torch.arange(5.0)
    p = Params([ParamSpec("emitter.phase", (), "emitter", prior=Gaussian(mean, 0.5))], 5,
               values={"emitter.phase": torch.zeros(5)})
    part = p.subset(torch.tensor([1, 3]))
    assert torch.equal(part.specs["emitter.phase"].prior.mean, torch.tensor([1.0, 3.0]))
    r, _ = part.prior_residuals("emitter")
    assert torch.allclose(r.ravel(), torch.tensor([-2.0, -6.0]))
    assert p.specs["emitter.phase"].prior.mean.shape == (5,)              # the whole set keeps its own


def test_the_in_situ_export_keeps_the_bead_exports_phase_convention(bead_model):
    """The same shared values exported from the molecule model and from the bead model: the same period,
    the same sign, the same I, A, B -- unlike uiPSF v1, whose in-situ export had the opposite sign."""
    shared = _shared(bead_model, **TRUTH)
    model = fi.molecule_model(bead_model, (R, R))
    a = calibration((bead_model, shared), z_range_nm=300, dz_nm=50)
    b = fi.calibration(fi.FourPiInSitu(model, shared, None, np.arange(5), {}), z_range_nm=300, dz_nm=50)
    assert a.phase_per_nm == pytest.approx(b.phase_per_nm, rel=1e-4) and b.phase_per_nm < 0
    assert np.allclose(a.volumes, b.volumes, atol=1e-6)
    assert b.parameters["method"] == "uiPSF v2 4Pi in-situ model"


@pytest.mark.long
def test_in_situ_learning_brings_back_the_samples_pupils_coherence_maps_and_misalignment(bead_model):
    """160 simulated molecules learnt from a bead model a day off: everything that moved comes back, much
    nearer the truth than the start was, and z lands on the right fringe for every molecule."""
    model = fi.molecule_model(bead_model, (R, R))
    truth, start = _shared(model, **TRUTH), _shared(model, **START)
    batch, fitter, xyz = _molecules(model, truth, 160)
    res = fi.learn_insitu(bead_model, start, batch, fitter, PERIOD_NM, settings=fi.InSituSettings(rounds=1))
    for name in ("pupil.phase", "lo.pupil.phase"):
        before = float((start[name] - truth[name]).norm())
        after = float((res.shared[name] - truth[name]).norm())
        assert after < 0.25 * before, name
    assert np.allclose(res.shared["fourpi.coherence"], truth["fourpi.coherence"], atol=0.05)
    assert np.allclose(res.shared["fourpi.delta"], truth["fourpi.delta"], atol=0.08)
    assert abs(float(res.shared["fourpi.offset"][2]) - 60.0) < 15
    assert np.allclose(res.shared["fourpi.offset"][:2], truth["fourpi.offset"][:2], atol=2)
    assert np.allclose(res.shared["ch1.transform"][[2, 5]], [0.3, -0.2], atol=0.06)
    dz = res.found["xyz"][:, 2] - xyz[:, 2]
    assert np.mean(np.abs(dz) > PERIOD_NM / 2) == 0
    assert np.median(np.abs(dz)) < 15


@pytest.mark.long
def test_tied_to_the_cavity_the_molecules_still_bring_back_the_sample_and_keep_the_fringe_period(bead_model):
    """The same learning with each molecule's phase held near the cavity's (docs/4pi.md §29): what moved comes
    back as untied, the model's fringe period is the truth's, and z lands on the right fringe."""
    model = fi.molecule_model(bead_model, (R, R))
    truth, start = _shared(model, **TRUTH), _shared(model, **START)
    batch, fitter, xyz = _molecules(model, truth, 160, cavity=True)
    res = fi.learn_insitu(bead_model, start, batch, fitter, PERIOD_NM,
                          settings=fi.InSituSettings(rounds=1, cavity_sigma_rad=0.2, cavity_window_frames=1000))
    for name in ("pupil.phase", "lo.pupil.phase"):
        assert float((res.shared[name] - truth[name]).norm()) < 0.25 * float((start[name] - truth[name]).norm())
    assert np.allclose(res.shared["fourpi.coherence"], truth["fourpi.coherence"], atol=0.05)
    assert res.rounds[0]["cavity_spread_rad"] < 1.0
    period = calibration((model, res.shared), z_range_nm=300, dz_nm=50).period_nm
    assert period == pytest.approx(calibration((model, truth), z_range_nm=300, dz_nm=50).period_nm, rel=0.01)
    dz = res.found["xyz"][:, 2] - xyz[:, 2]
    assert np.mean(np.abs(dz) > PERIOD_NM / 2) == 0 and np.median(np.abs(dz)) < 15
