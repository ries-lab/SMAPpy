"""The supercritical ring on a Zernike magnitude, on request (``zernike_ring``): the beads' exp(−2πκh) as its own
factor, as the minimal pupil has it, so that the Zernikes need not make its kink at the critical angle and a spline
for single molecules is drawn without it."""
import math

import pytest
import torch

from uipsf.export.model_file import build, describe, sample_psf
from uipsf.model import ZernikePupil
from uipsf.optics.zernike import noll_to_nm
from uipsf.workflows import BeadSettings, PSFModelSettings, bead_model, calibrate_beads

from .test_minimal_pupil import OPTICS, PHASE, ROI, SIGMA, simulated

RING_H = 60.0
#: Noll 4 (defocus-shaped fall) and 5 (an asymmetric magnitude): what the acquisition gives beside the ring
MAGNITUDE = {4: -0.08, 5: 0.05}


def _settings(magnitude="free", saf_damping=True, zernike_ring=True):
    return BeadSettings(psf=PSFModelSettings(zernike_order=4, magnitude=magnitude, saf_damping=saf_damping,
                                             zernike_ring=zernike_ring))


def _model(**kw):
    return bead_model(OPTICS, _settings(**kw), ROI, z_max_nm=1000, device="cpu")


def _magnitude(pupil):
    c = torch.zeros(len(pupil.magnitude_modes))
    for j, v in MAGNITUDE.items():
        c[pupil.magnitude_modes.index(j)] = v
    return c


@pytest.mark.parametrize("magnitude", ["free", "symmetric", "fixed"])
def test_every_zernike_magnitude_asked_for_it_has_the_beads_ring_beyond_the_critical_angle(magnitude):
    pupil = _model(magnitude=magnitude).psf.pupil
    assert isinstance(pupil, ZernikePupil) and pupil.ring and "pupil.ring_h" in {s.name for s in pupil.specs()}
    p = {"pupil.phase": torch.zeros(len(pupil.phase_modes)), "pupil.magnitude": torch.zeros(len(pupil.magnitude_modes)),
         "pupil.ring_h": torch.tensor(RING_H)}
    magnitude_map, _ = pupil(p)
    assert torch.allclose(magnitude_map, torch.exp(-2 * math.pi * pupil.kappa * RING_H))
    rho = pupil.grid.rho.real if pupil.grid.rho.is_complex() else pupil.grid.rho
    inside = rho < OPTICS.n_med / OPTICS.NA - 0.02                       # below the critical angle: untouched
    assert torch.allclose(magnitude_map[inside], torch.ones(()))
    assert pupil.values(p) == pytest.approx({"ring_h_nm": RING_H})


@pytest.mark.parametrize("asked", [dict(zernike_ring=False), dict(saf_damping=False)])
def test_a_zernike_magnitude_has_no_ring_unless_asked_with_saf_damping_on(asked):
    pupil = _model(**asked).psf.pupil
    assert not pupil.ring and "pupil.ring_h" not in {s.name for s in pupil.specs()}


def test_by_default_only_the_minimal_magnitude_has_the_ring():
    psf = PSFModelSettings(zernike_order=4)
    assert psf.saf_damping and not psf.zernike_ring
    assert bead_model(OPTICS, BeadSettings(psf=psf), ROI, z_max_nm=1000, device="cpu").psf.pupil.ring


def test_the_ring_survives_the_model_file_and_its_first_stage():
    model = _model()
    again = build(describe(model), device="cpu").psf.pupil
    assert again.saf_damping and again.ring and torch.allclose(again.kappa, model.psf.pupil.kappa)
    low = model.psf.pupil.truncated(2)
    assert low.ring and "pupil.ring_h" in {s.name for s in low.specs()}


def test_a_spline_from_a_zernike_model_is_drawn_without_the_beads_saf_damping():
    model = _model()
    pupil = model.psf.pupil
    shared = {"pupil.phase": torch.zeros(len(pupil.phase_modes)), "pupil.magnitude": _magnitude(pupil),
              "psf.sigma": torch.tensor([SIGMA])}
    xyz = torch.tensor([[0.0, 0.0, -300.0], [0.0, 0.0, 0.0], [0.0, 0.0, 300.0]])
    high = sample_psf(model, shared | {"pupil.ring_h": torch.tensor(200.0)}, xyz)
    glass = sample_psf(model, shared | {"pupil.ring_h": torch.tensor(0.0)}, xyz)
    assert torch.allclose(high, glass)                                  # h is the beads', not a molecule's


@pytest.mark.long                  # two calibrations, ~90 s
def test_a_zernike_calibration_recovers_the_ring_and_leaves_the_zernikes_the_rest():
    """Beads with a ring at 60 nm and a small magnitude in Noll 4 and 5: with the ring, h and the two modes come
    back and the others stay at 0; without it, the radial modes take up the ring's loss."""
    truth = _model()
    beads = simulated(truth, extra={"pupil.magnitude": _magnitude(truth.psf.pupil),
                                    "pupil.ring_h": torch.tensor(RING_H)})
    shared = calibrate_beads(beads, OPTICS, _settings(), device="cpu").shared()
    assert float(shared["pupil.ring_h"]) == pytest.approx(RING_H, abs=15.0)
    assert float(shared["psf.sigma"]) == pytest.approx(SIGMA, abs=4.0)
    assert torch.allclose(shared["pupil.phase"][:4].cpu(), PHASE, atol=0.02)
    modes = truth.psf.pupil.magnitude_modes
    found = dict(zip(modes, shared["pupil.magnitude"].cpu().tolist(), strict=True))
    for j, v in found.items():
        assert v == pytest.approx(MAGNITUDE.get(j, 0.0), abs=0.02), j
    bare = calibrate_beads(beads, OPTICS, _settings(zernike_ring=False), device="cpu").shared()
    without = dict(zip(modes, bare["pupil.magnitude"].cpu().tolist(), strict=True))
    radial = [j for j in modes if noll_to_nm(j)[1] == 0 and j != 4]
    assert max(abs(without[j]) for j in radial) > 2 * max(abs(found[j]) for j in radial)
