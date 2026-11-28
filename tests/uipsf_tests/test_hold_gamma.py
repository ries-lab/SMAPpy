"""*pupil magnitude* "minimal, γ held": a bead calibration that holds γ (and the NA learnt with it) from the
start model or the microscope instead of learning it (`smappy.calibrate.pupil.hold_gamma`,
`uipsf.model.build.spec_held`)."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from smappy.calibrate.pupil import hold_gamma, model_shown

from uipsf.model import MinimalPupil
from uipsf.workflows import BeadSettings, PSFModelSettings, bead_model, calibrate_beads

from .test_minimal_pupil import GAMMA, OPTICS, PHASE, RING_H, ROI, simulated

HELD = BeadSettings(psf=PSFModelSettings(zernike_order=4, magnitude="held"))


def start_model(gamma=0.27, na=1.47):
    """A learnt minimal model's stand-in (`LearnedPSF`'s ``model`` and ``shared``), its NA learnt."""
    model = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4, learn_na=True)), ROI,
                       z_max_nm=1000, device="cpu")
    pupil = model.psf.pupil
    shared = {"pupil.phase": torch.zeros(len(pupil.phase_modes)), "pupil.gamma": torch.tensor(gamma),
              "pupil.na": torch.tensor(na), "pupil.ring_h": torch.tensor(50.0), "psf.sigma": torch.tensor([20.0])}
    return SimpleNamespace(model=model, shared=shared)


def test_gamma_and_the_na_learnt_with_it_are_held_from_the_start_model():
    settings = hold_gamma(HELD, start_model())
    assert settings.psf.held_gamma == pytest.approx(0.27) and settings.psf.held_na == pytest.approx(1.47)
    assert settings.psf.held_source == "the start model"
    pupil = bead_model(OPTICS, settings, ROI, z_max_nm=1000, device="cpu").psf.pupil
    assert isinstance(pupil, MinimalPupil) and pupil.held_gamma == pytest.approx(0.27)
    assert pupil.na == pytest.approx(1.47) and not pupil.learn_na
    assert "pupil.gamma" not in {s.name for s in pupil.specs()}
    # every other choice is the minimal model's: the ring is still learnt, and the phase
    assert {"pupil.ring_h", "pupil.phase"} <= {s.name for s in pupil.specs()}


def test_a_held_gamma_is_never_learnt_or_guessed_where_nothing_gives_it():
    with pytest.raises(ValueError, match="needs a γ to hold"):
        hold_gamma(HELD, None, SimpleNamespace(microscope=None), 600.0)
    with pytest.raises(ValueError, match="needs a γ to hold"):
        bead_model(OPTICS, HELD, ROI, z_max_nm=1000, device="cpu")


def test_other_magnitudes_and_a_gamma_found_already_are_left_as_they_are():
    plain = BeadSettings()
    assert hold_gamma(plain, start_model()) is plain
    found = replace(HELD, psf=replace(HELD.psf, held_gamma=0.4))
    assert hold_gamma(found, start_model()) is found


def test_the_form_offers_learn_na_and_the_ring_for_a_held_gamma_as_for_a_learnt_one():
    held, learnt = model_shown("beads", "held"), model_shown("beads", "minimal")
    assert held == learnt
    assert model_shown("molecules", "held")["psf.learn_na"]


@pytest.mark.long             # 40 s: a whole calibration
def test_beads_learnt_with_the_true_gamma_held_give_back_the_aberrations_and_keep_the_hold():
    truth = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4)), ROI, z_max_nm=1000, device="cpu")
    beads = simulated(truth, extra={"pupil.gamma": torch.tensor(GAMMA), "pupil.ring_h": torch.tensor(RING_H)})
    settings = hold_gamma(HELD, start_model(gamma=GAMMA, na=OPTICS.NA))
    cal = calibrate_beads(beads, OPTICS, settings, device="cpu")
    assert "pupil.gamma" not in cal.shared() and cal.model.psf.pupil.held_gamma == pytest.approx(GAMMA)
    assert torch.allclose(cal.shared()["pupil.phase"][:4].cpu(), PHASE, atol=0.02)
    assert cal.options.psf.held_gamma == pytest.approx(GAMMA)        # what a run on halves rebuilds from
