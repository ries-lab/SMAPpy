"""The residual atlas, likelihood-ratio probes and held-out report (plan §8, M3) on simulated mismatch
(review §B).  Slow tier: each case fits a model, the probes refit it."""
import numpy as np
import pytest
import torch

from mismatch import TRUE_PHASE, mismatched
from uipsf.diagnostics import residual_atlas, select_terms
from uipsf.model import BeadStackGeometry, ForwardModel, PerObservation, PerSliceFactor
from uipsf.noise import ShiftedPoisson
from uipsf.workflows import fit_beads, learn_beads

pytestmark = pytest.mark.slow

SPECIFIC = ("photons per slice, shared", "photons per observation", "background ramp")


@pytest.mark.parametrize("kind,check", [("background ramp", "background ramp"),
                                        ("laser", "photons per slice, shared"),
                                        ("speckle", "photons per observation")])
def test_the_atlas_names_the_missing_physics(kind, check):
    model, batch = mismatched(kind)
    res = fit_beads(model, batch, first_order=None)
    atlas = residual_atlas(model, res.params, batch, ShiftedPoisson())
    flagged = atlas.flagged()
    assert check in flagged, atlas.checks
    assert [c for c in flagged if c in SPECIFIC] == [check], atlas.checks


@pytest.mark.long
@pytest.mark.parametrize("kind,term,kw", [("piezo scale", "z scale", dict(K=41, z_range_nm=1000.0)),
                                          ("background ramp", "planar background", {}),
                                          ("laser", "per-slice photons", {})])
def test_forward_selection_adds_the_missing_term_first(kind, term, kw):
    # A 5 % piezo scale leaves no pattern in the residuals (the pupil absorbs it); its probe finds it.
    model, batch = mismatched(kind, **kw)
    res = fit_beads(model, batch, first_order=None)
    if kind == "piezo scale":
        assert residual_atlas(model, res.params, batch, ShiftedPoisson()).flagged() in ([], ["radius"])
    _, params, added = select_terms(model, res.params, batch, ShiftedPoisson(), max_terms=1)
    assert [a.term for a in added] == [term]
    if term == "z scale":
        scale = float(params.physical()["geometry.z_scale"])
        assert abs(scale - 1.05) < 0.01, scale


def test_matched_beads_raise_no_flag_and_held_out_beads_are_unbiased():
    model, batch = mismatched("matched", n=16)
    cal = learn_beads(model, batch, heldout=0.25, first_order=None)
    assert cal.atlas.flagged() == [], cal.atlas.checks
    _, _, added = select_terms(model, cal.result.params, batch.subset(torch.tensor(cal.beads.used)),
                               ShiftedPoisson(), max_terms=1)
    assert added == []
    rep = cal.heldout_report
    assert len(rep.beads) == 4 and abs(rep.deviance - 1) < 0.03
    c = rep.central()
    se = rep.z_crlb_nm[c] / np.sqrt(len(rep.beads))         # 4 beads: a robust spread is too noisy a yardstick
    assert (np.abs(rep.z_bias_nm[c]) < 4 * se).mean() > 0.9
    assert 0.6 < rep.summary()["z_spread_over_crlb"] < 1.5


def test_a_fitted_z_scale_recovers_a_piezo_scale():
    model, batch = mismatched("piezo scale", K=41, z_range_nm=1000.0)
    model = ForwardModel(model.psf, geometry=BeadStackGeometry(z_scale=True))
    res = fit_beads(model, batch, first_order=None)
    scale, sd = float(res.params.physical()["geometry.z_scale"]), float(res.std["geometry.z_scale"])
    assert abs(scale - 1.05) < 4 * sd and sd < 0.01


@pytest.mark.parametrize("kind,right", [("laser", PerSliceFactor), ("speckle", PerObservation)])
def test_the_right_photon_model_recovers_the_pupil_and_clears_the_atlas(kind, right):
    # Plan §9 M3: with the photon option the data need, the pupil comes back and the atlas is quiet;
    # the default's error is larger, and its atlas names the problem (test_the_atlas_names_…).
    model, batch = mismatched(kind)
    ok = ForwardModel(model.psf, intensity=right())
    res = fit_beads(ok, batch, first_order=None)
    atlas = residual_atlas(ok, res.params, batch, ShiftedPoisson())
    assert not [c for c in atlas.flagged() if c in SPECIFIC], atlas.checks
    err = (res.params.physical()["pupil.phase"] - TRUE_PHASE).double()
    assert (err.abs() / res.std["pupil.phase"]).max() < 4
    wrong = fit_beads(model, batch, first_order=None)
    err_wrong = (wrong.params.physical()["pupil.phase"] - TRUE_PHASE).double()
    assert err_wrong.pow(2).mean() > err.pow(2).mean()
