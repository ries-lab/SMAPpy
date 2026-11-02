"""A coefficient field: astigmatism and coma that vary over the chip (`model.field`,
`docs/uipsf/field-dependence.md`)."""
import pytest
import torch

import uipsf
from uipsf.model import Channel, FieldPolynomial
from uipsf.model.field import monomials

f64 = torch.float64
OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.518, n_med=1.33, pixel_size_nm=(110, 110))
PX = (110.0, 110.0)


def field_beads(n=24, truth_field=None, seed=0, photons=20000.0, K=9, roi=(11, 11), dtype=f64):
    """(model with a field, truth Params, Batch): beads spread over a 1000 px chip, astigmatism and coma
    varying as ``truth_field`` says ({(Noll, i, j): rad})."""
    from uipsf.simulate import parameters, simulate, template
    grid = uipsf.PupilGrid.build(OPTICS, 24, device="cpu", dtype=dtype)
    pupil = uipsf.ZernikePupil(grid, n_max=4)
    field = FieldPolynomial((500.0, 500.0), PX)
    model = uipsf.ForwardModel([Channel(uipsf.VectorPSF(grid, pupil, roi, sigma_nm=(30.0, 30.0)), field=field)])
    g = torch.Generator().manual_seed(seed)
    xyz = (torch.rand(n, 3, generator=g, dtype=f64) - 0.5) * torch.tensor([200.0, 200.0, 300.0])
    chip = torch.rand(n, 2, generator=g, dtype=f64) * 1000.0
    phase = torch.zeros(len(pupil.phase_modes))
    phase[:3] = torch.tensor([0.1, 0.6, 0.05])
    coefficients = torch.tensor([(truth_field or {}).get(t, 0.0) for t in field.layout])
    values = {"pupil.phase": phase, "psf.sigma": torch.tensor([30.0, 30.0]), "field.phase": coefficients,
              "emitter.xyz": xyz, "emitter.photons": torch.full((n,), photons),
              "emitter.background": torch.full((n,), 10.0)}
    b0 = template(n, uipsf.Batch.z_from_objective(torch.linspace(-600, 600, K)), roi, dtype=dtype)
    from dataclasses import replace
    b0 = replace(b0, field_xy=chip.to(dtype))
    truth = parameters(model, b0, values, dtype=dtype)
    return model, truth, simulate(model, truth, b0, seed=seed + 1)


def test_the_monomials_run_by_total_order_without_a_constant():
    assert monomials(2) == [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]


def test_an_emitters_pupil_is_the_centres_plus_the_polynomial_at_its_position():
    grid = uipsf.PupilGrid.build(OPTICS, 16, device="cpu", dtype=f64)
    pupil = uipsf.ZernikePupil(grid, n_max=4)
    field = FieldPolynomial((100.0, 200.0), PX, scale_um=11.0)            # 100 px = 11 µm = u of 1
    base = torch.linspace(0.1, 0.9, len(pupil.phase_modes), dtype=f64)
    coef = torch.zeros(len(field.layout), dtype=f64)
    coef[field.layout.index((6, 2, 0))] = 0.3                            # Noll 6 += 0.3 u²
    coef[field.layout.index((7, 0, 1))] = -0.2                           # Noll 7 −= 0.2 v
    xy = torch.tensor([[100.0, 200.0], [300.0, 300.0]], dtype=f64)       # the centre; u = 2, v = 1
    out = field.apply({"pupil.phase": base, "field.phase": coef}, pupil, xy)["pupil.phase"]
    assert out.shape == (2, len(base)) and torch.allclose(out[0], base)
    change = out[1] - base
    i6, i7 = pupil.phase_modes.index(6), pupil.phase_modes.index(7)
    assert change[i6].item() == pytest.approx(1.2) and change[i7].item() == pytest.approx(-0.2)
    assert torch.count_nonzero(change).item() == 2


def test_a_field_needs_the_emitters_chip_positions():
    model, truth, batch = field_beads(n=2)
    from dataclasses import replace
    with pytest.raises(ValueError, match="field_xy"):
        model(truth.physical(), replace(batch, field_xy=None))


def test_a_zero_field_is_the_model_without_one():
    model, truth, batch = field_beads(n=3)
    plain = uipsf.ForwardModel(model.psf)
    p = truth.physical()
    assert torch.allclose(model(p, batch), plain(p, batch))


def test_the_pupil_linearisation_matches_the_jacobian_of_the_field_coefficients():
    # the solver builds the field's columns through `pupil_linearisation`; they must be the true derivative
    model, truth, batch = field_beads(n=3, truth_field={(5, 1, 0): 0.1, (6, 0, 2): -0.2, (8, 0, 1): 0.15})
    p = truth.physical()
    groups = model.pupil_linearisation(p, batch)
    _, names, field, apply = groups[0]
    assert "field.phase" in names
    T = p["field.phase"].numel()
    eye = torch.eye(T, dtype=f64)

    def with_field(v):
        return {**p, "field.phase": v}
    dA = torch.stack([torch.func.jvp(lambda v: field(with_field(v)), (p["field.phase"],), (e,))[1] for e in eye])
    columns = apply(dA)                                                  # [T, E, K, Y, X]
    exact = torch.stack([torch.func.jvp(lambda v: model(with_field(v), batch)[:, 0], (p["field.phase"],), (e,))[1]
                         for e in eye])
    assert torch.allclose(columns, exact, rtol=1e-6, atol=1e-9 * exact.abs().max().item())


@pytest.mark.slow
def test_beads_over_the_chip_give_back_the_field_they_were_simulated_with():
    from uipsf.workflows import learn_beads
    truth_field = {(6, 1, 0): 0.15, (5, 0, 1): 0.15, (6, 2, 0): 0.1, (8, 1, 0): 0.08}
    model, truth, batch = field_beads(truth_field=truth_field)
    cal = learn_beads(model, batch, first_order=None)
    learnt = cal.shared()["field.phase"].detach()
    true = truth.physical()["field.phase"]
    assert torch.allclose(learnt, true, atol=0.02), (learnt, true)


@pytest.mark.slow
def test_the_report_finds_a_real_field_significant_and_measures_its_changes():
    from uipsf.diagnostics import field_report
    from uipsf.workflows import learn_beads
    model, truth, batch = field_beads(truth_field={(6, 1, 0): 0.15, (5, 0, 1): 0.15})
    cal = learn_beads(model, batch, first_order=None)
    rep = field_report(cal)
    assert rep.significant and rep.dof == 14
    # the largest change of Noll 6 across beads up to 500 px (55 µm) from the centre: 0.15 rad per 50 µm
    d6, err6 = rep.changes[6]
    assert abs(d6) == pytest.approx(0.15 * 55 / 50, rel=0.15) and err6 < 0.01
    assert "significant" in rep.summary()


@pytest.mark.slow
def test_the_report_finds_no_field_where_there_is_none():
    from uipsf.diagnostics import field_report
    from uipsf.workflows import learn_beads
    model, truth, batch = field_beads()
    rep = field_report(learn_beads(model, batch, first_order=None))
    assert not rep.significant
    assert all(abs(d) < 5 * e for d, e in rep.changes.values())


def test_a_model_file_keeps_the_field_and_draws_the_psf_where_it_is_asked(tmp_path):
    from uipsf.export import load_model, save_model
    from uipsf.workflows import learn_beads
    model, truth, batch = field_beads(n=4, truth_field={(6, 1, 0): 0.3}, dtype=torch.float32)
    cal = learn_beads(model, batch, first_order=None)
    path = tmp_path / "field.uipsf.h5"
    save_model(cal, path)
    import h5py
    with h5py.File(path) as f:
        assert int(f.attrs["version"]) == 4
    learned = load_model(path, device="cpu")
    assert learned.model.psf is learned.model.channels[0].psf and learned.model.channels[0].field is not None
    xyz = [[0.0, 0.0, 0.0]]
    centre = learned.psf(xyz, roi_px=11)
    assert torch.allclose(centre, cal.psf(xyz, roi_px=11), atol=1e-6)
    far = learned.psf(xyz, roi_px=11, field_xy=(1000.0, 500.0))
    assert not torch.allclose(centre, far, atol=1e-4)
    assert torch.allclose(far, cal.psf(xyz, roi_px=11, field_xy=(1000.0, 500.0)), atol=1e-6)


@pytest.mark.slow
def test_a_learnt_field_becomes_node_splines_that_fit_spots_at_the_edge_without_bias():
    import numpy as np
    from smappy.calibrate.pupil import spline_calibration
    from smappy.psf import SplinePSF

    from uipsf.workflows import learn_beads
    model, truth, batch = field_beads(truth_field={(6, 1, 0): 0.4, (5, 0, 1): 0.3, (8, 1, 0): 0.15},
                                      dtype=torch.float32)
    cal = learn_beads(model, batch, first_order=None)
    spline = spline_calibration(cal, 20.0, 600.0, 15)
    field = spline.field_splines
    assert field is not None and field.shape[0] >= 3
    # spots simulated with the true model near a corner of the beads' span, fitted with and without the field
    xy = np.asarray(cal.beads_field_xy())
    corner = xy.max(0) - 20
    rng = np.random.default_rng(0)
    n = 1500
    xyz = np.c_[rng.uniform(-50, 50, (n, 2)), rng.uniform(-400, 400, n)]
    from uipsf.export import sample_psf
    true = {k: v for k, v in truth.physical().items() if truth.specs[k].scope == "shared"}
    mu = sample_psf(model, true, xyz, roi_px=11, field_xy=tuple(corner)).double().numpy()
    rois = rng.poisson(mu * 5000 + 10).astype(np.float32)
    psf = SplinePSF(spline)
    fielded = psf.unpack(psf.fit(rois, 50, 2, chip_xy=np.tile(corner, (n, 1))))
    plain = psf.unpack(psf.fit(rois, 50, 2))
    bins = np.digitize(xyz[:, 2], np.arange(-400, 401, 100))

    def bias(r):
        e = r["z_nm"] - xyz[:, 2]
        return np.sqrt(np.mean([np.median(e[bins == q]) ** 2 for q in np.unique(bins)]))
    assert bias(fielded) < 6 and bias(plain) > 2 * bias(fielded), (bias(fielded), bias(plain))


def test_a_spline_drawn_at_a_depth_in_the_sample_keeps_its_field_nodes():
    from smappy.calibrate.depth import depth_spline
    from smappy.calibrate.pupil import spline_calibration

    from uipsf.export import LearnedPSF, describe
    model, truth, _ = field_beads(n=2, truth_field={(6, 1, 0): 0.5}, dtype=torch.float32)
    shared = {k: v for k, v in truth.physical().items() if truth.specs[k].scope == "shared"}
    beads = [{"state": "used", "field_x": x, "field_y": y} for x, y in ((0.0, 0.0), (1000.0, 1000.0))]
    learned = LearnedPSF(model, shared, {}, describe(model), beads=beads)
    spline = spline_calibration(learned, 50.0, 500.0, 11)
    assert spline.field_splines is not None
    deep = depth_spline(spline, 2000.0, learned=learned)
    assert deep.field_splines.shape == spline.field_splines.shape
    assert deep.field_splines.coeff.shape[2:] == deep.coeff.shape[1:]
    # the nodes differ from each other at depth too: the field is still there
    nodes = torch.as_tensor(deep.field_splines.coeff)
    assert not torch.allclose(nodes[0], nodes[-1])


def test_a_spline_calibration_keeps_the_grid_limit_warning(monkeypatch):
    import smappy.field_splines as fs
    from smappy.calibrate.pupil import spline_calibration

    from uipsf.export import LearnedPSF, describe
    model, truth, _ = field_beads(n=2, truth_field={(6, 1, 0): 0.5}, dtype=torch.float32)
    shared = {k: v for k, v in truth.physical().items() if truth.specs[k].scope == "shared"}
    beads = [{"state": "used", "field_x": x, "field_y": y} for x, y in ((0.0, 0.0), (1000.0, 1000.0))]
    learned = LearnedPSF(model, shared, {}, describe(model), beads=beads)
    monkeypatch.setattr(fs, "GRIDS", (1, 2))      # the largest grid 2 × 2: too coarse for 0.5 rad per 50 µm
    with pytest.warns(UserWarning, match="grid of splines"):
        spline = spline_calibration(learned, 50.0, 500.0, 11)
    assert spline.field_splines.shape == (2, 2)
    assert "2 × 2" in spline.parameters["field_warning"] and spline.parameters["field_blend_error"] > 1e-6
