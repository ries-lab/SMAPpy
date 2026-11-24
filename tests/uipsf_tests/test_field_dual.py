"""Field dependence with two channels of a split chip (`docs/uipsf/field-dependence.md` §13.2): each channel its
own `FieldPolynomial` of the main channel's chip position, learnt from beads; each channel's node splines over
its own part of the chip in the dual calibration; and SMAPpy's two-channel fitter blending each channel's nodes
where its ROI is (`fit_cspline_field_global`)."""
import numpy as np
import pytest
import torch

import uipsf
from uipsf.model import Channel, FieldPolynomial
from uipsf.model.channels import inverse_map
from uipsf_tests.conftest import two_channel_beads

pytest.importorskip("smappy")
from smappy.calibrate.dual import split_geometry  # noqa: E402 -- after the skip

f64 = torch.float64


def _corner_error(field, learnt, true, corners) -> dict:
    """{Noll: largest change of the mode at ``corners`` [N, 2] between ``learnt`` and ``true`` coefficients}."""
    error = field.basis(torch.as_tensor(corners, dtype=f64)) * (learnt.double() - true.double())
    out = {}
    for j, _ in field.terms:
        rows = [i for i, (m, _, _) in enumerate(field.layout) if m == j]
        out[j] = float(error[:, rows].sum(1).abs().max())
    return out


@pytest.mark.slow
def test_dual_beads_give_back_each_channels_own_field(optics):
    from uipsf.workflows import learn_beads
    fields = ({(6, 1, 0): 0.5, (5, 0, 1): 0.6, (8, 1, 0): 0.2},
              {(6, 1, 0): -0.4, (5, 0, 1): 0.5, (7, 0, 1): 0.3})
    model, truth, batch = two_channel_beads(optics, n=24, kind="ratiometric", fields=fields, roi=(11, 11),
                                            photons=20000.0)
    cal = learn_beads(model, batch, first_order=None)
    corners = [[x, y] for x in (40.0, 440.0) for y in (40.0, 240.0)]
    for c, pre in enumerate(("", "ch1.")):
        errors = _corner_error(model.channels[c].field, cal.shared()[pre + "field.phase"].detach(),
                               truth.physical()[pre + "field.phase"], corners)
        assert max(errors.values()) < 0.05, (c, errors)
    from uipsf.diagnostics import field_report
    assert field_report(cal, 0).significant and field_report(cal, 1).significant


# ------------------------------------------------------------------ a known two-channel field, through SMAPpy
PIXEL = 110.0
SHAPE = (600, 300)                   # (height, width): the upper half main, the lower half mirrored
ANGLE = np.deg2rad(0.3)
# secondary → main chip pixels: y mirrored across the split, a small turn and shift
TRANSFORM = (np.array([[np.cos(ANGLE), -np.sin(ANGLE), 0.6], [np.sin(ANGLE), np.cos(ANGLE), -0.4], [0, 0, 1]])
             @ np.array([[1.0, 0, 0], [0, -1, 599], [0, 0, 1]]))
GEOMETRY = split_geometry("up-down mirrored", "upper", SHAPE, split_position=300)
CENTRE = (150.0, 150.0)
FIELDS = ({(6, 1, 0): 1.6, (5, 0, 1): 1.2, (8, 1, 0): 0.4}, {(6, 1, 0): -1.2, (5, 0, 1): 1.5, (7, 0, 1): 0.5})


def _model():
    """(LearnedPSF, model, shared): a ratiometric pair, each channel astigmatic with its own field; the
    'beads' over the main half."""
    from uipsf.export import LearnedPSF, describe
    channels = []
    for c, wavelength in enumerate((670.0, 700.0)):
        optics = uipsf.Optics(NA=1.43, wavelength_nm=wavelength, n_imm=1.516, n_med=1.335,
                              pixel_size_nm=(PIXEL, PIXEL))
        grid = uipsf.PupilGrid.build(optics, 32, device="cpu", dtype=f64)
        psf = uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=4), (15, 15), upsample=2)
        channels.append(Channel(psf, transform=TRANSFORM if c else None,
                                field=FieldPolynomial(CENTRE, (PIXEL, PIXEL))))
    model = uipsf.ForwardModel(channels)
    n = len(channels[0].psf.pupil.phase_modes)
    shared = {"pupil.phase": torch.zeros(n, dtype=f64), "ch1.pupil.phase": torch.zeros(n, dtype=f64)}
    shared["pupil.phase"][1] = 0.9
    shared["ch1.pupil.phase"][1] = -0.8
    for c, pre in enumerate(("", "ch1.")):
        shared[pre + "pupil.magnitude"] = torch.zeros(len(channels[0].psf.pupil.magnitude_modes), dtype=f64)
        shared[pre + "psf.sigma"] = torch.tensor([30.0, 30.0], dtype=f64)
        shared[pre + "field.phase"] = torch.tensor([FIELDS[c].get(t, 0.0) for t in channels[c].field.layout],
                                                   dtype=f64)
    shared["ch1.transform"] = torch.tensor((TRANSFORM / TRANSFORM[2, 2]).ravel()[:8])
    shared["ch1.share"] = torch.tensor(0.4, dtype=f64)
    shared["ch1.dz"] = torch.tensor(30.0, dtype=f64)
    beads = [{"state": "used", "field_x": x, "field_y": y} for x in (15.0, 285.0) for y in (15.0, 285.0)]
    learned = LearnedPSF(model, shared, {}, describe(model), beads=beads, geometry=dict(GEOMETRY))
    return learned, model, shared


def _frames(model, shared, emitters, n, rng, background=15.0):
    """Split frames: each emitter in both halves where the transform puts it, each channel's PSF drawn with
    its field at the emitter's place on the main chip."""
    from uipsf.export import sample_psf
    H = torch.tensor(TRANSFORM / TRANSFORM[2, 2])
    mean = np.full(SHAPE, background)
    for x, y, z, photons in emitters:
        second = inverse_map(H, torch.tensor([[x, y]], dtype=f64))[0].numpy()
        for c, (cx, cy) in enumerate(((x, y), tuple(second))):
            ix, iy = int(round(cx)), int(round(cy))
            local = [[(cx - ix) * PIXEL, (cy - iy) * PIXEL, z]]
            patch = sample_psf(model, shared, local, roi_px=15, channel=c, field_xy=(x, y))[0].double().numpy()
            mean[iy - 7:iy + 8, ix - 7:ix + 8] += photons * patch
    return rng.poisson(np.broadcast_to(mean, (n, *SHAPE))).astype(np.float32)


@pytest.mark.slow
def test_a_two_channel_field_becomes_node_splines_per_channel_that_the_global_fitter_blends(tmp_path):
    import copy

    from smappy.calibrate.dual import load_dual_color_calibration, save_dual_color_calibration
    from smappy.calibrate.pupil import dual_calibration
    from smappy.detect import AbsoluteCutoff, DoGFilter, PeakFinder
    from smappy.dualfit import LINK_XYZ
    from smappy.locs import in_pixels
    from smappy.metadata import CameraMetadata
    from smappy.multichannel import MultiChannelEngine
    from smappy.pipeline import FitSettings
    from smappy.psf import GlobalSplinePSF
    learned, model, shared = _model()
    cal = dual_calibration(learned, 40.0, 480.0, roi_size=15)
    main, second = cal.main.field_splines, cal.secondary.field_splines
    assert main is not None and second is not None and main.shape[0] >= 3 and second.shape[0] >= 3
    # each grid over its own channel's chip: the secondary's on the lower half, where the beads' partners are
    assert main.ys.max() < 300 and second.ys.min() > 300
    path = tmp_path / "pair_3dcal.h5"
    save_dual_color_calibration(path, cal)
    cal = load_dual_color_calibration(path)
    # float16 on disk, as a single calibration's nodes (`io.calibration.write_field_splines`)
    assert np.allclose(cal.secondary.field_splines.coeff, second.coeff, rtol=1e-3, atol=1e-7)
    assert cal.main.parameters["field_grid"] == list(main.shape)
    # emitters near the main half's corner, against the field's centre, fitted with the nodes and without
    rng = np.random.default_rng(0)
    zs = np.linspace(-360, 360, 7)
    emitters = [(30.0 + 20 * (k % 3), 30.0 + 20 * (k // 3), z, 15000.0) for k, z in enumerate(zs)]
    frames = _frames(model, shared, emitters, 25, rng)
    camera = CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=PIXEL / 1000, em_on=False)
    layout = cal.layout()
    plain = copy.deepcopy(cal)
    plain.main.field_splines = plain.secondary.field_splines = None

    def fitted(calibration):
        psf = GlobalSplinePSF(tuple(calibration.layer_splines()), LINK_XYZ)
        engine = MultiChannelEngine(camera, PeakFinder(DoGFilter(1.2), AbsoluteCutoff(20.0)), psf, layout,
                                    FitSettings(roisize=13), detection="pairs")
        engine.push(layout.cut(frames))
        locs = engine.flush()
        x, y = in_pixels(locs, "x_pix", "y_pix")
        errors = []
        for x0, y0, z0, _ in emitters:
            mine = np.hypot(x - x0, y - y0) < 3
            assert mine.sum() >= 12
            errors.append(float(np.median(locs["z_nm"][mine])) - z0)
        e = np.asarray(errors)
        return float(np.sqrt(np.mean((e - e.mean()) ** 2)))       # the bias's shape: the focus is a constant

    with_field, without = fitted(cal), fitted(plain)
    assert GlobalSplinePSF(tuple(cal.layer_splines()), LINK_XYZ).field_dependent
    assert with_field < 10.0 and without > 2.5 * with_field, (with_field, without)


# ------------------------------------------------------------------ the fitter, with splines known exactly
def _stack(sx_slope, nz=21, size=13):
    """An astigmatic Gaussian stack, (nz, size, size), whose widths change with z by ``sx_slope``."""
    z = np.linspace(-1, 1, nz)[:, None, None]
    y, x = np.mgrid[:size, :size] - (size - 1) / 2
    sx, sy = 1.3 + sx_slope * z, 1.3 - sx_slope * z
    psf = np.exp(-x ** 2 / (2 * sx ** 2) - y ** 2 / (2 * sy ** 2))
    return (psf / psf.sum(axis=(1, 2)).max()).astype(np.float32)


def _calibration(stack, field=None):
    from smappy.calibrate.core import spline_coefficients
    from smappy.io.calibration import SplineCalibration
    return SplineCalibration(spline_coefficients(stack), 50.0, (len(stack) - 1) / 2, x0=(stack.shape[1] - 1) / 2,
                             psf=stack, field_splines=field)


def test_the_global_fitter_fits_each_channel_with_the_node_splines_at_its_chip_position():
    from smappy.dualfit import LINK_XYZ
    from smappy.field_splines import FieldSplines
    from smappy.psf import GlobalSplinePSF
    a, b, c = _stack(0.5), _stack(-0.4), _stack(0.2)
    ca, cb, cc = (_calibration(s) for s in (a, b, c))
    # channel 0: a 2 × 2 grid whose first node is a, the others b; channel 1: one spline, c
    nodes = np.stack([ca.coeff, cb.coeff, cb.coeff, cb.coeff])
    fielded = GlobalSplinePSF((_calibration(a, FieldSplines(nodes, [0.0, 100.0], [0.0, 100.0])), cc), LINK_XYZ)
    rng = np.random.default_rng(0)
    n = 50
    rois = np.stack([rng.poisson(2000 * a[8] + 5), rng.poisson(1500 * c[8] + 5)], 1).astype(np.float32)
    rois = np.repeat(rois[None], 1, 0).reshape(1, 2, 13, 13).repeat(n, 0)
    link = np.zeros((n, 2, 2, 5), np.float32)
    link[:, 1] = 1.0
    plain_a = GlobalSplinePSF((ca, cc), LINK_XYZ).fit(rois, link)
    plain_b = GlobalSplinePSF((cb, cc), LINK_XYZ).fit(rois, link)
    on_a = np.tile([[0.0, 0.0], [500.0, 500.0]], (n, 1, 1))
    on_b = np.tile([[100.0, 0.0], [500.0, 500.0]], (n, 1, 1))
    assert np.allclose(fielded.fit(rois, link, chip_xy=on_a).theta, plain_a.theta, atol=1e-4)
    assert np.allclose(fielded.fit(rois, link, chip_xy=on_b).theta, plain_b.theta, atol=1e-4)
    # without chip positions, the grid's centre spline (the calibration's own), as before
    assert np.allclose(fielded.fit(rois, link).theta, plain_a.theta, atol=1e-4)
    with pytest.raises(ValueError, match="chip_xy"):
        fielded.fit(rois, link, chip_xy=on_a[:, :1])


def test_a_mirrored_spline_mirrors_every_node_and_keeps_where_they_are():
    from smappy.calibrate.dual import mirrored_spline
    from smappy.field_splines import FieldSplines
    a, b = _stack(0.5), _stack(-0.4)
    ca, cb = _calibration(a), _calibration(b)
    cal = _calibration(a, FieldSplines(np.stack([ca.coeff, cb.coeff]), [0.0, 100.0], [0.0]))
    flipped = mirrored_spline(cal, axis=1)
    assert np.allclose(flipped.field_splines.coeff[1], mirrored_spline(cb, axis=1).coeff)
    assert np.array_equal(flipped.field_splines.xs, cal.field_splines.xs)
    assert np.allclose(flipped.coeff, mirrored_spline(ca, axis=1).coeff)


@pytest.mark.slow
def test_a_dual_field_drawn_at_a_depth_in_the_sample_keeps_each_channels_nodes(monkeypatch):
    from smappy.calibrate import depth
    from smappy.calibrate.pupil import dual_calibration
    learned, _, _ = _model()
    cal = dual_calibration(learned, 100.0, 300.0, roi_size=11)
    monkeypatch.setattr(depth, "model_of", lambda calibration, channels=1: learned)
    deep = depth.dual_depth_calibration(cal, 1500.0)
    for before, after in ((cal.main, deep.main), (cal.secondary, deep.secondary)):
        assert after.field_splines.shape == before.field_splines.shape
        assert np.array_equal(after.field_splines.ys, before.field_splines.ys)
        assert after.field_splines.coeff.shape[1:] == after.coeff.shape
        nodes = after.field_splines.coeff
        assert not np.allclose(nodes[0], nodes[-1])           # the field is still there at depth
