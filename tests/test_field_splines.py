"""Splines that vary over the field, blended per spot (`smappy.field_splines`, the fitter's
`fit_cspline_field`; docs/uipsf/field-dependence.md)."""
import numpy as np
import pytest

from smappy import _fit3d
from smappy.calibrate.core import spline_coefficients
from smappy.field_splines import FieldSplines, bilinear, blend, blend_error, grid_size
from smappy.io.calibration import SplineCalibration, load_spline_calibration, save_spline_calibration
from smappy.psf import SplinePSF

NZ, BOX, ROI = 41, 15, 11


def astigmatic(ellipticity: float) -> np.ndarray:
    """A Gaussian stack whose two focal lines move apart with ``ellipticity``: a stand-in for a PSF whose
    astigmatism changes over the field."""
    z = np.linspace(-1, 1, NZ)[:, None, None]
    y, x = np.mgrid[:BOX, :BOX] - (BOX - 1) / 2
    sx = 1.3 + 0.8 * (z - ellipticity) ** 2
    sy = 1.3 + 0.8 * (z + ellipticity) ** 2
    psf = np.exp(-x ** 2 / (2 * sx ** 2) - y ** 2 / (2 * sy ** 2)) / (2 * np.pi * sx * sy)
    return psf.astype(np.float32)


def field_of(ellipticity_at, xs, ys) -> FieldSplines:
    return FieldSplines(np.stack([spline_coefficients(astigmatic(ellipticity_at(x, y))) for y in ys for x in xs]),
                        xs, ys)


def spots(psf, rng, n, photons=5000.0, background=10.0):
    """ROIs of ``psf`` with emitters within half a pixel of the centre at z planes 10..30."""
    k = rng.integers(10, 31, n)
    s = (BOX - ROI) // 2
    mu = psf[k, s:s + ROI, s:s + ROI] * photons + background
    return rng.poisson(mu).astype(np.float32), k


def test_the_weights_are_bilinear_and_a_position_outside_takes_its_edge():
    field = field_of(lambda x, y: 0.3, np.array([0.0, 100.0, 200.0]), np.array([0.0, 50.0]))
    nodes, w = bilinear(field, [[25.0, 10.0], [500.0, -20.0], [100.0, 50.0]])
    assert nodes[0].tolist() == [0, 1, 3, 4] and np.allclose(w[0], [0.6, 0.2, 0.15, 0.05])
    assert np.allclose(w.sum(1), 1)
    assert np.isclose(w[1] @ (nodes[1] == 2), 1)                 # beyond the corner (200, 0): node 2 alone
    assert np.isclose(w[2] @ (nodes[2] == 4), 1)                 # on a node: that node alone


def test_one_node_is_the_calibration_of_one_spline():
    field = field_of(lambda x, y: 0.3, np.array([5.0]), np.array([7.0]))
    nodes, w = bilinear(field, [[0.0, 0.0], [900.0, 900.0]])
    assert (nodes == 0).all() and np.allclose(w, [[1, 0, 0, 0]] * 2)


def test_the_blended_fit_is_the_fit_with_the_blended_spline():
    rng = np.random.default_rng(0)
    xs, ys = np.array([0.0, 400.0]), np.array([0.0, 400.0])
    field = field_of(lambda x, y: 0.1 + 0.4 * x / 400, xs, ys)
    rois, _ = spots(astigmatic(0.3), rng, 200)
    where = np.tile([[300.0, 120.0]], (len(rois), 1))
    nodes, w = bilinear(field, where)
    a = _fit3d.fit_cspline_field(rois, field.coeff, nodes, w, NZ / 2, 50, 2)
    b = _fit3d.fit_cspline(rois, blend(field, where[0]), NZ / 2, 50, 2)
    assert np.allclose(a[0], b[0], rtol=1e-4, atol=1e-3)


def test_spots_over_a_varying_field_are_unbiased_with_the_field_and_biased_without():
    rng = np.random.default_rng(1)
    xs = ys = np.linspace(0, 1000, 5)
    ellipticity = lambda x, y: 0.1 + 0.5 * x / 1000                       # noqa: E731
    field = field_of(ellipticity, xs, ys)
    centre = SplineCalibration(blend(field, (500, 500)), 50.0, NZ // 2, x0=(BOX - 1) / 2)
    with_field = SplineCalibration(centre.coeff, 50.0, NZ // 2, x0=(BOX - 1) / 2, field_splines=field)
    x = 950.0                                                              # near the field's edge
    rois, k = spots(astigmatic(ellipticity(x, 300.0)), rng, 2000)
    chip = np.tile([[x, 300.0]], (len(rois), 1))
    z_true = (NZ // 2 - k) * 50.0
    plain = SplinePSF(centre).unpack(SplinePSF(centre).fit(rois, 50, 2))
    model = SplinePSF(with_field)
    assert model.field_dependent and not SplinePSF(centre).field_dependent
    fielded = model.unpack(model.fit(rois, 50, 2, chip_xy=chip))

    def bias(r):
        """The z bias, rms over the planes: it changes sign with z, so one median over all hides it."""
        return np.sqrt(np.mean([np.median((r["z_nm"] - z_true)[k == q]) ** 2 for q in np.unique(k)]))
    assert bias(fielded) < 8 and bias(plain) > 3 * bias(fielded), (bias(fielded), bias(plain))


def test_a_calibration_file_keeps_its_field(tmp_path):
    field = field_of(lambda x, y: 0.1 + y / 1000, np.array([0.0, 500.0]), np.array([0.0, 250.0, 500.0]))
    cal = SplineCalibration(blend(field, (250, 250)), 50.0, NZ // 2, x0=(BOX - 1) / 2, field_splines=field)
    save_spline_calibration(tmp_path / "f.h5", cal)
    back = load_spline_calibration(tmp_path / "f.h5")
    assert back.field_splines.shape == (3, 2)
    assert np.array_equal(back.field_splines.coeff, field.coeff) and np.array_equal(back.field_splines.ys, field.ys)
    plain = SplineCalibration(cal.coeff, 50.0, NZ // 2, x0=(BOX - 1) / 2)
    save_spline_calibration(tmp_path / "p.h5", plain)
    assert load_spline_calibration(tmp_path / "p.h5").field_splines is None


def test_the_grid_is_the_smallest_whose_blend_keeps_to_the_tolerance():
    flat = lambda x, y: astigmatic(0.3)                                   # noqa: E731
    assert grid_size(flat, (0, 0), (1000, 1000)) == 1
    linear = lambda x, y: astigmatic(0.1 + 0.3 * x / 1000)                # noqa: E731
    n = grid_size(linear, (0, 0), (1000, 1000))
    assert n > 1 and blend_error(linear, (0, 0), (1000, 1000), n) <= 0.005


def test_the_nodes_must_fill_the_grid():
    with pytest.raises(ValueError, match="grid"):
        FieldSplines(np.zeros((3, 64, 2, 2, 2), np.float32), [0.0, 1.0], [0.0, 1.0])
