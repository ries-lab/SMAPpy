"""A SMAP spline of x-mirrored beads, converted on load to the camera's
orientation.

The old way kept the spline as it was and flipped every ROI in x before the
fit (and the fitted x back after it).  That path is gone from the code, so
these tests reproduce it with numpy -- flip the ROI, evaluate, flip back --
and hold the converted spline to it."""
import h5py
import numpy as np
import pytest
import scipy.io

from smappy import _fit3d
from smappy.calibrate.core import spline_coefficients
from smappy.io.calibration import (
    SplineCalibration, _smap_flipped, _unmirror_x, evaluate_spline,
    load_spline_calibration, save_spline_calibration,
)


def psf_volume(nz=41, size=17):
    """Astigmatic, with a side lobe that moves along +x with z: a PSF that is
    not symmetric in x, so a wrong flip cannot pass for a right one."""
    z, y, x = np.mgrid[:nz, :size, :size].astype(float)
    t = (z - (nz - 1) / 2) / 12
    sx, sy = 1.2 * np.sqrt(1 + (t - .6) ** 2), 1.2 * np.sqrt(1 + (t + .6) ** 2)
    c = size // 2
    g = np.exp(-.5 * (((x - c) / sx) ** 2 + ((y - c) / sy) ** 2)) / (2 * np.pi * sx * sy)
    lobe = 0.3 * np.exp(-.5 * (((x - c - 2.5 - 1.5 * t) / 1.0) ** 2
                               + ((y - c) / 1.3) ** 2)) / (2 * np.pi * 1.3)
    v = g + lobe
    return v / v.sum(axis=(1, 2), keepdims=True)


def mirrored_calibration(size):
    """``size`` knots across: nx = size - 1 cells."""
    vol = psf_volume(size=size)
    return SplineCalibration(spline_coefficients(vol), 10., 20., x0=(size - 1) / 2,
                             psf=vol, parameters={})


def through_flipped_roi(cal, x, y, z, roisize):
    """What the fitter modelled for a mirrored spline: the ROI flipped in x."""
    return evaluate_spline(cal, roisize - 1 - x, y, z, roisize)[:, ::-1]


@pytest.mark.parametrize("size", [17, 18])          # nx = 16 (even), 17 (odd)
def test_the_converted_spline_is_the_mirrored_one_seen_through_a_flipped_roi(size):
    cal = mirrored_calibration(size)
    conv = _unmirror_x(cal)
    rng = np.random.default_rng(size)
    for roisize in (7, 11, 13, 15):
        for _ in range(100):
            # past the ROI's edges too, where the grid's ends are clamped
            x, y, z = rng.uniform(-3, roisize + 2), rng.uniform(0, roisize - 1), rng.uniform(0, 39.9)
            old = through_flipped_roi(cal, x, y, z, roisize)
            new = evaluate_spline(conv, x, y, z, roisize)
            assert np.abs(new - old).max() <= 1e-5 * np.abs(old).max()


def test_an_odd_grid_gains_one_cell_and_the_knots_follow():
    cal = mirrored_calibration(18)                   # nx = 17
    conv = _unmirror_x(cal)
    assert cal.coeff.shape[-1] == 17 and conv.coeff.shape[-1] == 18
    # the knots, reversed, and one more at the end to stay nx + 1 across
    assert conv.psf.shape == (41, 18, 19)
    assert np.array_equal(conv.psf[..., :18], cal.psf[..., ::-1])
    assert conv.x0 == 17 - cal.x0
    assert "x-flipped" in conv.parameters["smappy_conversion"]
    # an even grid keeps its size
    assert _unmirror_x(mirrored_calibration(17)).coeff.shape == (64, 40, 16, 16)


def test_flipping_the_signs_of_the_odd_terms_alone_is_not_the_conversion():
    """S(nx - u) runs each cell backwards, t -> 1 - t, not t -> -t."""
    cal = mirrored_calibration(17)
    c = cal.coeff.reshape(4, 4, 4, *cal.coeff.shape[1:]).copy()
    c[:, :, 1::2] *= -1
    naive = SplineCalibration(np.ascontiguousarray(c[..., ::-1].reshape(cal.coeff.shape)), 10., 20.)
    old = through_flipped_roi(cal, 6.3, 6.0, 20.4, 13)
    assert np.abs(evaluate_spline(naive, 6.3, 6.0, 20.4, 13) - old).max() > 0.1 * old.max()
    assert np.abs(evaluate_spline(_unmirror_x(cal), 6.3, 6.0, 20.4, 13) - old).max() < 1e-5 * old.max()


@pytest.mark.parametrize("size", [17, 18])
def test_fits_through_the_converted_spline_match_the_flipped_path(size):
    """x, y, z and photons the same to the size of float32 rounding: a few
    ten-thousandths of a pixel, a hundredth of a z plane at most."""
    cal = mirrored_calibration(size)
    conv = _unmirror_x(cal)
    rng = np.random.default_rng(7)
    roisize, n = 13, 400
    truth = np.c_[rng.uniform(5, 7, n), rng.uniform(5, 7, n), rng.uniform(800, 5000, n),
                  rng.uniform(2, 30, n), rng.uniform(8, 32, n)]   # x, y, N, bg, z index
    rois = np.stack([rng.poisson(np.clip(N * evaluate_spline(conv, x, y, z, roisize) + b, 0, None))
                     for x, y, N, b, z in truth]).astype(np.float32)   # camera orientation
    old = _fit3d.fit_cspline(np.ascontiguousarray(rois[:, :, ::-1]), cal.coeff, 20., 50, 0)[0]
    new = _fit3d.fit_cspline(rois, conv.coeff, 20., 50, 0)[0]
    dx = np.abs((roisize - 1 - old[:, 0]) - new[:, 0])
    assert dx.max() < 2e-3 and np.median(dx) < 1e-5
    assert np.abs(old[:, 1] - new[:, 1]).max() < 2e-3
    assert np.abs(old[:, 4] - new[:, 4]).max() < 0.05
    assert np.max(np.abs(old[:, 2] - new[:, 2]) / old[:, 2]) < 1e-3
    # and the converted spline finds the truth
    assert np.sqrt(np.mean((new[:, 0] - truth[:, 0]) ** 2)) < 0.1


def _matlab_coeff(coeff):
    """(64, nz, ny, nx) -> SMAP's (ny, nx, nz, 64), py and px digits swapped."""
    idx = np.arange(64)
    swapped = coeff[16 * (idx // 16) + 4 * (idx % 4) + (idx // 4) % 4]
    return np.ascontiguousarray(swapped.transpose(2, 3, 1, 0))


def write_smap_file(path, cal, emmirror, em_on):
    """A ``_3dcal.mat`` as SMAP's bead calibrator writes it, its bead stack a
    sample wider than the knots on every side."""
    stack = np.pad(cal.psf, 1, mode="edge")                       # (nz+3, ny+3, nx+3)
    sxy = {"cspline": {"coeff": _matlab_coeff(cal.coeff), "dz": cal.dz, "z0": cal.z0,
                       "x0": cal.x0},
           "PSF": stack.transpose(1, 2, 0)}                       # MATLAB (y, x, z)
    if em_on is not None:
        sxy["EMon"] = em_on
    scipy.io.savemat(str(path), {"SXY": sxy, "parameters": {"emmirror": emmirror, "dz": 10.0}})


def test_a_smap_file_of_mirrored_beads_loads_in_the_cameras_orientation(tmp_path):
    cal = mirrored_calibration(18)
    write_smap_file(tmp_path / "m_3dcal.mat", cal, emmirror=3, em_on=1)
    loaded = load_spline_calibration(tmp_path / "m_3dcal.mat")
    assert np.allclose(loaded.coeff, _unmirror_x(cal).coeff, rtol=1e-6, atol=1e-9)
    assert loaded.em_on is True and "smappy_conversion" in loaded.parameters
    # the bead stack is turned with the spline: knot k is still sample k + 1
    _, nz, ny, nx = loaded.coeff.shape
    assert np.allclose(loaded.coeff[0, :, :, :nx - 1],
                       loaded.psf[1:1 + nz, 1:1 + ny, 1:nx], atol=1e-6)


def test_converting_a_smap_file_to_h5_is_a_load_and_a_save(tmp_path):
    cal = mirrored_calibration(17)
    write_smap_file(tmp_path / "m_3dcal.mat", cal, emmirror=1, em_on=1)
    mat = load_spline_calibration(tmp_path / "m_3dcal.mat")
    save_spline_calibration(tmp_path / "m.h5", mat)
    h5 = load_spline_calibration(tmp_path / "m.h5")
    assert np.array_equal(h5.coeff, mat.coeff)
    assert (h5.dz, h5.z0, h5.x0, h5.em_on) == (mat.dz, mat.z0, mat.x0, True)
    assert h5.parameters["emmirror"] == 1 and "smappy_conversion" in h5.parameters
    assert h5.psf is None              # SMAP's bead stack is not the spline's knots


def test_an_h5_saved_mirrored_by_an_older_version_is_converted_on_load(tmp_path):
    cal = mirrored_calibration(18)
    save_spline_calibration(tmp_path / "old.h5", cal)
    with h5py.File(tmp_path / "old.h5", "a") as f:
        f.attrs["em_mirror"] = True
    loaded = load_spline_calibration(tmp_path / "old.h5")
    assert np.array_equal(loaded.coeff, _unmirror_x(cal).coeff)
    assert np.array_equal(loaded.psf, _unmirror_x(cal).psf)
    # saved again, it stays converted: no second flip
    save_spline_calibration(tmp_path / "again.h5", loaded)
    assert np.array_equal(load_spline_calibration(tmp_path / "again.h5").coeff, loaded.coeff)


@pytest.mark.parametrize("emmirror,em_on,flipped", [
    (0, 0, False),      # never set: nothing done
    (2, 0, False),      # "no mirror" -- once read as a bool, and flipped
    (1, 0, False),      # "from metadata", and the metadata said no EM gain
    (1, 1, True),       # ... and it said EM gain
    (3, 1, True),       # "EM gain used (mirrored)"
    (1, None, True),    # no record of the flip: 1 was the old "mirrored" checkbox
    (3, None, True),
    (2, None, False),
    (None, 1, False),   # not SMAP's bead calibrator: nothing says it flipped
])
def test_smaps_record_of_the_flip_decides_over_its_popup(emmirror, em_on, flipped):
    assert _smap_flipped(emmirror, em_on, "x_3dcal.mat") is flipped


def test_a_popup_and_a_record_that_disagree_are_reported():
    with pytest.warns(UserWarning, match="emmirror=2 but EMon=1"):
        assert _smap_flipped(2, 1, "x_3dcal.mat") is True


def test_a_smap_file_saved_with_no_mirror_is_not_flipped(tmp_path):
    cal = mirrored_calibration(18)
    write_smap_file(tmp_path / "n_3dcal.mat", cal, emmirror=2, em_on=0)
    loaded = load_spline_calibration(tmp_path / "n_3dcal.mat")
    assert np.allclose(loaded.coeff, cal.coeff, rtol=1e-6, atol=1e-9)
    assert "smappy_conversion" not in loaded.parameters
