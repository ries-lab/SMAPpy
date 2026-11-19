"""SMAP's global two-channel calibrations read as SMAPpy's dual calibration (`calibrate.smap_dual`).

A small SMAP file is written here with scipy (two splines, the calibrator's parameters) and its
``_trafo.mat`` beside it, the registration a projective map chosen so that a known molecule's two
images are known; DECODE-Plex's Fig. 2a calibration and movie, in the cache, check the reading on a
real splitter (docs/decode/plex-plan.md §6).
"""
import os
import pathlib

import numpy as np
import pytest
import scipy.io

from smappy.calibrate import dual
from smappy.calibrate.smap_dual import (
    chip_transformation,
    load_smap_dual_calibration,
    spline_knots,
)
from smappy.io.calibration import load_spline_calibration

PLEX = pathlib.Path(os.environ.get("DECODE_SMLM_DATA", pathlib.Path.home() / ".cache" / "decode_smlm")) / "plex"

# the bead images' camera ROI (x, y, width, height), and SMAP's split row
ROI = (40, 8, 30, 60)
SPLIT = 31
N, NZ = 9, 7


def _spline(rng, scale):
    # MATLAB's (n1, n2, nz, 64): n1 the rows; asymmetric in x so that a mirror shows
    c = rng.normal(0, 0.01, (N, N, NZ, 64))
    c[..., 0] += scale * np.linspace(0.5, 1.5, N)[None, :, None]
    return c


def _smap_matrix(h_camera_local, em: bool, width: int):
    """SMAP's row-vector trafo_raw.tar for a map given on the bead image's camera pixels from 0."""
    h = np.asarray(h_camera_local, float)
    if em:
        mirror = np.array([[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]], float)
        h = mirror @ h @ mirror
    to_smap = np.array([[1e-3, 0, 1e-3], [0, 1e-3, 1e-3], [0, 0, 1]])
    m = to_smap @ h @ np.linalg.inv(to_smap)
    return (m / m[2, 2]).T


# the lower (secondary) half onto the upper: a mirror about the split, a shift, a small rotation,
# and a projective part, in the bead image's camera pixels from 0
H_LOCAL = np.array([[0.998, 0.012, 1.3], [0.011, -1.002, 2 * SPLIT - 0.7], [2e-4, -1e-4, 1.0]])


def _write(tmp_path, em: bool):
    rng = np.random.default_rng(3)
    w = ROI[2]
    sxy = np.zeros((1, 2), dtype=[("cspline", "O"), ("EMon", "O"), ("PSF", "O"), ("Xrange", "O"),
                                 ("Yrange", "O"), ("posind", "O")])
    coeffs = [_spline(rng, 1.0), _spline(rng, 0.8)]
    for k, c in enumerate(coeffs):
        sxy[0, k] = ({"coeff": c, "dz": 20.0, "z0": 3.0, "x0": 5.0, "mirror": 2.0 * k, "normf": 1.0 - 0.2 * k},
                     float(em), np.ones((N + 4, N + 4, NZ + 2)), np.array([-np.inf, np.inf]),
                     np.array([-np.inf, SPLIT - 5.0]) if k == 0 else np.array([SPLIT + 5.0, np.inf]),
                     np.array([1.0, k + 1.0]))
    par = {"roi": np.array([np.array(ROI, float)], dtype=object), "Tmode": "up-down mirror", "Tform": "projective",
           "Tsplitpos": float(SPLIT), "emmirror": 1.0 if em else 2.0, "pixelsize": np.array([0.1, 0.1]),
           "dz": 20.0, "modality": "global 2 channel"}
    cal = tmp_path / "beads_3dcal.mat"
    scipy.io.savemat(cal, {"SXY": sxy, "parameters_g": par, "parameters1": par, "parameters2": par})
    raw = _smap_matrix(H_LOCAL, em, w)
    inv = _smap_matrix(np.linalg.inv(H_LOCAL), em, w)
    scipy.io.savemat(tmp_path / "beads_trafo.mat", {"trafo_raw": {"ref": np.eye(3), "tar": raw},
                                                    "trafo_inv_raw": {"ref": np.eye(3), "tar": inv}})
    return cal, coeffs


def _apply(h, xy):
    p = h @ np.c_[xy, np.ones(len(xy))].T
    return (p[:2] / p[2]).T


@pytest.mark.parametrize("em", [False, True])
def test_the_registration_takes_a_molecules_secondary_image_to_its_main_one_on_the_chip(tmp_path, em):
    """SMAP's row-vector map on pixels from 1 in thousands, in the image it un-mirrored for EM, comes
    out as SMAPpy's 3x3 on the chip's pixels from 0, in the camera's orientation."""
    cal_path, _ = _write(tmp_path, em)
    cal = load_smap_dual_calibration(cal_path)
    x0, y0 = ROI[:2]
    lower = np.array([[3.0, 40.0], [20.5, 52.25], [27.0, 33.0]])          # bead image, camera pixels
    upper = _apply(H_LOCAL, lower)
    np.testing.assert_allclose(_apply(cal.transformation, lower + [x0, y0]), upper + [x0, y0], atol=1e-9)
    np.testing.assert_allclose(cal.to_secondary(upper + [x0, y0]), lower + [x0, y0], atol=1e-9)


@pytest.mark.parametrize("em", [False, True])
def test_the_decode_plex_reading_is_its_affine_map_without_the_division(tmp_path, em):
    """``reading="decode-plex"``: trafo_inv_raw (main to secondary) times the row vector, the first two
    elements kept -- DECODE-Plex's XYZTransformationMatrix -- on the un-mirrored bead image."""
    cal_path, _ = _write(tmp_path, em)
    cal = load_smap_dual_calibration(cal_path, reading="decode-plex")
    assert cal.parameters["reading"] == "decode-plex"
    t = scipy.io.loadmat(tmp_path / "beads_trafo.mat", struct_as_record=False, squeeze_me=True)["trafo_inv_raw"].tar
    x0, y0, w, _ = ROI
    upper = np.array([[3.0, 10.0], [20.5, 22.25], [27.0, 30.0]])            # bead image, camera pixels
    xs = (w - 1 - upper[:, 0]) if em else upper[:, 0]
    r = np.c_[(xs + 1) / 1000, (upper[:, 1] + 1) / 1000, np.ones(len(upper))] @ t
    expected = np.c_[r[:, 0] * 1000 - 1, r[:, 1] * 1000 - 1]
    if em:
        expected[:, 0] = w - 1 - expected[:, 0]
    np.testing.assert_allclose(cal.to_secondary(upper + [x0, y0]), expected + [x0, y0], atol=1e-9)
    # an affine map, which SMAP's projective one is not
    np.testing.assert_allclose(cal.transformation[2], [0, 0, 1], atol=1e-12)


def test_chip_transformation_reads_smaps_matrix_as_matlab_applies_it():
    """A row vector times SMAP's matrix, then the division: x' = (x T)[:2] / (x T)[2]."""
    t = np.array([[1.001, -0.036, 0.002], [-0.036, -1.0009, 0.0012], [0.0174, 0.5653, 1.0]])
    h = chip_transformation(t, roi=(0, 0, 512, 512), em_mirrored=False)
    for x, y in [(10.0, 300.0), (400.0, 480.0)]:
        r = np.array([(x + 1) / 1000, (y + 1) / 1000, 1.0]) @ t
        expected = r[:2] / r[2] * 1000 - 1
        np.testing.assert_allclose(_apply(h, [[x, y]])[0], expected, atol=1e-9)


@pytest.mark.parametrize("em", [False, True])
def test_the_splines_are_each_channel_as_the_camera_saw_it(tmp_path, em):
    """Each SXY entry read as SMAPpy reads a single SMAP spline, un-mirrored where SMAP mirrored the
    beads; the secondary, which SMAP keeps un-mirrored along the split (its ``mirror`` 2), flipped
    back along y to the camera's orientation."""
    from smappy.calibrate.dual import mirrored_spline

    cal_path, coeffs = _write(tmp_path, em)
    cal = load_smap_dual_calibration(cal_path)
    for k, model in enumerate((cal.main, cal.secondary)):
        single = tmp_path / f"single{k}_3dcal.mat"
        scipy.io.savemat(single, {"SXY": {"cspline": {"coeff": coeffs[k], "dz": 20.0, "z0": 3.0, "x0": 5.0},
                                          "EMon": float(em)},
                                  "parameters": {"emmirror": 1.0 if em else 2.0}})
        expected = load_spline_calibration(single)
        if k == 1:
            expected = mirrored_spline(expected, axis=1)
        np.testing.assert_allclose(model.coeff, expected.coeff, rtol=1e-6, atol=1e-7)
        assert model.dz == 20.0 and model.z0 == 3.0
    assert cal.parameters["smap"]["EMon"] == [em, em]
    assert cal.parameters["smap"]["normf"] == [1.0, 0.8]


def test_the_split_is_smaps_with_the_main_channel_above_and_the_other_mirrored(tmp_path):
    cal_path, _ = _write(tmp_path, False)
    geometry = load_smap_dual_calibration(cal_path).geometry
    x0, y0, w, h = ROI
    assert geometry["layout"] == "up-down mirrored" and geometry["main_channel"] == "upper"
    assert geometry["coordinate_system"] == "camera-chip"
    layout = dual.geometry_layout(geometry)
    assert [tuple(r) for r in layout.regions] == [(x0, y0, w, SPLIT), (x0, y0 + SPLIT, w, h - SPLIT)]
    assert dual.mirror_axis(geometry) == 1                                # mirrored in y


def test_the_knots_are_each_cells_cubic_at_its_corners():
    rng = np.random.default_rng(0)
    coeff = rng.normal(size=(64, 3, 4, 5))
    knots = spline_knots(coeff)
    assert knots.shape == (4, 5, 6)
    np.testing.assert_allclose(knots[:3, :4, :5], coeff[0], rtol=1e-6)      # a cell's start: its constant
    c = coeff.reshape(4, 4, 4, 3, 4, 5)
    np.testing.assert_allclose(knots[3, 4, 5], c[..., 2, 3, 4].sum(), rtol=1e-5)   # the far end: t = 1


def test_a_converted_calibration_is_saved_read_and_drawn_by_the_simulation(tmp_path):
    from smappy.simulate import camera

    cal_path, _ = _write(tmp_path, True)
    cal = load_smap_dual_calibration(cal_path)
    out = tmp_path / "beads_2ct.h5"
    dual.save_dual_color_calibration(out, cal)
    back = dual.load_dual_color_calibration(out)
    np.testing.assert_array_equal(back.transformation, cal.transformation)
    np.testing.assert_array_equal(back.secondary.coeff, cal.secondary.coeff)
    assert back.geometry == cal.geometry
    layout = back.layout()                         # the channel layer reads it
    assert len(layout.regions) == 2
    assert camera.dual_calibration(out) is not None


@pytest.mark.skipif(not (PLEX / "calibration").exists() or not (PLEX / "Fig2a-SmallTest" / "first300.tif").exists(),
                    reason=f"DECODE-Plex's Fig. 2a calibration and movie are not in {PLEX}")
def test_fig2a_s_registration_puts_its_movies_lower_half_onto_the_upper_as_decode_plex_does():
    """DECODE-Plex's Fig. 2a movie was taken after its beads, the splitter moved by 6 rows: its run
    shifts the movie's second channel by the Test offset (-29 rows) where its registration expects the
    Simulation's (-35).  Warped by the converted registration, the movie's lower half lies on the upper
    one but for that move (measured 5.3 rows, 0.1 columns)."""
    import tifffile
    from scipy.ndimage import map_coordinates

    cal = load_smap_dual_calibration(next((PLEX / "calibration").glob("Fig2a-*_3dcal.mat")))
    x0, y0, w, _ = cal.parameters["smap"]["roi"]
    frames = tifffile.imread(PLEX / "Fig2a-SmallTest" / "first300.tif").astype(float)
    signal = (frames - np.median(frames, 0)).clip(0).mean(0)
    split = 255
    yy, xx = np.mgrid[0:split, 0:w].astype(float)
    lower = cal.to_secondary(np.c_[xx.ravel() + x0, yy.ravel() + y0]) - [x0, y0]
    warped = map_coordinates(signal, [lower[:, 1], lower[:, 0]], order=1, cval=np.nan).reshape(split, w)
    ok = np.isfinite(warped)
    upper = np.where(ok, signal[:split] - np.nanmean(signal[:split][ok]), 0)
    warped = np.where(ok, warped - np.nanmean(warped[ok]), 0)
    cc = np.real(np.fft.ifft2(np.fft.fft2(upper) * np.conj(np.fft.fft2(warped))))
    i, j = np.unravel_index(cc.argmax(), cc.shape)
    shift = ((i + split // 2) % split - split // 2, (j + w // 2) % w - w // 2)
    assert shift in ((5, 0), (6, 0)), shift
