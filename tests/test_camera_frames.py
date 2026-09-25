"""The simulated camera frames: a fit of them gets the truth back."""
import numpy as np
import pytest

from smappy.simulate import camera_frames


def test_the_frames_are_the_camera_it_was_told_of():
    frames, truth = camera_frames(50, seed=1, background=20.0, conversion=0.5,
                                  offset=100.0)
    assert frames.dtype == np.uint16 and frames.shape == (50, 100, 100)
    # a pixel no molecule reaches is the background, converted: 20 / 0.5 + 100
    assert np.median(frames) == pytest.approx(140, abs=3)
    assert set(np.unique(truth["frame"])) <= set(range(50))


def test_a_fit_of_the_frames_finds_the_molecules_where_they_are(tmp_path):
    """Half a pixel off in x and y was the first version: pixel k is centred on
    k, the fitter's convention, not on k + 1/2."""
    tifffile = pytest.importorskip("tifffile")
    from scipy.spatial import cKDTree
    from smappy.plugins import Context
    from smappy.plugins.fit import (CameraSettings, GaussianFit, GaussianFitSettings,
                                    OutputSettings, SourceSettings)
    frames, truth = camera_frames(300, seed=2, conversion=0.5, offset=100.0)
    path = tmp_path / "frames.tif"
    tifffile.imwrite(path, frames)
    settings = GaussianFitSettings(
        source=SourceSettings(path=str(path)),
        camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1),
        output=OutputSettings(path=str(tmp_path / "out.hdf5")))
    locs = GaussianFit().run(Context(), settings).locs
    if locs is None:                     # saved rather than handed back
        from smappy.session import Session
        session = Session()
        session.load(tmp_path / "out.hdf5")
        locs = session.locs
    assert len(locs) > 0.8 * len(truth)
    dx, dy, sigma = [], [], []
    for f in np.unique(locs["frame"]):
        found = np.column_stack([locs["x_nm"], locs["y_nm"]])[locs["frame"] == f]
        true = np.column_stack([truth["x_nm"], truth["y_nm"]])[truth["frame"] == f]
        if len(found) and len(true):
            d, i = cKDTree(true).query(found)
            near = d < 50
            dx += list(found[near, 0] - true[i[near], 0])
            dy += list(found[near, 1] - true[i[near], 1])
            sigma += list(locs["xy_err_nm"][locs["frame"] == f][near])
    dx, dy, sigma = np.array(dx), np.array(dy), np.array(sigma)
    assert abs(np.median(dx)) < 1.0 and abs(np.median(dy)) < 1.0    # no offset
    # each error against that spot's own precision: robustly about one sigma
    spread = 1.4826 * np.median(np.abs(dx / sigma))
    assert spread == pytest.approx(1.0, abs=0.3)


def test_a_calibration_from_the_beads_fits_the_astigmatic_frames_in_z(tmp_path):
    """The whole 3D path on simulated data: beads -> calibration -> Spline 3D.

    A bead stack records where the *objective* was, so a bead drawn at +z
    rather than -z made a calibration that turned every z upside down (fitted
    against true, the slope was -0.96).  Both simulations use one PSF, so the
    fit must give z back with the right sign, no offset and no change of
    scale: before the frames dropped overlapping blinks the slope was 0.93,
    from the quarter of the spots with a neighbour in their ROI.
    """
    tifffile = pytest.importorskip("tifffile")
    from scipy.spatial import cKDTree
    from smappy.calibrate.core import CalibrationSettings, build_calibration, collect_beads
    from smappy.calibrate.input import BeadStack
    from smappy.plugins import Context
    from smappy.plugins.fit import (CameraSettings, OutputSettings, SourceSettings,
                                    SplineFit, SplineFitSettings, SplineModelSettings)
    from smappy.session import Session
    from smappy.simulate import ASTIGMATISM, bead_stacks
    stacks, z = bead_stacks(2, seed=0)
    beads = collect_beads([BeadStack(s.astype(np.float32), z, source=f"b{i}")
                           for i, s in enumerate(stacks)], CalibrationSettings())
    calibration = tmp_path / "beads_3dcal.h5"
    build_calibration(beads).save(calibration)
    frames, truth = camera_frames(400, seed=3, astigmatism=ASTIGMATISM)
    tifffile.imwrite(tmp_path / "frames.tif", frames)
    SplineFit().run(Context(), SplineFitSettings(
        source=SourceSettings(path=str(tmp_path / "frames.tif")),
        camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1),
        model=SplineModelSettings(calibration=str(calibration)),
        output=OutputSettings(path=str(tmp_path / "out.hdf5"))))
    session = Session()
    session.load(tmp_path / "out.hdf5")
    locs = session.locs
    fitted, true = [], []
    for f in np.unique(locs["frame"]):
        m, t = locs["frame"] == f, truth["frame"] == f
        if not t.any():
            continue
        d, i = cKDTree(np.column_stack([truth["x_nm"][t], truth["y_nm"][t]])).query(
            np.column_stack([locs["x_nm"][m], locs["y_nm"][m]]))
        near = d < 60
        fitted += list(locs["z_nm"][m][near])
        true += list(truth["z_nm"][t][i[near]])
    slope, intercept = np.polyfit(true, fitted, 1)
    assert slope == pytest.approx(1.0, abs=0.02)
    assert abs(intercept) < 10


@pytest.fixture(scope="module")
def bead_calibration():
    from smappy.calibrate.core import CalibrationSettings, build_calibration, collect_beads
    from smappy.calibrate.input import BeadStack
    from smappy.simulate import bead_stacks
    stacks, z = bead_stacks(2, seed=0)
    beads = collect_beads([BeadStack(s.astype(np.float32), z, source=f"b{i}")
                           for i, s in enumerate(stacks)], CalibrationSettings())
    return build_calibration(beads).calibration


def _spots(z_nm, neighbour_px=None, photons=5000.0, background=20.0, size=13):
    """Noise-free ROIs of one astigmatic spot at the centre and ``z_nm``, with
    an equally bright one ``neighbour_px`` to the right at the same z."""
    from scipy.special import erf
    from smappy.simulate import ASTIGMATISM, astigmatic_sigmas
    sx, sy = (w / 100.0 * np.sqrt(2.0) for w in astigmatic_sigmas(z_nm, 130.0, *ASTIGMATISM))
    pix, c = np.arange(size), (size - 1) / 2
    def spot(cx, k):
        px = 0.5 * (erf((pix + 0.5 - cx) / sx[k]) - erf((pix - 0.5 - cx) / sx[k]))
        py = 0.5 * (erf((pix + 0.5 - c) / sy[k]) - erf((pix - 0.5 - c) / sy[k]))
        return photons * np.outer(py, px)
    return np.array([background + spot(c, k) + (spot(c + neighbour_px, k) if neighbour_px else 0)
                     for k in range(len(z_nm))], np.float32)


def _fitted_z(calibration, rois):
    from smappy.psf import SplinePSF
    return calibration.z_index_to_nm(SplinePSF(calibration).fit(rois, iterations=100).theta[:, 4])


def test_the_bead_calibration_gives_isolated_spots_back_at_the_scale_they_were_drawn(
        bead_calibration):
    """The calibration and the fitter alone, without the acquisition: spots of
    the beads' own PSF at known z come back on a slope of one.  This is what
    said the 4-7 % compression of the fitted frames was not the calibration's
    (nor its z smoothing's: 0.99 unsmoothed, 1.00 at the default 20 nm)."""
    z = np.linspace(-500, 500, 41)
    slope, intercept = np.polyfit(z, _fitted_z(bead_calibration, _spots(z)), 1)
    assert slope == pytest.approx(1.0, abs=0.01)
    assert abs(intercept) < 8


def test_a_neighbour_inside_the_roi_pulls_z_towards_focus(bead_calibration):
    """Why `camera_frames` drops blinks with a neighbour: a second spot four
    pixels away is light the one-emitter model explains as a rounder spot,
    and the whole z range shrinks towards focus."""
    z = np.linspace(-300, 300, 25)
    slope, _ = np.polyfit(z, _fitted_z(bead_calibration, _spots(z, neighbour_px=4)), 1)
    assert slope < 0.9


def test_a_spline_fit_at_a_few_background_photons_does_not_stall_at_no_background(
        bead_calibration):
    """sCMOS frames with 1-5 background photons per pixel: the fit used to
    start the background at the ROI's minimum pixel, 0 here, and a third to
    two thirds of the fits ended there, their z frozen at the start and the
    slope of fitted z against true 0.2-0.6.  A start from the ROI's border
    and the expected information in the Hessian bring them all back."""
    rng = np.random.default_rng(0)
    z = rng.uniform(-400, 400, 600)
    for photons, background in ((1000, 1.0), (1000, 5.0), (5000, 2.0)):
        rois = rng.poisson(_spots(z, photons=photons, background=background))
        rois = (rois + rng.normal(0, 1.5, rois.shape)).astype(np.float32)
        from smappy.psf import SplinePSF
        fit = SplinePSF(bead_calibration).fit(rois, iterations=50)
        assert np.isfinite(fit.theta).all()
        assert np.mean(fit.theta[:, 3] < 0.05) < 0.02, (photons, background)
        slope = np.polyfit(z, bead_calibration.z_index_to_nm(fit.theta[:, 4]), 1)[0]
        assert slope == pytest.approx(1.0, abs=0.05), (photons, background)


def test_no_two_spots_in_a_frame_are_closer_than_the_minimum_separation():
    """The frames are meant to fit cleanly, and the structure's emitters
    cluster: without this a quarter of the blinks had a neighbour within a
    micrometre, which compressed fitted z by 7 %."""
    from scipy.spatial import cKDTree
    from smappy.simulate import MIN_SEPARATION_NM, dual_camera_frames
    for simulate in (camera_frames, dual_camera_frames):
        _, truth = simulate(200, seed=4)
        assert truth.metadata["n_unresolvable_dropped"] > 0
        for f in np.unique(truth["frame"]):
            t = truth["frame"] == f
            xy = np.column_stack([truth["x_nm"][t], truth["y_nm"][t]])
            assert not cKDTree(xy).query_pairs(MIN_SEPARATION_NM), (simulate.__name__, f)
    _, crowded = camera_frames(200, seed=4, min_separation_nm=0)
    assert crowded.metadata["n_unresolvable_dropped"] == 0


def test_the_split_camera_frames_give_back_their_transformation_and_their_dyes(tmp_path):
    """`dual_camera_frames` through the two-colour fit, as its tutorial runs it:
    the transformation measured from the movie is the simulated one, and the
    colours assigned from the photon split are the dyes."""
    tifffile = pytest.importorskip("tifffile")
    from scipy.spatial import cKDTree
    from smappy.calibrate.transform import load_transform
    from smappy.plugins import Context
    from smappy.plugins.fit import (CameraSettings, ChannelTransformSettings,
                                    DualGaussianFit, DualGaussianFitSettings,
                                    OutputSettings, SourceSettings)
    from smappy.simulate import dual_camera_frames, dual_transformation
    frames, truth = dual_camera_frames(600, seed=2)
    assert frames.shape[1:] == (200, 100)
    tifffile.imwrite(tmp_path / "two.tif", frames)
    result = DualGaussianFit().run(Context(), DualGaussianFitSettings(
        source=SourceSettings(path=str(tmp_path / "two.tif")),
        camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1),
        transform=ChannelTransformSettings(calibrate=True, calibrate_skip=0),
        output=OutputSettings(path=str(tmp_path / "two.hdf5"))))
    probe = np.array([[5.0, 105.0], [95.0, 195.0], [50.0, 150.0]])
    true = (np.c_[probe, np.ones(3)] @ dual_transformation().T)[:, :2]
    measured = load_transform(tmp_path / "two_2ct.h5").transform(probe)
    np.testing.assert_allclose(measured, true, atol=0.2)
    locs = result.locs
    got, want = [], []
    for f in np.unique(locs["frame"]):
        m, t = locs["frame"] == f, truth["frame"] == f
        if not t.any():
            continue
        d, i = cKDTree(np.column_stack([truth["x_nm"][t], truth["y_nm"][t]])).query(
            np.column_stack([locs["x_nm"][m], locs["y_nm"][m]]))
        near = (d < 60) & (locs["channel"][m] > 0)
        got += list(locs["channel"][m][near])
        want += list(truth["dye"][t][i[near]])
    # colour 1 is the lower (ch0 - ch1) / (ch0 + ch1): the dye with more in
    # the secondary half, which is the lines (dye 2)
    assert np.mean(np.asarray(got) == 3 - np.asarray(want)) > 0.95
