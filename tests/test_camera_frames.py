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
    fit must give z back with the right sign and no offset.
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
    assert slope == pytest.approx(1.0, abs=0.08)
    assert abs(intercept) < 15


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
