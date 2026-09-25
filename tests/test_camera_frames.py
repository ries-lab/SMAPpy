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
            sigma += list(locs["loc_precision_nm"][locs["frame"] == f][near])
    dx, dy, sigma = np.array(dx), np.array(dy), np.array(sigma)
    assert abs(np.median(dx)) < 1.0 and abs(np.median(dy)) < 1.0    # no offset
    # each error against that spot's own precision: robustly about one sigma
    spread = 1.4826 * np.median(np.abs(dx / sigma))
    assert spread == pytest.approx(1.0, abs=0.3)
