"""The short way in: `smappy.fit` and its friends.

What matters here is that the facade only *assembles* -- the same frames must
give the same localizations whether they come from a file or from memory, and
whether they are collected or streamed to disk.
"""
import numpy as np
import pytest

import smappy
from smappy.io.hdf5 import load_localizations
from smappy.metadata import CameraMetadata

SHAPE = (48, 48)
CAMERA = {"conversion": 1.0, "offset": 100.0, "pixelsize_um": 0.1,
          "em_on": False, "emgain": 1.0}
FIT = dict(camera=CAMERA, sigma=1.2, cutoff=40.0, roisize=9)


def frames(n=8) -> np.ndarray:
    """One bright emitter per frame, so every frame yields a fit."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        y, x = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
        cx, cy = rng.uniform(8, SHAPE[1] - 8), rng.uniform(8, SHAPE[0] - 8)
        clean = 2000.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 2.0)
        out.append((rng.poisson(clean) + 100).astype(np.uint16))
    return np.stack(out)


def test_fitting_frames_from_memory_needs_no_file():
    locs = smappy.fit(frames(), **FIT)
    assert len(locs) == 8
    assert set(("x_nm", "y_nm", "photons")) <= set(locs.keys())
    assert locs.metadata["units"] == "nm"


def test_the_returned_table_is_the_one_that_was_written(tmp_path):
    out = tmp_path / "locs.h5"
    locs = smappy.fit(frames(), out=out, **FIT)
    written = load_localizations(out)
    assert len(written) == len(locs)
    assert np.array_equal(written["x_nm"], locs["x_nm"])
    # how it was run, and what the run did, both travel with the file
    assert written.metadata["fit"]["roisize"] == 9
    assert written.metadata["stats"]["frames"] == 8


def test_collect_false_streams_without_keeping_anything(tmp_path):
    out = tmp_path / "locs.h5"
    locs = smappy.fit(frames(), out=out, collect=False, **FIT)
    assert len(locs) == 0
    assert locs.metadata["stats"]["localizations"] == 8
    assert len(load_localizations(out)) == 8


def test_blocks_do_not_change_the_result():
    stack = frames(8)
    whole = smappy.fit(stack, **FIT)
    in_two = smappy.fit([(0, stack[:4]), (4, stack[4:])], **FIT)
    assert np.allclose(np.sort(whole["x_nm"]), np.sort(in_two["x_nm"]))
    assert np.array_equal(np.sort(whole["frame"]), np.sort(in_two["frame"]))


def test_camera_may_be_a_dataclass_a_dict_or_a_file(tmp_path):
    config = tmp_path / "camera.yaml"
    config.write_text("camera:\n  conversion: 1.0\n  offset: 100.0\n"
                      "  pixelsize_um: 0.1\n  em_on: false\n  emgain: 1.0\n")
    stack = frames(4)
    reference = smappy.fit(stack, **FIT)["x_nm"]
    for camera in (CameraMetadata(**CAMERA), CAMERA, config, str(config)):
        locs = smappy.fit(stack, camera=camera, sigma=1.2, cutoff=40.0, roisize=9)
        assert np.array_equal(locs["x_nm"], reference)


def test_images_without_a_camera_say_so():
    with pytest.raises(ValueError, match="camera has to be given"):
        smappy.fit(frames(2), sigma=1.2, cutoff=40.0)


def test_an_incomplete_camera_names_what_is_missing():
    with pytest.raises(ValueError, match="conversion"):
        smappy.fit(frames(2), camera={"pixelsize_um": 0.1}, sigma=1.2, cutoff=40.0)


def test_view_accepts_a_saved_file(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    out = tmp_path / "locs.h5"
    smappy.fit(frames(), out=out, **FIT)

    seen = {}
    monkeypatch.setattr("smappy.viewer.show",
                        lambda locs, *a, **k: seen.setdefault("n", len(locs)))
    smappy.view(out)
    assert seen["n"] == 8


def test_importing_smappy_does_not_import_the_world():
    import subprocess
    import sys
    code = ("import sys, smappy; "
            "print(int(any(m in sys.modules for m in ('matplotlib', 'h5py'))))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "0"


def test_the_viewer_opens_on_a_file_as_well_as_a_table(tmp_path, monkeypatch):
    """`show` loads a path itself, so a saved file is one call away."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from smappy.viewer import show

    out = tmp_path / "locs.h5"
    locs = smappy.fit(frames(), out=out, **FIT)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    viewer = show(out, block=False)
    try:
        assert len(viewer.state.locs) == len(locs)
    finally:
        plt.close("all")
