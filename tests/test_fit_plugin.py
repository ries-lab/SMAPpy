"""The fitters as plugins: parts, nesting, a headless run, streaming."""
import numpy as np
import pytest
import tifffile

from smappy import plugins
from smappy.plugins.fit import (CameraSettings, DetectionSettings, GaussianFit,
                                GaussianFitSettings, OutputSettings, SourceSettings)

SHAPE = (48, 48)


@pytest.fixture
def stack(tmp_path):
    """20 frames, a few bright emitters each, plain TIFF (no metadata)."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    frames = []
    for _ in range(20):
        clean = np.zeros(SHAPE)
        for _ in range(3):
            cx, cy = rng.uniform(8, SHAPE[1] - 8), rng.uniform(8, SHAPE[0] - 8)
            clean += 3000.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 2.0)
        frames.append((rng.poisson(clean) + 100).astype(np.uint16))
    path = tmp_path / "run.tif"
    tifffile.imwrite(path, np.stack(frames))
    return path


def test_parts_show_up_as_children():
    specs = GaussianFit.specs()
    assert set(specs) == {"source", "camera", "detection", "model", "fit", "output"}
    assert specs["fit"].children["roisize"].info.label == "ROI size"   # via dotted extra
    assert specs["source"].children["path"].info.kind == "open_file"
    assert "Localize/Spline 3D" in plugins.available("Localize/")


def test_fit_runs_without_a_gui_and_streams(stack, tmp_path):
    settings = GaussianFitSettings(
        source=SourceSettings(path=str(stack), chunk=8),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=40.0),
        output=OutputSettings(path=str(tmp_path / "out.hdf5")))
    events, messages = [], []
    result = GaussianFit().run(None, None, settings, progress=messages.append,
                               stream=lambda e, p: events.append((e, p)))
    assert messages and "localizations" in messages[-1]
    assert 50 <= len(result.locs) <= 60          # 3 per frame, some overlap
    assert (tmp_path / "out.hdf5").exists()
    assert events[0][0] == "start" and events[0][1]["extent"] is not None
    blocks = [p for e, p in events if e == "block"]
    assert sum(len(b) for b in blocks) == len(result.locs)
    assert result.locs["x_nm"].max() < 4800


def test_react_fills_the_camera_from_a_preset(tmp_path):
    preset = tmp_path / "cam.yaml"
    preset.write_text("camera:\n  conversion: 6.7\n  offset: 400\n  pixelsize_um: 0.127\n")
    settings = GaussianFitSettings(camera=CameraSettings(preset=str(preset)))
    updates = GaussianFit().react("camera.preset", settings)
    assert updates == {"camera.conversion": 6.7, "camera.offset": 400,
                       "camera.pixelsize_um": 0.127}
