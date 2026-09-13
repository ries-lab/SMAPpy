"""The fitters as plugins: parts, nesting, a headless run, streaming."""
import numpy as np
import pytest
import tifffile

from smappy import plugins
from smappy.plugins import Context
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
    ctx = Context(progress=messages.append, stream=lambda e, p: events.append((e, p)))
    result = GaussianFit().run(ctx, settings)
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


def test_the_default_output_goes_next_to_the_image_folder(tmp_path):
    """Named by the acquisition, beside the folder its images are in."""
    from smappy.plugins.fit import default_output_path
    singles = tmp_path / "Nup96_EM100_2_Pos0"
    singles.mkdir()
    for i in range(3):
        tifffile.imwrite(singles / f"img_{i:09d}_Default_000.tif",
                         np.zeros((4, 4), np.uint16))
    expected = tmp_path / "Nup96_EM100_2_Pos0_locs.hdf5"
    assert default_output_path(singles) == expected
    assert default_output_path(singles / "img_000000001_Default_000.tif") == expected

    series = tmp_path / "run_1"
    series.mkdir()
    tifffile.imwrite(series / "run_1_MMStack_Pos0.ome.tif", np.zeros((2, 4, 4), np.uint16))
    assert default_output_path(series / "run_1_MMStack_Pos0.ome.tif") == \
        tmp_path / "run_1_locs.hdf5"

    lone = tmp_path / "stack.tif"
    tifffile.imwrite(lone, np.zeros((2, 4, 4), np.uint16))
    assert default_output_path(lone) == tmp_path / "stack_locs.hdf5"


def test_choosing_a_source_fills_the_output_but_keeps_a_typed_one(stack, tmp_path):
    plugin = GaussianFit()
    settings = GaussianFitSettings(source=SourceSettings(path=str(stack)))
    updates = plugin.react("source.path", settings)
    assert updates["output.path"].endswith("_locs.hdf5")

    # the default follows a new source while it is still the default ...
    settings.output.path = updates["output.path"]
    other = tmp_path / "other.tif"
    tifffile.imwrite(other, np.zeros((2, 8, 8), np.uint16))
    settings.source.path = str(other)
    assert plugin.react("source.path", settings)["output.path"] == \
        str(tmp_path / "other_locs.hdf5")

    # ... and a path the user typed is left alone
    settings.output.path = str(tmp_path / "mine.hdf5")
    assert "output.path" not in (plugin.react("source.path", settings) or {})
