"""The image tags: what the microscope recorded per frame, kept by the fit."""
import json

import h5py
import numpy as np
import pytest
import tifffile

from smappy.frametags import MAX_STATES, FrameTags, acquisition
from smappy.io.hdf5 import load_localizations, save_localizations
from smappy.io.tiff import open_stack
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.image_tags import ImageTags, ImageTagsSettings, names, per_frame

N_FRAMES = 60


def _plane(frame):
    """A frame's metadata as Micro-Manager 2 writes it, with what changes."""
    return {
        "Camera": "Andor", "Exposure-ms": 50.0, "ImageNumber": frame,
        "UUID": f"uuid-{frame}", "ElapsedTime-ms": 4147.0 + 52.3 * frame,
        "COM5-BaudRate": "115200", "PIZStage-Description": "PI E-709",
        "PIZStage-Position": f"{50.0 - 0.01 * frame:.4f}",
        "Laser Trigger-Duration0 (us)": str(0 if frame < 20 else 2 * frame),
        "Luxx405-Laser Operation Select": "Off" if frame < 20 else "On",
        "UserData": {"Andor-ElapsedTime-ms(HW)": {"type": "STRING",
                                                  "scalar": f"{51.7 * frame:.2f}"}},
    }


def test_only_the_tags_that_change_are_kept_and_filled_in_from_the_first_frame():
    tags = FrameTags()
    for frame in range(N_FRAMES):
        tags.add(frame, _plane(frame))
    table = tags.table()
    assert set(table) == {"frame", "ElapsedTime-ms", "PIZStage-Position",
                          "Laser Trigger-Duration0 (us)",
                          "Luxx405-Laser Operation Select",
                          "Andor-ElapsedTime-ms(HW)"}
    np.testing.assert_allclose(table["PIZStage-Position"],
                               50.0 - 0.01 * np.arange(N_FRAMES), atol=1e-9)
    trigger = table["Laser Trigger-Duration0 (us)"]
    assert trigger[:20].tolist() == [0] * 20 and trigger[30] == 60
    # a state stays text; the camera's clock comes out of UserData
    assert table["Luxx405-Laser Operation Select"][[0, 19, 20]].tolist() == \
        ["Off", "Off", "On"]
    assert table["Andor-ElapsedTime-ms(HW)"][2] == pytest.approx(103.4)


def test_a_text_tag_with_many_values_is_an_identifier_and_is_dropped():
    tags = FrameTags()
    for frame in range(MAX_STATES + 5):
        tags.add(frame, {"Stamp": f"13:16:{frame:02d}", "Filter": "A" if frame % 2 else "B"})
    assert "Stamp" not in tags.table() and tags.dropped == ["Stamp"]
    assert "Filter" in tags.table()


@pytest.fixture
def stack(tmp_path):
    """A Micro-Manager TIFF whose every page carries its own metadata."""
    path = tmp_path / "run_MMStack_Pos0.ome.tif"
    with tifffile.TiffWriter(path) as writer:
        for frame in range(N_FRAMES):
            md = json.dumps(_plane(frame))
            writer.write(np.full((8, 8), frame, np.uint16), metadata=None,
                         extratags=[(51123, "s", 0, md, True)])
    (tmp_path / "comments.txt").write_text(json.dumps(
        {"map": {"General annotation": {"type": "PROPERTY_MAP", "scalar": {
            "comments": {"type": "STRING", "scalar": "50ms\n638i30"}}}}}))
    return path


def test_the_tags_are_read_with_the_frames_and_survive_the_file(stack, tmp_path):
    source = open_stack(stack)
    source.tags = FrameTags()
    for _ in source.frames(chunk=7, start=5):
        pass
    table = source.tags.table()
    assert table["frame"].tolist() == list(range(5, N_FRAMES))

    static = acquisition(source)
    assert static["comment"] == "50ms\n638i30"
    assert "PIZStage-Description" in static["devices"]
    assert not any(k.startswith("COM5") for k in static["devices"])

    locs = Localizations({"frame": np.arange(N_FRAMES, dtype=np.int64)},
                         {"frame_tags": table, "acquisition": static})
    out = save_localizations(tmp_path / "t.hdf5", locs)
    with h5py.File(out, "r") as f:            # beside the table, not in the attribute
        assert "frame_tags" not in json.loads(f.attrs["metadata"])
    back = load_localizations(out).metadata
    np.testing.assert_array_equal(back["frame_tags"]["PIZStage-Position"],
                                  table["PIZStage-Position"])
    assert back["frame_tags"]["Luxx405-Laser Operation Select"].tolist() == \
        table["Luxx405-Laser Operation Select"].tolist()
    assert back["acquisition"] == json.loads(json.dumps(static))


def test_tags_are_found_by_part_of_their_name_and_the_rate_is_per_frame():
    table = {"frame": np.arange(3), "PIZStage-Position": np.zeros(3),
             "Laser Trigger-Duration0 (us)": np.zeros(3)}
    assert names(table, "piz") == ["PIZStage-Position"]
    assert names(table, "") == ["PIZStage-Position", "Laser Trigger-Duration0 (us)"]
    centres, rate = per_frame(np.repeat(np.arange(100), 3), 0, 99, bins=10)
    assert len(centres) == 10 and np.allclose(rate, 3.0)


def test_the_plugin_draws_a_tab_per_tag_and_says_the_piezo_range():
    tags = FrameTags()
    for frame in range(N_FRAMES):
        tags.add(frame, _plane(frame))
    locs = Localizations({"frame": np.arange(N_FRAMES, dtype=np.int64)},
                         {"frame_tags": tags.table()})
    result = ImageTags().run(Context(locs=locs), ImageTagsSettings(tags="PIZ, trigger"))
    assert list(result.plots) == ["PIZStage-Position", "Laser Trigger-Duration0 (us)"]
    assert "range 0.59" in result.text
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for draw in result.plots.values():
        draw(plt.figure().add_subplot())
    plt.close("all")


def test_a_table_without_tags_says_so():
    with pytest.raises(ValueError, match="no image tags"):
        ImageTags().run(Context(locs=Localizations({"frame": np.arange(3)})),
                        ImageTagsSettings())


def test_the_fit_keeps_the_tags_and_shows_the_piezo(tmp_path):
    """A simulated acquisition rewritten as a Micro-Manager TIFF with tags."""
    from smappy.plugins import get
    from smappy.plugins.fit import (CameraSettings, GaussianFit, GaussianFitSettings,
                                    OutputSettings, SourceSettings)
    from smappy.simulate import (CameraOutputSettings, LabellingSettings,
                                 SimulationSettings)
    made = get("File/Simulate/Blinking Structure")().run(Context(), SimulationSettings(
        output="camera", n_frames=N_FRAMES, seed=3,
        labelling=LabellingSettings(efficiency=0.1),
        camera=CameraOutputSettings(path=str(tmp_path / "run.sim.yaml"))))
    sim = open_stack(made.data["path"])
    frames = np.concatenate([b for _, b in sim.frames(chunk=30)])
    path = tmp_path / "acq" / "acq_MMStack_Pos0.ome.tif"
    path.parent.mkdir()
    with tifffile.TiffWriter(path) as writer:
        for frame, image in enumerate(frames):
            writer.write(image.astype(np.uint16), metadata=None,
                         extratags=[(51123, "s", 0, json.dumps(_plane(frame)), True)])
    cam = sim.camera
    out = tmp_path / "out.hdf5"
    result = GaussianFit().run(Context(), GaussianFitSettings(
        source=SourceSettings(path=str(path), chunk=25),
        camera=CameraSettings(conversion=cam["conversion"], offset=cam["offset"],
                              pixelsize_um=cam["pixelsize_um"], em_on=False),
        output=OutputSettings(path=str(out), raw_frames=0)))
    assert "PIZStage-Position" in result.plots            # shown by default
    assert "PIZStage-Position" in result.text
    for stored in (result.locs.metadata, load_localizations(out).metadata):
        np.testing.assert_allclose(stored["frame_tags"]["PIZStage-Position"],
                                   50.0 - 0.01 * np.arange(N_FRAMES), atol=1e-9)
        assert "devices" in stored["acquisition"]
