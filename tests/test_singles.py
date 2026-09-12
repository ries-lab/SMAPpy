"""Acquisitions written as one TIFF file per frame."""
import json

import numpy as np
import pytest
import tifffile

from smappy.io.singles import (open_singles, read_metadata_txt,
                               single_image_files)
from smappy.io.tiff import camera_metadata, open_stack


def write_set(folder, n=12, shape=(6, 5), metadata=True):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        frame = np.full(shape, i, np.uint16)
        tifffile.imwrite(folder / f"img_{i:09d}_Default_000.tif", frame)
    if metadata:
        (folder / "metadata.txt").write_text(json.dumps({
            "Summary": {"Frames": 200, "Date": "2019-10-22", "Width": shape[1],
                        "Height": shape[0], "Prefix": "run"},
            "FrameKey-0-0-0": {"Core-Camera": "Evolve512", "Evolve512-Offset": "500",
                               "Exposure-ms": 10, "ROI": "127-0-287-512",
                               "PixelSizeUm": 0.1},
        }))
    return folder


def test_any_image_of_the_folder_opens_the_whole_acquisition(tmp_path):
    folder = write_set(tmp_path / "run", n=12)
    for path in (folder, folder / "img_000000007_Default_000.tif"):
        source = open_stack(path)
        assert source.n_frames == 12
        assert source.shape == (6, 5)
        # the order is the frame number, not the string: 9 before 10
        assert [f.name for f in source.files] == sorted(
            f.name for f in source.files)
        np.testing.assert_array_equal(source.frame(9), np.full((6, 5), 9))


def test_frames_are_in_acquisition_order_past_ten(tmp_path):
    folder = write_set(tmp_path / "run", n=12)
    source = open_singles(folder)
    values = [int(block[0, 0, 0]) for _, block in source.frames(chunk=1)]
    assert values == list(range(12))
    _, block = next(source.frames(chunk=3, start=5))
    np.testing.assert_array_equal(block[:, 0, 0], [5, 6, 7])


def test_the_camera_comes_from_metadata_txt(tmp_path):
    folder = write_set(tmp_path / "run")
    summary, plane = read_metadata_txt(folder)
    assert summary["Frames"] == 200 and plane["Core-Camera"] == "Evolve512"
    cam = camera_metadata(open_singles(folder), require=False)
    assert cam.offset == 500 and cam.camera_name == "Evolve512"
    assert cam.exposure_ms == 10 and tuple(cam.roi) == (127, 0, 287, 512)


def test_a_truncated_metadata_txt_still_gives_the_camera(tmp_path):
    folder = write_set(tmp_path / "run")
    text = (folder / "metadata.txt").read_text()
    (folder / "metadata.txt").write_text(text[:text.rindex("}")])
    summary, plane = read_metadata_txt(folder)
    assert summary["Frames"] == 200 and plane["Core-Camera"] == "Evolve512"


def test_an_ome_series_or_a_lone_image_is_left_to_the_other_readers(tmp_path):
    ome = tmp_path / "ome"
    ome.mkdir()
    for name in ("run_MMStack.ome.tif", "run_MMStack_1.ome.tif"):
        tifffile.imwrite(ome / name, np.zeros((2, 4, 4), np.uint16))
    assert single_image_files(ome) == []
    lone = tmp_path / "lone"
    lone.mkdir()
    tifffile.imwrite(lone / "img_000000000_Default_000.tif", np.zeros((4, 4), np.uint16))
    assert single_image_files(lone) == []


def test_a_folder_without_metadata_still_opens(tmp_path):
    folder = write_set(tmp_path / "bare", n=4, metadata=False)
    source = open_singles(folder)
    assert source.n_frames == 4 and source.n_frames_declared is None
    with pytest.raises(NotImplementedError):
        next(source.watch())
