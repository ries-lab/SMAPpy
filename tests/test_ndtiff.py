"""Reading NDTiff datasets through their index.

The datasets here are written by hand rather than by pycro-manager, so the file
layout the reader assumes is stated in one place -- and the cases that matter
are the malformed ones: the zero padding at the end of the index, and the
records left behind by an acquisition that stopped mid-write.
"""
import json
import struct
import threading
import time

import numpy as np
import pytest

from smappy.io.ndtiff import (INDEX_NAME, SUMMARY_MARKER, SUMMARY_OFFSET,
                              is_ndtiff, open_ndtiff, read_index)
from smappy.io.tiff import camera_metadata, open_stack

SUMMARY = {"Frames": 4, "PixelSize_um": 0.1, "Height": 6, "Width": 5}
IMAGE_METADATA = {"Core-Camera": "Camera", "Camera-Offset": "100",
                  "ROI": "16-32-5-6", "Exposure-ms": 20.0, "ElapsedTime-ms": 0}


def write_ndtiff(folder, frames, *, pixel_type=1, pad=True, extra_records=0,
                 name="Stack_NDTiffStack.tif"):
    """A minimal NDTiff dataset: a stack file and the index that points into it.

    ``extra_records`` appends index entries for images that were never written,
    which is what an interrupted acquisition leaves behind.
    """
    folder.mkdir(parents=True, exist_ok=True)
    summary = json.dumps(SUMMARY).encode()
    blob = bytearray(b"\0" * SUMMARY_OFFSET)
    blob += struct.pack("<II", SUMMARY_MARKER, len(summary)) + summary

    index = bytearray()
    records = []
    for i, frame in enumerate(frames):
        metadata = json.dumps(dict(IMAGE_METADATA, ElapsedTime_ms=i)).encode()
        pixels = np.asarray(frame)
        records.append((len(blob), pixels.shape, len(blob) + pixels.nbytes,
                        len(metadata)))
        blob += pixels.tobytes() + metadata

    def record(i, offset, shape, md_offset, md_length, time_index):
        axes = json.dumps({"time": time_index}).encode()
        out = struct.pack("<I", len(axes)) + axes
        out += struct.pack("<I", len(name)) + name.encode()
        return out + struct.pack("<8I", offset, shape[1], shape[0], pixel_type,
                                 0, md_offset, md_length, 0)

    for i, (offset, shape, md_offset, md_length) in enumerate(records):
        index += record(i, offset, shape, md_offset, md_length, i)
    for k in range(extra_records):     # images the index promises but has not
        offset, shape, md_offset, md_length = records[-1]
        index += record(len(records) + k, len(blob) + k * shape[0] * shape[1] * 2,
                        shape, md_offset, md_length, len(records) + k)
    if pad:
        index += b"\0" * 64            # NDTiffStorage zero-pads the table

    (folder / name).write_bytes(bytes(blob))
    (folder / INDEX_NAME).write_bytes(bytes(index))
    return folder


def stack(n=4, shape=(6, 5)) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 4000, size=(n, *shape), dtype=np.uint16)


def test_frames_are_read_from_the_offsets_in_the_index(tmp_path):
    frames = stack()
    source = open_ndtiff(write_ndtiff(tmp_path / "ds", frames))

    assert source.n_frames == 4
    assert source.shape == (6, 5)
    assert source.dtype == np.uint16
    assert np.array_equal(source.frame(2), frames[2])
    read = np.concatenate([block for _, block in source.frames(chunk=3)])
    assert np.array_equal(read, frames)


def test_the_zero_padding_at_the_end_is_not_an_image(tmp_path):
    padded = read_index(write_ndtiff(tmp_path / "a", stack(), pad=True))
    bare = read_index(write_ndtiff(tmp_path / "b", stack(), pad=False))
    assert len(padded) == len(bare) == 4


def test_records_beyond_the_end_of_the_file_are_dropped(tmp_path):
    """An acquisition stopped mid-write leaves the index describing more."""
    folder = write_ndtiff(tmp_path / "ds", stack(), extra_records=3)
    source = open_ndtiff(folder)
    assert source.n_frames == 4
    assert len(list(source.frames(chunk=2))) == 2


def test_the_pixel_size_comes_from_where_the_metadata_starts(tmp_path):
    """pixelType has been seen to lie; the gap to the metadata has not."""
    frames = stack()
    folder = write_ndtiff(tmp_path / "ds", frames, pixel_type=0)  # claims 8-bit
    source = open_ndtiff(folder)
    assert source.dtype == np.uint16
    assert np.array_equal(source.frame(0), frames[0])


def test_metadata_reaches_the_camera_the_way_micro_manager_does(tmp_path):
    source = open_ndtiff(write_ndtiff(tmp_path / "ds", stack()))
    assert source.mm_metadata["Core-Camera"] == "Camera"
    assert source.mm_metadata["Format"] == "NDTiff"
    assert source.summary["Frames"] == 4

    camera = camera_metadata(source, overrides={"conversion": 1.0,
                                                "pixelsize_um": 0.1})
    assert camera.camera_name == "Camera"
    assert camera.offset == 100.0            # read from the image metadata
    assert camera.roi == (16, 32, 5, 6)      # so coordinates are absolute
    assert camera.exposure_ms == 20.0


def test_open_stack_picks_the_format_from_what_is_there(tmp_path):
    folder = write_ndtiff(tmp_path / "ds", stack())
    assert is_ndtiff(folder) and is_ndtiff(folder / INDEX_NAME)
    assert type(open_stack(folder)).__name__ == "NDTiffSource"
    assert not is_ndtiff(tmp_path)


def test_watching_a_dataset_that_is_still_being_written(tmp_path):
    """The index gains a record per finished image; watch follows it."""
    from smappy.io.watch import WatchSettings

    frames = stack(n=6)
    folder = write_ndtiff(tmp_path / "ds", frames[:2])
    source = open_ndtiff(folder)

    def keep_writing():
        for n in range(3, 7):
            time.sleep(0.05)
            write_ndtiff(folder, frames[:n])

    writer = threading.Thread(target=keep_writing)
    writer.start()
    settings = WatchSettings(poll=0.02, timeout=0.5)
    seen = [(start, len(block))
            for start, block in source.watch(chunk=2, settings=settings)]
    writer.join()

    assert sum(n for _, n in seen) == 6
    assert [start for start, _ in seen] == sorted(start for start, _ in seen)
    assert np.array_equal(source.frame(5), frames[5])


def test_a_stopped_watch_ends_at_once(tmp_path):
    from smappy.io.watch import WatchSettings

    folder = write_ndtiff(tmp_path / "ds", stack())
    source = open_ndtiff(folder)
    stop = threading.Event()
    stop.set()
    settings = WatchSettings(poll=0.01, timeout=5.0)
    assert list(source.watch(settings=settings, stop_event=stop)) == []


def test_a_compressed_dataset_says_so_rather_than_reading_noise(tmp_path):
    folder = write_ndtiff(tmp_path / "ds", stack())
    raw = bytearray((folder / INDEX_NAME).read_bytes())
    axes = len(json.dumps({"time": 0}).encode())
    name = len("Stack_NDTiffStack.tif")
    at = 4 + axes + 4 + name + 4 * 4          # the pixelCompression field
    raw[at:at + 4] = struct.pack("<I", 1)
    (folder / INDEX_NAME).write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="compressed"):
        open_ndtiff(folder)


def test_a_dataset_that_does_not_exist_yet_is_waited_for(tmp_path):
    """What a live fit does: the window opens before the microscope writes."""
    from smappy.io.watch import WatchSettings, open_growing_stack

    folder = tmp_path / "ds"
    frames = stack()

    def write_later():
        time.sleep(0.15)
        write_ndtiff(folder, frames)

    writer = threading.Thread(target=write_later)
    writer.start()
    source = open_growing_stack(folder, WatchSettings(poll=0.02, timeout=0.5,
                                                      appear_timeout=5.0))
    writer.join()
    assert type(source).__name__ == "NDTiffSource"
    assert source.shape == (6, 5)
    assert np.array_equal(source.frame(0), frames[0])


def test_a_micro_manager_tiff_is_not_mistaken_for_ndtiff(tmp_path):
    """The NDTiff check must not delay or divert an ordinary TIFF acquisition."""
    import tifffile

    from smappy.io.watch import WatchSettings, open_growing_stack

    directory = tmp_path / "mm"
    directory.mkdir()
    tifffile.imwrite(directory / "a_MMStack_Default.ome.tif", stack())
    source = open_growing_stack(directory, WatchSettings(poll=0.01, timeout=0.2,
                                                          appear_timeout=1.0))
    assert type(source).__name__ == "ImageSource"
