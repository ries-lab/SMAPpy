"""Reading a TIFF stack that an acquisition left unfinished."""
import numpy as np
import tifffile

from smappy.io.tiff import open_stack


def test_a_last_page_that_is_not_a_frame_is_neither_counted_nor_read(tmp_path):
    # An acquisition stopped while writing leaves a last page that is not a whole frame; a block that
    # reached it failed to stack.  The frames before it are all there.
    path = tmp_path / "movie_MMStack_Pos0.ome.tif"
    frames = np.arange(5 * 8 * 8, dtype=np.uint16).reshape(5, 8, 8)
    with tifffile.TiffWriter(path) as w:
        for f in frames:
            w.write(f, contiguous=False)
        w.write(np.zeros((1, 1), np.uint16), contiguous=False)
    src = open_stack(path)
    assert src.n_frames == 5
    blocks = list(src.frames(chunk=2, start=3))
    assert [b.shape for _, b in blocks] == [(2, 8, 8)]
    assert np.array_equal(np.concatenate([b for _, b in src.frames(chunk=3)]), frames)


def _series(tmp_path, n=(5, 4)):
    """A Micro-Manager-named series of files with ``n`` frames each, every pixel its frame's number."""
    frames = np.arange(sum(n), dtype=np.uint16)[:, None, None] * np.ones((1, 6, 7), np.uint16)
    paths, k = [], 0
    for i, count in enumerate(n):
        path = tmp_path / ("movie_MMStack_Pos0" + (f"_{i}" if i else "") + ".ome.tif")
        with tifffile.TiffWriter(path) as w:
            for f in frames[k:k + count]:
                w.write(f, contiguous=False)
        paths.append(path)
        k += count
    return paths, frames


def _index_map(monkeypatch, shuffle=True, broken=False):
    """tifffile's ``micromanager_metadata`` with an ``IndexMap`` as Micro-Manager writes one: an entry per
    frame, the page offset last, allocated longer than used (zeros).  tifffile cannot write the header."""
    def metadata(tf):
        offsets = np.array([p.offset for p in tf.pages], np.uint32) + (8 if broken else 0)
        rows = np.zeros((len(offsets) + 3, 5), np.uint32)
        rows[:len(offsets), 2], rows[:len(offsets), 4] = np.arange(len(offsets)), offsets
        if shuffle:
            rows[:len(offsets)] = rows[:len(offsets)][::-1]
        return {"Summary": {}, "IndexMap": rows}
    monkeypatch.setattr(tifffile.TiffFile, "micromanager_metadata", property(metadata))
    monkeypatch.setattr(tifffile.TiffFile, "is_micromanager", property(lambda tf: True), raising=False)


def test_frames_are_counted_and_read_through_micro_managers_index_map(tmp_path, monkeypatch):
    # Walking the pages to count them cost 28 s for 34 GB on iCloud Drive; the map in each file's
    # header gives the count and where each frame is
    paths, frames = _series(tmp_path)
    _index_map(monkeypatch)
    src = open_stack(paths[0])
    assert src.n_frames == 9 and src.offsets is not None and [len(o) for o in src.offsets] == [5, 4]
    assert np.array_equal(np.concatenate([b for _, b in src.frames(chunk=2)]), frames)
    across = list(src.frames(chunk=10, start=3, stop=7))
    assert across[0][0] == 3 and np.array_equal(across[0][1], frames[3:7])
    assert np.array_equal(src.frame(8), frames[8])


def test_a_map_that_does_not_point_at_the_pages_is_not_trusted(tmp_path, monkeypatch):
    paths, frames = _series(tmp_path)
    _index_map(monkeypatch, broken=True)
    src = open_stack(paths[0])
    assert src.offsets is None and src.n_frames == 9
    assert np.array_equal(src.frame(6), frames[6])
