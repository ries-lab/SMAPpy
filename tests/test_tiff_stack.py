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
