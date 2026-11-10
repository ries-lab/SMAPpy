"""The prefit (`smappy.training.prefit`): blocks spread over the acquisition, fitted and grouped."""
import numpy as np

from smappy.training.prefit import blocks, prefit


def test_the_blocks_span_the_acquisition_from_its_first_frame_to_its_last():
    b = blocks(20000, 10, 50)
    assert len(b) == 10 and b[0] == (0, 50) and b[-1] == (19950, 20000)
    assert all(stop - start == 50 for start, stop in b)
    assert all(b[i][1] <= b[i + 1][0] for i in range(9))          # apart, in order
    assert blocks(300, 10, 50) == [(0, 300)]                       # short: all of it


def test_each_block_is_fitted_and_grouped_on_its_own_frames():
    from smappy.simulate.camera import camera_frames
    frames, _ = camera_frames(n_frames=400, seed=1, size_px=64, photons=4000.0)
    read = []

    def reader(start, stop):
        read.append((start, stop))
        return frames[start:stop]

    p = prefit(reader, len(frames), {"conversion": 0.5, "offset": 100.0, "pixelsize_um": 0.1}, None,
               n_blocks=4, block_frames=30)
    assert read == [(0, 30), (123, 153), (247, 277), (370, 400)]
    f = np.asarray(p.localizations["frame"])
    assert all(any(a <= v < b for a, b in read) for v in np.unique(f))     # the acquisition's frame numbers
    gid, start = np.asarray(p.localizations["group_id"]), np.searchsorted([a for a, _ in read], f, side="right")
    for g in np.unique(gid):                                              # no blink across two blocks
        assert len(np.unique(start[gid == g])) == 1
    assert p.n_frames == 120 and p.area_px == 64 * 64
    blinks, locs = p.columns()
    assert {"group_id", "n_in_group", "photons", "background"} <= set(locs)
