"""4Pi in situ, SMAPpy's side: the molecules chosen from a fit and cut in every channel
(`calibrate/fourpi_insitu.py`; docs/4pi.md §27).  The learning is uipsf's
(`tests/uipsf_tests/test_fourpi_insitu.py`)."""

from types import SimpleNamespace

import numpy as np

from smappy import fourpi
from smappy.calibrate.fourpi_insitu import FourPiInSituSettings, alone, choose, cut
from smappy.channels import from_corners
from smappy.io.tiff import ImageSource


def _table(n, rng, frames=200):
    return {"x_nm": rng.uniform(1000, 5000, n), "y_nm": rng.uniform(1000, 5000, n),
            "z_env_nm": rng.uniform(-600, 600, n), "photons": rng.lognormal(7, 0.5, n),
            "phase_rad": rng.uniform(-3, 3, n), "frame": rng.integers(0, frames, n)}


def test_molecules_are_chosen_bright_alone_inside_the_range_and_spread_evenly_over_z():
    rng = np.random.default_rng(0)
    locs = _table(20000, rng, frames=20000)
    s = FourPiInSituSettings(molecules=800, z_margin_nm=100, z_bins=4)
    index = choose(locs, 600.0, s)
    assert len(index) == 800 and len(np.unique(index)) == 800
    z, photons = locs["z_env_nm"][index], locs["photons"][index]
    assert np.all(np.abs(z) <= 500) and photons.min() >= np.median(locs["photons"])
    counts = np.histogram(z, np.linspace(-500, 500, 5))[0]
    assert counts.min() >= 190
    odd = choose(locs, 600.0, s, frames=lambda f: f % 2 == 1)
    assert np.all(locs["frame"][odd] % 2 == 1)


def test_a_molecule_with_a_neighbour_in_its_frame_is_not_alone():
    x = np.array([0.0, 500.0, 5000.0, 0.0])
    frame = np.array([1, 1, 1, 2])
    assert alone(x, np.zeros(4), frame, 1000.0).tolist() == [False, False, True, True]


class _Frames(ImageSource):
    def frame(self, index):
        return self.stack[index]


def test_each_molecule_is_cut_in_every_channel_where_the_layout_maps_it():
    """Molecules drawn on a strip of four channels, two mirrored and one shifted by a fraction of a pixel:
    every box is centred on its molecule in every channel, and the remainder in channel 0 is the start's
    x, y."""
    cal = fourpi.model_calibration()
    layout = from_corners([0, 70, 140, 210], [0] * 4, 64, 64, flip_x=[0, 0, 1, 1])
    m = np.eye(3)
    m[0, 2], m[1, 2] = 0.6, -0.4
    layout = layout.with_maps([np.eye(3), np.eye(3), m, np.eye(3)])
    rng = np.random.default_rng(1)
    n = 40
    x, y = rng.uniform(12, 50, (2, n))
    frame = np.arange(n)
    frames = fourpi.render_frames(cal, layout, frame, x, y, np.zeros(n), 20000, phase0=np.zeros(n),
                                  n_frames=n, background=0.0)
    camera = SimpleNamespace(conversion=1.0, offset=0.0, adu_to_photons=1.0, roi_offset=(0, 0),
                             require=lambda *a: None)
    source = _Frames(files=[], shape=frames.shape[1:], dtype=np.float32, n_frames=n, n_frames_declared=n,
                     mm_metadata=None, summary=None)
    source.stack = frames
    px = 100.0
    locs = {"x_nm": layout.on_frame(0, np.c_[x, y])[:, 0] * px, "y_nm": layout.on_frame(0, np.c_[x, y])[:, 1] * px,
            "z_env_nm": np.zeros(n), "photons": np.full(n, 20000.0), "frame": frame}
    got = cut(source, layout, locs, np.arange(n), camera, 11, (px, px))
    assert len(got) == n
    assert np.allclose(got.start["xyz"][:, 0], (x - np.rint(x)) * px)
    assert np.allclose(got.start["xyz"][:, 1], (y - np.rint(y)) * px)
    summed = got.data.sum(0)                                   # every box's light, (C, 11, 11)
    for c in range(4):
        cy, cx = np.unravel_index(np.argmax(summed[c]), summed[c].shape)
        assert (cy, cx) == (5, 5)
    assert np.allclose(got.roi_xy[:, 0], np.rint(np.c_[x, y]))


def test_a_pupil_model_saved_without_its_recipe_is_refused_with_the_way_out(tmp_path):
    # P1's tubulin model (docs/4pi.md §26, §29) was saved the day before pupil models kept their recipe
    import pytest

    from smappy.calibrate.fourpi_insitu import bead_model_of
    cal = fourpi.model_calibration(size=15, z_range_nm=300.0)
    cal.parameters = {"method": "uiPSF v2 4Pi bead model"}
    fourpi.save_fourpi_calibration(tmp_path / "old_4pi.h5", cal)
    with pytest.raises(ValueError, match="learn it again"):
        bead_model_of(tmp_path / "old_4pi.h5")
    cal.parameters = {"method": "IAB from a bead scan"}
    fourpi.save_fourpi_calibration(tmp_path / "average_4pi.h5", cal, overwrite=True)
    with pytest.raises(ValueError, match="bead average"):
        bead_model_of(tmp_path / "average_4pi.h5")


def test_a_better_held_out_likelihood_with_z_on_other_fringes_or_a_moved_period_is_still_warned_of():
    # the 2017-11-28 tubulin (docs/4pi.md §29): 97 nats a molecule better, 15 % on another fringe, 294 -> 282.5 nm
    from smappy.calibrate.fourpi_insitu import FourPiInSituRun, warnings_of
    rng = np.random.default_rng(0)
    z = rng.uniform(-400, 400, 100)
    moved = z + 20 + np.where(rng.random(100) < 0.15, 294.2, 0.0)
    bead = SimpleNamespace(period_nm=294.2)

    def run(z_insitu, objective, period):
        return FourPiInSituRun(None, None, bead, None, np.arange(100), {"bead": 1000.0, "in situ": objective},
                               None, calibration=SimpleNamespace(period_nm=period),
                               held_z={"bead": z, "in situ": z_insitu})

    assert warnings_of(run(z + 20, 900.0, 293.0)) == []
    tubulin = run(moved, 900.0 - 9700, 282.5)
    assert 0.1 < tubulin.other_fringe < 0.2
    text = " ".join(warnings_of(tubulin))
    assert "another fringe" in text and "period moved" in text and "no better" not in text
    assert "no better" in " ".join(warnings_of(run(z, 1000.0, 294.2)))
