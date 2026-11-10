"""Overlay channels: a split camera's second half put on the first, one dye in each.

Tables in chip pixels with a known registration, and the numbers asserted back; then the plugin end to end, on bead movies it fits itself.
"""
from dataclasses import replace

import numpy as np
import pytest

from smappy.calibrate.dual import ArrangementSettings, ChannelTransform, split_geometry
from smappy.channel_overlay import (bead_key, cached_transform, overlay_table, secondary_rows,
                                    transform_path)
from smappy.locs import Localizations
from smappy.plugins import Context

SHAPE = (96, 64)
SPLIT = 48
PIXEL = 100.0


def registration(dx=0.8, dy=-0.5, angle_deg=0.0):
    """Secondary -> main in chip pixels: the lower half, turned by ``angle_deg`` and shifted."""
    a = np.deg2rad(angle_deg)
    forward = np.eye(3)                                  # main -> secondary
    forward[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    forward[:2, 2] = [dx, SPLIT + dy]
    return ChannelTransform(np.linalg.inv(forward),
                            split_geometry("up-down", "upper", SHAPE, split_position=SPLIT), {})


def two_dyes(t, n_frames=400, per=4, moved=(0.0, 0.0), crosstalk=0.04, sigma=0.02, seed=1):
    """A two-channel fit's table: two dyes, each in its own half and ``crosstalk`` of its light
    in the other; the second half off by ``moved`` (px) from where the bead map ``t`` puts it.
    The main-half positions are the truth, ``true_x`` / ``true_y``."""
    rng = np.random.default_rng(seed)
    forward = np.linalg.inv(t.transformation)
    cols = {k: [] for k in ("frame", "x", "y", "photons", "camera_channel", "dye", "tx", "ty")}

    def add(frame, xy, photons, half, dye, true):
        cols["frame"].append(frame)
        cols["x"].append(xy[0] + rng.normal(0, sigma / np.sqrt(photons / 5000)))
        cols["y"].append(xy[1] + rng.normal(0, sigma / np.sqrt(photons / 5000)))
        cols["photons"].append(photons)
        cols["camera_channel"].append(half)
        cols["dye"].append(dye)
        cols["tx"].append(true[0])
        cols["ty"].append(true[1])

    for f in range(n_frames):
        for _ in range(per):
            p = np.array([rng.uniform(4, SHAPE[1] - 4), rng.uniform(4, SPLIT - 4)])
            q = (forward @ np.r_[p, 1])[:2] + moved
            dye = int(rng.integers(0, 2))
            bright, dim = 5000.0 * (1 - crosstalk), 5000.0 * crosstalk
            own, other = ((p, 0), (q, 1)) if dye == 0 else ((q, 1), (p, 0))
            add(f, own[0], bright, own[1], dye, p)
            if crosstalk and rng.random() < 0.5:        # a dim partner is not always found
                add(f, other[0], dim, other[1], dye, p)
    c = {k: np.asarray(v) for k, v in cols.items()}
    return Localizations({"frame": c["frame"].astype(np.int64),
                          "x_nm": (c["x"] * PIXEL).astype(np.float64),
                          "y_nm": (c["y"] * PIXEL).astype(np.float64),
                          "photons": c["photons"].astype(np.float32),
                          "camera_channel": c["camera_channel"].astype(np.int8),
                          "dye": c["dye"], "true_x": c["tx"], "true_y": c["ty"]},
                         {"pixelsize_nm": PIXEL})


def chip(locs):
    return np.asarray(locs["x_nm"]) / PIXEL, np.asarray(locs["y_nm"]) / PIXEL


def error_px(table, half=1, rows=None):
    """Where each localization of ``half`` lands against its molecule, in px, bright ones only."""
    rows = (np.asarray(table["camera_channel"]) == half) & (np.asarray(table["photons"]) > 1000)
    x, y = chip(table)
    return np.c_[x[rows] - table["true_x"][rows], y[rows] - table["true_y"][rows]]


# --------------------------------------------------------------------------- moving the table
def test_the_second_half_lands_on_its_molecules_and_channel_is_the_half():
    t = registration(angle_deg=1.0)
    locs = two_dyes(t, n_frames=50, crosstalk=0.0)
    out = overlay_table(locs, t)
    assert np.max(np.abs(np.median(error_px(out), axis=0))) < 0.005
    # the main half is not moved
    np.testing.assert_array_equal(np.asarray(out["x_nm"])[~secondary_rows(locs)],
                                  np.asarray(locs["x_nm"])[~secondary_rows(locs)])
    np.testing.assert_array_equal(out["channel"], locs["camera_channel"])
    assert out.metadata["transformations"][-1]["name"] == "overlay channels"


def test_a_table_is_not_overlaid_twice():
    t = registration()
    once = overlay_table(two_dyes(t, n_frames=5, crosstalk=0.0), t)
    with pytest.raises(ValueError, match="already"):
        overlay_table(once, t)


def test_a_table_not_fitted_as_two_channels_is_refused_with_how_to_fit_it():
    locs = Localizations({"frame": np.zeros(3, np.int64), "x_nm": np.zeros(3),
                          "y_nm": np.zeros(3)}, {"pixelsize_nm": PIXEL})
    with pytest.raises(ValueError, match="two channels"):
        secondary_rows(locs)


# ---------------------------------------------------------------------------- keeping the beads
def test_a_kept_registration_is_used_only_for_the_same_beads_and_settings(tmp_path):
    beads = tmp_path / "beads.tif"
    beads.write_bytes(b"0" * 10)
    key = bead_key([beads], registration={"model": "projective"})
    t = registration()
    t.parameters["beads"] = key
    from smappy.calibrate.transform import save_channel_transform
    where = transform_path(beads)
    assert where == tmp_path / "beads_2ct.h5"
    save_channel_transform(where, t)
    assert cached_transform(where, key) is not None
    assert cached_transform(where, bead_key([beads], registration={"model": "polynomial"})) is None
    beads.write_bytes(b"0" * 11)                       # the beads changed
    assert cached_transform(where, bead_key([beads], registration={"model": "projective"})) is None
    folder = tmp_path / "bead folder"
    folder.mkdir()
    assert transform_path(folder) == tmp_path / "bead folder_2ct.h5"


# ------------------------------------------------------------------------------- the plugin
def bead_movie(path, rng, t, n_frames=15):
    """A split movie of beads, each seen in both halves, written to ``path``."""
    import tifffile
    from scipy.special import erf
    forward = np.linalg.inv(t.transformation)
    beads = np.c_[rng.uniform(6, SHAPE[1] - 6, 10), rng.uniform(6, SPLIT - 6, 10)]
    iy, ix = np.arange(SHAPE[0]), np.arange(SHAPE[1])

    def spot(frame, x, y, n, s):
        k = np.sqrt(1 / 2 / s ** 2)
        gx = .5 * (erf((ix - x + .5) * k) - erf((ix - x - .5) * k))
        gy = .5 * (erf((iy - y + .5) * k) - erf((iy - y - .5) * k))
        frame += n * np.outer(gy, gx)

    frames = []
    for _ in range(n_frames):
        frame = np.full(SHAPE, 15.0)
        for x, y in beads:
            q = (forward @ [x, y, 1])[:2]
            spot(frame, x, y, 20000, 1.1)
            spot(frame, q[0], q[1], 20000, 1.3)
        frames.append(rng.poisson(frame))
    tifffile.imwrite(path, (np.stack(frames) + 100).astype(np.uint16))


def plugin_settings(tmp_path, **changes):
    from smappy.plugins.overlay_channels import OverlaySettings
    return replace(OverlaySettings(
        source="movies", beads=str(tmp_path / "beads"),
        channels=ArrangementSettings(layout="up-down", main_channel="upper"),
        camera={"conversion": 1.0, "offset": 100.0, "pixelsize_um": 0.1}), **changes)


def test_the_plugin_fits_a_bead_folder_registers_it_and_keeps_it(tmp_path):
    from smappy.plugins.overlay_channels import OverlayChannels
    rng = np.random.default_rng(5)
    t = registration(dx=0.6, dy=-0.4, angle_deg=0.5)
    (tmp_path / "beads").mkdir()
    for k in range(2):
        bead_movie(tmp_path / "beads" / f"position{k}.tif", rng, t)
    locs = two_dyes(t, n_frames=30, crosstalk=0.0)
    result = OverlayChannels().run(Context(locs=locs), plugin_settings(tmp_path))
    assert "measured on 2 bead movies" in result.text
    assert np.max(np.abs(np.median(error_px(result.locs), axis=0))) < 0.05
    assert (tmp_path / "beads_2ct.h5").exists()
    again = OverlayChannels().run(Context(locs=locs), plugin_settings(tmp_path))
    assert "measured on these beads before" in again.text
    np.testing.assert_allclose(again.locs["x_nm"], result.locs["x_nm"])
    # and the file it kept serves the next sample as a transformation file
    third = OverlayChannels().run(Context(locs=locs), plugin_settings(
        tmp_path, source="file", path=str(tmp_path / "beads_2ct.h5")))
    np.testing.assert_allclose(third.locs["x_nm"], result.locs["x_nm"])


def test_a_fit_on_two_channels_comes_out_overlaid_from_its_after_the_fit(tmp_path):
    """The step inside a fitter: the beads read in the fitter's own arrangement and camera."""
    import tifffile
    from smappy.plugins.after_fit import AfterFitSettings
    from smappy.plugins.fit import (CameraSettings, ChannelsSettings, GaussianFit,
                                    GaussianFitSettings, OutputSettings, SourceSettings)
    rng = np.random.default_rng(6)
    t = registration(dx=0.6, dy=-0.4)
    (tmp_path / "beads").mkdir()
    bead_movie(tmp_path / "beads" / "beads.tif", rng, t)
    bead_movie(tmp_path / "sample.tif", rng, t, n_frames=5)
    overlay = plugin_settings(tmp_path, channels=ArrangementSettings(), camera={})
    settings = GaussianFitSettings(
        source=SourceSettings(path=str(tmp_path / "sample.tif")),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        channels=ChannelsSettings(split="two channels", layout="up-down", main_channel="upper"),
        output=OutputSettings(save=False))
    settings.finish = AfterFitSettings(overlay=True, overlay_channels=overlay)
    out = GaussianFit().run(Context(), settings).locs
    assert "channel" in out
    # every bead's two localizations now at one place
    from scipy.spatial import cKDTree
    first = out[np.asarray(out["camera_channel"]) == 0]
    second = out[np.asarray(out["camera_channel"]) == 1]
    d, _ = cKDTree(np.c_[1e6 * first["frame"], first["x_nm"], first["y_nm"]]).query(
        np.c_[1e6 * second["frame"], second["x_nm"], second["y_nm"]])
    assert np.median(d) < 5.0


def test_the_overlay_step_is_offered_only_after_a_fit_on_two_channels():
    from smappy.plugins.after_fit import AfterFitSettings, applicable, shown
    from smappy.plugins.fit import ChannelsSettings, GaussianFit, GaussianFitSettings
    s = AfterFitSettings(overlay=True)
    assert not applicable(s, two_channel=False, widths=False, split=False).overlay
    assert applicable(s, two_channel=False, widths=False, split=True).overlay
    assert shown(s, False, False, split=True)["overlay_channels"]
    assert not shown(s, False, False, split=False)["overlay"]
    fit = GaussianFitSettings(channels=ChannelsSettings(split="two channels"))
    fit.finish = s
    state = GaussianFit().describe(None, fit)
    assert state.shown["finish.overlay"]
    # the fitter's own arrangement is used, so the step does not ask it again
    assert state.shown["finish.overlay_channels.channels"] is False
    assert not GaussianFit().describe(None, GaussianFitSettings()).shown["finish.overlay"]


def test_the_step_takes_the_fitters_camera_and_arrangement():
    from smappy.plugins.fit import CameraSettings, ChannelsSettings, GaussianFitSettings
    from smappy.plugins.overlay_channels import OverlaySettings, with_fitter
    fit = GaussianFitSettings(
        camera=CameraSettings(conversion=2.0, offset=90.0),
        channels=ChannelsSettings(split="two channels", layout="right-left", main_channel="left"))
    s = with_fitter(OverlaySettings(), fit)
    assert s.camera == {"conversion": 2.0, "offset": 90.0}
    assert (s.channels.layout, s.channels.main_channel) == ("right-left", "left")


def test_a_fitter_without_a_split_cameras_channels_leaves_the_arrangement_alone():
    """Spline 4Pi's *channels* are four rectangles, not a split: nothing to take from them."""
    from smappy.plugins.fit_4pi import FourPiFitSettings
    from smappy.plugins.overlay_channels import OverlaySettings, with_fitter
    s = with_fitter(OverlaySettings(), FourPiFitSettings())
    assert s.channels == ArrangementSettings()

