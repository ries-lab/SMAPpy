"""Binding-event cores (Steen et al. 2024): the first and last frame of each
DNA-PAINT binding event dropped, where the imager bound or left during the
exposure -- `group.event_cores`, and *Remove Localizations*' region
*binding-event edges*."""
import numpy as np
import pytest

from smappy.group import event_cores
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.remove_locs import (EdgeSettings, RemoveLocalizations,
                                        RemoveSettings)
from smappy.simulate.binding import bound_intervals

PHOTONS = 3000.0


def bindings(n_sites=30, n_frames=20000, precision_nm=2.0, seed=1):
    """Sites 1 um apart, each bound for 8 frames on average every 1500; a
    localization in every frame a binding covers for at least half, with
    photons in proportion to how much of the frame it covered.  Returns the
    table, sorted by frame, and the covered fraction of each row."""
    rng = np.random.default_rng(seed)
    site, start, end = bound_intervals(n_sites, n_frames, 8.0, 1500.0, rng)
    first = np.floor(start).astype(np.int64)
    last = np.ceil(end).astype(np.int64) - 1
    which = np.repeat(np.arange(len(site)), last - first + 1)
    frame = first[which] + np.concatenate([np.arange(n) for n in last - first + 1])
    cover = np.minimum(end[which], frame + 1.0) - np.maximum(start[which], frame)
    lit = cover >= 0.5
    which, frame, cover = which[lit], frame[lit], cover[lit]
    order = np.argsort(frame, kind="stable")
    which, frame, cover = which[order], frame[order], cover[order]
    s = site[which]
    xy = (np.column_stack([s % 6, s // 6]) * 1000.0
          + rng.normal(0, precision_nm, (len(s), 2)))
    locs = Localizations({"x_nm": xy[:, 0].astype(np.float32),
                          "y_nm": xy[:, 1].astype(np.float32),
                          "frame": frame, "site_true": s.astype(np.int32),
                          "photons": rng.poisson(PHOTONS * cover).astype(np.float32),
                          "xy_err_nm": np.full(len(s), precision_nm, np.float32)},
                         {"pixelsize_nm": 130.0})
    return locs, cover


def true_cores(site, frame, min_locs=3):
    """The cores by their definition: per site, runs of consecutive frames
    are the events; an event of at least ``min_locs`` keeps its inner frames."""
    keep = np.zeros(len(frame), bool)
    for s in np.unique(site):
        rows = np.flatnonzero(site == s)
        rows = rows[np.argsort(frame[rows], kind="stable")]
        runs = np.split(rows, np.flatnonzero(np.diff(frame[rows]) > 1) + 1)
        for run in runs:
            if len(run) >= min_locs:
                keep[run[1:-1]] = True
    return keep


def test_the_cores_are_each_events_frames_but_its_first_and_last():
    locs, _ = bindings()
    core, event = event_cores(locs["x_nm"], locs["y_nm"], locs["frame"], 13.0)
    expected = true_cores(np.asarray(locs["site_true"]), np.asarray(locs["frame"]))
    assert 0.5 < expected.mean() < 0.9          # a test with something to drop
    assert np.array_equal(core, expected)
    # one event per run: never two sites in one event
    for e in np.unique(event):
        assert len(np.unique(locs["site_true"][event == e])) == 1


def test_the_cores_photons_are_unbiased_where_all_localizations_are_not():
    """The edges had the imager for part of the frame: their photons pull the
    mean down, the cores' are the whole frame's."""
    locs, cover = bindings()
    core, _ = event_cores(locs["x_nm"], locs["y_nm"], locs["frame"], 13.0)
    photons = np.asarray(locs["photons"], float)
    assert photons.mean() < 0.98 * PHOTONS
    assert abs(photons[core].mean() / PHOTONS - 1) < 0.005
    assert (cover[core] > 1 - 1e-9).all()       # every core frame was covered whole


def test_dark_frames_join_an_event_across_a_missed_frame_and_min_locs_drops_short_ones():
    x = np.zeros(7)
    frame = np.array([0, 1, 2, 4, 5, 6, 20])     # frame 3 missed; frame 20 alone
    core, event = event_cores(x, x, frame, 10.0)
    assert list(core) == [False, True, False, False, True, False, False]
    core, event = event_cores(x, x, frame, 10.0, dark_frames=1)
    assert list(core) == [False, True, True, True, True, False, False]
    assert len(np.unique(event)) == 2
    core, _ = event_cores(x, x, frame, 10.0, min_locs=4)
    assert not core.any()


def test_files_are_linked_apart():
    x = np.zeros(6)
    frame = np.array([0, 1, 2, 0, 1, 2])
    blocks = np.array([1, 1, 1, 2, 2, 2])
    core, event = event_cores(x, x, frame, 10.0, blocks=blocks)
    assert list(core) == [False, True, False] * 2 and len(np.unique(event)) == 2


def ctx_of(locs):
    ctx = Context(locs=locs)
    ctx.session = None
    return ctx


def test_remove_localizations_keeps_the_cores_and_links_within_a_tenth_of_a_pixel():
    locs, _ = bindings()
    settings = RemoveSettings(region="edges")
    result = RemoveLocalizations().run(ctx_of(locs), settings)
    expected = true_cores(np.asarray(locs["site_true"]), np.asarray(locs["frame"]))
    assert len(result.locs) == expected.sum()
    assert "linked within 13 nm" in result.text
    assert np.array_equal(np.sort(result.locs["frame"]),
                          np.sort(np.asarray(locs["frame"])[expected]))


def test_hidden_edges_leave_every_merged_blink_that_has_a_core():
    """Hidden, the edges are a flag whose merging rule is *any*: every blink
    has edges, and *all* would hide every one."""
    from smappy.group import group
    from smappy.mathparse import recipes
    locs, _ = bindings(n_frames=4000)
    result = RemoveLocalizations().run(ctx_of(locs),
                                       RemoveSettings(region="edges", action="hide"))
    assert [r["grouped"] for r in recipes(result.locs) if r["field"] == "use"] == ["any"]
    grouped, _ = group(result.locs)
    expected = true_cores(np.asarray(locs["site_true"]), np.asarray(locs["frame"]))
    has_core = np.bincount(np.asarray(result.locs["group_id"]), weights=expected) > 0
    assert np.array_equal(np.asarray(grouped["use"]) > 0.5, has_core[1:])


def test_without_a_pixel_size_the_radius_must_be_set():
    locs, _ = bindings(n_frames=2000)
    locs.metadata.pop("pixelsize_nm")
    with pytest.raises(ValueError, match="set within"):
        RemoveLocalizations().run(ctx_of(locs), RemoveSettings(region="edges"))
    result = RemoveLocalizations().run(
        ctx_of(locs), RemoveSettings(region="edges", edges=EdgeSettings(within_nm=13.0)))
    assert len(result.locs) < len(locs)


def test_the_edges_part_is_shown_only_for_that_region():
    plugin = RemoveLocalizations()
    assert plugin.describe(None, RemoveSettings()).shown == {"edges": False}
    assert plugin.describe(None, RemoveSettings(region="edges")).shown == {"edges": True}
