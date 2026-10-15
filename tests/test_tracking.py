"""Tracks: the simulation of Brownian tracks with states and blinking, and the
two linkers of Link tracks recovering them across dark frames."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.mathparse import recipes
from smappy.plugins import Context, Selection
from smappy.plugins.link_tracks import LinkTracks
from smappy.simulate import jump_nm, simulate_tracks
from smappy.tracking import candidates, link_tracks, score, track_ids


def link(locs, **values):
    return link_tracks(locs["x_nm"], locs["y_nm"], locs["frame"], **values)


def test_the_simulation_keeps_its_states_fractions_and_its_diffusion():
    locs = simulate_tracks(n_frames=400, density_um2=0.05, d_um2_s=(0.05, 1.0),
                           fractions=(0.3, 0.7), switch_probability=0.05, seed=3)
    state = np.asarray(locs["state_true"])
    assert abs((state == 0).mean() - 0.3) < 0.04
    # the jumps between consecutive frames of a track, in its fast state
    truth, frame = np.asarray(locs["track_id_true"]), np.asarray(locs["frame"])
    order = np.lexsort((frame, truth))
    x = np.asarray(locs["x_nm"], float)[order]
    y = np.asarray(locs["y_nm"], float)[order]
    step = (truth[order][1:] == truth[order][:-1]) & (np.diff(frame[order]) == 1)
    fast = step & (state[order][:-1] == 1) & (state[order][1:] == 1)
    msd = np.mean(np.diff(x)[fast] ** 2 + np.diff(y)[fast] ** 2)
    # 4 D t plus the precision of both ends, 4 sigma^2
    expected = jump_nm(1.0, 10.0) ** 2 + 4 * 20.0 ** 2
    assert msd == pytest.approx(expected, rel=0.08)


def test_blinking_leaves_dark_frames_inside_the_tracks():
    locs = simulate_tracks(n_frames=300, p_dark=0.2, dark_frames=2.0, seed=1)
    truth, frame = np.asarray(locs["track_id_true"]), np.asarray(locs["frame"])
    order = np.lexsort((frame, truth))
    same = truth[order][1:] == truth[order][:-1]
    gaps = np.diff(frame[order])[same] - 1
    # a dark run is geometric with mean 2 frames: some, not most, links skip frames
    assert 0.1 < (gaps > 0).mean() < 0.4
    assert gaps[gaps > 0].mean() == pytest.approx(2.0, rel=0.15)


def test_candidates_are_the_pairs_within_the_jump_and_the_memory():
    x = np.array([0.0, 100.0, 5000.0, 150.0, 120.0])
    y = np.zeros(5)
    frame = np.array([0, 1, 1, 3, 6])
    c = candidates(x, y, frame, max_jump_nm=200.0, memory=1)
    found = sorted(zip(c.from_.tolist(), c.to.tolist(), c.lag.tolist()))
    # 0->1 (one frame), 1->3 (two frames: memory 1); 0->3 is three frames,
    # 3->4 three frames, 2 is too far from everything
    assert found == [(0, 1, 1), (1, 3, 2)]
    assert np.allclose(sorted(c.d2), [100.0 ** 2, 50.0 ** 2][::-1])


@pytest.mark.parametrize("algorithm", ["simple", "global"])
def test_the_linkers_recover_sparse_tracks_across_dark_frames(algorithm):
    locs = simulate_tracks(n_frames=300, density_um2=0.05, d_um2_s=(0.5,), p_dark=0.15,
                           dark_frames=1.5, seed=5)
    max_jump = 3 * np.hypot(jump_nm(0.5, 10.0), 40.0)
    with_memory = score(link(locs, max_jump_nm=max_jump, memory=3, algorithm=algorithm)[0],
                        locs["track_id_true"], locs["frame"])
    assert with_memory["links_found"] > 0.97
    assert with_memory["links_wrong"] < 0.01
    assert with_memory["tracks_whole"] > 0.8
    # without memory, every dark frame breaks a track
    no_memory = score(link(locs, max_jump_nm=max_jump, memory=0, algorithm=algorithm)[0],
                      locs["track_id_true"], locs["frame"])
    assert no_memory["links_found"] < 0.9
    assert no_memory["tracks_whole"] < 0.5 * with_memory["tracks_whole"]


def test_the_global_assignment_resolves_a_crossing_the_simple_linker_gets_wrong():
    # two ends; the simple linker gives the nearest pair (b->c, 30 nm) first and
    # leaves a with nothing; the least squared jumps, an end left unlinked
    # costing the maximum jump squared, link a->c (70 nm) and b->d (90 nm)
    x = np.array([0.0, 100.0, 70.0, 190.0])
    frame = np.array([0, 0, 1, 1])
    y = np.zeros(4)
    simple, _ = link_tracks(x, y, frame, max_jump_nm=130.0, memory=0, algorithm="simple")
    best, _ = link_tracks(x, y, frame, max_jump_nm=130.0, memory=0, algorithm="global")
    assert simple.tolist() == [0, 1, 1, 0]
    assert best[0] == best[2] > 0 and best[1] == best[3] > 0 and best[0] != best[1]


def test_short_tracks_and_lone_localizations_get_track_id_0():
    # successors: 0->1->2 (three), 3->4 (two), 5 alone
    nxt = np.array([1, 2, -1, 4, -1, -1])
    frame = np.array([0, 1, 2, 5, 6, 0])
    track, length = track_ids(nxt, frame, min_length=3)
    assert track.tolist() == [1, 1, 1, 0, 0, 0]
    assert length.tolist() == [3, 3, 3, 0, 0, 0]
    track, length = track_ids(nxt, frame, min_length=2)
    assert track.tolist() == [1, 1, 1, 2, 2, 0]
    assert length.tolist() == [3, 3, 3, 2, 2, 0]


def test_groups_are_linked_apart():
    # a track in each file, at the same place: never one track of the two
    x = np.array([0.0, 10.0, 5.0, 15.0])
    frame = np.array([0, 1, 2, 3])
    groups = np.array([0, 0, 1, 1])
    track, _ = link_tracks(x, np.zeros(4), frame, max_jump_nm=50.0, groups=groups)
    assert track.tolist() == [1, 1, 2, 2]
    track, _ = link_tracks(x, np.zeros(4), frame, max_jump_nm=50.0)
    assert track.tolist() == [1, 1, 1, 1]
    # a group below 0 is in no track
    track, _ = link_tracks(x, np.zeros(4), frame, max_jump_nm=50.0,
                           groups=np.array([0, 0, -1, -1]))
    assert track.tolist() == [1, 1, 0, 0]


def test_link_in_3d_uses_the_jump_in_z():
    x = np.zeros(2)
    frame = np.array([0, 1])
    z = np.array([0.0, 400.0])
    assert link_tracks(x, x, frame, z, max_jump_nm=200.0)[0].tolist() == [0, 0]
    assert link_tracks(x, x, frame, None, max_jump_nm=200.0)[0].tolist() == [1, 1]


def test_the_plugin_writes_track_id_and_track_length_for_the_selection():
    locs = simulate_tracks(n_frames=200, density_um2=0.05, d_um2_s=(0.2,), p_dark=0.1,
                           seed=2)
    keep = np.asarray(locs["x_nm"]) < 10000
    plugin = LinkTracks()
    settings = LinkTracks.Settings(max_jump_nm=250.0, memory=3, min_length=3)
    result = plugin.run(Context(locs=locs, selection=Selection(keep)), settings)
    out = result.locs
    track, length = np.asarray(out["track_id"]), np.asarray(out["track_length"])
    assert (track[~keep] == 0).all()
    assert (track[keep] > 0).mean() > 0.9
    assert (length[track > 0] >= 3).all()
    assert (np.bincount(track)[track[track > 0]] == length[track > 0]).all()
    assert {r["field"]: r["grouped"] for r in recipes(out)} == {
        "track_id": "max", "track_length": "max"}
    assert "tracks" in result.text


def test_the_plugin_refuses_a_table_without_frames():
    locs = Localizations({"x_nm": np.zeros(3), "y_nm": np.zeros(3)}, {})
    with pytest.raises(ValueError, match="frame"):
        LinkTracks().run(Context(locs=locs), LinkTracks.Settings())


def test_track_columns_resolve_in_order_and_any_integer_column_can_be_chosen():
    from smappy.tracking.columns import track_column, track_column_choices
    base = {"x_nm": np.zeros(2), "frame": np.arange(2)}
    both = Localizations({**base, "tid": np.ones(2, int),
                          "track_id_true": np.ones(2, int)}, {})
    assert track_column(both) == "tid"
    assert track_column(both, "frame") == "frame"
    assert track_column_choices(both)[0] == ("", "auto (tid)")
    with pytest.raises(ValueError, match="frame, tid, track_id_true"):
        track_column(both, "group_id")
    # group_id is never auto: nearly every table has one
    with pytest.raises(ValueError, match="none of"):
        track_column(Localizations({**base, "group_id": np.ones(2, int)}, {}))


def test_link_tracks_writes_where_it_is_told_and_never_over_a_column_of_its_own():
    locs = simulate_tracks(n_frames=100, density_um2=0.05, d_um2_s=(0.2,), seed=2)
    first = LinkTracks().run(Context(locs=locs), LinkTracks.Settings(max_jump_nm=250.0))
    second = LinkTracks().run(Context(locs=first.locs),
                              LinkTracks.Settings(max_jump_nm=100.0, memory=0,
                                                  column="track_id_strict"))
    out = second.locs
    assert {"track_id", "track_length", "track_id_strict",
            "track_id_strict_length"} <= set(out.keys())
    assert out["track_id_strict"].max() > out["track_id"].max()  # stricter: more, shorter
    # again into its own column is fine; over the table's own is refused
    LinkTracks().run(Context(locs=out), LinkTracks.Settings(column="track_id_strict"))
    with pytest.raises(ValueError, match="table's own"):
        LinkTracks().run(Context(locs=out), LinkTracks.Settings(column="track_id_true"))
    with pytest.raises(ValueError, match="table's own"):
        LinkTracks().run(Context(locs=out), LinkTracks.Settings(column="frame"))
