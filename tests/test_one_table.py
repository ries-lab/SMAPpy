"""One table per session, one set of bounds per layer (`layer_state`).

The session owns the ungrouped table and one grouped table, a cache keyed by
the rows and the link settings; a layer owns its bounds, its files and its
grouped tick, and its filter is computed on the table it shows.
"""
import dataclasses

import numpy as np
import pytest

import smappy.layer_state as layer_state
from smappy.group import GroupSettings
from smappy.locs import Localizations
from smappy.plugins import Plugin, Result
from smappy.session import Session
from smappy.simulate import simulate


def a_session(n_frames=800, seed=3):
    return Session(simulate(n_frames=n_frames, seed=seed))


@pytest.fixture
def links(monkeypatch):
    """How many times anything was linked."""
    calls = []
    real = layer_state.group

    def counting(*args, **kwargs):
        calls.append(args[1] if len(args) > 1 else kwargs.get("settings"))
        return real(*args, **kwargs)
    monkeypatch.setattr(layer_state, "group", counting)
    return calls


def test_one_bound_keeps_different_localizations_grouped_and_ungrouped():
    session = a_session()
    layer = session.layers[0]
    layer.set_ranges({"xy_err_nm": (None, 8.0)})          # this bound alone
    grouped, selection = layer.grouped_selection(0)
    ungrouped = layer.selection(0).mask

    # each table's own values decide: the frames' precisions, and the
    # combined precision of each blink
    np.testing.assert_array_equal(ungrouped, np.asarray(session.locs["xy_err_nm"]) <= 8.0)
    np.testing.assert_array_equal(selection.mask, np.asarray(grouped["xy_err_nm"]) <= 8.0)
    assert 0 < ungrouped.mean() < 1
    # a blink's precision beats its frames', so more of the blinks pass
    assert selection.mask.mean() > ungrouped.mean()
    assert layer.ranges == {"xy_err_nm": (None, 8.0)}      # one setting
    assert layer.filter.ranges == layer.ranges             # drawn: the grouped table


def test_the_grouped_table_is_linked_again_when_the_link_settings_change(links):
    session = a_session()
    before = session.tables.peek_grouped()
    assert session.tables.is_current and len(links) == 1      # linked on open
    links.clear()

    session.set_group_settings(GroupSettings(dx=20.0, dt=1))
    after = session.tables.peek_grouped()
    assert len(links) == 1                                     # once, for every layer
    assert after is not before and session.tables.is_current
    assert len(after.locs) > len(before.locs)                  # a tighter box links less
    assert session.layers[0].locs is after.locs                # what the layer draws
    assert not hasattr(session.layers[0].state, "grouped_stale")


def test_the_grouped_table_is_linked_again_when_the_table_changes(links):
    session = a_session()
    links.clear()
    layer = session.layers[0]
    layer.set_bound("photons", 500.0, None)
    before = session.tables.peek_grouped()

    class Shift(Plugin):
        changes_locs = True
        path = "Test/Shift"
        grouping = "ungrouped"

        def run(self, ctx, settings=None):
            columns = {k: np.asarray(v).copy() for k, v in ctx.locs.columns.items()}
            columns["x_nm"] = columns["x_nm"] + 10.0
            return Result(locs=Localizations(columns, dict(ctx.locs.metadata)))

    session.run(Shift())
    after = session.tables.peek_grouped()
    assert len(links) == 1 and after is not before and session.tables.is_current
    assert layer.locs is after.locs
    assert layer.filter.ranges["photons"] == (500.0, None)     # the bound stands
    np.testing.assert_allclose(np.asarray(after.locs["x_nm"]),
                               np.asarray(before.locs["x_nm"]) + 10.0, atol=1e-3)


def test_a_second_layer_over_the_table_shares_the_one_grouped_table(links):
    session = a_session()
    links.clear()
    first = session.layers[0]
    second = session.add_layer(like=0)
    third = session.add_layer()                   # no template: grouped by default
    assert third.grouped and second.grouped
    assert len(links) == 0                        # linked on open, never again
    assert second.locs is first.locs is third.locs
    assert second.state.index is first.state.index           # one spatial index
    # and each keeps its own bounds over it
    first.set_bound("photons", 1000.0, None)
    second.set_bound("photons", None, 1000.0)
    a, b = first.filter.mask, second.filter.mask
    assert not (a & b).any() and (a | b).any()


def test_a_chain_copy_shares_the_tables_and_links_nothing(links):
    session = a_session()
    links.clear()
    session.add_layer(like=0)
    session.layers[1].set_bound("photons", 1000.0, None)
    copy = session.scratch()
    assert len(links) == 0
    assert copy.locs is session.locs
    assert copy.layers[1].locs is session.layers[1].locs
    assert copy.layers[1].ranges == session.layers[1].ranges
    copy.set_group_settings(GroupSettings(dx=20.0, dt=1))     # the copy's own
    assert len(links) == 1
    assert session.tables.is_current                          # the session's stands
    assert copy.layers[0].locs is not session.layers[0].locs


def test_the_roi_manager_follows_the_sessions_link_settings():
    locs = simulate(n_frames=800, seed=3)
    session = Session()
    from smappy.io.formats import FileInfo
    session.add_file(locs, FileInfo("a", "/a.h5", "smappy"))
    session.add_file(simulate(n_frames=800, seed=4), FileInfo("b", "/b.h5", "smappy"),
                     append=True)
    session.set_group_settings(GroupSettings(dx=30.0, dt=1))
    project = session.rois
    assert project.group_settings == session.group_settings
    file_id = list(project.sources)[1]
    state = project.state(file_id)
    state.use_grouped = True
    # the file's blinks are the session's blinks of that file: nothing relinked
    whole = session.tables.grouped().locs
    mine = np.asarray(whole["filenumber"]) == 1
    assert len(state.locs) == int(mine.sum())
    np.testing.assert_array_equal(np.asarray(state.locs["x_nm"]),
                                  np.asarray(whole["x_nm"])[mine])


def block(frame, n=300):
    rng = np.random.default_rng(frame)
    return Localizations({"x_nm": rng.uniform(0, 5000, n),
                          "y_nm": rng.uniform(0, 5000, n),
                          "frame": np.full(n, frame, np.int64),
                          "photons": rng.uniform(100, 2000, n).astype(np.float32),
                          "xy_err_nm": rng.uniform(5, 40, n)}, {})


def test_a_live_fit_appends_once_and_every_layer_follows():
    session = Session(simulate(n_frames=200, seed=1))
    before = session.locs
    session.begin_live((0, 5000, 0, 5000))
    session.append(block(1))
    second = session.add_layer(like=0)
    second.set_bound("photons", 1000.0, None)
    for frame in range(2, 6):
        session.append(block(frame))
    assert len(session.locs) == 5 * 300                       # each block once
    first = session.layers[0]
    assert first.filter.n_rows == second.filter.n_rows == len(session.locs)
    assert first.ranges["xy_err_nm"] == (None, 25.0)          # the defaults came with block 1
    photons = np.asarray(session.locs["photons"])
    np.testing.assert_array_equal(second.filter.mask,
                                  first.filter.mask & (photons >= 1000.0))

    # grouped while it grows: kept as linked until somebody asks again
    second.show_grouped(True)
    linked = session.tables.peek_grouped()
    session.append(block(6))
    assert not session.tables.is_current and second.locs is linked.locs
    assert len(session.locs) == 6 * 300
    locs, _ = session.table(1, "grouped")                      # asked: linked again
    assert session.tables.is_current and locs is not linked.locs

    # and the table before the fit is one undo away
    session.end_live()
    assert not session.live
    session.undo()
    assert session.locs is before
    assert session.layers[0].locs is session.tables.grouped().locs


def test_undo_brings_back_the_table_the_bounds_and_the_link_settings_stay():
    session = a_session()
    layer = session.layers[0]
    layer.set_bound("photons", 300.0, None)
    original = session.locs
    keep = np.arange(len(original)) % 2 == 0
    session.set_locs(original[keep], label="halve")
    session.set_group_settings(GroupSettings(dx=20.0, dt=1))
    session.undo()
    assert session.locs is original
    assert session.tables.is_current
    assert session.tables.settings == GroupSettings(dx=20.0, dt=1)
    assert layer.grouped and layer.ranges["photons"] == (300.0, None)
    grouped = session.tables.peek_grouped().locs
    np.testing.assert_array_equal(layer.filter.mask,
                                  layer.filter.mask & (np.asarray(grouped["photons"]) >= 300))
    assert dataclasses.asdict(session.group_settings)["dx"] == 20.0
