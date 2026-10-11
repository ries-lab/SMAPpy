"""A session whose table a fit is still appending to, and the plugins run on it.

Three ways the growing table used to break: a plugin's context grew under it,
a copy of the session (`scratch`, a chain's) linked the live table and wrote
columns into it, and a table change was applied to it -- after which the next
block could not be appended and the fit's own result threw the change away.
"""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.plugins import Plugin, Result
from smappy.session import LiveTableError, Session


def block(frame, n=50):
    rng = np.random.default_rng(frame)
    return Localizations({"x_nm": rng.uniform(0, 5000, n),
                          "y_nm": rng.uniform(0, 5000, n),
                          "frame": np.full(n, frame, np.int64),
                          "photons": rng.uniform(100, 2000, n)}, {})


def live_session(blocks=3):
    session = Session()
    session.begin_live((0, 5000, 0, 5000))
    for frame in range(1, blocks + 1):
        session.append(block(frame))
    return session


class Shift(Plugin):
    """Stands in for a drift correction: a new table, x moved."""
    path = "Analysis/Drift/Shift"
    name = "Shift"

    def run(self, ctx, settings):
        locs = Localizations(dict(ctx.locs.columns), dict(ctx.locs.metadata))
        locs.columns["x_nm"] = locs["x_nm"] + 100.0
        return Result(locs=locs, text="shifted")


def test_a_snapshot_keeps_its_length_while_the_table_is_extended():
    table = Localizations()
    table.extend(block(1))
    snap = table.snapshot()
    for frame in range(2, 20):              # through several reallocations
        table.extend(block(frame))
    assert len(snap) == 50 and len(table) == 950
    np.testing.assert_array_equal(snap["x_nm"], block(1)["x_nm"])


def test_a_context_made_during_a_live_fit_does_not_grow_with_it():
    session = live_session()
    ctx = session.context()
    session.append(block(4))
    assert len(session.locs) == 200
    assert len(ctx.locs) == len(ctx.selection.mask) == 150
    assert len(ctx.selection.apply(ctx.locs)) == 150
    locs, selection = ctx.table()
    assert locs is ctx.locs and len(selection.mask) == 150


def test_every_layers_selection_agrees_with_the_pinned_table():
    session = live_session()
    session.add_layer()
    session.layers[1].set_bound("photons", 1000, 2000)
    ctx = session.context()
    session.append(block(4))
    bright = ctx.layer_selection(1)
    assert len(bright.mask) == len(ctx.locs) == 150
    assert np.all(bright.apply(ctx.locs)["photons"] >= 1000)
    assert len(ctx.layer_selection(1, without=("photons",)).mask) == 150


def test_a_grouped_table_is_refused_during_a_live_fit_when_the_layer_has_none():
    ctx = live_session().context(grouping="grouped")
    with pytest.raises(ValueError, match="grouped"):
        ctx.table()


def test_a_table_change_is_refused_while_the_fit_appends_and_the_view_keeps_growing():
    session = live_session()
    plugin = Shift()
    result = plugin.run(session.context(), None)
    with pytest.raises(LiveTableError, match="Shift changes the localizations"):
        session.apply(plugin, result)
    assert session.live and not session.history and not session.can_undo
    session.append(block(4))
    assert len(session.locs) == 200


def test_the_fits_own_result_finishes_the_live_table():
    session = live_session()
    finished = session.locs.snapshot().compact()
    session.apply(Shift(), Result(locs=finished, text="fit"), finishes_live=True)
    assert not session.live and len(session.locs) == 150


def test_a_scratch_copy_of_a_live_session_leaves_the_live_table_appendable():
    session = live_session()
    copy = session.scratch()               # links its own layers by default
    assert copy.locs is not session.locs and len(copy.locs) == 150
    assert "group_id" not in session.locs
    session.append(block(4))
    assert len(session.locs) == 200 and len(copy.locs) == 150


def test_a_fit_that_fails_leaves_what_it_streamed_as_an_ordinary_table():
    session = live_session()
    session.end_live()
    assert not session.live and len(session.locs) == 150
    result = Shift().run(session.context(), None)
    session.apply(Shift(), result)
    assert session.can_undo


def test_a_layer_added_during_a_live_fit_takes_each_block_once():
    session = live_session()
    session.add_layer()
    session.append(block(4))
    assert len(session.locs) == 200
    assert "group_id" not in session.locs
    for layer in session.layers:
        assert layer.filter.n_rows == 200
