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
    changes_locs = True
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
    with pytest.raises(ValueError, match="grouped"):
        live_session().context(grouping="grouped")


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


# ------------------------------------------------------------ live recipes
def two_colour_block(frame, n=400):
    """Two dyes: one puts 80 % of its photons in channel 1, the other 20 %."""
    rng = np.random.default_rng(frame)
    dye = rng.integers(0, 2, n)
    total = rng.uniform(2000, 6000, n)
    share = np.where(dye == 0, 0.8, 0.2)
    first = rng.poisson(total * share).astype(float)
    second = rng.poisson(total * (1 - share)).astype(float)
    return Localizations({"x_nm": rng.uniform(0, 5000, n),
                          "y_nm": rng.uniform(0, 5000, n),
                          "frame": np.full(n, frame, np.int64),
                          "photons": first + second,
                          "photons_ch1": first, "photons_ch2": second,
                          "dye": dye}, {})


def test_an_expression_follows_the_live_table_block_by_block():
    import dataclasses
    from smappy import plugins
    session = live_session()
    cls = plugins.get("Analysis/Process/Math Parser")
    settings = dataclasses.replace(cls.Settings(), field="bright",
                                   expression="photons > 1000")
    plugin = cls()
    assert plugin.live_recipe_for(settings)
    session.append(block(4))                       # after the snapshot
    session.apply(plugin, plugin.run(session.context(), settings))
    session.append(block(5))
    assert session.live and len(session.locs) == 250
    np.testing.assert_array_equal(np.asarray(session.locs["bright"]) > 0.5,
                                  np.asarray(session.locs["photons"]) > 1000)
    assert session.history[-1]["what"] == "Analysis/Process/Math Parser"


def test_a_reduction_is_recomputed_over_the_whole_table_as_it_grows():
    import dataclasses
    from smappy import plugins
    session = live_session()
    cls = plugins.get("Analysis/Process/Math Parser")
    settings = dataclasses.replace(cls.Settings(), field="centred",
                                   expression="photons - median(photons)")
    session.apply(cls(), cls().run(session.context(), settings))
    session.layers[0].set_bound("centred", 0, None)
    session.append(block(4))
    photons = np.asarray(session.locs["photons"])
    np.testing.assert_allclose(session.locs["centred"],
                               photons - np.median(photons), rtol=1e-5, atol=1e-2)
    # the filter on the column follows its new values too
    assert session.layers[0].filter.mask.sum() == (session.locs["centred"] >= 0).sum()


def test_an_expression_over_the_selection_cannot_follow_a_live_table():
    import dataclasses
    from smappy import plugins
    cls = plugins.get("Analysis/Process/Math Parser")
    settings = dataclasses.replace(cls.Settings(), where="selection")
    assert not cls().live_recipe_for(settings)
    shift = dataclasses.replace(cls.Settings(), field="x_nm", expression="y_nm")
    assert not cls().live_recipe_for(shift)        # a position: not a recipe


def test_colours_follow_the_live_table_and_are_found_again_when_it_is_finished():
    from smappy.plugins.assign_colors import AssignColors
    session = Session()
    session.begin_live((0, 5000, 0, 5000))
    for frame in range(1, 4):
        session.append(two_colour_block(frame))
    plugin = AssignColors()
    settings = plugin.Settings()
    session.apply(plugin, plugin.run(session.context(), settings))
    session.append(two_colour_block(4))
    channel = np.asarray(session.locs["channel"])
    dye = np.asarray(session.locs["dye"])
    new = slice(1200, 1600)                       # the block after the run
    decided = channel[new] > 0
    assert decided.mean() > 0.8
    # one colour per dye, whichever number it got
    for d in (0, 1):
        assert len(set(channel[new][decided & (dye[new] == d)])) == 1

    finished = session.locs.snapshot().compact()
    for name in ("channel", "color_ratio"):
        finished.columns.pop(name)
    session.apply(Shift(), Result(locs=finished, text="fit"), finishes_live=True)
    assert not session.live and "channel" in session.locs
    assert session.history[-2]["what"] == "live recipes"
