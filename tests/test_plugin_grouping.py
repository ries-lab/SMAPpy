"""Plugins follow the layer's grouping (simplification plan B3).

A measuring plugin is handed the table the layer shows -- blinks, on a
grouped layer; one that must have the localizations says so with
``Plugin.grouping = "ungrouped"`` and gets them whatever the layer shows.
"""
import numpy as np
import pytest

from smappy import plugins
from smappy.locs import Localizations
from smappy.plugins import Plugin, Result
from smappy.session import Session
from smappy.simulate import simulate


def a_session():
    session = Session(simulate(n_frames=800, seed=3))
    assert session.layers[0].grouped           # as a file opens
    return session


def spy_on(cls):
    """``cls`` with its run replaced by one that keeps the context."""
    seen = {}

    class Spy(cls):
        def run(self, ctx, settings):
            seen["ctx"] = ctx
            return Result(text="looked")
    return Spy, seen


class Echo(Plugin):
    """A measurement that declares nothing."""
    path = "Test/Echo"

    def run(self, ctx, settings):
        return Result(text=f"{len(ctx.selection)} of {len(ctx.locs)}",
                      data={"ctx": ctx})


def test_a_plugin_on_a_grouped_layer_measures_the_grouped_table():
    session = a_session()
    layer = session.layers[0]
    layer.set_bound("photons", 2000.0, None)
    ctx = session.run(Echo()).data["ctx"]
    grouped = session.tables.grouped().locs
    assert ctx.locs is grouped and ctx.served_grouped
    # the layer's bound, on the blinks' summed photons
    np.testing.assert_array_equal(ctx.selection.mask, layer.filter.mask)
    assert (np.asarray(grouped["photons"])[ctx.selection.mask] >= 2000).all()
    # and the localizations are still there for a plugin that writes onto them
    whole, selection = ctx.ungrouped()
    assert whole is session.locs and len(selection.mask) == len(whole)

    session.show_grouped(0, False)
    ctx = session.run(Echo()).data["ctx"]
    assert ctx.locs is session.locs and not ctx.served_grouped


@pytest.mark.parametrize("path", ["Analysis/Drift/COMET", "Analysis/Drift/RCC",
                                  "Analysis/Measure/Localization Precision",
                                  "Analysis/Process/Remove Localizations"])
def test_drift_precision_and_removal_get_the_ungrouped_table_on_a_grouped_layer(path):
    session = a_session()
    cls = plugins.get(path)
    assert cls.grouping == "ungrouped"
    Spy, seen = spy_on(cls)
    session.run(Spy(), cls.Settings())
    ctx = seen["ctx"]
    assert ctx.locs is session.locs and not ctx.served_grouped
    assert len(ctx.selection.mask) == len(session.locs)


def test_line_profile_measures_what_each_layer_shows():
    from smappy.plugins.line_profile import LineProfileSettings, _groups
    from smappy.regions import Region
    session = a_session()
    session.add_layer(like=0)
    session.show_grouped(1, False)
    session.set_roi(Region.line((2000, 5000), (8000, 5000), 2000.0))
    ctx = session.context()
    found = _groups(ctx, LineProfileSettings(source="layers"))
    grouped, ungrouped = found[0][2], found[1][2]
    # the same ROI and bounds: blinks on the first, localizations on the second
    assert 0 < len(grouped) < len(ungrouped)


def test_a_new_table_made_from_blinks_is_refused_and_the_session_kept():
    class Halve(Plugin):
        path = "Test/Halve"            # written to the old rule: no grouping

        def run(self, ctx, settings):
            return Result(locs=ctx.locs[:len(ctx.locs) // 2])

    session = a_session()
    before = session.locs
    with pytest.raises(ValueError, match="grouping = \"ungrouped\""):
        session.run(Halve())
    assert session.locs is before and not session.can_undo


def test_a_loader_on_a_grouped_layer_is_not_refused_its_new_table():
    class Loads(Plugin):
        path = "File/Load/Test"            # makes a table of its own

        def run(self, ctx, settings):
            return Result(locs=simulate(n_frames=100, seed=9))

    session = a_session()
    session.run(Loads())
    assert len(session.locs) == len(simulate(n_frames=100, seed=9))


def test_dbscan_on_blinks_still_writes_onto_every_localization():
    session = a_session()
    cls = plugins.get("Analysis/Regions/Find/DBSCAN")
    session.run(cls(), cls.Settings(radius_nm=60.0, neighbours=5, use_z=False))
    assert "cluster_id" in session.locs and "(blinks)" in session.history[-1]["text"]
    assert len(session.locs["cluster_id"]) == len(session.locs)
    # a blink's localizations share its cluster
    gid = np.asarray(session.locs["group_id"])
    cid = np.asarray(session.locs["cluster_id"])
    for g in np.unique(gid)[:200]:
        assert len(np.unique(cid[gid == g])) == 1


def test_every_shipped_plugin_that_changes_the_table_reads_the_localizations():
    """A table change from the grouped table is refused, so a plugin that
    makes one either asks for the ungrouped table or maps blinks back
    itself -- DBSCAN, through `Context.ungrouped`."""
    maps_back = {"Analysis/Regions/Find/DBSCAN"}
    for path, ref in plugins.refs().items():
        cls = plugins.get(path)
        takes_input = not path.startswith(plugins.SOURCES)
        if cls.changes_locs and takes_input and path not in maps_back:
            assert cls.grouping == "ungrouped", path
