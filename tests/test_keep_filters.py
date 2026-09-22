"""A correction keeps the filters the user set.

Drift correction and colour assignment hand back the same localizations with a
column changed or added, and `Session.set_locs` rebinds the layers rather than
starting over -- so the bounds have to survive.  They did on the ungrouped
table and not on the grouped one, which is the one on screen by default, so
running either plugin looked like it had thrown the filters away.
"""
import numpy as np

from smappy.locs import Localizations
from smappy.plugins import Plugin, Result
from smappy.session import Session


def table(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({
        "x_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "frame": rng.integers(0, 200, n).astype(np.int64),
        "phot": rng.uniform(100, 5000, n).astype(np.float32),
        "loc_precision_nm": rng.uniform(5, 40, n).astype(np.float32),
        "logl_rel": rng.uniform(-3, 0, n).astype(np.float32),
    }, {})


class Corrector(Plugin):
    """Stands in for a drift correction: the same rows, x and y moved."""
    name = "corrector"
    path = "Test/Corrector"

    def run(self, ctx, settings=None) -> Result:
        locs = ctx.session.locs
        columns = {name: np.asarray(locs[name]).copy() for name in locs}
        columns["x_nm"] = columns["x_nm"] + 12.0
        return Result(locs=Localizations(columns, {}), text="moved")


def test_a_correction_keeps_the_bounds_on_both_tables():
    session = Session(table())
    layer = session.layers[0]
    assert layer.grouped                       # the grouped table is what is drawn
    layer.set_bound("phot", 500, None)
    before = layer.filter.mask.sum()
    assert 0 < before < len(layer.locs)        # the bound does something

    session.run(Corrector())

    layer = session.layers[0]
    assert layer.grouped
    for which in ("ungrouped", "grouped"):
        ranges = layer.state.sets[which].filter.ranges
        assert ranges.get("phot") == (500, None), which
        assert ranges.get("loc_precision_nm") == (None, 25.0), which
    assert layer.filter.mask.sum() == before   # and the same rows are kept


def test_a_correction_keeps_the_bounds_on_a_second_layer_too():
    """The second layer inherits the first's grouped *table* but has a filter
    of its own, so nothing is linked for it and the bounds have to be put on
    that filter explicitly -- which is what did not happen."""
    session = Session(table())
    session.add_layer(like=0)
    for i, layer in enumerate(session.layers):
        layer.set_bound("phot", 500 + 100 * i, None)
        if not layer.grouped:
            layer.show_grouped(True, session.layers[0] if i else None)
    before = [layer.filter.mask.sum() for layer in session.layers]
    assert before[0] != before[1]              # the two layers differ to begin with

    session.run(Corrector())

    for i, layer in enumerate(session.layers):
        assert layer.grouped
        for which in ("ungrouped", "grouped"):
            assert layer.state.sets[which].filter.ranges.get("phot") == (500 + 100 * i, None), \
                f"layer {i}, {which}"
        assert layer.filter.mask.sum() == before[i]


def test_a_layer_that_takes_over_a_grouped_table_takes_the_files_too():
    """Switching a layer to grouped when another layer has the table already
    reuses that table -- and its filter is empty, files included."""
    locs = table()
    locs.columns["filenumber"] = (np.arange(len(locs)) % 2).astype(np.int64)
    session = Session(locs)
    session.add_layer(like=0)
    session.layers[1].set_files([1])
    session.layers[1].show_grouped(False)
    session.show_grouped(1, True)
    grouped = session.layers[1].state.sets["grouped"]
    assert "files" in grouped.filter
    assert set(np.asarray(grouped.locs["filenumber"])[grouped.filter.mask]) == {1}
