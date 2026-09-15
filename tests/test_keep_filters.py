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
