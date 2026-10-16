"""A column with no value for some rows -- NaN -- is left out of what is
computed from it, not allowed to spoil it.

A MINFLUX `cfr` is measured in one iteration, for every localization in some
sequences and once per trace in others; the rest have none.  Merging blinks
averages over the localizations that have one, colouring by it draws the rest
grey, and the colour range is over the values there are.
"""
import numpy as np

from smappy import lut
from smappy.group import by_trace, combine
from smappy.locs import Localizations
from smappy.render import FieldOfView, RenderSettings, finite_range, render_locs


def traces():
    return Localizations({
        "x_nm": np.array([10, 11, 12, 50, 51, 90, 91], np.float32),
        "y_nm": np.array([10, 11, 12, 50, 51, 90, 91], np.float32),
        "frame": np.arange(7),
        "xy_err_nm": np.full(7, 5.0, np.float32),
        "tid": np.array([1, 1, 1, 2, 2, 3, 3]),
        "cfr": np.array([0.3, np.nan, np.nan, 0.2, 0.6, np.nan, np.nan], np.float32),
        "efo_err": np.array([2.0, np.nan, 2.0, 1.0, 1.0, np.nan, np.nan], np.float32),
    }, {"units": "nm"})


def test_merging_a_trace_averages_the_values_it_has_and_none_is_none():
    locs = traces()
    merged = combine(locs, by_trace(locs["tid"]))
    assert np.allclose(merged["cfr"][:2], [0.3, 0.4])          # its one; the mean of two
    assert np.isnan(merged["cfr"][2])                            # none measured
    # a precision pools the ones there are: 1/sqrt(1/4 + 1/4)
    assert np.allclose(merged["efo_err"][:2], [np.sqrt(2), 1 / np.sqrt(2)])
    assert np.isnan(merged["efo_err"][2])
    assert np.allclose(merged["x_nm"], [11, 50.5, 90.5])        # columns without NaN as before


def test_colouring_by_a_column_draws_a_missing_value_grey_and_ranges_over_the_rest():
    values = np.array([0.0, np.nan, 1.0, np.inf])
    colours = lut.colors(values, "hot", 0.0, 1.0)
    table = lut.get("hot")
    assert np.allclose(colours[0], table[0]) and np.allclose(colours[2], table[-1])
    assert np.allclose(colours[[1, 3]], lut.NO_VALUE)
    assert finite_range(values) == (0.0, 1.0) and finite_range([np.nan]) == (0.0, 1.0)
    locs = traces()
    image = render_locs(locs, FieldOfView(0, 0, 10.0, 10, 10),
                        RenderSettings(color_field="cfr"))
    assert np.isfinite(np.asarray(image.rgb if hasattr(image, "rgb") else image.weight)).all()
