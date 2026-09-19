"""Removing, or hiding, the localizations inside or outside the ROI."""
import numpy as np
import pytest

from smappy.group import GroupSettings, group
from smappy.locs import Localizations
from smappy.mathparse import recipes
from smappy.plugins import Context, Selection
from smappy.regions import Region
from smappy.plugins.remove_locs import (FILTER_BOUND, RemoveLocalizations,
                                        RemoveSettings, check_field, hide)


def table(n=20):
    """Twenty localizations in a row, 0 to 1900 nm apart."""
    return Localizations({
        "x_nm": np.arange(n, dtype=np.float32) * 100,
        "y_nm": np.zeros(n, np.float32),
        "frame": np.arange(n, dtype=np.int64),
        "photons": np.full(n, 500.0, np.float32),
        "loc_precision_nm": np.full(n, 10.0, np.float32),
    }, {"units": "nm"})


def context(locs, roi=None, mask=None):
    """A context with a ROI, as a session would build one."""
    if mask is None:
        mask = (roi.mask(locs["x_nm"], locs["y_nm"]) if roi is not None
                else np.ones(len(locs), bool))
    ctx = Context(locs=locs, selection=Selection(mask, roi=roi))
    ctx.session = None
    return ctx


HALF = Region.rect(-50, -50, 950, 50)          # the first ten localizations


# ------------------------------------------------------------- what goes

def test_removing_inside_the_roi_keeps_everything_else():
    locs = table()
    result = RemoveLocalizations().run(context(locs, HALF), RemoveSettings())
    assert len(result.locs) == 10
    assert np.allclose(result.locs["x_nm"], np.arange(10, 20) * 100)
    assert "removed 10 of 20" in result.text


def test_removing_outside_the_roi_keeps_only_what_is_in_it():
    locs = table()
    result = RemoveLocalizations().run(context(locs, HALF),
                                       RemoveSettings(which="outside"))
    assert np.allclose(result.locs["x_nm"], np.arange(10) * 100)


def test_the_selection_is_the_region_when_it_is_asked_for():
    """The filter and the ROI together: SMAP's "keep visible inside ROI"."""
    locs = table()
    bright = np.zeros(len(locs), bool)
    bright[:5] = True                       # a filter that keeps five of the ten
    ctx = context(locs, HALF, mask=bright & HALF.mask(locs["x_nm"], locs["y_nm"]))
    result = RemoveLocalizations().run(
        ctx, RemoveSettings(which="outside", region="selection"))
    assert len(result.locs) == 5


def test_a_run_that_would_change_nothing_or_everything_is_refused():
    locs = table()
    away = Region.rect(1e6, 1e6, 2e6, 2e6)
    with pytest.raises(ValueError, match="nothing is inside"):
        RemoveLocalizations().run(context(locs, away), RemoveSettings())
    with pytest.raises(ValueError, match="empty table"):
        RemoveLocalizations().run(context(locs, away),
                                  RemoveSettings(which="outside"))


def test_without_a_roi_it_says_so_rather_than_removing_everything():
    with pytest.raises(ValueError, match="no ROI"):
        RemoveLocalizations().run(context(table()), RemoveSettings())


def test_the_preview_counts_without_touching_the_table():
    locs = table()
    result = RemoveLocalizations().preview(context(locs, HALF), RemoveSettings())
    assert result.locs is None and "would remove 10 of 20" in result.text
    assert len(locs) == 20


# ------------------------------------------------------------ what hides

def test_hiding_keeps_the_rows_and_flags_them():
    locs = table()
    result = RemoveLocalizations().run(context(locs, HALF),
                                       RemoveSettings(action="hide"))
    assert len(result.locs) == 20                      # nothing was removed
    assert np.array_equal(np.asarray(result.locs["use"]),
                          np.repeat([0.0, 1.0], 10))
    # and the filter is opened on it, which is what makes them disappear
    assert result.data["bounds"] == {"use": FILTER_BOUND}


def test_hiding_twice_hides_both_regions():
    """Clicking around a field of view adds up; it does not start over."""
    locs = table()
    plugin, settings = RemoveLocalizations(), RemoveSettings(action="hide")
    once = plugin.run(context(locs, HALF), settings).locs
    other = Region.rect(1750, -50, 1950, 50)           # the last two
    twice = plugin.run(context(once, other), settings).locs
    assert np.array_equal(np.asarray(twice["use"]),
                          np.array([0] * 10 + [1] * 8 + [0] * 2, float))


def test_a_hidden_blink_is_one_whose_localizations_were_all_visible():
    """Grouping must not bring back half of what was hidden."""
    locs = Localizations({
        "x_nm": np.array([0, 0, 0, 1000, 1000, 1000], np.float32),
        "y_nm": np.zeros(6, np.float32),
        "frame": np.array([0, 1, 2, 0, 1, 2], np.int64),
        "photons": np.full(6, 500.0, np.float32),
    }, {"units": "nm"})
    keep = np.array([True, False, True, True, True, True])
    flagged = hide(locs, keep)
    assert [r["field"] for r in recipes(flagged)] == ["use"]
    grouped, _ = group(flagged, GroupSettings(dx=50.0, dt=1))
    assert np.array_equal(np.asarray(grouped["use"]), [0.0, 1.0])


def test_a_name_that_could_not_be_a_column_is_refused():
    assert check_field(" use ") == "use"
    with pytest.raises(ValueError, match="not usable"):
        check_field("2 things")
    with pytest.raises(ValueError, match="grouping"):
        check_field("n_in_group")


# ---------------------------------------------------------- in a session

def test_the_session_opens_the_filter_on_the_flag_and_undoes_a_removal():
    from smappy.session import Session

    session = Session(table())
    session.set_roi(HALF)
    session.run(RemoveLocalizations(), RemoveSettings(action="hide"),
                layer=0)
    assert session.layers[0].filter.ranges["use"] == FILTER_BOUND
    assert int(session.layers[0].filter.mask.sum()) == 10

    session.run(RemoveLocalizations(), RemoveSettings())
    assert len(session.locs) == 10
    session.undo()
    assert len(session.locs) == 20


def test_what_was_removed_is_in_the_log_with_its_settings():
    from smappy.session import Session

    session = Session(table())
    session.set_roi(HALF)
    session.run(RemoveLocalizations(), RemoveSettings(region="selection",
                                                      which="outside"))
    entry = session.history[-1]
    assert entry["what"] == "Analysis/Process/Remove Localizations"
    assert entry["settings"]["which"] == "outside"
