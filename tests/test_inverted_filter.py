"""An inverted bound: the layer shows what that one bound throws away,
under every other bound as it is."""
import numpy as np
import pytest

from smappy.filter import LocFilter
from smappy.locs import Localizations
from smappy.session import Session


def _table(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(-400, 400, n)
    z[::50] = np.nan                       # localizations without a z
    return Localizations({
        "x_nm": rng.uniform(0, 1e4, n), "y_nm": rng.uniform(0, 1e4, n),
        "z_nm": z, "xy_err_nm": rng.gamma(4, 3, n),
        "photons": rng.uniform(100, 2000, n),
        "frame": np.arange(n), "filenumber": np.repeat([0, 1], n // 2),
    }, {"units": "nm"})


def test_an_inverted_range_keeps_exactly_what_the_range_throws_away_nan_included():
    locs = _table()
    kept = LocFilter(locs, z_nm=(-200, 200)).mask
    turned = LocFilter(locs).set("z_nm", -200, 200, invert=True).mask
    assert np.array_equal(turned, ~kept)
    assert turned[np.isnan(locs["z_nm"])].all()


def test_the_other_ranges_keep_their_sense_while_one_is_inverted():
    locs = _table()
    f = LocFilter(locs, photons=(500, None))
    f.set("xy_err_nm", None, 10, invert=True)
    expected = (locs["photons"] >= 500) & ~(locs["xy_err_nm"] <= 10)
    assert np.array_equal(f.mask, expected)
    assert f.inverted == {"xy_err_nm"}
    f.set("xy_err_nm", None, 10)           # turned back
    assert np.array_equal(f.mask, (locs["photons"] >= 500) & (locs["xy_err_nm"] <= 10))
    assert not f.inverted


def test_an_inverted_range_stays_inside_the_spans_and_extends_to_new_rows():
    locs = _table()
    f = LocFilter(locs, spans=[(0, 1000)])
    f.set("xy_err_nm", None, 10, invert=True)
    assert not f.mask[1000:].any()
    assert np.array_equal(f.mask[:1000], ~(locs["xy_err_nm"][:1000] <= 10))

    grow = Localizations({k: locs[k][:1500] for k in locs.keys()}, {})
    g = LocFilter(grow).set("xy_err_nm", None, 10, invert=True)
    grow.extend(Localizations({k: locs[k][1500:] for k in locs.keys()}, {}))
    g.append(500)
    assert np.array_equal(g.mask, ~(locs["xy_err_nm"] <= 10))


def test_a_layer_keeps_its_inversion_through_a_new_bound_and_drops_it_with_the_bound():
    session = Session(_table())
    layer = session.layers[0]
    layer.set_ranges({"photons": (500, None), "xy_err_nm": (None, 10)})
    locs = layer.locs
    layer.invert_bound("xy_err_nm", True)
    assert np.array_equal(layer.filter.mask,
                          (locs["photons"] >= 500) & ~(locs["xy_err_nm"] <= 10))
    layer.set_bound("xy_err_nm", None, 12)          # a drag of the histogram
    assert layer.inverted == {"xy_err_nm"}
    assert np.array_equal(layer.filter.mask,
                          (locs["photons"] >= 500) & ~(locs["xy_err_nm"] <= 12))
    layer.invert_bound("photons", False)
    layer.invert_bound("frame", True)               # no bound: nothing to invert
    assert layer.inverted == {"xy_err_nm"}
    layer.remove_bound("xy_err_nm")
    assert not layer.inverted
    layer.set_bound("xy_err_nm", None, 12)
    assert not layer.inverted


def test_the_grouped_table_and_plugins_see_the_inverted_bound_too():
    session = Session(_table())
    layer = session.layers[0]
    layer.set_ranges({"xy_err_nm": (None, 10)}, inverted=["xy_err_nm"])
    ungrouped = session.tables.ungrouped.locs
    assert np.array_equal(layer.selection().mask, ~(ungrouped["xy_err_nm"] <= 10))
    layer.show_grouped(True)
    grouped = layer.locs
    assert np.array_equal(layer.filter.mask, ~(grouped["xy_err_nm"] <= 10))


def test_a_copied_layer_a_scratch_session_and_a_handoff_carry_the_inversion():
    from smappy import handoff
    session = Session(_table())
    layer = session.layers[0]
    layer.set_files(None)
    layer.set_ranges({"photons": (500, None), "xy_err_nm": (None, 10)},
                     inverted=["xy_err_nm"])
    added = session.add_layer(like=0)
    assert added.inverted == {"xy_err_nm"}
    assert session.scratch().layers[0].inverted == {"xy_err_nm"}
    assert session.layer_configs()[0]["inverted"] == ["xy_err_nm"]
    other = Session(_table())
    handoff.apply_layers(other, handoff.layer_states(session))
    assert other.layers[0].inverted == {"xy_err_nm"}
    assert np.array_equal(other.layers[0].filter.mask, layer.filter.mask)
    # a config without inversions is the whole filter: none
    other.set_layer_configs([{"bounds": {"xy_err_nm": [None, 10]}}])
    assert not other.layers[0].inverted


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_invert_tick_turns_the_shown_bound_around_and_marks_it_red(app):
    from smappy.gui.render_tab import INVERTED_COLOUR, FilterWidget
    session = Session(_table())
    layer = session.layers[0]
    layer.set_files(None)
    layer.set_ranges({"xy_err_nm": (None, 10)})
    widget = FilterWidget(session)
    widget.bind(layer)
    widget._select("photons")
    widget._show_field()
    assert not widget.invert.isEnabled()           # no bound on photons
    widget._select("xy_err_nm")
    widget._show_field()
    assert widget.invert.isEnabled() and not widget.invert.isChecked()
    widget.invert.setChecked(True)
    assert layer.inverted == {"xy_err_nm"}
    assert len(layer.filter) == int((~(layer.locs["xy_err_nm"] <= 10)).sum())
    assert INVERTED_COLOUR in widget.quick["xy_err_nm"].styleSheet()
    assert "not" in widget.bounds_line.text()
    widget.clear.click()                           # the bound goes, and its inversion
    assert not layer.inverted and not widget.invert.isChecked()
    assert INVERTED_COLOUR not in widget.quick["xy_err_nm"].styleSheet()
