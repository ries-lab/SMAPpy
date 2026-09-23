"""What chains and batch runs stand on: settings read back from the log, the
grouped table handed out on request, a fit opening as a file, a loader's log
entry surviving, and figures written without a window."""
from dataclasses import asdict, dataclass, field

import numpy as np
import pytest

from smappy.io.hdf5 import save_localizations
from smappy.locs import Localizations
from smappy.plugins import (Context, Plot, Plugin, Result, settings_from,
                            settings_values)
from smappy.session import Session


def blinks(n_emitters=40, on=4, seed=1):
    """Emitters far apart, each on for ``on`` consecutive frames: linking with
    SMAP's 50 nm / 1 frame gives exactly one group per emitter."""
    rng = np.random.default_rng(seed)
    x = np.repeat(np.arange(n_emitters) * 500.0, on) + rng.normal(0, 3, n_emitters * on)
    y = np.repeat(np.full(n_emitters, 1000.0), on) + rng.normal(0, 3, n_emitters * on)
    frame = (np.repeat(np.arange(n_emitters) * 10, on)
             + np.tile(np.arange(on), n_emitters)).astype(np.int64)
    precision = np.repeat(np.linspace(5, 40, n_emitters), on)
    return Localizations({"x_nm": x, "y_nm": y, "frame": frame,
                          "photons": np.full(len(x), 1000.0),
                          "loc_precision_nm": precision})


@dataclass
class Inner:
    roisize: int = 13
    method: str = "fast"


@dataclass
class Outer:
    radius: float = 50.0
    fit: Inner = field(default_factory=Inner)


def test_settings_logged_as_nested_dictionaries_read_back_as_they_were():
    settings = Outer(radius=12.5, fit=Inner(roisize=7, method="slow"))
    assert settings_from(Outer, asdict(settings)) == settings
    assert settings_from(Outer, settings_values(settings)) == settings
    # a dotted key beside the nested block wins, as the more specific spelling
    mixed = {"fit": {"roisize": 7}, "fit.roisize": 9}
    assert settings_from(Outer, mixed).fit.roisize == 9


def test_a_plugin_has_a_version_and_no_grouping_requirement_unless_it_says():
    class Plain(Plugin):
        pass

    class Picky(Plugin):
        version = "3"
        grouping = "grouped"

    assert Plain.version == "1" and Plain.grouping is None
    assert Picky.version == "3" and Picky.grouping == "grouped"


def test_the_grouped_table_is_handed_out_with_its_own_filter():
    locs = blinks()
    session = Session(locs)
    session.layers[0].show_grouped(False)
    session.layers[0].set_bound("loc_precision_nm", None, 20.0)

    ungrouped, sel = session.context().table()
    assert ungrouped is session.locs and len(sel) == np.sum(locs["loc_precision_nm"] <= 20)

    grouped, gsel = session.context(grouping="grouped").table()
    assert len(grouped) == 40
    assert len(gsel) == np.sum(grouped["loc_precision_nm"] <= 20)
    # what the layer shows is not changed by a plugin asking
    assert not session.layers[0].grouped


def test_a_layer_that_shows_grouped_hands_out_grouped_unless_overridden():
    session = Session(blinks())
    assert session.layers[0].grouped
    assert len(session.context().table()[0]) == 40
    assert len(session.context(grouping="ungrouped").table()[0]) == 160


def test_a_grouped_table_without_a_session_is_refused_with_a_reason():
    ctx = Context(locs=blinks(), grouping="grouped")
    with pytest.raises(ValueError, match="needs a session"):
        ctx.table()
    locs, sel = Context(locs=blinks()).table()
    assert len(locs) == 160 and len(sel) == 160


class _MadeFromNothing(Plugin):
    path = "Localize/Test"

    def run(self, ctx, settings):
        return Result(locs=blinks(), text="fitted", data={"path": "/tmp/x_locs.hdf5"})


def test_a_fit_in_an_empty_session_opens_as_a_file_with_its_log():
    session = Session()
    session.run(_MadeFromNothing())
    assert len(session.locs) == 160 and len(session.files) == 1
    assert session.files[0].name == "x_locs.hdf5"
    assert not session.undo_stack.can_undo
    assert [e["what"] for e in session.history][-1] == "Localize/Test"


def test_a_loaders_log_entry_survives_the_file_it_opens(tmp_path):
    from smappy.plugins import get
    path = tmp_path / "t.h5"
    save_localizations(path, blinks(), {"history": [{"what": "earlier", "text": ""}]})
    session = Session()
    loader = get("File/Load/Auto")()
    settings = loader.Settings(path=str(path))
    session.run(loader, settings)
    whats = [e["what"] for e in session.history]
    assert whats[0] == "earlier"                # the file's own log continues
    assert "File/Load/Auto" in whats            # and the load is in it


def test_figures_are_written_without_a_window(tmp_path):
    from smappy.figures import save, summary_plot

    def line(ax):
        ax.plot([0, 1], [0, 1])

    def two(figure):
        a, b = figure.subplots(1, 2)
        a.plot([1, 2]); b.plot([2, 1])

    plots = [Plot(line, name="drift: x(t)"), Plot(two, name="pair", panels=2),
             Plot(line, name="drift: x(t)")]
    written = save(plots + [summary_plot(plots)], tmp_path, prefix="rcc")
    names = sorted(p.name for p in written)
    assert names == ["rcc_all.png", "rcc_drift_x_t.png", "rcc_drift_x_t_2.png",
                     "rcc_pair.png"]
    assert all(p.stat().st_size > 1000 for p in written)
