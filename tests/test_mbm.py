import numpy as np
import pytest

from smappy.io.formats import FileInfo
from smappy.io.msr import bead_table
from smappy.locs import Localizations
from smappy.mbm import MBMSettings, with_overrides
from smappy.plugins import Context
from smappy.plugins.drift_mbm import MBMDrift
from smappy.simulate import simulate_bead_tracks

T = np.arange(10.0, 200.0, 2.0)
DRIFT = np.column_stack([10 * np.sin(T / 30), 0.05 * T, 3 * np.cos(T / 50)])
NAMES = ["R0", "R1", "R2", "R3", "R4"]


def measurement(label="260101-120000_sim", did="d1", jump=(2, 120.0, 25.0), seed=0):
    """A MINFLUX measurement as the reader leaves it: the localizations
    corrected by Imspector's mean over five beads, one of which jumps, and
    the beads' records in the metadata.  Also the true positions."""
    beads = simulate_bead_tracks(T, DRIFT, n_beads=5, jumps=[jump] if jump else [],
                                 precision_nm=1.0, seed=seed)
    gri = 61 + 62 * (beads["tid"] - 1)
    meta = {"label": label, "did": did, "dimensionality": 3, "z_scale": 1.0,
            "mbm": {"used": NAMES,
                    "points": {"gri": gri.tolist(), "x_nm": beads["x_nm"].tolist(),
                               "y_nm": beads["y_nm"].tolist(),
                               "z_nm": beads["z_nm"].tolist(),
                               "time_s": beads["time_s"].tolist(),
                               "strength": beads["photons"].tolist()},
                    "beads": [{"name": n, "gri": 61 + 62 * k, "used": True,
                               "sigpos_nm": 1.0} for k, n in enumerate(NAMES)]}}
    rng = np.random.default_rng(seed + 1)
    n = 3000
    t = np.sort(rng.uniform(T[0], T[-1], n))
    true = rng.uniform(0, 2000, (n, 3))
    drift = np.column_stack([np.interp(t, T, DRIFT[:, a]) for a in range(3)])
    raw = true + drift + rng.normal(0, 0.3, (n, 3))
    # Imspector: each bead's displacement from its first point, averaged
    shift = []
    for k in range(1, 6):
        rows = beads["tid"] == k
        p = np.column_stack([beads[f"{a}_nm"][rows] for a in "xyz"])
        shift.append(np.column_stack([np.interp(t, beads["time_s"][rows], p[:, a] - p[0, a])
                                      for a in range(3)]))
    imspector = np.mean(shift, axis=0)
    columns = {"time_s": t, "frame": np.arange(n), "tid": np.arange(n) // 5}
    for a, axis in enumerate("xyz"):
        columns[f"{axis}_nm"] = raw[:, a] - imspector[:, a]
        columns[f"{axis}_nc_nm"] = raw[:, a]
    locs = Localizations(columns, {"minflux": {"measurements": [meta]}})
    return locs, meta, true


def scatter_from_truth(locs, true):
    moved = np.column_stack([locs[f"{a}_nm"] for a in "xyz"]) - true
    return np.std(moved, axis=0)


def test_the_bead_that_jumps_is_left_out_and_the_correction_made_again_from_scratch():
    locs, _, true = measurement()
    plugin = MBMDrift()
    result = plugin.run(Context(locs=locs), MBMSettings())
    assert "5 beads: 4 used, 1 left out automatically" in result.text
    before = scatter_from_truth(locs, true)
    after = scatter_from_truth(result.locs, true)
    # Imspector's mean carries a fifth of the 25 nm jump, a 5 nm step, into
    # every localization: 1.8 nm rms per axis over the measurement
    assert before[:2].min() > 1.5 and after.max() < 0.6
    assert np.array_equal(result.locs["x_nc_nm"], locs["x_nc_nm"])
    assert result.data["drift"].times_s is not None
    # from scratch: run again, the same table
    again = plugin.run(Context(locs=result.locs), MBMSettings())
    assert np.allclose(again.locs["x_nm"], result.locs["x_nm"])


def test_a_bead_chosen_by_hand_stays_chosen_and_only_its_measurement_is_touched():
    locs, _, true = measurement()
    plugin = MBMDrift()
    ctx = Context(locs=locs)
    view = plugin.bead_view(ctx, MBMSettings())
    assert view.beads.names == NAMES and view.columns["Imspector"] == ["used"] * 5
    assert not view.beads.used[2] and "jump" in view.beads.reasons[2]
    settings = plugin.with_overrides(MBMSettings(beads="9:R0 out"), view, {"R2": True})
    assert settings.beads == "9:R0 out, R2 in"            # another measurement's kept
    view = plugin.bead_view(ctx, settings)
    assert view.beads.used.all() and view.beads.state(2) == "in, by hand"
    forced = plugin.run(ctx, settings)
    assert scatter_from_truth(forced.locs, true)[:2].min() > 1.5
    assert with_overrides(settings, NAMES, {}).beads == "9:R0 out"


def test_imspectors_choice_is_where_the_automatic_one_starts():
    locs, meta, _ = measurement(jump=None)
    meta["mbm"]["beads"][4]["used"] = False
    view = MBMDrift().bead_view(Context(locs=locs), MBMSettings())
    assert list(view.beads.used) == [True, True, True, True, False]
    view = MBMDrift().bead_view(Context(locs=locs), MBMSettings(start="all beads"))
    assert view.beads.used.all()


def test_the_bead_file_is_corrected_with_its_measurement():
    from smappy.session import Session
    locs, meta, _ = measurement()
    beads = bead_table([meta])
    session = Session()
    session.add_file(locs, FileInfo("m", "/data/m.msr", "Abberior MSR",
                                    metadata={"minflux": {"measurements": [meta]}}))
    session.add_file(beads, FileInfo("m (MBM beads)", "/data/m.msr", "Abberior MSR",
                                     metadata={"mbm_of": {"path": "/data/m.msr",
                                                          "measurements": ["d1"]},
                                               "mbm_beads": beads.metadata["mbm_beads"]}),
                     append=True)
    ctx = session.context(grouping="ungrouped")
    mask = np.asarray(session.locs["filenumber"]) == 0
    ctx = Context(locs=ctx.locs, session=session, selection=ctx.selection.__class__(mask))
    result = MBMDrift().run(ctx, MBMSettings())
    out = result.locs
    good = (np.asarray(out["filenumber"]) == 1) & (out["tid"] == 1)
    # a good bead, corrected, is still: what is left is its noise
    assert np.std(out["x_nm"][good]) < 1.5 < np.std(session.locs["x_nm"][good])


def test_the_notice_counts_the_beads_and_the_window_writes_a_choice_into_the_form():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from smappy.gui.bead_window import BeadWindow
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.session import Session
    app = QApplication.instance() or QApplication([])
    locs, meta, _ = measurement()
    session = Session()
    session.add_file(locs, FileInfo("m", "/data/m.msr", "Abberior MSR",
                                    metadata={"minflux": {"measurements": [meta]}}))
    opened = []

    def opener(settings=None, plugin=None, reply=None):
        opened.append(BeadWindow(session, plugin, settings, reply))
    opener.replies = True
    session.window_openers["bead selector"] = opener
    panel = PluginPanel(MBMDrift, session)
    panel._describe()
    assert "5 beads: 4 used, 1 left out automatically" in panel.notice.text()
    assert not panel.notice_button.isHidden()
    panel.notice_button.click()
    window = opened[0]
    assert window.table.rowCount() == 5
    assert window.table.item(2, 1).text() == "no" and "jump" in window.table.item(2, 9).text()
    # click R2's track in the figure: it is selected, and drawn over the others
    line = next(l for ax in window.figure.axes for l in ax.lines if l.get_gid() == "2")
    window._picked(type("Pick", (), {"artist": line})())
    assert window.selected == 2 and window.table.currentRow() == 2
    window.include.click()
    assert panel.form.value().beads == "R2 in"
    assert window.table.item(2, 1).text() == "yes (by hand)"
    assert "chosen by hand" in panel.notice.text()
    window.automatic.click()
    assert panel.form.value().beads == ""
    window.close()
    app.processEvents()
