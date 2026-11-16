"""MINFLUX: where its tools are, and a first analysis of motor traces.

For a reader who knows what MINFLUX is but not every word of abberior's:
the words come on a card and again where they are used.  The tools are
not on the shipped tabs (MINFLUX is a specialist's path), so the + of a
tab pins them, as a reader will.

On a simulated `.msr` (`simulate.minflux.motor_msr`): two measurements of
kinesin-like motors stepping 8 nm, one at saturating ATP and one where ATP
is scarce, tracked by the same 2D sequence, a tenth of the attempts
catching two molecules at once.  The storyboard checks what the words say:
the steps come out near 8 nm in both, the dwell is longer and its Erlang
shape lower at low ATP.
"""
from __future__ import annotations

import time

from .common import expand, field, open_section, pin, place_beside, show_tab

TITLE = "MINFLUX: opening, checking and analysing traces"
DESCRIPTION = ("Where SMAPpy's MINFLUX tools are, the words abberior uses, and a "
               "first analysis: the sequence, its quality, traces, steps, and two "
               "measurements compared.")

_FILE = "kinesin.msr"
_PATTERN = r"(?P<ATP_uM>\d+)uM"


def _data(d):
    from ...simulate.minflux import motor_msr
    return motor_msr(d.data_dir() / _FILE)


def _wait_for(d, done, what: str, timeout: float = 120.0) -> None:
    end = time.monotonic() + timeout
    while not done() and time.monotonic() < end:
        d.pump(0.1)
    assert done(), f"{what} did not finish"
    d.settle()


def _quietly_pin(d, plugin_tab, path: str, title: str) -> None:
    """Pin a plugin as `pin` does, without showing the + again."""
    if any(s.title.lower().startswith(title.lower()) for s in plugin_tab.sections):
        return
    from ...workspace import Instance
    plugin_tab.tab.instances.append(Instance(plugin=path))
    plugin_tab.rebuild()
    d.settle()


def _open(d, plugin_tab, path: str, title: str):
    _quietly_pin(d, plugin_tab, path, title)
    for s in plugin_tab.sections:            # one open at a time: the column stays short
        if s.button.isCheckable() and s.button.isChecked() \
                and not s.title.lower().startswith(title.lower()):
            s.button.click()
    return open_section(d, plugin_tab, title)[1]


def _set(panel, name: str, value) -> None:
    f = field(panel, name)
    expand(f)
    f.set(value)
    f.changed.emit()


def _run(d, panel, plot: bool = True, fraction: float = 0.85):
    panel.run_button.click()
    d.settle()
    assert panel.result is not None, f"{panel.plugin.name} gave no result"
    if not plot:
        return None
    panel.plot()
    d.settle()
    place_beside(d, panel._window, fraction)
    return panel._window


def _panels(d, window):
    """The figure's panels as {title: desktop rectangle}."""
    from ...gui.figures import FigurePane
    pane = [p for p in window.findChildren(FigurePane) if p.isVisible()][0]
    x, y, w, h = d.rect(pane.canvas, pad=0)
    out = {}
    for ax in pane.figure.axes:
        if not ax.get_visible():
            continue
        b = ax.get_position()
        x0, x1 = max(0.0, b.x0 - 0.04), min(1.0, b.x1 + 0.02)
        y0, y1 = max(0.0, b.y0 - 0.06), min(1.0, b.y1 + 0.05)
        out[ax.get_title()] = (x + x0 * w, y + (1 - y1) * h, (x1 - x0) * w, (y1 - y0) * h)
    return out


def _titled(panels, start: str):
    for title, rect in panels.items():
        if title.startswith(start):
            return rect
    raise LookupError(f"no panel {start!r}; there are " + ", ".join(panels))


def _busiest(locs, size: float = 1500.0):
    """A square of ``size`` nm around the longest trace: a few motors'
    tracks, large enough to see them walk."""
    import numpy as np
    tid = np.asarray(locs["tid"])
    final = np.asarray(locs["final"]) == 1 if "final" in locs else np.ones(len(tid), bool)
    ids, counts = np.unique(tid[final & (tid >= 0)], return_counts=True)
    rows = final & (tid == ids[np.argmax(counts)])
    cx = float(np.mean(np.asarray(locs["x_nm"])[rows]))
    cy = float(np.mean(np.asarray(locs["y_nm"])[rows]))
    return cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2


def _words():
    rows = (("iteration", "one step of the search: the beam pattern shrinks around "
                          "the molecule, and the last iteration gives the localizations"),
            ("sequence", "the iterations and their settings, what the instrument was told"),
            ("trace, tid", "the localizations of one molecule while it was caught; "
                           "<code>tid</code> numbers them"),
            ("efo", "the photon rate while localizing: twice as high with two molecules"),
            ("cfr", "how much the centre of the pattern still sees: high when a second "
                    "molecule is near"),
            ("thread", "a sequence of its own in the same measurement, often a second "
                       "colour"),
            ("MBM", "beads tracked during the measurement, for the drift"))
    return "".join(f"<p><b>{a}</b>: {b}</p>" for a, b in rows)


def make(d) -> None:
    session, control, render = d.session, d.control, d.render
    path = _data(d)

    d.chapter("MINFLUX in SMAPpy")
    d.card(TITLE,
           "<p><b>File, Open</b> reads abberior's <code>.msr</code>: each "
           "measurement becomes a file, with its sequence.</p>"
           "<p><b>Analysis, MINFLUX</b>: what only MINFLUX data has, the "
           "sequence, its quality, two colours, the beads.</p>"
           "<p><b>Analysis, Tracking</b>: traces, from any method: their "
           "statistics, their steps, one trace up close.</p>"
           "<p><b>Regions</b>: the numbers per trace, compared between "
           "measurements.</p>",
           say="MINFLUX files open from the File menu. The tools are in three "
               "places: MINFLUX for what only MINFLUX has, Tracking for traces, "
               "and Regions to compare.")
    d.card("Words abberior uses", _words(),
           say="A few of abberior's words, which the plugins use too: "
               "iteration, sequence, trace, efo, cfr, thread and MBM.")

    d.chapter("Opening an .msr")
    from PySide6.QtWidgets import QDialog
    from ...gui.msr_import import MsrImportDialog
    dialog = MsrImportDialog(path, control)
    dialog.show()
    d.settle()
    d.place(dialog, 80, 90)
    d.settle()
    d.shot("Opening an .msr asks which parts to read. Here two measurements of "
           "motors, one at high and one at low ATP.",
           spot=[dialog.tree], zoom=d.around(dialog, 1000))
    d.shot("Beside it, the selected measurement's sequence: what the instrument "
           "did in each iteration.",
           spot=[dialog.details], zoom=d.around(dialog, 1000))
    dialog.every.setChecked(True)
    d.settle()
    d.shot("Every iteration reads every step of every attempt, not only the "
           "localizations. Sequence QC needs it; the layers still show only the "
           "localizations.",
           spot=[dialog.every], point=dialog.every, click=True,
           zoom=d.around(dialog.every, 760))
    dialog.close()
    d.settle()

    def accept(self):
        self.every.setChecked(True)
        return QDialog.Accepted

    kept = MsrImportDialog.exec
    MsrImportDialog.exec = accept
    try:
        control.load_paths([str(path)], reset_view=True)
        _wait_for(d, lambda: len(session.files) == 2 and len(session.locs) > 0,
                  "opening the measurements")
    finally:
        MsrImportDialog.exec = kept

    tab = show_tab(d, "Render")
    d.settle()
    filt = tab.filter
    final = filt.bounds_line
    assert "final" in final.text(), final.text()
    d.shot("Each measurement is a file. With every iteration read, the layer "
           "keeps the localizations: final is 1.",
           spot=[final, d.rect(render.view.graphics, pad=-2)], zoom=d.around(filt, 760))
    i = filt.field.findText("filenumber")
    assert i >= 0, "no filenumber in the field list"
    filt.field.setCurrentIndex(i)
    d.settle()
    filt.all_files.click()
    d.settle()
    assert session.layers[0].files is None or len(session.layers[0].files) == 2
    d.shot("A layer starts on the first file. Choose filenumber and tick all "
           "to see both measurements.",
           spot=[filt.field, filt.files, filt.all_files], point=filt.all_files,
           click=True, zoom=d.around(filt, 760))
    assert tab.grouped.isChecked()
    d.shot("Merge blinks draws each trace as one point: its localizations "
           "averaged.",
           spot=[tab.grouped, d.rect(render.view.graphics, pad=-2)], point=tab.grouped)
    tab.grouped.click()
    d.settle()
    x0, x1, y0, y1 = _busiest(session.locs)
    render.view.frame_on((x0, x1), (y0, y1))
    d.settle_picture()
    d.shot("Unticked, every localization is drawn, and the traces show the "
           "motors walking.",
           spot=[tab.grouped, d.rect(render.view.graphics, pad=-2)], point=tab.grouped,
           click=True)

    d.chapter("The sequence")
    analysis = show_tab(d, "Analysis")
    pin(d, analysis, "Analysis/MINFLUX/Sequence table", "Sequence table")
    table = _open(d, analysis, "Analysis/MINFLUX/Sequence table", "Sequence table")
    _run(d, table, plot=False)
    assert "differ" not in (table.result.text or "").split("\n")[0], \
        "the two sequences should be the same"
    table.show_text()
    d.settle()
    text = table._text_window
    place_beside(d, text, 0.95)
    d.shot("Sequence table lists each measurement's sequence, an iteration a "
           "line, and first what differs between them. Here nothing: the "
           "difference is in the sample.",
           spot=[d.window_rect(text)])
    text.hide()
    conditions = _open(d, analysis, "Analysis/Process/File conditions", "File conditions")
    _set(conditions, "pattern", _PATTERN)
    d.settle()
    d.shot("File conditions writes that difference onto the localizations, here "
           "the ATP concentration read from the names.",
           spot=[field(conditions, "pattern")], point=field(conditions, "pattern"),
           zoom=d.around(field(conditions, "pattern"), 760))
    _run(d, conditions, plot=False)
    assert "ATP_uM" in session.locs, "no ATP_uM attribute"

    d.chapter("How the sequence ran")
    qc = _open(d, analysis, "Analysis/MINFLUX/Sequence QC", "Sequence QC")
    window = _run(d, qc, fraction=0.9)
    panels = _panels(d, window)
    d.shot("Sequence QC reads every iteration: how many attempts passed each "
           "one, and why the others ended.",
           spot=[_titled(panels, "attempts per iteration")])
    d.shot("Most attempts end in the first iteration, where nothing is there "
           "yet. Once caught, a molecule mostly stays.",
           spot=[_titled(panels, "attempts per iteration")],
           zoom=d.around(_titled(panels, "attempts per iteration"), 900))
    d.shot("efo against cfr: one molecule is a cloud at low cfr; two molecules "
           "at once sit higher on both.",
           spot=[_titled(panels, "efo against cfr")],
           zoom=d.around(_titled(panels, "efo against cfr"), 900))
    window.hide()

    d.chapter("Quality")
    quality = _open(d, analysis, "Analysis/MINFLUX/Quality filter", "Quality filter")
    window = _run(d, quality, fraction=0.7)
    bounds = quality.result.data["bounds"]
    assert "trace_cfr" in bounds and "efo" in bounds, bounds
    d.shot("Quality filter finds the cuts in the data: efo below the valley "
           "before the two-molecule peak, and cfr below the single molecules' "
           "spread.",
           spot=[d.window_rect(window)])
    window.hide()
    d.shot("cfr is measured once, at the start of a trace, so it is cut per "
           "trace: a trace is kept or dropped whole.",
           spot=[quality.output], zoom=d.around(quality.output, 900))
    show_tab(d, "Render")
    d.settle()
    d.shot("The cuts are ordinary filter bounds on the layer: they can be "
           "moved or cleared like any other.",
           spot=[filt.bounds_line], zoom=d.around(filt, 760))

    d.chapter("Traces")
    analysis = show_tab(d, "Analysis")
    stats = _open(d, analysis, "Analysis/Tracking/Trace statistics", "Trace statistics")
    window = _run(d, stats)
    d.shot("Trace statistics measures every trace at once: its speed, its "
           "shape, and its precision from consecutive localizations.",
           spot=[d.window_rect(window)])
    window.hide()
    d.shot("Each trace's numbers are kept as a table per tid, for filtering "
           "and comparing traces.",
           spot=[stats.output], zoom=d.around(stats.output, 900))
    inspector = _open(d, analysis, "Analysis/Tracking/Trace inspector", "Trace inspector")
    window = _run(d, inspector)
    d.shot("Trace inspector shows one trace up close: the longest, or the "
           "one you choose by its tid.",
           spot=[d.window_rect(window)])
    window.hide()

    d.chapter("Steps")
    steps = _open(d, analysis, "Analysis/Tracking/Find steps", "Find steps")
    _set(steps, "step_nm", 8.0)
    d.settle()
    d.shot("Find steps fits steps to each trace along its own axis. With the "
           "expected size set, steps that are too small are merged.",
           spot=[field(steps, "step_nm")], point=field(steps, "step_nm"),
           zoom=d.around(field(steps, "step_nm"), 760))
    window = _run(d, steps)
    panels = _panels(d, window)
    pooled = steps.result.data.get("pooled") or {}
    if pooled.get("step_mean_nm") is not None:
        assert abs(pooled["step_mean_nm"] - 8.0) < 1.0, pooled["step_mean_nm"]
    d.shot("The positions along the track, and the steps found in them.",
           spot=[_titled(panels, "the longest traces")],
           zoom=d.around(_titled(panels, "the longest traces"), 900))
    d.shot("The step sizes pooled over the traces, and the dwell times between "
           "steps, with three models compared.",
           spot=[d.window_rect(window)])
    window.hide()

    d.chapter("Comparing the measurements")
    compare = _open(d, analysis, "Analysis/Regions/Summarize/Compare groups",
                    "Compare groups")
    _set(compare, "evaluation", "per tid: Find steps")
    _set(compare, "group_by", "attribute:ATP_uM")
    _set(compare, "fields", "step_median_nm, dwell_mean_s, erlang_shape, velocity_nm_per_s")
    d.settle()
    d.shot("Compare groups compares what was measured per trace, grouped by "
           "file or by a condition: here ATP.",
           spot=[field(compare, "group_by")], point=field(compare, "group_by"),
           zoom=d.around(field(compare, "group_by"), 760))
    window = _run(d, compare)
    _check_atp(compare.result)
    d.shot("The steps are 8 nm in both. With little ATP the motor waits longer, "
           "and one event, ATP binding, sets the pace: the Erlang shape drops "
           "towards 1.",
           spot=[d.window_rect(window)])
    window.hide()

    d.chapter("Next")
    d.card("Also there",
           "<ul><li><b>Pair colours</b>: two threads' traces paired, and the "
           "offset between the colours.</li>"
           "<li><b>MBM drift correction</b> and <b>MBM registration</b>: the "
           "beads, to correct drift and to put measurements on top of each "
           "other.</li>"
           "<li><b>Stability</b>: how still the instrument was, from beads.</li>"
           "<li><b>Filter regions</b>: keep the traces whose numbers pass a "
           "condition.</li></ul>"
           "<p>Each plugin's <b>?</b> opens its page.</p>",
           say="There is more: two colours, the beads for drift, the "
               "instrument's stability, and filtering traces by their numbers. "
               "Each plugin's question mark explains it.")


def _check_atp(result) -> None:
    """The words say the steps agree and the dwell and shape differ: if the
    data stop showing it, the words are wrong."""
    median = result.data["table"]["median"]
    groups = [str(g) for g in result.data["groups"]]
    assert set(groups) == {"20", "1000"}, groups
    low = {f: median[f][g] for f in median for g in result.data["groups"] if str(g) == "20"}
    high = {f: median[f][g] for f in median for g in result.data["groups"] if str(g) == "1000"}
    assert abs(low["step_median_nm"] - high["step_median_nm"]) < 1.0, (low, high)
    assert low["dwell_mean_s"] > 1.5 * high["dwell_mean_s"], (low, high)
    assert low["erlang_shape"] < high["erlang_shape"], (low, high)
