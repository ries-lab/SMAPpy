"""Drift correction: seeing drift, and correcting it with RCC or COMET.

The simulation drifts on request (File -> Simulate, "add drift": a random
walk plus a slow creep of about a hundred nanometres) and records the drift it
added, so both corrections are checked against the truth here, not only
shown.  Drift is made visible the way the tour shows it -- the first half of
the acquisition in red, the second in green -- because a blur is easy to miss
and two colours that do not overlap are not.

RCC first, since it is the fast and robust one; COMET on the same data after
an undo, since what it does differently is the point.  COMET's question
before a long run (with RCC first as the answer that saves the afternoon) is
said rather than shown: on this simulation the run is seconds and it does not
ask.
"""
from __future__ import annotations

import numpy as np

from .common import (expand, field, menu, menu_item, open_section, place_beside,
                     show_tab, tab_button, type_into)

TITLE = "Drift correction: RCC and COMET"
DESCRIPTION = ("Seeing drift, correcting it with RCC or COMET, and checking the "
               "result.")

_RING = (6400, 8200), (3600, 5400)


def _halves(d, tab) -> None:
    """Layer 1 the first half of the frames in red, layer 2 the second in green."""
    session = d.session
    half = int(session.locs["frame"].max()) // 2
    filt = tab.filter
    filt.quick["frame"].click()
    filt.lo.setText("")
    filt.hi.setText(str(half))
    filt.lo.editingFinished.emit()
    filt.hi.editingFinished.emit()
    tab.lut.setCurrentText("red")
    tab.strip.add.menu().actions()[0].trigger()          # + -> localizations
    d.settle()
    filt.quick["frame"].click()
    filt.lo.setText(str(half + 1))
    filt.hi.setText("")
    filt.lo.editingFinished.emit()
    filt.hi.editingFinished.emit()
    tab.lut.setCurrentText("green")
    d.settle()


def _all_frames(d, tab) -> None:
    """Back to one layer showing every frame: a drift plugin estimates from what
    the current layer shows, and half the frames is half the drift."""
    d.session.remove_layer(1)
    tab.strip.select(0)
    tab.filter.quick["frame"].click()
    tab.filter.clear.click()
    tab.lut.setCurrentText("hot")
    d.settle()


def _error_nm(result, truth) -> float:
    """RMS difference between an estimated drift and the simulated one, per axis,
    both taken relative to their own mean -- a drift is known up to a shift."""
    drift = result.data["drift"]
    frames = np.asarray(drift.frames).astype(int)
    found = np.asarray(drift.drift)[:, :2]
    true = np.asarray(truth)[frames][:, :2]
    found = found - found.mean(0)
    true = true - true.mean(0)
    return float(np.sqrt(((found - true) ** 2).mean(0)).max())


def make(d) -> None:
    session, render = d.session, d.render

    d.chapter("Drift")
    d.card(TITLE,
           "<p>Over the minutes of an acquisition the sample moves, by tens to "
           "hundreds of nanometres. Each localization is off by where the "
           "sample was in its frame, and the picture blurs.</p>"
           "<p>Drift is estimated from the localizations themselves, then "
           "subtracted: no beads needed.</p>",
           say="Over an acquisition the sample drifts, and the picture blurs. "
               "SMAPpy estimates the drift from the data and takes it out.")

    file_tab = show_tab(d, "File")
    _, simulate = open_section(d, file_tab, "Blinking Structure")
    type_into(simulate, "drift", True)
    d.settle()
    d.shot("To practise, Simulate adds drift on request, and remembers how much, "
           "so the correction can be checked.",
           spot=[field(simulate, "drift")], point=field(simulate, "drift"),
           click=True, zoom=d.around(field(simulate, "drift"), 760))
    simulate.run_button.click()
    d.settle()
    truth = session.locs.metadata.get("drift_truth")
    assert truth, "the simulation recorded no drift"

    d.chapter("Seeing it")
    tab = show_tab(d, "Render")
    _halves(d, tab)
    render.view.frame_on(*_RING)
    d.settle()
    d.shot("The first half of the acquisition in red, the second in green. "
           "Without drift they would overlap in yellow; here they do not.",
           spot=[d.rect(render.view.graphics, pad=-2), tab.strip],
           point=d.at_data(7480, 4700))

    d.chapter("Two methods")
    d.card("RCC and COMET",
           "<p><b>RCC</b> renders the data in a few time windows and "
           "cross-correlates them. It is fast and robust, and limited by "
           "its rendering.</p>"
           "<p><b>COMET</b> fits the drift to the localizations themselves, "
           "finer and slower. Before a long run it says how long, and offers "
           "RCC first.</p>",
           say="Two methods: RCC correlates rendered time windows, fast and "
               "robust. COMET fits the localizations themselves, finer.")

    d.chapter("RCC")
    _all_frames(d, tab)
    d.shot("A drift plugin estimates from what the current layer shows, so give "
           "it all the frames back first: one layer, no frame filter.",
           spot=[tab.strip, d.rect(tab.filter)], zoom=d.around(tab.filter, 760))
    analysis = show_tab(d, "Analysis")
    _, rcc = open_section(d, analysis, "RCC")
    d.shot("RCC's main settings: how many time windows, the pixel of the "
           "rendering, and how far the sample may have moved.",
           spot=[rcc.form], point=tab_button(d, "Analysis"), click=True,
           zoom=d.around(rcc.form, 760))
    rcc.run_button.click()
    d.settle()
    error = _error_nm(rcc.result, truth)
    assert error < 15, f"RCC is {error:.0f} nm from the simulated drift"
    rcc.plot()
    d.settle()
    place_beside(d, rcc._window, 0.7)
    d.shot("The drift it found, in x, y and z over the acquisition. It is "
           "subtracted from every localization.",
           spot=[d.window_rect(rcc._window)])
    rcc._window.hide()
    tab = show_tab(d, "Render")
    _halves(d, tab)
    render.view.frame_on(*_RING)
    d.settle()
    d.shot("Red and green now overlap in yellow: the two halves agree again.",
           spot=[d.rect(render.view.graphics, pad=-2)], point=d.at_data(7480, 4700))

    d.chapter("COMET")
    file_menu = menu(d, "File")
    d.shot("To try COMET on the same data, undo the correction first: File, "
           "Undo, or Ctrl+Z.",
           spot=[menu_item(d, file_menu, "Undo")], point=menu_item(d, file_menu, "Undo"),
           click=True, zoom=d.around(file_menu, 700))
    file_menu.close()
    session.undo()
    _all_frames(d, tab)
    analysis = show_tab(d, "Analysis")
    _, comet = open_section(d, analysis, "COMET")
    d.shot("COMET's settings: the window size, how far the sample may have "
           "moved, and a spline that smooths the drift between windows.",
           spot=[comet.form], zoom=d.around(comet.form, 760))
    first = field(comet, "rcc_prepass")
    expand(first)
    d.shot("On a large dataset COMET says how long it will take before it "
           "starts, and offers RCC first: seconds for the bulk of the drift.",
           spot=[first], point=first, zoom=d.around(first, 760))
    comet.run_button.click()
    d.settle()
    error = _error_nm(comet.result, truth)
    assert error < 15, f"COMET is {error:.0f} nm from the simulated drift"
    comet.plot()
    d.settle()
    place_beside(d, comet._window, 0.7)
    d.shot("COMET's estimate of the same drift. On real data it is often the "
           "finer of the two; RCC first is a good way to start a long run.",
           spot=[d.window_rect(comet._window)])
    comet._window.hide()

    d.chapter("What is kept")
    d.shot("The drift curve is saved with the file, so Plot draws it again after "
           "the file is reopened, and the correction can be undone.",
           spot=[comet.plot_button], point=comet.plot_button,
           zoom=d.around(comet.plot_button, 760))

    d.chapter("Next")
    d.card("What to remember",
           "<ul><li>See drift with two layers, early and late frames, in two "
           "colours.</li>"
           "<li><b>RCC</b>: fast and robust. <b>COMET</b>: finer, and it warns "
           "before a long run.</li>"
           "<li>Correct drift before measuring; undo brings the old table "
           "back.</li></ul>",
           say="That is drift correction: see it in two colours, correct it with "
               "RCC or COMET, and measure after.")
