"""Plugins: how any plugin is found, set up, previewed, run and kept.

Every analysis in SMAPpy is a plugin, and they all work the same way, so
this is the one place the way is shown: the Analysis tab and its find box,
the settings form, Preview, *live*, Run and Text, a plugin in a window of
its own, and what a run leaves behind (undo, and the History log).

Line Profile is the example because it has everything -- a preview, *live*,
a fit to look at -- and because a line across one of the simulated lines has
a width the fit should find.  What the measuring plugins measure, and how
to read it, is the measuring tutorial's.
"""
from __future__ import annotations

from .common import field, open_section, place_beside, show_tab, tab_button, toolbar

TITLE = "Plugins: running, previewing, plotting"
DESCRIPTION = ("How every plugin works: finding it, its settings, Preview and "
               "live, Run, and what a run leaves behind.")

# a line of the simulated structure (`simulate.structure`): from (1000, 1500)
# to (9000, 7500) nm, so a line ROI across it is along (-0.6, 0.8)
_ALONG = (0.8, 0.6)
_ACROSS = (-0.6, 0.8)


def _across(at: float, half_length: float = 450.0, width: float = 250.0):
    """A line ROI across the simulated line, ``at`` of the way along it."""
    from ...regions import Region
    cx, cy = 1000 + 8000 * at, 1500 + 6000 * at
    p0 = (cx - _ACROSS[0] * half_length, cy - _ACROSS[1] * half_length)
    p1 = (cx + _ACROSS[0] * half_length, cy + _ACROSS[1] * half_length)
    return Region.line(p0, p1, width), (cx, cy)


def make(d) -> None:
    session, control, render = d.session, d.control, d.render

    # the tour's simulated table, made the way it shows (File -> Simulate)
    file_tab = show_tab(d, "File")
    _, simulate = open_section(d, file_tab, "Blinking Structure")
    simulate.run_button.click()
    d.settle()

    d.chapter("What a plugin is")
    d.card(TITLE,
           "<p>Every analysis in SMAPpy is a plugin: a form of settings and a "
           "Run button. They all work the same way, and this is that way.</p>"
           "<p>A plugin measures the selection: what passes the current layer's "
           "filter, inside the ROI if one is drawn.</p>",
           say="Every analysis in SMAPpy is a plugin, and they all work the "
               "same way. This tutorial shows that way.")

    d.chapter("Finding one")
    analysis = show_tab(d, "Analysis")
    titles = [s.button for s in analysis.sections]
    d.shot("The Analysis tab lists its plugins as sections, by what they do: "
           "drift, measuring, processing.",
           spot=[d.union(*titles)], point=tab_button(d, "Analysis"), click=True,
           zoom=d.around(analysis.sections[0], 760))
    analysis.search.setText("profile")
    d.settle()
    d.shot("Type in find to narrow the list.",
           spot=[analysis.search], point=analysis.search,
           zoom=d.around(analysis.search, 760))
    analysis.search.setText("")
    d.settle()
    first = analysis.sections[0]
    d.shot("Right-click a title to rename it, duplicate it with other settings, "
           "or remove it; + pins another plugin to the tab.",
           spot=[first.button], point=first.button,
           zoom=d.around(first.button, 760))

    d.chapter("Settings")
    section, panel = open_section(d, analysis, "Line Profile")
    d.shot("Opening a section shows its settings. Hold the mouse over a label "
           "to see what it does; more holds the rest.",
           spot=[panel.form], zoom=d.around(panel.form, 760))

    d.chapter("Preview and live")
    for name, value in (("axis", "along"), ("model", "gauss")):
        f = field(panel, name)
        f.set(value)
        f.changed.emit()
    # well clear of the ring, which crosses this line near 0.3 and 0.7
    roi, centre = _across(0.15)
    toolbar(d)._set_kind("line", "line")
    session.set_roi(roi)
    render.view.frame_on((centre[0] - 700, centre[0] + 700), (centre[1] - 500, centre[1] + 500))
    d.settle()
    d.shot("Draw a line across the structure. The profile along it is the "
           "structure's cross-section, here fitted with a Gaussian.",
           spot=[d.data_rect(centre[0] - 450, centre[1] - 450,
                             centre[0] + 450, centre[1] + 450, pad=4),
                 field(panel, "axis"), field(panel, "model")],
           point=d.at_data(*centre))
    d.shot("Preview does the work and draws it, without changing anything.",
           spot=[panel.preview_button], point=panel.preview_button, click=True,
           zoom=d.around(panel.preview_button, 760))
    panel.preview_button.click()
    d.settle()
    window = panel._window
    # the simulated line scatters by 15 nm (`simulate.structure`): a fit far
    # from that is a line drawn the wrong way, not a width
    sigma = panel.result.data["values"]["gauss"]["sigma"]
    assert 5 < sigma < 40, f"the profile's width is {sigma:.1f} nm"
    place_beside(d, window, 0.75)
    d.shot("The profile and its fit: the width of the line, with its "
           "uncertainty.",
           spot=[d.window_rect(window)])
    if panel.live is not None:
        panel.live.setChecked(True)
        d.settle()
        window.hide()
        d.shot("With live ticked, the figure follows the line while it is "
               "moved: a measurement you can aim.",
               spot=[panel.live], point=panel.live, click=True,
               zoom=d.around(panel.live, 760))
        roi, centre = _across(0.85)
        session.set_roi(roi)
        render.view.frame_on((centre[0] - 700, centre[0] + 700),
                             (centre[1] - 500, centre[1] + 500))
        d.pump(0.3)
        d.settle()
        window.show()
        place_beside(d, window, 0.75)
        d.shot("Moved further along, the same line measured again, at once.",
               spot=[d.window_rect(window)])
        window.hide()
        panel.live.setChecked(False)

    d.chapter("Run and Text")
    panel.run_button.click()
    d.settle()
    d.shot("Run does the same and keeps the result: the numbers appear below, "
           "Text opens them in full, Plot draws the figures again.",
           spot=[panel.output, panel.text_button, panel.plot_button],
           point=panel.run_button, click=True, zoom=d.around(panel.output, 760))

    d.chapter("Several at once")
    if section.detach_button is not None:
        d.shot("The arrow on a title opens the plugin in a window of its own, "
               "so several can stay open side by side.",
               spot=[section.detach_button], point=section.detach_button,
               click=True, zoom=d.around(section.detach_button, 760))
        section.detach_button.click()
        d.settle()
        floating = section._window
        place_beside(d, floating, 0.45)
        d.shot("The same plugin, with the same settings, now beside the picture. "
               "Close it to put it back in the tab.",
               spot=[d.window_rect(floating)])
        floating.close()
        d.settle()

    d.chapter("What is kept")
    history_section, history = open_section(d, analysis, "History")
    history.run_button.click()
    d.settle()
    text = getattr(history, "_text_window", None)
    if text is not None and text.isVisible():
        place_beside(d, text, 0.6)
    d.shot("A plugin that changes the table, like a drift correction, can be "
           "undone, and is written into the file's history. History shows it.",
           spot=[d.window_rect(text)] if text is not None and text.isVisible()
           else [history.output],
           point=history.run_button)
    if text is not None:
        text.hide()

    d.chapter("Next")
    d.card("What to remember",
           "<ul><li>Find a plugin in its tab, with find, or in the Plugins "
           "menu.</li>"
           "<li><b>Preview</b> shows the work without changing anything; "
           "<b>live</b> follows the ROI.</li>"
           "<li><b>Run</b> keeps the result; <b>Text</b> and <b>Plot</b> show "
           "it again.</li>"
           "<li>What changes the table can be undone, and is kept in the "
           "history.</li></ul>",
           say="That is how every plugin works. Next: the measuring plugins, and "
               "how to read what they find.")
