"""Finding your way around: the two windows, the tabs, filters, layers,
grouping, the selection, plugins and undo.

The second of the series and the first one made, because everything after it
assumes it.  Aimed at a biologist who knows what SMLM is: each idea gets one
slide of reminder (what grouping does, what a plugin measures) and then is
shown working, on the simulated ring-and-crossing-lines so that anyone can
follow along with File -> Simulate.  Nothing is fitted here -- that is the
fitting tutorial's job.
"""
from __future__ import annotations

TITLE = "Finding your way around SMAPpy"
DESCRIPTION = ("The two windows, the tabs, filters, layers, grouping, what a "
               "plugin works on, and undo.")

# ---------------------------------------------------------------- figures
# Drawn for the cards; `currentColor` and the player's CSS variables, so they
# follow its light and dark themes.

_GROUPING = """
<svg viewBox="0 0 520 190" role="img" aria-label="three localizations of one blink, in consecutive frames, become one">
  <g font-size="13" fill="currentColor">
    <text x="10" y="18" opacity=".7">frames</text>
    <g>
      <rect x="10" y="30" width="60" height="60" rx="6" class="panel"/>
      <rect x="80" y="30" width="60" height="60" rx="6" class="panel"/>
      <rect x="150" y="30" width="60" height="60" rx="6" class="panel"/>
      <rect x="220" y="30" width="60" height="60" rx="6" class="panel"/>
      <circle cx="42" cy="58" r="5" class="dot"/>
      <circle cx="109" cy="62" r="5" class="dot"/>
      <circle cx="181" cy="57" r="5" class="dot"/>
      <text x="40" y="108" text-anchor="middle">1</text>
      <text x="110" y="108" text-anchor="middle">2</text>
      <text x="180" y="108" text-anchor="middle">3</text>
      <text x="250" y="108" text-anchor="middle">4</text>
      <text x="250" y="65" text-anchor="middle" opacity=".6">off</text>
    </g>
    <path d="M300 60 h60" class="arrow" marker-end="url(#head)"/>
    <text x="330" y="48" text-anchor="middle" opacity=".7">grouped</text>
    <rect x="380" y="30" width="60" height="60" rx="6" class="panel"/>
    <circle cx="410" cy="59" r="7" class="dot strong"/>
    <text x="455" y="54">1 localization</text>
    <text x="455" y="72" opacity=".7">3 frames on</text>
    <text x="10" y="150" opacity=".85">One fluorophore, on for three frames: three localizations, a few nm apart.</text>
    <text x="10" y="172" opacity=".85">Grouped, they are one, with the photons added and a better precision.</text>
  </g>
  <defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>
</svg>
"""

_LAYERS = """
<svg viewBox="0 0 520 170" role="img" aria-label="two layers of the same localizations, added into one picture">
  <g font-size="13" fill="currentColor">
    <rect x="10" y="20" width="120" height="100" rx="8" fill="#000"/>
    <circle cx="70" cy="70" r="34" fill="none" stroke="#ff4040" stroke-width="4"/>
    <text x="70" y="140" text-anchor="middle">layer 1: frames, first half</text>
    <text x="155" y="76" font-size="26" text-anchor="middle">+</text>
    <rect x="180" y="20" width="120" height="100" rx="8" fill="#000"/>
    <circle cx="240" cy="70" r="34" fill="none" stroke="#40ff40" stroke-width="4"/>
    <text x="240" y="140" text-anchor="middle">layer 2: second half</text>
    <text x="325" y="76" font-size="26" text-anchor="middle">=</text>
    <rect x="350" y="20" width="120" height="100" rx="8" fill="#000"/>
    <circle cx="410" cy="70" r="34" fill="none" stroke="#ffff40" stroke-width="4"/>
    <text x="410" y="140" text-anchor="middle">the picture</text>
  </g>
</svg>
"""

_SELECTION = """
<svg viewBox="0 0 520 190" role="img" aria-label="the selection is what passes the layer's filter inside the ROI">
  <g font-size="13" fill="currentColor">
    <rect x="10" y="10" width="300" height="160" rx="8" class="panel"/>
    <g class="faint">
      <circle cx="40" cy="40" r="3"/><circle cx="80" cy="130" r="3"/><circle cx="130" cy="60" r="3"/>
      <circle cx="260" cy="40" r="3"/><circle cx="280" cy="140" r="3"/><circle cx="60" cy="90" r="3"/>
      <circle cx="170" cy="110" r="3"/><circle cx="215" cy="60" r="3"/><circle cx="235" cy="120" r="3"/>
    </g>
    <g class="dot">
      <circle cx="50" cy="60" r="4"/><circle cx="100" cy="150" r="4"/><circle cx="280" cy="80" r="4"/>
      <circle cx="160" cy="80" r="4"/><circle cx="190" cy="95" r="4"/><circle cx="175" cy="125" r="4"/>
      <circle cx="210" cy="115" r="4"/><circle cx="200" cy="70" r="4"/>
    </g>
    <rect x="140" y="55" width="100" height="85" class="roi"/>
    <text x="330" y="40"><tspan class="faint-text">o</tspan> filtered out</text>
    <text x="330" y="66"><tspan class="dot-text">o</tspan> passes the filter</text>
    <text x="330" y="92"><tspan class="roi-text">[ ]</tspan> the ROI</text>
    <text x="330" y="130" font-weight="bold">selection =</text>
    <text x="330" y="150">filter AND ROI,</text>
    <text x="330" y="168">for the current layer</text>
  </g>
</svg>
"""


from .common import menu as _menu, menu_item as _menu_item, open_section as _open
from .common import show_tab as _show_tab, tab_button as _tab_button, toolbar as _toolbar


# ---------------------------------------------------------------- the story

def make(d) -> None:
    session, control, render = d.session, d.control, d.render
    toolbar = _toolbar(d)

    d.chapter("Two windows")
    d.card(TITLE,
           "<p>The two windows, the tabs, and the four ideas the rest of SMAPpy "
           "is built on: <b>filters</b>, <b>layers</b>, <b>grouping</b> and "
           "<b>the selection</b>.</p>"
           "<p>Everything here uses simulated data, so you can follow along: "
           "<i>File &rarr; Simulate</i>.</p>",
           say="Welcome. This tutorial is a tour of SMAPpy: where things are, "
               "and four ideas that everything else is built on.")
    d.shot("SMAPpy opens two windows. On the left, the control window, with "
           "every setting and tool.",
           spot=[d.window_rect(control)])
    d.shot("On the right, the render window: the super-resolution image itself.",
           spot=[d.window_rect(render)])

    d.chapter("Getting data in")
    menu = _menu(d, "File")
    d.shot("Open a file from the File menu, or with Ctrl+O. SMAPpy reads "
           "localizations from the common fitting programs.",
           spot=[_menu_item(d, menu, "Open...")], point=_menu_item(d, menu, "Open..."),
           zoom=d.around(menu, 700))
    d.shot("Add file... puts a second file into the same table, for instance "
           "the next field of view.",
           spot=[_menu_item(d, menu, "Add file")], zoom=d.around(menu, 700))
    menu.close()
    file_tab = _show_tab(d, "File")
    d.shot("The File tab has the same, and more: saving, exporting an image, "
           "and simulating data.",
           spot=[_tab_button(d, "File")], point=_tab_button(d, "File"), click=True)
    section, panel = _open(d, file_tab, "Blinking Structure")
    d.shot("To practise, Simulate makes a dataset with a known structure: "
           "a ring and two crossing lines.",
           spot=[d.rect(section)], zoom=d.around(section, 760))
    d.shot("Click Run.", spot=[panel.run_button], point=panel.run_button,
           click=True, zoom=d.around(section, 760))
    panel.run_button.click()
    d.settle()
    d.shot(f"A moment later the localizations are in, and the picture is drawn.",
           spot=[d.window_rect(render)], point=d.at_data(5000, 5000))

    d.chapter("The tabs")
    d.shot("The tabs follow the workflow, from loading and fitting to the "
           "picture and its analysis.",
           spot=[d.rect(control.tabs.tabBar())],
           zoom=d.around(control.tabs.tabBar(), 640))

    d.chapter("Looking around")
    d.shot("In the render window, scroll to zoom and drag to move.",
           point=d.at_data(7500, 5000))
    render.view.frame_on((6400, 8200), (3600, 5400))
    d.settle()
    d.shot("Zoomed in, the ring is made of single localizations.",
           spot=[d.rect(render.view.graphics, pad=-2)], point=d.at_data(7480, 4700))
    render_tab = _show_tab(d, "Render")
    overview = render_tab.overview
    d.shot("The overview in the Render tab shows the whole field of view. The "
           "yellow box is where you are; click to move it.",
           spot=[d.rect(overview.graphics)], point=overview.graphics,
           zoom=d.around(overview, 700))
    reset = toolbar.widgetForAction(next(a for a in toolbar.actions()
                                         if a.text() == "Reset view"))
    d.shot("Reset view, or Ctrl+0, shows everything again.",
           spot=[reset], point=reset, click=True, zoom=d.around(reset, 800))
    render.view.reset()
    d.settle()
    menu = _menu(d, "View")
    d.shot("The View menu has Reset view too, next to the 3D view.",
           spot=[_menu_item(d, menu, "Reset view"), _menu_item(d, menu, "3D view")],
           zoom=d.around(menu, 700))
    menu.close()
    d.settle()

    d.chapter("Filters")
    d.card("Filters",
           "<p>Not every localization is a good one: dim ones are imprecise, and "
           "a bad fit lands in the wrong place.</p>"
           "<p>A <b>filter</b> keeps the localizations whose values lie in a range "
           "&ndash; precision below 10 nm, say. It changes what is drawn and what "
           "the analysis sees, <b>never the data</b>: clear it and everything is "
           "back.</p>",
           say="Filters. A filter keeps the localizations whose values lie in "
               "a range. It changes what is drawn and analysed, never the data.")
    filt = render_tab.filter
    quick = list(filt.quick.values())
    d.shot("The quick buttons choose what to filter on; the list below them "
           "has every other column.",
           spot=[d.union(*quick)], zoom=d.around(filt, 700))
    prec = filt.quick.get("loc_precision_nm")
    if prec is not None:
        prec.click()
    d.shot("The shaded part of the histogram is the range kept. Drag its "
           "edges, or type min and max.",
           spot=[d.rect(filt.plot), d.union(filt.lo, filt.hi)],
           zoom=d.around(filt, 700))
    default = filt.hi.text()
    if default not in ("", "-"):
        d.shot("SMAPpy starts with a sensible filter on the precision. "
               "Tighten it by typing a smaller maximum.",
               spot=[d.union(filt.lo, filt.hi)], point=filt.hi, click=True,
               zoom=d.around(filt, 700))
    filt.hi.setText("4")
    filt.hi.editingFinished.emit()
    d.settle()
    # the words no longer read the numbers out (STYLE.md), so they are checked
    # here instead: what a subtitle says must still be what the GUI does
    kept, total = (int(n) for n in filt.count.text().split("/"))
    assert 0 < kept < total, filt.count.text()
    d.shot("Here you see how many localizations pass the filter. Only those "
           "are drawn.",
           spot=[d.rect(filt.count), d.union(filt.lo, filt.hi)],
           zoom=d.around(filt, 700))
    filt.hi.setText(default if default not in ("", "-") else "")
    filt.hi.editingFinished.emit()
    d.shot("Type the old value back, or Clear the bound. Nothing was deleted: "
           "a filter only chooses.",
           spot=[d.union(filt.lo, filt.hi), filt.clear], point=filt.clear,
           zoom=d.around(filt, 700))

    d.chapter("Display")
    display = render_tab.mode.parentWidget()
    d.shot("Display sets how the layer is drawn: render mode, colour table, "
           "contrast.",
           spot=[display], zoom=d.around(display, 760))
    render_tab.color.setCurrentIndex(1)
    render_tab.color_field.setCurrentText("z_nm")
    d.settle()
    d.shot("Colour by a field, here z, and the picture becomes a depth map.",
           spot=[d.union(render_tab.color, render_tab.color_field)],
           point=render_tab.color_field)
    render_tab.color.setCurrentIndex(0)
    d.settle()

    d.chapter("Layers")
    d.card("Layers",
           "<p>A <b>layer</b> is one way of drawing the same localizations: its "
           "own filter, colour table and contrast. The visible layers are "
           "added into one picture.</p>"
           "<p>Two colour channels are two layers. So are the first and the second "
           "half of an acquisition &ndash; which is how to spot drift at a "
           "glance.</p>",
           figure=_LAYERS,
           say="Layers. A layer is one way of drawing the same localizations, "
               "with its own filter and colours. Visible layers add up.")
    strip = render_tab.strip
    d.shot("The layer strip: a button per layer, + to add one. Everything below "
           "edits the highlighted layer.",
           spot=[strip], zoom=d.around(strip, 640))
    half = int(session.locs["frame"].max()) // 2
    frame = filt.quick.get("frame")
    if frame is not None:
        frame.click()
    filt.hi.setText(str(half))
    filt.hi.editingFinished.emit()
    render_tab.lut.setCurrentText("red")
    d.settle()
    d.shot("Layer 1 now keeps the first half of the frames, drawn in red.",
           spot=[d.rect(filt), render_tab.lut], zoom=d.around(filt, 760))
    strip.add.menu().actions()[0].trigger()             # + -> localizations
    d.settle()
    frame = filt.quick.get("frame")
    if frame is not None:
        frame.click()
    filt.lo.setText(str(half + 1))
    filt.hi.setText("")
    filt.lo.editingFinished.emit()
    filt.hi.editingFinished.emit()
    render_tab.lut.setCurrentText("green")
    d.settle()
    d.shot("Layer 2 keeps the second half, in green.",
           spot=[strip, d.rect(filt), render_tab.lut], zoom=d.around(filt, 760))
    render.view.frame_on((6400, 8200), (3600, 5400))
    d.settle()
    d.shot("Where the halves agree, the picture is yellow. Red and green "
           "fringes would mean drift.",
           spot=[d.rect(render.view.graphics, pad=-2)], point=d.at_data(7480, 4700))
    session.remove_layer(1)
    render_tab.strip.select(0)
    filt.quick["frame"].click()
    filt.clear.click()
    render_tab.lut.setCurrentText("hot")
    render.view.reset()
    d.settle()

    d.chapter("Grouping")
    d.card("Grouping",
           "<p>A fluorophore is often on for several frames, and is localized in "
           "each of them.</p>"
           "<p><b>Grouping</b> links localizations that are close (50 nm) in "
           "consecutive frames (one dark frame allowed) into one, with the "
           "photons added. The ungrouped table stays; each layer chooses which "
           "one it shows.</p>",
           figure=_GROUPING,
           say="Grouping. One blink seen in several frames is several "
               "localizations. Grouping links them into one.")
    if not render_tab.grouped.isChecked():
        render_tab.grouped.click()
        d.settle()
    assert len(session.table(0)[0]) < len(session.locs), "grouping merged nothing"
    d.shot("SMAPpy groups a new table straight away, so what you see are "
           "blinks, not single frames.",
           spot=[render_tab.grouped], point=render_tab.grouped,
           zoom=d.around(render_tab.grouped, 700))
    render_tab.grouped.click()
    d.settle()
    d.shot("Untick grouped to see every localization of every frame.",
           spot=[render_tab.grouped, d.rect(render.view.graphics, pad=-2)],
           point=render_tab.grouped, click=True)
    render_tab.grouped.click()
    d.settle()
    d.shot("Parameters... sets how close, and how many dark frames apart, "
           "localizations are linked.",
           spot=[overview.parameters_button], point=overview.parameters_button,
           zoom=d.around(overview, 700))

    d.chapter("The selection")
    d.card("What a plugin works on",
           "<p>Every analysis gets the <b>selection</b>: the localizations of the "
           "current layer that pass its filter <i>and</i> lie inside the region of "
           "interest, if one is drawn.</p>"
           "<p>What you see is what is measured.</p>",
           figure=_SELECTION,
           say="The selection. A plugin measures what passes the current "
               "layer's filter, inside the ROI if one is drawn.")
    d.shot("To draw a region of interest, click ROI. Right-click it for a line, "
           "rectangle or polygon.",
           spot=[toolbar.roi_button], point=toolbar.roi_button, click=True,
           zoom=d.around(toolbar.roi_button, 800))
    from ...regions import Region
    toolbar._set_kind("rect", "rectangle")
    session.set_roi(Region.rect(2300, 2300, 7700, 7700))
    d.settle()
    assert 0 < len(session.shown_selection(0)) < len(session.table(0)[0])
    d.shot("Then drag over the picture. The toolbar shows how many "
           "localizations are inside, and plugins now measure only those.",
           spot=[d.data_rect(2300, 2300, 7700, 7700, pad=6), toolbar.counts],
           point=d.at_data(7700, 7700))

    d.chapter("Plugins")
    analysis = _show_tab(d, "Analysis")
    d.shot("The Analysis tab has a section per plugin, one open at a time.",
           spot=[d.rect(analysis)], point=_tab_button(d, "Analysis"), click=True,
           zoom=d.around(_tab_button(d, "Analysis"), 760))
    d.shot("Find narrows the list; + pins another plugin to the tab.",
           spot=[analysis.search], zoom=d.around(analysis.search, 640))
    section, panel = _open(d, analysis, "Localization Statistics")
    d.shot("Localization Statistics measures photons, precision and on-time.",
           spot=[d.rect(section)], zoom=d.around(section, 760))
    panel.run_button.click()
    d.settle()
    d.shot("Run it: a summary appears, and Plot draws the figures.",
           spot=[panel.output, panel.plot_button], point=panel.plot_button,
           click=True, zoom=d.around(panel.output, 760))
    panel.plot()
    d.settle()
    window = panel._window
    x, y, w, h = d.window_rect(render)
    d.place(window, x + 80, y + 40, w - 160, h - 60)
    d.settle()
    d.shot("The figures get a window of their own, reused by the next run.",
           spot=[d.window_rect(window)])
    window.hide()
    menu = _menu(d, "Plugins")
    d.shot("The Plugins menu has every plugin, to open once in a window of "
           "its own. Find a plugin..., Ctrl+Shift+P, searches them by name.",
           spot=[d.rect(menu)], zoom=d.around(menu, 700))
    menu.close()

    d.chapter("Undo and saving")
    menu = _menu(d, "File")
    d.shot("What changes the table, like a drift correction, can be undone. "
           "Grey here: filters and layers never change it.",
           spot=[_menu_item(d, menu, "Undo"), _menu_item(d, menu, "Undo steps")],
           zoom=d.around(menu, 700))
    d.shot("Save, or Ctrl+S, keeps the localizations, their history and your "
           "ROIs in one file.",
           spot=[_menu_item(d, menu, "Save"), _menu_item(d, menu, "Save as")],
           zoom=d.around(menu, 700))
    menu.close()
    d.settle()

    d.chapter("Next")
    d.card("That is the tour",
           "<ul><li><b>Filters</b> choose which localizations count.</li>"
           "<li><b>Layers</b> draw the same localizations in different ways, "
           "and add up.</li>"
           "<li><b>Grouping</b> turns one blink seen in several frames into one "
           "localization.</li>"
           "<li><b>The selection</b> &ndash; filter and ROI &ndash; is what a plugin "
           "measures.</li></ul>"
           "<p>Next: fitting raw data, rendering in 2D and 3D, and the ROI "
           "manager.</p>",
           say="That is the tour. Next: fitting raw data, rendering, and the "
               "ROI manager.")
