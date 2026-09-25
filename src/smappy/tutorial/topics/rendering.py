"""Rendering in 2D and 3D: how a table becomes a picture, and how to change it.

The render modes (and what each says about the data), contrast, the colour
table and a white background for a figure, saving, a picture whose axes are
other columns, and the 3D view with its slab.  On the tour's simulated ring
and lines, which have a z to show in 3D; the table comes from File ->
Simulate, as in the tour, without showing it again.

The 3D view uses its CPU engine here -- the offscreen platform has no GPU --
which is the same picture the GPU engine draws, only slower to turn.
"""
from __future__ import annotations

from .common import (expand, menu, menu_item, open_section, show_tab, tab_button,
                     toolbar)

TITLE = "Rendering in 2D and 3D"
DESCRIPTION = ("What is drawn (filters and grouping), render modes, contrast "
               "and colours, saving, other axes, and the 3D view.")

_RENDER = """
<svg viewBox="0 0 560 160" role="img" aria-label="points with an uncertainty, drawn as blurs or counted per pixel">
  <g font-size="13" fill="currentColor">
    <rect x="10" y="15" width="150" height="100" rx="8" class="panel"/>
    <circle cx="50" cy="50" r="3" class="dot"/><circle cx="62" cy="58" r="3" class="dot"/>
    <circle cx="110" cy="80" r="3" class="dot"/><circle cx="120" cy="40" r="3" class="dot"/>
    <circle cx="50" cy="50" r="7" fill="none" stroke="currentColor" opacity=".35"/>
    <circle cx="62" cy="58" r="11" fill="none" stroke="currentColor" opacity=".35"/>
    <circle cx="110" cy="80" r="5" fill="none" stroke="currentColor" opacity=".35"/>
    <circle cx="120" cy="40" r="9" fill="none" stroke="currentColor" opacity=".35"/>
    <text x="85" y="140" text-anchor="middle">positions, each with its precision</text>
    <path d="M170 65 h35" class="arrow" marker-end="url(#h5)"/>
    <rect x="215" y="15" width="150" height="100" rx="8" fill="#000"/>
    <circle cx="255" cy="50" r="7" fill="#ffb040" opacity=".9"/><circle cx="267" cy="58" r="11" fill="#ffb040" opacity=".45"/>
    <circle cx="315" cy="80" r="5" fill="#ffd070"/><circle cx="325" cy="40" r="9" fill="#ffb040" opacity=".55"/>
    <text x="290" y="140" text-anchor="middle">precision: as wide as each is uncertain</text>
    <rect x="400" y="15" width="150" height="100" rx="8" fill="#000"/>
    <rect x="435" y="40" width="20" height="20" fill="#ffb040"/><rect x="495" y="70" width="20" height="20" fill="#ffb040" opacity=".6"/>
    <rect x="515" y="30" width="20" height="20" fill="#ffb040" opacity=".6"/>
    <text x="475" y="140" text-anchor="middle">hist: counted per pixel</text>
  </g>
  <defs><marker id="h5" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>
</svg>
"""


def make(d) -> None:
    session, control, render = d.session, d.control, d.render
    bar = toolbar(d)

    # the tour's simulated table, made the way it shows (File -> Simulate)
    file_tab = show_tab(d, "File")
    _, simulate = open_section(d, file_tab, "Blinking Structure")
    simulate.run_button.click()
    d.settle()
    tab = show_tab(d, "Render")
    d.settle()

    d.chapter("From table to picture")
    d.card(TITLE,
           "<p>What is drawn, how SMAPpy draws it, what to change for a "
           "figure, how to save it, and the 3D view.</p>",
           say="This tutorial is about the picture: how it is drawn, how to "
               "change it, and the 3D view.")
    d.card("A picture of positions",
           "<p>A localization is a position with an uncertainty. Rendering "
           "draws each one as a small blur, or counts them per pixel.</p>"
           "<p>What the picture looks like is a choice; the table underneath "
           "stays the same.</p>",
           figure=_RENDER,
           say="A localization is a position with an uncertainty. Rendering "
               "draws each one as a blur, or counts them per pixel.")

    d.chapter("What is drawn")
    filt = tab.filter
    bold = [b for b in filt.quick.values() if "bold" in b.styleSheet()]
    assert bold, "a new table has no default filter"
    d.shot("Each layer draws only what passes its filter. A quick button in "
           "bold has a bound: a new table starts with a few, on precision and "
           "fit quality.",
           spot=bold, zoom=d.around(filt, 700))
    ll = filt.quick.get("logl_rel")
    if ll is not None:
        ll.click()
        d.settle()
        d.shot("The histogram shows the column, and the shaded range is what is "
               "kept. Here fit quality: fits worse than the bound are left out.",
               spot=[ll, d.rect(filt.plot)], point=ll, click=True,
               zoom=d.around(filt, 700))
    d.shot("Any column can be filtered: the list below the buttons has them "
           "all, and each layer keeps its own bounds.",
           spot=[filt.field], point=filt.field, zoom=d.around(filt, 700))
    ring = (6400, 8200), (3600, 5400)
    render.view.frame_on(*ring)
    d.settle()
    assert tab.grouped.isChecked(), "a new table is not grouped"
    d.shot("Grouped, each blink is one localization: its positions averaged "
           "and its photons added, so it is more precise.",
           spot=[tab.grouped, d.rect(render.view.graphics, pad=-2)],
           point=tab.grouped)
    tab.grouped.click()
    d.settle()
    d.shot("Ungrouped, every frame of every blink is drawn: more localizations, "
           "each one less precise.",
           spot=[tab.grouped, d.rect(render.view.graphics, pad=-2)],
           point=tab.grouped, click=True)
    tab.grouped.click()
    d.settle()
    from ...gui.dialogs import ParametersDialog
    dialog = ParametersDialog(session, control)
    dialog.show()
    d.settle()
    x, y, w, h = d.rect(tab.overview.parameters_button)
    d.place(dialog, x + w + 20, y - 40)
    d.shot("Parameters, beside the overview, sets how far a molecule may move "
           "between frames, and how many dark frames a blink may have.",
           spot=[tab.overview.parameters_button, d.window_rect(dialog)],
           point=tab.overview.parameters_button, click=True,
           zoom=d.around(dialog, 800))
    dialog.reject()
    d.settle()
    d.shot("The filter and the grouping also decide what the analysis plugins "
           "measure: what is drawn is what is measured.",
           spot=[tab_button(d, "Analysis"), d.rect(render.view.graphics, pad=-2)],
           point=tab_button(d, "Analysis"))

    d.chapter("Render modes")
    d.shot("Render precision draws each localization as wide as it is "
           "uncertain: well-localized ones are sharp, poor ones are soft.",
           spot=[tab.mode, d.rect(render.view.graphics, pad=-2)], point=tab.mode)
    tab.mode.setCurrentText("hist")
    d.settle()
    d.shot("Hist counts the localizations in each pixel. Nothing is smoothed, "
           "so up close the picture is grainy, but it hides nothing.",
           spot=[tab.mode, d.rect(render.view.graphics, pad=-2)], point=tab.mode)
    tab.mode.setCurrentText("gauss")
    d.settle()
    expand(tab.sigma)
    d.shot("Gauss draws every localization with the same width, set under more.",
           spot=[tab.mode, tab.sigma], point=tab.sigma,
           zoom=d.around(tab.sigma, 760))
    tab.mode.setCurrentText("precision")
    render.view.reset()
    d.settle()

    d.chapter("Brightness and colour")
    contrast = tab.contrast.value()
    tab.contrast.setValue(max(0.5, contrast - 0.5))
    d.settle()
    d.shot("Contrast sets how many of the brightest pixels are saturated. A "
           "smaller value saturates more, and faint structure comes up.",
           spot=[tab.contrast, d.rect(render.view.graphics, pad=-2)],
           point=tab.contrast)
    tab.contrast.setValue(contrast)
    tab.lut.setCurrentText("gray")
    d.settle()
    d.shot("The colour table, LUT, maps brightness to colour.",
           spot=[tab.lut, d.rect(render.view.graphics, pad=-2)], point=tab.lut)
    tab.lut.setCurrentText("hot")
    expand(tab.white)
    tab.white.setChecked(True)
    d.settle()
    d.shot("White background turns the picture over for a figure: ink on "
           "paper, with the colours kept.",
           spot=[tab.white, d.rect(render.view.graphics, pad=-2)], point=tab.white,
           click=True)
    tab.white.setChecked(False)
    d.settle()

    d.chapter("Saving")
    save = bar.widgetForAction(bar.actions()[0])
    save_menu = save.menu()
    from PySide6.QtCore import QPoint
    save_menu.popup(save.mapToGlobal(QPoint(0, save.height())))
    d.settle()
    d.shot("Save keeps a PNG as you see it, or renders a TIFF at a pixel size "
           "you choose: in colour, or as intensities to measure.",
           spot=[d.rect(save_menu)], zoom=d.around(save_menu, 800))
    save_menu.close()
    d.settle()

    d.chapter("Other axes")
    axes = tab.axes
    expand(axes.fields["y"])
    axes.fields["y"].setCurrentText("z_nm")
    d.settle()
    axes.fit.click()
    d.settle()
    d.shot("The picture's axes need not be x and y. With z as the second axis, "
           "the same table is a side view of the ring.",
           spot=[d.union(axes.fields["x"], axes.fields["y"]),
                 d.rect(render.view.graphics, pad=-2)],
           point=axes.fields["y"])
    d.shot("Any column works: photons against frame, say, to see bleaching. "
           "Reset brings back the ordinary picture.",
           spot=[axes.reset], point=axes.reset, click=True,
           zoom=d.around(axes.reset, 760))
    axes.reset.click()
    render.view.reset()
    d.settle()

    d.chapter("3D")
    view_menu = menu(d, "View")
    d.shot("View, 3D view opens the same layers in 3D, with their filters "
           "and colours.",
           spot=[menu_item(d, view_menu, "3D view")], point=menu_item(d, view_menu, "3D view"),
           click=True, zoom=d.around(view_menu, 700))
    view_menu.close()
    control.show_3d()
    d.settle()
    window = control.view3d_window
    d.pump(0.5)
    d.settle()
    _turn(window.view, 35.0, 55.0)
    d.pump(0.5)
    d.settle()
    d.shot("Drag to turn it, scroll to zoom. Top, front and side jump to those "
           "views; fit brings the whole box back.",
           spot=[d.window_rect(window)])
    panel = window.panel
    panel.depth_color.setChecked(True)
    d.pump(0.5)
    d.settle()
    d.shot("Colour by depth shows which parts are near and which are far.",
           spot=[panel.depth_color, d.window_rect(window)], point=panel.depth_color)
    window._preset("top")
    lo, hi = panel.ranges[2]
    lo.setValue(0.0)
    hi.setValue(150.0)
    d.pump(0.5)
    d.settle()
    d.shot("The slab is the box that is drawn. Narrowed in z it is a thin "
           "section; it can also follow a region drawn in 2D.",
           spot=[d.union(lo, hi), d.window_rect(window)], point=lo)
    assert any(r.isVisible() for r in (window, panel)), "the 3D view did not open"

    d.chapter("Next")
    d.card("What to remember",
           "<ul><li><b>Render mode:</b> precision for the picture, hist to see "
           "exactly what is there.</li>"
           "<li><b>Contrast, LUT, white background:</b> how it looks, never "
           "what it is.</li>"
           "<li><b>Save:</b> a PNG as shown, or a TIFF at a chosen pixel size.</li>"
           "<li><b>3D view:</b> the same layers, turned, with a slab.</li></ul>",
           say="That is rendering. Next: drift correction, grouping and "
               "filtering.")


def _turn(view, horizontal: float, vertical: float) -> None:
    """Turn the 3D view as a mouse drag does.  The view owns its projection,
    so setting the session's does nothing to what is on screen."""
    if view.projection.rotate_at_centre:
        view.projection.pivot_at_centre()
    view.projection.rotate_view(horizontal, vertical)
    view._draw_box()
    view.changed.emit()
    view.schedule()
