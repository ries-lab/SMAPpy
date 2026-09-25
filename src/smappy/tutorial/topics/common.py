"""What every storyboard needs: find a tab, a section, a menu item, on screen.

Each returns a widget or a desktop rectangle for `Director.shot`, and raises
naming what *is* there when the thing asked for is not -- which is how a
renamed tab or plugin shows up when a tutorial is rebuilt.
"""
from __future__ import annotations


def tab(d, name: str):
    tabs = d.control.tabs
    for i in range(tabs.count()):
        if tabs.tabText(i) == name:
            return i, tabs.widget(i)
    raise LookupError(f"no {name!r} tab; there are "
                      + ", ".join(tabs.tabText(i) for i in range(tabs.count())))


def show_tab(d, name: str):
    i, widget = tab(d, name)
    d.control.tabs.setCurrentIndex(i)
    return widget


def tab_button(d, name: str):
    """The tab's own button in the tab bar, as a rectangle."""
    i, _ = tab(d, name)
    bar = d.control.tabs.tabBar()
    r = bar.tabRect(i)
    x, y, _, _ = d.rect(bar, pad=0)
    return (x + r.x(), y + r.y(), r.width(), r.height())


def section(plugin_tab, title: str):
    """A plugin's section in a tab, by the start of its title."""
    for found in plugin_tab.sections:
        if found.title.lower().startswith(title.lower()):
            return found
    raise LookupError(f"no section {title!r}; there are "
                      + ", ".join(s.title for s in plugin_tab.sections))


def open_section(d, plugin_tab, title: str):
    """Expand a plugin's section and return it and its panel."""
    found = section(plugin_tab, title)
    if not found.button.isChecked():
        found.button.click()
    d.settle()
    panel = next(p for p in plugin_tab.panels()
                 if p.plugin.name.lower().startswith(title.lower()))
    return found, panel


def field(panel, path: str):
    """A plugin form's field by dotted name: ``field(panel, "camera.offset")``."""
    target = panel.form
    for part in path.split("."):
        target = target.fields[part]
    return target


def type_into(panel, path: str, value) -> None:
    """Set a field as a person would, so the plugin hears about it (`react`):
    the fitter fills the output path from the file only when told the file
    changed, which `SettingsForm.set_values` deliberately does not do."""
    target = field(panel, path)
    target.set(value)
    target.changed.emit()
    edit = getattr(target, "widget", None)
    if hasattr(edit, "setCursorPosition"):
        edit.setCursorPosition(len(edit.text()))     # a path shows its file name


def place_beside(d, window, width_fraction: float = 0.9):
    """Put a plugin's own window (a preview, a result) over the render window,
    where it would be looked at, rather than wherever Qt opens it."""
    x, y, w, h = d.window_rect(d.render)
    width = int(w * width_fraction)
    d.place(window, x + (w - width) // 2, y + 30, width, h - 50)
    d.settle()


def toolbar(d):
    from ...gui.render_view import RenderToolBar
    return d.render.findChild(RenderToolBar)


def menu(d, title: str):
    """Open one of the control window's menus where it would drop down."""
    from PySide6.QtCore import QPoint
    bar = d.control.menuBar()
    action = next(a for a in bar.actions() if a.text().replace("&", "") == title)
    geometry = bar.actionGeometry(action)
    menu = action.menu()
    menu.popup(bar.mapToGlobal(QPoint(geometry.x(), geometry.bottom() + 1)))
    d.settle()
    return menu


def menu_item(d, menu, text: str):
    """An item of an open menu, as a rectangle on the desktop."""
    action = next(a for a in menu.actions() if a.text().replace("&", "").startswith(text))
    r = menu.actionGeometry(action)
    g = menu.geometry()
    return (g.x() + r.x(), g.y() + r.y(), r.width(), r.height())


def section_of(panel, name: str):
    """A part of a plugin's form (``detection``), as the section that holds it."""
    from ...gui.widgets import CollapsibleSection
    widget = field(panel, name)
    while widget is not None and not isinstance(widget, CollapsibleSection):
        widget = widget.parentWidget()
    return widget if widget is not None else field(panel, name)


def expand(widget) -> None:
    """Open every collapsed section around a widget, as a person clicks the
    title of "more" before reaching what is inside it."""
    from ...gui.widgets import CollapsibleSection
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, CollapsibleSection) and not parent.button.isChecked():
            parent.set_expanded(True)
        parent = parent.parentWidget()


def figure_panels(d, window, pad: float = 0.03):
    """The panels of the figure a result window shows, as desktop rectangles,
    top to bottom -- read from the figure's own layout, since a figure's
    panels are not an equal division of it (titles and axes take room)."""
    from ...gui.figures import FigurePane
    panes = [p for p in window.findChildren(FigurePane) if p.isVisible()]
    if not panes:
        raise LookupError("the window shows no figure")
    pane = panes[0]
    x, y, w, h = d.rect(pane.canvas, pad=0)
    rects = []
    for ax in pane.figure.axes:
        if not ax.get_visible() or not ax.has_data():
            continue
        box = ax.get_position()
        x0, x1 = max(0.0, box.x0 - pad), min(1.0, box.x1 + pad / 2)
        y0, y1 = max(0.0, box.y0 - 2 * pad), min(1.0, box.y1 + 1.5 * pad)
        rects.append((x + x0 * w, y + (1 - y1) * h, (x1 - x0) * w, (y1 - y0) * h))
    return sorted(rects, key=lambda r: r[1])
