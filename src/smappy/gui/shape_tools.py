"""Drawing shapes on a figure's panel: the tools a `ShapeEdit` asks for.

A plugin's figure can say that one of its panels takes shapes (`ShapeEdit`):
the fit preview's frame takes the region to fit in.  This is the row of
buttons the figure window puts above such a figure, and the matplotlib
selectors that draw on the panel -- matplotlib's own, so the drawing works
in the figure as it is, zoomed and panned with its toolbar.

A shape is finished when it is closed -- a rectangle on release, a polygon
on clicking its first point again -- and is then added to the edit's shapes
in place and handed to ``ShapeEdit.on_change``, which the panel set to write
it into the form.  *edit* takes a drawn shape back: a click on it puts it in
matplotlib's interactive selector, a rectangle with handles on its corners
and edges and moved by its middle, a polygon by its vertices (shift-drag
moves it whole), and each release writes it back the same way.  Before
(2026-11) a shape could only be undone and drawn again.  The figure is then drawn again (``redraw``): the plot
draws the region from that same list, so what is on the panel is always what
is in the settings, after an undo as after a new shape.  The tool stays on
for the next shape until its button is clicked again.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QWidget

#: (kind, how drawn, button label)
TOOLS = (("include", "rect", "fit in: rectangle"),
         ("include", "poly", "fit in: polygon"),
         ("exclude", "rect", "leave out: rectangle"),
         ("exclude", "poly", "leave out: polygon"))
COLOURS = {"include": "#2ca02c", "exclude": "#d62728"}


def is_rectangle(points) -> bool:
    """Whether a shape is an upright rectangle -- four corners, edges along the axes -- and so
    edited with a rectangle's handles."""
    points = np.asarray(points, float)
    if points.shape != (4, 2):
        return False
    edges = np.roll(points, -1, axis=0) - points
    return bool(np.all(np.isclose(edges[:, 0], 0) | np.isclose(edges[:, 1], 0)))


class ShapeTools(QWidget):
    """The buttons for a `ShapeEdit`, and the drawing on its panel."""

    def __init__(self, canvas, redraw, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.redraw = redraw              # draws the figure again, then `attach`
        self.edit = None
        self.ax = None
        self._selector = None
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)
        row.addWidget(QLabel("draw:"))
        self.group = QButtonGroup(self)
        self.group.setExclusive(False)
        self.buttons = []
        for i, (kind, how, label) in enumerate(TOOLS):
            button = QPushButton(label, checkable=True)
            button.setToolTip("drag a rectangle" if how == "rect" else
                              "click the corners; click the first one again to close")
            self.group.addButton(button, i)
            row.addWidget(button)
            self.buttons.append(button)
        # takes a drawn shape back to move or resize it (`_pressed`)
        self.edit_button = QPushButton("edit", checkable=True)
        self.edit_button.setToolTip("click a shape to move or resize it: drag its handles or its "
                                    "middle; a polygon's corners one by one, shift-drag moves it")
        self.group.addButton(self.edit_button, len(TOOLS))
        row.addWidget(self.edit_button)
        self._editing = None              # the index of the shape in the selector
        self._press_id = None
        self.group.idClicked.connect(self._choose)
        self.undo_button = QPushButton("undo")
        self.undo_button.setToolTip("take the last shape away")
        self.undo_button.clicked.connect(self.undo)
        self.clear_button = QPushButton("clear")
        self.clear_button.setToolTip("take every shape away: fit the whole frame")
        self.clear_button.clicked.connect(self.clear)
        row.addWidget(self.undo_button)
        row.addWidget(self.clear_button)
        row.addStretch(1)
        self.hide()

    # ------------------------------------------------------------ the edit
    def attach(self, edit, figure) -> None:
        """Draw on ``edit``'s panel of ``figure`` (just drawn), or show
        nothing when there is no edit."""
        self._stop()
        self.edit = edit
        axes = [a for a in figure.axes] if figure is not None else []
        self.ax = axes[edit.panel] if edit is not None and edit.panel < len(axes) else None
        self.setVisible(self.ax is not None)
        # the tool that was on stays on over the figure drawn again, and so does the shape in edit
        on = next((i for i, b in enumerate(self.buttons) if b.isChecked()), None)
        if on is not None and self.ax is not None:
            kind, how, _ = TOOLS[on]
            self._start(kind, how)
        elif self.edit_button.isChecked() and self.ax is not None:
            self._listen(True)
            if self._editing is not None and self._editing < len(self.edit.shapes):
                self._start_editing(self._editing)
            else:
                self._editing = None

    def _choose(self, index: int) -> None:
        """A tool's button: on starts drawing that kind of shape (or, *edit*, taking one back),
        off stops."""
        every = self.buttons + [self.edit_button]
        on = every[index].isChecked()
        for i, button in enumerate(every):
            if i != index:
                button.setChecked(False)
        self._stop()
        self._editing = None
        self._listen(on and index == len(TOOLS))
        if on and self.ax is not None and index < len(TOOLS):
            kind, how, _ = TOOLS[index]
            self._start(kind, how)

    # ------------------------------------------------------------ editing
    def _listen(self, on: bool) -> None:
        """Clicks on the canvas pick the shape to edit while *edit* is on."""
        if self._press_id is not None:
            self.canvas.mpl_disconnect(self._press_id)
            self._press_id = None
        if on:
            self._press_id = self.canvas.mpl_connect("button_press_event", self._pressed)

    def shape_at(self, x: float, y: float):
        """The index of the shape drawn last under (x, y), within a few per cent of the view of its
        outline, so that a click on a handle finds it too; None when none."""
        from matplotlib.path import Path
        if self.edit is None or self.ax is None:
            return None
        (left, right), (bottom, top) = self.ax.get_xlim(), self.ax.get_ylim()
        near = 0.02 * max(abs(right - left), abs(top - bottom))
        for index in range(len(self.edit.shapes) - 1, -1, -1):
            points = np.asarray(self.edit.shapes[index]["points"], float)
            path = Path(np.vstack([points, points[:1]]), closed=True)
            if path.contains_point((x, y), radius=near) or path.contains_point((x, y), radius=-near):
                return index
        return None

    def _pressed(self, event) -> None:
        """A click with *edit* on: the shape under it into the selector, another one in its
        place, or none -- the selector let go -- when it is on no shape."""
        try:
            if event.inaxes is not self.ax or event.xdata is None:
                return
            if getattr(self.canvas, "toolbar", None) is not None and \
                    getattr(self.canvas.toolbar, "mode", ""):
                return                                   # zooming or panning
            index = self.shape_at(event.xdata, event.ydata)
            if index == self._editing:
                return                                   # the selector's own handles
            self._stop()
            self._editing = index
            if index is not None:
                self._start_editing(index)
        except Exception as error:                       # never out of a Qt slot
            print(f"the shape could not be taken up: {error}")

    def _start_editing(self, index: int) -> None:
        """Shape ``index`` in matplotlib's interactive selector; each release writes it back."""
        from matplotlib.widgets import PolygonSelector, RectangleSelector
        shape = self.edit.shapes[index]
        points = np.asarray(shape["points"], float)
        colour = COLOURS.get(shape.get("kind"), COLOURS["include"])
        if is_rectangle(points):
            def moved(press, release):
                x0, x1, y0, y1 = self._selector.extents
                if x1 - x0 > 1e-9 and y1 - y0 > 1e-9:
                    self._replace(index, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            self._selector = RectangleSelector(
                self.ax, moved, useblit=False, interactive=True, ignore_event_outside=True,
                props=dict(edgecolor=colour, fill=False, linewidth=2),
                handle_props=dict(markeredgecolor=colour))
            self._selector.extents = (points[:, 0].min(), points[:, 0].max(),
                                      points[:, 1].min(), points[:, 1].max())
        else:
            def moved(vertices):
                if len(vertices) >= 3:
                    self._replace(index, [[float(x), float(y)] for x, y in vertices])
            self._selector = PolygonSelector(self.ax, moved, useblit=False,
                                             props=dict(color=colour, linewidth=2),
                                             handle_props=dict(markeredgecolor=colour))
            self._selector.verts = [tuple(p) for p in points]
        self.canvas.draw_idle()

    def _replace(self, index: int, points) -> None:
        if self.edit is None or index >= len(self.edit.shapes):
            return
        shape = dict(self.edit.shapes[index])
        shape["points"] = [[float(x), float(y)] for x, y in points]
        if shape == self.edit.shapes[index]:
            return
        self.edit.shapes[index] = shape
        self._changed()

    def _start(self, kind: str, how: str) -> None:
        from matplotlib.widgets import PolygonSelector, RectangleSelector
        props = dict(color=COLOURS[kind], linewidth=1.5)
        if how == "rect":
            def done(press, release):
                x0, x1 = sorted((press.xdata, release.xdata))
                y0, y1 = sorted((press.ydata, release.ydata))
                if None in (x0, y0) or x1 - x0 < 1e-9 or y1 - y0 < 1e-9:
                    return
                self._add(kind, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            self._selector = RectangleSelector(self.ax, done, useblit=False,
                                               props=dict(edgecolor=COLOURS[kind],
                                                          fill=False, linewidth=1.5))
        else:
            def done(vertices):
                if len(vertices) >= 3:
                    self._add(kind, [[float(x), float(y)] for x, y in vertices])
            self._selector = PolygonSelector(self.ax, done, useblit=False, props=props)

    def _stop(self) -> None:
        if self._selector is not None:
            try:
                self._selector.set_active(False)
                self._selector.set_visible(False)
                self._selector.disconnect_events()
            except Exception:
                pass
            self._selector = None
            self.canvas.draw_idle()

    # ------------------------------------------------------------- shapes
    def _add(self, kind: str, points) -> None:
        self.edit.shapes.append(
            {"kind": kind, "points": [[float(x), float(y)] for x, y in points]})
        self._changed()

    def undo(self) -> None:
        if self.edit is None or not self.edit.shapes:
            return
        self.edit.shapes.pop()
        self._editing = None
        self._changed()

    def clear(self) -> None:
        if self.edit is None or not self.edit.shapes:
            return
        self.edit.shapes.clear()
        self._editing = None
        self._changed()

    def _changed(self) -> None:
        """Into the settings, then onto the panel."""
        if self.edit is not None and self.edit.on_change is not None:
            try:
                self.edit.on_change(list(self.edit.shapes))
            except Exception as error:               # never out of a Qt slot
                print(f"the drawn shapes could not be kept: {error}")
        self._stop()
        self.redraw()
