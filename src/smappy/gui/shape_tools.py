"""Drawing shapes on a figure's panel: the tools a `ShapeEdit` asks for.

A plugin's figure can say that one of its panels takes shapes (`ShapeEdit`):
the fit preview's frame takes the region to fit in.  This is the row of
buttons the figure window puts above such a figure, and the matplotlib
selectors that draw on the panel -- matplotlib's own, so the drawing works
in the figure as it is, zoomed and panned with its toolbar.

A shape is finished when it is closed -- a rectangle on release, a polygon
on clicking its first point again -- and is then added to the edit's shapes
in place and handed to ``ShapeEdit.on_change``, which the panel set to write
it into the form.  The figure is then drawn again (``redraw``): the plot
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
        # the tool that was on stays on over the figure drawn again
        on = next((i for i, b in enumerate(self.buttons) if b.isChecked()), None)
        if on is not None and self.ax is not None:
            kind, how, _ = TOOLS[on]
            self._start(kind, how)

    def _choose(self, index: int) -> None:
        """A tool's button: on starts drawing that kind of shape, off stops."""
        on = self.buttons[index].isChecked()
        for i, button in enumerate(self.buttons):
            if i != index:
                button.setChecked(False)
        self._stop()
        if on and self.ax is not None:
            kind, how, _ = TOOLS[index]
            self._start(kind, how)

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
        self._changed()

    def clear(self) -> None:
        if self.edit is None or not self.edit.shapes:
            return
        self.edit.shapes.clear()
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
