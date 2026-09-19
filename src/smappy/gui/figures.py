"""A result's figures, in one window, drawn when they are looked at.

A plugin with one figure gets a window with a figure in it, which is what it
always had.  A plugin with several gets them as tabs of that one window rather
than as a pile of windows on the screen -- and a tab can be torn off when two
of them have to be seen at once, which is the one thing tabs are bad at.

Nothing is drawn until it is on screen.  A plugin run again, or an ROI
selected, marks every tab stale and redraws only the one being looked at
(and any torn off, which are on screen by definition); the rest draw when
they are next clicked, and say so until then by greying their label.  That
is what makes six figures cost what one figure costs, and it is the whole
reason a site can be stepped through at all.

The window is Qt's, not pyplot's: pyplot keeps every figure it ever made in a
global registry, which for figures that are redrawn per ROI is a leak with a
warning attached.  Here a figure lives as long as the window it is in.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, QWidget)

from ..plugins import Plot

# the summary page's key.  A tab character cannot be in a plot's name as
# anyone would write one, so nothing a plugin calls its figure collides here
SUMMARY = "\tall"
SUMMARY_LABEL = "All"


def _label(name: str) -> str:
    """What a tab says: the plot's name, or what to call the two unnamed."""
    return SUMMARY_LABEL if name == SUMMARY else (name or "figure")


def _canvas_classes():
    """Imported late: matplotlib's Qt backend is not free to import."""
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg,
                                                   NavigationToolbar2QT)
    from matplotlib.figure import Figure
    return FigureCanvasQTAgg, NavigationToolbar2QT, Figure


class FigurePane(QWidget):
    """One figure with its toolbar, and what it is currently showing.

    It holds the `Plot` it was given rather than the picture: a plot is a
    closure over the data it draws, so keeping it is what lets the pane be
    redrawn later, when it is looked at, without running anything again.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        canvas_cls, toolbar_cls, figure_cls = _canvas_classes()
        self.figure = figure_cls(layout="constrained")
        self.canvas = canvas_cls(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar_cls(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        self.plot: Optional[Plot] = None
        self.stale = True

    def offer(self, plot: Plot) -> None:
        """This is what to draw next time the pane is looked at."""
        self.plot = plot
        self.stale = True

    def draw(self) -> None:
        """Draw what was last offered, unless it is already on the screen."""
        if not self.stale or self.plot is None:
            return
        self.figure.clear()
        self.plot.draw_into(self.figure)
        self.canvas.draw_idle()
        self.stale = False


def summary_plot(plots: Sequence[Plot]) -> Plot:
    """Every figure on one page, each in a subfigure of it.

    A page is worth having for the glance and for what gets exported, and it
    is worth nothing if it costs the figures being drawn twice, so it is a
    tab like any other and is drawn when it is looked at.

    Each plot is handed a `SubFigure`, which takes `subplots` exactly as a
    figure does -- which is why a plot declares its panels instead of
    helping itself to `ax.figure`: a plot that seizes the figure would take
    the whole page with it.
    """
    plots = list(plots)

    def draw(figure) -> None:
        columns = 1 if len(plots) == 1 else 2
        rows = -(-len(plots) // columns)
        cells = figure.subfigures(rows, columns, squeeze=False).ravel()
        for plot, cell in zip(plots, cells):
            if plot.name:
                cell.suptitle(plot.name, fontsize=9)
            plot.draw_into(cell)
        for cell in cells[len(plots):]:
            cell.set_visible(False)

    return Plot(draw=draw, name=SUMMARY, panels=len(plots) + 1)


class DetachedFigure(QMainWindow):
    """A pane taken out of the tabs, and the way back in.

    A window rather than an instance with its `closeEvent` replaced: patching
    a virtual onto one PySide object works until the interpreter takes the
    C++ side apart in an order nobody chose, and a segfault at exit is a
    poor trade for six saved lines.
    """

    closed = Signal(str)

    def __init__(self, name: str, title: str, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle(title)
        self._name = name

    def closeEvent(self, event) -> None:           # noqa: N802 (Qt's name)
        self.closed.emit(self._name)
        super().closeEvent(event)


class FigureTabs(QWidget):
    """The figures of one result: a pane, or a tab each beyond the first.

    A widget rather than a window, because it is both -- `ResultWindow` puts
    one in a window of its own, and the ROI site window puts one behind each
    evaluator's tab, where the outer tabs are who drew it and the inner ones
    what they drew.
    """

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self._subtitle = ""
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._on_tab)
        self.tabs.tabBarDoubleClicked.connect(self.detach)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.panes: Dict[str, FigurePane] = {}
        self.detached: Dict[str, QMainWindow] = {}
        self._names: List[str] = []
        self.size_hint_inches: Optional[Tuple[float, float]] = None
        detach = QAction("Open the current tab in its own window", self)
        detach.setShortcut("Ctrl+D")
        detach.triggered.connect(lambda: self.detach(self.tabs.currentIndex()))
        self.addAction(detach)

    @property
    def current(self) -> str:
        """The name of the plot being looked at."""
        pane = self.tabs.currentWidget()
        return next((n for n, p in self.panes.items() if p is pane), "")

    # ------------------------------------------------------------- showing
    def show_plots(self, plots: Sequence[Plot], subtitle: str = "") -> None:
        """Offer a new set of figures; draw only what is being looked at.

        The panes are kept across calls, keyed by the plot's name, so a
        plugin run twice or an ROI stepped past redraws in place.  A name
        that has gone takes its pane with it.
        """
        plots = list(plots)
        if len(plots) > 1:            # a page of all of them, drawn on demand
            plots.append(summary_plot(plots))
        self._subtitle = subtitle
        names = [plot.name for plot in plots]
        if names != self._names:
            self._rebuild(names)
        for plot in plots:
            self.panes[plot.name].offer(plot)
        for name, window in self.detached.items():
            if name in self.panes:                 # on screen: draw it now
                self.panes[name].draw()
                window.setWindowTitle(self._tab_title(name, subtitle))
        self._draw_current()
        self._mark_tabs()
        sizes = [plot.size for plot in plots if plot.size]
        if sizes:
            self.size_hint_inches = (max(w for w, _ in sizes),
                                     max(h for _, h in sizes))

    def _rebuild(self, names: Sequence[str]) -> None:
        """The set of figures changed: keep the panes that survive it."""
        while self.tabs.count():
            self.tabs.removeTab(0)
        kept = {name: pane for name, pane in self.panes.items() if name in names}
        for name, pane in self.panes.items():
            if name not in kept:
                pane.setParent(None)
                pane.deleteLater()
        self.panes = kept
        for name in names:
            if name not in self.panes:
                self.panes[name] = FigurePane(self)
            if name not in self.detached:
                self.tabs.addTab(self.panes[name], _label(name))
        self._names = list(names)
        self.tabs.tabBar().setVisible(self.tabs.count() > 1)

    def _draw_current(self) -> None:
        pane = self.tabs.currentWidget()
        if isinstance(pane, FigurePane) and self.isVisible():
            pane.draw()

    def dpi(self) -> float:
        pane = next(iter(self.panes.values()), None)
        return pane.figure.get_dpi() if pane is not None else 100.0

    def _on_tab(self, index: int) -> None:
        pane = self.tabs.widget(index)
        if isinstance(pane, FigurePane):
            pane.draw()
            self._mark_tabs()

    def _mark_tabs(self) -> None:
        """A tab that has not been drawn yet says so, greyed.

        Otherwise the last run's figure, still sitting in an untouched tab,
        reads as this run's -- which is worse than no figure at all.
        """
        for index in range(self.tabs.count()):
            pane = self.tabs.widget(index)
            if not isinstance(pane, FigurePane):
                continue
            colour = (Qt.GlobalColor.gray if pane.stale
                      else self.tabs.palette().text().color())
            self.tabs.tabBar().setTabTextColor(index, colour)

    def showEvent(self, event) -> None:            # noqa: N802 (Qt's name)
        super().showEvent(event)
        self._draw_current()
        self._mark_tabs()

    # ------------------------------------------------------------ tear-off
    def detach(self, index: int) -> None:
        """Take a tab out into a window of its own, to compare it with another.

        It stays live: the next result redraws it where it stands, and
        closing it hands the tab back.
        """
        pane = self.tabs.widget(index)
        if not isinstance(pane, FigurePane):
            return
        name = next((n for n, p in self.panes.items() if p is pane), None)
        if name is None or name in self.detached:
            return
        window = DetachedFigure(name, self._tab_title(name), self)
        window.closed.connect(self._reattach)
        self.tabs.removeTab(index)
        window.setCentralWidget(pane)
        window.resize(self.size())
        self.detached[name] = window
        pane.draw()
        window.show()
        self.tabs.tabBar().setVisible(self.tabs.count() > 1)

    def _reattach(self, name: str) -> None:
        """A torn-off window closed: its figure goes back into the tabs."""
        window = self.detached.pop(name, None)
        pane = self.panes.get(name)
        if window is None or pane is None:
            return
        window.takeCentralWidget()
        pane.setParent(self)
        index = min(self._names.index(name) if name in self._names else 0,
                    self.tabs.count())
        self.tabs.insertTab(index, pane, _label(name))
        self.tabs.tabBar().setVisible(self.tabs.count() > 1)
        self._mark_tabs()

    def _tab_title(self, name: str, subtitle: str = "") -> str:
        parts = [p for p in (self._title, subtitle or self._subtitle) if p]
        if name:
            parts.append(_label(name))
        return ": ".join(parts)

    def close_detached(self) -> None:
        """Whatever was torn off goes with the window it came from."""
        for window in list(self.detached.values()):
            window.close()


class ResultWindow(QMainWindow):
    """One result's figures in a window of their own.

    Reused for the life of whatever owns it -- a plugin's section -- so that
    running again redraws where the window already is, at the size it was
    given, rather than opening another one on top of it.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle(title)
        self._title = title
        self._sized = False
        self.figures = FigureTabs(title, self)
        self.setCentralWidget(self.figures)
        self.resize(640, 480)

    # the window is a thin skin over the tabs; the tests and the panel reach
    # through it rather than through a second set of forwarding methods
    @property
    def tabs(self):
        return self.figures.tabs

    @property
    def panes(self):
        return self.figures.panes

    @property
    def detached(self):
        return self.figures.detached

    def detach(self, index: int) -> None:
        self.figures.detach(index)

    def show_plots(self, plots: Sequence[Plot], subtitle: str = "") -> None:
        self.setWindowTitle(f"{self._title}: {subtitle}" if subtitle else self._title)
        self.figures.show_plots(plots, subtitle)
        self._take_size()

    def _take_size(self) -> None:
        """A plot's size hint sets the window, once.

        Only the first time: after that the window is where the user put it
        and at the size they gave it, and a three-panel figure is not a
        reason to undo that on every run.
        """
        hint = self.figures.size_hint_inches
        if self._sized or hint is None:
            return
        self._sized = True
        dpi = self.figures.dpi()
        self.resize(int(hint[0] * dpi), int(hint[1] * dpi))

    def closeEvent(self, event) -> None:           # noqa: N802 (Qt's name)
        self.figures.close_detached()
        super().closeEvent(event)
