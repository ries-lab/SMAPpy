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

from typing import Dict, List, Optional, Sequence

from PySide6.QtCore import Qt
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


class ResultWindow(QMainWindow):
    """Every figure of one result: one pane, or a tab each.

    Reused for the life of whatever owns it -- a plugin section, an ROI
    manager -- so that running again redraws where the window already is,
    at the size it was given, rather than opening another one on top of it.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle(title)
        self._title = title
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._on_tab)
        self.tabs.tabBarDoubleClicked.connect(self.detach)
        self.setCentralWidget(self.tabs)
        self.panes: Dict[str, FigurePane] = {}
        self.detached: Dict[str, QMainWindow] = {}
        self._names: List[str] = []
        self._sized = False
        detach = QAction("Open the current tab in its own window", self)
        detach.setShortcut("Ctrl+D")
        detach.triggered.connect(lambda: self.detach(self.tabs.currentIndex()))
        self.addAction(detach)
        self.resize(640, 480)

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
        self.setWindowTitle(f"{self._title}: {subtitle}" if subtitle else self._title)
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
        self._take_size(plots)

    def _take_size(self, plots: Sequence[Plot]) -> None:
        """A plot's size hint sets the window, once.

        Only the first time: after that the window is where the user put it
        and at the size they gave it, and a three-panel figure is not a
        reason to undo that on every run.
        """
        if self._sized:
            return
        sizes = [plot.size for plot in plots if plot.size]
        if not sizes:
            return
        self._sized = True
        dpi = next(iter(self.panes.values())).figure.get_dpi()
        self.resize(int(max(w for w, _ in sizes) * dpi),
                    int(max(h for _, h in sizes) * dpi))

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
        window = QMainWindow(self)
        window.setWindowFlag(Qt.Window, True)
        window.setWindowTitle(self._tab_title(name))
        self.tabs.removeTab(index)
        window.setCentralWidget(pane)
        window.resize(self.size())
        window.closeEvent = lambda event, n=name: self._reattach(n, event)
        self.detached[name] = window
        pane.draw()
        window.show()
        self.tabs.tabBar().setVisible(self.tabs.count() > 1)

    def _reattach(self, name: str, event) -> None:
        window = self.detached.pop(name, None)
        pane = self.panes.get(name)
        if window is not None and pane is not None:
            window.takeCentralWidget()
            pane.setParent(self)
            index = min(self._names.index(name) if name in self._names else 0,
                        self.tabs.count())
            self.tabs.insertTab(index, pane, _label(name))
            self.tabs.tabBar().setVisible(self.tabs.count() > 1)
            self._mark_tabs()
        event.accept()

    def _tab_title(self, name: str, subtitle: str = "") -> str:
        parts = [self._title]
        if subtitle:
            parts.append(subtitle)
        if name:
            parts.append(_label(name))
        return ": ".join(parts)

    def closeEvent(self, event) -> None:           # noqa: N802 (Qt's name)
        for window in list(self.detached.values()):
            window.close()
        super().closeEvent(event)


# ------------------------------------------------- one window per figure
#
# What the ROI manager still uses while its own window is built: a figure per
# plot, in a pyplot-managed window, reused by key.  It goes when the site
# window lands.

def draw_figure(cache: Dict[str, object], key: str, title: str, plot: Plot):
    """Draw `plot` into the pyplot figure `key` owns, and show it."""
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    figure = cache.get(key)
    if figure is not None and plt.fignum_exists(figure.number):
        figure.clear()
    else:
        figure = plt.figure()
        cache[key] = figure
    plot.draw_into(figure)
    figure.canvas.manager.set_window_title(title)
    figure.canvas.draw_idle()
    figure.show()
    return figure


def draw_figures(cache: Dict[str, object], plots: Sequence[Plot], title: str,
                 prefix: str = "") -> None:
    """Every figure of one result, one window each, named `title: plot`."""
    for plot in plots:
        draw_figure(cache, f"{prefix}{plot.name}",
                    f"{title}: {plot.name}" if plot.name else title, plot)
