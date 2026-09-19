"""What the evaluators drew for the site being looked at.

Outer tabs are who drew it, inner ones what they drew -- SMAP's site explorer
arranged the same way, and for the same reason: a pipeline of five evaluators
with three figures each is fifteen windows, which is not a thing anyone can
step a list of sites through.

The laziness is what makes it work.  Only the evaluator whose tab is open is
run for the site, and only the figure whose tab is open inside it is drawn;
picking another evaluator asks for that one, and the rest are never computed
at all.  The numbers are a different matter -- they are stored, so they come
from the last run and cost nothing, which is what lets a list be scrolled.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QMainWindow, QStackedWidget, QTabWidget

from ..plugins import Plot
from .figures import FigureTabs

FAILED_COLOUR = "#c07000"


class SiteWindow(QMainWindow):
    """One window for the selected ROI: a tab per evaluator.

    It asks rather than computes: `shown` says which evaluator is being
    looked at, and whoever owns the window runs that step and hands back its
    figures.  The window never runs anything itself, which is what keeps the
    policy -- what to re-evaluate, and when -- in one place.
    """

    shown = Signal(str)          # this evaluator's tab is open; draw it
    closed = Signal()            # nobody is looking: stop measuring for it

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("ROI evaluation")
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._on_tab)
        self.setCentralWidget(self.tabs)
        self.pages: Dict[str, QStackedWidget] = {}
        self.figures: Dict[str, FigureTabs] = {}
        self.messages: Dict[str, QLabel] = {}
        self._labels: List[str] = []
        self.resize(760, 560)

    @property
    def current(self) -> str:
        """The evaluator being looked at."""
        index = self.tabs.currentIndex()
        return self._labels[index] if 0 <= index < len(self._labels) else ""

    def set_steps(self, labels: Sequence[str]) -> None:
        """One tab per step of the pipeline, keeping the ones that survive."""
        labels = list(labels)
        if labels == self._labels:
            return
        keep = self.tabs.currentIndex()
        while self.tabs.count():
            self.tabs.removeTab(0)
        for label in list(self.figures):
            if label not in labels:
                self.figures.pop(label).setParent(None)
                self.messages.pop(label, None)
                self.pages.pop(label).setParent(None)
        for label in labels:
            if label not in self.pages:
                page = QStackedWidget(self)
                figures = FigureTabs(label, page)
                message = QLabel("", alignment=Qt.AlignCenter)
                message.setWordWrap(True)
                message.setStyleSheet("color: gray")
                page.addWidget(figures)
                page.addWidget(message)
                self.pages[label] = page
                self.figures[label] = figures
                self.messages[label] = message
            self.tabs.addTab(self.pages[label], label)
        self._labels = labels
        self.tabs.tabBar().setVisible(len(labels) > 1)
        if 0 <= keep < self.tabs.count():
            self.tabs.setCurrentIndex(keep)

    def show_plots(self, label: str, plots: Sequence[Plot], subtitle: str = "") -> None:
        """What this evaluator drew for the site now being looked at."""
        figures = self.figures.get(label)
        if figures is None:
            return
        self.pages[label].setCurrentWidget(figures)
        self._mark(label, ok=True)
        figures.show_plots(plots, subtitle)

    def show_message(self, label: str, text: str, failed: bool = False) -> None:
        """No figure from this evaluator, and why -- where the figure was."""
        message = self.messages.get(label)
        if message is None:
            return
        message.setText(text)
        self.pages[label].setCurrentWidget(message)
        self._mark(label, ok=not failed)

    def title_for(self, subtitle: str) -> None:
        self.setWindowTitle(f"ROI evaluation: {subtitle}" if subtitle
                            else "ROI evaluation")

    def _mark(self, label: str, ok: bool) -> None:
        """An evaluator that failed says so on its tab, not only inside it."""
        if label not in self._labels:
            return
        index = self._labels.index(label)
        self.tabs.tabBar().setTabTextColor(
            index, self.tabs.palette().text().color() if ok
            else QColor(FAILED_COLOUR))

    def _on_tab(self, index: int) -> None:
        if 0 <= index < len(self._labels):
            self.shown.emit(self._labels[index])

    def closeEvent(self, event) -> None:           # noqa: N802 (Qt's name)
        for figures in self.figures.values():
            figures.close_detached()
        self.closed.emit()
        super().closeEvent(event)
