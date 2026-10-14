"""Annotating ROIs in the ROI manager: a keyword from each of a few lists.

SMAP's ROI annotation (`SEAnnotation`): up to four lists, read from a text
file, a value of each per site, chosen with the arrow keys.  Here the lists
are written in the panel itself, one a line (``quality: good, bad``), as
many as are wanted, and saved with the project; a keyword is a key, 1 to 9
in the list that is active, so that going through a few hundred ROIs is a
key per ROI -- with *next ROI after a keyword*, not even the arrow.

Keys, in the whole window: 1-9 a keyword of the active list, 0 clears it,
Tab and Shift+Tab the next and previous list, Up and Down the next and
previous ROI, Space *use*.  A text field keeps its own keys: typing a 1 into
the lists is a 1.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QLabel, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)

from ..roi_manager.core import annotation_lists_text, parse_annotation_lists
from .widgets import CollapsibleSection

MAX_KEYS = 9                     # keywords per list a number key reaches


class AnnotationPanel(QWidget):
    """The selected ROI's keywords, *use*, and the lists they come from."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window             # the `ROIManagerWindow`
        self.active = 0                  # which list the number keys set
        self.rows: List[List[QPushButton]] = []
        self.titles: List[QLabel] = []
        self._built = None               # the lists the buttons were made for
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        top = QGridLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self.use = QCheckBox("use")
        self.use.setToolTip("include this ROI in the per-region steps and the "
                            "results (Space)")
        self.use.toggled.connect(self._on_use)
        self.advance = QCheckBox("next ROI after a keyword")
        self.advance.setToolTip("after a keyword is given, select the next ROI, "
                                "so a list is annotated a key per ROI")
        self.advance.toggled.connect(self._on_advance)
        top.addWidget(self.use, 0, 0)
        top.addWidget(self.advance, 0, 1)
        layout.addLayout(top)
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(2)
        self.grid.setVerticalSpacing(2)
        layout.addLayout(self.grid)
        self.keys = QLabel("1-9 keyword, 0 none, Tab next list, Up/Down next ROI")
        self.keys.setStyleSheet("color: gray")
        self.keys.setWordWrap(True)
        layout.addWidget(self.keys)
        self.lists = QPlainTextEdit()
        self.lists.setToolTip("one list a line: its name, a colon, its keywords "
                              "separated by commas -- quality: good, bad")
        self.lists.setFixedHeight(64)
        self.lists.setTabChangesFocus(True)
        self._apply_timer = QTimer(self, singleShot=True, interval=600)
        self._apply_timer.timeout.connect(self.apply_lists)
        self.lists.textChanged.connect(self._apply_timer.start)
        layout.addWidget(CollapsibleSection("lists", self.lists, expanded=False))

        for key in range(10):
            shortcut = QShortcut(QKeySequence(str(key)), window)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(lambda k=key: self.press(k))
        for key, step in ((Qt.Key_Tab, 1), (Qt.Key_Backtab, -1)):
            shortcut = QShortcut(QKeySequence(key), window)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(lambda s=step: self.next_list(s))
        self.refresh(lists=True)

    @property
    def project(self):
        return self.window.project

    # ------------------------------------------------------------ building
    def refresh(self, lists: bool = False) -> None:
        """The buttons from the project's lists, checked as the selected ROI
        has them; with ``lists`` the text of the lists too."""
        project = self.project
        if lists:
            blocked = self.lists.blockSignals(True)
            self.lists.setPlainText(annotation_lists_text(project.annotation_lists))
            self.lists.blockSignals(blocked)
        names = list(project.annotation_lists)
        shape = [(n, project.annotation_lists[n][:MAX_KEYS]) for n in names]
        if shape != self._built:
            self._build(names)
            self._built = shape
        self.active = min(self.active, max(len(names) - 1, 0))
        roi = self.window._selected()
        blocked = self.use.blockSignals(True)
        self.use.setChecked(bool(roi is not None and roi.use))
        self.use.setEnabled(roi is not None)
        self.use.blockSignals(blocked)
        blocked = self.advance.blockSignals(True)
        self.advance.setChecked(bool(project.navigation.get("annotate_advance")))
        self.advance.blockSignals(blocked)
        for i, (name, title, row) in enumerate(zip(names, self.titles, self.rows)):
            title.setText(("▶ " if i == self.active else "") + f"{name}:")
            title.setStyleSheet("font-weight: bold" if i == self.active else "")
            chosen = roi.annotations.get(name) if roi is not None else None
            for button in row:
                blocked = button.blockSignals(True)
                button.setChecked(button.property("keyword") == chosen)
                button.setEnabled(roi is not None)
                button.blockSignals(blocked)

    def _build(self, names: List[str]) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.titles, self.rows = [], []
        for r, name in enumerate(names):
            title = QLabel(f"{name}:")
            self.grid.addWidget(title, r, 0)
            row = []
            for k, keyword in enumerate(self.project.annotation_lists[name][:MAX_KEYS]):
                button = QPushButton(f"{k + 1} {keyword}")
                button.setCheckable(True)
                button.setProperty("keyword", keyword)
                button.setToolTip(f"{name}: {keyword} (key {k + 1} while this "
                                  "list is active)")
                button.setMinimumWidth(button.fontMetrics().horizontalAdvance(button.text()) + 12)
                button.clicked.connect(lambda on, i=r, w=keyword:
                                       self.choose(i, w if on else None))
                self.grid.addWidget(button, r, k + 1)
                row.append(button)
            self.titles.append(title)
            self.rows.append(row)
        self.grid.setColumnStretch(MAX_KEYS + 1, 1)

    # ------------------------------------------------------------- editing
    def choose(self, index: int, keyword: Optional[str]) -> None:
        """Give the selected ROI ``keyword`` from list ``index``, or none."""
        roi = self.window._selected()
        names = list(self.project.annotation_lists)
        if roi is None or not 0 <= index < len(names):
            return
        self.active = index
        self.project.annotate(roi.id, names[index], keyword)
        self.window.annotated()
        if keyword and self.project.navigation.get("annotate_advance"):
            self.window.step_roi(1)
        self.refresh()

    def press(self, key: int) -> None:
        """A number key: keyword ``key`` of the active list, 0 none."""
        names = list(self.project.annotation_lists)
        if not names:
            return
        words = self.project.annotation_lists[names[self.active]]
        if key == 0:
            self.choose(self.active, None)
        elif key <= min(len(words), MAX_KEYS):
            self.choose(self.active, words[key - 1])

    def next_list(self, step: int) -> None:
        n = len(self.project.annotation_lists)
        if n:
            self.active = (self.active + step) % n
            self.refresh()

    def apply_lists(self) -> None:
        """The lists as typed, into the project."""
        lists = parse_annotation_lists(self.lists.toPlainText())
        if lists != self.project.annotation_lists:
            self.project.set_annotation_lists(lists)
            self.window.annotated()
            self.refresh()

    def _on_use(self, on: bool) -> None:
        roi = self.window._selected()
        if roi is not None:
            self.window._set_use(roi.id, on)

    def _on_advance(self, on: bool) -> None:
        self.project.navigation["annotate_advance"] = bool(on)
