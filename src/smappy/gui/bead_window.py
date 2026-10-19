"""The bead selector: a table of beads, their tracks, and the choice by hand.

One window for every plugin that chooses beads: it knows nothing of MBM or
fiducials.  It asks the plugin for a `beads.BeadView` (`Plugin.bead_view`:
the beads judged with the settings as they stand) and gives back the choices
made here (`Plugin.with_overrides`), which go into the plugin's form through
the panel that opened it (`PluginPanel.take_settings`), so a run, a saved
workspace and a chain all keep them.  Nothing is corrected here: the choice
is the plugin's settings, and its Run applies them.

The table is sorted by nothing but the beads' names, so a bead stays in its
row while the choice changes around it; clicking a row or a track selects a
bead, and its track is drawn over the others.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView,
                               QLabel, QMainWindow, QPushButton, QSplitter,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

# the beads' numbers as the table shows them: column title, metric, format
METRIC_COLUMNS = (("noise (nm)", "noise_nm", "{:.1f}"), ("wander (nm)", "wander_nm", "{:.1f}"),
                  ("jump (nm)", "jump_nm", "{:.1f}"), ("seen", "coverage", "{:.0%}"),
                  ("points", "n", "{:.0f}"))


class BeadWindow(QMainWindow):
    """The beads of one plugin's settings, to look at and choose by hand."""

    def __init__(self, session, plugin=None, settings=None,
                 reply: Optional[Callable] = None, centre: Optional[Callable] = None,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("bead selector")
        self.session = session
        self.view = None
        self.selected: Optional[int] = None
        self._centre = centre
        layout_widget = QWidget()
        layout = QVBoxLayout(layout_widget)
        top = QHBoxLayout()
        self.source = QComboBox()
        self.source.setObjectName("bead source")
        self.source.setToolTip("the measurement whose beads are shown")
        self.source.currentIndexChanged.connect(lambda _=0: self.refresh())
        top.addWidget(QLabel("beads of"))
        top.addWidget(self.source, 1)
        layout.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget()
        self.table.setObjectName("bead table")
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.itemSelectionChanged.connect(self._row_selected)
        left_layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        self.include = QPushButton("Include")
        self.include.setToolTip("use this bead whatever the rule says")
        self.exclude = QPushButton("Exclude")
        self.exclude.setToolTip("never use this bead (Space toggles)")
        self.automatic = QPushButton("Automatic")
        self.automatic.setToolTip("let the rule decide for this bead again")
        self.centre_button = QPushButton("Centre in view")
        self.centre_button.setToolTip("move the render view to this bead")
        self.include.clicked.connect(lambda: self.choose(True))
        self.exclude.clicked.connect(lambda: self.choose(False))
        self.automatic.clicked.connect(lambda: self.choose(None))
        self.centre_button.clicked.connect(self.centre)
        for b in (self.include, self.exclude, self.automatic, self.centre_button):
            buttons.addWidget(b)
        left_layout.addLayout(buttons)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        left_layout.addWidget(self.status)
        split.addWidget(left)

        from matplotlib.figure import Figure
        from .canvas import canvas_class
        self.figure = Figure(figsize=(8, 6), layout="constrained")
        self.canvas = canvas_class()(self.figure)
        self.canvas.mpl_connect("pick_event", self._picked)
        split.addWidget(self.canvas)
        split.setSizes([460, 760])
        layout.addWidget(split, 1)
        self.setCentralWidget(layout_widget)
        QShortcut(QKeySequence(Qt.Key_Space), self.table, self.toggle)
        self.resize(1250, 640)
        self.plugin = self.settings = self.reply = None
        self.setWindowTitle("Select beads")
        if plugin is not None:
            self.open_for(plugin, settings, reply)
        else:
            self._enable()

    # ----------------------------------------------------------- the plugin
    def open_for(self, plugin, settings, reply: Optional[Callable] = None) -> None:
        """Show ``plugin``'s beads with ``settings``; choices go to ``reply``."""
        self.plugin, self.settings, self.reply = plugin, settings, reply
        self.setWindowTitle(f"Select beads: {plugin.name or plugin.path}")
        self.refresh(reset_sources=True)

    def _context(self):
        return self.session.context(grouping=self.plugin.grouping)

    def refresh(self, reset_sources: bool = False) -> None:
        """Judge the beads again with the settings as they are, and redraw."""
        source = max(self.source.currentIndex(), 0)
        try:
            self.view = self.plugin.bead_view(self._context(), self.settings, source)
        except Exception as e:                # no beads in the selection, say so
            self.view = None
            self.status.setText(str(e))
            self.table.setRowCount(0)
            self.figure.clear()
            self.canvas.draw_idle()
            return
        if reset_sources or self.source.count() != len(self.view.sources):
            self.source.blockSignals(True)
            self.source.clear()
            self.source.addItems(self.view.sources)
            self.source.setCurrentIndex(self.view.source)
            self.source.blockSignals(False)
            self.selected = None
        if self.selected is not None and self.selected >= len(self.view.beads):
            self.selected = None
        self._fill_table()
        self._draw()

    # ------------------------------------------------------------ the table
    def _labels(self):
        v = self.view
        return v.labels if v.labels is not None else v.beads.names

    def _fill_table(self) -> None:
        v = self.view
        beads = v.beads
        extra = list(v.columns)
        headers = ["bead", "use"] + [c for c, _, _ in METRIC_COLUMNS] + extra + ["why"]
        self.table.blockSignals(True)
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(beads))
        labels = self._labels()
        for b in range(len(beads)):
            state = beads.state(b)
            use = "yes" if beads.used[b] else "no"
            if beads.names[b] in beads.overrides:
                use += " (by hand)"
            values = [labels[b], use]
            for _, metric, fmt in METRIC_COLUMNS:
                x = beads.metrics[metric][b]
                values.append(fmt.format(x) if np.isfinite(x) else "")
            for name in extra:
                x = v.columns[name][b]
                values.append("" if x is None else (f"{x:.1f}" if isinstance(x, float) else str(x)))
            values.append(state if not beads.used[b] or "hand" in state else "")
            grey = QBrush(QColor(150, 150, 150)) if not beads.used[b] else None
            for c, text in enumerate(values):
                item = QTableWidgetItem(text)
                if grey is not None:
                    item.setForeground(grey)
                self.table.setItem(b, c, item)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        if self.selected is not None:
            self.table.selectRow(self.selected)
        self.table.blockSignals(False)
        n = len(beads)
        self.status.setText(f"{int(beads.used.sum())} of {n} beads used"
                            + (f"; {int(beads.outlier.sum())} left out by the rule"
                               if beads.outlier.any() else ""))
        self._enable()

    def _enable(self) -> None:
        on = self.selected is not None
        for b in (self.include, self.exclude, self.automatic):
            b.setEnabled(on)
        self.centre_button.setEnabled(on and self._centre is not None
                                      and self.view is not None
                                      and self.view.where_nm is not None)

    def _row_selected(self) -> None:
        rows = {i.row() for i in self.table.selectedItems()}
        self.selected = min(rows) if rows else None
        self._enable()
        self._draw()

    def _picked(self, event) -> None:
        gid = event.artist.get_gid()
        if gid is None or self.view is None:
            return
        self.selected = int(gid)
        self.table.selectRow(self.selected)

    # ----------------------------------------------------------- the choice
    def choose(self, use: Optional[bool]) -> None:
        """Bead in (True), out (False), or back to the rule (None), into the
        plugin's settings and through ``reply`` into its form."""
        if self.selected is None or self.view is None:
            return
        beads = self.view.beads
        overrides = dict(beads.overrides)
        name = beads.names[self.selected]
        if use is None:
            overrides.pop(name, None)
        else:
            overrides[name] = use
        self.settings = self.plugin.with_overrides(self.settings, self.view, overrides)
        if self.reply is not None:
            self.reply(self.settings)
        self.refresh()

    def toggle(self) -> None:
        if self.selected is not None and self.view is not None:
            self.choose(not bool(self.view.beads.used[self.selected]))

    def centre(self) -> None:
        if self.selected is None or self._centre is None or self.view is None \
                or self.view.where_nm is None:
            return
        x, y = self.view.where_nm[self.selected]
        self._centre(float(x), float(y))

    # ----------------------------------------------------------- the figure
    def _draw(self) -> None:
        from ..beads import draw_bead_tracks
        self.figure.clear()
        if self.view is not None:
            draw_bead_tracks(self.figure, self.view.beads, self.view.axes,
                             selected=self.selected, labels=self._labels(),
                             time_unit=self.view.time_unit)
        self.canvas.draw_idle()
