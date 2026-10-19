"""The Regions tab's *region filter*: conditions on what was measured per region.

The filter is `smappy.region_filter` (the rules: per region, OR within a
source, AND across sources, out where there is no region); this is the hand
editor of its conditions, which *Filter regions* sets in a chain.  From the
top: the conditions applied, each with what it keeps and a cross; a tree of
everything that can be filtered -- source, evaluation, value, and a level
more where a plugin hands back statistics of one quantity
(`RegionTable.groups`) -- with a search box, since a few evaluations make a
hundred values; and the chosen value's distribution over the regions, with
the range to keep, or for a keyword the texts to accept.

The histogram is the Render tab's, on region values: a bound dragged to the
edge is no bound.  The regions counted are all of them, not only those the
filter keeps -- the bounds are set on what there is.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QToolButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..chain import BLOCK_ROIS
from ..region_filter import Condition, region_values, sources

HIST_BINS = 60
ROLE = Qt.UserRole


def _numeric(values) -> Optional[np.ndarray]:
    """The values as floats, or None when they are text or flags."""
    present = [v for v in values if v is not None]
    if not present or any(isinstance(v, (str, bool)) for v in present):
        return None
    out = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    return out


class RegionFilterPanel(QWidget):
    """The conditions, the tree of values and the chosen value's histogram."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        from .render_tab import _Bound
        self.session = session
        self.chosen: Optional[tuple] = None          # (source, label, value)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.applied = QVBoxLayout()
        self.applied.setSpacing(1)
        layout.addLayout(self.applied)
        self.none = QLabel("no conditions: every localization is shown")
        self.none.setStyleSheet("color: gray")
        layout.addWidget(self.none)
        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        self.warnings.setStyleSheet("color: #c06000")
        layout.addWidget(self.warnings)
        self.clear_all = QPushButton("clear all")
        self.clear_all.setToolTip("remove every condition")
        self.clear_all.clicked.connect(lambda: self.session.set_region_filter([]))
        layout.addWidget(self.clear_all, 0, Qt.AlignLeft)

        self.search = QLineEdit()
        self.search.setPlaceholderText("search the values")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter_tree)
        layout.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setFixedHeight(150)
        self.tree.setToolTip("the values measured per region: choose one to filter on")
        self.tree.currentItemChanged.connect(self._on_item)
        layout.addWidget(self.tree)

        self.what = QLabel("")
        self.what.setWordWrap(True)
        self.what.setStyleSheet("color: gray")
        layout.addWidget(self.what)
        self.plot = pg.PlotWidget(background=None)
        self.plot.setFixedHeight(100)
        self.plot.setMenuEnabled(False)
        self.plot.hideAxis("left")
        self.plot.setMouseEnabled(x=True, y=False)
        self.bars = pg.PlotCurveItem([0.0, 1.0], [0.0], pen=None,
                                     brush=(120, 120, 120, 160), fillLevel=0,
                                     stepMode="center")
        self.plot.addItem(self.bars)
        self.region = pg.LinearRegionItem(brush=(60, 120, 220, 50))
        self.region.sigRegionChangeFinished.connect(self._on_region)
        self.plot.addItem(self.region)
        layout.addWidget(self.plot)
        self.texts = QListWidget()
        self.texts.setFixedHeight(90)
        self.texts.setToolTip("tick the texts to accept")
        layout.addWidget(self.texts)
        row = QHBoxLayout()
        self.lo, self.hi = _Bound(), _Bound()
        row.addWidget(QLabel("min"))
        row.addWidget(self.lo)
        row.addWidget(QLabel("max"))
        row.addWidget(self.hi)
        self.add = QPushButton("add")
        self.add.setToolTip("filter on this value: a region must pass every "
                            "condition on its source")
        self.add.clicked.connect(self._on_add)
        row.addWidget(self.add)
        row.addStretch(1)
        layout.addLayout(row)
        self._edges = (0.0, 1.0)
        self._text_mode = False            # a keyword's texts, not a histogram
        self._show_value(None)
        session.on_change(self._on_session)
        self.refresh()

    # ------------------------------------------------------------ session
    def _on_session(self, what: str) -> None:
        if what in ("locs", "rois", "region_filter"):
            self.refresh()

    def refresh(self) -> None:
        self._show_applied()
        self._fill_tree()

    def _show_applied(self) -> None:
        while self.applied.count():
            item = self.applied.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        found = self.session.region_filter
        outcome = self.session.region_filter_outcome()
        for k, condition in enumerate(found):
            kept, total = outcome.counts[k] if k < len(outcome.counts) else (0, 0)
            line = QWidget()
            h = QHBoxLayout(line)
            h.setContentsMargins(0, 0, 0, 0)
            text = QPushButton(f"{condition.text()}   {kept} / {total}")
            text.setFlat(True)
            text.setStyleSheet("text-align: left")
            text.setToolTip("show this condition's value below")
            text.clicked.connect(lambda _=False, c=condition: self._select(c))
            h.addWidget(text, 1)
            remove = QToolButton(text="×")
            remove.setToolTip("remove this condition")
            remove.clicked.connect(lambda _=False, k=k: self._remove(k))
            h.addWidget(remove)
            self.applied.addWidget(line)
        self.none.setVisible(not found)
        self.clear_all.setVisible(bool(found))
        self.warnings.setText("\n".join(outcome.warnings))
        self.warnings.setVisible(bool(outcome.warnings))

    def _fill_tree(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        for source, labels in sources(self.session).items():
            top = QTreeWidgetItem(["ROIs" if source == BLOCK_ROIS else f"per {source}"])
            self.tree.addTopLevelItem(top)
            for label, found in labels.items():
                node = QTreeWidgetItem([label])
                top.addChild(node)
                grouped = set()
                for group, members in sorted(found["groups"].items()):
                    gnode = QTreeWidgetItem([group])
                    node.addChild(gnode)
                    for member, name in members.items():
                        leaf = QTreeWidgetItem([member])
                        leaf.setData(0, ROLE, (source, label, name))
                        leaf.setToolTip(0, name)
                        gnode.addChild(leaf)
                        grouped.add(name)
                for name in found["values"]:
                    if name not in grouped:
                        leaf = QTreeWidgetItem([name])
                        leaf.setData(0, ROLE, (source, label, name))
                        node.addChild(leaf)
            top.setExpanded(True)
        self.tree.blockSignals(False)
        self._filter_tree(self.search.text())
        if self.chosen is not None:
            self._show_value(self.chosen)

    def _filter_tree(self, text: str) -> None:
        """Show what matches ``text``: a value by its name, an evaluation or
        a source by its own (then all of it), and the branches leading there."""
        text = text.strip().lower()

        def visit(item, inherited: bool) -> bool:
            own = not text or inherited or text in item.text(0).lower()
            if item.childCount() == 0:
                visible = own
            else:
                below = [visit(item.child(i), own and bool(text) or inherited)
                         for i in range(item.childCount())]
                visible = any(below) or (own and not text)
            item.setHidden(not visible)
            if text and visible and item.childCount():
                item.setExpanded(True)
            return visible

        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i), False)

    # ------------------------------------------------------------ a value
    def _on_item(self, item, _previous=None) -> None:
        data = None if item is None else item.data(0, ROLE)
        if data is not None:
            self._show_value(tuple(data))

    def _select(self, condition: Condition) -> None:
        self._show_value((condition.source, condition.label, condition.value), condition)

    def _existing(self, chosen) -> Optional[int]:
        for k, c in enumerate(self.session.region_filter):
            if (c.source, c.label, c.value) == tuple(chosen):
                return k
        return None

    def _show_value(self, chosen, condition: Optional[Condition] = None) -> None:
        self.chosen = chosen
        has = chosen is not None
        for w in (self.plot, self.texts, self.lo, self.hi, self.add):
            w.setEnabled(has)
        if not has:
            self.what.setText("choose a value above")
            self.texts.hide()
            self._text_mode = False
            return
        source, label, name = chosen
        ids, values = region_values(self.session, source, label)
        column = values.get(name, [])
        k = self._existing(chosen)
        if condition is None and k is not None:
            condition = self.session.region_filter[k]
        self.add.setText("update" if k is not None else "add")
        numbers = _numeric(column)
        where = "ROIs" if source == BLOCK_ROIS else f"per {source}"
        self.what.setText(f"{where} › {label} › {name}: {len(ids)} regions")
        self._text_mode = numbers is None
        if numbers is None:
            self.plot.hide()
            self.lo.setEnabled(False)
            self.hi.setEnabled(False)
            self.texts.show()
            self.texts.clear()
            accept = set(condition.accept or []) if condition is not None else set()
            from ..region_filter import _as_text
            for text in sorted({_as_text(v) for v in column if v is not None}):
                item = QListWidgetItem(text)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if text in accept else Qt.Unchecked)
                self.texts.addItem(item)
            return
        self.texts.hide()
        self.plot.show()
        finite = numbers[np.isfinite(numbers)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if hi <= lo:
                lo, hi = lo - 0.5, hi + 0.5
            counts, edges = np.histogram(finite, bins=HIST_BINS, range=(lo, hi))
        else:
            counts, edges = np.zeros(1), np.array([0.0, 1.0])
        self._edges = (float(edges[0]), float(edges[-1]))
        self.bars.setData(edges, counts.astype(np.float64))
        low = None if condition is None else condition.lo
        high = None if condition is None else condition.hi
        self.lo.set(low)
        self.hi.set(high)
        self.region.blockSignals(True)
        self.region.setRegion((self._edges[0] if low is None else low,
                               self._edges[1] if high is None else high))
        self.region.blockSignals(False)
        self.plot.setXRange(*self._edges, padding=0.02)

    def _on_region(self) -> None:
        lo, hi = self.region.getRegion()
        # as in the Render tab: an end at the edge is no bound
        self.lo.set(None if lo <= self._edges[0] else float(lo))
        self.hi.set(None if hi >= self._edges[1] else float(hi))

    # ------------------------------------------------------------ editing
    def condition(self) -> Optional[Condition]:
        """The condition the panel describes now, or None for none."""
        if self.chosen is None:
            return None
        source, label, name = self.chosen
        if self._text_mode:
            accept = [self.texts.item(i).text() for i in range(self.texts.count())
                      if self.texts.item(i).checkState() == Qt.Checked]
            return Condition(source, label, name, accept=accept)
        try:
            lo, hi = self.lo.value(), self.hi.value()
        except ValueError:
            return None
        if lo is None and hi is None:
            return None
        return Condition(source, label, name, lo, hi)

    def _on_add(self) -> None:
        condition = self.condition()
        if condition is None:
            self.what.setText("set a range (min, max or both), or tick a text, first")
            return
        found = list(self.session.region_filter)
        k = self._existing(self.chosen)
        if k is None:
            found.append(condition)
        else:
            found[k] = condition
        self.session.set_region_filter(found)

    def _remove(self, k: int) -> None:
        found = list(self.session.region_filter)
        if 0 <= k < len(found):
            del found[k]
            self.session.set_region_filter(found)
