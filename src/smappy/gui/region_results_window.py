"""The region results window: what was measured per region, as one table.

A row per region -- an ROI, or a region of an id column (a cluster, a track)
-- and a column per value each evaluation measured there, so that the
numbers the per-region steps keep can be read, sorted and compared in one
place: before it the Regions tab showed one ROI's values at a time, and an
id column's regions none at all.

The header's right click colours the localizations by a value
(`region_colour`, the Render tab's *region value*) or opens the region
filter on it; a click on a row shows the region -- an ROI in the ROI
manager, a region of an id column centred in the view.  The table is the
region tables as they stand (`region_filter.region_values`, what the filter
and the colours read), rebuilt when they change; a value whose result is out
of date is drawn in the warning colour.

It is also the GUI's export of the table, which *Export table* was (that
plugin stays for chains and batches, `chain_only`): *Save...* and *Copy*
write the columns ticked under *select* -- a tick in each header, with
*all* and *none*, shown only while choosing, so that a table of thirty
values is narrowed in a few clicks and the choice is out of the way
otherwise -- in the table's order, the numbers exact (the cells show four
digits), a stale value left empty.  For the ROIs it adds each ROI's centre,
file and comment (`POSITION`), what is needed to find an ROI again from a
row of a spreadsheet, and *used ROIs only* (on) keeps the ROIs whose *use*
is ticked, the ones every summary reads.
"""
from __future__ import annotations

import csv
import io
import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QMenu, QPushButton, QTableView,
                               QVBoxLayout, QWidget)

from ..chain import BLOCK_ROIS as ROIS
from .widgets import note_style

#: the role the proxy sorts by: a number where there is one
SORT_ROLE = Qt.UserRole + 1
#: a value whose result is out of date (`CLAUDE.md`: reads on light and dark)
STALE = "#c06000"
#: the label of the columns an ROI has of its own, not measured: its place,
#: its file and its comment (*Export table*'s *ROI position*)
POSITION = "ROI"
POSITION_VALUES = ("x_nm", "y_nm", "file", "comment")


# ---------------------------------------------------------- the numbers, Qt-free

def results_table(session, source: str, used_only: bool = False):
    """``(regions, names, columns, values, stale)`` of one source: each
    region's id and its name in the first column, the ``(label, value)`` of
    every column, the values as rows (None where not measured), and the
    ``(row, column)`` cells whose result is out of date.  For the ROIs the
    first columns are their own (`POSITION`), and ``used_only`` keeps the
    ROIs with *use* ticked."""
    from ..region_filter import ANNOTATIONS, region_values, sources
    from ..region_results import Evaluation, is_current
    labels = sources(session).get(source) or {}
    if source == ROIS:
        project = session.rois
        regions = [r for r in project.ordered_ids()
                   if not used_only or project.rois[r].use]
        numbers = project.numbers()
        names = [str(numbers.get(r, "")) for r in regions]
    else:
        found = set()
        for label in labels:
            found.update(region_values(session, source, label)[0])
        regions = sorted(found, key=_id_key)
        names = [str(r) for r in regions]
    row_of = {r: i for i, r in enumerate(regions)}
    columns: List[Tuple[str, str]] = []
    values: List[List] = [[] for _ in regions]
    stale = set()
    ctx = session.context()
    for label in labels:
        ids, by_value = region_values(session, source, label)
        old_rows = set()
        if source == ROIS and label != ANNOTATIONS:
            project = session.rois
            for roi_id in ids:
                state = (project.latest(roi_id)[1] or {}).get(label)
                if state not in (None, "current") and roi_id in row_of:
                    old_rows.add(row_of[roi_id])
        elif source != ROIS and not is_current(ctx, Evaluation(source, label)):
            old_rows = set(range(len(regions)))
        for value in sorted(by_value):
            k = len(columns)
            columns.append((label, value))
            column = [None] * len(regions)
            for region, v in zip(ids, by_value[value]):
                if region in row_of:
                    column[row_of[region]] = v
            for i, v in enumerate(column):
                values[i].append(v)
                if v is not None and i in old_rows:
                    stale.add((i, k))
    if source == ROIS:
        own = [_own_values(session.rois, r) for r in regions]
        columns = [(POSITION, v) for v in POSITION_VALUES] + columns
        values = [o + row for o, row in zip(own, values)]
        stale = {(i, k + len(POSITION_VALUES)) for i, k in stale}
    return regions, names, columns, values, stale


def _own_values(project, roi_id) -> list:
    """An ROI's centre, its file's name and its comment (`POSITION_VALUES`)."""
    roi = project.rois[roi_id]
    source = project.sources.get(roi.file_id)
    return [float(roi.center[0]), float(roi.center[1]),
            source.name if source is not None else "", roi.comment or ""]


def exact_text(value) -> str:
    """A value as a file takes it: a number to every digit it has (the
    table shows four), NaN empty."""
    if isinstance(value, (float, np.floating)):
        return "" if math.isnan(value) else repr(float(value))
    return as_text(value)


def _id_key(region):
    try:
        return (0, float(region), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(region))


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def _sort_key(value):
    """Numbers before texts, missing last whichever way it is sorted up."""
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        v = float(value)
        return v if not math.isnan(v) else None
    return None if value is None else str(value)


def as_csv(header: Sequence[str], rows: Sequence[Sequence[str]], delimiter=",") -> str:
    out = io.StringIO()
    writer = csv.writer(out, delimiter=delimiter, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue()


# ---------------------------------------------------------------- the model

class ResultsModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.source = ROIS
        self.regions: list = []
        self.names: list = []
        self.columns: list = []
        self.values: list = []
        self.stale: set = set()
        self._stale_brush = QColor(STALE)
        # *select*: a tick in each header; the columns left out, by
        # (label, value), so that a column a later step adds is in
        self.selecting = False
        self.left_out: set = set()

    def fill(self, session, source: str, used_only: bool = False) -> None:
        self.beginResetModel()
        self.source = source
        (self.regions, self.names, self.columns, self.values,
         self.stale) = results_table(session, source, used_only)
        self.endResetModel()

    def picked(self, column: int) -> bool:
        return column == 0 or self.columns[column - 1] not in self.left_out

    def pick(self, columns, on: bool) -> None:
        for c in columns:
            if c > 0:
                key = self.columns[c - 1]
                (self.left_out.discard if on else self.left_out.add)(key)
        if self.columns:
            self.headerDataChanged.emit(Qt.Horizontal, 0, len(self.columns))
            self.dataChanged.emit(self.index(0, 0),
                                  self.index(max(0, len(self.regions) - 1), len(self.columns)))

    def header_text(self, column: int) -> str:
        if column == 0:
            return "ROI" if self.source == ROIS else self.source
        label, value = self.columns[column - 1]
        return f"{value}\n{label}"

    def rowCount(self, parent=QModelIndex()) -> int:          # noqa: N802 (Qt's)
        return 0 if parent.isValid() else len(self.regions)

    def columnCount(self, parent=QModelIndex()) -> int:       # noqa: N802
        return 0 if parent.isValid() else 1 + len(self.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if orientation != Qt.Horizontal:
            return None
        if role == Qt.DisplayRole:
            text = self.header_text(section)
            if self.selecting and section > 0:
                text = ("\u2611 " if self.picked(section) else "\u2610 ") + text
            return text
        if role == Qt.ForegroundRole and self.selecting and not self.picked(section):
            return _left_out_brush()
        if role == Qt.ToolTipRole and self.selecting and section > 0:
            return "click: tick or untick this column for Copy and Save"
        if role == Qt.ToolTipRole and section > 0 and self.columns[section - 1][0] == POSITION:
            return "the ROI's own, not measured"
        if role == Qt.ToolTipRole and section > 0:
            label, value = self.columns[section - 1]
            return (f"{value}, measured by {label}.  Right-click: colour the "
                    "localizations by it, or filter the regions on it")
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row, column = index.row(), index.column()
        if column == 0:
            if role == Qt.DisplayRole:
                return self.names[row]
            if role == SORT_ROLE:
                return _id_key(self.names[row])
            return None
        value = self.values[row][column - 1]
        if role == Qt.DisplayRole:
            return as_text(value)
        if role == SORT_ROLE:
            return _sort_key(value)
        if role == Qt.ForegroundRole and self.selecting and not self.picked(column):
            return _left_out_brush()
        if role == Qt.ForegroundRole and (row, column - 1) in self.stale:
            return self._stale_brush
        if role == Qt.ToolTipRole and (row, column - 1) in self.stale:
            return "out of date: measured before the region or the step changed"
        return None


def _left_out_brush() -> QColor:
    """A column left out while choosing: the palette's disabled text, which
    reads on light and dark windows."""
    return QGuiApplication.palette().color(QPalette.Disabled, QPalette.Text)


class _Proxy(QSortFilterProxyModel):
    """Sorts numbers as numbers, missing values last."""

    def lessThan(self, left, right) -> bool:                  # noqa: N802
        a, b = left.data(SORT_ROLE), right.data(SORT_ROLE)
        if a is None or b is None:
            # missing last both ways: "less" flips with the order
            up = self.sortOrder() == Qt.AscendingOrder
            return (a is not None) if up else (a is None and b is not None)
        if type(a) is not type(b):
            return isinstance(a, float)
        return a < b


# --------------------------------------------------------------- the window

class RegionResultsWindow(QWidget):
    """``colour_by(field)`` colours the Render tab's layer by a region value,
    ``filter_on(source, label, value)`` opens the region filter on one,
    ``show_roi(roi_id)`` and ``center_on(x, y)`` show a region; any may be
    None (a test, or a window without them), and its menu entry is then
    left out."""

    def __init__(self, session, parent=None, *,
                 colour_by: Optional[Callable[[str], None]] = None,
                 filter_on: Optional[Callable[[str, str, str], None]] = None,
                 show_roi: Optional[Callable[[str], None]] = None,
                 center_on: Optional[Callable[[float, float], None]] = None):
        super().__init__(parent, Qt.Window)
        from .help_window import attach_help
        attach_help(self, "Panels/Region results")
        self.setWindowTitle("Region results")
        self.session = session
        self.colour_by, self.filter_on = colour_by, filter_on
        self.show_roi, self.center_on = show_roi, center_on

        self.source = QComboBox()
        self.source.setToolTip("whose regions: the ROIs, or an id column's "
                               "regions (clusters, tracks)")
        self.count = QLabel("")
        self.count.setStyleSheet(note_style())
        self.copy = QPushButton("Copy")
        self.copy.setToolTip("the table to the clipboard, tab-separated, for a "
                             "spreadsheet")
        self.save = QPushButton("Save...")
        self.save.setToolTip("the table as a CSV file, the numbers exact -- for "
                             "statistics and figures made elsewhere")
        self.used = QCheckBox("used ROIs only")
        self.used.setChecked(True)
        self.used.setToolTip("the ROIs whose use is ticked, which every summary "
                             "reads; off: every ROI")
        self.select = QPushButton("select")
        self.select.setCheckable(True)
        self.select.setToolTip("choose the columns Copy and Save write: a tick in "
                               "each header, click to tick or untick")
        top = QHBoxLayout()
        top.addWidget(QLabel("regions"))
        top.addWidget(self.source, 1)
        top.addWidget(self.count)
        top.addWidget(self.used)
        top.addWidget(self.select)
        top.addWidget(self.copy)
        top.addWidget(self.save)
        # shown while choosing, gone otherwise
        self.select_row = QWidget()
        pick = QHBoxLayout(self.select_row)
        pick.setContentsMargins(0, 0, 0, 0)
        for text, on in (("all", True), ("none", False)):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, on=on: self.pick_all(on))
            pick.addWidget(button)
        self.select_note = QLabel("")
        self.select_note.setStyleSheet(note_style())
        pick.addWidget(self.select_note, 1)
        self.select_row.hide()

        self.model = ResultsModel()
        self.proxy = _Proxy(self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.horizontalHeader().customContextMenuRequested.connect(self._header_menu)
        self.table.clicked.connect(self._on_row)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.select_row)
        layout.addWidget(self.table, 1)
        self.resize(720, 480)

        self.source.currentIndexChanged.connect(self._on_source)
        self.used.toggled.connect(lambda _: self._fill())
        self.select.toggled.connect(self.set_selecting)
        self.table.horizontalHeader().sectionClicked.connect(self._on_header_click)
        self.copy.clicked.connect(lambda: self.copy_table())
        self.save.clicked.connect(self._save)
        session.on_change(self._on_session)
        self.destroyed.connect(lambda *_: session.off_change(self._on_session))
        self.refresh()

    # ------------------------------------------------------------ contents
    def sources(self) -> List[str]:
        from ..region_filter import sources
        return list(sources(self.session))

    def show_source(self, source: str) -> None:
        """Show ``source``'s regions (ROIS, or an id column), if it has any."""
        k = self.source.findData(source)
        if k >= 0:
            self.source.setCurrentIndex(k)

    def refresh(self) -> None:
        chosen = self.source.currentData()
        found = self.sources()
        self.source.blockSignals(True)
        self.source.clear()
        for source in found:
            self.source.addItem("ROIs" if source == ROIS else f"per {source}", source)
        k = self.source.findData(chosen)
        self.source.setCurrentIndex(k if k >= 0 else 0)
        self.source.blockSignals(False)
        self._fill()

    def _fill(self) -> None:
        source = self.source.currentData()
        header = self.table.horizontalHeader()
        order = (header.sortIndicatorSection(), header.sortIndicatorOrder())
        if source is None:
            self.model.beginResetModel()
            self.model.regions, self.model.names = [], []
            self.model.columns, self.model.values, self.model.stale = [], [], set()
            self.model.endResetModel()
            self.count.setText("nothing measured per region yet")
        else:
            self.model.fill(self.session, source, used_only=self.used.isChecked())
            n = len(self.model.regions)
            self.count.setText(f"{n} region{'s' if n != 1 else ''}")
        self.used.setVisible(source == ROIS)
        self._say_picked()
        self.table.resizeColumnsToContents()
        if 0 <= order[0] < self.model.columnCount():
            self.table.sortByColumn(*order)
        for w in (self.copy, self.save):
            w.setEnabled(bool(self.model.regions))

    def _on_source(self) -> None:
        self._fill()

    def _on_session(self, what: str) -> None:
        if what in ("locs", "rois") and self.isVisible():
            self.refresh()

    def showEvent(self, event) -> None:                         # noqa: N802
        self.refresh()          # what changed while it was hidden
        super().showEvent(event)

    # ------------------------------------------------------------- columns
    def field_of(self, column: int) -> Optional[str]:
        """The colour field of a column (`region_colour.name`); None for the
        regions' own."""
        from ..region_colour import name
        if column <= 0 or column > len(self.model.columns):
            return None
        label, value = self.model.columns[column - 1]
        return name(self.model.source, label, value)

    def numeric(self, column: int) -> bool:
        from ..region_colour import _is_number
        return any(_is_number(row[column - 1]) for row in self.model.values
                   if row[column - 1] is not None)

    def _header_menu(self, at) -> None:
        column = self.table.horizontalHeader().logicalIndexAt(at)
        menu = self.header_menu(column)
        if menu is not None:
            menu.exec(self.table.horizontalHeader().mapToGlobal(at))

    def header_menu(self, column: int) -> Optional[QMenu]:
        """The right-click menu of a column's header; built, not shown."""
        if column <= 0:
            return None
        label, value = self.model.columns[column - 1]
        menu = QMenu(self)
        if label == POSITION:              # the ROI's own: nothing to colour by
            menu.addAction("Copy this column", lambda: self.copy_table([column]))
            return menu
        if self.colour_by is not None:
            act = menu.addAction("Colour localizations by this",
                                 lambda: self.colour_by(self.field_of(column)))
            act.setEnabled(self.numeric(column))
        if self.filter_on is not None:
            menu.addAction("Filter regions on this...",
                           lambda: self.filter_on(self.model.source, label, value))
        menu.addAction("Copy this column", lambda: self.copy_table([column]))
        return menu

    # ------------------------------------------------------------- select
    def set_selecting(self, on: bool) -> None:
        """*select*: the headers show ticks and a click ticks or unticks a
        column instead of sorting by it; *all* and *none* appear."""
        on = bool(on)
        if self.select.isChecked() != on:
            self.select.setChecked(on)          # comes back here
            return
        self.model.selecting = on
        self.select_row.setVisible(on)
        # a click on a header ticks while choosing; sorting resumes after
        self.table.setSortingEnabled(not on)
        self.model.headerDataChanged.emit(Qt.Horizontal, 0, self.model.columnCount() - 1)
        self.table.viewport().update()
        self.table.resizeColumnsToContents()
        self._say_picked()

    def _on_header_click(self, column: int) -> None:
        if self.model.selecting and column > 0:
            self.model.pick([column], not self.model.picked(column))
            self._say_picked()

    def pick_all(self, on: bool) -> None:
        self.model.pick(range(1, self.model.columnCount()), on)
        self._say_picked()

    def chosen_columns(self) -> List[int]:
        """The columns Copy and Save write: the region's, and those ticked."""
        return [c for c in range(self.model.columnCount()) if self.model.picked(c)]

    def _say_picked(self) -> None:
        n, of = len(self.chosen_columns()) - 1, self.model.columnCount() - 1
        self.select_note.setText(f"{max(n, 0)} of {max(of, 0)} columns to copy and save")
        tip = "" if n == of else f" -- {max(n, 0)} of {of} columns, as ticked under select"
        self.copy.setToolTip("the table to the clipboard, tab-separated, for a "
                             "spreadsheet" + tip)

    # ---------------------------------------------------------------- rows
    def _on_row(self, index) -> None:
        row = self.proxy.mapToSource(index).row()
        if row < 0 or row >= len(self.model.regions):
            return
        region = self.model.regions[row]
        if self.model.source == ROIS:
            if self.show_roi is not None:
                self.show_roi(region)
            return
        if self.center_on is None:
            return
        locs = self.session.locs
        column = self.model.source
        if column not in locs or "x_nm" not in locs:
            return
        try:
            mine = np.asarray(locs[column]) == float(region)
        except (TypeError, ValueError):
            return
        if mine.any():
            self.center_on(float(np.nanmean(np.asarray(locs["x_nm"])[mine])),
                           float(np.nanmean(np.asarray(locs["y_nm"])[mine])))

    # ---------------------------------------------------------- the table out
    def rows_as_text(self, columns: Optional[Sequence[int]] = None):
        """``(header, rows)`` to write: the ticked columns (`chosen_columns`)
        or ``columns``, in the table's order, the numbers exact and a stale
        value empty -- a file cannot show it orange, and a number that no
        longer describes its region should not be read as one that does."""
        columns = self.chosen_columns() if columns is None \
            else [0] + [c for c in columns if c != 0]
        header = [self.model.header_text(c).replace("\n", " / ") for c in columns]
        rows = []
        for r in range(self.proxy.rowCount()):
            source_row = self.proxy.mapToSource(self.proxy.index(r, 0)).row()
            rows.append([self.model.names[source_row] if c == 0 else
                         "" if (source_row, c - 1) in self.model.stale else
                         exact_text(self.model.values[source_row][c - 1])
                         for c in columns])
        return header, rows

    def copy_table(self, columns: Optional[Sequence[int]] = None) -> None:
        header, rows = self.rows_as_text(columns)
        QGuiApplication.clipboard().setText(as_csv(header, rows, delimiter="\t"))

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save region results", "",
                                              "CSV (*.csv)")
        if path:
            self.save_csv(path)

    def save_csv(self, path) -> None:
        header, rows = self.rows_as_text()
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(as_csv(header, rows))
