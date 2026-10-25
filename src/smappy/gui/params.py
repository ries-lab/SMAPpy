"""A settings dataclass as a form, and the form back as a dataclass.

A field that is itself a dataclass (a *part*) becomes a collapsible section
with its own form inside, so a fitter assembled from parts reads as one.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QSizePolicy,
                               QHBoxLayout, QLabel, QLineEdit, QSpinBox,
                               QToolButton, QVBoxLayout, QWidget)

from ..plugins import ParamSpec, param_specs
from . import folders
from .widgets import CollapsibleSection, note_style


class _Field(QWidget):
    """One parameter.  Optional ones get an "auto" box that stands for None.

    An auto field still shows a number: whatever *auto* would resolve to, in
    grey, put there by `set_hint`.  "auto" alone tells the user that the value
    comes from somewhere else but not what it is, and the camera fields are
    read far more often than they are typed.
    """

    changed = Signal()
    # an optional bool: "auto" goes in front, as for any optional choice
    ON_OFF = ((True, "on"), (False, "off"))

    def __init__(self, spec: ParamSpec):
        super().__init__()
        self.spec = spec
        self._hint = ""
        # a list the plugin worked out from the session (`Plugin.describe`),
        # which replaces what the field's own callable offers
        self._supplied: Optional[list] = None
        info = spec.info
        row = QHBoxLayout(self)
        # the macOS style paints a check box wider than its size hint and
        # clips it at zero margin; a little slack on every row fixes that
        row.setContentsMargins(2, 0, 6, 0)
        self.auto: Optional[QCheckBox] = None
        # the microscope control's way (`type_over`): no *auto* box, the found value typed over
        self._typed_over = False
        self._checking = False
        self.locked = False
        self.reset_button: Optional[QToolButton] = None
        choices = info.choices() if callable(info.choices) else info.choices
        if choices is None and spec.type is bool and spec.optional:
            # three states as words: a tri-state tick box's third state is a
            # dash nobody reads as "auto", and it says nothing of what auto is
            choices = self.ON_OFF
        if choices is not None:
            self.widget = QComboBox()
            self._values = []
            self._fill(choices)
            self.widget.currentIndexChanged.connect(self.changed)
        elif info.kind == "text":
            # as wide as the form gives it, and monospaced: this is for
            # something read back character by character -- an expression --
            # where the 80 px a number gets would show a fifth of it
            self.widget = QLineEdit()
            self.widget.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
            self.widget.editingFinished.connect(self.changed)
            row.addWidget(self.widget, 1)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        elif info.kind in ("open_file", "save_file", "dir"):
            self.widget = QLineEdit()
            self.widget.editingFinished.connect(self.changed)
            browse = QToolButton(text="...")
            browse.clicked.connect(self._browse)
            row.addWidget(self.widget, 1)
            row.addWidget(browse)
        elif spec.type is bool:
            self.widget = QCheckBox()
            self.widget.toggled.connect(self.changed)
        elif spec.type is int and not spec.optional:
            self.widget = QSpinBox()
            self.widget.setRange(int(info.min) if info.min is not None else -10**9,
                                 int(info.max) if info.max is not None else 10**9)
            self.widget.valueChanged.connect(self.changed)
        else:
            # a line edit for floats and optionals: it takes "1e-7", and it is
            # the compact thing.  Validation happens in `value`.
            self.widget = QLineEdit()
            self.widget.setMaximumWidth(80)
            self.widget.editingFinished.connect(self._check)
            self.widget.editingFinished.connect(self.changed)
        if isinstance(self.widget, QComboBox):
            # first call on the room, up to its cap: see _fit_combo
            row.addWidget(self.widget, 100)
        elif info.kind not in ("open_file", "save_file", "dir", "text"):
            row.addWidget(self.widget)
        if spec.optional and choices is None and spec.type is not bool:
            self.auto = QCheckBox("auto")
            self.auto.toggled.connect(self._on_auto)
            row.addWidget(self.auto)
        if info.unit:
            row.addWidget(QLabel(info.unit))
        if info.kind not in ("open_file", "save_file", "dir", "text"):
            row.addStretch(1)
        if info.help:
            self.setToolTip(info.help)
        for box in (self.widget, self.auto):
            if isinstance(box, QCheckBox):
                box.setMinimumWidth(box.sizeHint().width() + 12)
        self.set(spec.default)

    def _fill(self, choices) -> None:
        """Put ``choices`` in the box.  Used to build it and to refresh it."""
        self.widget.clear()
        self._values = []
        choices = list(choices)
        if self.spec.optional:
            choices.insert(0, (None, "auto"))
        for choice in choices:
            value, label = choice if isinstance(choice, tuple) else (choice, str(choice))
            self.widget.addItem(label, value)
            self._values.append(value)
        if getattr(self, "_choice_hint", None) is not None:
            self.set_hint(self._choice_hint)
        self._fit_combo()

    def _fit_combo(self) -> None:
        """As wide as its longest choice when there is room, narrower when not.

        A combo box asks for its longest choice as its *minimum*, and one long
        choice ("ring and cross, with scattered points (demo)") made the whole
        section wider than the control column: it scrolled sideways and took
        the title bar's ? and detach arrow out of view.  So the minimum is a
        few characters, the natural width is the cap, and the list that drops
        down is still wide enough to read every choice.
        """
        box = self.widget
        box.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        natural = box.sizeHint().width()
        box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        box.setMinimumContentsLength(8)
        box.setMaximumWidth(max(natural, box.minimumSizeHint().width()))
        box.view().setMinimumWidth(natural)

    def refresh(self) -> None:
        """Re-read choices that are computed rather than fixed.

        A panel is built once and lives as long as the session, so a list that
        the plugin works out -- the expressions used before, the files that are
        open -- would otherwise be the list as it was when the section was
        first opened.  What is selected survives if it is still there.
        """
        choices = self.spec.info.choices
        if not callable(choices) or not isinstance(self.widget, QComboBox):
            return
        self._refill(self._supplied if self._supplied is not None else choices())

    def set_choices(self, choices) -> None:
        """The list the plugin works out from the session; see `Plugin.describe`.  For a path, what
        was found for it (a fitter's calibrations): offered in a menu, ▾ and a right click."""
        if self.spec.info.kind in ("open_file", "dir"):
            self._offer_paths(list(choices))
            return
        if not isinstance(self.widget, QComboBox):
            return
        self._supplied = list(choices)
        self._refill(self._supplied)

    def _refill(self, choices) -> None:
        current = self.value()
        blocked = self.widget.blockSignals(True)
        try:
            self._fill(choices)
            self.set(current)
        finally:
            self.widget.blockSignals(blocked)

    def type_over(self) -> None:
        """No *auto* box: the field shows the value found (the hint, in grey) and typing over it makes
        the value the user's; ↺ or clearing the field gives it back to the file.  The microscope control
        (`gui.microscope_form`): there every value has been found somewhere, and a box beside each that
        said so was one more thing to read.  `value()` is unchanged -- None while it is the found one."""
        if self.auto is None or not isinstance(self.widget, QLineEdit) or self._typed_over:
            return
        self._typed_over = True
        self.auto.setVisible(False)
        self.reset_button = QToolButton(text="\u21ba", autoRaise=True)
        self.reset_button.setToolTip("back to the value found: the file's, the database's")
        self.reset_button.clicked.connect(lambda: self.auto.setChecked(True))
        self.layout().insertWidget(self.layout().indexOf(self.auto), self.reset_button)
        self.widget.textEdited.connect(self._typed)
        self._on_auto(self.auto.isChecked())

    def _typed(self, text: str) -> None:
        if self.auto.isChecked():
            blocked = self.auto.blockSignals(True)       # the text typed stays: it is the value
            self.auto.setChecked(False)
            self.auto.blockSignals(blocked)
            self._enable()
            self._paint()

    def set_locked(self, locked: bool) -> None:
        """Locked: the value can be read, not typed (the microscope control's lock)."""
        self.locked = locked
        self._enable()

    def _enable(self) -> None:
        on = self.auto is not None and self.auto.isChecked()
        self.widget.setEnabled(not self.locked and (self._typed_over or not on))
        if self.reset_button is not None:
            self.reset_button.setVisible(not on)
            self.reset_button.setEnabled(not self.locked)

    def _offer_paths(self, choices: list) -> None:
        self._found_paths = choices
        if getattr(self, "found_button", None) is None:
            self.found_button = QToolButton(text="\u25be", autoRaise=True)
            self.found_button.setToolTip("the files found for this: choose one, or auto")
            self.found_button.clicked.connect(lambda: self._path_menu().exec(
                self.found_button.mapToGlobal(self.found_button.rect().bottomLeft())))
            self.layout().insertWidget(self.layout().indexOf(self.widget) + 1, self.found_button)
            self.widget.setContextMenuPolicy(Qt.CustomContextMenu)
            self.widget.customContextMenuRequested.connect(
                lambda at: self._path_menu().exec(self.widget.mapToGlobal(at)))
        self.found_button.setVisible(bool(choices))

    def _path_menu(self):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        auto = menu.addAction("auto" + (f" -- {self.widget.placeholderText()[6:]}"
                                         if self.widget.placeholderText().startswith("auto: ") else ""))
        auto.triggered.connect(lambda: self._take_path(""))
        menu.addSeparator()
        for value, label in getattr(self, "_found_paths", []):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(value == self.widget.text())
            action.triggered.connect(lambda _=False, v=value: self._take_path(v))
        return menu

    def _take_path(self, value: str) -> None:
        self.widget.setText(value)
        self.changed.emit()

    def _on_auto(self, on: bool) -> None:
        self._enable()
        if on and (self._hint or self._typed_over) and isinstance(self.widget, QLineEdit):
            self.widget.setText(self._hint)      # leaving it blank says less
        self._paint()
        self.changed.emit()

    def _paint(self) -> None:
        """Grey while the value on show is a hint rather than a setting."""
        if isinstance(self.widget, QLineEdit):
            hinting = self.auto is not None and self.auto.isChecked()
            self.widget.setStyleSheet(note_style("font-style: italic")
                                      if hinting else "")

    def set_hint(self, value: Any) -> None:
        """What *auto* currently resolves to, for showing greyed out.

        A hint is never a value: `value()` still returns None while auto is
        on, so nothing here can turn a read-from-the-file number into one the
        user is taken to have set.
        """
        if isinstance(self.widget, QComboBox) and self._values and (
                self._values[0] is None and self.spec.optional
                or self._values[0] == "" and self.widget.itemText(0).startswith("auto")):
            # a choice on auto says what auto is: "auto (on)"
            self._choice_hint = value          # kept: a refill of the list (`_fill`) says it again
            labels = dict(zip(self._values, (self.widget.itemText(i)
                                             for i in range(self.widget.count()))))
            shown = labels.get(value, None if value is None else str(value))
            self.widget.setItemText(0, f"auto ({shown})" if shown else "auto")
            return
        if self.spec.info.kind in ("open_file", "save_file", "dir") \
                and isinstance(self.widget, QLineEdit):
            # an empty path that something stands in for: said in grey, inside
            # the field -- for a file to write, the name a run would use
            self.widget.setPlaceholderText("" if value is None else str(value))
            return
        if self.auto is None or not isinstance(self.widget, QLineEdit):
            return
        self._hint = ("" if value is None else f"{value:g}"
                      if isinstance(value, float) else str(value))
        self.setToolTip("\n".join(filter(None, (
            self.spec.info.help, f"auto: {self._hint}" if self._hint else ""))))
        if self.auto.isChecked():
            self.widget.setText(self._hint)
            self._paint()

    def _browse(self) -> None:
        kind, name_filter = self.spec.info.kind, self.spec.info.file_filter
        # empty: what the plugin suggests (`set_hint`), the localization
        # file's folder and name for a file to write
        current = self.widget.text() or self.widget.placeholderText()
        if not current or Path(current).parent == Path("."):
            current = folders.start(current)    # nothing better to go on
        if kind == "open_file":
            path, _ = QFileDialog.getOpenFileName(self, self.spec.name, current, name_filter)
        elif kind == "save_file":
            path, _ = QFileDialog.getSaveFileName(self, self.spec.name, current, name_filter)
        else:
            path = QFileDialog.getExistingDirectory(self, self.spec.name, current)
        if path:
            folders.remember(path, stack=kind == "open_file")
            self.widget.setText(path)
            self.changed.emit()

    def set(self, value: Any) -> None:
        if self.auto is not None:
            self.auto.setChecked(value is None)
            self._paint()
            if value is None:
                if isinstance(self.widget, QLineEdit):
                    self.widget.setText(self._hint)
                return
        w = self.widget
        if isinstance(w, QComboBox):
            if value not in self._values and self._supplied is not None \
                    and value not in (None, ""):
                # a name the session does not have (yet): kept, and said so,
                # rather than quietly becoming the first entry
                w.addItem(f"{value} (not available)", value)
                self._values.append(value)
            w.setCurrentIndex(self._values.index(value) if value in self._values else 0)
        elif isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, QSpinBox):
            w.setValue(int(value))
        else:
            w.setText("" if value is None else f"{value:g}" if isinstance(value, float)
                      else str(value))
            self._set_text = w.text()
            if self.spec.type in (int, float):
                self._check()

    def value(self) -> Any:
        if self.auto is not None and self.auto.isChecked():
            return None
        w = self.widget
        if isinstance(w, QComboBox):
            return w.currentData()
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QSpinBox):
            return w.value()
        text = w.text().strip()
        if self.spec.type in (int, float):
            return self._number(text)
        return text

    @property
    def label(self) -> str:
        return self.spec.info.label or self.spec.name.replace("_", " ")

    def _number(self, text: str):
        """``text`` as the field's number, or a ValueError that names the field and says what it takes.

        Checked here, not left to the plugin: a word in a number field said only Python's "could not
        convert string to float", without the field, a decimal in a whole-number field was cut to its
        integer part, and a value beyond the field's bounds reached the run.  Refused, never clamped: a
        clamped value is one the user did not type.  The bounds are the GUI's, for what is typed: a value
        set from a chain, a script or a controlling program (`set`) beyond them is shown red but passed on,
        as it would be without a form (MicroClaw's live fits stop after half a second idle)."""
        info, whole = self.spec.info, self.spec.type is int
        try:
            number = float(text)
        except ValueError:
            raise ValueError(f"{self.label}: '{text}' is not a number" if text
                             else f"{self.label}: empty, a number is needed") from None
        if number != number:
            raise ValueError(f"{self.label}: '{text}' is not a number")
        if whole and number != int(number):
            raise ValueError(f"{self.label}: '{text}' is not a whole number")
        unit = f" {info.unit}" if info.unit else ""
        if text == getattr(self, "_set_text", None) and not self._checking:
            return int(number) if whole else number
        if info.min is not None and number < info.min:
            raise ValueError(f"{self.label}: {text} is below its least, {info.min:g}{unit}")
        if info.max is not None and number > info.max:
            raise ValueError(f"{self.label}: {text} is above its most, {info.max:g}{unit}")
        return int(number) if whole else number

    def problem(self) -> str:
        """What is wrong with the value as shown, or "" -- `_number`'s message, bounds included."""
        self._checking = True
        try:
            self.value()
        except (ValueError, TypeError) as error:
            return str(error)
        finally:
            self._checking = False
        return ""

    def _check(self) -> None:
        """When a number is typed: red and the reason in the tooltip if it will not do.  The run refuses
        it too, with the same words (`_number`)."""
        if self._typed_over and not self.widget.text().strip() and not self.auto.isChecked():
            self.auto.setChecked(True)               # emptied: the found value again
            return
        problem = self.problem()
        self.widget.setProperty("invalid", bool(problem))
        if problem:
            self.widget.setStyleSheet("border: 1.5px solid #d62728; border-radius: 3px")
            self.widget.setToolTip(problem)
        else:
            self.widget.setToolTip("")
            self._paint()


class LayersField(QWidget):
    """The `Analysis/Process/Layers` step's layers: per layer a grouping, where its bounds
    start from, and rows of field, lo, hi, quantile, required.

    A table of rows rather than a field per column, because which columns a
    table has is not known until a file is open -- and in a batch it is a
    different file each time.  With a session, the field box offers its
    columns and **from current layers** copies what the Render tab shows;
    without one, a field is typed.
    """

    changed = Signal()
    COLUMNS = ("field", "lo", "hi", "quantile", "required")
    GROUPING = ((None, "as it is"), (True, "merged"), (False, "not merged"))

    def __init__(self, spec: ParamSpec):
        super().__init__()
        from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QPushButton,
                                       QTableWidget)
        from ..plugins.chain_layers import STARTS, default_layers
        self.spec = spec
        self.session = None                 # set by the panel that owns the form
        self._default = default_layers
        self._layers: List[Dict[str, Any]] = []
        self._current = 0
        self._loading = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.which = QComboBox()
        self.which.currentIndexChanged.connect(self._switch)
        add_layer = QPushButton("+ layer")
        add_layer.clicked.connect(self._add_layer)
        remove_layer = QPushButton("- layer")
        remove_layer.clicked.connect(self._remove_layer)
        self.take = QPushButton("from current layers")
        self.take.setToolTip("the bounds, and whether blinks are merged, as the "
                             "Render tab shows them now")
        self.take.clicked.connect(self.take_from_session)
        # "from current layers" on a line of its own: beside the layer
        # buttons the row was wider than the control column
        for w in (self.which, add_layer, remove_layer):
            top.addWidget(w)
        for button in (add_layer, remove_layer, self.take):
            button.setMinimumWidth(button.fontMetrics().horizontalAdvance(button.text()) + 20)
        top.addStretch(1)
        layout.addLayout(top)
        take = QHBoxLayout()
        take.addWidget(self.take)
        take.addStretch(1)
        layout.addLayout(take)
        self.grouped = QComboBox()
        for value, label in self.GROUPING:
            self.grouped.addItem(label, value)
        self.start = QComboBox()
        for value, label in STARTS:
            self.start.addItem(value, value)
            self.start.setItemData(self.start.count() - 1, label, Qt.ToolTipRole)
        for w in (self.grouped, self.start):
            w.currentIndexChanged.connect(self._edited)
        # one per line, for the same reason
        options = QFormLayout()
        options.setContentsMargins(0, 0, 0, 0)
        for label, box in (("merge blinks", self.grouped), ("start from", self.start)):
            line = QHBoxLayout()
            line.addWidget(box)
            line.addStretch(1)
            options.addRow(label, line)
        layout.addLayout(options)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(110)
        self.table.itemChanged.connect(self._edited)
        layout.addWidget(self.table)
        rows = QHBoxLayout()
        self.add_field = QComboBox(editable=True)
        self.add_field.setPlaceholderText("field")
        self.add_field.lineEdit().setPlaceholderText("add a bound on...")
        self.add_field.activated.connect(lambda _i: self._add_row(self.add_field.currentText()))
        self.add_field.lineEdit().returnPressed.connect(
            lambda: self._add_row(self.add_field.currentText()))
        remove_row = QPushButton("- row")
        remove_row.clicked.connect(self._remove_rows)
        rows.addWidget(self.add_field, 1)
        rows.addWidget(remove_row)
        layout.addLayout(rows)
        if spec.info.help:
            self.setToolTip(spec.info.help)
        self.set(spec.default)

    # the interface a form expects of a field
    def refresh(self) -> None:
        self.add_field.clear()
        if self.session is not None and len(self.session.locs):
            self.add_field.addItems(sorted(self.session.locs.keys()))
        self.add_field.setCurrentText("")
        self.take.setEnabled(self.session is not None)

    def set_hint(self, value) -> None:
        pass

    def set(self, value) -> None:
        import copy
        self._layers = copy.deepcopy(list(value)) if value else self._default()
        self._current = 0
        self._reload()

    def value(self):
        import copy
        self._commit()
        return copy.deepcopy(self._layers)

    # what the widgets hold, to and from the layer on show
    def _reload(self) -> None:
        self._loading = True
        try:
            self.which.clear()
            self.which.addItems([f"layer {n + 1}" for n in range(len(self._layers))])
            self.which.setCurrentIndex(self._current)
            layer = self._layers[self._current]
            grouped = layer.get("grouped")
            self.grouped.setCurrentIndex([v for v, _ in self.GROUPING].index(
                None if grouped is None else bool(grouped)))
            index = self.start.findData(layer.get("start", "defaults"))
            self.start.setCurrentIndex(max(index, 0))
            self.table.setRowCount(0)
            for row in layer.get("bounds") or []:
                self._put_row(row)
        finally:
            self._loading = False

    def _put_row(self, row: Dict[str, Any]) -> None:
        from PySide6.QtWidgets import QTableWidgetItem
        n = self.table.rowCount()
        self.table.insertRow(n)
        for c, name in enumerate(self.COLUMNS):
            item = QTableWidgetItem()
            if name in ("quantile", "required"):
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                item.setCheckState(Qt.Checked if row.get(name) else Qt.Unchecked)
            else:
                value = row.get(name)
                item.setText("" if value is None else f"{value:g}"
                             if isinstance(value, float) else str(value))
            self.table.setItem(n, c, item)

    def _commit(self) -> None:
        if not self._layers:
            return
        rows = []
        for r in range(self.table.rowCount()):
            cells = {name: self.table.item(r, c) for c, name in enumerate(self.COLUMNS)}
            name = cells["field"].text().strip() if cells["field"] else ""
            if not name:
                continue
            row: Dict[str, Any] = {"field": name}
            for bound in ("lo", "hi"):
                text = cells[bound].text().strip() if cells[bound] else ""
                if text:
                    row[bound] = float(text)
            for flag in ("quantile", "required"):
                if cells[flag] is not None and cells[flag].checkState() == Qt.Checked:
                    row[flag] = True
            rows.append(row)
        layer = self._layers[self._current]
        layer["bounds"] = rows
        layer["start"] = self.start.currentData()
        grouped = self.grouped.currentData()
        if grouped is None:
            layer.pop("grouped", None)
        else:
            layer["grouped"] = bool(grouped)

    def _edited(self, *args) -> None:
        if self._loading:
            return
        try:
            self._commit()
        except ValueError:
            return                          # a half-typed number: wait for the rest
        self.changed.emit()

    def _switch(self, index: int) -> None:
        if self._loading or index < 0:
            return
        self._commit()
        self._current = index
        self._reload()

    def _add_layer(self) -> None:
        self._commit()
        self._layers.append({"start": "defaults", "bounds": []})
        self._current = len(self._layers) - 1
        self._reload()
        self.changed.emit()

    def _remove_layer(self) -> None:
        if len(self._layers) <= 1:
            return
        self._layers.pop(self._current)
        self._current = max(0, self._current - 1)
        self._reload()
        self.changed.emit()

    def _add_row(self, name: str) -> None:
        name = (name or "").strip()
        if not name:
            return
        self._put_row({"field": name})
        self.add_field.setCurrentText("")
        self._edited()

    def _remove_rows(self) -> None:
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)
        self._edited()

    def take_from_session(self) -> None:
        """The session's layers, as they are: every bound, and the grouping."""
        if self.session is None:
            return
        layers = []
        for config in self.session.layer_configs():
            rows = [{"field": f, **({"lo": lo} if lo is not None else {}),
                     **({"hi": hi} if hi is not None else {})}
                    for f, (lo, hi) in config["bounds"].items()]
            layers.append({"grouped": config["grouped"], "start": "empty",
                           "bounds": rows})
        if layers:
            self.set(layers)
            self.changed.emit()


class ColumnsField(QWidget):
    """A list of names to tick: which columns an export writes, say.

    The value is the ticked names, or ``[]`` when every one is ticked: "all
    columns" then stays all of them for a table that has more, rather than
    becoming the list this table happened to have.  With the field's
    ``empty_is_all`` off, ``[]`` is none ticked instead, for a list of
    extras.  The names come from the session (`FormState.choices`); a name
    in the value that the session does not offer is kept, ticked, and
    marked as not there.

    *none* on a list where ``[]`` is all has nothing to say as a value: it
    leaves every name unticked on screen (``_none``) until one is ticked,
    rather than the next refresh ticking them all again, which is what made
    it look like a second *all*.
    """

    changed = Signal()

    def __init__(self, spec: ParamSpec):
        super().__init__()
        from PySide6.QtWidgets import QListWidget, QPushButton
        self.spec = spec
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 6, 0)
        self.list = QListWidget()
        self.list.setMaximumHeight(140)
        self.list.itemChanged.connect(lambda _: self.changed.emit())
        layout.addWidget(self.list)
        row = QHBoxLayout()
        for label, on in (("all", True), ("none", False)):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, on=on: self._tick_all(on))
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        if spec.info.help:
            self.setToolTip(spec.info.help)
        self._names: List[str] = []
        self._value: List[str] = list(spec.default() if callable(spec.default)
                                      else spec.default or [])
        self.empty_is_all = getattr(spec.info, "empty_is_all", True)
        self._none = False

    def _tick_all(self, on: bool) -> None:
        self._none = not on
        blocked = self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.list.blockSignals(blocked)
        self.changed.emit()

    def _fill(self, names: List[str], ticked: List[str]) -> None:
        from PySide6.QtWidgets import QListWidgetItem
        blocked = self.list.blockSignals(True)
        self.list.clear()
        everything = self.empty_is_all and not ticked and not self._none
        shown = list(names) + [n for n in ticked if n not in names]
        for name in shown:
            item = QListWidgetItem(name if name in names else f"{name} (not there)")
            item.setData(Qt.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if everything or name in ticked else Qt.Unchecked)
            self.list.addItem(item)
        self.list.blockSignals(blocked)

    # the interface a form expects of a field
    def set_choices(self, choices) -> None:
        current = self.value()
        self._names = [c[0] if isinstance(c, tuple) else c for c in choices]
        self._fill(self._names, current)

    def refresh(self) -> None:
        pass

    def set_hint(self, value) -> None:
        pass

    def set(self, value) -> None:
        self._none = False
        self._fill(self._names, list(value or []))

    def value(self) -> List[str]:
        items = [self.list.item(i) for i in range(self.list.count())]
        if not items:
            return list(self._value)
        ticked = [it.data(Qt.UserRole) for it in items if it.checkState() == Qt.Checked]
        if not self.empty_is_all:
            return ticked
        if ticked:
            self._none = False
        return [] if len(ticked) == len(items) else ticked


class ShapesField(QWidget):
    """Shapes drawn elsewhere -- on a preview's figure (`ShapeEdit`) -- kept
    as a setting: what the form shows of them is how many, and a button to
    take them away.  The shapes themselves are drawn and seen on the figure.
    """

    changed = Signal()

    def __init__(self, spec: ParamSpec):
        super().__init__()
        from PySide6.QtWidgets import QPushButton
        self.spec = spec
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 0, 6, 0)
        self.label = QLabel("")
        self.clear_button = QPushButton("clear")
        self.clear_button.clicked.connect(lambda: (self.set([]), self.changed.emit()))
        row.addWidget(self.label)
        row.addWidget(self.clear_button)
        row.addStretch(1)
        if spec.info.help:
            self.setToolTip(spec.info.help)
        self._shapes: List[dict] = []
        self.set([])

    def set(self, value) -> None:
        self._shapes = [dict(s) for s in (value or []) if isinstance(s, dict)]
        inside = sum(1 for s in self._shapes if s.get("kind") == "include")
        outside = len(self._shapes) - inside
        self.label.setText("none: see Preview" if not self._shapes else
                           ", ".join(bit for bit in (
                               f"{inside} to fit in" if inside else "",
                               f"{outside} left out" if outside else "") if bit))
        self.clear_button.setEnabled(bool(self._shapes))

    def value(self) -> List[dict]:
        return [dict(s) for s in self._shapes]

    def refresh(self) -> None:
        pass

    def set_hint(self, value) -> None:
        pass


#: px: a key parameter's label wraps beyond this, a component's name before it
KEY_LABEL_WIDTH = 150


class PresetField(QWidget):
    """A name from saved settings (`smappy.presets`), with *save...* and
    *delete* beside it: ``kind="preset"``.

    Choosing a name is an edit like any other -- the plugin's `react` fills
    the settings from it -- and saving asks the form for its values, since a
    preset is the whole form, not this field.
    """

    changed = Signal()
    save_requested = Signal(str)

    def __init__(self, spec: ParamSpec):
        super().__init__()
        self.spec = spec
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 0, 6, 0)
        self.widget = QComboBox()
        self.widget.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.widget.setMinimumContentsLength(8)
        self.save_button = QToolButton(text="save...")
        self.save_button.setToolTip("keep these settings under a name of yours")
        self.delete_button = QToolButton(text="\u2212")
        self.delete_button.setToolTip("delete the chosen preset of yours")
        # "modified" once the settings have moved from the preset's (`set_hint`)
        self.state = QLabel("")
        self.state.setStyleSheet("color: grey")
        row.addWidget(self.widget, 1)
        row.addWidget(self.state)
        row.addWidget(self.save_button)
        row.addWidget(self.delete_button)
        if spec.info.help:
            self.setToolTip(spec.info.help)
        self.widget.currentIndexChanged.connect(self._chosen)
        self.save_button.clicked.connect(self._save)
        self.delete_button.clicked.connect(self._delete)
        self._fill(spec.default or "")

    def folders(self) -> list:
        store = self.spec.info.store
        return list(store()) if callable(store) else []

    def _fill(self, current: str) -> None:
        from .. import presets
        self.widget.blockSignals(True)
        self.widget.clear()
        self.widget.addItem("-", "")
        for name in presets.names(self.folders()):
            self.widget.addItem(name, name)
        index = self.widget.findData(current)
        if index < 0 and current:
            self.widget.addItem(f"{current} (missing)", current)
            index = self.widget.count() - 1
        self.widget.setCurrentIndex(max(index, 0))
        self.widget.blockSignals(False)
        self._enable()

    def _enable(self) -> None:
        from .. import presets
        name = self.value()
        folders = self.folders()
        self.delete_button.setEnabled(bool(name and folders
                                           and presets.is_users(folders, name)))
        self.save_button.setEnabled(bool(folders))

    def _chosen(self) -> None:
        self._enable()
        self.changed.emit()

    def _save(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save preset", "name:", text=self.value())
        if ok and name.strip():
            self.save_requested.emit(name.strip())

    def _delete(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        from .. import presets
        name = self.value()
        try:
            presets.delete(self.folders(), name)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Delete preset", str(error))
            return
        self._fill("")
        self.changed.emit()

    def saved(self, name: str) -> None:
        """A preset was written: list it, and show it chosen."""
        self._fill(name)

    def set(self, value) -> None:
        self._fill(value or "")

    def value(self) -> str:
        return self.widget.currentData() or ""

    def refresh(self) -> None:
        self._fill(self.value())

    def set_hint(self, value) -> None:
        """A word beside the name: "modified" when the settings have moved
        from the preset's (`FormState.hints`)."""
        self.state.setText(str(value) if value else "")


class ListField(QWidget):
    """A list of parts -- a simulated scene's components -- as a list with a
    form below it for the item selected.

    The items are kept here as settings objects; the form edits the one
    selected, and an edit writes it back.  Each row is named by the item's
    `ParamInfo.item_name` field and ticked by its `item_on` one.  The plugin
    addresses an item as ``<field>.<i>.<name>`` in its `FormState` and its
    `react` (`route`); what it says about an item not on show is kept and
    shown when it is selected.  ``set_choices`` on the field itself gives the
    *add* menu its entries, ``(label, values)``; saved items come from the
    folders `ParamInfo.store` names.
    """

    changed = Signal()
    item_changed = Signal(str)        # "<i>.<dotted name>", or "" for the list

    def __init__(self, spec: ParamSpec):
        super().__init__()
        from PySide6.QtWidgets import QListWidget, QMenu
        self.spec = spec
        self.item_type = spec.item_type
        self.items: List[Any] = []
        self._states: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self._offers: List[tuple] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.list = QListWidget()
        if spec.info.help:
            self.list.setToolTip(spec.info.help)
        layout.addWidget(self.list)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        self.add_button = QToolButton(text="+")
        self.add_button.setToolTip("add an item: a new one, or one saved before")
        self.add_button.setPopupMode(QToolButton.InstantPopup)
        self.add_menu = QMenu(self.add_button)
        self.add_menu.aboutToShow.connect(self._fill_add_menu)
        self.add_button.setMenu(self.add_menu)
        self.copy_button = QToolButton(text="duplicate")
        self.copy_button.setToolTip("a copy of the selected item, after it")
        self.remove_button = QToolButton(text="\u2212")
        self.remove_button.setToolTip("remove the selected item")
        self.up_button = QToolButton(text="\u2191")
        self.down_button = QToolButton(text="\u2193")
        self.save_button = QToolButton(text="save...")
        self.save_button.setToolTip("keep the selected item under a name, to add "
                                    "to another list")
        self.save_button.setVisible(callable(spec.info.store))
        for b in (self.add_button, self.copy_button, self.remove_button,
                  self.up_button, self.down_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)
        self.editor = SettingsForm(self.item_type, spec.items)
        if spec.info.item_on in spec.items:          # the list's tick is it
            self.editor.set_shown({spec.info.item_on: False})
        layout.addWidget(self.editor)
        self.copy_button.clicked.connect(self._duplicate)
        self.remove_button.clicked.connect(self._remove)
        self.up_button.clicked.connect(lambda: self._move(-1))
        self.down_button.clicked.connect(lambda: self._move(1))
        self.save_button.clicked.connect(self._save)
        self.list.currentRowChanged.connect(self._select)
        self.list.itemChanged.connect(self._ticked)
        self.editor.field_changed.connect(self._edited)
        self.set(spec.default or [])

    # ------------------------------------------------------------ the list
    def current(self) -> int:
        return self.list.currentRow()

    def _name(self, item, i: int) -> str:
        name = getattr(item, self.spec.info.item_name, "") if self.spec.info.item_name else ""
        return str(name) or f"{i + 1}"

    def _rebuild(self, select: int) -> None:
        on = self.spec.info.item_on
        self.list.blockSignals(True)
        self.list.clear()
        from PySide6.QtWidgets import QListWidgetItem
        for i, item in enumerate(self.items):
            row = QListWidgetItem(self._name(item, i))
            if on and hasattr(item, on):
                row.setFlags(row.flags() | Qt.ItemIsUserCheckable)
                row.setCheckState(Qt.Checked if getattr(item, on) else Qt.Unchecked)
            self.list.addItem(row)
        self.list.blockSignals(False)
        # as tall as its rows, up to five: two components need no empty box
        rows = max(len(self.items), 1)
        self.list.setFixedHeight(min(rows, 5) * self.list.sizeHintForRow(0)
                                 + 2 * self.list.frameWidth() + 2
                                 if self.items else 2 * self.list.frameWidth() + 20)
        has = bool(self.items)
        for b in (self.copy_button, self.remove_button, self.up_button,
                  self.down_button, self.save_button):
            b.setEnabled(has)
        self.editor.setVisible(has)
        if has:
            self.list.setCurrentRow(min(max(select, 0), len(self.items) - 1))
            self._select(self.current())

    def _select(self, i: int) -> None:
        if not 0 <= i < len(self.items):
            return
        self.editor.blockSignals(True)
        self.editor.set(self.items[i])
        self.editor.blockSignals(False)
        # one form for every item: what was hidden or greyed for another
        # item comes back before this one's state is shown
        own = self._states.get(i, {})
        for method in ("set_shown", "set_active"):
            reset = {sub: True for state in self._states.values()
                     for sub in state.get(method, {}) if sub not in own.get(method, {})}
            if reset:
                getattr(self.editor, method)(reset)
        for method, values in own.items():
            getattr(self.editor, method)(values)

    def _edited(self, sub: str) -> None:
        i = self.current()
        if not 0 <= i < len(self.items):
            return
        try:
            self.items[i] = self.editor.value()
        except (ValueError, TypeError):
            return                         # a number half typed: the old stands
        row = self.list.item(i)
        if row is not None:
            self.list.blockSignals(True)
            row.setText(self._name(self.items[i], i))
            on = self.spec.info.item_on
            if on and hasattr(self.items[i], on):
                row.setCheckState(Qt.Checked if getattr(self.items[i], on) else Qt.Unchecked)
            self.list.blockSignals(False)
        self.item_changed.emit(f"{i}.{sub}")
        self.changed.emit()

    def _ticked(self, row) -> None:
        i = self.list.row(row)
        on = self.spec.info.item_on
        if not (0 <= i < len(self.items) and on and hasattr(self.items[i], on)):
            return
        self.items[i] = dataclasses.replace(self.items[i],
                                            **{on: row.checkState() == Qt.Checked})
        if i == self.current():
            self._select(i)
        self.item_changed.emit(f"{i}.{on}")
        self.changed.emit()

    def _structural(self, select: int) -> None:
        self._states.clear()
        self._rebuild(select)
        self.item_changed.emit("")
        self.changed.emit()

    def add(self, item=None) -> None:
        """Append ``item`` -- a settings object or a map of its values -- or a
        new one, and select it."""
        self.items.append(self._item(item))
        self._structural(len(self.items) - 1)

    def _duplicate(self) -> None:
        i = self.current()
        if 0 <= i < len(self.items):
            import copy
            twin = copy.deepcopy(self.items[i])
            name = self.spec.info.item_name
            if name and hasattr(twin, name):
                twin = dataclasses.replace(twin, **{name: f"{getattr(twin, name)} copy"})
            self.items.insert(i + 1, twin)
            self._structural(i + 1)

    def _remove(self) -> None:
        i = self.current()
        if 0 <= i < len(self.items):
            del self.items[i]
            self._structural(i)

    def _move(self, by: int) -> None:
        i = self.current()
        j = i + by
        if 0 <= i < len(self.items) and 0 <= j < len(self.items):
            self.items[i], self.items[j] = self.items[j], self.items[i]
            self._structural(j)

    def _fill_add_menu(self) -> None:
        from .. import presets
        menu = self.add_menu
        menu.clear()
        menu.addAction("new", lambda: self.add())
        for label, values in self._offers:
            menu.addAction(label, lambda v=values: self.add(v))
        store = self.spec.info.store
        folders = list(store()) if callable(store) else []
        saved = presets.names(folders)
        if saved:
            menu.addSeparator()
            for name in saved:
                menu.addAction(name, lambda n=name: self._add_saved(folders, n))

    def _add_saved(self, folders, name: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        from .. import presets
        try:
            values = presets.read(folders, name)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Add", str(error))
            return
        self.add(values)

    def _save(self) -> None:
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        from .. import presets
        from ..plugins import settings_values
        i = self.current()
        store = self.spec.info.store
        if not (0 <= i < len(self.items)) or not callable(store):
            return
        name, ok = QInputDialog.getText(self, "Save", "name:",
                                        text=self._name(self.items[i], i))
        if not (ok and name.strip()):
            return
        try:
            presets.write(list(store()), name.strip(), settings_values(self.items[i]))
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Save", str(error))

    def _item(self, value):
        from ..plugins import _from_specs
        if isinstance(value, self.item_type):
            return value
        if value is None:
            return self.item_type()
        return _from_specs(self.item_type, self.spec.items,
                           value if isinstance(value, dict) else {})

    # ------------------------------------------------------- as a field
    def set(self, value) -> None:
        import copy
        self.items = [self._item(copy.deepcopy(v)) for v in (value or [])]
        self._states.clear()
        self._rebuild(0)

    def value(self) -> list:
        import copy
        return copy.deepcopy(self.items)

    def values(self) -> list:
        """The items as flat maps, for a file (`SettingsForm.values`)."""
        from ..plugins import settings_values
        return [settings_values(item) for item in self.items]

    def refresh(self) -> None:
        self.editor.refresh()

    def set_hint(self, value) -> None:
        pass

    def set_choices(self, choices) -> None:
        """What *add* offers beside a new item: ``(label, values)`` pairs."""
        self._offers = [tuple(c) for c in (choices or [])]

    def route(self, method: str, path: str, value) -> None:
        """``<i>.<name>`` of a `SettingsForm` method (``set_hints``, ...):
        to the form if item ``i`` is on show, kept for when it is."""
        index, _, sub = path.partition(".")
        try:
            i = int(index)
        except ValueError:
            return
        if not (0 <= i < len(self.items)) or not sub:
            return
        if method == "set_values":
            from ..plugins import _from_specs, settings_values
            flat = settings_values(self.items[i])
            flat[sub] = value
            self.items[i] = _from_specs(self.item_type, self.spec.items, flat)
            if i == self.current():
                self._select(i)
                row = self.list.item(i)
                if row is not None:
                    self.list.blockSignals(True)
                    row.setText(self._name(self.items[i], i))
                    self.list.blockSignals(False)
            return
        self._states.setdefault(i, {}).setdefault(method, {})[sub] = value
        if i == self.current():
            getattr(self.editor, method)({sub: value})


class SettingsForm(QWidget):
    """The fields of a settings dataclass; ``value()`` builds the instance.

    Fields marked advanced sit under a collapsed "more" section; a part (a
    dataclass-typed field) is a section of its own, open unless advanced or
    collapsed.  Rows and parts come in the order they are declared, so a
    plugin whose plain settings say what it does can put them first.
    ``field_changed`` carries the dotted name of what was edited.
    """

    changed = Signal()
    field_changed = Signal(str)

    def __init__(self, settings_cls: type, specs: Optional[Dict[str, ParamSpec]] = None,
                 parent=None):
        super().__init__(parent)
        self.settings_cls = settings_cls
        self.fields: Dict[str, Any] = {}          # _Field or SettingsForm
        self.labels: Dict[str, QLabel] = {}       # the row label of each field
        self.sections: Dict[str, CollapsibleSection] = {}   # each part's
        # each row's form layout, to hide the row (`set_shown`), and the
        # "more" section, hidden with the last of its rows
        self.row_layouts: Dict[str, QFormLayout] = {}
        self.more_section: Optional[CollapsibleSection] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        def form_layout():
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setVerticalSpacing(2)
            # macOS's default keeps every field at its size hint, so an
            # expression got a box of 160 px in a column twice as wide; only
            # what asks to grow does (`kind="text"`), on every platform
            form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            return form
        more = form_layout()
        main = None                          # the rows since the last part
        # the widget each row is in the form layout by: the field, or the
        # holder of a field and those `beside` it
        self.row_widgets: Dict[str, QWidget] = {}
        self.beside: Dict[str, tuple] = {}   # a field beside another: (label, widget)
        last_row: Optional[QHBoxLayout] = None
        # `specs if None`, not `specs or`: {} is a plugin that declares no
        # settings at all -- a loader with sensible defaults, say -- and it
        # must give an empty form rather than fall back to introspection
        specs = param_specs(settings_cls) if specs is None else specs
        shown_specs = [(n, sp) for n, sp in specs.items() if not sp.info.hidden]
        followed = {shown_specs[k][0] for k in range(len(shown_specs) - 1)
                    if shown_specs[k + 1][1].info.beside}
        for name, spec in shown_specs:
            if spec.items is not None:
                lst = ListField(spec)
                lst.item_changed.connect(
                    lambda sub, n=name: self.field_changed.emit(f"{n}.{sub}" if sub else n))
                lst.changed.connect(self.changed)
                self.fields[name] = lst
                main = last_row = None
                section = CollapsibleSection(spec.info.label or name, lst,
                                             expanded=not (spec.info.advanced
                                                           or spec.info.collapsed))
                self.sections[name] = section
                layout.addWidget(section)
                continue
            if spec.children is not None:
                if spec.info.kind == "microscope":
                    # one line here, the rows in a window (`gui.microscope_form`)
                    from .microscope_form import MicroscopeForm
                    part = MicroscopeForm(spec.type, spec.children)
                else:
                    part = SettingsForm(spec.type, spec.children)
                part.field_changed.connect(lambda sub, n=name: self.field_changed.emit(f"{n}.{sub}"))
                part.changed.connect(self.changed)
                self.fields[name] = part
                main = last_row = None
                section = CollapsibleSection(spec.info.label or name, part,
                                             expanded=not (spec.info.advanced
                                                           or spec.info.collapsed))
                self.sections[name] = section
                layout.addWidget(section)
                continue
            w = (LayersField(spec) if spec.info.kind == "layers"
                 else ColumnsField(spec) if spec.info.kind == "columns"
                 else ShapesField(spec) if spec.info.kind == "shapes"
                 else PresetField(spec) if spec.info.kind == "preset" else _Field(spec))
            w.changed.connect(self.changed)
            w.changed.connect(lambda n=name: self.field_changed.emit(n))
            if isinstance(w, PresetField):
                w.save_requested.connect(lambda preset, f=w: self._save_preset(preset, f))
            self.fields[name] = w
            label = QLabel(spec.info.label or name)
            if spec.info.help:
                label.setToolTip(spec.info.help)
            self.labels[name] = label
            if spec.info.beside and last_row is not None:
                # on the row before: its label a small word between the two
                last_row.insertWidget(last_row.count() - 1, label)
                last_row.insertWidget(last_row.count() - 1, w)
                self.beside[name] = (label, w)
                self.row_layouts[name] = self.row_layouts[self._last_name]
                self.row_widgets[name] = self.row_widgets[self._last_name]
                continue
            row_widget = w
            last_row = None
            if name in followed:
                row_widget = QWidget()
                last_row = QHBoxLayout(row_widget)
                last_row.setContentsMargins(0, 0, 0, 0)
                last_row.setSpacing(4)
                last_row.addWidget(w)
                last_row.addStretch(1)
            self._last_name = name
            self.row_widgets[name] = row_widget
            target = more if spec.info.advanced else None
            if target is None:
                if main is None:
                    main = form_layout()
                    layout.addLayout(main)
                target = main
            target.addRow(label, row_widget)
            self.row_layouts[name] = target
        if more.rowCount():
            box = QWidget()
            box.setLayout(more)
            self.more_section = CollapsibleSection("more", box, expanded=False)
            layout.addWidget(self.more_section)
        # the key parameters (`FormState.keys`): copies of a few fields, at
        # the top after the first rows, shown once the plugin names some
        self._specs = specs
        self._keys: Dict[str, str] = {}
        self.key_fields: Dict[str, QWidget] = {}
        keys_box = QWidget()
        self._keys_form = form_layout()
        keys_box.setLayout(self._keys_form)
        self.keys_section = CollapsibleSection("key parameters", keys_box, expanded=True)
        self.keys_section.hide()
        first_section = next((k for k in range(layout.count())
                              if isinstance(layout.itemAt(k).widget(), CollapsibleSection)),
                             layout.count())
        layout.insertWidget(first_section, self.keys_section)
        self.changed.connect(self._sync_keys)

    def value(self):
        if self.settings_cls is None:
            return None                     # a plugin with nothing to configure
        return self.settings_cls(**{n: f.value() for n, f in self.fields.items()})

    def set(self, settings) -> None:
        for name, f in self.fields.items():
            f.set(getattr(settings, name))

    def refresh(self) -> None:
        """Re-read every field whose choices are computed.  See `_Field.refresh`."""
        for f in self.fields.values():
            f.refresh()

    def leaves(self):
        """Every field widget, parts' included, depth first."""
        for f in self.fields.values():
            if isinstance(f, SettingsForm):
                yield from f.leaves()
            else:
                yield f

    def values(self) -> Dict[str, Any]:
        """Every field by dotted name, for saving.

        Reads the widgets rather than building the settings object: a form
        left mid-edit with a bad number still has to be saveable, and a value
        that will not parse is simply left out.
        """
        out: Dict[str, Any] = {}
        for name, f in self.fields.items():
            if isinstance(f, SettingsForm):
                out.update({f"{name}.{k}": v for k, v in f.values().items()})
                continue
            if isinstance(f, ListField):
                out[name] = f.values()
                continue
            try:
                out[name] = f.value()
            except (ValueError, TypeError):
                pass
        return out

    def restore(self, values: Dict[str, Any]) -> List[str]:
        """Set what we recognise and report what we did not.

        Saved values outlive the plugin that wrote them, so a renamed or
        removed field must cost that field and not the whole form.
        """
        dropped = []
        for path, value in values.items():
            try:
                self.set_values({path: value})
            except (KeyError, AttributeError, ValueError, TypeError):
                dropped.append(path)
        return dropped

    # ------------------------------------------------------- key parameters
    def spec_at(self, path: str) -> Optional[ParamSpec]:
        """The spec of the field at ``path``: through parts, and through a
        list's item (``components.0.photons``)."""
        specs = self._specs
        parts = path.split(".")
        spec = None
        k = 0
        while k < len(parts):
            spec = specs.get(parts[k]) if specs else None
            if spec is None:
                return None
            if spec.items is not None:
                specs = spec.items
                k += 2                      # the item's index
                continue
            specs = spec.children
            k += 1
        return spec

    def value_at(self, path: str):
        """The value at ``path`` as the form holds it now."""
        parts = path.split(".")
        target: Any = self
        k = 0
        while k < len(parts):
            if isinstance(target, SettingsForm):
                target = target.fields[parts[k]]
                k += 1
            elif isinstance(target, ListField):
                item = target.items[int(parts[k])]
                for name in parts[k + 1:]:
                    item = getattr(item, name)
                return item
            else:
                break
        return target.value()

    def set_keys(self, keys: Dict[str, str]) -> None:
        """Show these fields again at the top (`FormState.keys`), with these
        labels; an edit there sets the field itself and says so as an edit
        of it, so the plugin hears ``components.0.blinking.photons``."""
        keys = dict(keys)
        if keys == self._keys:
            self._sync_keys()
            return
        self._keys = keys
        while self._keys_form.rowCount():
            self._keys_form.removeRow(0)
        self.key_fields = {}
        for path, label in keys.items():
            spec = self.spec_at(path)
            if spec is None or spec.children is not None or spec.items is not None:
                continue                    # a key names one field, never a part
            proxy = _Field(dataclasses.replace(spec, info=dataclasses.replace(
                spec.info, label=label, beside=False)))
            proxy.changed.connect(lambda p=path, w=proxy: self._key_edited(p, w))
            name = QLabel(label)
            # "second label: labelling efficiency" on two lines, not wider
            name.setWordWrap(True)
            name.setFixedWidth(min(name.fontMetrics().horizontalAdvance(label) + 4,
                                   KEY_LABEL_WIDTH))
            if spec.info.help:
                name.setToolTip(spec.info.help)
            self._keys_form.addRow(name, proxy)
            self.key_fields[path] = proxy
        self.keys_section.setVisible(bool(self.key_fields))
        self._sync_keys()

    def _key_edited(self, path: str, proxy) -> None:
        try:
            value = proxy.value()
        except (ValueError, TypeError):
            return
        self.set_values({path: value})
        self.field_changed.emit(path)

    def _sync_keys(self) -> None:
        for path, proxy in self.key_fields.items():
            try:
                value = self.value_at(path)
            except (KeyError, IndexError, AttributeError, ValueError, TypeError):
                continue
            proxy.blockSignals(True)
            proxy.set(value)
            proxy.blockSignals(False)

    def _routed(self, method: str, path: str, value) -> bool:
        """Hand ``path`` to a list of parts on the way, if there is one:
        ``components.2.photons`` is item 2's ``photons`` (`ListField.route`)."""
        parts = path.split(".")
        target = self
        for k, part in enumerate(parts[:-1]):
            field = target.fields.get(part) if isinstance(target, SettingsForm) else None
            if isinstance(field, ListField):
                field.route(method, ".".join(parts[k + 1:]), value)
                return True
            if field is None:
                return False
            target = field
        return False

    def _save_preset(self, name: str, field: "PresetField") -> None:
        """*save...* of a preset field: the whole form, but the preset's own
        name, under ``name`` in the user's folder."""
        from PySide6.QtWidgets import QMessageBox
        from .. import presets
        own = next((n for n, f in self.fields.items() if f is field), None)
        values = {k: v for k, v in self.values().items() if k != own}
        try:
            presets.write(field.folders(), name, values)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Save preset", str(error))
            return
        field.saved(name)

    def set_hints(self, values: Dict[str, Any]) -> None:
        """Show, by dotted name, what each field's *auto* resolves to.

        Unknown names are ignored rather than raised: a hint is an extra, and
        a plugin that offers one for a field it no longer has must not break
        the form the way a stale saved value must not.
        """
        for path, value in values.items():
            if self._routed("set_hints", path, value):
                continue
            target = self
            parts = path.split(".")
            try:
                for part in parts[:-1]:
                    target = target.fields[part]
                field = target.fields[parts[-1]]
            except (KeyError, AttributeError):
                continue
            if hasattr(field, "set_hint"):
                field.set_hint(value)
            elif hasattr(field, "set_details"):
                # a part that shows what the function found (`gui.microscope_form`)
                field.set_details(value)

    def set_choices(self, lists: Dict[str, Any]) -> None:
        """Replace, by dotted name, the lists of fields whose choices the
        session decides (`Plugin.describe`).  Unknown names are ignored, as
        for hints."""
        for path, choices in lists.items():
            if self._routed("set_choices", path, choices):
                continue
            target = self
            parts = path.split(".")
            try:
                for part in parts[:-1]:
                    target = target.fields[part]
                field = target.fields[parts[-1]]
            except (KeyError, AttributeError):
                continue
            if hasattr(field, "set_choices"):
                field.set_choices(choices)

    def set_titles(self, titles: Dict[str, str]) -> None:
        """A part's title by dotted name (`FormState.titles`), cut to the
        control column with the whole of it in the tooltip: a title is one
        line and must not widen the form.  Unknown names are ignored."""
        from .widgets import CONTROL_WIDTH
        for path, title in titles.items():
            if self._routed("set_titles", path, title):
                continue
            target = self
            parts = path.split(".")
            try:
                for part in parts[:-1]:
                    target = target.fields[part]
                section = target.sections[parts[-1]]
            except (KeyError, AttributeError):
                continue
            metrics = section.button.fontMetrics()
            shown = metrics.elidedText(title, Qt.ElideRight, CONTROL_WIDTH - 90)
            section.set_title(shown)
            section.button.setToolTip(title if shown != title else "")

    def set_active(self, flags: Dict[str, bool]) -> None:
        """Grey out the fields the current settings do not read.

        Cosmetic on purpose: a disabled widget keeps its value and still
        answers `value()`, so greying a field cannot change what a run does.
        Unknown names are ignored, as in `set_hints`.
        """
        for path, on in flags.items():
            if self._routed("set_active", path, on):
                continue
            target = self
            parts = path.split(".")
            try:
                for part in parts[:-1]:
                    target = target.fields[part]
                field = target.fields[parts[-1]]
            except (KeyError, AttributeError):
                continue
            field.setEnabled(bool(on))
            label = target.labels.get(parts[-1])
            if label is not None:
                label.setEnabled(bool(on))

    def set_shown(self, flags: Dict[str, bool]) -> None:
        """Hide the fields and parts the settings leave out (`FormState.shown`):
        a row with its label, a part with its title bar, and "more" once none
        of its rows is left.  Cosmetic, as `set_active` is: a hidden widget
        keeps its value and still answers `value()`.  Unknown names are
        ignored.
        """
        touched = set()
        for path, on in flags.items():
            if self._routed("set_shown", path, on):
                continue
            target = self
            parts = path.split(".")
            try:
                for part in parts[:-1]:
                    target = target.fields[part]
                name = parts[-1]
                if name in target.sections:
                    target.sections[name].setVisible(bool(on))
                    continue
                if name in target.beside:
                    for widget in target.beside[name]:
                        widget.setVisible(bool(on))
                    continue
                target.row_layouts[name].setRowVisible(
                    target.row_widgets.get(name, target.fields[name]), bool(on))
            except (KeyError, AttributeError):
                continue
            touched.add(target)
        for form in touched:
            form._update_more()

    def _update_more(self) -> None:
        if self.more_section is None:
            return
        more = self.more_section.content
        rows = [n for n, lay in self.row_layouts.items() if lay is more.layout()]
        self.more_section.setVisible(any(
            self.row_layouts[n].isRowVisible(self.row_widgets.get(n, self.fields[n]))
            for n in rows))

    def set_values(self, values: Dict[str, Any]) -> None:
        """Set some fields by dotted name, without firing `field_changed`."""
        for path, value in values.items():
            if self._routed("set_values", path, value):
                continue
            target = self
            parts = path.split(".")
            for part in parts[:-1]:
                target = target.fields[part]
            field = target.fields[parts[-1]]
            field.blockSignals(True)
            field.set(value)
            field.blockSignals(False)
        self.changed.emit()
