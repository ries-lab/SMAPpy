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
from .widgets import CollapsibleSection


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
        """The list the plugin works out from the session; see `Plugin.describe`."""
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

    def _on_auto(self, on: bool) -> None:
        self.widget.setEnabled(not on)
        if on and self._hint and isinstance(self.widget, QLineEdit):
            self.widget.setText(self._hint)      # leaving it blank says less
        self._paint()
        self.changed.emit()

    def _paint(self) -> None:
        """Grey while the value on show is a hint rather than a setting."""
        if isinstance(self.widget, QLineEdit):
            hinting = self.auto is not None and self.auto.isChecked()
            self.widget.setStyleSheet("color: gray; font-style: italic"
                                      if hinting else "")

    def set_hint(self, value: Any) -> None:
        """What *auto* currently resolves to, for showing greyed out.

        A hint is never a value: `value()` still returns None while auto is
        on, so nothing here can turn a read-from-the-file number into one the
        user is taken to have set.
        """
        if isinstance(self.widget, QComboBox) and self.spec.optional \
                and self._values and self._values[0] is None:
            # a choice on auto says what auto is: "auto (on)"
            labels = dict(zip(self._values, (self.widget.itemText(i)
                                             for i in range(self.widget.count()))))
            shown = labels.get(value, None if value is None else str(value))
            self.widget.setItemText(0, f"auto ({shown})" if shown else "auto")
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
        current = self.widget.text()
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
        if self.spec.type is int:
            return int(float(text))
        if self.spec.type is float:
            return float(text)
        return text


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
    becoming the list this table happened to have.  The names come from the
    session (`FormState.choices`); a name in the value that the session does
    not offer is kept, ticked, and marked as not there.
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

    def _tick_all(self, on: bool) -> None:
        blocked = self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.list.blockSignals(blocked)
        self.changed.emit()

    def _fill(self, names: List[str], ticked: List[str]) -> None:
        from PySide6.QtWidgets import QListWidgetItem
        blocked = self.list.blockSignals(True)
        self.list.clear()
        everything = not ticked
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
        self._fill(self._names, list(value or []))

    def value(self) -> List[str]:
        items = [self.list.item(i) for i in range(self.list.count())]
        if not items:
            return list(self._value)
        ticked = [it.data(Qt.UserRole) for it in items if it.checkState() == Qt.Checked]
        return [] if len(ticked) == len(items) else ticked


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
        # `specs if None`, not `specs or`: {} is a plugin that declares no
        # settings at all -- a loader with sensible defaults, say -- and it
        # must give an empty form rather than fall back to introspection
        for name, spec in (param_specs(settings_cls) if specs is None
                           else specs).items():
            if spec.info.hidden:
                continue
            if spec.children is not None:
                part = SettingsForm(spec.type, spec.children)
                part.field_changed.connect(lambda sub, n=name: self.field_changed.emit(f"{n}.{sub}"))
                part.changed.connect(self.changed)
                self.fields[name] = part
                main = None
                section = CollapsibleSection(spec.info.label or name, part,
                                             expanded=not (spec.info.advanced
                                                           or spec.info.collapsed))
                self.sections[name] = section
                layout.addWidget(section)
                continue
            w = (LayersField(spec) if spec.info.kind == "layers"
                 else ColumnsField(spec) if spec.info.kind == "columns" else _Field(spec))
            w.changed.connect(self.changed)
            w.changed.connect(lambda n=name: self.field_changed.emit(n))
            self.fields[name] = w
            label = QLabel(spec.info.label or name)
            if spec.info.help:
                label.setToolTip(spec.info.help)
            self.labels[name] = label
            if spec.info.advanced:
                more.addRow(label, w)
                self.row_layouts[name] = more
                continue
            if main is None:
                main = form_layout()
                layout.addLayout(main)
            main.addRow(label, w)
            self.row_layouts[name] = main
        if more.rowCount():
            box = QWidget()
            box.setLayout(more)
            self.more_section = CollapsibleSection("more", box, expanded=False)
            layout.addWidget(self.more_section)

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

    def set_hints(self, values: Dict[str, Any]) -> None:
        """Show, by dotted name, what each field's *auto* resolves to.

        Unknown names are ignored rather than raised: a hint is an extra, and
        a plugin that offers one for a field it no longer has must not break
        the form the way a stale saved value must not.
        """
        for path, value in values.items():
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

    def set_choices(self, lists: Dict[str, Any]) -> None:
        """Replace, by dotted name, the lists of fields whose choices the
        session decides (`Plugin.describe`).  Unknown names are ignored, as
        for hints."""
        for path, choices in lists.items():
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
            target = self
            parts = path.split(".")
            try:
                for part in parts[:-1]:
                    target = target.fields[part]
                name = parts[-1]
                if name in target.sections:
                    target.sections[name].setVisible(bool(on))
                    continue
                target.row_layouts[name].setRowVisible(target.fields[name], bool(on))
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
            self.row_layouts[n].isRowVisible(self.fields[n]) for n in rows))

    def set_values(self, values: Dict[str, Any]) -> None:
        """Set some fields by dotted name, without firing `field_changed`."""
        for path, value in values.items():
            target = self
            parts = path.split(".")
            for part in parts[:-1]:
                target = target.fields[part]
            field = target.fields[parts[-1]]
            field.blockSignals(True)
            field.set(value)
            field.blockSignals(False)
        self.changed.emit()
