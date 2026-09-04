"""A settings dataclass as a form, and the form back as a dataclass."""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QSpinBox, QVBoxLayout, QWidget)

from ..plugins import ParamSpec, param_specs
from .widgets import CollapsibleSection


class _Field(QWidget):
    """One parameter.  Optional ones get an "auto" box that stands for None."""

    changed = Signal()

    def __init__(self, spec: ParamSpec):
        super().__init__()
        self.spec = spec
        info = spec.info
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.auto: Optional[QCheckBox] = None
        if info.choices:
            self.widget = QComboBox()
            self._values = []
            choices = list(info.choices)
            if spec.optional:
                choices.insert(0, (None, "auto"))
            for choice in choices:
                value, label = choice if isinstance(choice, tuple) else (choice, str(choice))
                self.widget.addItem(label, value)
                self._values.append(value)
            self.widget.currentIndexChanged.connect(self.changed)
        elif spec.type is bool and spec.optional:   # three states: on, off, auto
            self.widget = QCheckBox(tristate=True)
            self.widget.stateChanged.connect(self.changed)
        elif spec.type is bool:
            self.widget = QCheckBox()
            self.widget.toggled.connect(self.changed)
        elif spec.type is int and not spec.optional:
            self.widget = QSpinBox()
            self.widget.setRange(int(info.min) if info.min is not None else -10**9,
                                 int(info.max) if info.max is not None else 10**9)
            if info.step:
                self.widget.setSingleStep(int(info.step))
            self.widget.valueChanged.connect(self.changed)
        else:
            # a line edit for floats and optionals: it takes "1e-7", and it is
            # the compact thing.  Validation happens in `value`.
            self.widget = QLineEdit()
            self.widget.setMaximumWidth(80)
            self.widget.editingFinished.connect(self.changed)
        row.addWidget(self.widget)
        if spec.optional and not info.choices and spec.type is not bool:
            self.auto = QCheckBox("auto")
            self.auto.toggled.connect(self._on_auto)
            row.addWidget(self.auto)
        if info.unit:
            row.addWidget(QLabel(info.unit))
        row.addStretch(1)
        if info.help:
            self.setToolTip(info.help)
        self.set(spec.default)

    def _on_auto(self, on: bool) -> None:
        self.widget.setEnabled(not on)
        self.changed.emit()

    def set(self, value: Any) -> None:
        if self.auto is not None:
            self.auto.setChecked(value is None)
            if value is None:
                return
        w = self.widget
        if isinstance(w, QComboBox):
            w.setCurrentIndex(self._values.index(value) if value in self._values else 0)
        elif isinstance(w, QCheckBox):
            if w.isTristate():
                w.setCheckState(Qt.PartiallyChecked if value is None
                                else Qt.Checked if value else Qt.Unchecked)
            else:
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
            if w.isTristate():
                state = w.checkState()
                return None if state == Qt.PartiallyChecked else state == Qt.Checked
            return w.isChecked()
        if isinstance(w, QSpinBox):
            return w.value()
        text = w.text().strip()
        if self.spec.type is int:
            return int(float(text))
        if self.spec.type is float:
            return float(text)
        return text


class SettingsForm(QWidget):
    """The fields of a settings dataclass; ``value()`` builds the instance.

    Fields marked advanced sit under a collapsed "more" section.
    """

    changed = Signal()

    def __init__(self, settings_cls: type, specs: Optional[Dict[str, ParamSpec]] = None,
                 parent=None):
        super().__init__(parent)
        self.settings_cls = settings_cls
        self.fields: Dict[str, _Field] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        main, more = QFormLayout(), QFormLayout()
        for form in (main, more):
            form.setContentsMargins(0, 0, 0, 0)
            form.setVerticalSpacing(2)
        for name, spec in (specs or param_specs(settings_cls)).items():
            if spec.info.hidden:
                continue
            w = _Field(spec)
            w.changed.connect(self.changed)
            self.fields[name] = w
            (more if spec.info.advanced else main).addRow(spec.info.label or name, w)
        layout.addLayout(main)
        if more.rowCount():
            box = QWidget()
            box.setLayout(more)
            layout.addWidget(CollapsibleSection("more", box, expanded=False))

    def value(self):
        return self.settings_cls(**{n: f.value() for n, f in self.fields.items()})

    def set(self, settings) -> None:
        for name, f in self.fields.items():
            f.set(getattr(settings, name))
