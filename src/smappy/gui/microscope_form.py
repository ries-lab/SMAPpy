"""The microscope control: one line where a file is opened, its values a button away.

Every place a stack is opened -- the fitters, *PSF calibration* -- has the same control
(`docs/microscope-database.md`, "Second round").  In the panel it is a line, the
microscope and its state as found from the file (`microscope_db.one_line`), the values the
function uses below it in grey, and *Microscope...*.  The button opens one window: the
microscope and the camera, recognised or chosen, and the values with where each comes
from.  The values are locked: a lock opens them for editing, so that overriding what the
file and the database say is a decision, and an edited value's label is coloured.

It is a `SettingsForm` -- a settings part of kind ``"microscope"`` -- whose rows live in
the window instead of the panel: the dotted names, the hints, the choices, what is shown,
saving and restoring are a form's, unchanged, so a chain or a GUI file that set
``camera.conversion`` still does.  The function says what it found through one hint on
the part itself, a dict (`set_details`): the line, a status, the values in a sentence,
and each value's source.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget

from .params import SettingsForm

#: an edited value's label: the user's word over the file's
EDITED = "color: #1d5fb4; font-weight: 600"
#: the line, by status: all known, something the function uses missing, nothing recognised
STATUS = {"ok": "color: gray", "missing": "color: #c07000", "unrecognised": "color: #c07000",
          "fallback": "color: #c07000"}
#: the fields that choose rather than state a value, never locked
CHOOSERS = ("microscope", "camera")


class MicroscopeForm(SettingsForm):
    """A settings part shown as one line and a window (module docstring)."""

    def __init__(self, settings_cls: type, specs=None, parent=None):
        super().__init__(settings_cls, {}, parent)       # an empty form: this widget is the line
        self.inner = SettingsForm(settings_cls, specs)
        # the inner form's tables are this form's: every SettingsForm method works on them
        self.fields, self.labels = self.inner.fields, self.inner.labels
        self.sections, self.row_layouts = self.inner.sections, self.inner.row_layouts
        self.more_section = self.inner.more_section
        self.inner.changed.connect(self.changed)
        self.inner.field_changed.connect(self.field_changed)
        self.inner.changed.connect(self._paint)

        line = self.layout()
        self.line = QLabel("no file")
        self.line.setWordWrap(True)
        line.addWidget(self.line)
        self.values_line = QLabel("")
        self.values_line.setWordWrap(True)
        self.values_line.setStyleSheet("color: gray")
        line.addWidget(self.values_line)
        row = QHBoxLayout()
        self.button = QPushButton("Microscope...")
        self.button.setToolTip("the microscope and the camera found for this file, the values "
                               "used and where each comes from; choose them, or unlock to edit")
        self.button.clicked.connect(self.open_window)
        row.addWidget(self.button)
        row.addStretch(1)
        line.addLayout(row)

        self.window = QWidget(self, Qt.Window)
        self.window.setWindowTitle("Microscope")
        box = QVBoxLayout(self.window)
        self.window_line = QLabel("")
        self.window_line.setWordWrap(True)
        box.addWidget(self.window_line)
        box.addWidget(self.inner)
        self.sources = QLabel("")
        self.sources.setWordWrap(True)
        self.sources.setStyleSheet("color: gray")
        box.addWidget(self.sources)
        buttons = QHBoxLayout()
        self.lock = QToolButton()
        self.lock.setCheckable(True)
        self.lock.setToolTip("locked: the values are the file's and the database's.  Unlock to "
                             "type one: it wins over them, for this run, and is kept with its "
                             "settings")
        self.lock.toggled.connect(self._locked)
        buttons.addWidget(self.lock)
        buttons.addStretch(1)
        self.manager_button = QPushButton("Microscope manager...")
        self.manager_button.setToolTip("the microscope entries: devices, states, values, cameras")
        self.manager_button.clicked.connect(self._manager)
        buttons.addWidget(self.manager_button)
        box.addLayout(buttons)
        #: what *Microscope manager...* calls: the main window's, else a manager of its own
        self.open_manager = None
        self._manager_window = None
        self._sources: Dict[str, str] = {}
        self.lock.setChecked(True)
        self._locked(True)

    # ------------------------------------------------------------------ the lock
    def _value_fields(self):
        return {n: f for n, f in self.fields.items() if n not in CHOOSERS}

    def _locked(self, locked: bool) -> None:
        self.lock.setText("\U0001F512 locked" if locked else "\U0001F513 editing")
        for field in self._value_fields().values():
            field.setEnabled(not locked)

    def _paint(self) -> None:
        """Colour the label of each value typed over the file's: it is the user's word."""
        for name, field in self._value_fields().items():
            label = self.labels.get(name)
            if label is None:
                continue
            try:
                edited = field.value() is not None
            except ValueError:
                edited = True
            label.setStyleSheet(EDITED if edited else "")
            label.setToolTip(("edited: wins over the file and the database.  " if edited else "")
                             + self._sources.get(name, ""))

    def set(self, settings) -> None:
        super().set(settings)
        self._paint()
        # a settings file with values typed opens unlocked, so they are seen as the user's
        if any(f.value() is not None for f in self._value_fields().values()):
            self.lock.setChecked(False)

    # ---------------------------------------------------------------- what was found
    def set_details(self, details: Optional[Dict[str, Any]]) -> None:
        """The function's account of the file: ``line``, ``status`` (`STATUS`), ``values`` (the
        values used, in a sentence) and ``sources`` ({field: where its value comes from})."""
        details = details or {}
        status = details.get("status", "ok")
        text = details.get("line", "no file")
        self.line.setText(text)
        self.line.setStyleSheet(STATUS.get(status, ""))
        self.window_line.setText(text)
        self.window_line.setStyleSheet(STATUS.get(status, ""))
        self.values_line.setText(details.get("values", ""))
        self._sources = dict(details.get("sources") or {})
        self.sources.setText("\n".join(f"{name}: {source}" for name, source in self._sources.items()
                                       if name in self.fields and self._shown(name)))
        self._paint()

    def _shown(self, name: str) -> bool:
        layout = self.row_layouts.get(name)
        return layout is None or layout.isRowVisible(self.fields[name])

    # ------------------------------------------------------------------ windows
    def open_window(self) -> None:
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def _manager(self) -> None:
        opener = self.open_manager
        if opener is None:
            panel = self.parent()
            while panel is not None and not hasattr(panel, "session"):
                panel = panel.parent()
            openers = getattr(getattr(panel, "session", None), "window_openers", {}) or {}
            opener = openers.get("microscope manager")
        if opener is not None:
            opener()
            return
        from .microscope_editor import MicroscopeManager
        if self._manager_window is None:
            self._manager_window = MicroscopeManager(parent=self)
        self._manager_window.show()
        self._manager_window.raise_()
