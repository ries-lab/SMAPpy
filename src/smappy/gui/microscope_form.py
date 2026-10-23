"""The microscope control: a folded section where a file is opened, its values in it.

Every place a stack is opened -- the fitters, *PSF calibration* -- has the same control
(`docs/microscope-database.md`, "Second round").  Folded, as it starts, the section's title is
one line: the microscope and its state as found from the file (`microscope_db.one_line`),
orange when something the function needs is missing or nothing was recognised.  Open, it lists
the microscope, the camera and only the values the function uses, each showing the value found
-- grey -- with where it comes from in its tooltip.  The values are locked: the lock opens them,
typing over one makes it the user's (its label coloured, ↺ beside it to give it back), so that
overriding what the file and the database say is a decision.  The microscope stays choosable.

It replaced a window behind a *Microscope...* button (2026-10): the window put the values a
click and a window away from where they are used, and its grey list of sources under the values
was read by nobody.  No *auto* boxes either: every value here has been found somewhere, and the
field shows it (`_Field.type_over`).

It is a `SettingsForm` -- a settings part of kind ``"microscope"`` -- whose rows are an inner
form's: the dotted names, the hints, the choices, what is shown, saving and restoring are a
form's, unchanged, so a chain or a GUI file that set ``camera.conversion`` still does.  The
function says what it found through one hint on the part itself, a dict (`set_details`): the
line, a status, and each value's source.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QToolButton

from .params import SettingsForm

#: an edited value's label: the user's word over the file's
EDITED = "color: #1d5fb4; font-weight: 600"
#: the line, by status: all known, something the function uses missing, nothing recognised
STATUS = {"ok": "", "missing": "color: #c07000", "unrecognised": "color: #c07000",
          "fallback": "color: #c07000"}
#: the field that chooses where the values come from: never locked
CHOOSER = "microscope"


class MicroscopeForm(SettingsForm):
    """A settings part shown folded as one line, its values inside (module docstring)."""

    def __init__(self, settings_cls: type, specs=None, parent=None):
        super().__init__(settings_cls, {}, parent)       # an empty form: the inner one holds the rows
        self._sources: Dict[str, str] = {}
        self.inner = SettingsForm(settings_cls, specs)
        # the inner form's tables are this form's: every SettingsForm method works on them
        self.fields, self.labels = self.inner.fields, self.inner.labels
        self.sections, self.row_layouts = self.inner.sections, self.inner.row_layouts
        self.more_section = self.inner.more_section
        self.inner.changed.connect(self.changed)
        self.inner.field_changed.connect(self.field_changed)
        self.inner.changed.connect(self._paint)
        for name, field in self._value_fields().items():
            if hasattr(field, "type_over"):
                field.type_over()

        box = self.layout()
        # said inside only when something is wrong: the title says the rest
        self.line = QLabel("no file")
        self.line.setWordWrap(True)
        self.line.setVisible(False)
        box.addWidget(self.line)
        box.addWidget(self.inner)
        buttons = QHBoxLayout()
        self.lock = QToolButton()
        self.lock.setCheckable(True)
        self.lock.setToolTip("locked: the values are the file's and the database's.  Unlock to "
                             "type over one: it wins over them, and is kept with the settings")
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
        self.status = "ok"
        #: the line: the section's title
        self.title = "no file"
        self.lock.setChecked(True)
        self._locked(True)

    # ------------------------------------------------------------------ the lock
    def _value_fields(self):
        return {n: f for n, f in self.fields.items() if n != CHOOSER}

    def _locked(self, locked: bool) -> None:
        self.lock.setText("\U0001F512 locked" if locked else "\U0001F513 editing")
        for field in self._value_fields().values():
            if hasattr(field, "set_locked"):
                field.set_locked(locked)
            else:
                field.setEnabled(not locked)

    def _paint(self) -> None:
        """Colour the label of each value typed over the file's: it is the user's word; and say in
        each label's tooltip where its value comes from."""
        for name, field in self.fields.items():
            label = self.labels.get(name)
            if label is None:
                continue
            try:
                edited = name != CHOOSER and field.value() not in (None, "")
            except ValueError:
                edited = True
            label.setStyleSheet(EDITED if edited else "")
            source = self._sources.get(name, "")
            label.setToolTip(("edited: wins over the file and the database" if edited
                              else f"found: {source}" if source else ""))

    def set(self, settings) -> None:
        super().set(settings)
        self._paint()
        # a settings file with values typed opens unlocked, so they are seen as the user's
        if any(f.value() not in (None, "") for f in self._value_fields().values()):
            self.lock.setChecked(False)

    # ---------------------------------------------------------------- what was found
    def set_details(self, details: Optional[Dict[str, Any]]) -> None:
        """The function's account of the file: ``line``, ``status`` (`STATUS`), ``values`` (the
        values used, in a sentence: said inside when something is wrong) and ``sources`` ({field:
        where its value comes from}, in each label's tooltip).  The line is the section's title."""
        details = details or {}
        self.status = details.get("status", "ok")
        text = self.title = details.get("line", "no file")
        section = self._section()
        if section is not None:
            from .widgets import CONTROL_WIDTH
            from PySide6.QtCore import Qt
            shown = section.button.fontMetrics().elidedText(text, Qt.ElideRight, CONTROL_WIDTH - 90)
            section.set_title(shown)
            section.button.setToolTip(text if shown != text else "")
            style = "border: none; font-weight: bold;" + (" " + STATUS[self.status] + ";"
                                                         if STATUS.get(self.status) else "")
            section.button.setStyleSheet(f"QToolButton {{ {style} }}")
        wrong = STATUS.get(self.status, "")
        self.line.setVisible(bool(wrong))
        self.line.setStyleSheet(wrong)
        self.line.setText({"missing": "missing: " + details.get("values", "").split("missing: ")[-1],
                           "fallback": "nothing in the file recognises a microscope: the last one "
                                       "used stands in",
                           "unrecognised": "no microscope recognised: choose one, or unlock and type "
                                           "the values"}.get(self.status, ""))
        self._sources = dict(details.get("sources") or {})
        self._paint()

    def _shown(self, name: str) -> bool:
        layout = self.row_layouts.get(name)
        return layout is None or layout.isRowVisible(self.fields[name])

    def _section(self):
        """The collapsible section this form is the content of, if any."""
        from .widgets import CollapsibleSection
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, CollapsibleSection) and parent.content is self:
                return parent
            parent = parent.parent()
        return None

    # ------------------------------------------------------------------ the manager
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
