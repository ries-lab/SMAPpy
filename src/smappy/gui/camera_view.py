"""What the fit thinks the camera is, and where every number came from.

A wrong conversion is invisible: the fit runs, the image looks like an image,
and every photon count in the file is out by a factor.  The only defence is
being able to look -- not at the five fields the form shows, which say what a
value *is*, but at where each one came from: this tag of this file, this
readout mode, the database, or something typed into the form.

Hidden by default, because the answer is usually "the camera was recognised
and the numbers are its own".  It is one button away for the times it is not.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

MISSING_COLOUR = "#c07000"
USED = "#2e9e4f"


class CameraParameters(QWidget):
    """The resolved camera parameters of the acquisition being fitted."""

    def __init__(self, plugin=None, settings=None, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Camera parameters")
        self.plugin = plugin
        self._settings = settings

        layout = QVBoxLayout(self)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        layout.addWidget(self.summary)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["parameter", "value", "from"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: gray")
        layout.addWidget(self.note)

        refresh = QPushButton("Refresh")
        refresh.setToolTip("read the file again: the camera, its readout mode "
                           "and every value the fit will use")
        refresh.clicked.connect(self.refresh)
        buttons = QHBoxLayout()
        buttons.addWidget(refresh)
        # offered when the database does not know the camera or its mode
        self.add_button = QPushButton("Add to camera database...")
        self.add_button.setToolTip("open the camera editor on this file, to add "
                                   "its camera or its readout mode once and "
                                   "for every file after it")
        self.add_button.clicked.connect(self.edit)
        self.add_button.hide()
        buttons.addWidget(self.add_button)
        self.manager_button = QPushButton("Microscope manager...")
        self.manager_button.setToolTip("the microscope entries: which devices set the pixel "
                                       "and the optics, and in which state")
        self.manager_button.clicked.connect(self.manage)
        buttons.addWidget(self.manager_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        # what opens the editor: the main window's, which tells the fitters
        # when a camera is added; a view on its own opens its own
        self.open_editor = None
        self.open_manager = None
        self._manager = None
        self._resolved = None
        self._editor = None
        self.resize(620, 420)

    # ------------------------------------------------------------------ data
    def show_for(self, plugin, settings) -> None:
        """Show the camera of this plugin's current source."""
        self.plugin, self._settings = plugin, settings
        self.refresh()
        self.show()
        self.raise_()

    def refresh(self) -> None:
        resolved = self._resolve()
        self._resolved = resolved
        self.table.setRowCount(0)
        self.add_button.hide()
        if resolved is None:
            self.summary.setText("<b>no acquisition</b>")
            self.note.setText("choose a file in the Localize tab; its camera is "
                              "read from the file's own metadata.")
            return
        self._fill(resolved)

    def _resolve(self):
        plugin, settings = self.plugin, self._settings
        if plugin is None or settings is None:
            return None
        try:
            return plugin.resolution(settings)
        except Exception:
            return None

    def _fill(self, resolved) -> None:
        from ..io.tiff import SIMULATED
        origin = (resolved.sources or {}).get("camera_name")
        if resolved.camera is None and origin is not None \
                and (origin.kind, origin.detail) == SIMULATED:
            # a simulation says exactly what its camera is: not a failure
            self.summary.setText(
                "<b>a simulation</b><br>its camera is not in the database and "
                "need not be: the simulation says what it is.")
        elif resolved.camera is None:
            self.summary.setText(
                "<b>camera not recognised</b><br>Nothing in this file matches a "
                "camera in the database, so only what the file itself says is "
                "known.  Pick one under <i>camera</i> in the Localize tab, state "
                "the values there, or add the camera to the database.")
            self.add_button.show()
        else:
            state = (f"<br>readout mode: {resolved.state_name}"
                     if resolved.state_name else
                     "<br><b>readout mode not recognised</b>: what depends on it "
                     "is missing below")
            if not resolved.state_name and any(
                    p.state for p in resolved.camera.parameters.values()):
                self.add_button.show()
            self.summary.setText(
                f"<b>{resolved.camera_name}</b>"
                f"<br>identified by {resolved.camera.identified_by(resolved.metadata)}"
                f"{state}")

        from .. import microscope_db
        scope = resolved.microscope
        self.summary.setText(self.summary.text() + "<br>microscope: " + microscope_db.one_line(
            scope.summary() if scope is not None else None))

        rows = resolved.rows()
        self.table.setRowCount(len(rows))
        for row, (name, value, source, used) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            text = "-" if value is None else _text(value)
            item = QTableWidgetItem(text)
            if value is None:
                item.setForeground(Qt.red if used else Qt.gray)
            self.table.setItem(row, 1, item)
            self.table.setItem(row, 2, QTableWidgetItem(source))
            if not used:
                for column in range(3):
                    cell = self.table.item(row, column)
                    cell.setForeground(Qt.gray)
                    cell.setToolTip("known, but not something the fit reads")

        if scope is not None:
            from .microscope_view import fill_rows
            first = self.table.rowCount()
            fill_rows(self.table, [(f"microscope: {what}", value, source)
                                   for what, value, source in scope.rows()], first)
            for row in range(first, self.table.rowCount()):
                for column in range(3):
                    self.table.item(row, column).setToolTip(
                        "the microscope's: the pixel above follows it")

        missing = resolved.missing
        if missing:
            self.note.setText(
                "The fit cannot run without " + ", ".join(missing) +
                ".  State them in the Localize tab, or add this camera to the "
                "database (Add to camera database).")
            self.note.setStyleSheet(f"color: {MISSING_COLOUR}")
        else:
            self.note.setText("Every value the fit needs is known.  Greyed rows "
                              "are parameters the database carries and the fit "
                              "does not read.")
            self.note.setStyleSheet("color: gray")


    def manage(self) -> None:
        """The microscope manager: the main window's, or one of this view's own."""
        if self.open_manager is not None:
            self.open_manager()
            return
        from .microscope_editor import MicroscopeManager
        if self._manager is None:
            self._manager = MicroscopeManager(parent=self)
        self._manager.show()
        self._manager.raise_()

    def edit(self) -> None:
        """The camera editor on the file this view is showing."""
        resolved = self._resolved
        if resolved is None:
            return
        if self.open_editor is not None:
            self.open_editor(resolved.metadata, resolved.camera)
            return
        from .camera_editor import open_editor
        first = self._editor is None
        self._editor = open_editor(resolved.metadata, resolved.camera, self,
                                   self._editor)
        if first:
            self._editor.saved.connect(lambda name: self.refresh())


def _text(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(v) for v in value)
    return str(value)


def open_for(plugin, settings, parent=None,
             window: Optional[CameraParameters] = None) -> CameraParameters:
    """Show the parameter view for this plugin, reusing the open one."""
    if window is None:
        window = CameraParameters(parent=parent)
    window.show_for(plugin, settings)
    return window
