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
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QLabel,
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
        layout.addWidget(refresh)
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
        self.table.setRowCount(0)
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
        if resolved.camera is None:
            self.summary.setText(
                "<b>camera not recognised</b><br>Nothing in this file matches a "
                "camera in the database, so only what the file itself says is "
                "known.  Pick one under <i>camera</i> in the Localize tab, or "
                "state the values there.")
        else:
            state = (f"<br>readout mode: {resolved.state_name}"
                     if resolved.state_name else
                     "<br><b>readout mode not recognised</b>: what depends on it "
                     "is missing below")
            self.summary.setText(
                f"<b>{resolved.camera_name}</b>"
                f"<br>identified by {resolved.camera.identified_by(resolved.metadata)}"
                f"{state}")

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

        missing = resolved.missing
        if missing:
            self.note.setText(
                "The fit cannot run without " + ", ".join(missing) +
                ".  State them in the Localize tab, or add this camera to "
                "cameras.json.")
            self.note.setStyleSheet(f"color: {MISSING_COLOUR}")
        else:
            self.note.setText("Every value the fit needs is known.  Greyed rows "
                              "are parameters the database carries and the fit "
                              "does not read.")
            self.note.setStyleSheet("color: gray")


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
