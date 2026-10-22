"""The microscope manager: the microscope entries, edited and checked against a file.

Opened from the menu (*File > Microscope manager...*), and from what a file opener shows of
the microscope it found (`gui.microscope_view`).  An entry is what `microscope_db` resolves a
file with: the tags that recognise the microscope, its devices and their named states, the
parameters that follow them, the arms of a split and the cameras it has had
(`docs/microscope-database.md`).  Here each part is a table, and what does not fit a cell
is typed the way the camera editor types a map: ``on: 28000; off: 4000``, a pixel as
``127 116``.

An entry starts from an EMU configuration (*Import EMU configuration...*, `io.emu`), from
scratch, or from a shipped one; *Check with a file...* shows, as it is edited, what the
entry makes of a stack.  Saving writes the user's ``microscopes.json`` and keeps what the
entry said before as a version, dated (*old version until*: the EMU configuration's date
after an import, else today), so older files are still read as they were taken.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .. import microscope_db
from ..microscope_db import FROM_NAME, READS, Microscope

# ------------------------------------------------------------- text in a cell

def value_text(value: Any) -> str:
    """A value as it is typed: a number, numbers by spaces, or text."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(value_text(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def parse_value(text: str) -> Any:
    """What `value_text` wrote: ``1.43`` -> 1.43, ``127 116`` -> [127.0, 116.0], else the text."""
    text = str(text).strip()
    if not text:
        return None
    parts = text.split()
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return text
    return numbers[0] if len(numbers) == 1 else numbers


def map_text(table: Dict[str, Any]) -> str:
    """``{"on": 28000, "off": 4000}`` -> ``on: 28000; off: 4000``."""
    return "; ".join(f"{key}: {value_text(value)}" for key, value in (table or {}).items())


def parse_map(text: str, values: bool = True) -> Dict[str, Any]:
    """What `map_text` wrote.  Split on the last colon of each part, so a state's name may hold
    one; `values` False keeps what follows as text (a state's reading is compared as text)."""
    out: Dict[str, Any] = {}
    for part in str(text).split(";"):
        if ":" not in part:
            continue
        key, value = part.rsplit(":", 1)
        out[key.strip()] = parse_value(value) if values else value.strip()
    return out


def follows_text(raw: Any) -> tuple:
    """A parameter as two cells: what it follows ("" : fixed) and its value or table."""
    if isinstance(raw, dict) and "device" in raw:
        values = raw.get("values")
        return raw["device"], FROM_NAME if values == FROM_NAME else map_text(values or {})
    if isinstance(raw, dict):
        return "", value_text(raw.get("fixed"))
    return "", value_text(raw)


def parse_follows(device: str, text: str) -> Any:
    device, text = device.strip(), str(text).strip()
    if not device:
        return parse_value(text)
    return {"device": device,
            "values": FROM_NAME if text == FROM_NAME else parse_map(text)}


# ----------------------------------------------------------------- the window

def _table(headers: List[str], tip: str) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setToolTip(tip)
    header = table.horizontalHeader()
    for column in range(len(headers) - 1):
        header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(len(headers) - 1, QHeaderView.Stretch)
    return table


def _cell(table: QTableWidget, row: int, column: int) -> str:
    item = table.item(row, column)
    return item.text() if item is not None else ""


def _put(table: QTableWidget, cells: List[str]) -> None:
    row = table.rowCount()
    table.insertRow(row)
    for column, text in enumerate(cells):
        table.setItem(row, column, QTableWidgetItem(str(text)))


TABS = (
    ("identify", ["tag", "value"],
     "tags that must all match for a file to be this microscope: take a device that stays "
     "with it (a laser's serial number), not the camera, which moves"),
    ("devices", ["name", "tags", "read", "tolerance", "states", "EMU control"],
     "each device: the Micro-Manager tags it is read from (an older name after a ' | '), how "
     "(exact, nearest, value), and its states as 'name: reading; ...'"),
    ("parameters", ["parameter", "follows", "value"],
     "NA, n_immersion, wavelength_nm, pixel_nm, modality ...: a value, or a device it follows "
     "and 'state: value; ...' ('from the name' reads a filter's band)"),
    ("channels", ["arm", "filter", "side"],
     "the split: its arms, transmitted first, the filter device each follows and the half of "
     "the chip it lands on with this camera (left, right, upper, lower)"),
)


class MicroscopeManager(QWidget):
    """The entries on the left, the one chosen as tables on the right, and what it makes of a
    file below.  `saved` carries the name of a microscope written to the user's file."""

    saved = Signal(str)

    def __init__(self, parent=None, path: Optional[str] = None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Microscope manager")
        self._versions: List[Dict[str, Any]] = []
        self._metadata: Dict[str, Any] = {}
        self._file = ""
        layout = QHBoxLayout(self)

        left = QVBoxLayout()
        self.entries = QListWidget()
        self.entries.setToolTip("the microscopes known here: smappy's and your own")
        self.entries.currentItemChanged.connect(lambda item, _: self.load(item.data(Qt.UserRole))
                                                if item is not None else None)
        left.addWidget(self.entries, 1)
        new = QPushButton("New")
        new.setToolTip("an empty entry")
        new.clicked.connect(lambda: self.show_entry(Microscope(name="")))
        left.addWidget(new)
        emu = QPushButton("Import EMU configuration...")
        emu.setToolTip("devices and their state names from an EMU (ht-SMLM) configuration: a new "
                       "entry, or the chosen one updated from a newer configuration")
        emu.clicked.connect(self.import_emu)
        left.addWidget(emu)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setToolTip("how the microscope is called; a user entry of a shipped one's name "
                             "takes its place")
        form.addRow("name", self.name)
        self.comment = QLineEdit()
        form.addRow("comment", self.comment)
        camera_row = QHBoxLayout()
        self.camera = QComboBox()
        self.camera.setToolTip("the camera the microscope is set up with, from the camera "
                               "database; a new camera is a new setup: save, and the old one is "
                               "kept for older files")
        camera_row.addWidget(self.camera, 1)
        self.camera_button = QPushButton("Edit camera...")
        self.camera_button.setToolTip("the camera editor: the camera's conversion and offset per "
                                      "readout mode, on the file being checked with")
        self.camera_button.clicked.connect(self.edit_camera)
        camera_row.addWidget(self.camera_button)
        form.addRow("camera", camera_row)
        right.addLayout(form)
        self._fill_cameras("")
        self.camera.activated.connect(lambda _: self.check())
        self._camera_editor = None

        self.tabs = QTabWidget()
        self.tables: Dict[str, QTableWidget] = {}
        for title, headers, tip in TABS:
            page = QWidget()
            box = QVBoxLayout(page)
            table = _table(headers, tip)
            self.tables[title] = table
            box.addWidget(table)
            buttons = QHBoxLayout()
            add, remove = QPushButton("Add row"), QPushButton("Remove row")
            add.clicked.connect(lambda _=False, t=table: t.insertRow(t.rowCount()))
            remove.clicked.connect(lambda _=False, t=table: t.removeRow(t.currentRow()))
            buttons.addWidget(add)
            buttons.addWidget(remove)
            buttons.addStretch(1)
            box.addLayout(buttons)
            self.tabs.addTab(page, title)
        right.addWidget(self.tabs, 2)

        check = QHBoxLayout()
        self.check_button = QPushButton("Check with a file...")
        self.check_button.setToolTip("what this entry, as it stands, makes of a stack: whether it "
                                     "recognises it, each device's state, each value")
        self.check_button.clicked.connect(self.choose_file)
        check.addWidget(self.check_button)
        self.file_label = QLabel("no file")
        self.file_label.setStyleSheet("color: gray")
        check.addWidget(self.file_label, 1)
        right.addLayout(check)
        self.result = _table(["", "value", "from"], "what the entry makes of the file")
        self.result.setEditTriggers(QAbstractItemView.NoEditTriggers)
        right.addWidget(self.result, 1)

        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("old version until"))
        self.until = QLineEdit()
        self.until.setPlaceholderText(datetime.date.today().isoformat())
        self.until.setToolTip("saving a changed entry keeps the old one for files taken before "
                              "this day: the EMU configuration's date after an import, else today")
        save_row.addWidget(self.until)
        self.versions_label = QLabel("")
        self.versions_label.setStyleSheet("color: gray")
        save_row.addWidget(self.versions_label, 1)
        self.save_button = QPushButton("Save")
        self.save_button.setToolTip("write the entry into your microscopes.json")
        self.save_button.clicked.connect(self.save)
        save_row.addWidget(self.save_button)
        right.addLayout(save_row)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        right.addWidget(self.message)
        layout.addLayout(right, 3)

        for table in self.tables.values():
            table.itemChanged.connect(lambda _: self.check())
        self.name.textChanged.connect(lambda _: self.check())
        self.refresh_list()
        if path:
            self.check_file(path)
        self.resize(1000, 640)

    def _fill_cameras(self, current: str) -> None:
        from ..camera_db import database as cameras
        self.camera.clear()
        self.camera.addItem("(none)", "")
        names = cameras(reload=False).names()
        for name in names:
            self.camera.addItem(name, name)
        if current and current not in names:
            found = cameras().get(current)             # a former name of a camera
            current = found.name if found is not None else current
            if self.camera.findData(current) < 0:
                self.camera.addItem(current, current)
        self.camera.setCurrentIndex(max(self.camera.findData(current), 0))

    def edit_camera(self) -> None:
        """The camera editor on the chosen camera and the file being checked with."""
        from ..camera_db import database as cameras
        from .camera_editor import open_editor
        first = self._camera_editor is None
        self._camera_editor = open_editor(self._metadata or None,
                                          cameras().get(self.camera.currentData() or ""),
                                          self, self._camera_editor)
        if first:
            self._camera_editor.saved.connect(lambda name: (self._fill_cameras(name), self.check()))

    # -------------------------------------------------------------- the list
    def refresh_list(self, select: str = "") -> None:
        mine = set()
        try:
            from ..camera_db import user_file
            mine = set(microscope_db.MicroscopeDatabase.load(user_file()).names()) \
                if user_file().exists() else set()
        except (OSError, ValueError):
            pass
        self.entries.blockSignals(True)
        self.entries.clear()
        for name in microscope_db.database(reload=True).names():
            item = QListWidgetItem(name + ("  (yours)" if name in mine else ""))
            item.setData(Qt.UserRole, name)
            self.entries.addItem(item)
            if name == select:
                self.entries.setCurrentItem(item)
        self.entries.blockSignals(False)

    def load(self, name: str) -> None:
        found = microscope_db.database().get(name)
        if found is not None:
            self.show_entry(found)

    # ---------------------------------------------------------- entry <-> tables
    def show_entry(self, entry: Microscope) -> None:
        for table in self.tables.values():
            table.blockSignals(True)
            table.setRowCount(0)
        self.name.setText(entry.name)
        self.comment.setText(entry.comment)
        for tag, value in entry.identify.items():
            _put(self.tables["identify"], [tag, value])
        for name, device in entry.devices.items():
            _put(self.tables["devices"], [name, " | ".join(device.tags), device.read,
                                          value_text(device.tolerance), map_text(device.states),
                                          device.emu])
        for name, raw in entry.parameters.items():
            _put(self.tables["parameters"], [name, *follows_text(raw)])
        for channel in entry.channels:
            side = map_text(channel.side) if isinstance(channel.side, dict) else channel.side
            _put(self.tables["channels"], [channel.name, channel.filter, side])
        self._fill_cameras(entry.camera)
        for table in self.tables.values():
            table.blockSignals(False)
        self._versions = list(entry.versions)
        self.versions_label.setText(
            "versions until " + ", ".join(str(v.get("until")) for v in self._versions)
            if self._versions else "no older versions")
        self.check()

    def entry(self) -> Microscope:
        """The entry as the tables say; a ValueError naming what does not parse."""
        rows = lambda key: [[_cell(self.tables[key], r, c)
                             for c in range(self.tables[key].columnCount())]
                            for r in range(self.tables[key].rowCount())]
        devices = {}
        for name, tags, read, tolerance, states, emu in rows("devices"):
            if not name.strip():
                continue
            if read.strip() and read.strip() not in READS:
                raise ValueError(f"device {name}: read is one of {', '.join(READS)}, not {read!r}")
            devices[name.strip()] = {
                "tag": [t.strip() for t in tags.split("|") if t.strip()],
                "read": read.strip() or "exact", "states": parse_map(states, values=False),
                **({"tolerance": float(tolerance)} if tolerance.strip() else {}),
                **({"emu": emu.strip()} if emu.strip() else {})}
        raw = {
            "name": self.name.text().strip(), "comment": self.comment.text().strip(),
            "identify": {tag.strip(): value.strip() for tag, value in rows("identify") if tag.strip()},
            "devices": devices,
            "parameters": {name.strip(): parse_follows(device, value)
                           for name, device, value in rows("parameters") if name.strip()},
            "channels": {arm.strip(): {"filter": filter_.strip(),
                                       "side": parse_map(side, values=False) if ":" in side
                                       else side.strip()}
                         for arm, filter_, side in rows("channels") if arm.strip()},
            "camera": self.camera.currentData() or "", "versions": self._versions,
        }
        return Microscope.from_dict(raw)

    # ---------------------------------------------------------------- checking
    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "A stack to check the entry with", "",
                                              "Image stacks (*.tif *.tiff *.ome.tif);;All files (*)")
        if path:
            self.check_file(path)

    def check_file(self, path: str) -> None:
        try:
            from ..io.tiff import camera_tags, open_stack
            self._metadata = camera_tags(open_stack(path))
            self._file = str(path)
        except Exception as error:
            self._metadata, self._file = {}, ""
            self.message.setText(f"{path}: {error}")
        self.check()

    def check(self) -> None:
        """What the entry as typed makes of the file being checked with."""
        self.result.setRowCount(0)
        if not self._file:
            self.file_label.setText("no file")
            return
        from pathlib import Path
        try:
            entry = self.entry()
        except (ValueError, TypeError) as error:
            self.file_label.setText(f"{Path(self._file).name}: {error}")
            return
        from ..camera_db import database as cameras
        camera = cameras().identify(self._metadata)
        found = entry.resolve(self._metadata, camera=camera.name if camera else "", channels=2)
        recognised = entry.identifies(self._metadata)
        self.file_label.setText(f"{Path(self._file).name}: "
                                + ("recognised" if recognised else
                                   "not recognised by the identify tags (shown as if chosen)"))
        from .microscope_view import fill_rows
        fill_rows(self.result, found.rows())

    # ------------------------------------------------------------------ EMU
    def import_emu(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "An EMU configuration", "",
                                              "EMU configurations (*.uicfg);;All files (*)")
        if path:
            self.import_emu_file(path)

    def import_emu_file(self, path: str, three_d: Optional[str] = None, three_d_in: str = "on",
                        update: Optional[bool] = None) -> List[str]:
        """Import `path`: a new entry, or the shown one's devices updated from it.  The 3D lens
        and its *in* state are asked for when not given (EMU has no role for them)."""
        from ..io import emu
        config = emu.read(path)
        current = self.name.text().strip()
        if update is None:
            update = bool(current) and any(_cell(self.tables["devices"], r, 5)
                                           for r in range(self.tables["devices"].rowCount())) \
                and QMessageBox.question(
                    self, "Import EMU configuration",
                    f"Update {current}'s devices from this configuration (the names and every "
                    f"value kept), or start a new entry?",
                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes
        if update:
            entry, warnings = emu.update(self.entry(), config)
        else:
            if three_d is None:
                found, _ = emu.devices(config)
                switches = [n for n, d in found.items() if d.emu.startswith("Two-state")]
                choice, ok = QInputDialog.getItem(self, "The 3D lens",
                                                  "Which device puts the cylindrical lens in?",
                                                  ["(none)"] + switches, 0, False)
                three_d = choice if ok and choice != "(none)" else ""
                if three_d:
                    state, ok = QInputDialog.getItem(self, "The 3D lens", f"Which state of {three_d} "
                                                     "is the lens in?", list(found[three_d].states), 0,
                                                     False)
                    three_d_in = state if ok else three_d_in
            entry, warnings = emu.draft(config, current if update else "", three_d=three_d or "",
                                        three_d_in=three_d_in)
            entry.name = ""
        self.show_entry(entry)
        if config.modified is not None:
            self.until.setText(config.modified.isoformat())
        self.message.setText("Imported " + config.path.name + ("; " + "; ".join(warnings) if warnings
                                                                 else "") +
                             ".  Name it, add what EMU does not know (NA, pixel, identify), save.")
        return warnings

    # ------------------------------------------------------------------ save
    def save(self) -> Optional[str]:
        try:
            entry = self.entry()
        except (ValueError, TypeError) as error:
            self.message.setText(str(error))
            return None
        if not entry.name:
            self.message.setText("the microscope needs a name")
            return None
        until = self.until.text().strip() or None
        entry.versions = []                         # save_microscope keeps the stored ones
        path = microscope_db.save_microscope(entry, until=until)
        self.refresh_list(select=entry.name)
        self.load(entry.name)
        self.message.setText(f"saved to {path}")
        self.saved.emit(entry.name)
        return str(path)
