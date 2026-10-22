"""Adding a camera to the database, from a file it took.

The database is what turns a Micro-Manager file into photons: it knows the
conversion of each readout mode, which the file never records.  A camera that
is not in it has to have its numbers typed into every fit.  This window writes
the entry once instead, and it starts from the file just opened, because that
file already says almost everything the entry needs: the device, its serial
number, the tags that carry the gain and the readout mode.  What it cannot
say -- the conversion and the baseline -- is typed here, once per readout
mode, from the camera's data sheet or a calibration.

What is saved goes into the user's ``microscopes.json`` (`camera_db.save_camera`),
never into the shipped list an update would overwrite.  A shipped camera that
is edited here -- given a readout mode it lacked -- is saved as the user's own
copy under the same name, which then takes the shipped one's place.

The window edits a `Camera` and shows, as it is edited, what that camera
would make of the file: the same table the camera-parameter view shows, so
that what is saved is checked against the file it was written for.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QScrollArea, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from .. import camera_db
from ..camera_db import Camera, Parameter, expand, generalise

MISSING_COLOUR = "#c07000"

# (parameter, label, unit) in the order the window shows them
ROWS = (("conversion", "conversion", "e-/ADU"),
        ("offset", "offset", "ADU"),
        ("pixelsize_um", "pixel size", "µm"),
        ("em_on", "EM gain on", ""),
        ("emgain", "EM gain", ""),
        ("read_noise_e", "read noise", "e-"),
        ("exposure_ms", "exposure", "ms"),
        ("frame_time_ms", "frame time", "ms"),
        ("roi", "camera ROI", "px"))

SOURCES = (("fixed", "fixed"), ("tag", "from a tag"),
           ("state", "per readout mode"), ("none", "not set"))

READS = (("number", "a number"), ("numbers", "numbers"), ("boolean", "yes / no"),
         ("positive", "above zero"), ("equals", "equals"),
         ("not_equals", "is not"), ("map", "per value"), ("text", "text"))

#: how a ``map`` reading is typed: what the tag reads, a colon, the value
MAP_HINT = "In: 0.127 0.116; Out: 0.127"

NEW = "a new camera, from this file"


class _Row:
    """One parameter's widgets: where its value comes from, and the value."""

    def __init__(self, editor: "CameraEditor", name: str, label: str, unit: str):
        self.name = name
        self.label = QLabel(f"{label} ({unit})" if unit else label)
        self.source = QComboBox()
        for key, text in SOURCES:
            self.source.addItem(text, key)
        self.value = QLineEdit()
        self.value.setPlaceholderText("value")
        self.tag = QComboBox()
        self.tag.setEditable(True)
        self.tag.setInsertPolicy(QComboBox.NoInsert)
        self.tag.setMinimumContentsLength(16)
        self.tag.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.read = QComboBox()
        for key, text in READS:
            self.read.addItem(text, key)
        self.argument = QLineEdit()
        self.argument.setPlaceholderText("compared with")
        self.argument.setMaximumWidth(110)
        self.source.currentIndexChanged.connect(self.shown)
        self.read.currentIndexChanged.connect(self.shown)
        for signal in (self.source.currentIndexChanged, self.read.currentIndexChanged,
                       self.value.textEdited, self.tag.currentTextChanged,
                       self.argument.textEdited):
            signal.connect(editor.changed)
        self.value.setToolTip(_VALUE_HELP.get(name, "a number"))

    @property
    def kind(self) -> str:
        return self.source.currentData()

    def shown(self) -> None:
        kind = self.kind
        read = self.read.currentData()
        # a value read from a tag has a default, for a file without the tag
        self.value.setVisible(kind in ("fixed", "state")
                              or (kind == "tag" and read == "map"))
        self.value.setPlaceholderText("this file's mode" if kind == "state" else
                                      "default" if kind == "tag" else "value")
        self.tag.setVisible(kind == "tag")
        self.read.setVisible(kind == "tag")
        self.argument.setVisible(kind == "tag"
                                 and read in ("equals", "not_equals", "map"))
        self.argument.setPlaceholderText(MAP_HINT if read == "map" else "compared with")
        self.argument.setMaximumWidth(16777215 if read == "map" else 110)

    def set(self, parameter: Optional[Parameter], prefix: str, value: Any) -> None:
        """Show `parameter`; `value` is this file's mode's, for a modal one."""
        if parameter is None:
            kind = "none"
        elif parameter.state:
            kind = "state"
        elif parameter.tag:
            kind = "tag"
        else:
            kind = "fixed"
            value = parameter.fixed
        if kind == "tag":
            value = parameter.fixed                     # the default
        self.source.setCurrentIndex(self.source.findData(kind))
        self.value.setText(_text(value))
        if parameter is not None and parameter.tag:
            self.tag.setCurrentText(expand(parameter.tag, prefix))
            self.read.setCurrentIndex(max(self.read.findData(parameter.read), 0))
            self.argument.setText("" if parameter.argument is None else
                                  _map_text(parameter.argument)
                                  if isinstance(parameter.argument, dict)
                                  else str(parameter.argument))
        else:
            self.tag.setCurrentText("")
            self.argument.setText("")
        self.shown()

    def parameter(self, prefix: str) -> Optional[Parameter]:
        kind = self.kind
        if kind == "none":
            return None
        if kind == "state":
            return Parameter(state=True)
        if kind == "tag":
            read = self.read.currentData()
            argument = (self.argument.text().strip() if read in ("equals", "not_equals")
                        else None)
            default = None
            if read == "map":
                argument = self._map()
                default = self.number()
            return Parameter(tag=generalise(self.tag.currentText().strip(), prefix),
                             read=read, argument=argument or None, fixed=default)
        return Parameter(fixed=self.number())

    def number(self, text: Optional[str] = None) -> Any:
        """The value field (or ``text``), read as this parameter needs it;
        None if empty."""
        text = (self.value.text() if text is None else text).strip()
        if not text:
            return None
        if self.name == "em_on":
            return camera_db.read_boolean(text)
        if self.name in ("pixelsize_um", "roi"):
            numbers = camera_db.read_numbers(text)
            if numbers is None:
                return None
            if self.name == "roi":
                return [int(v) for v in numbers]
            return numbers[0] if len(numbers) == 1 else numbers[:2]
        return camera_db.read_number(text)


    def _map(self) -> Dict[str, Any]:
        """``In: 0.127 0.116; Out: 0.127`` -> {"In": [0.127, 0.116], "Out": 0.127}.

        Split at the last colon, so a tag that reads ``Label: In`` still
        maps; an entry without a value is left out."""
        out: Dict[str, Any] = {}
        for entry in self.argument.text().split(";"):
            key, colon, value = entry.rpartition(":")
            found = self.number(value) if colon else None
            if key.strip() and found is not None:
                out[key.strip()] = found
        return out


def _map_text(table: Dict[str, Any]) -> str:
    return "; ".join(f"{key}: {_text(value)}" for key, value in table.items())


_VALUE_HELP = {
    "conversion": "photoelectrons per count, from the data sheet or a "
                  "calibration of this readout mode",
    "offset": "the count of a pixel that saw no light: the mean of a dark frame",
    "pixelsize_um": "the pixel's size in the sample, after the magnification; "
                    "two numbers for x and y if they differ",
    "em_on": "yes or no",
    "read_noise_e": "rms per pixel, in electrons: about 1 for an sCMOS; "
                    "not set, 1 -- and with EM gain on it is not used",
    "roi": "x, y, width, height on the chip",
    "frame_time_ms": "from one frame to the next: the exposure plus the "
                     "camera's dead time; not set, the exposure",
}


class CameraEditor(QWidget):
    """Write a camera into the user's database, starting from a file."""

    #: the camera's name, once it is in the user's database
    saved = Signal(str)

    def __init__(self, metadata: Optional[Dict[str, Any]] = None,
                 camera: Optional[Camera] = None, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Camera editor")
        self.metadata: Dict[str, Any] = {}
        self._base = Camera(name="")
        self._loading = False

        # scrolled, all but the buttons: on a laptop screen the window is
        # taller than the screen, and Qt then squeezes the groups until their
        # rows overlap rather than giving up height it does not have
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        form = QFormLayout()
        self.start_from = QComboBox()
        self.start_from.setToolTip("edit a camera the database has -- to give it a "
                                   "readout mode it lacks -- or write a new one")
        self.start_from.activated.connect(self._start_from)
        form.addRow("start from", self.start_from)
        self.name = QLineEdit()
        self.name.setToolTip("what the camera is called in the Localize tab; "
                             "a shipped camera's name replaces that camera")
        self.name.textEdited.connect(self.changed)
        form.addRow("name", self.name)
        self.prefix = QLineEdit()
        self.prefix.setToolTip("the Micro-Manager device name its tags start with; "
                               "a lab that renames the device edits only this")
        self.prefix.textEdited.connect(self.changed)
        form.addRow("device", self.prefix)
        identify = QHBoxLayout()
        self.identify_tag = QComboBox()
        self.identify_tag.setEditable(True)
        self.identify_tag.setInsertPolicy(QComboBox.NoInsert)
        self.identify_tag.setToolTip("the tag that names this one camera: "
                                     "a serial number is best")
        self.identify_tag.currentTextChanged.connect(self._identify_tag_changed)
        self.identify_value = QLineEdit()
        self.identify_value.setToolTip("what that tag reads in this camera's files")
        self.identify_value.textEdited.connect(self.changed)
        self.identify_match = QComboBox()
        self.identify_match.addItem("is", "exact")
        self.identify_match.addItem("contains", "contains")
        self.identify_match.currentIndexChanged.connect(self.changed)
        identify.addWidget(self.identify_tag, 2)
        identify.addWidget(self.identify_match)
        identify.addWidget(self.identify_value, 2)
        form.addRow("recognised by", identify)
        layout.addLayout(form)

        modes = QGroupBox("readout mode")
        modes_layout = QVBoxLayout(modes)
        explain = QLabel("Tick the tags that set the conversion and the offset; "
                         "what they read here is this file's mode.")
        explain.setWordWrap(True)
        explain.setStyleSheet("color: gray")
        modes_layout.addWidget(explain)
        self.tag_filter = QLineEdit()
        self.tag_filter.setPlaceholderText("filter the tags")
        self.tag_filter.textChanged.connect(self._filter_tags)
        modes_layout.addWidget(self.tag_filter)
        self.mode_tags = QListWidget()
        self.mode_tags.setFixedHeight(130)
        self.mode_tags.itemChanged.connect(self.changed)
        modes_layout.addWidget(self.mode_tags)
        self.modes_note = QLabel("")
        self.modes_note.setWordWrap(True)
        self.modes_note.setStyleSheet("color: gray")
        modes_layout.addWidget(self.modes_note)
        layout.addWidget(modes)

        parameters = QGroupBox("parameters")
        grid = QGridLayout(parameters)
        self.rows: Dict[str, _Row] = {}
        for i, (name, label, unit) in enumerate(ROWS):
            row = _Row(self, name, label, unit)
            self.rows[name] = row
            grid.addWidget(row.label, i, 0)
            grid.addWidget(row.source, i, 1)
            value = QHBoxLayout()
            for widget in (row.value, row.tag, row.read, row.argument):
                value.addWidget(widget)
            grid.addLayout(value, i, 2)
        grid.setColumnStretch(2, 1)
        layout.addWidget(parameters)

        check = QGroupBox("this file, read with this entry")
        check_layout = QVBoxLayout(check)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["parameter", "value", "from"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setMinimumHeight(120)
        check_layout.addWidget(self.table)
        self.problems = QLabel("")
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet(f"color: {MISSING_COLOUR}")
        check_layout.addWidget(self.problems)
        layout.addWidget(check, 1)

        buttons = QHBoxLayout()
        self.where = QLabel("")
        self.where.setStyleSheet("color: gray")
        self.where.setWordWrap(True)
        buttons.addWidget(self.where, 1)
        self.save_button = QPushButton("Save")
        self.save_button.setToolTip("write this camera into your microscopes.json; "
                                    "the next file it took is recognised")
        self.save_button.clicked.connect(self.save)
        buttons.addWidget(self.save_button)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        outer.addLayout(buttons)
        self.resize(720, 900)

        self.start(metadata or {}, camera)

    # ---------------------------------------------------------------- loading
    def start(self, metadata: Dict[str, Any], camera: Optional[Camera] = None) -> None:
        """Edit `camera` against this file, or a new camera drafted from it."""
        self.metadata = dict(metadata or {})
        known = camera_db.database()
        self.start_from.clear()
        self.start_from.addItem(NEW, "")
        for name in known.names():
            self.start_from.addItem(name, name)
        self.start_from.setCurrentIndex(
            max(self.start_from.findData(camera.name), 0) if camera else 0)

        tags = sorted(self.metadata, key=lambda t: (not _is_device_tag(t, self.metadata),
                                                    t.lower()))
        for combo in [self.identify_tag] + [r.tag for r in self.rows.values()]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(tags)
            combo.blockSignals(False)

        device = camera_db.device_of(self.metadata)
        if not self.metadata:
            self.summary.setText("<b>no file</b>: tags are typed by hand.  Open the "
                                 "editor from a fitter that is set to a file, and "
                                 "they are offered from it.")
        else:
            self.summary.setText(
                f"from the file's {len(self.metadata)} tags"
                + (f", taken by the device <b>{device}</b>" if device else "")
                + ".  The conversion and the offset are never among them: type "
                  "them from the data sheet or a calibration.")
        self.where.setText(f"saves to {camera_db.user_file()}")
        self._load(copy.deepcopy(camera) if camera is not None
                   else camera_db.draft(self.metadata))

    def _start_from(self) -> None:
        name = self.start_from.currentData()
        found = camera_db.database().get(name) if name else None
        self._load(copy.deepcopy(found) if found is not None
                   else camera_db.draft(self.metadata))

    def _load(self, camera: Camera) -> None:
        self._loading = True
        self._base = camera
        prefix = camera.prefix
        self.name.setText(camera.name)
        self.prefix.setText(prefix)
        if camera.identify is not None and camera.identify.tag:
            self.identify_tag.setCurrentText(expand(camera.identify.tag, prefix))
            self.identify_value.setText(str(camera.identify.argument or ""))
            self.identify_match.setCurrentIndex(
                max(self.identify_match.findData(camera.identify.read), 0))
        else:
            self.identify_tag.setCurrentText("")
            self.identify_value.setText("")
        state = camera.state_for(self.metadata)
        values = state.values if state is not None else {}
        for name, row in self.rows.items():
            row.set(camera.parameters.get(name), prefix, values.get(name))

        ticked = [expand(t, prefix) for t in
                  (state.match if state is not None else camera_db.mode_tags(camera))]
        self.mode_tags.blockSignals(True)
        self.mode_tags.clear()
        shown = [t for t in sorted(self.metadata, key=lambda t: (
            t not in ticked, not _is_device_tag(t, self.metadata), t.lower()))]
        shown += [t for t in ticked if t not in self.metadata]
        for tag in shown:
            here = tag in self.metadata
            item = QListWidgetItem(f"{tag} = {self.metadata[tag]}" if here
                                   else f"{tag}  (not in this file)")
            item.setData(Qt.UserRole, tag)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if tag in ticked else Qt.Unchecked)
            self.mode_tags.addItem(item)
        self.mode_tags.blockSignals(False)
        self._filter_tags(self.tag_filter.text())
        others = len(camera.states) - (state is not None)
        self.modes_note.setText(
            (f"{others} other readout mode{'s' if others != 1 else ''} of this "
             "camera kept as they are." if others else "")
            + ("" if state is None or not self.metadata else
               "  This file's mode is in the database: saving replaces its values."))
        self._loading = False
        self.changed()

    def _identify_tag_changed(self, tag: str) -> None:
        if not self._loading and tag in self.metadata:
            self.identify_value.setText(str(self.metadata[tag]).strip())
        self.changed()

    def _filter_tags(self, text: str) -> None:
        text = text.strip().lower()
        for i in range(self.mode_tags.count()):
            item = self.mode_tags.item(i)
            item.setHidden(bool(text) and text not in item.text().lower()
                           and item.checkState() != Qt.Checked)

    # ------------------------------------------------------------ the camera
    def ticked(self) -> List[str]:
        return [self.mode_tags.item(i).data(Qt.UserRole)
                for i in range(self.mode_tags.count())
                if self.mode_tags.item(i).checkState() == Qt.Checked]

    def camera(self) -> Camera:
        """The entry as the window shows it."""
        prefix = self.prefix.text().strip()
        base = self._base
        parameters = {k: v for k, v in base.parameters.items()
                      if k not in self.rows}             # a frame interval, kept
        modal: Dict[str, Any] = {}
        for name, row in self.rows.items():
            parameter = row.parameter(prefix)
            if parameter is None:
                continue
            parameters[name] = parameter
            if parameter.state:
                modal[name] = row.number()
        tag = self.identify_tag.currentText().strip()
        identify = (Parameter(tag=generalise(tag, prefix),
                              read=self.identify_match.currentData(),
                              argument=self.identify_value.text().strip())
                    if tag else None)
        camera = Camera(name=self.name.text().strip(), prefix=prefix,
                        comment=base.comment, identify=identify,
                        parameters=parameters, states=copy.deepcopy(base.states))
        if modal:
            camera = camera_db.with_mode(camera, self.metadata, self.ticked(), modal)
        return camera

    def changed(self, *_) -> None:
        """Show what the entry, as it stands, makes of the file."""
        if self._loading:
            return
        camera = self.camera()
        rows = camera.resolve(self.metadata).rows() if self.metadata else []
        self.table.setRowCount(len(rows))
        for i, (name, value, source, used) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(name))
            item = QTableWidgetItem("-" if value is None else _text(value))
            if value is None and used:
                item.setForeground(Qt.red)
            self.table.setItem(i, 1, item)
            self.table.setItem(i, 2, QTableWidgetItem(source))
        found = camera_db.problems(camera, self.metadata, camera_db.database())
        if _replaces_shipped(camera.name):
            found.append(f"{camera.name} is a shipped camera: your copy in "
                         "microscopes.json will take its place")
        self.problems.setText("\n".join(found))
        self.save_button.setEnabled(bool(camera.name))

    # ----------------------------------------------------------------- saving
    def save(self) -> Optional[str]:
        """Write the entry into the user's database; its name, or None."""
        camera = self.camera()
        if not camera.name:
            return None
        path = camera_db.save_camera(camera)
        self._base = camera
        self.where.setText(f"saved {camera.name} to {path}")
        self.saved.emit(camera.name)
        self.changed()
        return camera.name


def _replaces_shipped(name: str) -> bool:
    """Whether saving under this name puts the user's copy over a shipped
    camera for the first time."""
    from ..camera_db import SHIPPED, CameraDatabase, user_file
    try:
        if not name or CameraDatabase.load(SHIPPED).get(name) is None:
            return False
        path = user_file()
        return not (path.exists() and CameraDatabase.load(path).get(name))
    except (OSError, ValueError):
        return False


def _is_device_tag(tag: str, metadata: Dict[str, Any]) -> bool:
    device = camera_db.device_of(metadata)
    return bool(device) and str(tag).startswith(device + "-")


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(v) for v in value)
    return str(value)


def open_editor(metadata: Optional[Dict[str, Any]] = None,
                camera: Optional[Camera] = None, parent=None,
                window: Optional[CameraEditor] = None) -> CameraEditor:
    """Show the editor on this file, reusing the open one."""
    if window is None:
        window = CameraEditor(parent=parent)
    window.start(metadata or {}, camera)
    window.show()
    window.raise_()
    return window
