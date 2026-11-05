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
entry makes of a stack, and the *channels* tab draws the split on its first frame.  Saving
writes the user's ``microscopes.json`` and keeps what the entry said before as a version,
dated (*old version until*: the EMU configuration's date after an import, else today), so
older files are still read as they were taken; the *versions* tab shows them, their dates
editable.  The objective's pupils (*PSF*) are kept by PSF calibration, never typed: the tab
draws them and can drop one.

What is drawn is drawn by module-level functions (`draw_channels`, `draw_pupil`) on the
entry and arrays, so a test or a page reads the same picture without the window.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import numpy as np
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


# ------------------------------------------------------------------ drawings

#: an entry's ``layout`` where its arms have no sides yet: the split's axis, which arm is where
#: read from data later (`Microscope.arrangement`); a mirror is the arm's
LAYOUTS = ("", "right-left", "up-down")

#: immersions offered; any other is typed
IMMERSIONS = ("oil", "water", "silicone", "glycerol", "air")


def _noll(j: int):
    """Noll's index as (n, m), the order of `uipsf.optics.zernike.noll_to_nm` -- copied, not
    imported: uipsf imports torch, and the manager must open without it."""
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    m = n % 2 + 2 * ((j - n * (n + 1) // 2 - 1 + (n + 1) % 2) // 2)
    return n, (-m if j % 2 else m) if m else 0


def zernike(j: int, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Noll-normalised Zernike polynomial ``j``, unit RMS over the disc, as uiPSF's."""
    from math import factorial
    n, m = _noll(j)
    a = abs(m)
    radial = sum((-1) ** k * factorial(n - k)
                 / (factorial(k) * factorial((n + a) // 2 - k) * factorial((n - a) // 2 - k))
                 * rho ** (n - 2 * k) for k in range((n - a) // 2 + 1))
    norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * n + 2)
    return norm * radial * (np.sin(a * phi) if m < 0 else np.cos(a * phi))


def pupil_phase(pupil: Dict[str, Any], size: int = 65) -> np.ndarray:
    """A kept pupil's phase over the aperture, nm of wavefront, NaN outside: the sum of its
    Zernikes (`phase_nm`, Noll's index to nm), on a ``size`` square grid across the disc."""
    axis = np.linspace(-1, 1, size)
    x, y = np.meshgrid(axis, axis)
    rho, phi = np.hypot(x, y), np.arctan2(y, x)
    phase = np.zeros_like(rho)
    for j, value in (pupil.get("phase_nm") or {}).items():
        phase += float(value) * zernike(int(j), rho, phi)
    phase[rho > 1] = np.nan
    return phase


def draw_pupil(ax, pupil: Optional[Dict[str, Any]]) -> None:
    """The pupil's phase, its largest modes and γ or the magnitude's count in the title."""
    ax.set_axis_off()
    if not pupil:
        ax.text(0.5, 0.5, "choose a pupil", ha="center", va="center", transform=ax.transAxes,
                color="gray")
        return
    phase = pupil_phase(pupil)
    limit = max(float(np.nanmax(np.abs(phase))), 1.0)
    image = ax.imshow(phase, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower")
    ax.figure.colorbar(image, ax=ax, shrink=0.8, label="phase (nm)")
    top = sorted((pupil.get("phase_nm") or {}).items(), key=lambda kv: -abs(float(kv[1])))[:3]
    modes = ", ".join(f"Z{j} {float(v):+.0f}" for j, v in top)
    extra = (f"γ {pupil['gamma']:g}" if pupil.get("gamma") is not None else
             f"{len(pupil.get('magnitude') or {})} magnitude modes")
    ax.set_title(f"{pupil.get('kind', '?')}: {modes}; {extra}", fontsize=9)


def draw_channels(ax, entry: Microscope, camera: str = "", roi=None,
                  frame: Optional[np.ndarray] = None) -> None:
    """The entry's channels on the chip, over ``frame`` where there is one: its two-arm split as
    the fitters cut it (`Microscope.split_layout`) -- the arms' regions, else the camera ROI
    ``roi`` (x0, y0, width, height) halved -- or its regions (a 4Pi's), the main channel solid, each labelled with its arm and its
    flip.  What keeps it from being drawn is said in its place."""
    ax.clear()
    if frame is not None and roi is not None:
        x0, y0, w, h = roi
        ax.imshow(frame, cmap="gray", origin="upper",
                  extent=(x0 - 0.5, x0 + w - 0.5, y0 + h - 0.5, y0 - 0.5))
    arms = entry.ordered_channels()
    try:
        if len(arms) == 2:
            found = entry.arrangement(camera)
            if found is not None and not found.get("regions") and roi is None:
                raise ValueError("check with a file to draw the split on its frame")
            layout = entry.split_layout(roi or (0, 0, 2, 2), camera)
            names = [c.name for c in arms]
        else:
            layout = entry.channel_layout(camera)
            names = [c.name for c in entry.channels if c.region_for(camera or entry.camera)]
        if layout is None:
            raise ValueError("no arrangement: give the arms their sides, or a layout")
    except ValueError as error:
        ax.text(0.5, 0.5, str(error), ha="center", va="center", wrap=True,
                transform=ax.transAxes, color="gray")
        ax.set_axis_off()
        return
    from matplotlib.patches import Rectangle
    for c, ((x, y, w, h), (fx, fy)) in enumerate(zip(layout.regions, layout.flips)):
        main = c == 0
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), w, h, fill=False, lw=2 if main else 1.5,
                               ls="-" if main else "--", color="#e0453a" if main else "#3a8ee0"))
        flip = " ".join(axis for axis, on in (("x", fx), ("y", fy)) if on)
        label = (names[c] if c < len(names) else f"channel {c}") + (" (main)" if main else "") \
            + (f", mirrored {flip}" if flip else "")
        ax.text(x + w / 2 - 0.5, y + h / 2 - 0.5, label, ha="center", va="center", fontsize=9,
                color="white", bbox=dict(facecolor="black", alpha=0.5, lw=0))
    xs = [r[0] for r in layout.regions] + [r[0] + r[2] for r in layout.regions]
    ys = [r[1] for r in layout.regions] + [r[1] + r[3] for r in layout.regions]
    if roi is not None:
        xs += [roi[0], roi[0] + roi[2]]
        ys += [roi[1], roi[1] + roi[3]]
    # a margin, so that a channel's edge on the frame's is not hidden under the axes' frame
    pad = 0.02 * max(max(xs) - min(xs), max(ys) - min(ys))
    ax.set_xlim(min(xs) - 0.5 - pad, max(xs) - 0.5 + pad)
    ax.set_ylim(max(ys) - 0.5 + pad, min(ys) - 0.5 - pad)
    ax.set_aspect("equal")
    ax.set_xlabel("chip x (px)")
    ax.set_ylabel("chip y (px)")


def version_text(version: Dict[str, Any]) -> str:
    """What a version holds, by section and name: what the entry said differently before."""
    parts = []
    for key, value in version.items():
        if key == "until":
            continue
        parts.append(f"{key}: {', '.join(map(str, value))}" if isinstance(value, dict) else key)
    return "; ".join(parts) or "nothing"


def unnamed_filters(entry: Microscope) -> List[str]:
    """The states of the filter devices a wavelength is read from whose name gives none, and that
    the filters table does not type: what a fit in them would miss."""
    devices = {c.filter for c in entry.channels if c.filter}
    devices |= {raw["device"] for raw in entry.parameters.values()
                if isinstance(raw, dict) and raw.get("values") == FROM_NAME}
    return [state for name in devices if name in entry.devices
            for state in entry.devices[name].states
            if microscope_db.filter_wavelength(state, entry.filters) is None]


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
    ("filters", ["filter", "wavelength_nm", "band from", "band to"],
     "a filter whose name is not its band (a long pass, an empty slot behind the splitting "
     "dichroic): the wavelength a PSF is drawn at, or the band it passes; a filter named as its "
     "band (676/37) needs no row"),
    ("devices", ["name", "tags", "read", "tolerance", "states", "EMU control"],
     "each device: the Micro-Manager tags it is read from (an older name after a ' | '), how "
     "(exact, nearest, value), and its states as 'name: reading; ...'"),
    ("parameters", ["parameter", "follows", "value"],
     "n_coverslip, coverslip_design_um, wavelength_nm, pixel_nm, modality, astigmatism_nm ...: a "
     "value, or a device it follows "
     "and 'state: value; ...' ('from the name' reads a filter's band)"),
    ("channels", ["arm", "filter", "side", "mirror", "region"],
     "the split: its arms, transmitted first, the filter device each follows, the side of "
     "the chip it lands on with this camera (left, right, upper, lower), the axis the optics "
     "flip it along against the other arm (x, y; empty for none), and its rectangle on the "
     "chip, 'x y width height', where it lies exactly: a 4Pi's arms, or a split that does not "
     "meet in the middle of the frame (its side then follows from it)"),
)

#: the order of the tabs; the tables' in `TABS`, the forms' built in place
ORDER = ("identify", "objective", "filters", "devices", "parameters", "channels", "PSF",
         "versions")


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
        self._frame: Optional[np.ndarray] = None
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
        pages: Dict[str, QWidget] = {}
        for title, headers, tip in TABS:
            page = QWidget()
            box = QVBoxLayout(page)
            table = _table(headers, tip)
            self.tables[title] = table
            box.addWidget(table, 1)
            buttons = QHBoxLayout()
            add, remove = QPushButton("Add row"), QPushButton("Remove row")
            add.clicked.connect(lambda _=False, t=table: t.insertRow(t.rowCount()))
            remove.clicked.connect(lambda _=False, t=table: (t.removeRow(t.currentRow()),
                                                             self.check()))
            buttons.addWidget(add)
            buttons.addWidget(remove)
            buttons.addStretch(1)
            box.addLayout(buttons)
            pages[title] = page
        pages["objective"] = self._objective_page()
        self.unnamed = QLabel("")
        self.unnamed.setStyleSheet("color: gray")
        self.unnamed.setWordWrap(True)
        pages["filters"].layout().insertWidget(1, self.unnamed)
        self._arrangement(pages["channels"])
        pages["PSF"] = self._psf_page()
        pages["versions"] = self._versions_page()
        for title in ORDER:
            self.tabs.addTab(pages[title], title)
        right.addWidget(self.tabs, 3)

        check = QHBoxLayout()
        self.check_button = QPushButton("Check with a file...")
        self.check_button.setToolTip("what this entry, as it stands, makes of a stack: whether it "
                                     "recognises it, each device's state, each value; the channels "
                                     "drawn on its first frame")
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
        save_row.addStretch(1)
        self.save_button = QPushButton("Save")
        self.save_button.setToolTip("write the entry into your microscopes.json")
        self.save_button.clicked.connect(self.save)
        save_row.addWidget(self.save_button)
        right.addLayout(save_row)
        self._pupils: Dict[str, Dict[str, Any]] = {}
        self.message = QLabel("")
        self.message.setWordWrap(True)
        right.addWidget(self.message)
        layout.addLayout(right, 3)

        for table in self.tables.values():
            table.itemChanged.connect(lambda _: self.check())
        self.version_table.itemChanged.connect(lambda _: self.check())
        self.name.textChanged.connect(lambda _: self.check())
        self.refresh_list()
        if path:
            self.check_file(path)
        self.resize(1100, 760)

    # ------------------------------------------------------------- the pages
    def _objective_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.objective: Dict[str, Any] = {}
        tips = {"name": "the objective as the lab calls it, 'Olympus 100x NA1.49'",
                "NA": "the numerical aperture, the objective's nominal one: a pupil model may "
                      "learn a slightly smaller one",
                "immersion": "oil, water, silicone ...: what the objective is designed for",
                "n_immersion": "the immersion's refractive index, 1.518 for oil"}
        for name in microscope_db.OBJECTIVE:
            if name == "immersion":
                field = QComboBox()
                field.setEditable(True)
                field.addItems(("",) + IMMERSIONS)
                field.currentTextChanged.connect(lambda _: self.check())
            else:
                field = QLineEdit()
                field.textChanged.connect(lambda _: self.check())
            field.setToolTip(tips.get(name, ""))
            self.objective[name] = field
            form.addRow(name, field)
        note = QLabel("No magnification: the pixel on the sample is the camera's, per state of the "
                      "3D lens, under parameters.")
        note.setStyleSheet("color: gray")
        note.setWordWrap(True)
        form.addRow(note)
        return page

    def _arrangement(self, page: QWidget) -> None:
        """Under the arms: the main one, the layout while they have no sides, and the split
        drawn.  Where the arms meet is their regions, not a field of its own: the entry says it
        once."""
        box = page.layout()
        form = QFormLayout()
        self.main_channel = QComboBox()
        self.main_channel.setToolTip("the arm the other is registered onto and whose PSF is a dual "
                                     "calibration's main; empty: the first arm")
        self.main_channel.currentTextChanged.connect(lambda _: self.check())
        form.addRow("main channel", self.main_channel)
        self.layout_choice = QComboBox()
        self.layout_choice.addItems(LAYOUTS)
        self.layout_choice.setToolTip("the split's axis while the arms have no sides or regions "
                                      "yet; the main arm is then on the left or upper half")
        self.layout_choice.currentTextChanged.connect(lambda _: self.check())
        form.addRow("layout", self.layout_choice)
        box.addLayout(form)
        from matplotlib.figure import Figure
        from .canvas import canvas_class
        self.channels_figure = Figure(figsize=(4, 3), layout="constrained")
        self.channels_canvas = canvas_class()(self.channels_figure)
        self.channels_canvas.setMinimumHeight(220)
        box.addWidget(self.channels_canvas, 2)
        self.tables["channels"].itemChanged.connect(lambda _: self._fill_main())
        self.tables["channels"].itemChanged.connect(lambda _: self._layout_needed())

    def _psf_page(self) -> QWidget:
        page = QWidget()
        box = QHBoxLayout(page)
        column = QVBoxLayout()
        self.pupil_table = _table(["key", "kind", "camera", "from", "date"],
                                  "the objective's pupils, kept by PSF calibration's Add to "
                                  "microscope...: one per state of the 3D lens and colour")
        self.pupil_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pupil_table.itemSelectionChanged.connect(self.draw_pupil)
        column.addWidget(self.pupil_table, 1)
        row = QHBoxLayout()
        remove = QPushButton("Remove pupil")
        remove.setToolTip("drop the chosen pupil from the entry; Save keeps the entry before as "
                          "a version")
        remove.clicked.connect(self.remove_pupil)
        row.addWidget(remove)
        row.addStretch(1)
        column.addLayout(row)
        box.addLayout(column, 1)
        from matplotlib.figure import Figure
        from .canvas import canvas_class
        self.pupil_figure = Figure(figsize=(3, 3), layout="constrained")
        self.pupil_canvas = canvas_class()(self.pupil_figure)
        box.addWidget(self.pupil_canvas, 1)
        return page

    def _versions_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        self.version_table = _table(["until", "what it said otherwise"],
                               "the entry's older setups: each holds what the entry said before a "
                               "change, for files taken before its day")
        box.addWidget(self.version_table, 1)
        row = QHBoxLayout()
        remove = QPushButton("Remove version")
        remove.setToolTip("forget the chosen version: the files of its days are then read as the "
                          "next setup says")
        remove.clicked.connect(self.remove_version)
        row.addWidget(remove)
        row.addStretch(1)
        box.addLayout(row)
        return page

    def _fill_main(self) -> None:
        arms = [_cell(self.tables["channels"], r, 0).strip()
                for r in range(self.tables["channels"].rowCount())]
        current = self.main_channel.currentText()
        self.main_channel.blockSignals(True)
        self.main_channel.clear()
        self.main_channel.addItems([""] + [a for a in arms if a])
        if current and self.main_channel.findText(current) < 0:
            self.main_channel.addItem(current)
        self.main_channel.setCurrentText(current)
        self.main_channel.blockSignals(False)

    def _layout_needed(self) -> None:
        """The *layout* only while no arm has a side or a region: once one has, the arms say the
        split, and a second description could only disagree with them."""
        table = self.tables["channels"]
        said = any(_cell(table, r, c).strip() for r in range(table.rowCount()) for c in (2, 4))
        self.layout_choice.setEnabled(not said)
        self.layout_choice.setToolTip(
            "the arms' sides or regions say the split" if said else
            "the split's axis while the arms have no sides or regions yet; the main arm is then "
            "on the left or upper half")

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
    def _quiet(self, quiet: bool) -> None:
        for widget in [*self.tables.values(), self.version_table, *self.objective.values(),
                       self.main_channel, self.layout_choice]:
            widget.blockSignals(quiet)

    def show_entry(self, entry: Microscope) -> None:
        self._quiet(True)
        for table in [*self.tables.values(), self.version_table]:
            table.setRowCount(0)
        self.name.setText(entry.name)
        self.comment.setText(entry.comment)
        for tag, value in entry.identify.items():
            _put(self.tables["identify"], [tag, value])
        for name, field in self.objective.items():
            text = value_text(entry.objective.get(name))
            field.setCurrentText(text) if isinstance(field, QComboBox) else field.setText(text)
        for name, typed in entry.filters.items():
            band = typed.get("band_nm") or [None, None]
            _put(self.tables["filters"], [name, value_text(typed.get("wavelength_nm")),
                                          value_text(band[0]), value_text(band[1])])
        for name, device in entry.devices.items():
            _put(self.tables["devices"], [name, " | ".join(device.tags), device.read,
                                          value_text(device.tolerance), map_text(device.states),
                                          device.emu])
        for name, raw in entry.parameters.items():
            _put(self.tables["parameters"], [name, *follows_text(raw)])
        for channel in entry.channels:
            side = map_text(channel.side) if isinstance(channel.side, dict) else channel.side
            region = (map_text(channel.region) if isinstance(channel.region, dict)
                      else value_text(channel.region))
            _put(self.tables["channels"], [channel.name, channel.filter, side, channel.mirror,
                                           region])
        self._fill_main()
        self.main_channel.setCurrentText(entry.main_channel)
        self.layout_choice.setCurrentText(entry.layout)
        self._layout_needed()
        self._fill_cameras(entry.camera)
        self._versions = [dict(v) for v in entry.versions]
        for version in self._versions:
            _put(self.version_table, [str(version.get("until", "")), version_text(version)])
            self.version_table.item(self.version_table.rowCount() - 1, 1).setFlags(Qt.ItemIsEnabled)
        self._pupils = dict(entry.pupils)
        self._fill_pupils()
        self._quiet(False)
        self.check()

    def _fill_pupils(self) -> None:
        self.pupil_table.setRowCount(0)
        for key, pupil in self._pupils.items():
            _put(self.pupil_table, [key, pupil.get("kind") or "old format: keep it again",
                                    pupil.get("camera") or "any", pupil.get("source") or "?",
                                    pupil.get("date") or "?"])
        if self._pupils:
            self.pupil_table.selectRow(0)
        self.draw_pupil()

    def draw_pupil(self) -> None:
        row = self.pupil_table.currentRow()
        key = _cell(self.pupil_table, row, 0) if row >= 0 else ""
        self.pupil_figure.clear()
        pupil = self._pupils.get(key)
        draw_pupil(self.pupil_figure.add_subplot(), pupil if pupil and pupil.get("kind") else None)
        self.pupil_canvas.draw_idle()

    def remove_pupil(self) -> None:
        row = self.pupil_table.currentRow()
        if row >= 0:
            self._pupils.pop(_cell(self.pupil_table, row, 0), None)
            self._fill_pupils()

    def remove_version(self) -> None:
        row = self.version_table.currentRow()
        if row >= 0:
            self.version_table.removeRow(row)
            del self._versions[row]
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
        filters = {}
        for name, wavelength, low, high in rows("filters"):
            if not name.strip():
                continue
            typed = {}
            if wavelength.strip():
                typed["wavelength_nm"] = float(wavelength)
            if low.strip() or high.strip():
                typed["band_nm"] = [float(low) if low.strip() else None,
                                    float(high) if high.strip() else None]
            filters[name.strip()] = typed
        objective = {}
        for name, field in self.objective.items():
            text = field.currentText() if isinstance(field, QComboBox) else field.text()
            if text.strip():
                objective[name] = parse_value(text) if name != "name" else text.strip()
        versions = []
        for row, version in enumerate(self._versions):
            until = _cell(self.version_table, row, 0).strip()
            if not microscope_db.read_date(until):
                raise ValueError(f"version {row + 1}: {until!r} is not a day (2026-11-04)")
            versions.append(dict(version, until=until))
        raw = {
            "name": self.name.text().strip(), "comment": self.comment.text().strip(),
            "identify": {tag.strip(): value.strip() for tag, value in rows("identify") if tag.strip()},
            "devices": devices,
            "parameters": {name.strip(): parse_follows(device, value)
                           for name, device, value in rows("parameters") if name.strip()},
            "objective": objective, "filters": filters,
            "channels": {arm.strip(): {"filter": filter_.strip(),
                                       "side": parse_map(side, values=False) if ":" in side
                                       else side.strip(),
                                       "mirror": mirror.strip(),
                                       **({"region": parse_map(region) if ":" in region
                                           else parse_value(region)} if region.strip() else {})}
                         for arm, filter_, side, mirror, region in rows("channels") if arm.strip()},
            "main_channel": self.main_channel.currentText().strip(),
            # the axis only while the arms do not say it: two descriptions would disagree
            "layout": self.layout_choice.currentText() if self.layout_choice.isEnabled() else "",
            "camera": self.camera.currentData() or "", "versions": versions,
            "pupils": self._pupils,
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
            from ..io.tiff import _parse_roi, camera_tags, open_stack
            source = open_stack(path)
            self._metadata = camera_tags(source)
            # the brightest of the first frames: a frame of a movie shows a few molecules, the
            # maximum over fifty shows where each channel has light
            self._frame = np.max(next(source.frames(chunk=50, stop=min(50, len(source))))[1],
                                 axis=0)
            height, width = self._frame.shape
            self._roi = _parse_roi(self._metadata.get("ROI")) or (0, 0, width, height)
            self._file = str(path)
        except Exception as error:
            self._metadata, self._file, self._frame = {}, "", None
            self.message.setText(f"{path}: {error}")
        self.check()

    def check(self) -> None:
        """What the entry as typed makes of the file being checked with, and its channels drawn."""
        from pathlib import Path
        self.result.setRowCount(0)
        try:
            entry = self.entry()
        except (ValueError, TypeError) as error:
            entry = None
            self.file_label.setText((f"{Path(self._file).name}: " if self._file else "") + str(error))
        camera = None
        if self._file:
            from ..camera_db import database as cameras
            camera = cameras().identify(self._metadata)
        self.unnamed.setText("")
        if entry is not None:
            missing = unnamed_filters(entry)
            if missing:
                self.unnamed.setText("no wavelength in the name, and none typed here: "
                                     + ", ".join(missing))
            self.draw_channels(entry, camera.name if camera else "")
        if not self._file:
            if entry is not None:
                self.file_label.setText("no file")
            return
        if entry is None:
            return
        found = entry.resolve(self._metadata, camera=camera.name if camera else "", channels=2)
        recognised = entry.identifies(self._metadata)
        self.file_label.setText(f"{Path(self._file).name}: "
                                + ("recognised" if recognised else
                                   "not recognised by the identify tags (shown as if chosen)"))
        from .microscope_view import fill_rows
        fill_rows(self.result, found.rows())

    def draw_channels(self, entry: Microscope, camera: str = "") -> None:
        self.channels_figure.clear()
        draw_channels(self.channels_figure.add_subplot(), entry, camera,
                      getattr(self, "_roi", None) if self._file else None,
                      self._frame if self._file else None)
        self.channels_canvas.draw_idle()

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
        # the versions as the tab has them -- a date moved, one removed -- and the change saved now
        # added to them as a new one
        path = microscope_db.save_microscope(entry, until=until, versions=entry.versions)
        self.refresh_list(select=entry.name)
        self.load(entry.name)
        self.message.setText(f"saved to {path}")
        self.saved.emit(entry.name)
        return str(path)
