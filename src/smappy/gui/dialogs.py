"""Small dialogs: mapping csv columns, asking for a pixel size, a file's
frame time."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QLabel, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from ..io.formats import CSV_NAMES, csv_columns, guess_csv_mapping

TARGETS = ["-", "x_nm", "y_nm", "z_nm", "frame", "photons", "xy_err_nm",
           "z_err_nm", "sigma_nm", "sigma_y_nm", "background", "logl_rel",
           "channel", "id"]


class CsvMappingDialog(QDialog):
    """Which csv column is which, what the lengths are in, and the number of
    the first frame -- and a saved conversion to do all of it by name
    (`io.conversions`, SMAP's *conversion structure*)."""

    UNITS = (("nm", "nm"), ("um", "µm"), ("mm", "mm"), ("px", "camera pixels"))

    def __init__(self, path, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QHBoxLayout, QInputDialog, QPushButton, QSpinBox
        from ..io.conversions import conversion_for, conversions
        self._input = QInputDialog
        self.setWindowTitle(f"columns of {Path(path).name}")
        headers, sample, has_header = csv_columns(Path(path))
        self.headers = headers
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("conversion"))
        self.conversion = QComboBox()
        self.conversion.addItem("-", "")
        for name in conversions():
            self.conversion.addItem(name, name)
        top.addWidget(self.conversion, 1)
        save = QPushButton("save as...")
        save.setToolTip("keep these columns, units and first frame under a name, "
                        "found again by the headers")
        save.clicked.connect(self._save)
        top.addWidget(save)
        layout.addLayout(top)
        layout.addWidget(QLabel("x and y are needed; the rest is optional.  "
                                "Other columns are kept under their own name."))
        self.table = QTableWidget(len(headers), 3)
        self.table.setHorizontalHeaderLabels(["column", "first value", "means"])
        self.combos = []
        for i, (h, v) in enumerate(zip(headers, sample)):
            self.table.setItem(i, 0, QTableWidgetItem(h))
            self.table.setItem(i, 1, QTableWidgetItem(v))
            combo = QComboBox()
            combo.addItems(TARGETS + [f"keep as '{h}'"])
            self.table.setCellWidget(i, 2, combo)
            self.combos.append((h, combo))
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)
        form = QFormLayout()
        self.units = QComboBox()
        for value, label in self.UNITS:
            self.units.addItem(label, value)
        self.pixelsize = QDoubleSpinBox(minimum=0.1, maximum=10000, value=100.0, decimals=2)
        self.units.currentIndexChanged.connect(
            lambda _: self.pixelsize.setEnabled(self.units.currentData() == "px"))
        self.first_frame = QSpinBox(minimum=0, maximum=1, value=0)
        self.first_frame.setToolTip("1 for ThunderSTORM and SMAP, 0 for smappy")
        form.addRow("lengths in", self.units)
        form.addRow("camera pixel (nm)", self.pixelsize)
        form.addRow("first frame", self.first_frame)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(520, 420)
        self.conversion.currentIndexChanged.connect(self._apply_conversion)
        found = conversion_for(headers)
        if found is not None:
            self.conversion.setCurrentIndex(self.conversion.findData(found.name))
        self._apply_conversion()

    def _apply_conversion(self, *_) -> None:
        """Fill the table and the units from the conversion chosen, or guess."""
        from ..io.conversions import conversions
        conv = conversions().get(self.conversion.currentData() or "")
        guess = guess_csv_mapping(self.headers)
        mapping = dict(guess, **(conv.mapping if conv else {}))
        for h, combo in self.combos:
            target = mapping.get(h, "-")
            combo.setCurrentText(target if target in TARGETS else f"keep as '{h}'")
        units = conv.units if conv else "nm"
        self.units.setCurrentIndex(max(0, self.units.findData(units)))
        if conv and conv.pixelsize_nm:
            self.pixelsize.setValue(conv.pixelsize_nm)
        self.pixelsize.setEnabled(units == "px")
        self.first_frame.setValue(conv.first_frame if conv else 0)

    def _mapping(self) -> Dict[str, str]:
        mapping = {}
        for h, combo in self.combos:
            t = combo.currentText()
            if t.startswith("keep as"):
                mapping[h] = ""                # read under its own name
            elif t != "-":
                mapping[h] = t
        return mapping

    def _save(self) -> None:
        from ..io.conversions import Conversion, save_conversion
        name, ok = self._input.getText(self, "Save conversion", "name")
        name = name.strip()
        if not ok or not name:
            return
        mapping = {h: t for h, t in self._mapping().items() if t}
        save_conversion(Conversion(
            name=name, headers=list(self.headers), mapping=mapping,
            units=self.units.currentData(),
            pixelsize_nm=self.pixelsize.value() if self.units.currentData() == "px" else None,
            first_frame=self.first_frame.value()))
        if self.conversion.findData(name) < 0:
            self.conversion.addItem(name, name)
        blocked = self.conversion.blockSignals(True)
        self.conversion.setCurrentIndex(self.conversion.findData(name))
        self.conversion.blockSignals(blocked)

    def reader_args(self) -> Dict:
        units = self.units.currentData()
        return {"mapping": {h: t for h, t in self._mapping().items() if t},
                "units": units, "first_frame": self.first_frame.value(),
                "pixelsize_nm": self.pixelsize.value() if units == "px" else None}


class GroupingDialog(QDialog):
    """How localizations are linked into blinks: session-wide, belonging to no
    layer.  OK regroups every layer."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        from ..group import GroupSettings
        from ..plugins import ParamInfo, param_specs
        from .params import SettingsForm
        self.session = session
        self.setWindowTitle("Merge blinks")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Merge blinks</b>: the localizations of one blinking "
                                "event, in consecutive frames, become one"))
        note = QLabel("Merging looks at x and y only: emitters this close in consecutive "
                      "frames overlap in the raw data and were fitted as one anyway, so "
                      "a z window would only split blinks whose z scatters.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        layout.addWidget(note)
        self.form = SettingsForm(GroupSettings, param_specs(GroupSettings))
        self.form.set(session.group_settings)
        self.form.field_changed.connect(self._show_mode)
        self._show_mode("link")
        layout.addWidget(self.form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # the fields each mode reads; the other mode's are kept, only hidden
    MODE_FIELDS = {"fixed": ("dx",), "precision": ("k", "dx_min", "dx_max")}

    def _show_mode(self, name: str) -> None:
        if name != "link":
            return
        mode = self.form.fields["link"].value()
        for m, names in self.MODE_FIELDS.items():
            for n in names:
                self.form.fields[n].setVisible(m == mode)
                self.form.labels[n].setVisible(m == mode)

    def _apply(self) -> None:
        try:
            settings = self.form.value()
        except ValueError:
            return
        if settings != self.session.group_settings:
            self.session.set_group_settings(settings)
        self.accept()


class PixelSizeDialog(QDialog):
    """An image without a pixel size in its tags: ask, plus where it sits."""

    def __init__(self, name: str, pixelsize: Optional[float] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"place {name}")
        form = QFormLayout(self)
        self.pixelsize = QDoubleSpinBox(minimum=0.01, maximum=1e6, decimals=2,
                                        value=pixelsize or 100.0)
        self.x0 = QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=1)
        self.y0 = QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=1)
        form.addRow("pixel size (nm)", self.pixelsize)
        form.addRow("x of the first pixel (nm)", self.x0)
        form.addRow("y of the first pixel (nm)", self.y0)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> Tuple[float, float, float]:
        return self.pixelsize.value(), self.x0.value(), self.y0.value()


class StackAxesDialog(QDialog):
    """A stack of pages whose file does not say what they are: ask.

    Asked when the file is opened rather than guessed, because a wrong guess
    is silent -- two channels taken for a z-stack still show a picture, one
    that is wrong every other plane.  The answer can be corrected afterwards
    in the Render tab (*planes are*), all but the channels, which are split
    into images here.
    """

    def __init__(self, name: str, n_planes: int, shape, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QLabel, QSpinBox
        self.n_planes = int(n_planes)
        self.setWindowTitle(f"the planes of {name}")
        form = QFormLayout(self)
        form.addRow(QLabel(f"{n_planes} planes of {shape[1]} x {shape[0]} pixels, "
                           "and the file does not say what they are."))
        self.channels = QSpinBox(minimum=1, maximum=max(1, self.n_planes), value=1)
        self.channels.setToolTip("how many channels the planes hold")
        self.order = QComboBox()
        self.order.addItem("interleaved: c1, c2, c1, c2, ...", True)
        self.order.addItem("in blocks: all of c1, then c2", False)
        self.axis = QComboBox()
        self.axis.addItem("z-slices", "z")
        self.axis.addItem("time frames", "t")
        self.axis.addItem("neither", "")
        # a stack of thousands of planes is a movie; a z-stack is a few dozen
        self.axis.setCurrentIndex(1 if self.n_planes > 500 else 0)
        self.step = QDoubleSpinBox(minimum=0.0, maximum=1e9, decimals=3)
        self.step_label = QLabel()
        self.error = QLabel("")
        self.error.setStyleSheet("color: #e0453a")
        form.addRow("channels", self.channels)
        form.addRow("channel order", self.order)
        form.addRow("planes are", self.axis)
        form.addRow(self.step_label, self.step)
        form.addRow(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.axis.currentIndexChanged.connect(self._axis_changed)
        self.channels.valueChanged.connect(self._axis_changed)
        self._axis_changed()

    def _axis_changed(self) -> None:
        axis = self.axis.currentData()
        self.order.setEnabled(self.channels.value() > 1)
        self.step.setEnabled(bool(axis))
        self.step_label.setText("frame time (s)" if axis == "t" else "z step (nm)")
        if axis == "z" and self.step.value() <= 0 or axis == "t" and self.step.value() > 60:
            self.step.setValue(100.0 if axis == "z" else 0.0)
        self.error.setText("" if self.n_planes % self.channels.value() == 0 else
                           f"{self.n_planes} planes do not divide into "
                           f"{self.channels.value()} channels")

    def _accept(self) -> None:
        if self.n_planes % self.channels.value() == 0:
            self.accept()

    def values(self) -> Dict:
        """The keywords of `images.assign_axes`."""
        axis = self.axis.currentData()
        return {"axis": axis, "n_channels": self.channels.value(),
                "interleaved": bool(self.order.currentData()),
                "z_step_nm": self.step.value() if axis == "z" else None,
                "frame_time_s": self.step.value() if axis == "t" else None}


def read_image_file(path, parent=None):
    """A file's images (`images.load_images`), asking what the file does not
    say: its pixel size (`PixelSizeDialog`), and what its pages are
    (`StackAxesDialog`).  None if either is cancelled."""
    from ..images import assign_axes, load_images
    name = Path(path).name
    try:
        images = load_images(path)
    except ValueError:                                  # no pixel size in the file
        dialog = PixelSizeDialog(name, parent=parent)
        if dialog.exec() != QDialog.Accepted:
            return None
        px, x0, y0 = dialog.values()
        images = load_images(path, px, x0, y0)
    out = []
    for image in images:
        if not image.axes_unknown:
            out.append(image)
            continue
        dialog = StackAxesDialog(image.name or name, image.n_frames, image.shape, parent)
        if dialog.exec() != QDialog.Accepted:
            return None
        out.extend(assign_axes(image, **dialog.values()))
    return out


class CopyLayerDialog(QDialog):
    """Which layer to copy from, and which of its settings to take.

    Everything is the wrong default: what makes two layers two -- the filter,
    or which file each one shows -- is usually exactly what must not be
    carried over, and "make this one look like that one" is the thing people
    actually want.  So the display is ticked and the rest is not.
    """

    PARTS = (("display", "display and rendering",
              "colour, LUT, contrast, mode and the rendering width"),
             ("bounds", "filter bounds",
              "replaces this layer's bounds with the other layer's"),
             ("files", "which files are shown", ""),
             ("grouping", "merge blinks or not", ""))

    def __init__(self, session, target: int, parent=None):
        super().__init__(parent)
        self.session = session
        self.target = target
        self.setWindowTitle(f"copy settings to {session.layers[target].name}")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Take these settings from:"))
        self.source = QComboBox()
        for i, layer in enumerate(session.layers):
            if i != target and not layer.is_image:
                self.source.addItem(f"{i + 1}: {layer.name}", i)
        layout.addWidget(self.source)
        self.parts = {}
        for name, label, help_text in self.PARTS:
            box = QCheckBox(label, checked=(name == "display"))
            if help_text:
                box.setToolTip(help_text)
            layout.addWidget(box)
            self.parts[name] = box
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setEnabled(self.source.count() > 0)

    def _apply(self) -> None:
        source = self.source.currentData()
        if source is not None:
            self.session.copy_layer(source, self.target,
                                    **{n: b.isChecked() for n, b in self.parts.items()})
        self.accept()


class FilePropertiesDialog(QDialog):
    """The open files and what one frame of each is in time.

    The frame time is the one property a file may lack that something reads
    later -- seconds in Stability, Blinking, Time Course -- and nothing else
    let one set it: a simulation or a bare csv records none.  The pixel size
    is shown, not edited: the nm columns were made with it.
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("File properties")
        self.session = session
        layout = QVBoxLayout(self)
        files = session.files
        self.table = QTableWidget(len(files), 4, self)
        self.table.setHorizontalHeaderLabels(
            ["file", "localizations", "pixel size (nm)", "frame time (ms)"])
        self.times = []
        for row, info in enumerate(files):
            for col, text in enumerate((info.name, f"{info.n:,}",
                                        f"{info.pixelsize_nm:g}" if info.pixelsize_nm
                                        else "-")):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setToolTip(str(info.path))
                self.table.setItem(row, col, item)
            spin = QDoubleSpinBox(minimum=0.0, maximum=1e7, decimals=3)
            spin.setSpecialValueText("not known")
            spin.setValue(info.frame_time_ms or 0.0)
            spin.setToolTip("the time from one frame to the next; 0: not known")
            self.table.setCellWidget(row, 3, spin)
            self.times.append(spin)
        self.table.resizeColumnsToContents()
        layout.addWidget(QLabel("the frame time turns frames into seconds wherever "
                                "a plugin reports time"))
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(560, 140 + 30 * min(len(files), 10))

    def apply(self) -> None:
        """Set the frame times that changed."""
        for number, (info, spin) in enumerate(zip(self.session.files, self.times)):
            value = spin.value() or None
            if value != info.frame_time_ms:
                self.session.set_frame_time(number, value)
