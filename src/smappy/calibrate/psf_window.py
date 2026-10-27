"""The PSF calibration window: one place to make a PSF calibration, by the method chosen.

Only widgets.  The methods are entries in `psf_methods` -- *bead average*
(the bead calibration's) and *pupil model (uiPSF)* -- each saying which of
the sections below it reads, how it runs, its pages and its bead table; the
window shows the chosen one's and knows none of them (`docs/uipsf/plan.md` §7.5).

A third method, *in situ (blinking molecules)*, learns from movies: its files are
movies, its section *in situ*, its *Detect molecules* rings them on their frames,
and its table counts them by state (`psf_methods.InSitu`).  A fourth,
*elliptical Gaussian*, makes no PSF: the curves of an elliptical Gaussian's
widths in z, for Astigmatic z, from its own section (`psf_methods.EllipticalGaussian`);
its *Use* button and the `calibrated` signal hand them to that plugin, and it is
never listed for the fitters.

What every method shares is the window's: the stacks and their *Stacks*
page (`stack_browser`), *mode* (single channel or dual colour), the two steps
*Detect beads* and *Calibrate* with a progress bar and the time left, *save
model* -- saved beside the stacks under the bead calibration's name and
handed to the fitter -- *Save as...*, the bead table with its marks for the
next *Calibrate*, the stack browser (`stack_view`, data beside the model)
and the worker thread.

torch is imported only by the pupil model's run: without it the window opens,
the pupil model says how to install it, and the bead average works as ever.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import fields
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QProcess, Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QListWidget, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
                               QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..gui import folders
from ..gui.params import SettingsForm
from ..gui.widgets import CollapsibleSection
from .core import CalibrationSettings
from .dual import DualColorSettings
from .elliptical import EllipticalCalibrationSettings
from .input import acquisition_kind, discover_acquisitions
from .psf_methods import AVERAGE_ONLY, METHODS, LearningProgress, duration
from .insitu import InSituSettings
from .insitu_run import InSituModelSettings
from .pupil import BeadSettings, OpticsSettings, RunSettings, SplineSettings
from .qt_gui import _bead_specs


def torch_missing():
    """What the pupil model lacks, with the command that brings it; None when a usable torch is here."""
    from .. import torch_install
    problem = torch_install.problem()
    if problem is None:
        return None
    return (f"The pupil model needs PyTorch: {problem}.  Install PyTorch puts {torch_install.described()} "
            f"into this Python; or, in a terminal:\n{shell_line(torch_install.command())}")


def shell_line(parts, windows: Optional[bool] = None) -> str:
    """``parts`` as a line to paste into a terminal.

    Joined with spaces, a Python under ``C:\\Program Files`` was cut at the
    space and the command found no ``C:\\Program``: cmd's quoting on Windows,
    the shell's elsewhere.  On Windows a part with ``>`` is quoted as well,
    which `subprocess.list2cmdline` leaves bare: ``torch>=2.4`` is a
    redirection in cmd and in PowerShell alike.
    """
    import shlex
    import subprocess
    if not (sys.platform.startswith("win") if windows is None else windows):
        return shlex.join(parts)
    quoted = []
    for part in parts:
        text = subprocess.list2cmdline([part])
        if not text.startswith('"') and any(c in text for c in "<>|&^"):
            text = f'"{text}"'
        quoted.append(text)
    return " ".join(quoted)


def _chosen(box):
    """A page control's value: a tick's state or a choice's text."""
    return box.isChecked() if isinstance(box, QCheckBox) else box.currentText()


class _Worker(QObject):
    done = Signal(str, object)
    progress = Signal(object)

    def __init__(self, kind, function):
        super().__init__()
        self.kind, self.function = kind, function

    def run(self):
        try:
            self.done.emit(self.kind, self.function(self.progress.emit))
        except Exception as error:                       # shown, never fatal
            self.done.emit("error", error)


class PSFCalibrationWindow(QMainWindow):
    """Files and settings on the left; the method's pages and the bead table on the right."""

    # the saved calibration's path and whether it is dual colour: which fitter it goes to
    calibrated = Signal(str, bool)

    def __init__(self, paths=(), parent=None):
        super().__init__(parent)
        from ..gui.help_window import attach_help
        attach_help(self, "Panels/PSF calibration")
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("SMAPpy - PSF calibration")
        self.setAcceptDrops(True)           # stacks dropped on it are added (`add_dropped`)
        self.paths: List[str] = [str(p) for p in paths]
        self.method = METHODS[0]
        self.run_result = None
        self.detected = None                # what Detect beads found, until a setting it read changes
        self.marks: dict = {}
        self.busy = False
        self._thread = None
        self._cancel = threading.Event()
        self.stack_viewer = None
        self.pages: list = []
        self.choices: list = []                    # per page, its choices' boxes (`Page.controls`)

        scroll = QScrollArea(widgetResizable=True)
        scroll.setWidget(self._controls())
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(430)

        from .stack_browser import StackBrowser
        self.tabs = QTabWidget()
        self.tabs.setElideMode(Qt.ElideNone)
        self.tabs.currentChanged.connect(lambda _: self.redraw())
        # the first page, always there: the stacks as recorded, the detected beads ringed on them
        self.stack_browser = StackBrowser()
        self.stack_browser.on_pick = self._select_bead
        self.tabs.addTab(self.stack_browser, "Stacks")
        self.bead_table = QTableWidget(0, 0)
        self.bead_table.verticalHeader().setVisible(False)
        self.bead_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bead_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.bead_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.bead_table.setMinimumHeight(140)
        self.bead_table.itemSelectionChanged.connect(self._selection_changed)
        toggle = QShortcut(QKeySequence(Qt.Key_Space), self)
        toggle.setContext(Qt.WindowShortcut)
        toggle.activated.connect(self.toggle_beads)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.tabs)
        table_side = QWidget()
        tlayout = QVBoxLayout(table_side)
        tlayout.setContentsMargins(2, 2, 2, 2)
        self.table_hint = QLabel()
        tlayout.addWidget(self.table_hint)
        tlayout.addWidget(self.bead_table)
        row = QHBoxLayout()
        self.toggle_button = QPushButton("Exclude / include")
        self.toggle_button.clicked.connect(self.toggle_beads)
        self.browse_button = QPushButton("Browse stacks...")
        self.browse_button.setToolTip("the beads beside the model and the residual, the average or one bead at "
                                      "a time, slice by slice in any direction, in its own window")
        self.browse_button.clicked.connect(self.browse_stack)
        self.halves_button = QPushButton("Uncertainty from halves...")
        self._halves_tip = ("learn the calibration again on random halves of its used beads, each from the "
                            "run's own start, and show how far they disagree beside the CRLB: an error bar "
                            "that bead-to-bead differences are in too")
        self.halves_button.setToolTip(self._halves_tip)
        self.halves_button.clicked.connect(self.halves)
        row.addWidget(self.toggle_button)
        row.addWidget(self.browse_button)
        row.addWidget(self.halves_button)
        row.addStretch(1)
        tlayout.addLayout(row)
        right.addWidget(table_side)
        right.setSizes([620, 220])

        whole = QSplitter(Qt.Horizontal)
        whole.addWidget(scroll)
        whole.addWidget(right)
        whole.setSizes([390, 1000])
        self.setCentralWidget(whole)
        self.status_bar = self.statusBar()
        self.resize(1400, 900)
        self._refresh_files()
        self._show_method()
        for form in (*self.detection_forms.values(), self.insitu_detection_form):
            form.changed.connect(self._detection_changed)

    # ------------------------------------------------------------ controls
    def _controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        files = QWidget()
        flayout = QVBoxLayout(files)
        flayout.setContentsMargins(0, 0, 0, 0)
        self.file_list = QListWidget()
        self.file_list.setToolTip("the bead stacks; drop files or folders here (several at once) to add them")
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setMaximumHeight(120)
        flayout.addWidget(self.file_list)
        brow = QHBoxLayout()
        for label, slot in (("Add files...", self.add_files), ("Add folder...", self.add_directory),
                            ("Remove", self.remove_paths)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            brow.addWidget(button)
        flayout.addLayout(brow)
        self.subdirectories = QCheckBox("search subdirectories")
        self.subdirectories.setChecked(True)
        flayout.addWidget(self.subdirectories)
        self.files_section = CollapsibleSection("bead stacks", files, expanded=True)
        layout.addWidget(self.files_section)

        def choice(label, items, slot):
            line = QHBoxLayout()
            line.addWidget(QLabel(label))
            box = QComboBox()
            box.addItems(items)
            box.currentTextChanged.connect(slot)
            line.addWidget(box, 1)
            layout.addLayout(line)
            return box
        self.method_box = choice("method", [m.label for m in METHODS], self.method_changed)
        self.mode_box = choice("mode", ["Single channel", "Dual colour"], self.mode_changed)

        # one RunSettings, two forms: *start from* is a choice of the model, *save model* of the run
        self.run_form = SettingsForm(RunSettings)
        self.run_form.set_shown({"start_from": False})
        layout.addWidget(self.run_form)
        self.start_form = SettingsForm(RunSettings)
        self.start_form.set_shown({"save_model": False})
        # the microscope control (`gui.microscope_form`): the first stack's microscope in one
        # line, its values a button away, locked until unlocked
        from ..gui.microscope_form import MicroscopeForm
        self.optics_form = MicroscopeForm(OpticsSettings)
        self.optics_form.field_changed.connect(self._optics_changed)
        self.optics_camera = None
        self.model_form = SettingsForm(BeadSettings)
        model = QWidget()
        mlayout = QVBoxLayout(model)
        mlayout.setContentsMargins(0, 0, 0, 0)
        mlayout.addWidget(self.start_form)
        mlayout.addWidget(self.model_form)
        self.spline_form = SettingsForm(SplineSettings)
        holder = QWidget()
        hlayout = QVBoxLayout(holder)
        hlayout.setContentsMargins(0, 0, 0, 0)
        self.detection_forms = {}
        for cls in (CalibrationSettings, DualColorSettings):
            form = SettingsForm(cls, _bead_specs(cls))
            hlayout.addWidget(form)
            self.detection_forms[cls] = form
        # in situ: its own choices (the focus, the start, the coverslip, the rounds) and, under more, how the
        # molecules are found and chosen
        self.insitu_form = SettingsForm(InSituModelSettings)
        self.insitu_form.changed.connect(self._show_astigmatism)
        self.start_form.changed.connect(self._show_astigmatism)
        self.insitu_detection_form = SettingsForm(InSituSettings)
        insitu = QWidget()
        ilayout = QVBoxLayout(insitu)
        ilayout.setContentsMargins(0, 0, 0, 0)
        ilayout.addWidget(self.insitu_form)
        ilayout.addWidget(self.insitu_detection_form)
        # elliptical Gaussian: the fit box and the curves' terms
        self.elliptical_form = SettingsForm(EllipticalCalibrationSettings)
        # 4Pi beads: where the channels are, the scan's z step and mirror steps, the model's order
        from .fourpi_beads import FourPiBeadSettings
        self.fourpi_form = SettingsForm(FourPiBeadSettings)
        self.fourpi_form.changed.connect(self._show_fourpi)
        self.sections = {"microscope": CollapsibleSection("microscope", self.optics_form, expanded=False),
                         "model": CollapsibleSection("model", model, expanded=True),
                         "in situ": CollapsibleSection("in situ", insitu, expanded=True),
                         "elliptical": CollapsibleSection("elliptical Gaussian", self.elliptical_form,
                                                          expanded=True),
                         "4Pi": CollapsibleSection("4Pi beads", self.fourpi_form, expanded=True),
                         "beads": CollapsibleSection("beads", holder, expanded=True),
                         "spline": CollapsibleSection("spline", self.spline_form, expanded=False)}
        for section in self.sections.values():
            layout.addWidget(section)

        self.notice = QLabel(torch_missing() or "")
        self.notice.setWordWrap(True)
        self.notice.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.notice)
        self.install_button = QPushButton("Install PyTorch")
        self.install_button.setToolTip("install PyTorch into the Python running SMAPpy; pip's progress shows "
                                       "below, and pressing again stops it")
        self.install_button.clicked.connect(self.install_torch)
        layout.addWidget(self.install_button)
        self._installer = None

        self.detect_button = QPushButton("Detect beads")
        self.detect_button.setToolTip("find the beads, ring them on the Stacks page and say how long "
                                      "calibrating them will take")
        self.detect_button.clicked.connect(self.detect)
        self.run_button = QPushButton("Calibrate")
        self.run_button.setToolTip("calibrate from the detected beads (detecting them first if they are not); "
                                   "the fit quality and the model against the data follow")
        self.run_button.clicked.connect(self.run)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setToolTip("stop the pupil model's learning after the current iteration; nothing is saved")
        self.stop_button.clicked.connect(self._cancel.set)
        self.found_label = QLabel("")
        self.found_label.setWordWrap(True)
        self.found_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        self.save_as_button = QPushButton("Save as...")
        self.save_as_button.setToolTip("save this calibration somewhere else, or when save model was off")
        self.save_as_button.clicked.connect(self.save_as)
        self.use_button = QPushButton("Use in the Spline 3D fitter")
        self.use_button.setToolTip("put the saved calibration into the fitter's calibration field")
        self.use_button.clicked.connect(self._use_in_fitter)
        self.keep_button = QPushButton("Add to microscope...")
        self.keep_button.setToolTip("keep this model's pupil -- every Zernike of phase and magnitude -- with "
                                    "the microscope, for the state of its 3D lens: in situ learning starts "
                                    "from it")
        self.keep_button.clicked.connect(self._keep_pupil)
        steps = QHBoxLayout()
        steps.addWidget(self.detect_button)
        steps.addWidget(self.run_button)
        steps.addWidget(self.stop_button)
        layout.addLayout(steps)
        layout.addWidget(self.found_label)
        layout.addWidget(self.progress_bar)
        for button in (self.save_as_button, self.use_button, self.keep_button):
            layout.addWidget(button)
        layout.addStretch(1)
        return panel

    @property
    def is_dual(self) -> bool:
        return self.mode_box.currentText() == "Dual colour"

    @property
    def detection_form(self) -> SettingsForm:
        return self.detection_forms[DualColorSettings if self.is_dual else CalibrationSettings]

    def _show_method(self) -> None:
        """Show what the chosen method reads, and only that."""
        reads = self.method.sections
        insitu = reads.get("in situ", False)
        for name in ("model", "spline"):
            self.sections[name].setVisible(reads[name])
        # every method reads the camera and the pixel; a model the optics too (`_show_optics`)
        self.sections["microscope"].setVisible(True)
        if hasattr(self, "paths"):
            self._show_optics()
        self.sections["in situ"].setVisible(insitu)
        self.sections["elliptical"].setVisible(reads.get("elliptical", False))
        fourpi = reads.get("4Pi", False)
        self.sections["4Pi"].setVisible(fourpi)
        self._show_fourpi()
        # 4Pi finds its beads on channel 0 by itself, and its channels are its section's
        self.sections["beads"].setVisible(not insitu and not fourpi)
        # in situ decides the magnitude itself, by its *hold*: held from the start, or symmetric and learnt
        self.model_form.set_shown({"magnitude": not insitu})
        self.files_section.button.setText(getattr(self.method, "files", "bead stacks"))
        self.detect_button.setText(f"Detect {getattr(self.method, 'finds', 'beads')}")
        if (insitu or fourpi) and self.is_dual:
            self.mode_box.setCurrentText("Single channel")          # one channel only, for now
        self.mode_box.setEnabled(not insitu and not fourpi)
        self.stack_browser.set_movie(getattr(self.method, "files", "") == "movies")
        self.start_form.setVisible(reads["start from"])
        wanted = DualColorSettings if self.is_dual else CalibrationSettings
        shared = {f.name for f in fields(CalibrationSettings)}
        for cls, form in self.detection_forms.items():
            form.setVisible(cls is wanted)
            form.set_shown({name: reads["averaging"] for name in AVERAGE_ONLY if name in shared})
        pupil = self.method.key in ("pupil", "insitu") or getattr(self.method, "needs_torch", False)
        missing = torch_missing()
        self.notice.setText(missing or "")
        self.notice.setVisible(pupil and missing is not None)
        # shown only while torch is missing or too old: nothing to press once it is right
        self.install_button.setVisible(pupil and (missing is not None or self._installer is not None))
        self.stop_button.setVisible(pupil)
        self.browse_button.setVisible(self.method.has_stack)
        self.toggle_button.setVisible(getattr(self.method, "marks", True))
        self.halves_button.setVisible(getattr(self.method, "has_halves", False))
        self.table_hint.setText("beads - space marks a used bead to be excluded"
                                + (" or a rejected one to be included" if self.method.marks_include else "")
                                + ", then Calibrate")
        self.use_button.setText(getattr(self.method, "use_label", None) or (
            "Use in the Spline 3D 2C fitter" if self.is_dual else "Use in the Spline 3D fitter"))
        self._update_buttons()

    def _show_fourpi(self) -> None:
        """The typed corners only while the channels are typed."""
        typed = self.fourpi_form.value().layout == "typed"
        self.fourpi_form.set_shown({n: typed for n in ("x_corners", "y_corners", "width", "height", "mirrored")})

    def _forget(self, message: str) -> None:
        """A change of method or mode makes the result another calibration's: dropped, settings kept."""
        if self.busy:
            return
        self.run_result = None
        self.marks = {}
        self._clear_pages()
        self.bead_table.setRowCount(0)
        self._undetect(None)
        self._show_method()
        self.status_bar.showMessage(message)

    def _detection_changed(self) -> None:
        self.stack_browser.set_paths(self.paths, self._dz_nm())
        self._undetect("detection settings changed")
        self._show_optics()               # a dual's main half chooses the pixel and the emission

    def _undetect(self, why) -> None:
        """Drop what Detect beads found: the files or a setting it read changed.  A calibration made from
        it stays, with its table; only its rings go from the Stacks page until the next detection."""
        if self.busy or self.detected is None:
            return
        self.detected = None
        self.found_label.setText(f"{why}: detect again" if why else "")
        if self.run_result is None:
            self.marks = {}                       # they named beads of that detection
            self.stack_browser.set_marks([])
            self.refresh_table()
            self._update_buttons()
    def method_changed(self, label: str) -> None:
        self.method = next(m for m in METHODS if m.label == label)
        self._forget("method changed; files and settings kept - calibrate again")

    def mode_changed(self, _text: str) -> None:
        self._forget("mode changed; files and settings kept - calibrate again")
        self._show_optics()

    def _optics_changed(self, name: str) -> None:
        """A microscope, a camera or a camera number chosen: the stacks read again, the beads found
        before dropped (their photons and pixel may change)."""
        if name in ("microscope", "camera", "conversion", "offset", "emgain", "pixel_x_nm", "pixel_y_nm"):
            self._undetect("the microscope changed")
            self._show_optics()
        elif name in ("n_immersion", "n_coverslip"):     # the coverslip's thickness and tilt shown or not
            self._show_optics()

    def _main_half(self):
        """The dual calibration's main half as set, or None on *auto*."""
        if not self.is_dual:
            return None
        try:
            main = self.detection_form.value().registration.main_channel
        except (ValueError, AttributeError):
            return None
        return None if main == "auto" else main

    def _stacks_camera(self, require: bool = False):
        """The first stack's camera, resolved with the microscope and the camera chosen here and the
        camera numbers typed here -- what every run of this window uses."""
        from ..io.tiff import open_stack
        from ..plugins.fit import CameraSettings
        from . import pupil
        try:
            optics = self.optics_form.value()
        except ValueError:
            optics = OpticsSettings()
        # a fitter's camera settings: the same resolution, the last microscope used included
        settings = CameraSettings(microscope=optics.microscope, camera=optics.camera,
                                  **optics.camera_overrides())
        path = discover_acquisitions(self.paths[:1])[0]
        if require:
            return pupil.camera_for(path, settings)
        return settings.resolve(open_stack(path), require=False)

    def _show_optics(self) -> None:
        """The first stack's camera and microscope (`microscope_db`): the line, the optics on *auto*
        greyed in the window, and only the rows the method uses."""
        from ..microscope_db import one_line
        self.optics_camera = None
        hints, details = {}, {"line": "microscope: no stacks", "status": "ok"}
        channels = 2 if self.is_dual else 1
        if self.paths:
            try:
                camera = self.optics_camera = self._stacks_camera()
                hints = OpticsSettings().hints(camera, channels, self._main_half())
                hints.update({k: getattr(camera, k) for k in ("conversion", "offset", "emgain")})
                hints["camera"] = camera.camera_name or None
                scope = camera.microscope
                missing = self._optics_missing(camera)
                status = ("missing" if missing else "fallback" if scope and scope.get("fallback")
                          else "unrecognised" if not scope else "ok")
                details = {"line": one_line(scope) if scope else "no microscope recognised",
                           "status": status, "values": self._optics_sentence(hints, missing),
                           "sources": {"microscope": (f"recognised: {scope['name']}" if scope and
                                                      not scope.get("fallback") else
                                                      "the last one used" if scope else "none")}}
            except Exception as error:               # a file that cannot be read says so when it is run
                print(f"PSF calibration: the stacks' camera: {error}")
                details = {"line": f"the stacks cannot be read: {error}", "status": "missing"}
        self.optics_form.set_hints({name: hints.get(name) for name in
                                    ("NA", "n_immersion", "wavelength_nm", "wavelength2_nm",
                                     "n_coverslip", "coverslip_um",
                                     "pixel_x_nm", "pixel_y_nm", "conversion", "offset", "emgain",
                                     "camera")})
        self.optics_form.set_details(details)
        self._show_astigmatism()
        optics = self.method.sections.get("microscope", False)      # a model: the optics too
        layered = optics and self._coverslip_matters()
        self.optics_form.set_shown({"NA": optics, "n_immersion": optics, "wavelength_nm": optics,
                                    "wavelength2_nm": optics and self.is_dual, "n_medium": optics,
                                    "n_coverslip": optics, "coverslip_um": layered,
                                    "emgain": bool(self.optics_camera is not None
                                                   and self.optics_camera.is_em)})
        # the tilt: always in situ, where molecules above the glass see it; for beads only through a coverslip
        # of another index (a water or silicone objective) -- with oil, beads on the coverslip see none
        self.model_form.set_shown({"fit_tilt": self.method.sections.get("in situ", False) or layered})

    def _coverslip_matters(self) -> bool:
        """Whether the coverslip is a layer of its own for these stacks, with the indices as typed or found."""
        try:
            optics = self.optics_form.value()
        except ValueError:
            return False
        try:
            return optics.coverslip_matters(self.optics_camera)
        except Exception:                            # no stacks, or a camera that cannot be read
            return optics.n_coverslip is not None and optics.n_coverslip != (optics.n_immersion or 1.518)

    def _show_astigmatism(self) -> None:
        """In situ's start astigmatism: the microscope entry's for its 3D lens, as the field's *auto*, under
        *more*; gone with *start from*, whose model starts instead; and *more* opened, the label orange, when
        neither gives one -- a microscope the database does not know -- since the run then needs it typed
        (`insitu_run.start_astigmatism`)."""
        from .insitu_run import start_astigmatism
        try:
            start_from = self.start_form.value().start_from
            typed = self.insitu_form.value().astigmatism_nm
        except ValueError:
            return
        found, why = start_astigmatism(InSituModelSettings(), self.optics_camera)
        self.insitu_form.set_hints({"astigmatism_nm": found})
        self.insitu_form.set_shown({"astigmatism_nm": not start_from})
        # said only once there are movies to judge: before, nothing is known to be missing
        needed = bool(getattr(self, "paths", None)) and not start_from and found is None and typed is None
        label = self.insitu_form.labels.get("astigmatism_nm")
        if label is not None:
            label.setStyleSheet("color: #c07000" if needed else "")
            label.setToolTip(f"needed: {why}" if needed else (f"found: {why}" if found is not None else ""))
        if needed and self.insitu_form.more_section is not None:
            self.insitu_form.more_section.set_expanded(True)

    def _optics_missing(self, camera) -> list:
        if not self.method.sections.get("microscope", False):
            return []
        try:
            optics = self.optics_form.value()
        except ValueError:
            return []
        return optics.missing(camera, 2 if self.is_dual else 1, self._main_half())

    def _optics_sentence(self, hints: dict, missing: list) -> str:
        bits = []
        if self.method.sections.get("microscope", False):
            for key, text in (("NA", "NA {:g}"), ("n_immersion", "immersion {:g}"),
                              ("wavelength_nm", "emission {:g} nm"), ("wavelength2_nm", "2nd {:g} nm")):
                if hints.get(key) is not None and (key != "wavelength2_nm" or self.is_dual):
                    bits.append(text.format(hints[key]))
        if hints.get("pixel_x_nm") is not None:
            x, y = hints["pixel_x_nm"], hints.get("pixel_y_nm", hints["pixel_x_nm"])
            bits.append(f"pixel {x:g} nm" if x == y else f"pixel {x:g}\u00d7{y:g} nm")
        if hints.get("conversion") is not None:
            bits.append(f"{hints['conversion']:g} e-/ADU")
        if missing:
            bits.append("missing: " + ", ".join(missing))
        return ", ".join(bits)

    def _update_buttons(self) -> None:
        ready = not ((self.method.key in ("pupil", "insitu") or getattr(self.method, "needs_torch", False))
                     and torch_missing() is not None)
        done = self.run_result is not None
        self.toggle_button.setEnabled(done or self.detected is not None)
        self.detect_button.setEnabled(ready and not self.busy)
        self.run_button.setEnabled(ready and not self.busy)
        self.stop_button.setEnabled(self.busy)
        self.save_as_button.setEnabled(done and not self.busy)
        self.browse_button.setEnabled(done)
        why = self.method.halves_needed(self.run_result) if done and getattr(self.method, "has_halves", False) else None
        self.halves_button.setEnabled(done and not self.busy and why is None)
        self.halves_button.setToolTip(why or self._halves_tip)
        self.use_button.setEnabled(done and not self.busy and self.run_result.saved is not None)
        learnt = done and self._learnt() is not None
        self.keep_button.setVisible(learnt)
        self.keep_button.setEnabled(learnt and not self.busy)

    def _learnt(self):
        """The uipsf model of the last run -- a pupil model's or in situ's -- or None (a bead average)."""
        result = getattr(self.run_result, "result", None)
        try:
            cal = getattr(result, "cal", None)
        except Exception:
            return None
        return cal if cal is not None and hasattr(cal, "shared") else None

    def _keep_pupil(self) -> None:
        """*Add to microscope...*: the run's pupil kept with the microscope the beads were taken on, for
        the state of its 3D lens (`microscope_db.keep_pupil`), after saying what and where."""
        from .. import microscope_db
        from .pupil import objective_pupil
        cal = self._learnt()
        scope = getattr(getattr(self, "optics_camera", None), "microscope", None) or {}
        if cal is None:
            return
        source = str(self.run_result.saved or (self.paths[0] if self.paths else ""))
        pupil = objective_pupil(cal, source=Path(source).name)
        entry = microscope_db.database().get(str(scope.get("name") or ""))
        key = entry.pupil_key(scope.get("states") or {}) if entry is not None else "?"
        before = entry.pupils.get(key) if entry is not None else None
        text = (f"Keep this pupil with {scope.get('name') or 'no microscope'}, for {key}: "
                f"{len(pupil['phase_nm'])} phase and {len(pupil['magnitude'])} magnitude Zernikes, "
                f"learnt at {pupil['wavelength_nm']:g} nm.  In situ learning starts from it."
                + (f"\n\nIt replaces the one from {before.get('source') or '?'} ({before.get('date')}), "
                   f"which is kept in a dated version of the entry." if before else ""))
        if self._confirm(text) != QMessageBox.Ok:
            return
        try:
            key, path = microscope_db.keep_pupil(scope, pupil)
        except Exception as error:
            self.error(error)
            return
        self.status_bar.showMessage(f"{scope.get('name')}: pupil for {key} kept in {path}")

    def _confirm(self, text: str):
        """Ask before a change to the user's microscope file (a test answers it)."""
        return QMessageBox.question(self, "Add to microscope", text, QMessageBox.Ok | QMessageBox.Cancel)

    # --------------------------------------------------------------- files
    def _refresh_files(self) -> None:
        self.file_list.clear()
        for path in self.paths:
            self.file_list.addItem(path)
        self.stack_browser.set_paths(self.paths, self._dz_nm())
        self._undetect("the files changed")
        self._show_optics()

    def _box_px(self) -> int:
        try:
            return int(self.detection_form.value().roi_size)
        except ValueError:
            return self.stack_browser.box_px

    def _dz_nm(self):
        """The detection's z step for the Stacks page; the files' own while the field is being typed in."""
        try:
            return self.detection_form.value().dz_nm
        except ValueError:
            return self.stack_browser.dz_nm

    def add_paths(self, paths) -> None:
        new = []
        for path in paths:
            if path and str(path) not in self.paths:
                self.paths.append(str(path))
                new.append(str(path))
        self._check_kind(new)
        self._refresh_files()

    def _check_kind(self, paths) -> None:
        """A movie added to a bead method, or a bead z-stack to in situ: said at once, with the method that
        reads it offered (`input.acquisition_kind`).  A movie read as a bead stack read every page into
        memory and ended in a refusal long enough to widen the window."""
        movies = getattr(self.method, "files", "") == "movies"
        wrong = []
        for path in paths:
            kind, n = acquisition_kind(path)
            if kind == ("stack" if movies else "movie"):
                wrong.append((path, n))
        if not wrong:
            return
        names = "\n".join(Path(p).name for p, _ in wrong[:5]) + ("\n..." if len(wrong) > 5 else "")
        other = next(m for m in METHODS if (getattr(m, "files", "") == "movies") != movies)
        what = ("bead z-stacks, not blinking movies" if movies else
                f"blinking movies ({wrong[0][1]} frames, no z steps), not bead z-stacks")
        box = QMessageBox(QMessageBox.Warning, "PSF calibration",
                          f"{'This looks like' if len(wrong) == 1 else 'These look like'} {what}:\n\n{names}\n\n"
                          f"{self.method.label} cannot learn from {'it' if len(wrong) == 1 else 'them'}.", parent=self)
        switch = box.addButton(f"Use {other.label}", QMessageBox.AcceptRole)
        remove = box.addButton("Remove", QMessageBox.DestructiveRole)
        box.addButton("Keep", QMessageBox.RejectRole)
        box.setDefaultButton(switch)
        self._ask(box)
        if box.clickedButton() is switch:
            self.method_box.setCurrentText(other.label)
        elif box.clickedButton() is remove:
            for path, _ in wrong:
                self.paths.remove(path)

    def _ask(self, box) -> None:
        """Show a question and wait (a test replaces it with a click)."""
        box.exec()

    def add_files(self) -> None:
        """On a Mac the system's open panel, files and folders together, several at once (`mac_panel`),
        each sorted as a drop is (`add_dropped`); elsewhere, or should the panel fail, Qt's file dialog."""
        from ..gui import mac_panel
        chosen = mac_panel.choose_files_and_folders("Bead stacks: files or folders, several at once",
                                                    folders.start())
        if chosen is not None:
            self.add_dropped(chosen)
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Bead stacks", folders.start(),
                                                "Image stacks (*.tif *.tiff *.ome.tif *.dcimg *.mat);;All files (*)")
        folders.remember(paths, stack=True)
        self.add_paths(paths)

    def add_directory(self) -> None:
        paths = folders.choose_folders(self, "Folders of bead stacks")
        if not paths:
            return
        if self.subdirectories.isChecked():
            try:
                found = [str(p) for p in discover_acquisitions(paths)]
            except Exception as error:
                self.error(error)
                return
            self.add_paths(found or paths)
        else:
            self.add_paths(paths)

    def add_dropped(self, paths) -> None:
        """Files and folders dropped on the window, several at once: what no folder dialog does -- the
        Mac's and Qt's both take one folder.  Each is searched as *Add folder...* searches (a folder that is
        a stack is that stack; *search subdirectories* goes deeper); one in which nothing is found is added
        as it is, for the detection to say what is wrong with it."""
        paths = [str(p) for p in paths]
        if not paths:
            return
        folders.remember(paths, stack=True)
        found = []
        for path in paths:
            try:
                hits = [str(p) for p in discover_acquisitions([path], recursive=self.subdirectories.isChecked())]
            except Exception as error:
                self.error(error)
                return
            found += hits or [path]
        self.add_paths(found)
        self.status_bar.showMessage(f"{len(found)} stack{'s' if len(found) != 1 else ''} added from "
                                    f"{len(paths)} dropped")

    def dragEnterEvent(self, event) -> None:
        from ..gui import drops
        if drops.accepts(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        from ..gui import drops
        paths = drops.local_paths(event.mimeData())
        event.acceptProposedAction()
        self.add_dropped(paths)

    def remove_paths(self) -> None:
        for item in self.file_list.selectedItems():
            if item.text() in self.paths:
                self.paths.remove(item.text())
        self._refresh_files()

    # ---------------------------------------------------------------- work
    def error(self, exc) -> None:
        self.status_bar.showMessage(f"{type(exc).__name__}: {exc}")
        QMessageBox.warning(self, "PSF calibration", f"{type(exc).__name__}: {exc}")

    def _work(self, kind: str, function) -> None:
        if self.busy:
            return
        self.busy = True
        self._cancel.clear()
        self._tracker = LearningProgress()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(True)
        self._update_buttons()
        self._thread = QThread()
        self._thread.setStackSize(32 * 1024 * 1024)
        self._worker = _Worker(kind, function)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress)
        self._worker.done.connect(self._finished)
        self._worker.done.connect(self._thread.quit)
        self._thread.start()

    def _progress(self, entry) -> None:
        fraction, left, text = self._tracker.update(entry)
        if "detected" in entry:                          # Calibrate detected first: shown at once
            self._show_detected(entry["detected"])
        if fraction is None:
            self.progress_bar.setRange(0, 0)             # busy, no fraction yet
        else:
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(int(round(1000 * fraction)))
            self.progress_bar.setFormat("%p%" + ("" if left is None or left < 1 else f" - about {duration(left)} left"))
        self.status_bar.showMessage(text)

    def _finished(self, kind: str, value) -> None:
        self.busy = False
        self.progress_bar.setVisible(False)
        if kind == "error":
            self._update_buttons()
            self.error(value)
            return
        if kind == "halves":
            self._update_buttons()
            if value is None:
                self.status_bar.showMessage("stopped: no uncertainty from halves")
                return
            self.status_bar.showMessage(f"uncertainty from {value.n_splits} splits into halves"
                                        + (f"; saved into {self.run_result.saved.name}"
                                           if self.run_result.saved is not None else ""))
            self._show_result(focus=next((t for t in (self.tabs.tabText(i) for i in range(self.tabs.count()))
                                          if t.startswith("Pupil and Zernikes")), None))
            return
        if kind == "detected":
            self._show_detected(value)
            self._update_buttons()
            self.status_bar.showMessage(value.text)
            return
        self.run_result = value
        self.marks = self.method.initial_marks(value)
        self._show_result()
        if value.saved is not None:
            self._list_calibration(value.saved, value.dual)
            self.calibrated.emit(str(value.saved), value.dual)

    def _list_calibration(self, path, dual: bool) -> None:
        """Add a saved calibration to the list the fitters' *auto* chooses from (`calibration_list`),
        described by the beads' microscope as this window resolved it -- a bead average does not
        record it in its file -- and the day they were taken."""
        from .. import calibration_list
        from ..io.tiff import camera_tags, open_stack
        from ..microscope_db import acquisition_date
        if not getattr(self.method, "listed", True):
            return                                 # not a PSF: the elliptical Gaussian's curves
        camera = getattr(self, "optics_camera", None)
        made = None
        try:
            day = acquisition_date(camera_tags(open_stack(self.paths[0]))) if self.paths else None
            made = day.isoformat() if day else None
        except Exception:
            pass
        try:
            calibration_list.add(path, microscope=getattr(camera, "microscope", None) or None,
                                 em_on=getattr(camera, "em_on", None), dual=dual, made=made,
                                 method=self.method.label)
        except Exception as error:                 # the list is a convenience: never the save's failure
            print(f"PSF calibration: could not list {path}: {error}")

    def _inputs(self) -> dict:
        run = self.run_form.value()
        run.start_from = self.start_form.value().start_from
        insitu = self.method.sections.get("in situ", False)
        inputs = dict(detection=self.insitu_detection_form.value() if insitu else self.detection_form.value(), run=run)
        if insitu:
            inputs["insitu"] = self.insitu_form.value()
        if self.method.sections.get("elliptical", False):
            inputs["elliptical"] = self.elliptical_form.value()
        if self.method.sections.get("4Pi", False):
            inputs["fourpi"] = self.fourpi_form.value()
        if self.paths:
            # a value typed that will not do stops the run, naming it: `_stacks_camera` would
            # otherwise put the file's in its place without a word
            self.optics_form.value()
            # the camera every run takes: as this window resolved it, chosen and typed values
            # included.  Incomplete, the method resolves its own and says what is missing
            try:
                inputs["camera"] = self._stacks_camera(require=True)
            except ValueError:
                pass
        if self.method.sections["microscope"]:
            inputs.update(optics=self.optics_form.value(), model=self.model_form.value(),
                          spline=self.spline_form.value())
            camera = getattr(self, "optics_camera", None)
            missing = inputs["optics"].missing(camera, 2 if self.is_dual else 1, self._main_half())
            if missing:
                scope = (getattr(camera, "microscope", None) or {}).get("name")
                raise ValueError(f"give the microscope's {' and '.join(missing)}: "
                                 + (f"{scope} does not give it for these stacks" if scope else
                                    "no microscope in the database recognises these stacks"))
        return inputs

    def _show_detected(self, detected) -> None:
        """A new detection: its beads in the table, ringed on the Stacks page.  A calibration from an
        earlier one is dropped -- its files stay on disk -- since the table can only hold one set of beads."""
        if self.run_result is not None:
            self.run_result = None
            self._clear_pages()
        self.marks = {}
        self.detected = detected
        self.found_label.setText(detected.text)
        sources = getattr(detected.found, "sources", None) or []
        self.stack_browser.box_px = self._box_px()
        states = getattr(self.method, "detected_states", None)
        self.stack_browser.set_marks(detected.positions, sources, states(detected) if states else None)
        self.refresh_table()
        self._update_buttons()

    def detect(self) -> None:
        """Find the beads and say how many and how long calibrating them will take; nothing is learnt."""
        if not self.paths:
            QMessageBox.information(self, "no stacks", "add bead stacks first")
            return
        try:
            inputs = self._inputs()
        except ValueError as error:
            self.error(error)
            return
        method, paths, dual = self.method, list(self.paths), self.is_dual
        self.status_bar.showMessage("detecting beads...")
        self._work("detected", lambda progress: method.detect(paths, dual, inputs, progress))

    def run(self) -> None:
        """Calibrate with the settings as they are: from the detected beads, detecting them first if
        they are not (a setting detection reads drops them, `_undetect`), with the marks made on them, then
        `Method.finish`.

        One button for a first calibration and every one after it: a separate *Recalculate* reused the
        last run's beads whatever the settings said, and which of the two to press was a question.
        What it had over a new run is kept -- with every setting as the last run's and the same beads,
        only the marks changed, the learning starts from the last fit (`previous`), which is quicker."""
        if not self.paths:
            QMessageBox.information(self, "no stacks", "add bead stacks first")
            return
        try:
            inputs = self._inputs()
        except ValueError as error:
            self.error(error)
            return
        method, paths, dual, detected = self.method, list(self.paths), self.is_dual, self.detected
        # the marks name beads of this detection; detecting again would number the beads anew
        marks = dict(self.marks) if detected is not None else {}
        last = self.run_result
        again = (last is not None and detected is not None and last.dual == dual
                 and last.extra.get("inputs") == inputs)
        self.status_bar.showMessage("calibrating again from the last fit, with your marks..." if again else
                                    "calibrating..." if detected else "detecting beads...")

        def work(progress):
            found = detected
            if found is None:
                found = method.detect(paths, dual, inputs, progress)
                progress({"message": found.text, "detected": found})
                progress({"restart": True})        # one bar for the calibration; detection had its own
            run = method.run(paths, dual, inputs, progress, self._cancel, detected=found, marks=marks or None,
                             previous=last if again else None)
            run.extra["inputs"] = inputs           # what the next Calibrate compares its settings with
            return method.finish(run, progress)
        self._work("result", work)

    def halves(self, splits=None) -> None:
        """*Uncertainty from halves*: how many splits, with what it will cost, then 2 x splits cold fits
        in the worker (`Method.halves`), *Stop* as for a calibration.  ``splits`` given (a test, a
        script) asks nothing."""
        run = self.run_result
        if run is None or not getattr(self.method, "has_halves", False) or self.busy:
            return
        why = self.method.halves_needed(run)
        if why:
            self.error(ValueError(why))
            return
        try:
            inputs = self._inputs()
        except ValueError as error:
            self.error(error)
            return
        if splits is None:
            from PySide6.QtWidgets import QInputDialog
            per = self.method.halves_seconds(run, run.extra.get("inputs", inputs))
            cost = f"about {duration(per)} a split, " if per else ""
            used = len(run.result.cal.beads.used)
            splits, ok = QInputDialog.getInt(
                self, "Uncertainty from halves",
                f"Splits of the {used} used beads into two halves, each half learnt from the run's own start "
                f"({cost}two fits each).\n3 gives the error bar to about ±40 %, 5 to about ±30 %.", 3, 2, 10)
            if not ok:
                return
        method = self.method
        self.status_bar.showMessage(f"learning {2 * splits} halves...")
        self._work("halves", lambda progress: method.halves(run, inputs, int(splits), progress, self._cancel))

    # -------------------------------------------------------------- results
    def _clear_pages(self) -> None:
        """Every result page goes; the Stacks page stays."""
        self.tabs.blockSignals(True)
        while self.tabs.count() > 1:
            widget = self.tabs.widget(1)
            self.tabs.removeTab(1)
            widget.deleteLater()
        self.tabs.blockSignals(False)
        self.pages = []
        self.choices = []

    def _table_page(self, page) -> QWidget:
        """A page that is a table (`Page.table`): numbers to read, and *Copy* puts them on the clipboard
        tab-separated, as a spreadsheet takes them."""
        columns, rows, *colours = page.table()
        colours = colours[0] if colours else [None] * len(rows)
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        table = QTableWidget(len(rows), len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        def text(value):
            return f"{value:.4g}" if isinstance(value, float) else str(value)
        from PySide6.QtGui import QColor
        for i, (row, colour) in enumerate(zip(rows, colours)):
            for j, value in enumerate(row):
                item = QTableWidgetItem(text(value))
                if colour:                       # a row the figure marks, in its colour
                    tint = QColor(colour)
                    tint.setAlpha(70)
                    item.setBackground(tint)
                table.setItem(i, j, item)
        table.resizeColumnsToContents()
        copy = QPushButton("Copy")
        copy.setToolTip("the whole table to the clipboard, tab-separated")

        def to_clipboard():
            from PySide6.QtWidgets import QApplication
            lines = ["\t".join(columns)] + ["\t".join(text(v) for v in row) for row in rows]
            QApplication.clipboard().setText("\n".join(lines))
            self.status_bar.showMessage(f"{len(rows)} rows copied")
        copy.clicked.connect(to_clipboard)
        line = QHBoxLayout()
        line.addWidget(copy)
        line.addStretch(1)
        layout.addLayout(line)
        layout.addWidget(table, 1)
        holder.table = table
        return holder

    def _page_widget(self, canvas, page) -> tuple:
        """The canvas, with the page's choices (`Page.controls`) in a row above it; (widget, boxes)."""
        if not page.controls:
            return canvas, {}
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        boxes = {}
        for label, choices in page.controls.items():
            if isinstance(choices, bool):
                box = QCheckBox(label)
                box.setChecked(choices)
                box.toggled.connect(lambda _: self.redraw())
            else:
                row.addWidget(QLabel(label))
                box = QComboBox()
                box.addItems(list(choices))
                box.currentTextChanged.connect(lambda _: self.redraw())
            row.addWidget(box)
            boxes[label] = box
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(canvas, 1)
        return holder, boxes

    def _show_result(self, focus=None) -> None:
        from matplotlib.figure import Figure

        from ..gui.canvas import canvas_class
        current = focus or (self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() > 1 else None)
        if current in (None, "Stacks"):
            current = "Model vs data"
        chosen = {page.title: {k: _chosen(b) for k, b in boxes.items()}
                  for (_, _, page), boxes in zip(self.pages, self.choices)}
        self._clear_pages()
        self.tabs.blockSignals(True)
        for page in self.method.pages(self.run_result):
            if page.table is not None:
                self.tabs.addTab(self._table_page(page), page.title)
                self.pages.append((None, None, page))
                self.choices.append({})
                continue
            figure = Figure(figsize=(9, 6), constrained_layout=True)
            canvas = canvas_class()(figure)
            canvas.mpl_connect("pick_event", self.pick_pair)
            widget, boxes = self._page_widget(canvas, page)
            for label, value in chosen.get(page.title, {}).items():      # the choices survive a new Calibrate
                if label in boxes:
                    box = boxes[label]
                    box.setChecked(value) if isinstance(box, QCheckBox) else box.setCurrentText(value)
            self.tabs.addTab(widget, page.title)
            self.pages.append((figure, canvas, page))
            self.choices.append(boxes)
        titles = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        if current in titles:
            self.tabs.setCurrentIndex(titles.index(current))
        self.tabs.blockSignals(False)
        self.stack_browser.set_marks(self.detected.positions if self.detected else self._positions(),
                                     self._sources(), self._states())
        self.refresh_table()
        self._update_buttons()
        self.redraw()
        saved = self.run_result.saved
        self.status_bar.showMessage(self.method.summary(self.run_result) + (
            f"; saved {saved.name}" if saved else "; not saved (save model is off)"))
        if self.stack_viewer is not None and self.stack_viewer.isVisible():
            self.browse_stack()

    def _positions(self) -> list:
        from .psf_methods import _positions
        own = getattr(self.method, "positions", None)
        if own is not None:
            return own(self.run_result)
        records = getattr(getattr(self.run_result.result, "rois", None), "records", None)
        if records is None:
            records = self.run_result.result.beads.records
        return _positions(records)

    def _sources(self) -> list:
        result = self.run_result.result if self.run_result is not None else None
        found = self.detected.found if self.detected is not None else (
            getattr(result, "rois", None) or getattr(result, "beads", None))
        return list(getattr(found, "sources", None) or [])

    def _states(self) -> dict:
        """Each bead's state for its ring, from the bead table's last column the methods share; a method
        whose table counts rather than lists (in situ) says them itself."""
        own = getattr(self.method, "states", None)
        if own is not None:
            return own(self.run_result)
        rows = self.method.rows(self.run_result, {})
        state = self.method.columns.index("state")
        return {i: str(values[state]).split(":")[0] for i, (values, _) in enumerate(rows)}

    def _page_index(self) -> int:
        """The result page in front, an index into ``pages``; -1 for the Stacks page."""
        return self.tabs.currentIndex() - 1

    def redraw(self) -> None:
        """Draw the page in front, with the table's selection and marks."""
        index = self._page_index()
        if self.run_result is None or not 0 <= index < len(self.pages):
            return
        figure, canvas, page = self.pages[index]
        if figure is None:                       # a table: filled when made
            return
        excluded = {b for b, wish in self.marks.items() if wish == "exclude"}
        try:
            figure.clear()
            if page.controls:
                chosen = {k: _chosen(b) for k, b in self.choices[index].items()}
                message = page.draw(figure, self.selected_beads(), excluded, chosen)
            else:
                message = page.draw(figure, self.selected_beads(), excluded)
        except Exception as error:
            self.status_bar.showMessage(f"could not draw: {type(error).__name__}: {error}")
            return
        if isinstance(message, str) and message:
            self.status_bar.showMessage(message)
        canvas.draw_idle()

    def _selection_changed(self) -> None:
        self.stack_browser.select(self.selected_beads())
        index = self._page_index()
        if 0 <= index < len(self.pages) and self.pages[index][2].follows_selection:
            self.redraw()

    def _select_bead(self, bead: int) -> None:
        """A ring clicked on the Stacks page: its bead selected in the table, if there is one yet."""
        if bead < self.bead_table.rowCount():
            self.bead_table.selectRow(bead)

    def refresh_table(self) -> None:
        """The calibration's beads, or before one the beads detected (`Method.detected_rows`)."""
        if self.run_result is None and self.detected is None:
            self.bead_table.setRowCount(0)
            return
        chosen = self.selected_beads()
        rows = (self.method.rows(self.run_result, self.marks) if self.run_result is not None
                else self.method.detected_rows(self.detected, self.marks))
        self.bead_table.blockSignals(True)
        self.bead_table.setColumnCount(len(self.method.columns))
        self.bead_table.setHorizontalHeaderLabels(self.method.columns)
        self.bead_table.setRowCount(len(rows))
        for i, (values, greyed) in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if greyed:
                    item.setForeground(Qt.gray)
                self.bead_table.setItem(i, column, item)
        self.bead_table.resizeColumnsToContents()
        self.bead_table.blockSignals(False)
        for i in chosen:
            if i < self.bead_table.rowCount():
                self.bead_table.selectRow(i)

    def selected_beads(self) -> List[int]:
        return sorted({index.row() for index in self.bead_table.selectionModel().selectedRows()})

    def toggle_beads(self) -> None:
        """Mark the selected beads for the next Calibrate (`Method.mark`): to be left out, or for the pupil
        model a rejected one to be let in; pressed again, unmark them."""
        if self.run_result is None and self.detected is None:
            return
        for bead in self.selected_beads():
            if bead in self.marks:
                del self.marks[bead]
            else:
                self.marks[bead] = ("exclude" if self.run_result is None
                                    else self.method.mark(self.run_result, bead))
        self.refresh_table()
        self.redraw()

    def pick_pair(self, event) -> None:
        ids = getattr(event.artist, "pair_ids", None)
        if ids is None or not len(ids):
            return
        # a scatter of several beads says which point was hit; a curve is one bead
        hit = getattr(event, "ind", None)
        index = int(ids[hit[0]] if hit is not None and len(hit) and len(ids) > 1 else list(ids)[0])
        if index < self.bead_table.rowCount():
            self.bead_table.selectRow(index)

    def browse_stack(self) -> None:
        if self.run_result is None or not self.method.has_stack:
            return
        from .stack_view import StackViewer
        views, calibration = self.method.stacks(self.run_result)
        if self.stack_viewer is None:
            self.stack_viewer = StackViewer(views.data, calibration, views.titles, views.mirror_axis, self,
                                            views=views)
        else:
            self.stack_viewer.calibration = calibration
            self.stack_viewer.set_views(views)
        self.stack_viewer.show()
        self.stack_viewer.raise_()

    # ------------------------------------------------------- installing torch
    def install_torch(self) -> None:
        """pip, in a process of its own: the window stays usable, its output is the progress, and a second
        press stops it.  torch was never imported here, so what pip installs is used without a restart."""
        if self._installer is not None:
            self._installer.kill()
            return
        from .. import torch_install
        program, *arguments = torch_install.command()
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._installer_output)
        process.finished.connect(self._installer_finished)
        self._installer = process
        self.install_button.setText("Stop installing")
        self.status_bar.showMessage("installing " + torch_install.described() + "...")
        process.start(program, arguments)

    def _installer_output(self) -> None:
        lines = bytes(self._installer.readAllStandardOutput()).decode(errors="replace").strip().splitlines()
        if lines:
            self.status_bar.showMessage(lines[-1][:200])

    def _installer_finished(self, code, _status=None) -> None:
        import importlib
        importlib.invalidate_caches()
        self._installer.deleteLater()
        self._installer = None
        self.install_button.setText("Install PyTorch")
        missing = torch_missing()
        self.status_bar.showMessage("PyTorch is installed: the pupil model is ready" if missing is None
                                    else f"the install stopped (exit code {code}); {missing.splitlines()[0]}")
        self._show_method()

    # ------------------------------------------------------------- saving
    def save_as(self) -> None:
        if self.run_result is None:
            return
        from .pupil import default_save_path
        start = self.run_result.saved or default_save_path(self.paths, self.run_result.dual)
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration", str(start), "Calibration (*.h5)")
        if not path:
            return
        try:
            saved = self.method.save_as(self.run_result, path)
        except Exception as error:
            self.error(error)
            return
        self._update_buttons()
        self.status_bar.showMessage(f"saved {saved}")
        self._list_calibration(saved, self.run_result.dual)
        self.calibrated.emit(str(saved), self.run_result.dual)

    def _use_in_fitter(self) -> None:
        if self.run_result is None or self.run_result.saved is None:
            return
        self.calibrated.emit(str(self.run_result.saved), self.run_result.dual)
        fitter = ("Spline 4Pi fitter" if self.method.key == "fourpi" else
                  "Astigmatic z" if not getattr(self.method, "listed", True) else
                  "Spline 3D 2C fitter" if self.run_result.dual else "Spline 3D fitter")
        self.status_bar.showMessage(f"the {fitter} now uses {Path(self.run_result.saved).name}")


def show_psf_calibration(paths=()):
    """Open the window, making a QApplication if there is none."""
    from PySide6.QtWidgets import QApplication

    from ..gui import make_app
    standalone = QApplication.instance() is None
    app = make_app()
    window = PSFCalibrationWindow(paths)
    window.show()
    if standalone:
        app.exec()
    return window
