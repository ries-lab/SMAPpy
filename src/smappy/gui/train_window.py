"""The DECODE training window: prepare a job from a movie, train it here or on a workstation, follow it, try it.

The design is the proof of concept's (h) (`docs/decode/poc-plan.md`, "(h), the design"); the window
holds no training itself.  A training is always a process of its own -- ``python -m decode_smlm.cli
train FOLDER`` here, ``smappy-decode train`` on a training host -- and the window reads what it writes:
``progress.jsonl``, ``lock.json``, ``model.safetensors`` (`smappy.training`).  So nothing here imports
torch at load time; *Prefit* runs SMAPpy's own fit, and only the simulated frames of the preparation
figure and *Try* need torch, which they say when it is not installed.

The work is module-level functions -- `run_prefit`, `write_job`, `start_here`, `draw_progress` -- that
a script or a test calls without a window; the window is their buttons.  Long work runs on a
`QThread` worker (the PSF window's pattern: errors come back as a result, never a crash), and the
jobs are read again on a repeating `QTimer`, never a ``singleShot`` that could raise.
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPlainTextEdit, QPushButton, QSplitter, QStatusBar, QTabWidget,
                               QVBoxLayout, QWidget, QFileDialog)

from ..plugins import param
from ..plugins.fit import TIFF_FILTER, CameraSettings

METHOD = "decode-v0.10.2"
CALIBRATION_FILTER = "Bead calibrations (*.h5 *.hdf5 *_3dcal.mat *.mat);;All files (*)"
DENSITIES = (("LD", "LD, 15 in 40 x 40 px"), ("HD", "HD, 25 in 40 x 40 px"), ("UHD", "UHD, 50 in 40 x 40 px"))
REFRESH_MS = 5000
TRY_FRAMES = 20
FOLLOW_S = 15                           # a host's jobs followed this often
GROW = 4                                # blocks times this while the prefit has too few fully-on frames


def _where_choices():
    from ..training import jobs
    return [("here", "this computer")] + [(h.name, h.name) for h in jobs.hosts()]


@dataclass
class PrepareSettings:
    """What a job is prepared from: the movie, its camera, the calibration, the density to train at."""
    movie: str = param("", label="movie", kind="open_file", file_filter=TIFF_FILTER,
                       help="the acquisition the model is for: the prefit reads blocks of it, the job lands beside it")
    calibration: str = param("", label="calibration", kind="open_file", file_filter=CALIBRATION_FILTER,
                             help="the bead calibration the network learns the PSF from, the one the data were taken with")
    density: str = param("LD", label="density", choices=DENSITIES,
                         help="how many molecules a training frame holds: LD, HD or UHD, as SMAP's")
    camera: CameraSettings = param(default_factory=CameraSettings, label="microscope", collapsed=True, kind="microscope")
    blocks: int = param(10, label="blocks", min=1, advanced=True,
                        help="blocks of the movie the prefit fits, spread over it")
    block_frames: int = param(50, label="frames a block", min=5, advanced=True, help="frames in each block")


@dataclass
class EstimateSettings:
    """The training's ranges, as the prefit estimated them; each may be changed before the job is written."""
    photons: float = param(5000.0, label="photons", unit="a frame", min=1.0,
                           help="the mean photons of a molecule in a frame it is on in")
    photons_sd: float = param(1000.0, label="spread", min=0.0, beside=True, help="their standard deviation")
    background_min: float = param(10.0, label="background", unit="photons/px", min=0.0,
                                  help="the least background a frame is drawn with")
    background_max: float = param(100.0, label="to", min=0.0, beside=True, help="the most")
    z_min_nm: float = param(-800.0, label="z", unit="nm", help="the lowest z drawn")
    z_max_nm: float = param(800.0, label="to", beside=True, help="the highest z drawn")
    on_time_frames: float = param(2.0, label="on-time", unit="frames", min=1.0,
                                  help="frames a blink is seen in, on average")


@dataclass
class TrainSettings:
    """Where and how long a job trains."""
    where: str = param("here", label="where", choices=lambda: _where_choices(),
                       help="this computer, or a training host of Preferences > Training hosts")
    epochs: int = param(1000, label="epochs", min=1, help="how long it trains; the test loss is within 2 % of its last by about 150")
    device: str = param("auto", label="device", choices=(("auto", "auto"), ("cuda", "CUDA"), ("mps", "Apple GPU"),
                                                         ("cpu", "CPU")), advanced=True,
                        help="where the network trains; auto: the fastest there")


# ----------------------------------------------------------------------------- the work, without a window
@dataclass
class Prepared:
    """What *Prefit* found, and what writing the job needs."""
    settings: PrepareSettings
    camera: object                         # the movie's CameraMetadata, as resolved
    estimates: object                      # decode_smlm.prefit.Estimates
    blinks: dict
    localizations: dict
    n_frames: int
    sample: np.ndarray = field(repr=False, default=None)   # a few frames, counts, for the figure


def _source(path):
    from ..plugins.fit import SourceSettings, _open
    return _open(SourceSettings(path=str(path)), watch=False)


def run_prefit(settings: PrepareSettings, report=lambda entry: None) -> Prepared:
    """Spline 3D on blocks of the movie (`training.prefit`) and DECODE's estimates of them."""
    from decode_smlm import prefit as estimates_of

    from ..io.calibration import load_spline_calibration
    from ..training.prefit import prefit
    if not settings.movie or not settings.calibration:
        raise ValueError("choose the movie and its bead calibration")
    source = _source(settings.movie)
    camera = settings.camera.resolve(source)
    report({"message": f"fitting {settings.blocks} blocks of {settings.block_frames} frames with Spline 3D"})

    def read(a, b):
        return np.concatenate([block for _, block in source.frames(chunk=b - a, start=a, stop=b)])
    cal = load_spline_calibration(settings.calibration)
    blocks = settings.blocks
    while True:
        found = prefit(read, source.n_frames, camera, settings.calibration, n_blocks=blocks,
                       block_frames=settings.block_frames)
        blinks, locs = found.columns()
        est = estimates_of.estimates(blinks, locs, area_px=found.area_px, n_frames=found.n_frames,
                                     z_limit_nm=estimates_of.z_limit(cal), max_iterations=found.max_iterations)
        # a sparse movie (the SMLM challenge's: one molecule a frame) has too few fully-on frames in
        # the default blocks, and SMAP's rule from whole blinks, which takes over then, put the
        # challenge's photons at 3 times the truth: fit more of it, up to the whole movie
        more = min(blocks * GROW, source.n_frames // settings.block_frames)
        if est.n_fully_on >= estimates_of.FULLY_ON_ENOUGH or more <= blocks:
            break
        report({"message": f"{est.n_fully_on} frames a molecule was fully on in, too few: "
                           f"fitting {more} blocks of {settings.block_frames} frames"})
        blocks = more
    return Prepared(settings, camera, est, blinks, locs, source.n_frames, sample=read(0, min(4, source.n_frames)))


def estimate_settings(est) -> EstimateSettings:
    return EstimateSettings(photons=float(est.intensity_mu_sig[0]), photons_sd=float(est.intensity_mu_sig[1]),
                            background_min=float(est.background[0]), background_max=float(est.background[1]),
                            z_min_nm=float(est.z_nm[0]), z_max_nm=float(est.z_nm[1]),
                            on_time_frames=float(est.on_time_frames))


def with_settings(est, s: EstimateSettings):
    """The estimates as edited in the form."""
    from decode_smlm.prefit import lifetime
    return replace(est, intensity_mu_sig=(s.photons, s.photons_sd), background=(s.background_min, s.background_max),
                   z_nm=(s.z_min_nm, s.z_max_nm), on_time_frames=s.on_time_frames, lifetime_frames=lifetime(s.on_time_frames))


def camera_section(camera) -> dict:
    """DECODE's ``Camera`` section from the movie's camera."""
    from decode_smlm.prefit import camera as section
    return section(camera.conversion, camera.offset, camera.pixelsize_nm_or_default[0],
                   em_gain=camera.emgain if camera.is_em else None, read_noise_e=camera.read_noise_e)


def method_params(prepared: Prepared, est) -> dict:
    from decode_smlm.prefit import method_params as params
    return params(est, camera_section(prepared.camera), prepared.settings.density)


def default_folder(movie, calibration) -> Path:
    """``<movie>_decode/<calibration>_<date>``: beside the data the model is for."""
    movie = Path(movie)
    stem = movie.name.split(".")[0]
    cal = Path(calibration).name.split(".")[0]
    day = _dt.date.today().isoformat()
    base = movie.parent / f"{stem}_decode" / f"{cal}_{day}"
    folder, k = base, 2
    while folder.exists():
        folder, k = base.with_name(f"{base.name}_{k}"), k + 1
    return folder


def write_job(prepared: Prepared, est, folder, params: Optional[dict] = None, seed: int = 0) -> Path:
    """The job folder (`training.job.write`), listed (`training.jobs.add`)."""
    from decode_smlm.prefit import simulation

    from ..io.calibration import load_spline_calibration
    from ..training import job as jobfile
    from ..training import jobs
    s = prepared.settings
    params = params or method_params(prepared, est)
    cam = prepared.camera
    job = jobfile.Job.new(method=METHOD, seed=seed, device="auto", ranges=simulation(est, s.density),
                          camera={**camera_section(cam), "name": cam.camera_name},
                          calibration={"source": str(Path(s.calibration).resolve())},
                          data={"path": str(Path(s.movie).resolve()), "frames": prepared.n_frames,
                                "camera_roi": None if cam.roi is None else list(cam.roi),
                                "em_on": bool(cam.is_em)})
    out = jobfile.write(folder, job, params, load_spline_calibration(s.calibration))
    jobs.add(out)
    return out


def torch_here() -> bool:
    return importlib.util.find_spec("torch") is not None


def start_here(folder, train: TrainSettings, python: str = sys.executable) -> subprocess.Popen:
    """``python -m decode_smlm.cli train FOLDER`` detached, its output in ``train.log``: the training
    outlives the window, and the window follows it through its files."""
    from ..handoff import own_session
    folder = Path(folder)
    command = [python, "-m", "decode_smlm.cli", "train", str(folder), "--epochs", str(train.epochs)]
    if train.device != "auto":
        command += ["--device", train.device]
    log = open(folder / "train.log", "a", encoding="utf-8")
    try:
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(folder), **own_session())
    finally:
        log.close()                       # the child keeps its own copy of the handle


def stop(folder) -> Path:
    """Ask a training here to stop at its next checkpoint: the STOP file it reads after every epoch."""
    path = Path(folder) / "STOP"
    path.touch()
    return path


def draw_progress(figure, lines: list, reference: Optional[dict] = None) -> str:
    """The curves of a job's ``progress.jsonl``: the losses, Jaccard, the RMSEs, the seconds an epoch;
    returns a line saying where it is and how long it has left.  ``reference``, the median spot's
    CRLB (`training.reference`), is drawn on the RMSE as dashed lines."""
    epochs = [line for line in lines if line["kind"] == "epoch"]
    figure.clear()
    axes = figure.subplots(2, 2)
    if not epochs:
        for a in axes.flat:
            a.set_axis_off()
        axes[0, 0].text(0.5, 0.5, "no epoch yet", ha="center", va="center", transform=axes[0, 0].transAxes)
        return "no epoch yet"
    e = np.array([line["epoch"] for line in epochs])

    def series(key, metric=False):
        return np.array([np.nan if (line.get("metrics", {}) if metric else line).get(key) is None
                         else (line["metrics"] if metric else line)[key] for line in epochs], float)
    axes[0, 0].plot(e, series("loss_train"), label="train")
    axes[0, 0].plot(e, series("loss_test"), label="test")
    axes[0, 0].set(yscale="log", title="loss", xlabel="epoch")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].plot(e, series("jaccard", True))
    axes[0, 1].set(title="Jaccard (test set)", xlabel="epoch")
    lateral = axes[1, 0].plot(e, series("rmse_lat_nm", True), label="lateral")[0]
    axial = axes[1, 0].plot(e, series("rmse_ax_nm", True), label="axial")[0]
    if reference:
        axes[1, 0].axhline(reference["lateral_nm"], ls="--", lw=0.8, color=lateral.get_color(), label="CRLB, median spot")
        axes[1, 0].axhline(reference["axial_nm"], ls="--", lw=0.8, color=axial.get_color())
    axes[1, 0].set(title="RMSE (nm, test set)", xlabel="epoch")
    axes[1, 0].legend(fontsize=7)
    seconds = series("seconds")
    axes[1, 1].plot(e, seconds)
    axes[1, 1].set(title="seconds an epoch", xlabel="epoch")
    for a in axes.flat:
        a.grid(alpha=0.3)
        a.title.set_fontsize(9)
    start = next((line for line in reversed(lines) if line["kind"] == "start"), {})
    total = start.get("epochs")
    per = float(np.nanmedian(seconds[-10:])) if np.isfinite(seconds[-10:]).any() else None
    text = f"epoch {int(e[-1])}" + (f" of {total}" if total else "")
    if total and per:
        left = max(total - 1 - int(e[-1]), 0) * per
        text += f", {per:.1f} s an epoch, about {int(left // 3600)} h {int(left % 3600 // 60):02d} min left"
    return text


# ----------------------------------------------------------------------------- the window
class _Worker(QObject):
    done = Signal(str, object)
    progress = Signal(object)

    def __init__(self, kind, function):
        super().__init__()
        self.kind, self.function = kind, function

    def run(self):
        try:
            value = self.function(self.progress.emit)
        except Exception as error:                         # shown, never fatal
            value, self.kind = error, "error"
        self.done.emit(self.kind, value)


class TrainWindow(QWidget):
    """Jobs on the left; *Prepare*, *Train*, *Follow* and *Try* on the right."""
    model_chosen = Signal(str)            # *Use this model*: the DECODE fitter's model

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        from ..gui.params import SettingsForm
        from .canvas import canvas_class
        from matplotlib.figure import Figure
        self.setWindowFlag(Qt.Window)
        self.setWindowTitle("DECODE training")
        self.busy, self.prepared = False, None
        self._thread = self._worker = None
        self.follower = HostFollower()
        self.references = {}              # a job's CRLB reference, by folder

        # the jobs
        self.job_list = QListWidget()
        self.job_list.currentItemChanged.connect(lambda *_: self.refresh())
        add_button, remove_button = QPushButton("Add folder..."), QPushButton("Remove from list")
        add_button.clicked.connect(self.add_folder)
        remove_button.clicked.connect(self.remove_selected)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("Jobs"))
        lv.addWidget(self.job_list)
        row = QHBoxLayout()
        row.addWidget(add_button)
        row.addWidget(remove_button)
        lv.addLayout(row)

        # Prepare
        self.prepare_form = SettingsForm(PrepareSettings)
        self.estimate_form = SettingsForm(EstimateSettings)
        self.estimate_form.setEnabled(False)
        self.simulated = None             # frames the training would draw, in photons; with torch only
        self._simulate_later = QTimer(self)
        self._simulate_later.setSingleShot(True)
        self._simulate_later.setInterval(800)
        self._simulate_later.timeout.connect(self.simulate)
        self.estimate_form.changed.connect(self._estimates_changed)
        self.yaml_edit = QPlainTextEdit()
        self.yaml_edit.setPlaceholderText("method.yaml, after the prefit: DECODE's parameters, used as written")
        self.yaml_edit.setVisible(False)
        self.prefit_button, self.write_button = QPushButton("Prefit"), QPushButton("Write job")
        self.yaml_button = QPushButton("Advanced: method.yaml")
        self.yaml_button.setCheckable(True)
        self.yaml_button.toggled.connect(self.yaml_edit.setVisible)
        self.prefit_button.clicked.connect(self.prefit)
        self.write_button.clicked.connect(self.write)
        self.write_button.setEnabled(False)
        self.prepare_figure = Figure(constrained_layout=True)
        self.prepare_canvas = canvas_class()(self.prepare_figure)
        prepare = QWidget()
        pv = QVBoxLayout(prepare)
        pv.addWidget(self.prepare_form)
        pv.addWidget(self.prefit_button)
        pv.addWidget(QLabel("Estimates"))
        pv.addWidget(self.estimate_form)
        pv.addWidget(self.yaml_button)
        pv.addWidget(self.yaml_edit)
        pv.addWidget(self.write_button)
        pv.addWidget(self.prepare_canvas, 1)

        # Train
        self.train_form = SettingsForm(TrainSettings)
        self.start_button, self.stop_button = QPushButton("Start"), QPushButton("Stop")
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.train_note = QLabel("")
        self.train_note.setWordWrap(True)
        train = QWidget()
        tv = QVBoxLayout(train)
        tv.addWidget(self.train_form)
        row = QHBoxLayout()
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        tv.addLayout(row)
        tv.addWidget(self.train_note)
        tv.addStretch(1)

        # Follow
        self.follow_label = QLabel("no job chosen")
        self.follow_figure = Figure(constrained_layout=True)
        self.follow_canvas = canvas_class()(self.follow_figure)
        follow = QWidget()
        fv = QVBoxLayout(follow)
        fv.addWidget(self.follow_label)
        fv.addWidget(self.follow_canvas, 1)

        # Try
        self.try_button, self.use_button = QPushButton("Try on my data"), QPushButton("Use this model")
        self.try_button.clicked.connect(self.try_model)
        self.use_button.clicked.connect(self.use_model)
        self.try_figure = Figure(constrained_layout=True)
        self.try_canvas = canvas_class()(self.try_figure)
        tryit = QWidget()
        yv = QVBoxLayout(tryit)
        row = QHBoxLayout()
        row.addWidget(self.try_button)
        row.addWidget(self.use_button)
        yv.addLayout(row)
        yv.addWidget(self.try_canvas, 1)

        self.tabs = QTabWidget()
        for widget, title in ((prepare, "Prepare"), (train, "Train"), (follow, "Follow"), (tryit, "Try")):
            self.tabs.addTab(widget, title)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 3)
        self.status_bar = QStatusBar()
        layout = QVBoxLayout(self)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status_bar)
        try:
            from .help_window import attach_help
            attach_help(self, "Panels/DECODE training")
        except Exception:                                  # help is a convenience
            pass

        self.timer = QTimer(self)
        self.timer.setInterval(REFRESH_MS)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        if settings is not None:
            self.take_fitter(settings)
        self.reload_jobs()
        self.resize(1100, 800)

    # ---------------------------------------------------------- jobs
    def take_fitter(self, settings) -> None:
        """The DECODE fitter's movie, camera and calibration, as a start for *Prepare*."""
        values = {}
        source = getattr(settings, "source", None)
        if source is not None and getattr(source, "path", ""):
            values["movie"] = source.path
        model = getattr(settings, "model", None)
        if model is not None and getattr(model, "calibration", ""):
            values["calibration"] = model.calibration
        camera = getattr(settings, "camera", None)
        if camera is not None:
            from dataclasses import asdict
            values.update({f"camera.{k}": v for k, v in asdict(camera).items() if v not in (None, "")})
        if values:
            self.prepare_form.set_values(values)

    def reload_jobs(self) -> None:
        from ..training import jobs
        chosen = self.selected_folder()
        self.job_list.blockSignals(True)
        self.job_list.clear()
        for entry in jobs.entries():
            status = jobs.status(entry)
            where = "" if entry.host is None else f" on {entry.host}"
            epoch = "" if status.epoch is None else f", epoch {status.epoch}"
            item = QListWidgetItem(f"{entry.name}: {status.state}{epoch}{where}")
            item.setData(Qt.UserRole, entry.folder)
            problem = self.follower.problems.get(entry.folder)
            item.setToolTip(entry.folder + ("" if entry.remote is None else f"\non {entry.host}: {entry.remote}")
                            + ("" if problem is None else f"\nnot reached: {problem}"))
            if problem is not None:
                item.setText(item.text() + " (not reached)")
            self.job_list.addItem(item)
            if entry.folder == chosen:
                self.job_list.setCurrentItem(item)
        self.job_list.blockSignals(False)

    def selected_folder(self) -> Optional[str]:
        item = self.job_list.currentItem()
        return None if item is None else item.data(Qt.UserRole)

    def selected_entry(self):
        from ..training import jobs
        folder = self.selected_folder()
        return next((e for e in jobs.entries() if e.folder == folder), None)

    def add_folder(self) -> None:
        from ..training import jobs
        folder = QFileDialog.getExistingDirectory(self, "A DECODE job folder")
        if folder:
            if not (Path(folder) / "job.json").exists():
                self.status_bar.showMessage(f"{folder} has no job.json: not a job folder")
                return
            jobs.add(folder)
            self.reload_jobs()

    def remove_selected(self) -> None:
        from ..training import jobs
        folder = self.selected_folder()
        if folder:
            jobs.remove(folder)
            self.reload_jobs()

    def refresh(self) -> None:
        """The list's states, and the chosen job's curves; on the timer, so never raising."""
        try:
            self.follower.poke()
            self.reload_jobs()
            entry = self.selected_entry()
            if entry is None:
                self.follow_label.setText("no job chosen")
                return
            from ..training import jobs, progress
            status = jobs.status(entry)
            if entry.folder not in self.references:        # once a job: a second, from its own files
                from ..training.reference import for_job
                self.references[entry.folder] = for_job(entry.folder)
            text = draw_progress(self.follow_figure, progress.read(entry.folder), self.references[entry.folder])
            self.follow_canvas.draw_idle()
            self.follow_label.setText(f"{entry.name}: {status.state}; {text}"
                                      + (f" -- {status.message}" if status.message else ""))
        except Exception as error:                         # a half-written file, a folder going away
            print(f"DECODE training: {error}")

    # ---------------------------------------------------------- work
    def _work(self, kind: str, function) -> None:
        if self.busy:
            return
        self.busy = True
        self._thread = QThread()
        self._worker = _Worker(kind, function)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda entry: self.status_bar.showMessage(entry.get("message", "")))
        self._worker.done.connect(self._finished)
        self._worker.done.connect(self._thread.quit)
        self._thread.start()

    def _finished(self, kind: str, value) -> None:
        self.busy = False
        if kind == "error":
            self.status_bar.showMessage(f"{type(value).__name__}: {value}")
            QMessageBox.warning(self, "DECODE training", f"{type(value).__name__}: {value}")
            return
        if kind == "prefit":
            self.show_prepared(value)
        elif kind == "try":
            self.show_try(*value)
        elif kind == "use":
            self._use(value)
        elif kind == "simulated":
            self.simulated = value
            if self.prepared is not None:
                self.draw_prepared()
        elif kind == "started":
            self.train_note.setText(value)
            self.tabs.setCurrentIndex(2)
            self.reload_jobs()
        elif kind == "stopping":
            self.train_note.setText("asked to stop: it stops after the epoch it is in, its checkpoint written there")

    def prefit(self) -> None:
        settings = self.prepare_form.value()
        self.status_bar.showMessage("prefit: reading the movie")
        self._work("prefit", lambda report: run_prefit(settings, report))

    def show_prepared(self, prepared: Prepared) -> None:
        import yaml
        self.prepared = prepared
        est = prepared.estimates
        self.estimate_form.set(estimate_settings(est))
        self.estimate_form.setEnabled(True)
        self.yaml_edit.setPlainText(yaml.safe_dump(method_params(prepared, est), sort_keys=False))
        self.write_button.setEnabled(True)
        self.simulated = None
        self.draw_prepared()
        self.simulate()
        self.status_bar.showMessage(f"prefit: {est.n_blinks} blinks, {est.n_localizations} localizations; "
                                    f"photons from {est.intensity_from} ({est.n_fully_on} fully-on frames)")

    def draw_prepared(self) -> None:
        from decode_smlm.figures import draw_frames, draw_ranges, photons
        self.prepare_figure.clear()
        est = with_settings(self.prepared.estimates, self.estimate_form.value())
        params = method_params(self.prepared, est)
        frames = self.simulated is not None and self.prepared.sample is not None
        axes = self.prepare_figure.subplots(2 if frames else 1, 3, squeeze=False)
        draw_ranges(axes[0], self.prepared.localizations, params)
        if frames:
            draw_frames(axes[1], photons(self.prepared.sample[:1], params["Camera"]), self.simulated[:1])
        self.prepare_canvas.draw_idle()

    def _estimates_changed(self) -> None:
        if self.prepared is not None:
            self.draw_prepared()
            self._simulate_later.start()

    def simulate(self) -> None:
        """Frames as the training would draw them with the estimates as they stand, beside the data:
        with torch here only (DECODE's own simulation); on a timer, so never raising."""
        try:
            if self.prepared is None or not torch_here():
                return
            if self.busy:
                self._simulate_later.start()
                return
            est = with_settings(self.prepared.estimates, self.estimate_form.value())
            params, calibration = method_params(self.prepared, est), self.prepared.settings.calibration
            self._work("simulated", lambda report: simulated_frames(params, calibration))
        except Exception as error:
            print(f"DECODE training: {error}")

    def write(self) -> None:
        import yaml
        if self.prepared is None:
            return
        est = with_settings(self.prepared.estimates, self.estimate_form.value())
        params = method_params(self.prepared, est)
        if self.yaml_button.isChecked():
            try:
                params = yaml.safe_load(self.yaml_edit.toPlainText()) or params
            except yaml.YAMLError as error:
                self.status_bar.showMessage(f"method.yaml does not read: {error}")
                return
        s = self.prepared.settings
        try:
            folder = write_job(self.prepared, est, default_folder(s.movie, s.calibration), params)
        except Exception as error:
            self.status_bar.showMessage(f"{type(error).__name__}: {error}")
            return
        self.reload_jobs()
        for i in range(self.job_list.count()):
            if self.job_list.item(i).data(Qt.UserRole) == str(folder):
                self.job_list.setCurrentRow(i)
        self.status_bar.showMessage(f"job written: {folder}")
        self.tabs.setCurrentIndex(1)

    def start(self) -> None:
        folder = self.selected_folder()
        if folder is None:
            self.train_note.setText("choose a job on the left")
            return
        train = self.train_form.value()
        if train.where != "here":
            self._work("started", lambda report: start_on_host(folder, train, report))
            return
        if not torch_here():
            self.train_note.setText("this Python has no torch: install SMAPpy's decode extra "
                                    "(pip install smappy-smlm[decode]), or train on a training host")
            return
        start_here(folder, train)
        self.train_note.setText(f"started here: {Path(folder).name}; its output in train.log")
        self.tabs.setCurrentIndex(2)
        QTimer.singleShot(1500, self._refresh_safely)

    def _refresh_safely(self) -> None:
        try:
            self.refresh()
        except Exception as error:                         # a singleShot callable must never raise
            print(f"DECODE training: {error}")

    def stop(self) -> None:
        folder = self.selected_folder()
        if folder is None:
            return
        entry = self.selected_entry()
        if entry is not None and entry.host is not None:
            from ..training import remote
            self._work("stopping", lambda report: remote.stop(entry))
            return
        stop(folder)
        self.train_note.setText("asked to stop: it stops after the epoch it is in, its checkpoint written")

    # ---------------------------------------------------------- try
    def try_model(self) -> None:
        entry = self.selected_entry()
        if entry is None:
            return
        if not torch_here():
            self.status_bar.showMessage("trying a model needs torch here: pip install smappy-smlm[decode]")
            return
        folder = Path(entry.folder)

        def run(report):
            fetch_if_remote(entry, report)
            return try_on_data(folder)
        self._work("try", run)

    def show_try(self, photons, locs) -> None:
        self.try_figure.clear()
        ax = self.try_figure.subplots()
        ax.imshow(photons, cmap="gray", vmax=np.percentile(photons, 99.8) or None)
        if len(locs):
            ax.scatter(locs[:, 0], locs[:, 1], marker="+", s=30, c="gold")
        ax.set(title=f"first frame: {len(locs)} localizations at prob >= 0.5")
        self.try_canvas.draw_idle()

    def use_model(self) -> None:
        entry = self.selected_entry()
        if entry is None:
            return
        if entry.host is not None:
            self._work("use", lambda report: fetch_if_remote(entry, report))
            return
        self._use(entry)

    def _use(self, entry) -> None:
        model = Path(entry.folder) / "model.safetensors"
        if not model.exists():
            self.status_bar.showMessage("no model yet: it is written after the first epoch")
            return
        self.model_chosen.emit(str(model))
        self.status_bar.showMessage(f"the DECODE fitter now uses {model}")


def simulated_frames(params: dict, calibration, n_frames: int = 2) -> np.ndarray:
    """Frames as the training simulates them with these settings, in photons (needs torch)."""
    from decode_smlm.figures import simulate_frames
    return simulate_frames(params, calibration, n_frames=n_frames)


def start_on_host(folder, train: TrainSettings, report=lambda entry: None) -> str:
    """The job sent to the training host ``train.where`` (unless it is there already) and started there."""
    from ..training import jobs, remote
    h = jobs.host(train.where)
    if h is None:
        raise ValueError(f"no training host {train.where!r}: add it in Preferences > Training hosts")
    entry = next((e for e in jobs.entries() if e.folder == str(Path(folder).resolve())), None)
    if entry is None or entry.host != h.name or not entry.remote:
        report({"message": f"sending {Path(folder).name} to {h.name}"})
        entry = remote.send(folder, h)
    report({"message": f"starting it on {h.name}"})
    remote.start(entry, train.epochs, train.device)
    return f"started on {h.name}: {entry.remote}; its output in train.log there"


def fetch_if_remote(entry, report=lambda entry: None):
    """The newest model of a job on a host copied back; the entry, for *Use this model*."""
    if entry.host is not None:
        from ..training import remote
        report({"message": f"copying the model from {entry.host}"})
        remote.fetch_model(entry)
    return entry


class HostFollower:
    """The progress of the jobs on training hosts copied back, in a thread of its own every
    ``FOLLOW_S``: ssh takes a second or more, the window's timer must not wait for it.  Touches no
    Qt object (the window reads the files it writes); a finished job's model is fetched once."""

    def __init__(self):
        self.last, self.thread, self.problems = 0.0, None, {}

    def poke(self, now=None) -> None:
        import threading
        import time
        now = time.monotonic() if now is None else now
        if self.thread is not None and self.thread.is_alive() or now - self.last < FOLLOW_S:
            return
        self.last = now
        self.thread = threading.Thread(target=self.follow_all, daemon=True, name="DECODE hosts")
        self.thread.start()

    def follow_all(self) -> None:
        from ..training import jobs, remote
        for entry in jobs.entries():
            if entry.host is None or not Path(entry.folder).is_dir():
                continue
            state = jobs.status(entry).state
            model = Path(entry.folder) / remote.MODEL
            if state in ("finished", "stopped", "error") and model.exists():
                continue                                   # nothing more to come from there
            try:
                remote.follow(entry)
                if jobs.status(entry).state in ("finished", "stopped"):
                    remote.fetch_model(entry)
                self.problems.pop(entry.folder, None)
            except Exception as error:                     # an unreachable host: said in the list's tooltip
                self.problems[entry.folder] = str(error)


def try_on_data(folder, frames: int = TRY_FRAMES):
    """The job's newest model on the first frames of its movie: (first frame in photons, its
    localizations' (x, y) in pixels at prob >= 0.5)."""
    from ..metadata import CameraMetadata
    from ..plugins.fit_decode import _mirrored
    from ..training import job as jobfile
    from ..training import model_file
    from ..window_engine import WindowEngine
    from decode_smlm.predict import load_predictor
    job, _ = jobfile.read(folder)
    model = Path(folder) / "model.safetensors"
    meta = model_file.read_metadata(model)
    cam = job.camera
    camera = CameraMetadata(conversion=cam["e_per_adu"], offset=cam["baseline"], em_on=bool(cam.get("em_gain")),
                            emgain=cam.get("em_gain"), pixelsize_um=1.0)
    source = _source(job.data["path"])
    n = min(frames, source.n_frames)
    stack = np.concatenate([b for _, b in source.frames(chunk=n, start=0, stop=n)])
    predictor = load_predictor(model, None, frames_mirrored=_mirrored(meta, camera))
    engine = WindowEngine(camera, predictor)
    parts = [engine.push(stack, 0), engine.end()]
    xs, ys = [], []
    for p in parts:
        if p is None:
            continue
        sel = (np.asarray(p["frame"]) == 0) & (np.asarray(p["prob"]) >= 0.5)
        xs.append(np.asarray(p["x_nm"])[sel] / 1000.0)
        ys.append(np.asarray(p["y_nm"])[sel] / 1000.0)
    from ..camera import to_photons
    locs = np.stack([np.concatenate(xs), np.concatenate(ys)], 1) if xs else np.zeros((0, 2))
    return to_photons(stack[0], camera), locs
