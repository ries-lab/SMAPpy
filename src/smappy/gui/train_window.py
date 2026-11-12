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
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QStatusBar,
                               QTableWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout,
                               QWidget)

from ..calibrate.depth import SampleSettings
from ..plugins import param
from .widgets import note_style
from ..plugins.fit import TIFF_FILTER, CameraSettings
from ..plugins.fit_decode import MODEL_FILTER

METHOD = "decode-v0.10.2"
CALIBRATION_FILTER = "Bead calibrations (*.h5 *.hdf5 *_3dcal.mat *.mat);;All files (*)"
DENSITIES = (("LD", "LD, 15 in 40 x 40 px"), ("HD", "HD, 25 in 40 x 40 px"), ("UHD", "UHD, 50 in 40 x 40 px"))
REFRESH_MS = 5000
TRY_FRAMES = 20
FOLLOW_S = 15                           # a host's jobs followed this often
SCAN_S = 60                             # the hosts' folders looked through for jobs not listed here this often
GROW = 4                                # blocks times this while the prefit has too few fully-on frames


# ----------------------------------------------------------------------------- the jobs table
COLUMNS = ("", "job", "state", "run on", "result")
#: the list's groups (`jobs.Status.group`) in its order, each with a symbol and a colour: mid
#: tones that read on a light and a dark window, the symbol for whoever does not see the colour
ORDER = ("running", "not started", "paused", "failed", "done")
SYMBOLS = {"running": "\u25b6", "not started": "\u25cb", "paused": "\u23f8", "failed": "\u2715", "done": "\u2713"}
COLOURS = {"running": "#2e9e44", "not started": "#8a8a8a", "paused": "#c06000", "failed": "#e0453a",
           "done": "#3a7bd5"}
FOUND = "found:"                          # the key of a row found on a host, not on this list


def state_text(status) -> str:
    """A job's state as the list says it, with its epoch."""
    epoch = "" if status.epoch is None else f"epoch {status.epoch}"
    of = f" of {status.epochs}" if status.epochs and status.group == "running" else ""
    if status.group == "done":
        how = f"converged at epoch {status.epoch}" if status.converged else f"all {(status.epoch or 0) + 1} epochs"
        return f"done, {how}"
    if status.group == "failed":
        died = "died" if status.state == "interrupted" else "failed"
        return f"{died}" + (f" at {epoch}" if epoch else "") + (", Start resumes" if status.epoch is not None else "")
    return status.group + (f", {epoch}{of}" if epoch else "")


def state_word(status) -> str:
    """The state alone, for the Follow line, whose curves say the epoch: *done, converged*."""
    if status.group == "done":
        return "done, converged" if status.converged else "done, all epochs"
    if status.group == "failed":
        return "died, Start resumes" if status.state == "interrupted" else "failed"
    return status.group


def _result(status) -> str:
    metrics = (status.last or {}).get("metrics") or {}
    if status.group == "failed":
        return status.message
    if not metrics or metrics.get("jaccard") is None:
        return ""
    lateral = metrics.get("rmse_lat_nm")
    return f"test set: Jaccard {metrics['jaccard']:.3f}" + (f", lateral {lateral:.0f} nm" if lateral else "")


def job_row(entry, status, problem=None) -> dict:
    """What the table shows of a job on this computer's list."""
    tip = entry.folder + ("" if entry.remote is None else f"\non {entry.host}: {entry.remote}")
    if status.message:
        tip += f"\n{status.message}"
    if problem is not None:
        tip += f"\nnot reached: {problem}"
    return {"key": entry.folder, "group": status.group, "name": entry.name, "foreign": False, "tip": tip,
            "state": state_text(status) + (" (not reached)" if problem is not None else ""),
            "run_on": "this computer" if entry.host is None else entry.host, "result": _result(status)}


def found_row(found) -> dict:
    """What the table shows of a job on a host that this list lacks: started from another computer."""
    from ..training.jobs import GROUPS, Status
    status = Status(found.state, epoch=found.epoch, converged=found.converged)
    return {"key": FOUND + found.host + ":" + found.remote, "group": GROUPS[found.state],
            "name": "\u2197 " + found.name, "foreign": True, "state": state_text(status), "run_on": found.host,
            "result": "from another computer: double-click to add it here",
            "tip": f"on {found.host}: {found.remote}\nnot on this computer's list -- started from another computer, "
                   f"or taken off this list.  Double-click to add it: its progress is followed from then on."}


def _where_choices():
    from ..training import jobs
    return [("here", "this computer")] + [(h.name, h.name) for h in jobs.hosts()]


@dataclass
class PrepareSettings:
    """What a job is prepared from: the movie, its camera, the calibration, the density to train at."""
    movie: str = param("", label="movie", kind="open_file", file_filter=TIFF_FILTER,
                       help="the acquisition the model is for: the prefit reads blocks of it, the job lands beside it")
    calibration: str = param("", label="PSF calibration", kind="open_file", file_filter=CALIBRATION_FILTER,
                             help="the bead calibration the network learns the PSF from, the one the data were taken with")
    # a pupil (Zernike) calibration only: drawn for the focus in the sample, as Spline 3D's *focus* draws it
    sample: SampleSettings = param(default_factory=SampleSettings, label="in the sample",
                                   help="where in the sample the pupil calibration is drawn for: the network "
                                        "learns that PSF, and its z is the height above the coverslip")
    density: str = param("LD", label="density", choices=DENSITIES,
                         help="how many molecules a training frame holds: LD, HD or UHD, as SMAP's")
    start_from: str = param("", label="start from", kind="open_file", file_filter=MODEL_FILTER,
                            help="a trained model the training starts from (a warm start), even one for "
                                 "other data: it trains faster than from scratch; empty: from scratch")
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
    run_on: str = param("here", label="run on", choices=lambda: _where_choices(),
                        help="this computer, or a training host (Training hosts... above)")
    epochs: int = param(1000, label="epochs", min=1, help="the most it trains")
    until_converged: bool = param(True, label="stop when it no longer improves",
                                  help="finish before the epochs when the test loss, Jaccard, efficiencies "
                                       "and RMSEs have not improved over 100 epochs (after 150 at least)")
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
    calibration: object = field(repr=False, default=None)  # the spline the network learns: drawn in the sample, or the beads'
    focus: object = None                                     # calibrate.depth.AutoDepth: how *auto* or *in situ* found it


def _source(path):
    from ..plugins.fit import SourceSettings, _open
    return _open(SourceSettings(path=str(path)), watch=False)


def in_the_sample(settings: PrepareSettings, source, camera, report=lambda entry: None):
    """``(spline, found)``: the calibration drawn for the focus in the sample (`calibrate.depth.for_sample`),
    *auto* from the movie's spots as Spline 3D finds them with its default detection and fit box; a bead
    average, or *as calibrated*, as it is."""
    from ..calibrate import depth
    from ..io.calibration import load_spline_calibration
    cal = load_spline_calibration(settings.calibration)
    sample = settings.sample
    if not depth.has_model(cal) or sample.focus in ("calibrated", "typed"):
        return depth.for_sample(cal, sample)
    from ..plugins.fit import DetectionSettings, FitSettings, SourceSettings, sample_rois
    report({"message": f"focus {'on auto' if sample.focus == 'auto' else 'in situ'}: spots from the movie"})
    images = sample_rois(source, camera, DetectionSettings().finder(), FitSettings().roisize,
                         SourceSettings(path=str(settings.movie)))

    def tell(text):
        report({"message": text})
    return depth.for_sample(cal, sample, images=images, paths=[str(settings.movie)], camera=camera,
                            report=tell)


def prefit_model(cal, camera):
    """What the prefit fits with: a spline drawn in the sample as Spline 3D fits it -- mirrored for the
    camera's EM, each fit started at the plane in focus -- or the bead calibration's path, which
    `api.fit` loads, mirrors and checks against the microscope itself."""
    from ..calibrate.depth import in_focus
    from ..io.calibration import matched_to_em
    from ..psf import SplinePSF
    if not (cal.parameters or {}).get("sample_depth"):
        return cal.source
    return SplinePSF(matched_to_em(cal, camera.em_on), z_start_nm=in_focus(cal))


def z_limits(cal):
    """The z the estimates may reach (`decode_smlm.prefit.bounds`): a drawn spline's two ends, heights above
    the coverslip; a bead spline's half range about its focus."""
    from decode_smlm import prefit as estimates_of

    from ..calibrate.depth import z_bounds
    return z_bounds(cal) if (cal.parameters or {}).get("sample_depth") else estimates_of.z_limit(cal)


def run_prefit(settings: PrepareSettings, report=lambda entry: None) -> Prepared:
    """Spline 3D on blocks of the movie (`training.prefit`) and DECODE's estimates of them, with the
    calibration drawn for the focus in the sample first where it carries a pupil model (`in_the_sample`)."""
    from decode_smlm import prefit as estimates_of

    from ..training.prefit import prefit
    if not settings.movie or not settings.calibration:
        raise ValueError("choose the movie and its bead calibration")
    source = _source(settings.movie)
    camera = settings.camera.resolve(source)
    cal, focus = in_the_sample(settings, source, camera, report)
    report({"message": f"fitting {settings.blocks} blocks of {settings.block_frames} frames with Spline 3D"})

    def read(a, b):
        return np.concatenate([block for _, block in source.frames(chunk=b - a, start=a, stop=b)])
    model = prefit_model(cal, camera)
    blocks = settings.blocks
    while True:
        found = prefit(read, source.n_frames, camera, model, n_blocks=blocks,
                       block_frames=settings.block_frames)
        blinks, locs = found.columns()
        est = estimates_of.estimates(blinks, locs, area_px=found.area_px, n_frames=found.n_frames,
                                     z_limit_nm=z_limits(cal), max_iterations=found.max_iterations)
        # a sparse movie (the SMLM challenge's: one molecule a frame) has too few fully-on frames in
        # the default blocks, and SMAP's rule from whole blinks, which takes over then, put the
        # challenge's photons at 3 times the truth: fit more of it, up to the whole movie
        more = min(blocks * GROW, source.n_frames // settings.block_frames)
        if est.n_fully_on >= estimates_of.FULLY_ON_ENOUGH or more <= blocks:
            break
        report({"message": f"{est.n_fully_on} frames a molecule was fully on in, too few: "
                           f"fitting {more} blocks of {settings.block_frames} frames"})
        blocks = more
    return Prepared(settings, camera, est, blinks, locs, source.n_frames, sample=read(0, min(4, source.n_frames)),
                    calibration=cal, focus=focus)


def prepared_by_hand(settings: PrepareSettings, s: EstimateSettings) -> Prepared:
    """What a job is written from when no prefit was run: the estimates as typed, the camera from
    the movie (its file and the microscope part), a few of its frames for the figure."""
    from decode_smlm.prefit import Estimates, lifetime
    if not settings.movie or not settings.calibration:
        raise ValueError("choose the movie and its bead calibration")
    source = _source(settings.movie)
    camera = settings.camera.resolve(source)
    cal, focus = in_the_sample(settings, source, camera)
    est = Estimates(on_time_frames=s.on_time_frames, lifetime_frames=lifetime(s.on_time_frames),
                    intensity_mu_sig=(s.photons, s.photons_sd), intensity_from="by hand", n_fully_on=0,
                    background=(s.background_min, s.background_max), z_nm=(s.z_min_nm, s.z_max_nm),
                    emitters_per_frame=float("nan"), n_blinks=0, n_localizations=0)
    n = min(4, source.n_frames)
    sample = np.concatenate([block for _, block in source.frames(chunk=n, start=0, stop=n)])
    empty = {"photons": np.zeros(0), "background": np.zeros(0)}
    return Prepared(settings, camera, est, empty, empty, source.n_frames, sample=sample, calibration=cal,
                    focus=focus)


def by_hand(prepared: Optional[Prepared]) -> bool:
    return prepared is None or prepared.estimates.intensity_from == "by hand"


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
                          calibration={"source": str(Path(s.calibration).resolve()),
                                       **calibration_conditions(s.calibration),
                                       **sample_record(prepared.calibration)},
                          data={"path": str(Path(s.movie).resolve()), "frames": prepared.n_frames,
                                "camera_roi": None if cam.roi is None else list(cam.roi),
                                "em_on": bool(cam.is_em)})
    if s.start_from:
        problem = start_problem(s.start_from, params)
        if problem:
            raise ValueError(problem)
        job.base_model = BASE_MODEL
    cal = prepared.calibration if prepared.calibration is not None else load_spline_calibration(s.calibration)
    out = jobfile.write(folder, job, params, cal)
    if s.start_from:
        shutil.copyfile(s.start_from, out / BASE_MODEL)
    jobs.add(out)
    return out


BASE_MODEL = "base_model.safetensors"


def sample_record(cal) -> dict:
    """``{"sample_depth": ...}`` of a spline drawn for a focus in the sample, for the job and the model
    file (``calibration_sample_depth``), so the fitter can say what the network's z is; {} for the beads'."""
    record = (getattr(cal, "parameters", None) or {}).get("sample_depth")
    if not record:
        return {}
    keep = ("focus_nm", "n_med", "in_focus_height_nm", "method", "blur_nm")
    return {"sample_depth": {k: record[k] for k in keep if record.get(k) is not None}}


def calibration_conditions(path) -> dict:
    """What the calibration's beads were taken under -- the microscope, its devices' states, the
    channel, EM gain (`calibration_list`: the list's description, else the file's) -- for the model
    file, so the fitter can say when data were taken otherwise (`calibration_list.differences`)."""
    from .. import calibration_list
    path = Path(path).resolve()
    entry = next((e for e in calibration_list.entries() if Path(e.path).resolve() == path), None) \
        or calibration_list.describe(path)
    if entry is None:
        return {}
    return {"microscope": entry.microscope, "em_on": entry.em_on, "made": entry.made}


def start_problem(path, params: dict) -> str:
    """Why the model at ``path`` cannot start this job's training, or "": a DECODE model with the
    network the job builds (its architecture and frame window; the method's or DECODE's defaults)."""
    import yaml
    from importlib.resources import files
    from ..training import model_file
    if not Path(path).is_file():
        return f"{Path(path).name}: no such model file"
    meta = model_file.read_metadata(path)
    if meta.get("method") != METHOD:
        return f"{Path(path).name} is not a DECODE model"
    # DECODE's defaults under the method's own (its reference file, read here: decode_smlm's
    # param_io imports torch, which this window must not)
    reference = yaml.safe_load((files("decode_smlm") / "utils" / "reference_files" / "reference.yaml").read_text(encoding="utf-8"))
    hyper = {**reference["HyperParameter"], **(params.get("HyperParameter") or {})}
    for key in ("architecture", "channels_in"):
        if key in meta and str(meta[key]) != str(hyper.get(key)):
            return (f"{Path(path).name} has another network ({key} {meta[key]}, the job's {hyper.get(key)}): "
                    f"start from scratch, or from a model of the same network")
    return ""


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
    if not train.until_converged:
        command += ["--all-epochs"]
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


#: epochs left out of a curve's y range: the first ones are far off what follows (a loss of thousands,
#: an RMSE of hundreds of nm while few molecules are matched) and would flatten the rest
SKIP_SHARE, SKIP_MIN = 0.05, 5


def _fit_range(ax, *curves, log=False, include=()) -> None:
    """The y range of ``curves`` without their first epochs (`SKIP_SHARE`, at least `SKIP_MIN`),
    and of the values ``include`` (lines drawn across it), with a margin; the first epochs still
    drawn, beyond the edge."""
    later = [c[min(len(c) - 1, max(SKIP_MIN, int(SKIP_SHARE * len(c)))):] for c in curves if len(c)]
    values = np.concatenate(later + [np.asarray(include, float)])
    values = values[np.isfinite(values) & ((values > 0) if log else True)]
    if not len(values):
        return
    lo, hi = float(values.min()), float(values.max())
    if log:
        pad = (hi / lo) ** 0.05 if hi > lo else 1.1
        ax.set_ylim(lo / pad, hi * pad)
    else:
        pad = 0.05 * (hi - lo) if hi > lo else 0.05 * abs(hi) or 1.0
        ax.set_ylim(lo - pad, hi + pad)


def draw_progress(figure, lines: list, reference: Optional[dict] = None) -> str:
    """The curves of a job's ``progress.jsonl``: the losses, Jaccard, the RMSEs and the efficiencies
    of the test set, each y range without the first epochs (`_fit_range`); returns a line saying
    where it is and how long it has left at most.  ``reference``, the median spot's CRLB
    (`training.reference`), is drawn on the RMSE as dashed lines."""
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
    train, test = series("loss_train"), series("loss_test")
    axes[0, 0].plot(e, train, label="train")
    axes[0, 0].plot(e, test, label="test")
    axes[0, 0].set(yscale="log", title="loss", xlabel="epoch")
    _fit_range(axes[0, 0], train, test, log=True)
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].plot(e, series("jaccard", True))
    axes[0, 1].set(title="Jaccard (test set)", xlabel="epoch")
    lat, ax_ = series("rmse_lat_nm", True), series("rmse_ax_nm", True)
    lateral = axes[1, 0].plot(e, lat, label="lateral")[0]
    axial = axes[1, 0].plot(e, ax_, label="axial")[0]
    bounds = ()
    if reference:
        axes[1, 0].axhline(reference["lateral_nm"], ls="--", lw=0.8, color=lateral.get_color(), label="CRLB, median spot")
        axes[1, 0].axhline(reference["axial_nm"], ls="--", lw=0.8, color=axial.get_color())
        bounds = (reference["lateral_nm"], reference["axial_nm"])
    axes[1, 0].set(title="RMSE (nm, test set)", xlabel="epoch")
    _fit_range(axes[1, 0], lat, ax_, include=bounds)
    axes[1, 0].legend(fontsize=7)
    # the SMLM Challenge's score of the Jaccard and the RMSE together: what the early stop reads
    # beside them, and a better sign of progress than either alone (the RMSE rises as more is found)
    eff_lat, eff_ax = series("eff_lat", True), series("eff_ax", True)
    axes[1, 1].plot(e, eff_lat, label="lateral")
    axes[1, 1].plot(e, eff_ax, label="axial")
    axes[1, 1].set(title="efficiency (test set)", xlabel="epoch")
    _fit_range(axes[1, 1], eff_lat, eff_ax)
    axes[1, 1].legend(fontsize=7)
    for a in axes.flat:
        a.grid(alpha=0.3)
        a.title.set_fontsize(9)
    start = next((line for line in reversed(lines) if line["kind"] == "start"), {})
    total = start.get("epochs")
    seconds = series("seconds")
    per = float(np.nanmedian(seconds[-10:])) if np.isfinite(seconds[-10:]).any() else None
    ended = any(line["kind"] in ("finished", "stopped", "error") for line in lines[-2:])
    text = f"epoch {int(e[-1])}" + (f" of at most {total}" if total else "")
    if per:
        text += f", {per:.1f} s an epoch"
    # the time trained: the epochs' own seconds, so a pause between a Stop and a Start is not counted
    spent = float(np.nansum(seconds))
    if spent > 0:
        text += f", {_duration(spent)} {'in all' if ended else 'so far'}"
    if total and per and not ended:
        text += f", at most {_duration(max(total - 1 - int(e[-1]), 0) * per)} left"
    return text


def _duration(seconds: float) -> str:
    """``1 h 05 min``, or ``12 min`` under an hour."""
    minutes = int(round(seconds / 60))
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


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

    def __init__(self, settings=None, parent=None, open_hosts=None):
        super().__init__(parent)
        self._open_hosts = open_hosts       # Preferences > Training hosts, the main window's
        from ..gui.params import SettingsForm
        from .canvas import canvas_class
        from matplotlib.figure import Figure
        self.setWindowFlag(Qt.Window)
        self.setWindowTitle("DECODE training")
        self.busy, self.prepared = False, None
        self._thread = self._worker = None
        self.follower = HostFollower()
        self.references = {}              # a job's CRLB reference, by folder

        # the jobs: a table across the window, a symbol and a colour a state, those running first
        self.job_list = QTableWidget(0, len(COLUMNS))
        self.job_list.setHorizontalHeaderLabels(COLUMNS)
        self.job_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.job_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.job_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.job_list.verticalHeader().setVisible(False)
        self.job_list.verticalHeader().setDefaultSectionSize(self.job_list.fontMetrics().height() + 6)
        header = self.job_list.horizontalHeader()
        for column in range(len(COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.Stretch)
        self.job_list.currentCellChanged.connect(lambda *_: self.refresh())
        self.job_list.cellDoubleClicked.connect(lambda row, _: self.adopt_row(row))
        self.others = []                  # jobs on the hosts this list lacks, as the follower last found them
        add_button, remove_button = QPushButton("Add existing job..."), QPushButton("Remove from list")
        add_button.setToolTip("put a job folder back on the list: one taken off it, or made on another computer")
        remove_button.setToolTip("take the job off the list; its folder, and a training running, stay")
        add_button.clicked.connect(self.add_folder)
        remove_button.clicked.connect(self.remove_selected)
        # beside the jobs, not under Try: it acts on the job chosen, tried or not
        self.use_button = QPushButton("Use this model")
        self.use_button.setToolTip("put the chosen job's newest model into the DECODE fitter's trained model "
                                   "(from a training host, copied back first)")
        self.use_button.clicked.connect(self.use_model)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Jobs"))
        row.addStretch(1)
        row.addWidget(self.use_button)
        row.addWidget(add_button)
        row.addWidget(remove_button)
        lv.addLayout(row)
        lv.addWidget(self.job_list)

        # Prepare
        self.prepare_form = SettingsForm(PrepareSettings)
        self._calibration_key = self._calibration_value = None
        self.prepare_form.changed.connect(self._prepare_changed)
        # typed by hand, or filled by *Prefit* and changed after
        self.estimate_form = SettingsForm(EstimateSettings)
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
        self.prefit_button.setToolTip("fit blocks of the movie with Spline 3D and estimate the training's "
                                      "ranges from them; or type the estimates by hand and write the job")
        self.write_button.setToolTip("write the job folder beside the movie, with the estimates as they stand")
        self.yaml_button.setToolTip("DECODE's parameters as the job will have them, to edit by hand")
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
        self.start_button.setToolTip("train the job chosen on the left where run on says, or go on from its checkpoint")
        self.stop_button.setToolTip("stop it after the epoch it is in; Start goes on from there")
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
        self.try_button = QPushButton("Try on my data")
        self.try_button.clicked.connect(self.try_model)
        self.try_button.setToolTip("localize the movie's first frame with the job's newest model")
        self.try_figure = Figure(constrained_layout=True)
        self.try_canvas = canvas_class()(self.try_figure)
        tryit = QWidget()
        yv = QVBoxLayout(tryit)
        row = QHBoxLayout()
        row.addWidget(self.try_button)
        row.addStretch(1)
        yv.addLayout(row)
        yv.addWidget(self.try_canvas, 1)

        self.tabs = QTabWidget()
        for widget, title in ((prepare, "Prepare"), (train, "Train"), (follow, "Follow"), (tryit, "Try")):
            self.tabs.addTab(widget, title)
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(left)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([170, 780])
        self.status_bar = QStatusBar()
        layout = QVBoxLayout(self)
        # what the window is for, the hosts it trains on, and the "?" every plugin's title bar has
        top = QHBoxLayout()
        intro = QLabel("Train a DECODE network for your data: prepare a job from the movie and its PSF "
                       "calibration, train it here or on a training host, then use it in the DECODE fitter.")
        intro.setWordWrap(True)
        intro.setStyleSheet(note_style())
        top.addWidget(intro, 1)
        self.hosts_button = QPushButton("Training hosts...")
        self.hosts_button.setToolTip("the workstations run on offers: Preferences > Training hosts")
        self.hosts_button.clicked.connect(self.open_hosts)
        top.addWidget(self.hosts_button)
        self.help_button = QToolButton(text="?", autoRaise=True)
        self.help_button.setToolTip("how to train a model: the steps, every control, and what to look at (F1)")
        self.help_button.clicked.connect(self.show_help)
        top.addWidget(self.help_button)
        layout.addLayout(top)
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
        self._prepare_changed()
        self.resize(900, 1000)

    # ---------------------------------------------------------- jobs
    def show_help(self) -> None:
        from .help_window import show_help
        show_help("Panels/DECODE training", self)

    def open_hosts(self) -> None:
        """Preferences > Training hosts, then *run on*'s list again."""
        if self._open_hosts is not None:
            self._open_hosts()
        self.train_form.refresh()

    def take_fitter(self, settings) -> None:
        """The DECODE fitter's movie, camera, calibration and *run on*, as a start."""
        values = {}
        source = getattr(settings, "source", None)
        if source is not None and getattr(source, "path", ""):
            values["movie"] = source.path
        model = getattr(settings, "model", None)
        if model is not None and getattr(model, "path", ""):
            values["start_from"] = model.path          # the fitter's model: a warm start for these data
            try:                                       # and its calibration, where it is still there
                from ..training import model_file
                source = model_file.read_metadata(model.path).get("calibration_source")
                if source and Path(source).is_file():
                    values["calibration"] = source
            except Exception:
                pass
        camera = getattr(settings, "camera", None)
        if camera is not None:
            from dataclasses import asdict
            values.update({f"camera.{k}": v for k, v in asdict(camera).items() if v not in (None, "")})
        if values:
            self.prepare_form.set_values(values)
        run_on = getattr(model, "run_on", None)
        if run_on:
            self.train_form.set_values({"run_on": run_on})

    def reload_jobs(self) -> None:
        from PySide6.QtGui import QBrush, QColor, QFont
        from ..training import jobs
        chosen = self.job_list.currentRow()
        chosen = None if chosen < 0 else self.job_list.item(chosen, 0).data(Qt.UserRole)
        self.others = list(self.follower.others)
        rows = [job_row(entry, jobs.status(entry), self.follower.problems.get(entry.folder))
                for entry in jobs.entries()]
        rows += [found_row(found) for found in self.others]
        rows.sort(key=lambda r: ORDER.index(r["group"]))
        self.job_list.blockSignals(True)
        self.job_list.setRowCount(len(rows))
        for i, r in enumerate(rows):
            cells = (SYMBOLS[r["group"]], r["name"], r["state"], r["run_on"], r["result"])
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(r["tip"])
                if column == 0:
                    item.setData(Qt.UserRole, r["key"])
                    item.setForeground(QBrush(QColor(COLOURS[r["group"]])))
                    item.setTextAlignment(Qt.AlignCenter)
                if r["foreign"]:
                    font = QFont()
                    font.setItalic(True)
                    item.setFont(font)
                self.job_list.setItem(i, column, item)
            if r["key"] == chosen:
                self.job_list.setCurrentCell(i, 0)
        self.job_list.blockSignals(False)

    def selected_folder(self) -> Optional[str]:
        """The chosen job's folder here; None for none, or for a job found on a host and not added."""
        row = self.job_list.currentRow()
        if row < 0 or self.job_list.item(row, 0) is None:
            return None
        key = self.job_list.item(row, 0).data(Qt.UserRole)
        return None if key is None or key.startswith(FOUND) else key

    def select_job(self, folder) -> None:
        for i in range(self.job_list.rowCount()):
            if self.job_list.item(i, 0).data(Qt.UserRole) == str(folder):
                self.job_list.setCurrentCell(i, 0)

    def adopt_row(self, row: int) -> None:
        """A job found on a host, double-clicked: put on this list, its progress followed from now."""
        item = self.job_list.item(row, 0)
        key = None if item is None else item.data(Qt.UserRole)
        if not key or not key.startswith(FOUND):
            return
        found = next((f for f in self.others if FOUND + f.host + ":" + f.remote == key), None)
        if found is None:
            return
        from ..training import remote
        self.status_bar.showMessage(f"adding {found.name} from {found.host}")
        self._work("adopted", lambda report: remote.adopt(found))

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
                row = self.job_list.currentRow()
                key = None if row < 0 or self.job_list.item(row, 0) is None else self.job_list.item(row, 0).data(Qt.UserRole)
                self.follow_label.setText("on a training host, not on this list: double-click it to add it"
                                          if key and key.startswith(FOUND) else "no job chosen")
                return
            from ..training import jobs, progress
            status = jobs.status(entry)
            if entry.folder not in self.references:        # once a job: a second, from its own files
                from ..training.reference import for_job
                self.references[entry.folder] = for_job(entry.folder)
            text = draw_progress(self.follow_figure, progress.read(entry.folder), self.references[entry.folder])
            self.follow_canvas.draw_idle()
            # the state without its epoch: the curves' line says the epoch, once
            self.follow_label.setText(f"{entry.name}: {state_word(status)}; {text}"
                                      + (f" -- {status.message}" if status.message else ""))
        except Exception as error:                         # a half-written file, a folder going away
            print(f"DECODE training: {error}")

    # ---------------------------------------------------------- work
    def _work(self, kind: str, function) -> None:
        if self.busy:
            return
        self.busy = True
        # the last job's thread may still be winding down: its `done`, which runs this window's
        # `_finished`, is emitted before its loop has quit, and *Prefit*'s starts the simulated
        # frames from there.  Dropping the last reference to a QThread still running aborts the
        # program (Qt: "QThread: Destroyed while thread is still running") -- it crashed SMAPpy
        # after every prefit with torch installed.  So it is let finish first; its work is done,
        # so this is the quit of an idle loop.
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = QThread()
        # Qt's default thread stack (512 kB on macOS) is too small for HDF5, which the prefit reads
        # (the calibration): a bus error, as in the calibration windows
        self._thread.setStackSize(32 * 1024 * 1024)
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
        elif kind == "adopted":
            self.follower.others = [f for f in self.follower.others
                                    if (f.host, f.name) != (value.host, Path(value.remote).name)]
            self.reload_jobs()
            self.status_bar.showMessage(f"{value.name} is on this list now: followed from {value.host}")
        elif kind == "stopping":
            self.train_note.setText("asked to stop: it stops after the epoch it is in, its checkpoint written there")

    def _prepare_changed(self) -> None:
        """*in the sample* only for a pupil (Zernike) calibration (`calibrate.depth.sample_state`), its rows as
        *focus* reads them.  On a signal, so it may raise; it is said, never thrown."""
        from ..calibrate.depth import sample_state
        try:
            s = self.prepare_form.value()
            shown, active, hints = sample_state(self._calibration(s.calibration), s.sample)
            shown["sample"] = shown.pop("sample.focus")
            self.prepare_form.set_shown(shown)
            self.prepare_form.set_active(active)
            self.prepare_form.set_hints(hints)
        except Exception as error:
            print(f"DECODE training: {error}")

    def _calibration(self, path):
        """The calibration at ``path``, loaded once while it is unchanged, or None: the form asks with every change."""
        try:
            key = (str(path), Path(path).stat().st_mtime)
        except (OSError, TypeError):
            return None
        if key != self._calibration_key:
            from ..io.calibration import load_spline_calibration
            try:
                value = load_spline_calibration(path)
            except Exception:
                value = None
            self._calibration_key, self._calibration_value = key, value
        return self._calibration_value

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
        from ..calibrate.depth import describe_sample
        where = describe_sample(prepared.calibration)
        self.status_bar.showMessage(f"prefit: {est.n_blinks} blinks, {est.n_localizations} localizations; "
                                    f"photons from {est.intensity_from} ({est.n_fully_on} fully-on frames)"
                                    + (f"; {where}" if where else ""))

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
            params = method_params(self.prepared, est)
            calibration = self.prepared.calibration or self.prepared.settings.calibration
            self._work("simulated", lambda report: simulated_frames(params, calibration))
        except Exception as error:
            print(f"DECODE training: {error}")

    def write(self) -> None:
        import yaml
        if by_hand(self.prepared):
            # no prefit: the estimates as typed, the camera from the movie as the form has them now
            try:
                self.prepared = prepared_by_hand(self.prepare_form.value(), self.estimate_form.value())
            except Exception as error:
                self.status_bar.showMessage(f"{type(error).__name__}: {error}")
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
        self.select_job(folder)
        self.status_bar.showMessage(f"job written: {folder}")
        self.tabs.setCurrentIndex(1)

    def start(self) -> None:
        folder = self.selected_folder()
        if folder is None:
            self.train_note.setText("choose a job on the left")
            return
        train = self.train_form.value()
        if train.run_on != "here":
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
    """The job sent to the training host ``train.run_on`` (unless it is there already) and started there."""
    from ..training import jobs, remote
    h = jobs.host(train.run_on)
    if h is None:
        raise ValueError(f"no training host {train.run_on!r}: add it in Preferences > Training hosts")
    entry = next((e for e in jobs.entries() if e.folder == str(Path(folder).resolve())), None)
    if entry is None or entry.host != h.name or not entry.remote:
        report({"message": f"sending {Path(folder).name} to {h.name}"})
        entry = remote.send(folder, h)
    report({"message": f"starting it on {h.name}"})
    remote.start(entry, train.epochs, train.device, all_epochs=not train.until_converged)
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
        self.others, self.host_problems, self.last_scan = [], {}, -1e9     # jobs on the hosts not listed here

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
        import time
        if time.monotonic() - self.last_scan >= SCAN_S:
            self.last_scan = time.monotonic()
            self.others, self.host_problems = remote.others(jobs.entries())


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
