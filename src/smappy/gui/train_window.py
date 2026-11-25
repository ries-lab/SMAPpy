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
from ..plugins.fit import TIFF_FILTER, CameraSettings, ChannelMoveSettings, move_shown
from ..plugins.fit_decode import MODEL_FILTER
from ..channels import FourPiChannelsSettings
from ..plugins.fit_4pi import FourPiRegistrationSettings

METHOD = "decode-v0.10.2"
NETFIT = "netfit-v1"
#: the network a job trains (docs/decode/gui-plan.md §1), offered as the calibration's channels allow
#: (`model_choices`): for one channel SMAPpy's own by default and DECODE's port beside it; for two,
#: SMAPpy's own with channel 1 transformed onto channel 0's pixels and both read together from the first
#: layer (early fusion; the key stays "optimised", which saved settings name), or the channels read apart
#: and joined later, as DECODE-Plex does
ONE_CHANNEL_MODELS = (("optimised", "DECODE, optimised"), ("decode", "DECODE"))
TWO_CHANNEL_CHOICES = (("optimised", "DECODE, transformed channels"), ("plexlike", "DECODE-Plex-like"))
#: a 4Pi model trains SMAPpy's network alone, reading the four phase channels and giving the phase (docs/4pi.md §30)
FOURPI_CHOICES = (("optimised", "DECODE, 4Pi"),)
MODELS = ONE_CHANNEL_MODELS + TWO_CHANNEL_CHOICES[1:]
TWO_CHANNEL_MODELS = tuple(key for key, _ in TWO_CHANNEL_CHOICES)
#: frames the network reads around the one it fits (*frames read*, auto): one channel's time model was
#: chosen at 7 (poc-plan, "Step 4: the time model": 7 is 1.6 nm better than 5 at on-time 3, 3 loses
#: 1.3 nm at on-time 1), two channels' at 3 (the C = 2 study's, plex-plan §16)
FRAMES_READ = (("auto", "auto"), ("3", "3"), ("5", "5"), ("7", "7"))
FRAMES_AUTO = {1: 7, 2: 3, 4: 3}
#: the mean on-time the optimised models train over by default, in frames (an exponential's mean, as
#: netfit's ``lifetime_frames``): one channel the mixture 1 -- 3, within 2 % of either specialist
#: (poc-plan, "Step 4: the time model"); two channels 1, which tied 1 -- 3 on molecules on for 3
#: frames and beat it on the brief dim ones (plex-plan §16, "The time model").  Widened to take in
#: the prefit's where it lies outside.
ON_TIME_RANGE = {1: (1.0, 3.0), 2: (1.0, 1.0), 4: (1.0, 3.0)}
#: a 4Pi prefit fits this many frames at least: *4Pi z* follows the cavity's phase over windows of a quarter of them
FOURPI_PREFIT_FRAMES = 2000
CALIBRATION_FILTER = "Bead calibrations (*.h5 *.hdf5 *_3dcal.mat *.mat);;All files (*)"
#: the densities offered, molecules per um^2 in a frame -- the sample's, whatever the camera's pixel
#: (plex-plan §16, Jonas 2026-11-22): HD robust from sparse data to dense, UHD for dense data, any other
#: typed.  The sweep's pixels were 117 x 127 nm, a 40 x 40 frame 23.8 um^2: HD near its 25 (1.05), UHD a
#: little above its 50 (2.1); LD (15, 0.63) gained at most 0.012 in Jaccard over 25, on sparse data alone
DENSITIES = (("HD", "HD, 1 per µm²"), ("UHD", "UHD, 2.5 per µm²"), ("manual", "manual"))
PER_UM2 = {"HD": 1.0, "UHD": 2.5}
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
class TwoChannelPrepare:
    """A two-channel calibration's training (docs/decode/gui-plan.md, step 4): shown for one alone."""
    kind: str = param("auto", label="channels are", choices=(("auto", "auto"), ("colour", "two colours"),
                                                              ("biplane", "two focal planes")),
                      help="two colours or two focal planes (biplane); auto: from the calibration, a biplane's "
                           "planes coming into focus 200 nm or more apart")
    main: str = param("auto", label="read as the grid", choices=(("auto", "auto"), ("0", "the main half"),
                                                                 ("1", "the other half")),
                      help="the half the network reads on its own pixels, the other resampled onto them; "
                           "auto: the brighter, from the prefit")
    dyes: str = param("independent", label="dyes", choices=lambda: _dye_choices(),
                      help="colour: how each molecule's share in the second channel is drawn -- each channel "
                           "its own count, any dyes; or around the dyes the prefit found, or typed")
    typed: str = param("", label="shares", beside=True,
                       help="the dyes' shares of their photons in the calibration's second channel, as "
                            "0.78, 0.92")
    spread: float = param(0.03, label="spread", min=0.0, max=0.5,
                          help="how far a molecule's share may lie from its dye's: 0.03 narrow, 0.06 broad "
                               "(fewer outliers in crowding, the dimmest found less)")
    # how far the second channel moved on the chip since the beads, the fitters' *channel registration*'s move (gui-plan
    # §4b): the prefit pairs the channels with it; the network learns the beads' map and the fitter moves it
    move: ChannelMoveSettings = param(default_factory=ChannelMoveSettings, label="channel move", collapsed=True)


@dataclass
class FourPiPrepare:
    """A 4Pi model's training (docs/4pi.md §30): where the four channels are on the movie's frames and how they map
    onto the first, as *Spline 4Pi* takes them -- the prefit fits with them, and the job draws its training in
    them.  Shown for a 4Pi model alone."""
    channels: FourPiChannelsSettings = param(default_factory=FourPiChannelsSettings, label="channels", collapsed=True)
    registration: FourPiRegistrationSettings = param(default_factory=FourPiRegistrationSettings,
                                                     label="registration", collapsed=True)


#: *uneven backgrounds*: auto is on for a two-channel job and off for a one-channel one
UNEVEN = (("auto", "auto"), ("on", "on"), ("off", "off"))


def uneven_on(choice: str, two_channels: bool) -> bool:
    return choice == "on" or (choice == "auto" and two_channels)


def _dye_choices():
    from ..netfit.job import DYES
    return DYES


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
    # the sample's and the acquisition's, chosen for the data, not the model's; HD by default, never far
    # from the best from sparse data to UHD's (Jonas, 2026-11-22; plex-plan §16)
    density: str = param("HD", label="density", choices=DENSITIES,
                         help="how many molecules are on in a training frame, per µm² of the sample: HD, UHD or typed")
    density_manual: float = param(1.0, label="molecules", unit="per µm²", min=0.02, max=10.0, beside=True,
                                  help="the density typed: molecules on in a frame, per µm² of the sample")
    # SMAPpy's networks only: DECODE's simulator draws a flat background (plex-plan §16, Jonas 2026-11-23)
    # on for two channels, off for one (Jonas, 2026-11-23): at C = 2 it halved the pixelation and was worth a quarter
    # of the error on a live cell; at C = 1 it cost 0.005 in Jaccard and doubled the pixelation (plex-plan §16)
    uneven_background: str = param("auto", label="uneven backgrounds", choices=UNEVEN,
                                   help="half the training's movies drawn with a background that varies over the "
                                        "field, as a cell's does; auto: on for two channels, off for one")
    two_channels: TwoChannelPrepare = param(default_factory=TwoChannelPrepare, label="two channels",
                                            collapsed=True)
    fourpi: FourPiPrepare = param(default_factory=FourPiPrepare, label="4Pi", collapsed=True)
    model: str = param("optimised", label="model", choices=MODELS, advanced=True,
                       help="the network trained, as the calibration allows: one channel DECODE, optimised "
                            "(SMAPpy's, the default) or the original DECODE's; two channels transformed "
                            "channels (the default) or DECODE-Plex-like; all fit with the DECODE fitter")
    frames_read: str = param("auto", label="frames read", choices=FRAMES_READ, advanced=True,
                             help="frames the network reads around the one it fits; auto: 7 for one "
                                  "channel, 3 for two")
    samples_m: float = param(6.0, label="samples", unit="million", min=0.1, advanced=True,
                             help="simulated samples it trains on at most; 6 million is the studies'")
    start_from: str = param("", label="start from", kind="open_file", file_filter=MODEL_FILTER,
                            help="a trained model the training starts from (a warm start), even one for "
                                 "other data: it trains faster than from scratch; empty: from scratch")
    camera: CameraSettings = param(default_factory=CameraSettings, label="microscope", collapsed=True, kind="microscope")
    prefit_frames: int = param(500, label="prefit frames", min=10, advanced=True,
                               help="frames of the movie the prefit fits, in blocks of 50 spread over it; "
                                    "four times as many while too few molecules are fully on in them")


def prefit_blocks(settings: PrepareSettings) -> tuple:
    """(blocks, frames a block) of *prefit frames*: blocks of `training.prefit.BLOCK_FRAMES`
    consecutive frames, which grouping into blinks needs, as many as the frames make."""
    from ..training.prefit import BLOCK_FRAMES
    frames = int(settings.prefit_frames)
    if frames < BLOCK_FRAMES:
        return 1, frames
    return max(1, round(frames / BLOCK_FRAMES)), BLOCK_FRAMES


def per_um2(s: PrepareSettings) -> float:
    """The density chosen, molecules per µm²: *HD*'s, *UHD*'s or the one typed under *manual*."""
    return float(s.density_manual) if s.density == "manual" else PER_UM2[s.density]


def training_density(s: PrepareSettings, camera) -> float:
    """The density a job trains at, molecules in a 40 x 40 frame of the movie's pixels, as DECODE's and
    netfit's jobs take it: the density per µm² over the area those pixels cover in the sample."""
    px, py = camera.pixelsize_nm_or_default[0]
    return per_um2(s) * (40 * px / 1000) * (40 * py / 1000)


@dataclass
class EstimateSettings:
    """The training's ranges, as the prefit estimated them; each may be changed before the job is written."""
    photons: float = param(5000.0, label="photons", unit="a frame", min=1.0,
                           help="the mean photons of a molecule in a frame it is on in")
    photons_sd: float = param(1000.0, label="spread", min=0.0, beside=True, help="their standard deviation")
    # SMAPpy's networks draw the photons uniformly over a range, DECODE's normal from the mean and spread
    photons_min: float = param(500.0, label="photons", unit="a frame", min=1.0,
                               help="the fewest photons a molecule is drawn with in a frame it is on in")
    photons_max: float = param(30000.0, label="to", min=1.0, beside=True, help="the most")
    background_min: float = param(10.0, label="background", unit="photons/px", min=0.0,
                                  help="the least background a frame is drawn with")
    background_max: float = param(100.0, label="to", min=0.0, beside=True, help="the most")
    z_min_nm: float = param(-800.0, label="z", unit="nm", help="the lowest z drawn")
    z_max_nm: float = param(800.0, label="to", beside=True, help="the highest z drawn")
    on_time_frames: float = param(2.0, label="on-time", unit="frames", min=1.0,
                                  help="frames a blink is seen in, on average")
    # SMAPpy's networks draw each sample's mean on-time over a range, one network for every dye in it
    on_time_min: float = param(1.0, label="on-time", unit="frames", min=0.5,
                               help="the shortest mean time on a training sample is drawn with (an "
                                    "exponential's mean: a blink is seen in about a frame more)")
    on_time_max: float = param(3.0, label="to", min=0.5, beside=True,
                               help="the longest; the same as the shortest: one on-time")


@dataclass
class TrainSettings:
    """Where and how long a job trains."""
    run_on: str = param("here", label="run on", choices=lambda: _where_choices(),
                        help="this computer, or a training host (Training hosts... above)")
    epochs: int = param(1000, label="epochs", min=1,
                        help="the most it trains (DECODE's; an optimised model trains to its samples)")
    until_converged: bool = param(True, label="stop when it no longer improves",
                                  help="finish before the epochs when the test loss, Jaccard, efficiencies "
                                       "and RMSEs have not improved over the last epochs (Follow says which)")
    stop_window: int = param(100, label="over", unit="epochs", min=10, advanced=True,
                             help="the last epochs the rule compares with the best before them")
    stop_min_epochs: int = param(150, label="never before", unit="epochs", min=0, advanced=True,
                                 help="the epochs it trains at least")
    stop_loss_tol: float = param(1.0, label="loss within", unit="% of its fall", min=0.0, advanced=True,
                                 help="how much the loss may still gain over the window, as a share of how "
                                      "far it has fallen so far")
    stop_jaccard_tol: float = param(0.005, label="Jaccard and efficiencies within", min=0.0, advanced=True,
                                    help="how much they may still gain over the window")
    device: str = param("auto", label="device", choices=(("auto", "auto"), ("cuda", "CUDA"), ("mps", "Apple GPU"),
                                                         ("cpu", "CPU")), advanced=True,
                        help="where the network trains; auto: the fastest there")
    joint_bounds: bool = param(True, label="joint CRLB", advanced=True,
                               help="error over CRLB also against each molecule's bound with its neighbours "
                                    "and frames (Follow's dashed curves and bottom row); off: up to 20 s less "
                                    "at each start")


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
    two: object = None                                       # netfit.prefit_two.TwoChannels: a two-channel calibration's
    #: a 4Pi model's: `netfit.fourpi.prefit`'s estimates and the movie's channel layout with its maps (``layout``)
    fourpi: Optional[dict] = None
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


def psf_kind(path, spline=None) -> tuple:
    """``(channels, line)``: what kind of PSF the calibration at ``path`` is -- 1 (one channel), 2 (a
    dual ``_2ct.h5``: biplane or two colours) or 4 (4Pi) -- and a line saying so under the field;
    ``(None, "")`` for no file, ``(None, why)`` for one this does not read.  ``spline``: the file
    already loaded as one channel's, or None.  By the headers where they say it, so cheap to ask
    with every change of the form."""
    if not path or not Path(path).is_file():
        return None, ""
    from ..fourpi import is_fourpi_calibration
    from ..netfit.prefit_two import is_dual
    try:
        if Path(path).suffix.lower() in (".h5", ".hdf5") and is_fourpi_calibration(path):
            return 4, "4Pi, four phase channels: DECODE, 4Pi -- the phase too, z from it by 4Pi z"
        if is_dual(path):
            from ..calibrate.dual import load_dual_color_calibration
            from ..netfit.prefit_two import kind_of
            kind = kind_of(load_dual_color_calibration(path))
            return 2, ("two channels, " + ("two focal planes (biplane)" if kind == "biplane" else "two colours")
                       + ": the two-channel models")
        from ..calibrate.depth import has_model
        from ..field_correction import has_field
        if spline is None:
            from ..io.calibration import load_spline_calibration
            spline = load_spline_calibration(path)
        how = "a pupil model, drawn at the focus in the sample" if has_model(spline) else "a bead average"
        return 1, "one channel, " + how + (", varying over the chip" if has_field(path) else "")
    except Exception as error:
        return None, f"not a PSF calibration this reads: {error}"


def model_choices(channels) -> tuple:
    """The models a calibration of ``channels`` trains (`ONE_CHANNEL_MODELS`, `TWO_CHANNEL_CHOICES`)."""
    return TWO_CHANNEL_CHOICES if channels == 2 else FOURPI_CHOICES if channels == 4 else ONE_CHANNEL_MODELS


def two_channel_problem(settings: PrepareSettings) -> str:
    """Why these settings cannot train on their calibration's channels, or "": a two-channel calibration
    trains SMAPpy's network alone, and *DECODE-Plex-like* needs one."""
    from ..netfit.prefit_two import is_dual
    dual = bool(settings.calibration) and is_dual(settings.calibration)
    if dual and settings.model not in TWO_CHANNEL_MODELS:
        return ("a two-channel calibration trains DECODE, transformed channels or DECODE-Plex-like: "
                "choose one as model")
    if not dual and settings.model == "plexlike":
        return "DECODE-Plex-like reads two channels: choose a two-channel calibration, or another model"
    return ""


def typed_shares(text: str) -> list:
    """``"0.78, 0.92"`` as shares."""
    import re
    return [float(v) for v in re.split(r"[\s,;]+", text.strip()) if v]


def run_prefit_two(settings: PrepareSettings, report=lambda entry: None) -> Prepared:
    """Spline 3D 2C on blocks of the movie and DECODE's estimates of its molecules' total photons, with
    the two channels' own: their kind, backgrounds, brighter half and dyes (`netfit.prefit_two`)."""
    from decode_smlm import prefit as estimates_of

    from ..calibrate.dual import load_dual_color_calibration
    from ..netfit.prefit_two import prefit_two
    source = _source(settings.movie)
    camera = settings.camera.resolve(source)
    t = settings.two_channels
    n_blocks, block_frames = prefit_blocks(settings)
    report({"message": f"fitting {n_blocks} blocks of {block_frames} frames with Spline 3D 2C"})
    moved = prefit_move(settings, report)
    found, two = prefit_two(settings.movie, settings.camera, settings.calibration, t.kind,
                            moved, n_blocks=n_blocks,
                            block_frames=block_frames, n_frames=source.n_frames,
                            report=lambda text: report({"message": text}))
    blinks, locs = found.columns()
    cal = load_dual_color_calibration(settings.calibration)
    est = estimates_of.estimates(blinks, locs, area_px=found.area_px, n_frames=found.n_frames,
                                 z_limit_nm=estimates_of.z_limit(cal.main), max_iterations=found.max_iterations)

    def read(a, b):
        return np.concatenate([block for _, block in source.frames(chunk=b - a, start=a, stop=b)])
    return Prepared(settings, camera, est, blinks, locs, source.n_frames, sample=read(0, min(4, source.n_frames)),
                    calibration=str(settings.calibration), two=two)


def prefit_move(settings: PrepareSettings, report=lambda entry: None) -> tuple:
    """(columns, rows) the prefit moves the registration by: the part's, typed or measured as the
    two-channel fitters measure it (Spline 3D 2C, x and y free per channel; cached beside the movie), its
    mean over time -- the prefit pairs the channels, and blocks spread over the movie do not need more."""
    m = settings.two_channels.move
    if m.move == "none":
        return (0.0, 0.0)
    from ..plugins.fit import (DualModelSettings, DualSplineFit, DualSplineFitSettings, SourceSettings,
                               calibration_map)
    spline = DualSplineFitSettings(source=SourceSettings(path=str(settings.movie)), camera=settings.camera,
                                   model=DualModelSettings(calibration=str(settings.calibration),
                                                           registration=calibration_map(m)))
    move, _ = DualSplineFit().channel_move(spline, lambda text: report({"message": text}))
    if move is None:
        return (0.0, 0.0)
    report({"message": move.line()})
    return tuple(float(v) for v in move.mean)


def run_prefit(settings: PrepareSettings, report=lambda entry: None) -> Prepared:
    """Spline 3D on blocks of the movie (`training.prefit`) and DECODE's estimates of them, with the
    calibration drawn for the focus in the sample first where it carries a pupil model (`in_the_sample`);
    Spline 3D 2C for a two-channel calibration (`run_prefit_two`)."""
    from decode_smlm import prefit as estimates_of

    from ..netfit.prefit_two import is_dual
    from ..training.prefit import prefit
    if not settings.movie or not settings.calibration:
        raise ValueError("choose the movie and its bead calibration")
    problem = two_channel_problem(settings)
    if problem:
        raise ValueError(problem)
    if psf_kind(settings.calibration)[0] == 4:
        return run_prefit_4pi(settings, report)
    if is_dual(settings.calibration):
        return run_prefit_two(settings, report)
    source = _source(settings.movie)
    camera = settings.camera.resolve(source)
    cal, focus = in_the_sample(settings, source, camera, report)
    blocks, block_frames = prefit_blocks(settings)
    report({"message": f"fitting {blocks} blocks of {block_frames} frames with Spline 3D"})

    def read(a, b):
        return np.concatenate([block for _, block in source.frames(chunk=b - a, start=a, stop=b)])
    model = prefit_model(cal, camera)
    while True:
        found = prefit(read, source.n_frames, camera, model, n_blocks=blocks,
                       block_frames=block_frames)
        blinks, locs = found.columns()
        est = estimates_of.estimates(blinks, locs, area_px=found.area_px, n_frames=found.n_frames,
                                     z_limit_nm=z_limits(cal), max_iterations=found.max_iterations)
        # a sparse movie (the SMLM challenge's: one molecule a frame) has too few fully-on frames in
        # the default blocks, and SMAP's rule from whole blinks, which takes over then, put the
        # challenge's photons at 3 times the truth: fit more of it, up to the whole movie
        more = min(blocks * GROW, source.n_frames // block_frames)
        if est.n_fully_on >= estimates_of.FULLY_ON_ENOUGH or more <= blocks:
            break
        report({"message": f"{est.n_fully_on} frames a molecule was fully on in, too few: "
                           f"fitting {more} blocks of {block_frames} frames"})
        blocks = more
    return Prepared(settings, camera, est, blinks, locs, source.n_frames, sample=read(0, min(4, source.n_frames)),
                    calibration=cal, focus=focus)


def fourpi_fit_settings(settings: PrepareSettings, frames: Optional[int] = None):
    """*Spline 4Pi*'s settings for the prefit: the movie's first ``frames``, its microscope, the 4Pi model, the
    *4Pi* part's channels and registration; nothing saved; *4Pi z* over windows of a quarter of the frames."""
    from ..plugins.fit import OutputSettings, SourceSettings
    from ..plugins.fit_4pi import FourPiFitSettings, FourPiModelSettings
    f = FourPiFitSettings(source=SourceSettings(path=str(settings.movie), stop=frames), camera=settings.camera,
                          channels=settings.fourpi.channels,
                          model=FourPiModelSettings(calibration=str(settings.calibration),
                                                    registration=settings.fourpi.registration),
                          output=OutputSettings(path=""))
    if frames:
        f.finish.fourpi.window_frames = max(50, int(frames) // 4)
    return f


def fourpi_layout(settings: PrepareSettings) -> dict:
    """The movie's channel layout with its maps, as Spline 4Pi takes it (`FourPiFit.channel_layout`), as a record."""
    from ..plugins.fit_4pi import FourPiFit
    return FourPiFit().channel_layout(fourpi_fit_settings(settings)).to_dict()


def run_prefit_4pi(settings: PrepareSettings, report=lambda entry: None) -> Prepared:
    """*Spline 4Pi* and *4Pi z* on the movie's first frames (*prefit frames*, at least `FOURPI_PREFIT_FRAMES`), and
    `netfit.fourpi.prefit` of the table: the photons' range, each channel's background, z and the cavity's drift.
    A 4Pi table is not grouped into blinks, so the on-time is not measured: the training draws 1 -- 3 frames."""
    from decode_smlm.prefit import Estimates, lifetime

    from ..netfit.fourpi import prefit
    from ..plugins import Context
    from ..plugins.fit_4pi import FourPiFit
    source = _source(settings.movie)
    camera = settings.camera.resolve(source)
    n = min(source.n_frames, max(int(settings.prefit_frames), FOURPI_PREFIT_FRAMES))
    report({"message": f"fitting the first {n} frames with Spline 4Pi"})
    fit = fourpi_fit_settings(settings, n)
    plugin = FourPiFit()
    locs = plugin.run(Context(), fit).locs
    if locs is None or not len(locs):
        raise ValueError("Spline 4Pi found no molecules in the prefit's frames: check the movie's channels and the 4Pi model")
    cols = {k: np.asarray(locs[k]) for k in locs.keys()}
    pre = prefit(cols)
    pre["layout"] = plugin.channel_layout(fit).to_dict()
    ph = cols["photons"]
    bg = np.mean([cols[k] for k in cols if k.startswith("background_ch")], axis=0)
    mid, spread = float(np.median(ph)), float(1.4826 * np.median(np.abs(ph - np.median(ph))))
    on = 2.0
    est = Estimates(on_time_frames=on, lifetime_frames=lifetime(on), intensity_mu_sig=(mid, spread),
                    intensity_from="every Spline 4Pi fit", n_fully_on=0,
                    background=(min(b[0] for b in pre["background"]), max(b[1] for b in pre["background"])),
                    z_nm=tuple(pre["z_nm"]), emitters_per_frame=len(ph) / n, n_blinks=0, n_localizations=len(ph),
                    photons_range=tuple(pre["photons_range"]))
    localizations = {"photons": ph, "background": bg, "z_nm": cols.get("z_env_nm", cols.get("z_nm"))}

    def read(a, b):
        return np.concatenate([block for _, block in source.frames(chunk=b - a, start=a, stop=b)])
    return Prepared(settings, camera, est, localizations, localizations, source.n_frames,
                    sample=read(0, min(4, source.n_frames)), calibration=str(settings.calibration), fourpi=pre)


def fourpi_params(prepared: Prepared, est, seed: int = 0, on_time: Optional[tuple] = None) -> dict:
    """A 4Pi model's ``settings.json`` (`netfit.fourpi.settings_from_prefit`, the recipe of docs/4pi.md §30) with the
    form's estimates: the photons' range, z, and each channel's background -- the prefit's per channel, or the
    form's range for all four where it was changed -- the movie's camera, the density, samples and frames read."""
    from ..netfit.fourpi import settings_from_prefit
    from ..netfit.settings import CameraModel, SimulationSettings
    from ..training import job as jobfile
    s = prepared.settings
    pre = dict(prepared.fourpi)
    overall = (min(b[0] for b in pre["background"]), max(b[1] for b in pre["background"]))
    if not np.allclose(overall, est.background):
        pre["background"] = [tuple(float(v) for v in est.background)] * len(pre["background"])
    pre["photons_range"] = tuple(est.photons_range or pre["photons_range"])
    pre["z_nm"] = tuple(est.z_nm)
    sec = camera_section(prepared.camera)
    py, px = sec["px_size"]
    cam = CameraModel(conversion=float(sec["e_per_adu"]), offset=float(sec["baseline"]),
                      em_gain=float(sec.get("em_gain") or 0.0), read_noise_e=float(sec["read_sigma"]),
                      qe=float(sec.get("qe", 1.0)), pixel_nm=(float(px), float(py)))
    t = settings_from_prefit(pre, cam, jobfile.CALIBRATION, density=training_density(s, prepared.camera), seed=seed)
    lo, hi = on_time or on_time_range(est, 4)
    t.simulation = SimulationSettings(**{**t.simulation.__dict__, "window": frames_read(s, 4),
                                         "lifetime_frames": float(lo), "lifetime_max_frames": float(hi) if hi > lo else 0.0,
                                         "background_uneven": 0.5 if s.uneven_background == "on" else 0.0})
    t.parts.schedule.settings["max_samples"] = float(s.samples_m) * 1e6
    t.parts.schedule.settings["snapshots"] = []
    return t.to_dict()


def prepared_by_hand(settings: PrepareSettings, s: EstimateSettings) -> Prepared:
    """What a job is written from when no prefit was run: the estimates as typed, the camera from
    the movie (its file and the microscope part), a few of its frames for the figure."""
    from decode_smlm.prefit import Estimates, lifetime
    if not settings.movie or not settings.calibration:
        raise ValueError("choose the movie and its bead calibration")
    from ..netfit.prefit_two import TwoChannels, is_dual, kind_of
    source = _source(settings.movie)
    camera = settings.camera.resolve(source)
    two = fourpi = None
    if psf_kind(settings.calibration)[0] == 4:
        # typed by hand: every channel the background typed, the cavity's drift the training's default
        from ..netfit.settings import ChannelSettings
        d = ChannelSettings()
        layout = fourpi_layout(settings)
        fourpi = {"photons_range": (s.photons_min, s.photons_max), "z_nm": (s.z_min_nm, s.z_max_nm),
                  "background": [(s.background_min, s.background_max)] * len(layout["regions"]),
                  "cavity_rate_rad": d.cavity_rate_rad, "cavity_walk_rad": d.cavity_walk_rad, "layout": layout}
        cal, focus = str(settings.calibration), None
    elif is_dual(settings.calibration):
        problem = two_channel_problem(settings)
        if problem:
            raise ValueError(problem)
        from ..calibrate.dual import load_dual_color_calibration
        kind = settings.two_channels.kind
        kind = kind_of(load_dual_color_calibration(settings.calibration)) if kind == "auto" else kind
        # typed by hand: both channels the background typed; the grid the main half where auto
        two = TwoChannels(kind, [(s.background_min, s.background_max)] * 2, 0)
        cal, focus = str(settings.calibration), None
    else:
        cal, focus = in_the_sample(settings, source, camera)
    est = Estimates(on_time_frames=s.on_time_frames, lifetime_frames=lifetime(s.on_time_frames),
                    intensity_mu_sig=(s.photons, s.photons_sd), intensity_from="by hand", n_fully_on=0,
                    background=(s.background_min, s.background_max), z_nm=(s.z_min_nm, s.z_max_nm),
                    emitters_per_frame=float("nan"), n_blinks=0, n_localizations=0,
                    photons_range=(s.photons_min, s.photons_max))
    n = min(4, source.n_frames)
    sample = np.concatenate([block for _, block in source.frames(chunk=n, start=0, stop=n)])
    empty = {"photons": np.zeros(0), "background": np.zeros(0)}
    return Prepared(settings, camera, est, empty, empty, source.n_frames, sample=sample, calibration=cal,
                    focus=focus, two=two, fourpi=fourpi)


def by_hand(prepared: Optional[Prepared]) -> bool:
    return prepared is None or prepared.estimates.intensity_from == "by hand"


def channels_of(prepared: Optional[Prepared]) -> int:
    if prepared is not None and prepared.fourpi is not None:
        return 4
    return 1 if prepared is None or prepared.two is None else 2


def on_time_range(est, channels: int = 1) -> tuple:
    """The (shortest, longest) mean on-time an optimised model trains over: `ON_TIME_RANGE`, widened to
    take in the prefit's -- its frames seen, as DECODE's lifetime (`decode_smlm.prefit.lifetime`)."""
    from decode_smlm.prefit import lifetime
    lo, hi = ON_TIME_RANGE[channels]
    measured = float(lifetime(est.on_time_frames)) if est is not None and np.isfinite(est.on_time_frames) else lo
    return min(lo, measured), max(hi, measured)


def estimate_settings(est, channels: int = 1) -> EstimateSettings:
    default = EstimateSettings()
    lo, hi = on_time_range(est, channels)
    drawn = est.photons_range or (default.photons_min, default.photons_max)
    return EstimateSettings(photons=float(est.intensity_mu_sig[0]), photons_sd=float(est.intensity_mu_sig[1]),
                            photons_min=float(drawn[0]), photons_max=float(drawn[1]),
                            background_min=float(est.background[0]), background_max=float(est.background[1]),
                            z_min_nm=float(est.z_nm[0]), z_max_nm=float(est.z_nm[1]),
                            on_time_frames=float(est.on_time_frames), on_time_min=lo, on_time_max=hi)


def with_settings(est, s: EstimateSettings):
    """The estimates as edited in the form."""
    from decode_smlm.prefit import lifetime
    return replace(est, intensity_mu_sig=(s.photons, s.photons_sd), photons_range=(s.photons_min, s.photons_max),
                   background=(s.background_min, s.background_max),
                   z_nm=(s.z_min_nm, s.z_max_nm), on_time_frames=s.on_time_frames, lifetime_frames=lifetime(s.on_time_frames))


def camera_section(camera) -> dict:
    """DECODE's ``Camera`` section from the movie's camera."""
    from decode_smlm.prefit import camera as section
    return section(camera.conversion, camera.offset, camera.pixelsize_nm_or_default[0],
                   em_gain=camera.emgain if camera.is_em else None, read_noise_e=camera.read_noise_e)


def method_params(prepared: Prepared, est) -> dict:
    """DECODE's parameter file of the job: what the figure draws its ranges from, and the job's own for
    a DECODE model (`netfit_params` for an optimised one: the same ranges)."""
    from decode_smlm.prefit import method_params as params
    return params(est, camera_section(prepared.camera), training_density(prepared.settings, prepared.camera))


def frames_read(settings: PrepareSettings, channels: int) -> int:
    """*frames read* as a number: auto `FRAMES_AUTO`'s for the channels."""
    return FRAMES_AUTO[channels] if settings.frames_read == "auto" else int(settings.frames_read)


def netfit_params(prepared: Prepared, est, seed: int = 0, on_time: Optional[tuple] = None) -> dict:
    """An optimised model's ``settings.json`` (`smappy.netfit.job.settings_from_estimates`): the
    estimates' ranges, the movie's camera, netfit's defaults for the rest; for a two-channel
    calibration its channels as the prefit found them and the form chose (`two_channel_settings`).
    ``on_time``: the (shortest, longest) mean on-time drawn, the form's; None: `on_time_range`."""
    from ..netfit.job import settings_from_estimates, two_channel_settings
    if prepared.fourpi is not None:
        return fourpi_params(prepared, est, seed, on_time)
    s = prepared.settings
    channels, fuse = None, "early"
    if prepared.two is not None:
        t = s.two_channels
        channels, fuse = two_channel_settings(prepared.two, main=t.main, dyes=t.dyes, typed=typed_shares(t.typed),
                                              spread=t.spread, fuse="late" if s.model == "plexlike" else "early")
    return settings_from_estimates(est, camera_section(prepared.camera), training_density(s, prepared.camera), seed=seed,
                                   channels=channels, fuse=fuse,
                                   uneven=uneven_on(s.uneven_background, prepared.two is not None),
                                   window=frames_read(s, channels_of(prepared)),
                                   lifetime=on_time or on_time_range(est, channels_of(prepared)),
                                   max_samples=float(s.samples_m) * 1e6).to_dict()


def method_of(settings: PrepareSettings) -> str:
    """SMAPpy's network for *DECODE, optimised* and *DECODE-Plex-like* (its channels joined late), DECODE's port
    for *DECODE*."""
    return METHOD if settings.model == "decode" else NETFIT


def job_method(folder) -> Optional[str]:
    """The method a job folder trains (``job.json``'s), or None where it does not read."""
    import json
    try:
        return json.loads((Path(folder) / "job.json").read_text(encoding="utf-8")).get("method")
    except (OSError, ValueError):
        return None


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


def write_job(prepared: Prepared, est, folder, params: Optional[dict] = None, seed: int = 0,
              on_time: Optional[tuple] = None) -> Path:
    """The job folder (`training.job.write`), listed (`training.jobs.add`)."""
    from decode_smlm.prefit import simulation

    from ..io.calibration import load_spline_calibration
    from ..training import job as jobfile
    from ..training import jobs
    s = prepared.settings
    method = method_of(s)
    if method == NETFIT:
        params = params or netfit_params(prepared, est, seed, on_time)
    else:
        params = params or method_params(prepared, est)
    cam = prepared.camera
    job = jobfile.Job.new(method=method, seed=seed, device="auto", ranges=simulation(est, training_density(s, prepared.camera)),
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
    if prepared.fourpi is not None:
        # the 4Pi model with the movie's channel layout and maps written in: the training draws the geometry the
        # data have (`netfit.fourpi.FourChannels.from_calibration`), whatever the beads' scans had
        cal = fourpi_job_calibration(s.calibration, prepared.fourpi["layout"], Path(folder))
        job.data["fourpi_layout"] = prepared.fourpi["layout"]
    elif prepared.two is not None:              # a two-channel calibration, copied as it is
        cal = s.calibration
    else:
        cal = prepared.calibration if prepared.calibration is not None else load_spline_calibration(s.calibration)
    out = jobfile.write(folder, job, params, cal)
    if s.start_from:
        shutil.copyfile(s.start_from, out / BASE_MODEL)
    jobs.add(out)
    return out


BASE_MODEL = "base_model.safetensors"


def fourpi_job_calibration(path, layout: dict, folder: Path) -> Path:
    """The 4Pi model at ``path`` with ``layout`` (the movie's, with its maps) as its own, written as the job's
    calibration in ``folder``."""
    from ..channels import ChannelLayout
    from ..fourpi import load_fourpi_calibration, save_fourpi_calibration
    from ..training import job as jobfile
    cal = load_fourpi_calibration(path)
    lay = ChannelLayout.from_dict(layout)
    cal.layout = lay.to_dict()
    cal.transforms = None if lay.maps is None else np.stack([np.asarray(m, float) for m in lay.maps])
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / jobfile.CALIBRATION
    save_fourpi_calibration(target, cal, overwrite=True)
    return target


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
    if "simulation" in params:              # an optimised model's job: its own network, the same frames
        two = bool((params.get("channels") or {}).get("kind"))
        if meta.get("method") != NETFIT or bool(meta.get("channels")) != two:
            what = "two-channel" if two else "one-channel"
            return f"{Path(path).name} is not an optimised {what} DECODE model: start from scratch, or from one"
        if two and (meta.get("settings") or {}).get("channels", {}).get("kind") != params["channels"]["kind"]:
            return f"{Path(path).name} was trained for other channels: start from scratch, or from a model of these"
        theirs, ours = meta.get("settings") or {}, params
        for part in ("backbone", "time_model", "output"):
            if (theirs.get("parts") or {}).get(part) != ours["parts"].get(part):
                return (f"{Path(path).name} has another network ({part}): start from scratch, or from a "
                        f"model of the same network")
        if int(meta.get("window", 0)) != int(ours["simulation"]["window"]):
            return f"{Path(path).name} reads {meta.get('window')} frames, the job {ours['simulation']['window']}"
        return ""
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


def stop_rule(train: TrainSettings) -> dict:
    """The Train tab's numbers as the rule's (`training.progress.EARLY_STOP`), handed to the training."""
    return {"window": int(train.stop_window), "min_epochs": int(train.stop_min_epochs),
            "loss_tol": float(train.stop_loss_tol) / 100.0, "jaccard_tol": float(train.stop_jaccard_tol),
            "eff_tol": float(train.stop_jaccard_tol)}


def start_here(folder, train: TrainSettings, python: str = sys.executable) -> subprocess.Popen:
    """``python -m decode_smlm.cli train FOLDER`` detached, its output in ``train.log``: the training
    outlives the window, and the window follows it through its files."""
    from ..handoff import own_session
    folder = Path(folder)
    command = [python, "-m", "decode_smlm.cli", "train", str(folder), "--epochs", str(train.epochs)]
    if train.device != "auto":
        command += ["--device", train.device]
    if not train.joint_bounds:
        command += ["--no-joint-bounds"]
    if not train.until_converged:
        command += ["--all-epochs"]
    else:
        import json
        command += ["--stop-rule", json.dumps(stop_rule(train))]
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


#: the Follow figure's bottom-right-of-centre panel: "crlb" (each localization's error over its
#: molecule's CRLB) or "efficiency" (the SMLM Challenge's score of Jaccard and RMSE together)
SCORE_PANEL = "crlb"


def job_rule(lines: list) -> Optional[dict]:
    """The rule the job's latest start stops by (its ``stop_rule``), the default where a training
    from before said none, None where it trains every epoch."""
    from ..training.progress import rule
    start = next((line for line in reversed(lines) if line["kind"] == "start"), {})
    if "stop_rule" in start:
        return start["stop_rule"] or None
    return rule()


def _window(ax, e, values, window) -> None:
    """The light extras: the last ``window`` epochs shaded, a dashed line at their median (of the
    11-epoch running median the rule reads) -- where the curve stands now."""
    from ..training.progress import _running_median
    if window is None or len(e) <= window:
        return
    v = np.asarray(_running_median(list(values), 11), float)[-window:]
    if not np.isfinite(v).any():
        return
    if not getattr(ax, "_window_shaded", False):           # once a panel, whatever its curves
        ax.axvspan(e[-window], e[-1], color="0.5", alpha=0.12, lw=0)
        ax._window_shaded = True
    line = ax.lines[-1] if ax.lines else None
    ax.axhline(float(np.nanmedian(v)), ls="--", lw=0.8, color=line.get_color() if line is not None else "k")


#: the bounds the error is divided by, as the Follow panels name and colour them
#: (`decode_smlm.training.test_crlbs`): the molecule alone, with its frame's neighbours, and with
#: the frames the network reads around it too
BOUNDS = (("single", "alone", "C0"), ("joint", "+ neighbours", "C2"), ("context", "+ neighbours, frames", "C3"))


def _split_panel(ax, split: Optional[dict], key: str, axis: str, title: str, xlabel: str) -> None:
    """One of the bottom row's panels: the latest epoch's error / CRLB width per bin of ``key``
    against each bound; a note where the training reported none."""
    if not split or not split.get(key):
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"{title}:\nnot reported by this training", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return
    if key == "edges_nm":                   # the bins side by side: the closest ones are the few that matter
        edges = np.asarray(split[key], float)
        x = np.arange(len(edges), dtype=float)
        labels = [f"{int(a)}-{int(b)}" for a, b in zip(edges[:-1], edges[1:])] + [f">{int(edges[-1])}"]
    else:
        x = np.asarray(split[key], float)
        labels = [str(int(v)) for v in x]
    for name, label, colour in BOUNDS:
        values = split.get(f"{name}_{axis}")
        if values is None:
            continue
        y = np.array([np.nan if v is None else v for v in values], float)
        ax.plot(x, y, "o-", color=colour, label=label, ms=3)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xticks(x, labels, fontsize=6, rotation=30 if key == "edges_nm" else 0)
    n = split.get("n") or []
    ax.set(title=title, xlabel=xlabel)
    if n:
        ax.text(0.99, 0.97, "n = " + " ".join(str(int(v)) for v in n), ha="right", va="top",
                transform=ax.transAxes, fontsize=6, color="0.4")
    ax.legend(fontsize=6, loc="best")


def draw_progress(figure, lines: list) -> str:
    """The curves of a job's ``progress.jsonl``, in three rows of three: the loss, the Jaccard and
    where in their pixel this epoch's localizations fall; the RMSE, each localization's error over
    its molecule's CRLB (or the efficiencies, `SCORE_PANEL`), and the seconds an epoch; and the
    latest epoch's error over CRLB against each of the three bounds (`BOUNDS`), by the distance to
    the nearest other molecule laterally and axially, and by the frames read the molecule is on in.
    Each y range
    without the first epochs (`_fit_range`); on the curves the stop rule's window shaded and a line
    at its median (`_window`).  Returns a line saying where it is and how long it has left at most.
    The median spot's CRLB is no longer drawn on the RMSE: it stretched its axis to where the curves
    never come, and the error over each molecule's own CRLB says the same per molecule."""
    epochs = [line for line in lines if line["kind"] == "epoch"]
    figure.clear()
    axes = figure.subplots(3, 3)
    if not epochs:
        for a in axes.flat:
            a.set_axis_off()
        axes[0, 0].text(0.5, 0.5, "no epoch yet", ha="center", va="center", transform=axes[0, 0].transAxes)
        return "no epoch yet"
    e = np.array([line["epoch"] for line in epochs])
    rule_now = job_rule(lines)
    window = int(rule_now["window"]) if rule_now else None

    def series(key, metric=False):
        return np.array([np.nan if (line.get("metrics", {}) if metric else line).get(key) is None
                         else (line["metrics"] if metric else line)[key] for line in epochs], float)
    train, test = series("loss_train"), series("loss_test")
    axes[0, 0].plot(e, train, label="train")
    axes[0, 0].plot(e, test, label="test")
    _window(axes[0, 0], e, test, window)
    axes[0, 0].set(yscale="log", title="loss", xlabel="epoch")
    _fit_range(axes[0, 0], train, test, log=True)
    axes[0, 0].legend(fontsize=7)
    jac = series("jaccard", True)
    axes[0, 1].plot(e, jac)
    _window(axes[0, 1], e, jac, window)
    axes[0, 1].set(title="Jaccard (test set)", xlabel="epoch")
    # how often it fails: matched molecules beyond 5 times their bound (an optimised model's, plex-plan §16)
    out_share = series("outliers", True)
    if np.isfinite(out_share).any():
        twin = axes[0, 1].twinx()
        twin.plot(e, 100 * out_share, color="C3", lw=0.9, label="outliers")
        twin.set_ylabel("outliers (%)", fontsize=7, color="C3")
        twin.tick_params(labelsize=7, colors="C3")
        found_from = np.nan_to_num(jac) >= 0.25 * np.nanmax(np.r_[jac, 0.0])
        _fit_range(twin, np.where(found_from, 100 * out_share, np.nan))
    # pixelation: where in their pixel this epoch's localizations fall, uniform where unbiased
    last = (epochs[-1].get("metrics") or {})
    rows, cols = last.get("pixel_hist_rows"), last.get("pixel_hist_cols")
    if rows and cols:
        c = (np.arange(len(rows)) + 0.5) / len(rows)
        for h, name in ((cols, "x"), (rows, "y")):
            h = np.asarray(h, float)
            axes[0, 2].step(c, h / max(h.mean(), 1e-9), where="mid", label=name)
        axes[0, 2].axhline(1.0, color="k", lw=0.8, ls=":")
        axes[0, 2].set(title=f"pixelation, epoch {int(e[-1])}", xlabel="place in the pixel (0: centre, 0.5: edge)",
                       ylabel="relative count", ylim=(0.6, 1.4))
        axes[0, 2].legend(fontsize=7)
    else:
        axes[0, 2].set_axis_off()
        axes[0, 2].text(0.5, 0.5, "pixelation: not reported\nby this training", ha="center", va="center",
                        transform=axes[0, 2].transAxes, fontsize=8)
    # the RMSEs and the error over CRLB before the network finds much are over a handful of
    # matches, hundreds of nm off: left out of their ranges, from where Jaccard reaches a quarter
    # of its best on
    found = np.flatnonzero(np.nan_to_num(jac) >= 0.25 * np.nanmax(np.r_[jac, 0.0])) if np.isfinite(jac).any() else []
    first = int(found[0]) if len(found) else 0

    def later(v):
        return np.where(np.arange(len(v)) >= first, v, np.nan)
    lat, ax_ = series("rmse_lat_nm", True), series("rmse_ax_nm", True)
    axes[1, 0].plot(e, lat, label="lateral")
    _window(axes[1, 0], e, lat, window)
    axes[1, 0].plot(e, ax_, label="axial")
    _window(axes[1, 0], e, ax_, window)
    axes[1, 0].set(title="RMSE (nm, test set)", xlabel="epoch")
    _fit_range(axes[1, 0], later(lat), later(ax_))
    axes[1, 0].legend(fontsize=7)
    if SCORE_PANEL == "crlb":
        # each localization's error over its molecule's single-emitter CRLB, as the width of a
        # Gaussian (1.4826 MAD: no cut-off for the outliers); 1 where the network reaches the bound;
        # dashed against the bound with its neighbours and frames (`BOUNDS`), where reported
        wl, wa = series("crlb_width_lat", True), series("crlb_width_ax", True)
        cl, ca = series("crlb_width_lat_context", True), series("crlb_width_ax_context", True)
        if np.isfinite(wl).any() or np.isfinite(wa).any():
            axes[1, 1].plot(e, wl, label="lateral", color="C0")
            _window(axes[1, 1], e, wl, window)
            axes[1, 1].plot(e, wa, label="axial", color="C1")
            _window(axes[1, 1], e, wa, window)
            if np.isfinite(cl).any():
                axes[1, 1].plot(e, cl, "--", label="lateral, joint", color="C0", alpha=0.7)
                axes[1, 1].plot(e, ca, "--", label="axial, joint", color="C1", alpha=0.7)
            # the inliers' RMSE over the bound, a bias included (an optimised model's): how good the rest is
            il, ia = series("inlier_rmse_crlb_lat", True), series("inlier_rmse_crlb_ax", True)
            if np.isfinite(il).any():
                axes[1, 1].plot(e, il, ":", label="lateral, inliers' RMSE", color="C0")
                axes[1, 1].plot(e, ia, ":", label="axial, inliers' RMSE", color="C1")
                cl, ca = np.fmax(cl, il), np.fmax(ca, ia)
            axes[1, 1].axhline(1.0, color="k", lw=0.8, ls=":")
            axes[1, 1].set(title="error / CRLB, width (test set)", xlabel="epoch")
            _fit_range(axes[1, 1], later(wl), later(wa), later(cl), later(ca), include=(1.0,))
        else:
            axes[1, 1].set_axis_off()
            axes[1, 1].text(0.5, 0.5, "error / CRLB: not reported\nby this training", ha="center", va="center",
                            transform=axes[1, 1].transAxes, fontsize=8)
    else:
        # the SMLM Challenge's score of the Jaccard and the RMSE together: what the early stop reads
        # beside them, and a better sign of progress than either alone (the RMSE rises as more is found)
        eff_lat, eff_ax = series("eff_lat", True), series("eff_ax", True)
        axes[1, 1].plot(e, eff_lat, label="lateral")
        axes[1, 1].plot(e, eff_ax, label="axial")
        axes[1, 1].set(title="efficiency (test set)", xlabel="epoch")
        _fit_range(axes[1, 1], eff_lat, eff_ax)
    if axes[1, 1].lines:
        axes[1, 1].legend(fontsize=7)
    # the latest epoch's error / CRLB split: where crowding and the neighbouring frames matter
    neighbour, frames_on = last.get("crlb_by_neighbour"), last.get("crlb_by_frames_on")
    _split_panel(axes[2, 0], neighbour, "edges_nm", "lat", f"lateral by neighbour, epoch {int(e[-1])}",
                 "nearest other molecule (nm)")
    _split_panel(axes[2, 1], neighbour, "edges_nm", "ax", f"axial by neighbour, epoch {int(e[-1])}",
                 "nearest other molecule (nm)")
    _split_panel(axes[2, 2], frames_on, "frames", "lat", f"lateral by frames on, epoch {int(e[-1])}",
                 "frames read the molecule is on in")
    for a in axes[2]:
        if a.lines:
            a.set_ylabel("error / CRLB", fontsize=7)
    seconds = series("seconds")
    axes[1, 2].plot(e, seconds)
    axes[1, 2].set(title="seconds an epoch", xlabel="epoch")
    _fit_range(axes[1, 2], seconds)
    for a in axes.flat:
        a.grid(alpha=0.3)
        a.title.set_fontsize(9)
    start = next((line for line in reversed(lines) if line["kind"] == "start"), {})
    total = start.get("epochs")
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


def rule_line(lines: list) -> str:
    """What the stop rule waits for, under the Follow curves (`training.progress.status_text`)."""
    from ..training.progress import converged, status_text, stop_status
    end = next((line for line in reversed(lines) if line["kind"] in ("finished", "stopped", "error")), None)
    if end is not None and end.get("converged"):
        return f"converged at epoch {end['epoch']}: no better over the last epochs"
    rule_now = job_rule(lines)
    if rule_now is None:
        return "trains every epoch (stop when it no longer improves is off)"
    status = stop_status(lines, rule_now)
    if not status:
        return f"the stop rule judges from epoch {max(int(rule_now['min_epochs']), int(rule_now['window']) + 1)}"
    text = status_text(status)
    return ("would stop now -- " + text) if converged(lines, rule_now) else text


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
        self.prepare_form.field_changed.connect(self._prepare_field)
        # what kind of PSF the calibration is, in a line under its field (`psf_kind`)
        self.psf_note = QLabel()
        self.psf_note.setStyleSheet(note_style())
        self.psf_note.setWordWrap(True)
        # the row's width: the form grows only the fields that ask to (`ExpandingFieldsGrow`)
        from PySide6.QtWidgets import QSizePolicy
        self.psf_note.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.psf_note.setVisible(False)
        rows = self.prepare_form.row_layouts["calibration"]
        at, _ = rows.getWidgetPosition(self.prepare_form.row_widgets["calibration"])
        rows.insertRow(at + 1, "", self.psf_note)
        self._model_choices = None
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
        self.yaml_button.setToolTip("DECODE's parameters as the job will have them, to edit by hand "
                                    "(model DECODE, under the form's more)")
        self._model_changed()
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
        # what the stop rule waits for, and with which numbers (`rule_line`)
        self.rule_label = QLabel("")
        self.rule_label.setWordWrap(True)
        self.rule_label.setStyleSheet(note_style())
        fv.addWidget(self.rule_label)

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
            # an optimised model trains to its samples (*samples* when it was prepared), not to epochs
            self.train_form.set_shown({"epochs": entry is None or job_method(entry.folder) != NETFIT})
            if entry is None:
                row = self.job_list.currentRow()
                key = None if row < 0 or self.job_list.item(row, 0) is None else self.job_list.item(row, 0).data(Qt.UserRole)
                self.follow_label.setText("on a training host, not on this list: double-click it to add it"
                                          if key and key.startswith(FOUND) else "no job chosen")
                return
            from ..training import jobs, progress
            status = jobs.status(entry)
            lines = progress.read(entry.folder)
            text = draw_progress(self.follow_figure, lines)
            self.follow_canvas.draw_idle()
            # the state without its epoch: the curves' line says the epoch, once
            self.follow_label.setText(f"{entry.name}: {state_word(status)}; {text}"
                                      + (f" -- {status.message}" if status.message else ""))
            self.rule_label.setText(rule_line(lines))
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

    def _prepare_field(self, name: str) -> None:
        """The sample's index left *typed*: its number dropped; a number typed over: *typed*
        (`calibrate.medium.cleared`).  *model*: method.yaml is DECODE's, for its model only."""
        if name == "model":
            self._model_changed()
            return
        if name not in ("sample.medium", "sample.sample_index"):
            return
        from ..calibrate.medium import cleared
        try:
            sample = self.prepare_form.value().sample
        except ValueError:
            return
        drop = cleared(name, sample.medium, sample.sample_index, "sample.medium", "sample.sample_index")
        if drop:
            self.prepare_form.set_values(drop)

    def _model_changed(self) -> None:
        try:
            decode = self.prepare_form.value().model == "decode"
        except ValueError:
            return
        if not decode:
            self.yaml_button.setChecked(False)
        self.yaml_button.setEnabled(decode)
        # DECODE draws the photons normal (mean, spread), SMAPpy's networks uniformly (a range); DECODE
        # one on-time, SMAPpy's networks each sample's over a range
        self.estimate_form.set_shown({"photons": decode, "photons_sd": decode,
                                      "photons_min": not decode, "photons_max": not decode,
                                      "on_time_frames": decode, "on_time_min": not decode,
                                      "on_time_max": not decode})

    def _prepare_changed(self) -> None:
        """*in the sample* only for a pupil (Zernike) calibration (`calibrate.depth.sample_state`), its rows as
        *focus* reads them.  On a signal, so it may raise; it is said, never thrown."""
        from ..calibrate.depth import sample_state
        try:
            s = self.prepare_form.value()
            shown, active, hints = sample_state(self._calibration(s.calibration), s.sample)
            shown["sample"] = shown.pop("sample.focus")
            from ..netfit.prefit_two import is_dual
            dual = bool(s.calibration) and is_dual(s.calibration)
            shown["two_channels"] = dual
            shown["fourpi"] = psf_kind(s.calibration)[0] == 4
            shown["density_manual"] = s.density == "manual"
            shown["uneven_background"] = s.model != "decode"
            shown["frames_read"] = shown["samples_m"] = s.model != "decode"
            channels, line = psf_kind(s.calibration, self._calibration(s.calibration) if not dual else None)
            self.psf_note.setText(line)
            self.psf_note.setVisible(bool(line))
            # the models this calibration trains; the choice moves to the first where it is not one
            choices = model_choices(channels)
            if choices != self._model_choices:
                self._model_choices = choices
                # the value first (the default is in both lists): a list without the value on show
                # keeps it, marked not available
                if s.model not in dict(choices):
                    self.prepare_form.set_values({"model": choices[0][0]})
                self.prepare_form.set_choices({"model": list(choices)})
            shown.update(move_shown("two_channels.move", s.two_channels.move))
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
        self.estimate_form.set(estimate_settings(est, channels_of(prepared)))
        self.estimate_form.setEnabled(True)
        self.yaml_edit.setPlainText(yaml.safe_dump(method_params(prepared, est), sort_keys=False))
        self.write_button.setEnabled(True)
        self.simulated = None
        self.draw_prepared()
        self.simulate()
        from ..calibrate.depth import describe_sample
        where = describe_sample(prepared.calibration) if prepared.two is None and prepared.fourpi is None else ""
        two = prepared.two
        said = "" if two is None else (
            f"; {'two colours' if two.kind == 'colour' else 'two focal planes'}, the brighter half "
            f"{'the main' if two.brighter == 0 else 'the other'}"
            + (f", dyes at {', '.join(f'{v:.2f}' for v in two.dyes)}" if two.dyes else ""))
        if prepared.fourpi is not None:
            f = prepared.fourpi
            said = (f"; the cavity drifts {f['rate_measured']:+.1e} rad a frame (trained over "
                    f"±{f['cavity_rate_rad']:.0e})") if "rate_measured" in f else ""
            self.status_bar.showMessage(f"prefit: {est.n_localizations} Spline 4Pi localizations, "
                                        f"{est.emitters_per_frame:.1f} a frame" + said)
            return
        self.status_bar.showMessage(f"prefit: {est.n_blinks} blinks, {est.n_localizations} localizations; "
                                    f"photons from {est.intensity_from} ({est.n_fully_on} fully-on frames)"
                                    + (f"; {where}" if where else "") + said)

    def draw_prepared(self) -> None:
        from decode_smlm.figures import draw_frames, draw_ranges, photons
        self.prepare_figure.clear()
        est = with_settings(self.prepared.estimates, self.estimate_form.value())
        params = method_params(self.prepared, est)
        frames = self.simulated is not None and self.prepared.sample is not None
        axes = self.prepare_figure.subplots(2 if frames else 1, 3, squeeze=False)
        draw_ranges(axes[0], self.prepared.localizations, params)
        if frames and self.prepared.fourpi is not None:
            real = fourpi_side_by_side(self.prepared, photons(self.prepared.sample[:1], params["Camera"])[0])
            draw_frames(axes[1], real[None], self.simulated[:1], size=4 * PREVIEW_PX)
        elif frames and self.prepared.two is not None:
            # each channel's corner of the data, cut as the network reads it, beside the simulated pair
            real = channels_side_by_side(self.prepared, photons(self.prepared.sample[:1], params["Camera"])[0])
            draw_frames(axes[1], real[None], self.simulated[:1], size=2 * PREVIEW_PX)
        elif frames:
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
            if self.prepared.fourpi is not None:
                params = fourpi_params(self.prepared, est, on_time=self.on_time())
                layout = self.prepared.fourpi["layout"]
                calibration = self.prepared.settings.calibration
                self._work("simulated", lambda report: simulated_frames_4pi(params, calibration, layout))
                return
            if self.prepared.two is not None:
                # DECODE's simulation draws one channel: a two-channel job's frames drawn as netfit draws them
                params = netfit_params(self.prepared, est, on_time=self.on_time())
                calibration = self.prepared.settings.calibration
                self._work("simulated", lambda report: simulated_frames_two(params, calibration))
                return
            params = method_params(self.prepared, est)
            calibration = self.prepared.calibration or self.prepared.settings.calibration
            self._work("simulated", lambda report: simulated_frames(params, calibration))
        except Exception as error:
            print(f"DECODE training: {error}")

    def on_time(self) -> tuple:
        """The form's on-time range, for an optimised model."""
        e = self.estimate_form.value()
        return float(e.on_time_min), float(e.on_time_max)

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
        if method_of(self.prepared.settings) == NETFIT:
            params = None                       # the optimised model's settings (`write_job`); no method.yaml
        elif self.yaml_button.isChecked():
            try:
                params = yaml.safe_load(self.yaml_edit.toPlainText()) or params
            except yaml.YAMLError as error:
                self.status_bar.showMessage(f"method.yaml does not read: {error}")
                return
        s = self.prepared.settings
        try:
            folder = write_job(self.prepared, est, default_folder(s.movie, s.calibration), params,
                               on_time=self.on_time())
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

    def show_try(self, photons, locs, found_per_um2=float("nan"), trained_per_um2=float("nan")) -> None:
        self.try_figure.clear()
        ax = self.try_figure.subplots()
        ax.imshow(photons, cmap="gray", vmax=np.percentile(photons, 99.8) or None)
        if len(locs):
            ax.scatter(locs[:, 0], locs[:, 1], marker="+", s=30, c="gold")
        ax.set(title=f"first frame: {len(locs)} localizations at prob >= 0.5")
        self.try_canvas.draw_idle()
        warning = crowding_warning(found_per_um2, trained_per_um2)
        self.status_bar.showMessage(warning or f"{found_per_um2:.2f} molecules per µm² a frame in the first "
                                               f"{TRY_FRAMES} frames, trained at {trained_per_um2:.2f}")

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


PREVIEW_PX = 40                     # a two-channel preview's channels, side by side, this many pixels each


def simulated_frames_two(params: dict, calibration, n_frames: int = 2, seed: int = 0) -> np.ndarray:
    """(n, PREVIEW_PX, 2 PREVIEW_PX): a two-channel job's middle frames as its training draws them
    (`netfit.simulate.draw_batch_2c`), channel 0 (the grid's, the main half) left of channel 1, in
    photons (needs torch)."""
    import torch

    from ..netfit.train import Drawer, settings_from_dict
    d = {**params, "calibration": str(calibration),
         "simulation": {**params["simulation"], "frame_px": PREVIEW_PX, "movie_windows": 1}}
    s = settings_from_dict(d)
    b = Drawer(s, "cpu")(n_frames, torch.Generator().manual_seed(seed))
    f = b.frames[:, :, b.frames.shape[2] // 2].numpy()                  # (n, 2, s, s)
    return np.concatenate([f[:, 0], f[:, 1]], axis=-1)


def simulated_frames_4pi(params: dict, calibration, layout: dict, n_frames: int = 2, seed: int = 0) -> np.ndarray:
    """(n, PREVIEW_PX, 4 PREVIEW_PX): a 4Pi job's middle frames as its training draws them (`netfit.fourpi.draw_batch`,
    in the movie's layout), the four channels side by side on the network's grid, in photons (needs torch)."""
    import tempfile

    import torch

    from ..netfit.train import Drawer, settings_from_dict
    with tempfile.TemporaryDirectory() as tmp:
        cal = fourpi_job_calibration(calibration, layout, Path(tmp))
        d = {**params, "calibration": str(cal),
             "simulation": {**params["simulation"], "frame_px": PREVIEW_PX, "movie_windows": 1}}
        b = Drawer(settings_from_dict(d), "cpu")(n_frames, torch.Generator().manual_seed(seed))
    f = b.frames[:, :, b.frames.shape[2] // 2].numpy()                  # (n, 4, s, s)
    return np.concatenate(list(f.transpose(1, 0, 2, 3)), axis=-1)


def fourpi_side_by_side(prepared, frame_photons) -> np.ndarray:
    """A frame of the movie cut into the four channels on the network's grid (the layout the prefit took), each
    channel's corner of PREVIEW_PX, side by side: what `simulated_frames_4pi` draws, from the data."""
    from ..channels import ChannelLayout
    from ..netfit.fourpi import FourChannels
    four = FourChannels.from_layout(ChannelLayout.from_dict(prepared.fourpi["layout"]))
    cut = four.cut(np.asarray(frame_photons, float)[None])[0]
    return np.concatenate([c[:PREVIEW_PX, :PREVIEW_PX] for c in cut], axis=-1)


def channels_side_by_side(prepared, frame_photons) -> np.ndarray:
    """A frame of the movie cut into the two channels as the network reads them (the calibration's
    registration at the movie's camera ROI, the main half first), each channel's corner of PREVIEW_PX,
    side by side: what `simulated_frames_two` draws, from the data."""
    from ..calibrate.dual import load_dual_color_calibration
    from ..netfit.channels import TwoChannels
    cal = load_dual_color_calibration(prepared.settings.calibration)
    main = netfit_main(prepared)
    frame = np.asarray(frame_photons, float)
    two = TwoChannels.from_calibration(cal, prepared.camera.roi_offset, main=main, shape=frame.shape)
    cut = two.cut(frame[None])[0]                                        # (2, h, w)
    return np.concatenate([cut[0, :PREVIEW_PX, :PREVIEW_PX], cut[1, :PREVIEW_PX, :PREVIEW_PX]], axis=-1)


def netfit_main(prepared) -> int:
    """The calibration's half the network reads as its grid, as the job will have it."""
    from ..netfit.job import two_channel_settings
    t = prepared.settings.two_channels
    ch, _ = two_channel_settings(prepared.two, main=t.main, dyes="independent")
    return int(ch.main)


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
    remote.start(entry, train.epochs, train.device, all_epochs=not train.until_converged,
                 stop_rule=stop_rule(train) if train.until_converged else None, joint_bounds=train.joint_bounds)
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


def crowding_warning(found_per_um2: float, trained_per_um2: float) -> str:
    """A line when a model finds more molecules a frame than it was trained at; "" else.

    The prefit cannot say so: on DECODE-Plex's Fig. 2c, a live cell crowded in its first frames,
    Spline 3D separated 0.16 molecules per µm² where DECODE-Plex found 1.3, and a model trained at
    HD (1 per µm²) found half what one trained at 2 did (plex-plan §16).  A network trained too
    sparse still finds more than it was trained at, so this is the sign; training too dense costs
    less than training too sparse, so it only ever advises a higher density."""
    if not (np.isfinite(found_per_um2) and np.isfinite(trained_per_um2)) or found_per_um2 <= trained_per_um2:
        return ""
    better = "UHD" if found_per_um2 <= PER_UM2["UHD"] else f"at least {found_per_um2:.1f} typed"
    return (f"It finds {found_per_um2:.1f} molecules per µm² a frame, more than the {trained_per_um2:.1f} it "
            f"was trained at: a model trained at {better} will find more and place them better.")


def per_um2_of(per_40px: float, px_size_nm) -> float:
    """Molecules per µm² from a job's molecules in 40 x 40 pixels and its camera's pixel (DECODE's px_size)."""
    py, px = (float(v) for v in px_size_nm)
    return float(per_40px) / (40 * px / 1000) / (40 * py / 1000)


def try_on_data(folder, frames: int = TRY_FRAMES):
    """The job's newest model on the first frames of its movie: (first frame in photons, its
    localizations' (x, y) in pixels at prob >= 0.5, molecules per µm² a frame over the frames read,
    molecules per µm² it was trained at); a two-channel model's at channel 0's spots, over a half's area."""
    from ..metadata import CameraMetadata
    from ..plugins.fit_decode import _mirrored
    from ..training import job as jobfile
    from ..training import model_file
    from ..training.predictors import fourpi, load, two_channel
    from ..window_engine import WindowEngine
    job, _ = jobfile.read(folder)
    model = Path(folder) / "model.safetensors"
    meta = model_file.read_metadata(model)
    cam = job.camera
    camera = CameraMetadata(conversion=cam["e_per_adu"], offset=cam["baseline"], em_on=bool(cam.get("em_gain")),
                            emgain=cam.get("em_gain"), pixelsize_um=1.0)
    source = _source(job.data["path"])
    n = min(frames, source.n_frames)
    stack = np.concatenate([b for _, b in source.frames(chunk=n, start=0, stop=n)])
    area_share = 1.0
    if fourpi(meta):
        # cut into its channels by the movie's layout the job keeps, as the DECODE fitter cuts them
        from ..channels import ChannelLayout
        from ..fourpi import load_fourpi_calibration
        from ..netfit.fourpi import FourPiPredictor
        from ..netfit.predict import Predictor
        layout = ChannelLayout.from_dict(job.data["fourpi_layout"])
        cal = load_fourpi_calibration(Path(folder) / job.calibration.get("file", jobfile.CALIBRATION))
        predictor = FourPiPredictor(Predictor(model, None), layout, cal)
        w, h = layout.sizes[0]
        area_share = w * h / (stack.shape[-2] * stack.shape[-1])
    elif two_channel(meta):
        # cut into its channels by the job's calibration at the movie's camera ROI, as the fitter cuts them
        # (`fit_decode.two_channel_model`): positions on the frame, channel 0's
        from ..calibrate.dual import load_dual_color_calibration
        from ..netfit.fit_two import TwoChannelPredictor
        from ..netfit.predict import Predictor
        cal = load_dual_color_calibration(Path(folder) / job.calibration.get("file", jobfile.CALIBRATION))
        roi = job.data.get("camera_roi")
        predictor = TwoChannelPredictor(Predictor(model, None), cal, (roi[0], roi[1]) if roi else (0, 0),
                                        em=bool(cam.get("em_gain")))
    else:
        predictor = load(model, None, frames_mirrored=_mirrored(meta, camera))
    engine = WindowEngine(camera, predictor)
    parts = [engine.push(stack, 0), engine.end()]
    xs, ys = [], []
    found = 0
    for p in parts:
        if p is None:
            continue
        found += int((np.asarray(p["prob"]) >= 0.5).sum())
        sel = (np.asarray(p["frame"]) == 0) & (np.asarray(p["prob"]) >= 0.5)
        xs.append(np.asarray(p["x_nm"])[sel] / 1000.0)
        ys.append(np.asarray(p["y_nm"])[sel] / 1000.0)
    from ..camera import to_photons
    locs = np.stack([np.concatenate(xs), np.concatenate(ys)], 1) if xs else np.zeros((0, 2))
    py, px = (float(v) for v in cam["px_size"])
    area = stack.shape[-2] * stack.shape[-1] * (0.5 if two_channel(meta) else area_share) * px * py * 1e-6
    trained = job.ranges.get("emitter_av")
    return (to_photons(stack[0], camera), locs, found / n / area,
            float("nan") if trained is None else per_um2_of(trained, cam["px_size"]))
