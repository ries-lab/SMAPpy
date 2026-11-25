"""A netfit training as a job folder: the training window's and the training hosts' way to train.

A job folder (`smappy.training.job`) holds ``job.json``, the calibration and the method's own
settings; for netfit (method ``netfit-v1``) those are ``settings.json``, a `TrainingSettings` whose
calibration is the folder's ``calibration.h5``, so the folder moves.  `train_folder` trains it as
``smappy-decode train FOLDER`` trains a DECODE job (`decode_smlm.cli` hands it here by the method):

* **a lock** (``lock.json``: host, pid, started), so two trainings never share a folder, and a
  window can tell a training that is running from one that was interrupted;
* **a stop** at the end of an epoch, by a ``STOP`` file (*Stop*, ``smappy-decode stop``) or SIGTERM:
  ``stopped`` in ``progress.jsonl``; run again, it resumes from ``checkpoint.pt``;
* **a base model** (``base_model`` in the job): its weights to start from (`train.warm_start`).

`settings_from_estimates` turns the prefit's estimates into the simulation's settings, as
`decode_smlm.prefit.method_params` turns them into DECODE's: the same background range, z range,
on-time and density, the movie's camera; the photons a frame drawn uniformly over the prefit's
range where it has one (DECODE draws them normal, from the mean and spread), and a share of the
movies with an uneven background (DECODE's are flat).
"""
from __future__ import annotations

import json
import os
import signal
import socket
import threading
import time
from pathlib import Path

from .settings import CameraModel, SimulationSettings, TrainingSettings

SETTINGS, STOP, LOCK = "settings.json", "STOP", "lock.json"
GMN7C_WINDOW = 7                                          # one channel's frames read (poc-plan, step 4)
DECAY_SAMPLES = 327680                                    # the studies' schedule: 10 epochs of 32 768
TARGET_PHOTONS_MIN = 100.0                                # the studies' targets
#: the share of the training's movies drawn with an uneven background, *uneven backgrounds* on: as the
#: C = 2 study's B models, which over three seeds lost nothing to flat ones and pixelated half as much
#: (plex-plan §16, 2026-11-23), and on Fig. 2c's live cell matched DECODE-Plex 0.62 against flat's 0.41
UNEVEN_SHARE = 0.5


class JobError(Exception):
    """A folder that cannot be trained, said in a sentence."""


#: how a two-colour training draws each molecule's share in channel 1 (Jonas, 2026-11-21): free -- each
#: channel its own count, any dyes --, or around the dyes, from the prefit or typed, by a spread
DYES = (("independent", "independent counts"), ("prefit", "from the prefit"), ("typed", "typed"))
BROAD_SD, NARROW_SD = 0.06, 0.03         # plex-plan §16: the broad prior, and the narrow one


def two_channel_settings(two, *, main="auto", dyes="independent", typed=(), spread: float = NARROW_SD,
                         fuse: str = "early"):
    """The training's `ChannelSettings` from the two-channel prefit (`prefit_two.TwoChannels`), in the
    grid's order: ``main`` the calibration's channel the network reads as its grid ("auto": the
    brighter, Jonas 2026-11-21: the bright one main), each channel's background, and for colour the dyes
    as shares in the grid's channel 1 (the calibration's, turned when its main half is not the grid's)."""
    from .settings import ChannelSettings
    m = two.brighter if main == "auto" else int(main)
    shares = list(two.dyes if dyes == "prefit" else typed if dyes == "typed" else [])
    if dyes in ("prefit", "typed") and not shares:
        raise ValueError("no dyes to train around: the prefit found none, or none were typed; "
                         "choose independent counts")
    grid = [round(1.0 - v, 4) if m == 1 else float(v) for v in shares]
    bg = [list(two.background[m]), list(two.background[1 - m])]
    out = ChannelSettings(kind=two.kind, background=bg, main=m)
    if two.kind == "colour" and grid:
        out.dyes, out.share_sd = sorted(grid), float(spread)
    return out, fuse


def settings_from_estimates(est, camera_section: dict, density, calibration: str = "calibration.h5",
                            window: int | None = None, seed: int = 0, channels=None, fuse: str = "early",
                            uneven: bool = True, lifetime: tuple | None = None,
                            max_samples: float | None = None) -> TrainingSettings:
    """A training's settings from the prefit's estimates (`decode_smlm.prefit.Estimates`) and the
    movie's camera (`decode_smlm.prefit.camera`'s section); everything else netfit's defaults.  With
    ``channels`` (`two_channel_settings`) a two-channel training: the calibration a two-channel one, the
    channels read together from the first layer (``fuse`` "early", *transformed channels*) or apart and
    joined later ("late", *DECODE-Plex-like*).  ``uneven``: `UNEVEN_SHARE` of the movies with an uneven
    background (*uneven backgrounds*), else all flat.  ``window``: the frames the network reads (None:
    `GMN7C_WINDOW` for one channel, 3 for two, the studies'); ``lifetime``: the (shortest, longest) mean
    on-time each sample is drawn with (None: one channel 1 -- 3, two channels 1); ``max_samples``: the
    samples it trains on at most (None: netfit's 6 M)."""
    from decode_smlm.prefit import density_per_frame
    per_40px = density_per_frame(density)          # a name (LD, HD, UHD) or emitters in 40 x 40 typed
    s = TrainingSettings()
    mean, sd = (float(v) for v in est.intensity_mu_sig)
    py, px = camera_section["px_size"]                  # DECODE's order: rows, then columns
    s.simulation = SimulationSettings(**{**s.simulation.__dict__, "window": int(window or 3), "photons_mean": mean,
                                         "photons_sd": sd, "background": tuple(float(v) for v in est.background),
                                         "z_nm": tuple(float(v) for v in est.z_nm),
                                         "lifetime_frames": float(est.lifetime_frames),
                                         "density_per_40px": per_40px,
                                         "photons_range": (None if getattr(est, "photons_range", None) is None
                                                           else tuple(float(v) for v in est.photons_range)),
                                         "background_uneven": UNEVEN_SHARE if uneven else 0.0})
    s.camera = CameraModel(conversion=float(camera_section["e_per_adu"]), offset=float(camera_section["baseline"]),
                           em_gain=float(camera_section.get("em_gain") or 0.0),
                           read_noise_e=float(camera_section["read_sigma"]), qe=float(camera_section.get("qe", 1.0)),
                           pixel_nm=(float(px), float(py)))
    s.calibration = calibration
    s.seed = int(seed)
    # no snapshots in a job: the model file of the last epoch is the one kept, as DECODE's jobs keep it
    s.parts.schedule.settings["snapshots"] = []
    # as the studies' winners trained (GMN7C, the C = 2 L_ runs): the rate falling every 327 680 samples
    # whatever the batch, molecules from 100 photons a frame the targets
    s.parts.schedule.settings["decay_samples"] = DECAY_SAMPLES
    s.simulation = SimulationSettings(**{**s.simulation.__dict__, "target_photons_min": TARGET_PHOTONS_MIN})
    if channels is None:
        # a calibration that varies over the chip is drawn so, and the network told where it is
        # (`netfit.field`, docs/decode/fd-plan.md): "auto" turns it on for one with node splines
        s.simulation = SimulationSettings(**{**s.simulation.__dict__, "field": "auto"})
    from .settings import Part
    if channels is None:
        # one channel: step 4's time model, GMN7C (docs/decode/poc-plan.md, "Step 4: the time model"):
        # a ConvGRU with its input normalised over 7 frames, the mean on-time drawn per sample over 1 -- 3
        # frames (one network for brief and long blinks, within 2 % of either specialist), streamed.
        # `settings.Parts` keeps version 1's frame stack, so a model saved before is built as it was.
        # Not yet here: its fine-tune to the on-time the prefit measured (1 -- 2 M samples at lr 1e-4)
        frames = int(window or GMN7C_WINDOW)
        s.parts.time_model = Part("convgru", {"frames": frames, "norm": True})
        s.simulation = SimulationSettings(**{**s.simulation.__dict__, "window": frames,
                                             "lifetime_frames": 1.0, "lifetime_max_frames": 3.0,
                                             "movie_windows": 8})
    if channels is not None:
        # the C = 2 study's settings, the ones its comparisons were made with (plex-plan §16;
        # `benchmarks/decode/netfit_c2/study/L_*.json`): 64 features, the ConvGRU joining the frames,
        # spots over the spline's whole grid, targets from 100 photons.  Streamed, as one channel's
        # (windows cut from one movie: half the drawing); at C = 2 one seed of it was a little worse than
        # unstreamed (plex-plan §16, "The C runs so far"), being tested with more seeds before deciding
        s.channels = channels
        s.parts.backbone.settings.update(two_fuse=fuse, features=64)
        s.parts.time_model = Part("convgru", {"frames": int(window or 3), "norm": True})
        # on-time 1: at C = 2 it tied mixed on-times 1 -- 3 on molecules on for 3 frames and beat them on
        # the brief dim ones (plex-plan §16, "The time model"); the prefit's on-time is not used
        s.simulation = SimulationSettings(**{**s.simulation.__dict__, "movie_windows": 8, "spots": "whole",
                                             "lifetime_frames": 1.0, "lifetime_max_frames": 0.0})
    if lifetime is not None:
        # the window's *on-time* range: one on-time where both ends are the same
        lo, hi = sorted(float(v) for v in lifetime)
        s.simulation = SimulationSettings(**{**s.simulation.__dict__, "lifetime_frames": lo,
                                             "lifetime_max_frames": hi if hi > lo else 0.0})
    if max_samples:
        s.parts.schedule.settings["max_samples"] = float(max_samples)
    return s


def read_settings(folder) -> TrainingSettings:
    """The folder's ``settings.json``, its calibration resolved inside the folder."""
    from .train import load_settings
    folder = Path(folder)
    s = load_settings(folder / SETTINGS)
    if s.calibration and not Path(s.calibration).is_absolute():
        s.calibration = str((folder / s.calibration).resolve())
    return s


def train_folder(folder, *, device: str | None = None, epochs: int | None = None, all_epochs: bool = False,
                 stop_rule: dict | None = None, allow_other_version: bool = False, log=print) -> str:
    """Train the netfit job in ``folder`` from where it stands; ``"finished"`` or ``"stopped"``."""
    from .. import __version__
    from ..training import job as jobs
    from ..training import progress
    from .train import METHOD, train
    folder = Path(folder)
    problems = jobs.check(folder)
    if problems:
        raise JobError(f"{folder}: " + "; ".join(problems))
    job, _ = jobs.read(folder)
    if job.method != METHOD:
        raise JobError(f"{folder} is a {job.method} job; this trains {METHOD}")
    if job.smappy_version != __version__ and not allow_other_version:
        raise JobError(f"{folder} was prepared by SMAPpy {job.smappy_version}, this is {__version__}: "
                       f"prepare it again, or allow another version")
    settings = read_settings(folder)
    if job.base_model:
        settings.parts.schedule.settings["init"] = str(folder / job.base_model)
    import torch
    want = device or job.device
    device = want if want not in (None, "auto") else (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    rule = None if all_epochs else dict(stop_rule or {})     # over netfit's, scaled to its epochs (`train.stop_rule`)
    stop = {"why": None}

    def on_term(signum, frame):
        stop["why"] = "SIGTERM"
    main = threading.current_thread() is threading.main_thread()
    old = signal.signal(signal.SIGTERM, on_term) if main else None
    try:
        with _Lock(folder):
            return train(folder, settings, device, epochs, log=log, early_stop=rule is not None, rule=rule,
                         stop=lambda: _stop_asked(folder, stop))
    except Exception as e:
        progress.append(folder, "error", None, message=f"{type(e).__name__}: {e}")
        raise
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)


def _stop_asked(folder: Path, stop: dict):
    """Why to stop now, or None: a ``STOP`` file (taken away, so the next run trains) or a signal."""
    if (folder / STOP).exists():
        (folder / STOP).unlink(missing_ok=True)
        return "STOP file"
    return stop["why"]


class _Lock:
    """``lock.json`` while a training runs: refused where a live process here, or another host, holds it."""

    def __init__(self, folder: Path):
        self.path = folder / LOCK

    def __enter__(self):
        if self.path.exists():
            try:
                info = json.loads(self.path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                info = {}
            host, pid = info.get("host"), info.get("pid")
            if host == socket.gethostname() and pid and _alive(pid):
                raise JobError(f"{self.path.parent} is being trained already, by process {pid}")
            if host and host != socket.gethostname():
                raise JobError(f"{self.path.parent} is locked by a training on {host}; if that is gone, "
                               f"delete {self.path.name}")
        self.path.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                                         "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}), encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
