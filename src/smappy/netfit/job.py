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
`decode_smlm.prefit.method_params` turns them into DECODE's: the same photons a frame (mean and
spread), background range, z range, on-time and density, the movie's camera.
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
DENSITIES = {"LD": 15.0, "HD": 25.0, "UHD": 50.0}       # emitters in a 40 x 40 frame, as DECODE's jobs


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


def settings_from_estimates(est, camera_section: dict, density: str, calibration: str = "calibration.h5",
                            window: int = 3, seed: int = 0, channels=None, fuse: str = "early") -> TrainingSettings:
    """A training's settings from the prefit's estimates (`decode_smlm.prefit.Estimates`) and the
    movie's camera (`decode_smlm.prefit.camera`'s section); everything else netfit's defaults.  With
    ``channels`` (`two_channel_settings`) a two-channel training: the calibration a two-channel one, the
    channels read together from the first layer (``fuse`` "early", *aligned channels*) or apart and
    joined later ("late", *DECODE-Plex-like*)."""
    if density not in DENSITIES:
        raise ValueError(f"density {density!r}: one of {', '.join(DENSITIES)}")
    s = TrainingSettings()
    mean, sd = (float(v) for v in est.intensity_mu_sig)
    py, px = camera_section["px_size"]                  # DECODE's order: rows, then columns
    s.simulation = SimulationSettings(**{**s.simulation.__dict__, "window": int(window), "photons_mean": mean,
                                         "photons_sd": sd, "background": tuple(float(v) for v in est.background),
                                         "z_nm": tuple(float(v) for v in est.z_nm),
                                         "lifetime_frames": float(est.lifetime_frames),
                                         "density_per_40px": DENSITIES[density]})
    s.camera = CameraModel(conversion=float(camera_section["e_per_adu"]), offset=float(camera_section["baseline"]),
                           em_gain=float(camera_section.get("em_gain") or 0.0),
                           read_noise_e=float(camera_section["read_sigma"]), qe=float(camera_section.get("qe", 1.0)),
                           pixel_nm=(float(px), float(py)))
    s.calibration = calibration
    s.seed = int(seed)
    # no snapshots in a job: the model file of the last epoch is the one kept, as DECODE's jobs keep it
    s.parts.schedule.settings["snapshots"] = []
    if channels is not None:
        # the C = 2 study's settings, the ones its comparisons were made with (plex-plan §16;
        # `benchmarks/decode/netfit_c2/study/L_*.json`): 64 features, the ConvGRU joining the frames,
        # spots over the spline's whole grid, targets from 100 photons; streamed, windows cut from one
        # movie, which halves the drawing ("Cheaper checks")
        from .settings import Part
        s.channels = channels
        s.parts.backbone.settings.update(two_fuse=fuse, features=64)
        s.parts.time_model = Part("convgru", {"frames": int(window), "norm": True})
        s.simulation = SimulationSettings(**{**s.simulation.__dict__, "movie_windows": 8, "spots": "whole",
                                             "target_photons_min": 100.0})
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
