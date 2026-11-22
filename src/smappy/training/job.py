"""A training's job folder: everything a training needs, in one movable folder.

The layout is docs/decode/poc-plan.md §5's.  What is written here, before a training:

    job.json          schema 1: method, smappy_version, created, the data it was prepared on,
                      the calibration (its file here and the sha256 of its coefficients), the
                      camera, the ranges the training draws from, base_model, seed, device
    method.yaml       the method's own settings, paths relative to the folder: DECODE's parameter
    settings.json     file (``decode-v0.10.2``), or netfit's `TrainingSettings` (``netfit-v1``)
    calibration.h5    the calibration as SMAPpy loaded it (`save_spline_calibration`), so the far
                      end reads one format whatever the user had

The training adds its checkpoint, model file and ``progress.jsonl`` beside them (milestone (f)).

**Every path inside is relative to the folder**, so a folder copied to another machine, or moved
on this one, still trains; the data's path is kept only as a record of where the ranges came
from.  **Written, read and written again, the folder is byte-identical**: `write` writes JSON and
YAML in one fixed form, and copies an existing ``calibration.h5`` byte for byte rather than
writing it again -- HDF5 written twice is not.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = 1
JOB, METHOD, CALIBRATION = "job.json", "method.yaml", "calibration.h5"
NETFIT_SETTINGS = "settings.json"
NETFIT = "netfit-v1"


def params_file(method: str) -> str:
    """The file a method's own settings are kept in."""
    return NETFIT_SETTINGS if method == NETFIT else METHOD


@dataclass
class Job:
    method: str                                  # "decode-v0.10.2" or "netfit-v1"
    ranges: dict                                 # what the training draws: photons, background, z, density
    camera: dict                                 # the camera's values (and its database entry, if any)
    calibration: dict                            # {"file": "calibration.h5", "sha256": ..., "source": ...}
    data: dict = field(default_factory=dict)     # path, frames, camera ROI: where the ranges came from
    base_model: str | None = None                # a model file to start from, relative to the folder
    seed: int = 0
    device: str = "auto"
    smappy_version: str = ""
    created: str = ""
    schema: int = SCHEMA

    @classmethod
    def new(cls, **fields) -> Job:
        from .. import __version__
        fields.setdefault("smappy_version", __version__)
        fields.setdefault("created", datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
        return cls(**fields)


def two_channel(method: str, params: dict) -> bool:
    """Whether a job trains on a split camera's two channels: a netfit job whose channels have a kind."""
    return method == NETFIT and bool((params.get("channels") or {}).get("kind"))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def coefficients_sha256(cal) -> str:
    """The sha256 of a calibration's coefficients as the fitter uses them: (64, nz, ny, nx), float32."""
    return hashlib.sha256(np.ascontiguousarray(cal.coeff, dtype=np.float32).tobytes()).hexdigest()


def _json(job: Job) -> str:
    return json.dumps(asdict(job), indent=1, ensure_ascii=False) + "\n"


def _yaml(params: dict) -> str:
    import yaml
    return yaml.safe_dump(params, sort_keys=False, default_flow_style=None, allow_unicode=True)


def write(folder, job: Job, method_params: dict, calibration) -> Path:
    """Write a job folder.  ``calibration`` is a `SplineCalibration`, or the path of a
    ``calibration.h5`` to copy (a folder written again from one read).  Refuses a folder that
    holds a job already, unless it is the same job being written again over itself."""
    from ..io.calibration import load_spline_calibration, save_spline_calibration

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    netfit = job.method == NETFIT
    own = method_params.get("calibration") if netfit else method_params["InOut"]["calibration_file"]
    if own and Path(own).is_absolute():
        raise ValueError(f"{params_file(job.method)}'s calibration file must be relative to the job folder")
    target = folder / CALIBRATION
    if two_channel(job.method, method_params):
        # a two-channel calibration (netfit's C = 2): copied as it is, its file's sha256 the model's
        # check (`train._sha256`, `fit_decode.two_channel_calibration`)
        source = Path(calibration)
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        sha = file_sha256(target)
        if job.calibration.get("sha256") not in (None, sha):
            raise ValueError("the job's calibration is not the one being written beside it")
        job.calibration = {**job.calibration, "file": CALIBRATION, "sha256": sha}
        text = json.dumps(method_params, indent=1) + "\n"
        (folder / params_file(job.method)).write_text(text, encoding="utf-8")
        (folder / JOB).write_text(_json(job), encoding="utf-8")
        return folder
    if isinstance(calibration, (str, Path)):
        source = Path(calibration)
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        cal = load_spline_calibration(target)
    else:
        cal = calibration
        save_spline_calibration(target, cal, overwrite=True)
    sha = coefficients_sha256(cal)
    if job.calibration.get("sha256") not in (None, sha):
        raise ValueError("the job's calibration is not the one being written beside it")
    job.calibration = {**job.calibration, "file": CALIBRATION, "sha256": sha}
    text = json.dumps(method_params, indent=1) + "\n" if netfit else _yaml(method_params)
    (folder / params_file(job.method)).write_text(text, encoding="utf-8")
    (folder / JOB).write_text(_json(job), encoding="utf-8")
    return folder


def read(folder) -> tuple[Job, dict]:
    """The job and the method's parameters of a folder; the calibration is ``folder / job.calibration["file"]``."""
    import yaml
    folder = Path(folder)
    raw = json.loads((folder / JOB).read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{folder / JOB}: schema {raw.get('schema')!r}, this SMAPpy reads {SCHEMA}")
    known = {f for f in Job.__dataclass_fields__}
    job = Job(**{k: v for k, v in raw.items() if k in known})
    text = (folder / params_file(job.method)).read_text(encoding="utf-8")
    return job, json.loads(text) if job.method == NETFIT else yaml.safe_load(text)


def check(folder) -> list[str]:
    """What is wrong with a folder, as sentences; empty when it can be trained."""
    folder = Path(folder)
    if not (folder / JOB).exists():
        return [f"{JOB} is missing"]
    try:
        method = json.loads((folder / JOB).read_text(encoding="utf-8")).get("method", "")
    except ValueError as e:
        return [f"{JOB}: {e}"]
    problems = []
    if not (folder / params_file(method)).exists():
        problems.append(f"{params_file(method)} is missing")
    if problems:
        return problems
    try:
        job, params = read(folder)
    except ValueError as e:
        return [str(e)]
    cal_file = folder / job.calibration.get("file", CALIBRATION)
    if not cal_file.exists():
        return [f"the calibration {cal_file.name} is missing"]
    from ..io.calibration import load_spline_calibration
    sha = file_sha256(cal_file) if two_channel(job.method, params) else coefficients_sha256(load_spline_calibration(cal_file))
    if sha != job.calibration.get("sha256"):
        problems.append(f"{cal_file.name} is not the calibration the job was prepared with")
    if job.method == NETFIT:
        value = params.get("calibration")
        if value and Path(value).is_absolute():
            problems.append(f"{NETFIT_SETTINGS}'s calibration is an absolute path: the folder cannot move")
    for section, key in (() if job.method == NETFIT else (("InOut", "calibration_file"), ("InOut", "experiment_out"))):
        value = (params.get(section) or {}).get(key)
        if value is not None and Path(value).is_absolute():
            problems.append(f"method.yaml's {section}.{key} is an absolute path: the folder cannot move")
    if job.base_model and not (folder / job.base_model).exists():
        problems.append(f"the base model {job.base_model} is missing")
    return problems
