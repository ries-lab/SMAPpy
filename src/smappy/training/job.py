"""A training's job folder: everything a training needs, in one movable folder.

The layout is docs/decode/poc-plan.md §5's.  What is written here, before a training:

    job.json          schema 1: method, smappy_version, created, the data it was prepared on,
                      the calibration (its file here and the sha256 of its coefficients), the
                      camera, the ranges the training draws from, base_model, seed, device
    method.yaml       the method's own settings (DECODE's parameter file), paths relative to the
                      folder
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


@dataclass
class Job:
    method: str                                  # "decode-v0.10.2"
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
    if Path(method_params["InOut"]["calibration_file"]).is_absolute():
        raise ValueError("method.yaml's calibration file must be relative to the job folder")
    target = folder / CALIBRATION
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
    (folder / METHOD).write_text(_yaml(method_params), encoding="utf-8")
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
    return job, yaml.safe_load((folder / METHOD).read_text(encoding="utf-8"))


def check(folder) -> list[str]:
    """What is wrong with a folder, as sentences; empty when it can be trained."""
    folder = Path(folder)
    problems = []
    for name in (JOB, METHOD):
        if not (folder / name).exists():
            problems.append(f"{name} is missing")
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
    if coefficients_sha256(load_spline_calibration(cal_file)) != job.calibration.get("sha256"):
        problems.append(f"{cal_file.name} is not the calibration the job was prepared with")
    for section, key in (("InOut", "calibration_file"), ("InOut", "experiment_out")):
        value = (params.get(section) or {}).get(key)
        if value is not None and Path(value).is_absolute():
            problems.append(f"method.yaml's {section}.{key} is an absolute path: the folder cannot move")
    if job.base_model and not (folder / job.base_model).exists():
        problems.append(f"the base model {job.base_model} is missing")
    return problems
