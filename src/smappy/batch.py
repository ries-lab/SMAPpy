"""A chain over many files: each loaded, processed, saved and cleared alone.

`docs/batch.md` has the decisions; this is the runner and the job file.  Its
shape comes from what SMAP's `BatchAnalysis` got wrong as much as from what
it did: every file here runs in a fresh session with nothing carried over,
one failing file costs that file and the batch goes on, the numbers a step
returns reach a table (`summary.csv`) rather than a struct, and nothing needs
a window -- the GUI's batch window runs this very module as a subprocess and
reads the lines it prints.

A job is read tolerantly where that is harmless and strictly where it is not:
`validate` refuses every settings key the plugin does not have, because the
tolerant reading that lets a two-year-old file open (`settings_from`) would
turn a typo in a file somebody wrote this morning into a silent default.
"""
from __future__ import annotations

import csv
import dataclasses
import fnmatch
import hashlib
import json
import math
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import chain as chains
from .chain import (ChainAborted, ChainError, ChainSkipped, ChainSpec,
                    chain_class, is_source)
from .plugins import ParamSpec, settings_from, settings_values

SCHEMA = "smappy-batch-v1"
RESULT_SCHEMA = "smappy-batch-result-v1"
REPORT_SCHEMA = "smappy-batch-report-v1"
PREFLIGHT = ("skip", "proceed", "abort")
RERUN = ("skip_identical", "always")
FIGURES = ("png", "pdf", "svg", "none")
FITTED = ("beside_image", "output")
# what a folder rule looks for when it does not say
PATTERNS = {"images": ["*.tif", "*.tiff"],
            "localizations": ["*.h5", "*.hdf5", "*_sml.mat", "*.csv"]}
# the files a step's data is cut down to before it is written: enough for a
# curve or a histogram's counts, not a column of a million localizations
MAX_LIST = 1000


class BatchError(ValueError):
    """A job that cannot run: the message says what to fix."""


# ------------------------------------------------------------------ the job

@dataclass
class Input:
    """Files to process: one file, or a folder and the patterns to look for."""
    file: str = ""
    folder: str = ""
    pattern: List[str] = field(default_factory=list)     # [] : by input kind
    exclude: List[str] = field(default_factory=list)
    recursive: bool = True
    overrides: Dict[str, Any] = field(default_factory=dict)
    include: bool = True

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"file": self.file} if self.file else {"folder": self.folder}
        if self.folder:
            if self.pattern:
                out["pattern"] = list(self.pattern)
            if self.exclude:
                out["exclude"] = list(self.exclude)
            if not self.recursive:
                out["recursive"] = False
        if self.overrides:
            out["overrides"] = dict(self.overrides)
        if not self.include:
            out["include"] = False
        return out

    @classmethod
    def from_dict(cls, raw) -> "Input":
        if isinstance(raw, str):
            return cls(file=raw)
        if not isinstance(raw, dict) or not (raw.get("file") or raw.get("folder")):
            raise BatchError(f"an input needs `file` or `folder`: {raw!r}")
        listed = lambda v: [v] if isinstance(v, str) else list(v or [])
        return cls(file=str(raw.get("file") or ""), folder=str(raw.get("folder") or ""),
                   pattern=listed(raw.get("pattern")), exclude=listed(raw.get("exclude")),
                   recursive=bool(raw.get("recursive", True)),
                   overrides=dict(raw.get("overrides") or {}),
                   include=bool(raw.get("include", True)))


@dataclass
class Output:
    folder: str = ""               # "" : <job>_output beside the job file
    suffix: str = "_proc"
    locs: bool = True
    figures: str = "png"
    fitted: str = "beside_image"
    delete_fitted: bool = False
    # the summary page's plots across the files: `summary.csv` columns, or
    # patterns of them; [] is every number that differs between files
    trends: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        if not self.trends:
            del out["trends"]          # older readers refuse a key they lack
        return out

    @classmethod
    def from_dict(cls, raw) -> "Output":
        raw = dict(raw or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise BatchError(f"output: no option {', '.join(unknown)}; there are "
                             f"{', '.join(sorted(known))}")
        if isinstance(raw.get("trends"), str):
            raw["trends"] = [raw["trends"]]
        return cls(**raw)


@dataclass
class Job:
    chain: ChainSpec
    inputs: List[Input] = field(default_factory=list)
    output: Output = field(default_factory=Output)
    overrides: Dict[str, Any] = field(default_factory=dict)
    name: str = ""
    preflight: str = "skip"
    rerun: str = "skip_identical"
    chain_file: str = ""           # where the chain came from, if a file
    origin: Optional[Path] = None  # the job file; not written

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"schema": SCHEMA}
        if self.name:
            out["name"] = self.name
        out["chain"] = self.chain_file or self.chain.to_dict()
        if self.overrides:
            out["overrides"] = dict(self.overrides)
        out["inputs"] = [i.to_dict() for i in self.inputs]
        out["output"] = self.output.to_dict()
        out["preflight"] = self.preflight
        out["rerun"] = self.rerun
        return out

    @classmethod
    def from_dict(cls, raw, origin: Optional[Path] = None) -> "Job":
        if not isinstance(raw, dict):
            raise BatchError(f"{origin or 'a job'} is not a mapping")
        schema = raw.get("schema", SCHEMA)
        if schema != SCHEMA:
            raise BatchError(f"{origin or 'this'} is a {schema!r} file, not a batch "
                             f"job ({SCHEMA})")
        base = origin.parent if origin is not None else Path.cwd()
        given = raw.get("chain")
        if isinstance(given, str):
            chain_file = str(_resolve(given, base))
            spec = chains.read(chain_file)
        elif isinstance(given, dict):
            chain_file, spec = "", ChainSpec.from_dict(given)
        else:
            raise BatchError("a job needs a `chain`: a chain file, or the chain itself")
        inputs = raw.get("inputs") or []
        if not isinstance(inputs, list):
            raise BatchError("`inputs` must be a list")
        return cls(chain=spec, inputs=[Input.from_dict(i) for i in inputs],
                   output=Output.from_dict(raw.get("output")),
                   overrides=dict(raw.get("overrides") or {}),
                   name=str(raw.get("name") or ""),
                   preflight=str(raw.get("preflight", "skip")),
                   rerun=str(raw.get("rerun", "skip_identical")),
                   chain_file=chain_file, origin=origin)

    def output_folder(self) -> Path:
        if self.output.folder:
            base = self.origin.parent if self.origin is not None else Path.cwd()
            return _resolve(self.output.folder, base)
        if self.origin is not None:
            stem = self.origin.name.split(".")[0]
            return self.origin.parent / f"{stem}_output"
        return Path.cwd() / "smappy_batch"


def _resolve(path: str, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p)


def read_job(path) -> Job:
    path = Path(path).expanduser()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        import yaml
        raw = yaml.safe_load(text)
    return Job.from_dict(raw, origin=path.resolve())


def write_job(job: Job, path) -> Path:
    import yaml
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(chains._plain(job.to_dict()), sort_keys=False,
                                   default_flow_style=False), encoding="utf-8")
    return path


# ------------------------------------------------------------------ inputs

@dataclass
class Item:
    """One file the batch will process, and what it will be called."""
    path: Path
    name: str                              # the output subfolder
    overrides: Dict[str, Any] = field(default_factory=dict)


def resolve_inputs(job: Job, kind: str) -> List[Item]:
    """The files the job names, in order, each once; excluded ones left out.

    A file named twice -- by a folder rule and again on its own, to give it
    overrides -- keeps its first place and gains the later entry's
    overrides.  The output folder is never searched: a second run of a job
    whose output sits inside its input folder would otherwise pick up its own
    `_proc.h5` files as input.
    """
    out_folder = job.output_folder().resolve()
    base = job.origin.parent if job.origin is not None else Path.cwd()
    found: List[Tuple[Path, Dict[str, Any]]] = []
    seen: Dict[Path, int] = {}
    for entry in job.inputs:
        if not entry.include:
            continue
        if entry.file:
            paths = [_resolve(entry.file, base)]
        else:
            folder = _resolve(entry.folder, base)
            paths = []
            for pattern in entry.pattern or PATTERNS[kind]:
                walker = folder.rglob if entry.recursive else folder.glob
                paths += [p for p in walker(pattern) if p.is_file()]
            paths = sorted(set(paths), key=lambda p: str(p).lower())
            paths = [p for p in paths
                     if not _excluded(p, folder, entry.exclude)
                     and out_folder not in p.resolve().parents]
        for p in paths:
            key = p.resolve()
            if key in seen:
                found[seen[key]][1].update(entry.overrides)
                continue
            seen[key] = len(found)
            found.append((p, dict(entry.overrides)))
    return [Item(path=p, name=n, overrides=o)
            for (p, o), n in zip(found, _names([p for p, _ in found]))]


def _excluded(path: Path, folder: Path, patterns: Sequence[str]) -> bool:
    relative = str(path.relative_to(folder)) if folder in path.parents else path.name
    return any(fnmatch.fnmatch(path.name, x) or fnmatch.fnmatch(relative, x)
               for x in patterns)


def _stem(path: Path) -> str:
    name = path.name
    for suffix in (".ome.tiff", ".ome.tif", "_sml.mat"):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def _names(paths: Sequence[Path]) -> List[str]:
    """Output folder names: the stem, with the parent folder's name in front
    for stems that occur twice, numbered if even that is not enough."""
    stems = [_stem(p) for p in paths]
    names = [f"{p.parent.name}_{s}" if stems.count(s) > 1 else s
             for p, s in zip(paths, stems)]
    out: List[str] = []
    for name in names:
        candidate, n = name, 2
        while candidate in out:
            candidate, n = f"{name}_{n}", n + 1
        out.append(candidate)
    return out


# -------------------------------------------------------------- validation

@dataclass
class Problem:
    level: str            # "error" | "warning"
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.where}: {self.message}"


def _flat_specs(specs: Dict[str, ParamSpec], prefix: str = "") -> Dict[str, ParamSpec]:
    out: Dict[str, ParamSpec] = {}
    for name, spec in specs.items():
        if spec.children is not None:
            out.update(_flat_specs(spec.children, f"{prefix}{name}."))
        else:
            out[prefix + name] = spec
    return out


def _check_value(spec: ParamSpec, value) -> Optional[str]:
    """Why ``value`` will not do for this field, or None if it will."""
    if value is None:
        return None if spec.optional else "may not be empty"
    choices = spec.info.choices
    if choices is not None and not callable(choices):
        allowed = [c[0] if isinstance(c, tuple) else c for c in choices]
        if value not in allowed:
            return f"{value!r} is not one of {', '.join(map(repr, allowed))}"
    if spec.type is bool and not isinstance(value, bool):
        return f"{value!r} is not true or false"
    if spec.type in (int, float) and (isinstance(value, bool)
                                      or not isinstance(value, (int, float))):
        try:
            float(value)
        except (TypeError, ValueError):
            return f"{value!r} is not a number"
    if spec.type is int and isinstance(value, float) and not value.is_integer():
        return f"{value!r} is not a whole number"
    if spec.type is str and not isinstance(value, str):
        return f"{value!r} is not text"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if spec.info.min is not None and value < spec.info.min:
            return f"{value} is below the minimum, {spec.info.min:g}"
        if spec.info.max is not None and value > spec.info.max:
            return f"{value} is above the maximum, {spec.info.max:g}"
    return None


def _check_values(values: Dict[str, Any], fields: Dict[str, ParamSpec], where: str,
                  problems: List[Problem]) -> None:
    for key, value in values.items():
        if isinstance(value, dict) and not any(k == key for k in fields) \
                and any(k.startswith(key + ".") for k in fields):
            # the nested spelling of a part: check what is inside it
            _check_values({f"{key}.{k}": v for k, v in value.items()}, fields,
                          where, problems)
            continue
        spec = fields.get(key)
        if spec is None:
            near = [k for k in fields if k.split(".")[-1] == key.split(".")[-1]]
            hint = (f"; did you mean {' or '.join(near)}?" if near else
                    f"; it has {', '.join(sorted(fields))}")
            problems.append(Problem("error", where, f"no field {key!r}{hint}"))
            continue
        if spec.items is not None:
            # a list of parts: each item checked against the item's fields
            if not isinstance(value, (list, tuple)):
                problems.append(Problem("error", where, f"{key}: {value!r} is not a list"))
                continue
            for i, item in enumerate(value):
                if not isinstance(item, dict):
                    problems.append(Problem("error", where, f"{key}.{i}: not a mapping"))
                    continue
                _check_values(item, _flat_specs(spec.items), f"{where}: {key}.{i}",
                              problems)
            continue
        why = _check_value(spec, value)
        if why:
            problems.append(Problem("error", where, f"{key}: {why}"))


def _check_steps(cls, fields: Dict[str, ParamSpec], where: str,
                 problems: List[Problem]) -> None:
    """Each step's values against its fields, into the steps of a block: a
    block has no values of its own, and is a chain of its steps (`blocks.
    BlockPlugin`), its fields under its key."""
    for key, step, plugin in cls.steps:
        mine = {k[len(key) + 1:]: v for k, v in fields.items() if k.startswith(key + ".")}
        mine.pop(chains.GROUPING_FIELD, None)
        here = f"{where}: step {step.title()!r}"
        if isinstance(step, chains.Block):
            _check_steps(plugin, mine, here, problems)
            continue
        _check_values(step.values, mine, here, problems)
        if step.version and step.version != plugin.version:
            problems.append(Problem(
                "warning", here,
                f"saved with version {step.version} of {plugin.path}; version "
                f"{plugin.version} is installed"))


def validate_chain(spec: ChainSpec, overrides: Optional[Dict[str, Any]] = None,
                   where: str = "chain") -> Tuple[Optional[type], List[Problem]]:
    """The chain's plugin class, and everything wrong with the chain."""
    problems: List[Problem] = []
    try:
        cls = chain_class(spec)
    except ChainError as error:
        return None, [Problem("error", where, str(error))]
    if not cls.steps:
        problems.append(Problem("error", where, "no enabled steps"))
    fields = _flat_specs(cls.specs())
    _check_steps(cls, fields, where, problems)
    for n, (key, step, plugin) in enumerate(cls.steps):
        if n and is_source(plugin):
            problems.append(Problem(
                "warning", f"{where}: step {step.title()!r}",
                f"{plugin.path} makes a table of its own, and replaces what the "
                "steps before it did"))
    if overrides:
        _check_values(overrides, fields, f"{where}: overrides", problems)
    return cls, problems


def validate(job: Job, need_files: bool = True) -> List[Problem]:
    """Everything wrong with a job, errors first.  Empty is ready to run.

    ``need_files=False`` lets the inputs resolve to nothing: a watched folder
    is empty until the first acquisition lands in it."""
    problems: List[Problem] = []
    for name, value, allowed in (("preflight", job.preflight, PREFLIGHT),
                                 ("rerun", job.rerun, RERUN),
                                 ("output.figures", job.output.figures, FIGURES),
                                 ("output.fitted", job.output.fitted, FITTED)):
        if value not in allowed:
            problems.append(Problem("error", name, f"{value!r} is not one of "
                                                   f"{', '.join(allowed)}"))
    cls, found = validate_chain(job.chain, job.overrides)
    problems += found
    if cls is not None:
        fields = _flat_specs(cls.specs())
        kind = cls.input_kind()
        base = job.origin.parent if job.origin is not None else Path.cwd()
        for n, entry in enumerate(job.inputs):
            where = f"inputs[{n}]"
            if entry.file and not _resolve(entry.file, base).exists():
                problems.append(Problem("error", where, f"{entry.file} does not exist"))
            if entry.folder and not _resolve(entry.folder, base).is_dir():
                problems.append(Problem("error", where, f"{entry.folder} is not a folder"))
            if entry.overrides:
                _check_values(entry.overrides, fields, f"{where}: overrides", problems)
        if need_files and not any(p.level == "error" and p.where.startswith("inputs")
                                  for p in problems):
            if not resolve_inputs(job, kind):
                problems.append(Problem("error", "inputs",
                                        f"no {kind} files: nothing to process"))
    return sorted(problems, key=lambda p: p.level != "error")


# ------------------------------------------------------------------ running

def json_safe(value, depth: int = 0):
    """What of a step's `data` can be written as JSON, and small enough to.

    A curve of a few hundred points goes in; a column of a million values, a
    table, or an object nobody can read back is left out rather than guessed
    at.  NaN becomes null, which is what JSON has for "no number".
    """
    import numpy as np
    from .locs import Localizations
    if depth > 6:
        return None
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.size > MAX_LIST:
            return None
        return json_safe(value.tolist(), depth + 1)
    if isinstance(value, Localizations):
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return json_safe(dataclasses.asdict(value), depth + 1)
        except Exception:
            return None
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            safe = json_safe(v, depth + 1)
            if safe is not None or v is None:
                out[str(k)] = safe
        return out
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_LIST:
            return None
        return [json_safe(v, depth + 1) for v in value]
    return None


def scalars(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """The single numbers and words in a step's data, by dotted name: the
    columns of `summary.csv`."""
    out: Dict[str, Any] = {}
    for key, value in (data or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(scalars(value, name + "."))
        elif isinstance(value, (bool, int, float, str)) and value is not None:
            out[name] = value
    return out


def settings_for(cls, job: Job, item: Item) -> Tuple[Any, Dict[str, Any], Optional[Path]]:
    """The chain's settings for one file: the chain's values, the job's
    overrides, the file's, and the file itself handed to the first step if
    that step makes the table.  Also the flat values, for the fingerprint, and
    where a fit will write its own file."""
    flat = settings_values(cls.Settings())
    fields = _flat_specs(cls.specs())
    flat.update(_flattened(job.overrides, fields))
    flat.update(_flattened(item.overrides, fields))
    fitted = None
    if cls.steps and is_source(cls.steps[0][2]):
        key, _, first = cls.steps[0]
        if f"{key}.source.path" in flat:                     # a fit
            flat[f"{key}.source.path"] = str(item.path)
            flat[f"{key}.output.save"] = True
            if job.output.fitted == "output":
                fitted = job.output_folder() / item.name / f"{_stem(item.path)}_locs.hdf5"
                flat[f"{key}.output.path"] = str(fitted)
            else:
                from .plugins.fit import default_output_path
                flat[f"{key}.output.path"] = ""
                fitted = default_output_path(item.path)
        elif f"{key}.path" in flat:                          # a loader
            flat[f"{key}.path"] = str(item.path)
    return settings_from(cls.Settings, flat), flat, fitted


def _flattened(values: Dict[str, Any], fields: Dict[str, ParamSpec],
               prefix: str = "") -> Dict[str, Any]:
    """Overrides as flat dotted names.  A dictionary given for a *part*
    (``drift: {segments: 8}``) is spelled out; one given for a field whose
    value is itself a mapping stays whole."""
    out: Dict[str, Any] = {}
    for key, value in (values or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and name not in fields \
                and any(f.startswith(name + ".") for f in fields):
            out.update(_flattened(value, fields, name + "."))
        else:
            out[name] = value
    return out


def fingerprint(cls, flat: Dict[str, Any], job: Job, path: Path) -> str:
    """What makes a re-run the same run: the resolved values, the versions of
    the plugins, the options that change what is written, and the input as it
    is on disk."""
    stat = path.stat() if path.exists() else None
    what = {"values": chains._plain(flat),
            "versions": [(p.path, p.version) for _, _, p in cls.steps],
            "output": {k: getattr(job.output, k) for k in ("locs", "suffix", "figures",
                                                            "fitted")},
            "input": {"path": str(path.resolve()),
                      "size": stat.st_size if stat else None,
                      "mtime": stat.st_mtime if stat else None}}
    text = json.dumps(what, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _write_json(path: Path, value) -> None:
    """Written beside and renamed: a report half-written by a killed run must
    not be mistaken for a finished one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False, default=str),
                         encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_one(cls, job: Job, item: Item, report: Callable[[str], None] = lambda t: None
            ) -> Dict[str, Any]:
    """Process one file in a session of its own; its record, as written to
    ``results.json``.  Raises only `ChainAborted`: everything else that goes
    wrong is this file's and goes in the record."""
    from .session import Session
    from . import figures
    folder = job.output_folder() / item.name
    settings, flat, fitted = settings_for(cls, job, item)
    record: Dict[str, Any] = {
        "schema": RESULT_SCHEMA, "input": str(item.path), "name": item.name,
        "fingerprint": fingerprint(cls, flat, job, item.path),
        "started": datetime.now().isoformat(timespec="seconds"),
        "input_mtime": _mtime(item.path),
        "status": "failed", "output": {}, "steps": []}
    started = time.perf_counter()
    try:
        folder.mkdir(parents=True, exist_ok=True)    # before a fit writes into it
        session = Session()
        if not (cls.steps and is_source(cls.steps[0][2])):
            report("loading")
            session.load(item.path)
        runner = cls()
        runner.preflight_policy = job.preflight
        result = runner.run(session.context(0, progress=report), settings)
        session.apply(runner, result)
        if result.written:
            fitted = Path(result.written)
        if job.output.locs:
            target = folder / f"{_stem(item.path)}{job.output.suffix}.h5"
            session.save(target, gui_state=False)
            record["output"]["locs"] = str(target)
        written: List[str] = []
        for key, step_result in (result.data or {}).get("results", {}).items():
            if job.output.figures != "none":
                written += [str(p) for p in figures.save(
                    step_result.figures(), folder / "figures", prefix=key,
                    fmt=job.output.figures)]
        record["output"]["figures"] = written
        results = (result.data or {}).get("results", {})
        for step in (result.data or {}).get("steps", []):
            done = results.get(step["key"])
            record["steps"].append({**chains._plain({k: v for k, v in step.items()}),
                                    "data": json_safe(done.data if done else {}),
                                    **({"written": done.written}
                                       if done is not None and done.written else {})})
        record["n_localizations"] = len(session.locs)
        record["acquired"] = acquisition_time(session.locs.metadata)
        if fitted is not None:
            record["output"]["fitted"] = str(fitted)
            if job.output.delete_fitted and Path(fitted).exists() and \
                    Path(fitted).resolve() != Path(record["output"].get("locs", "")).resolve():
                Path(fitted).unlink()
                record["output"]["fitted_deleted"] = True
        record["status"] = "done"
    except ChainAborted:
        raise
    except ChainSkipped as why:
        record["status"] = "skipped"
        record["message"] = str(why)
    except Exception as error:
        record["message"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
    record["seconds"] = round(time.perf_counter() - started, 2)
    _write_json(folder / "results.json", record)
    return record


def emitter(stream=None) -> Callable[..., None]:
    """The line protocol the batch window reads: ``SMAPPY_<EVENT>\\t...``."""
    import sys
    stream = stream or sys.stdout

    def emit(event: str, *fields) -> None:
        print("\t".join([f"SMAPPY_{event}", *map(str, fields)]), file=stream, flush=True)
    return emit


def run(job: Job, limit: Optional[int] = None, force: bool = False,
        emit: Optional[Callable[..., None]] = None) -> Dict[str, Any]:
    """Run the job; the batch report, as written to ``batch_report.json``.

    Validates first and raises `BatchError` listing every error, so nothing
    starts on a job that cannot finish.  Files run one after another -- the
    plugins are parallel already, and two fits on one GPU are slower than one
    after the other.  The summary, the page and the report are written again
    after every file, so they can be looked at while the batch runs.
    """
    emit = emit or (lambda *a: None)
    cls = _validated(job, emit)
    items = resolve_inputs(job, cls.input_kind())
    if limit is not None:
        items = items[:limit]
    report = _new_report(job, cls)
    emit("JOBS", len(items))
    aborted = None
    for n, item in enumerate(items, 1):
        entry, aborted = _process(cls, job, item, f"{n}/{len(items)}", force, emit)
        report["files"].append(entry)
        if aborted:
            break
        publish(job, report)
    statuses = [f["status"] for f in report["files"]]
    report["status"] = ("aborted" if aborted else
                        "partial" if "failed" in statuses else "complete")
    report["finished"] = datetime.now().isoformat(timespec="seconds")
    publish(job, report)
    emit("REPORT", job.output_folder() / "batch_report.json")
    return report


def _validated(job: Job, emit, need_files: bool = True):
    problems = validate(job, need_files=need_files)
    errors = [p for p in problems if p.level == "error"]
    for problem in problems:
        emit("WARNING" if problem.level == "warning" else "ERROR", problem)
    if errors:
        raise BatchError("\n".join(map(str, errors)))
    return chain_class(job.chain)


def _new_report(job: Job, cls) -> Dict[str, Any]:
    job.output_folder().mkdir(parents=True, exist_ok=True)
    return {"schema": REPORT_SCHEMA,
            "started": datetime.now().isoformat(timespec="seconds"),
            "job": chains._plain(job.to_dict()), "chain": chains._plain(job.chain.to_dict()),
            "versions": {"smappy": _version(),
                         "plugins": {p.path: p.version for _, _, p in cls.steps}},
            "files": [], "status": "running"}


def _process(cls, job: Job, item: Item, where: str, force: bool, emit
             ) -> Tuple[Dict[str, Any], Optional[str]]:
    """One file, skipped if it is done already: its entry in the report, and
    why the batch must stop if a step said so."""
    out = job.output_folder()
    previous = _read_json(out / item.name / "results.json")
    _, flat, _ = settings_for(cls, job, item)
    if (not force and job.rerun == "skip_identical" and previous
            and previous.get("status") == "done"
            and previous.get("fingerprint") == fingerprint(cls, flat, job, item.path)):
        emit("SKIP", where, item.path, "already done with these settings")
        return {"input": str(item.path), "name": item.name, "status": "unchanged",
                "folder": str(out / item.name)}, None
    emit("START", where, item.path)
    try:
        record = run_one(cls, job, item,
                         report=lambda text: emit("PROGRESS", where, text))
    except ChainAborted as why:
        emit("FAILED", where, f"batch stopped: {why}")
        return {"input": str(item.path), "name": item.name, "status": "aborted",
                "message": str(why)}, str(why)
    if record["status"] == "done":
        emit("DONE", where, out / item.name)
    elif record["status"] == "skipped":
        emit("SKIP", where, item.path, record.get("message", ""))
    else:
        emit("FAILED", where, record.get("message", ""))
    return ({k: record.get(k) for k in ("input", "name", "status", "message", "seconds",
                                        "traceback")} | {"folder": str(out / item.name)},
            None)


def publish(job: Job, report: Dict[str, Any]) -> None:
    """`summary.csv`, the summary page and `batch_report.json`, as the report
    stands: after every file, so a long batch or a watched folder can be
    followed in them.  The page is a convenience, and a figure it cannot
    shrink costs that thumbnail, never the batch."""
    write_summary(job, report)
    try:
        from .batch_page import write_page
        write_page(job, report)
    except Exception:
        traceback.print_exc()
    _write_json(job.output_folder() / "batch_report.json", report)


# ------------------------------------------------------------------ watching

def is_complete(path: Path, seen: Dict[Path, Tuple[Any, float]], settle: float,
                now: Optional[float] = None) -> bool:
    """Whether a file in a watched folder is written to the end.

    A smappy localization file says so itself: its writer marks it
    ``writing`` until it closes it (`io.hdf5.LocalizationWriter`), and that
    word wins over any silence, since a live fit can wait minutes for the
    next frames.  Any other file is complete once neither its size nor its
    modification time -- of every file of a Micro-Manager series, which fills
    one file and then begins the next -- has changed for ``settle`` seconds,
    by its modification time or by what was seen of it.  ``seen`` holds each
    file's sizes and times, and when they were first seen as they are.
    """
    now = time.time() if now is None else now
    try:
        from .io.tiff import _series_files
        files = (_series_files(path) if path.name.lower().endswith((".ome.tif", ".ome.tiff"))
                 else [path])
        stats = [f.stat() for f in files]
    except OSError:
        return False
    state = tuple((s.st_size, s.st_mtime) for s in stats)
    if seen.get(path, (None,))[0] != state:
        seen[path] = (state, now)
    writing = _smappy_writing(path)
    if writing is not None:
        return not writing
    quiet = max(now - seen[path][1], now - max(s.st_mtime for s in stats))
    return quiet >= settle


def _smappy_writing(path: Path) -> Optional[bool]:
    """True while a smappy writer has the file open, False once it closed it,
    None for any other file -- or one that cannot be read just now, which is
    what a file being written by a program holding a lock on it looks like."""
    if path.suffix.lower() not in (".h5", ".hdf5"):
        return None
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if f.attrs.get("format") != "smappy-localizations" or "writing" not in f.attrs:
                return None
            return bool(f.attrs["writing"])
    except Exception:
        return True


def watch(job: Job, settle: float = 10.0, poll: float = 2.0, stop=None,
          emit: Optional[Callable[..., None]] = None, force: bool = False,
          idle: Optional[float] = None) -> Dict[str, Any]:
    """Run the job on every input file, and on every new one as it arrives.

    The inputs are looked at again every ``poll`` seconds; a file is run once
    it is complete (`is_complete`), and again if it changes afterwards.  What
    is done already with the same settings is skipped, as by `run`, so a
    watch restarted on the same folder picks up where it was.  Each file is
    given its name once and keeps it: a later file with the same stem does
    not rename the folder of one done before it.

    ``stop`` (anything with ``is_set``) ends the watch between files, and
    KeyboardInterrupt (Ctrl-C, SIGTERM in the command line) ends it at once,
    leaving the file in hand unfinished, to be redone.  Either way the report,
    the summary and the page are written, with the status ``stopped``.
    ``idle`` ends it after that many seconds without a new file: for tests,
    and for a script that watches one session's acquisitions.
    """
    emit = emit or (lambda *a: None)
    cls = _validated(job, emit, need_files=False)
    kind = cls.input_kind()
    report = _new_report(job, cls)
    report["status"] = "watching"
    entries: Dict[str, Dict[str, Any]] = {}
    names: Dict[Path, str] = {}
    handled: Dict[Path, Any] = {}           # a file's state when it was run
    seen: Dict[Path, Tuple[Any, float]] = {}
    stopped = (lambda: stop is not None and stop.is_set())
    folders = sorted({str(_resolve(i.folder or str(Path(i.file).parent),
                                   job.origin.parent if job.origin else Path.cwd()))
                      for i in job.inputs if i.include})
    emit("WATCHING", *folders)
    publish(job, report)
    n, last_new = 0, time.time()
    try:
        while not stopped():
            for item in resolve_inputs(job, kind):
                if stopped():
                    break
                key = item.path.resolve()
                if not is_complete(key, seen, settle) or handled.get(key) == seen[key][0]:
                    continue
                item.name = names.setdefault(key, _free_name(item.name, names.values()))
                n += 1
                entry, aborted = _process(cls, job, item, str(n), force, emit)
                handled[key] = seen[key][0]
                last_new = time.time()
                entries[item.name] = entry
                report["files"] = list(entries.values())
                publish(job, report)
                emit("REPORT", job.output_folder() / "batch_report.json")
                if aborted:
                    report["status"] = "aborted"
                    raise _Stop
            if idle is not None and time.time() - last_new >= idle:
                break
            _wait(poll, stopped)
    except (KeyboardInterrupt, _Stop):
        pass
    if report["status"] == "watching":
        report["status"] = "stopped"
    report["finished"] = datetime.now().isoformat(timespec="seconds")
    publish(job, report)
    emit("REPORT", job.output_folder() / "batch_report.json")
    return report


class _Stop(Exception):
    pass


def _free_name(name: str, taken) -> str:
    taken = set(taken)
    candidate, n = name, 2
    while candidate in taken:
        candidate, n = f"{name}_{n}", n + 1
    return candidate


def _wait(seconds: float, stopped) -> None:
    end = time.time() + seconds
    while time.time() < end and not stopped():
        time.sleep(min(0.2, max(end - time.time(), 0)))


def stop_on_stdin(stream=None):
    """An event set when ``stop`` is read from ``stream`` (stdin): how the
    batch window asks a watch to end after the file in hand, on every
    platform -- a signal on Windows can only kill.  The end of the stream is
    not a stop: a watch started from a scheduler has nothing on its stdin."""
    import sys
    import threading
    stream = stream or sys.stdin
    event = threading.Event()

    def read() -> None:
        try:
            for line in stream:
                if line.strip().lower() == "stop":
                    event.set()
                    return
        except (OSError, ValueError):
            pass
    threading.Thread(target=read, daemon=True, name="batch-stop").start()
    return event


def acquisition_time(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """When the data were taken, as ISO text, from Micro-Manager's
    ``StartTime`` wherever the table's metadata carries it (the file's own,
    or a file entry's); None when it does not say."""
    found = _find_key(metadata or {}, ("StartTime",), depth=0)
    if not found:
        return None
    import re
    m = re.match(r"\s*(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})", str(found))
    return f"{m.group(1)}T{m.group(2)}" if m else None


def _find_key(value, keys, depth: int):
    if depth > 4:
        return None
    if isinstance(value, dict):
        for k in keys:
            if value.get(k):
                return value[k]
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value[:50]
    else:
        return None
    for child in children:
        found = _find_key(child, keys, depth + 1)
        if found:
            return found
    return None


def _mtime(path: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return None


def summary_rows(job: Job, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per file, one column per scalar a step returned.

    Read from each file's ``results.json``, so a file skipped as unchanged
    still has its row from the run that produced it.
    """
    out = job.output_folder()
    rows: List[Dict[str, Any]] = []
    for entry in report["files"]:
        record = _read_json(out / entry["name"] / "results.json") or {}
        row: Dict[str, Any] = {"file": entry["input"], "status": entry["status"],
                               "seconds": record.get("seconds", ""),
                               "n_localizations": record.get("n_localizations", ""),
                               "message": entry.get("message") or record.get("message", "")}
        for step in record.get("steps", []):
            row.update(scalars(step.get("data") or {}, f"{step.get('key')}."))
        rows.append(row)
    return rows


def write_summary(job: Job, report: Dict[str, Any]) -> Path:
    """`summary_rows` as ``summary.csv``; columns in the order first met."""
    rows = summary_rows(job, report)
    columns: List[str] = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    path = job.output_folder() / "summary.csv"
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns or ["file", "status"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _version() -> str:
    try:
        from . import __version__
        return str(__version__)
    except Exception:
        return ""


# --------------------------------------------------------------- describing

def describe_plugin(path: str) -> str:
    """A plugin's fields as a chain names them, for a person or an agent
    writing one: flat dotted names, type, default, unit, bounds, choices."""
    from . import plugins
    cls = plugins.get(path)
    lines = [cls.path, f"  {cls.description}" if cls.description else "",
             f"  version {cls.version}"
             + (f"; only works {cls.grouping}" if cls.grouping else "")
             + ("; makes its own table (a first step: images or a file)"
                if is_source(cls) else "")]
    fields = _flat_specs(cls.specs())
    if not fields:
        lines.append("  (no settings)")
    for name, spec in fields.items():
        kind = getattr(spec.type, "__name__", str(spec.type))
        bits = [f"{kind}{' or null' if spec.optional else ''}",
                f"default {spec.default!r}"]
        if spec.info.unit:
            bits.append(spec.info.unit)
        if spec.info.min is not None or spec.info.max is not None:
            bits.append(f"range {spec.info.min} .. {spec.info.max}")
        choices = spec.info.choices
        if choices is not None and not callable(choices):
            bits.append("one of " + ", ".join(
                repr(c[0] if isinstance(c, tuple) else c) for c in choices))
        if spec.info.advanced:
            bits.append("advanced")
        lines.append(f"  {name}: {'; '.join(bits)}")
        if spec.info.help:
            lines.append(f"      {spec.info.help}")
    return "\n".join(l for l in lines if l)


def describe_chain(spec: ChainSpec) -> str:
    """A chain's steps, their keys, and the values each field will run with --
    the names a job's overrides use."""
    cls = chain_class(spec)
    lines = [f"{cls.path}  ({cls.input_kind()} in)",
             f"  grouping: {spec.grouping}"]
    settings = cls.Settings()
    for key, step, plugin in cls.steps:
        lines.append(f"  {key}: {plugin.path} (version {plugin.version})")
        for name, value in settings_values(getattr(settings, key)).items():
            lines.append(f"      {key}.{name} = {value!r}")
    return "\n".join(lines)
