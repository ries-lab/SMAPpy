"""The calibrations made so far, as links, and which of them fits a file: the spline fitters' *auto*.

A calibration is chosen far more often than it is made, and the choice was a path typed or browsed
for every fit.  What decides it is all known: the microscope that took the data, the state of its
devices the PSF follows (the 3D lens, the filters), the camera, EM gain on or off, one channel or
two (`microscope_db`).  So every calibration *PSF calibration* saves is added here with that
description, and a fitter left on *auto* takes the one that matches the data -- nearest in time to
the data where the data say when they were taken -- and says which.

**A list of links, not a search.**  Beads and data sit in different folders, often on iCloud, and
looking through folders for calibrations is slow; the list is one small JSON file in the
configuration folder (`list_path`), each line a path and a description.  A file moved or deleted is
skipped.  Besides the list, the data's own folder and its parent are looked in (`beside`): where
beads are saved next to the data they serve, they are found without ever having been listed.

**The description is the list's, not only the file's.**  A bead-average calibration does not record
its microscope; the window that made it knows it, and says so when it adds the file (`add`).  A file
whose microscope is not known anywhere is offered (`find`, ``matching=False``) but never chosen on
*auto*: a calibration of the wrong lens state fits, and fits badly, without a word.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

#: the list, in the configuration folder
LIST_NAME = "calibrations.json"
#: files looked at beside the data, at most: a folder of thousands of fits is not searched through
BESIDE_LIMIT = 60


@dataclass
class Entry:
    """One calibration: where it is and what it is for."""
    path: str
    dual: bool = False
    microscope: Optional[Dict[str, Any]] = None     # `MicroscopeResolution.summary` of the beads
    em_on: Optional[bool] = None
    made: str = ""                                  # ISO date: the beads', else the file's
    method: str = ""                                # "bead average", "pupil model", "in situ", ...

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def day(self) -> Optional[_dt.date]:
        try:
            return _dt.date.fromisoformat(self.made[:10])
        except ValueError:
            return None

    def label(self) -> str:
        """``name (2026-09-17, M2 · 3DA on)``, as the fitter's menu shows it."""
        bits = [self.made[:10]] if self.made else []
        scope = self.microscope or {}
        if scope.get("name"):
            states = scope.get("states") or {}
            bits.append(" · ".join([str(scope["name"])] + [f"{d} {states[d]}" for d in
                                                          scope.get("followed") or []
                                                          if states.get(d) is not None][:2]))
        else:
            bits.append("microscope unknown")
        return f"{self.name} ({', '.join(bits)})"


def list_path() -> Path:
    from . import config
    return config.config_dir() / LIST_NAME


def entries() -> List[Entry]:
    """The list as saved; an unreadable one reads as empty (it is links, made again by use)."""
    path = list_path()
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    out = []
    for item in raw.get("calibrations", []) if isinstance(raw, dict) else []:
        try:
            out.append(Entry(**{k: item[k] for k in item if k in Entry.__dataclass_fields__}))
        except TypeError:
            continue
    return out


def _write(items: List[Entry]) -> None:
    path = list_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "calibrations": [asdict(e) for e in items]}, indent=1))


def add(path, microscope: Optional[Dict[str, Any]] = None, em_on: Optional[bool] = None,
        dual: Optional[bool] = None, made: Optional[str] = None, method: str = "") -> Entry:
    """List the calibration at ``path``; what is given wins over what the file says (`describe`).
    A path listed before is replaced: its description is the latest."""
    path = Path(path).expanduser().resolve()
    found = describe(path) or Entry(str(path))
    entry = Entry(str(path),
                  dual=found.dual if dual is None else bool(dual),
                  microscope=microscope or found.microscope,
                  em_on=found.em_on if em_on is None else bool(em_on),
                  made=made or found.made, method=method or found.method)
    _write([e for e in entries() if Path(e.path) != path] + [entry])
    return entry


def remove(path) -> None:
    path = Path(path).expanduser().resolve()
    _write([e for e in entries() if Path(e.path) != path])


def describe(path) -> Optional[Entry]:
    """What a calibration file says about itself, or None when it is not one.  Reads the attributes and
    the parameters only, never the spline: it is asked of every candidate beside the data."""
    path = Path(path)
    if not path.is_file():
        return None
    made = _dt.date.fromtimestamp(path.stat().st_mtime).isoformat()
    if path.suffix.lower() == ".mat":
        if not path.name.lower().endswith("_3dcal.mat"):
            return None
        return Entry(str(path), dual=False, made=made, method="SMAP")
    try:
        import h5py
        with h5py.File(path, "r") as f:
            kind = f.attrs.get("format")
            if kind == "smappy-spline-calibration":
                parameters = json.loads(f["parameters_json"][()])
                em_on = bool(f.attrs["em_on"]) if "em_on" in f.attrs else None
                dual = False
            elif kind == "smappy-dual-color-calibration":
                parameters = json.loads(f["main"]["parameters_json"][()])
                parameters = dict(json.loads(f["parameters_json"][()]), **parameters)
                em_on = bool(f["main"].attrs["em_on"]) if "em_on" in f["main"].attrs else None
                dual = True
            else:
                return None
    except Exception:
        return None
    return Entry(str(path), dual=dual, microscope=parameters.get("microscope"), em_on=em_on, made=made,
                 method=str(parameters.get("method") or ""))


def recorded_calibration(path) -> Optional[str]:
    """The calibration a fitted localization file was fitted with, as it recorded it, or None when
    ``path`` is not a fitted file.  (Its provenance, `pipeline.provenance`: ``calibration.file``.)"""
    try:
        import h5py
        with h5py.File(Path(path), "r") as f:
            if not isinstance(f.get("locs"), h5py.Group) or "metadata" not in f.attrs:
                return None
            metadata = json.loads(f.attrs["metadata"])
    except Exception:
        return None
    found = (metadata.get("calibration") or {}).get("file") if isinstance(metadata, dict) else None
    return str(found) if found else None


def beside(data) -> List[Entry]:
    """Calibrations in the data's folder and its parent, by name first (``cal`` in it), at most
    `BESIDE_LIMIT` files read."""
    if not data:
        return []
    data = Path(data)
    folder = data if data.is_dir() else data.parent
    out, read = [], 0
    for place in (folder, folder.parent):
        try:
            names = sorted(p for p in place.iterdir() if p.is_file()
                           and p.suffix.lower() in (".h5", ".hdf5", ".mat") and "cal" in p.name.lower())
        except OSError:
            continue
        for p in names:
            if read >= BESIDE_LIMIT:
                return out
            read += 1
            found = describe(p)
            if found is not None:
                out.append(found)
    return out


def matches(entry: Entry, scope: Optional[Dict[str, Any]], em_on: Optional[bool], dual: bool) -> bool:
    """Whether ``entry`` is for data taken with ``scope`` (`MicroscopeResolution.summary`), EM gain
    ``em_on``, one or two channels: the same microscope and setup version, camera, the same state of
    every device the PSF follows, the same EM setting where both are known.  Unknown microscopes on
    either side never match: *auto* does not guess."""
    if entry.dual != bool(dual):
        return False
    if entry.em_on is not None and em_on is not None and entry.em_on != em_on:
        return False
    beads = entry.microscope or {}
    if not beads.get("name") or not scope or not scope.get("name"):
        return False
    if beads.get("name") != scope.get("name") or beads.get("version") != scope.get("version"):
        return False
    if beads.get("camera") and scope.get("camera") and beads["camera"] != scope["camera"]:
        return False
    before, now = beads.get("states") or {}, scope.get("states") or {}
    followed = set(beads.get("followed") or []) | set(scope.get("followed") or [])
    return all(before.get(d) == now.get(d) for d in followed
               if before.get(d) is not None or now.get(d) is not None)


def find(scope: Optional[Dict[str, Any]], em_on: Optional[bool], dual: bool, data=None,
         day: Optional[_dt.date] = None) -> List[tuple]:
    """``[(entry, matching)]``: the listed calibrations and those beside ``data`` that exist, one or two
    channels as asked, matching ones first -- nearest in time to ``day`` (the data's), else newest --
    then the rest, newest first.  The first, if matching, is *auto*'s."""
    seen, pool = set(), []
    for entry in entries() + beside(data):
        key = str(Path(entry.path).resolve())
        if key in seen or not Path(entry.path).is_file() or entry.dual != bool(dual):
            continue
        seen.add(key)
        pool.append(entry)

    def newest(e):
        return -(e.day or _dt.date.min).toordinal()

    def nearest(e):
        return abs(((e.day or _dt.date.min) - day).days) if day is not None else newest(e)
    good = sorted((e for e in pool if matches(e, scope, em_on, dual)), key=nearest)
    rest = sorted((e for e in pool if not matches(e, scope, em_on, dual)), key=newest)
    return [(e, True) for e in good] + [(e, False) for e in rest]


def chosen(scope, em_on, dual, data=None, day=None) -> Optional[Entry]:
    """*auto*'s calibration, or None."""
    found = find(scope, em_on, dual, data, day)
    return found[0][0] if found and found[0][1] else None
