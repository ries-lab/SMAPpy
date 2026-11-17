"""A PSF calibration's run kept in its file, so that the calibration can be looked at again without its beads.

A calibration file held the spline and, for the pupil model, uipsf's model (`pupil.save`) -- enough to fit
with, not enough to judge it by: the *Model vs data* page needs the beads, the fit quality a refit of every
bead, the bead table their positions, and nothing said where the bead stacks were or with which settings the
calibration was made.  A calibration and its stacks part ways (copied to another computer, the stacks archived),
and its results could then not be seen again at all.

So the PSF calibration window writes, beside what the fitters read, a group of its own, `GROUP`, with what
its pages show of a run (the user's, November 2026):

* **the settings** of every part the method read (`psf_run.inputs`), restored into the window's forms;
* **the sources**: each stack's path, the path relative to the calibration and its size and time, so the
  stacks can be found again -- where they were, beside the calibration as they were, or in a folder the
  user points to (`find_sources`);
* **the bead table** as the window showed it, and each bead's place for the rings on the Stacks page;
* **the model against the data**, averaged over the used beads (`model_views.ModelViews` without the single
  beads: a bead's own volume is the stacks', and comes back with them);
* **the fit quality**, the refit of every bead;
* for the pupil model, **its bead table** (uipsf's `BeadRow`s, for the *Beads* page) and **the model it
  started from**, as a uipsf model group of its own (`START`), for the Zernike pages' comparison;
* **provenance**: the camera as resolved, the versions, the date, the run's summary and warning.

`load` reads it back as a `Run` whose result is a `SavedRun`, and `ENTRY` is the method entry the window
uses for such a run (pages, table, summary), whatever method made it.  All of it is a few hundred kB.

For the pupil model and the bead average, one channel or two (`RECORDED`): their pages are what is kept
here.  The other methods' files are opened as before, from what the fitters read.

The layout: the group's values as a tree -- a dict a group, a list or tuple a group of numbered entries
(attribute ``kind``), an array a dataset, anything else a JSON string dataset (attribute ``json``): JSON in
datasets rather than attributes, which HDF5 limits to 64 kB.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

#: the group of a calibration file the window's run is kept in
GROUP = "psf_window"
#: the model a pupil model's run started from, as a uipsf model group of the file (`uipsf.load_model`'s group)
START = "psf_window_start"
VERSION = 1
#: the methods whose runs are kept, by key
RECORDED = ("pupil", "average")


# ------------------------------------------------------------------ the tree

def _plain(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    raise TypeError(f"cannot keep {type(o).__name__} in a calibration's run")


def _put(group, name: str, value) -> None:
    if isinstance(value, dict):
        g = group.create_group(name)
        g.attrs["kind"] = "dict"
        g.attrs["keys"] = json.dumps([[str(k), type(k).__name__] for k in value])
        for i, v in enumerate(value.values()):
            _put(g, str(i), v)
    elif isinstance(value, (list, tuple)) and any(isinstance(v, (np.ndarray, dict, list, tuple)) for v in value):
        g = group.create_group(name)
        g.attrs["kind"] = "tuple" if isinstance(value, tuple) else "list"
        g.attrs["n"] = len(value)
        for i, v in enumerate(value):
            _put(g, str(i), v)
    elif isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
        group.create_dataset(name, data=value, compression="gzip" if value.size > 256 else None)
    elif isinstance(value, np.ndarray):                 # strings, objects: an array again when read
        ds = group.create_dataset(name, data=json.dumps(value.tolist(), default=_plain))
        ds.attrs["json"] = True
        ds.attrs["array"] = True
    else:
        ds = group.create_dataset(name, data=json.dumps(value, default=_plain))
        ds.attrs["json"] = True


def _get(obj):
    import h5py
    if isinstance(obj, h5py.Group):
        kind = obj.attrs.get("kind", "dict")
        if kind == "dict":
            keys = json.loads(obj.attrs["keys"])
            return {(int(k) if t == "int" else k): _get(obj[str(i)]) for i, (k, t) in enumerate(keys)}
        items = [_get(obj[str(i)]) for i in range(int(obj.attrs["n"]))]
        return tuple(items) if kind == "tuple" else items
    if obj.attrs.get("json", False):
        text = obj[()]
        value = json.loads(text.decode() if isinstance(text, bytes) else text)
        return np.asarray(value) if obj.attrs.get("array", False) else value
    return np.asarray(obj[()])


# ------------------------------------------------------------------ writing

def recordable(method, run) -> bool:
    """Whether ``run`` of ``method`` is kept: a saved run of a recorded method (`RECORDED`), not stopped."""
    if getattr(method, "key", None) not in RECORDED or run is None or run.saved is None:
        return False
    if run.extra.get("loaded"):
        return False                          # its file holds it already
    cal = getattr(run.result, "cal", None)
    return getattr(getattr(cal, "result", None), "stop_reason", None) != "cancelled"


def _source(path, calibration: Path) -> dict:
    p = Path(path)
    entry = {"path": str(p.resolve()), "name": p.name}
    try:
        entry["relative"] = os.path.relpath(p.resolve(), calibration.resolve().parent)
    except ValueError:                        # another drive on Windows
        entry["relative"] = None
    if p.is_file():
        stat = p.stat()
        entry.update(size=int(stat.st_size), mtime=float(stat.st_mtime))
    return entry


def _camera(camera) -> Optional[dict]:
    if camera is None:
        return None
    try:
        return json.loads(json.dumps(camera, default=_plain))
    except TypeError:
        return {k: v for k, v in vars(camera).items() if isinstance(v, (str, int, float, bool, type(None)))}


def _records(run):
    result = run.result
    rois = getattr(result, "rois", None)
    return getattr(rois, "records", None) if rois is not None else result.beads.records


def _settings(inputs: dict) -> dict:
    return {name: dataclasses.asdict(value) for name, value in inputs.items()
            if dataclasses.is_dataclass(value) and not isinstance(value, type) and name != "camera"}


def _views(views) -> Optional[dict]:
    if views is None:
        return None
    return {"data": [np.asarray(v, np.float32) for v in views.data],
            "model": [np.asarray(v, np.float32) for v in views.model],
            "z_nm": np.asarray(views.z_nm, float), "titles": list(views.titles),
            "mirror_axis": views.mirror_axis, "averaged": bool(views.averaged)}


def record_of(method, run, paths=None) -> dict:
    """What `write` keeps of ``run``, as a tree of plain values and arrays (module docstring)."""
    from .. import __version__
    from .psf_methods import _positions, sample_of
    inputs = run.extra.get("inputs") or {}
    paths = list(paths if paths is not None else run.extra.get("paths") or [])
    marks = method.initial_marks(run)
    rows = method.rows(run, marks)
    record = {
        "version": VERSION, "method": method.key, "sample": sample_of(method),
        "mode": "Dual colour" if run.dual else "Single channel", "dual": bool(run.dual),
        "date": datetime.now().isoformat(timespec="seconds"),
        "smappy": __version__, "settings": _settings(inputs), "camera": _camera(inputs.get("camera")),
        "sources": [_source(p, Path(run.saved)) for p in paths],
        "columns": list(method.columns),
        "rows": [[str(v) for v in values] for values, _ in rows],
        "greyed": [bool(g) for _, g in rows],
        "marks": {int(k): v for k, v in (marks or {}).items()},
        "positions": [list(p) for p in _positions(_records(run))],
        "summary": method.summary(run), "warning": run.extra.get("warning") or "",
        "detected": run.extra.get("detected") or "",
        "views": _views(run.extra.get("views")), "quality": run.extra.get("quality"),
    }
    cal = getattr(run.result, "cal", None)
    if cal is not None:
        import uipsf
        record["uipsf"] = uipsf.__version__
        record["bead_rows"] = [dataclasses.asdict(r) for r in cal.beads.rows]
    return record


def write(method, run, paths=None) -> bool:
    """Keep ``run`` in its saved file (`GROUP`, replaced if there), and the model it started from (`START`);
    False, and nothing written, where the run is not kept (`recordable`)."""
    import h5py
    if not recordable(method, run):
        return False
    record = record_of(method, run, paths)
    with h5py.File(run.saved, "a") as f:
        for name in (GROUP, START):
            if name in f:
                del f[name]
        g = f.create_group(GROUP)
        g.attrs["version"] = VERSION
        for key, value in record.items():
            _put(g, key, value)
    _keep_start(run)
    return True


def _keep_start(run) -> None:
    """The model the run started from (*start from*), copied whole into `START`: the Zernike pages draw it
    beside the learnt one, and its own file may be gone when this one is opened."""
    import h5py
    inputs = run.extra.get("inputs") or {}
    start = getattr(inputs.get("run"), "start_from", None) if inputs else None
    if not start or getattr(run.result, "start", None) is None or not Path(start).exists():
        return
    from uipsf.export.model_file import model_group
    from .pupil import model_path
    source = Path(start)
    try:
        with h5py.File(source, "r") as f:
            found = model_group(f) is not None
    except OSError:
        return
    if not found and model_path(source).exists():
        source = model_path(source)          # an old save's model file beside its spline
    with h5py.File(source, "r") as src, h5py.File(run.saved, "a") as dst:
        group = model_group(src)
        if group is not None:
            src.copy(group, dst, name=START)


# ------------------------------------------------------------------ reading

def has_record(path) -> bool:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return GROUP in f
    except OSError:
        return False


@dataclass
class SavedRun:
    """A run read back from its file (`load`): what its pages draw, without the beads' stacks."""
    path: Path
    record: dict
    calibration: object                         # the spline, or the dual calibration
    learned: object = None                      # uipsf's `LearnedPSF`, for a pupil model
    start: object = None                        # the `LearnedPSF` it started from, if kept
    sources: list = field(default_factory=list)     # as the window's Stacks page reads them

    @property
    def method(self) -> str:
        return self.record.get("method", "")

    @property
    def rois(self):
        """The window's Stacks page reads its sources here, as a run's own."""
        from types import SimpleNamespace
        return SimpleNamespace(sources=self.sources, records=[])


def _load_calibration(path, dual: bool):
    if dual:
        from .dual import load_dual_color_calibration
        return load_dual_color_calibration(path)
    from ..io.calibration import load_spline_calibration
    return load_spline_calibration(path)


def _views_of(saved: Optional[dict]):
    if not saved:
        return None
    from .model_views import ModelViews
    return ModelViews([np.asarray(v, float) for v in saved["data"]], [np.asarray(v, float) for v in saved["model"]],
                      np.asarray(saved["z_nm"], float), list(saved["titles"]),
                      mirror_axis=saved.get("mirror_axis"), averaged=bool(saved.get("averaged", True)))


def load(path):
    """The run kept in the calibration at ``path`` as a `psf_methods.Run` (`extra["loaded"]`), its result a
    `SavedRun`; a ValueError where the file keeps none (saved before November 2026, or by another method)."""
    import h5py

    from .psf_methods import Run
    path = Path(path)
    with h5py.File(path, "r") as f:
        if GROUP not in f:
            raise ValueError(f"{path.name} keeps no run of the PSF calibration window: it was saved before "
                             f"November 2026 or by a method that keeps none, and only its spline can be used")
        version = int(f[GROUP].attrs.get("version", 0))
        if version > VERSION:
            raise ValueError(f"{path.name} keeps its run as version {version}; this SMAPpy reads up to "
                             f"{VERSION}: update SMAPpy")
        record = {name: _get(f[GROUP][name]) for name in f[GROUP]}
        has_start = START in f
    dual = bool(record.get("dual"))
    saved = SavedRun(path, record, _load_calibration(path, dual))
    if record.get("method") == "pupil":
        import uipsf
        saved.learned = uipsf.load_model(path)
        if has_start:
            saved.start = uipsf.load_model(path, group=START)
    saved.sources = [dict(path=s["path"]) for s in record.get("sources", [])]
    extra = {"loaded": True, "views": _views_of(record.get("views")), "quality": record.get("quality"),
             "warning": record.get("warning") or "", "detected": record.get("detected") or ""}
    return Run(saved, dual, path, extra)


def settings_of(record: dict) -> dict:
    """The parts' settings as kept, as the dataclasses the window's forms hold: run, optics, model, spline
    and detection (the dual form's in dual colour), each only where it was kept."""
    from ..plugins import settings_from
    from .core import CalibrationSettings
    from .dual import DualColorSettings
    from .pupil import BeadSettings, OpticsSettings, RunSettings, SplineSettings
    classes = {"run": RunSettings, "optics": OpticsSettings, "model": BeadSettings, "spline": SplineSettings,
               "detection": DualColorSettings if record.get("dual") else CalibrationSettings}
    kept = record.get("settings") or {}
    return {name: settings_from(cls, kept[name]) for name, cls in classes.items() if name in kept}


# ------------------------------------------------------------------ finding the stacks again

def _same(path: Path, source: dict) -> bool:
    """Whether ``path`` is the stack ``source`` names: the size where it was kept (a copy changes the time)."""
    if not path.exists():
        return False
    size = source.get("size")
    return size is None or not path.is_file() or path.stat().st_size == size


def find_sources(record: dict, calibration, folder=None) -> list:
    """Where each stack the run was made from is now, or None for one not found: where it was, at the same
    place relative to the calibration (a folder moved whole), and with ``folder`` (the user's pick) by its
    name anywhere in that folder or below, of the same size."""
    calibration = Path(calibration)
    found = []
    by_name = None
    for source in record.get("sources", []):
        hit = None
        candidates = [Path(source["path"])]
        if source.get("relative"):
            candidates.append(calibration.parent / source["relative"])
        for candidate in candidates:
            if _same(candidate, source):
                hit = candidate
                break
        if hit is None and folder is not None:
            if by_name is None:
                by_name = {}
                for p in Path(folder).rglob("*"):
                    by_name.setdefault(p.name, []).append(p)
            hit = next((p for p in by_name.get(source["name"], []) if _same(p, source)), None)
        found.append(None if hit is None else str(hit))
    return found


# ------------------------------------------------------------------ the entry for a loaded run

class Loaded:
    """The window's method entry for a run read from its file (`load`): its pages from what was kept, its
    table as it was shown.  Marks, *Uncertainty from halves* and the single beads need the stacks: with them
    found, *Calibrate* makes the run again."""
    has_stack = True
    has_halves = True                 # shown as the method's, greyed with the reason (`halves_needed`)
    marks_include = True

    @staticmethod
    def columns_of(run) -> tuple:
        return tuple(run.result.record.get("columns", ()))

    def rows(self, run, marks: dict) -> list:
        record = run.result.record
        return [(tuple(values), bool(grey)) for values, grey in zip(record.get("rows", []),
                                                                    record.get("greyed", []), strict=False)]

    def initial_marks(self, run) -> dict:
        return {}

    def mark(self, run, bead: int) -> str:
        return "exclude"

    def halves_needed(self, run) -> Optional[str]:
        return "opened from a file: calibrate again from its stacks first"

    def positions(self, run) -> list:
        return [(int(s), float(x), float(y), int(b)) for s, x, y, b in run.result.record.get("positions", [])]

    def summary(self, run) -> str:
        record = run.result.record
        return (f"opened {run.result.path.name} ({record.get('date', '?')}): "
                f"{record.get('summary', '')}")

    def stacks(self, run):
        cal = run.result.calibration
        return run.extra.get("views"), getattr(cal, "main", cal)

    def save_as(self, run, path) -> Path:
        import shutil
        if Path(path).resolve() != Path(run.saved).resolve():
            shutil.copyfile(run.saved, path)
        return Path(path)

    def pages(self, run) -> list:
        from types import SimpleNamespace

        from . import plots
        from .psf_methods import Page, _model_page, _not_yet
        saved = run.result
        method, dual = saved.method, run.dual
        pages = []
        if saved.learned is not None:
            from uipsf.diagnostics.figures import draw_atlas, draw_pupil_and_zernikes, unit_choices, zernike_table
            learned, start = saved.learned, saved.start
            n = learned.model.n_channels
            for c in range(n):
                units = unit_choices()
                pages.append(Page(f"Pupil and Zernikes{'' if n == 1 else f' {c + 1}'}",
                                  lambda f, s, e, chosen, c=c: draw_pupil_and_zernikes(
                                      f, learned, c, start, units=chosen["units"], halves=learned.halves),
                                  follows_selection=False, controls={"units": units}))
            atlas = _atlas(learned.diagnostics.get("atlas"))
            if atlas is not None:
                pages.append(Page("Residual atlas", lambda f, s, e: draw_atlas(f, atlas), follows_selection=False))
            pages.append(Page("Zernike table", None, follows_selection=False,
                              table=lambda: zernike_table(learned, reference=start, halves=learned.halves)))
        pages.append(_model_page(run))
        quality = run.extra.get("quality")

        def fit_quality(figure, selected, excluded):
            if quality is None:
                _not_yet(figure, "no fit quality was kept with this calibration")
                return None
            if method == "pupil":
                return plots.draw_pupil_fit_quality(figure, quality, selected, excluded)
            if dual:
                return plots.draw_paired_fit_quality(figure, SimpleNamespace(calibration=saved.calibration),
                                                     quality[0], quality[1], selected)
            return plots.draw_fit_quality(figure, quality[0], quality[1], selected)
        pages.append(Page("Fit quality", fit_quality))
        pages.append(Page("CRLB", lambda f, s, e: plots.draw_crlb(
            f, SimpleNamespace(_crlb_pixel_nm=_pixel_nm(saved.record)), dual=dual, calibration=saved.calibration),
            follows_selection=False))
        rows = _bead_rows(saved.record)
        if rows and any(r.field_xy is not None for r in rows):
            from uipsf.diagnostics.figures import draw_beads
            from uipsf.workflows.rejection import BeadTable
            table = SimpleNamespace(beads=BeadTable(rows, {}))
            pages.append(Page("Beads", lambda f, s, e: draw_beads(f, table, s)))
        return pages


def _pixel_nm(record: dict):
    pixel = (record.get("camera") or {}).get("pixelsize_um")
    if isinstance(pixel, (list, tuple)):
        pixel = pixel[0]
    return None if pixel is None else float(pixel) * 1000


def _bead_rows(record: dict) -> list:
    rows = record.get("bead_rows")
    if not rows:
        return []
    from uipsf.workflows.rejection import BeadRow
    names = {f.name for f in dataclasses.fields(BeadRow)}
    out = []
    for r in rows:
        r = {k: v for k, v in r.items() if k in names}
        if r.get("field_xy") is not None:
            r["field_xy"] = tuple(r["field_xy"])
        out.append(BeadRow(**r))
    return out


def _atlas(saved: Optional[dict]):
    """uipsf's `Atlas` from what its model file keeps of it (`model_file._atlas`)."""
    if not saved:
        return None
    from uipsf.diagnostics.atlas import Atlas, Finding, Group
    groups = {k: Group(**{n: (np.asarray(v) if isinstance(v, list) else v) for n, v in g.items()})
              for k, g in saved.get("groups", {}).items()}
    return Atlas(float(saved["mean_deviance"]), groups, {k: tuple(v) for k, v in saved.get("checks", {}).items()},
                 [Finding(**f) for f in saved.get("findings", [])])


ENTRY = Loaded()


def entry_of(method, run):
    """The entry that answers for ``run``: `ENTRY` for a run read from its file, else ``method``."""
    return ENTRY if run is not None and run.extra.get("loaded") else method
