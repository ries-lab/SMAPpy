"""A bead-learnt pupil model as the spline of a sample at depth: Spline 3D's *focus*.

A bead calibration is taken with the beads on the coverslip and the objective moved through them, so its
spline is the PSF of an emitter *at the coverslip*, its z the objective's movement.  An emitter in the
sample, with the focus some way into it, differs on both counts: the index mismatch between immersion
and sample adds a spherical aberration that grows with the depth, and the light beyond the critical
angle (SAF) reaches the objective only from emitters within a wavelength of the coverslip.  A bead
average cannot be moved to another depth; a pupil learnt from the beads (uipsf, `pupil`) can: its
model file, beside the spline (`pupil.model_path`), is drawn here for emitters at every height above
the coverslip with the focus at ``focus_nm`` (`uipsf.export.sample_psf` with the emitter's depth), the
aberrations the beads' and the sample's index the beads' medium unless given.

**The spline's z is the height above the coverslip.**  The focus and the heights fix one another
(`docs/uipsf/architecture.md`, "M5, step 1"), and the height is what a sample's structure is measured
in; the bead spline's "z above focus" is the objective's movement, about 20 % off it in water.

**It reaches below the coverslip, into the glass** (`BELOW_NM`): an emitter there is drawn as one at
the coverslip seen with the focus as much farther away, glass being the immersion's index.  Nothing
real is there, but a fit with a focus slightly wrong puts the lowest molecules there, and a spline
that ended at 0 parked them on its edge, where the auto focus below could no longer see them.  Fits
below 0 are filtered afterwards, as any other.

**Its range** is the bead spline's: as many planes about the plane in focus, the range the user chose
as meaningful when calibrating, never past `BELOW_NM` into the glass.

**The auto focus** (`auto_focus`) takes the focus from the lowest localisations, as in situ learning
does (`uipsf.workflows.insitu_rounds`, whose `coverslip_edge` it uses): no molecule is below the
coverslip and in most samples some are on it, so the localised heights have a lower edge, which must be
at 0.  A first fit with the bead spline gives the edge in the objective's movement, which is what the
focus is measured in; then the drawn spline is fitted, and the focus moved by the edge's shift times
the focus per height (n_imm/n_med paraxially, then a secant on the passes), until the edge is at 0
within `EDGE_TOLERANCE_NM` or `AUTO_PASSES` passes.  Only the brightest spots are fitted (`brightest`)
and with fewer iterations (`AUTO_ITERATIONS`): the edge needs a few hundred good heights, not all.
Without molecules on the coverslip the edge is the lowest molecules', above it, and the depth comes out
too small by their distance from it -- which matters little: what depends on the depth there, the SAF
and the index mismatch's aberration, changes little over a few hundred nm.

**In situ, every aberration held** (`insitu_focus`) is the accurate way: the in situ calibration
(`insitu_run.run_insitu`) with the bead model's whole pupil held, learning only what belongs to the sample
-- the focus and the blur -- from the molecules, started where the auto focus puts it.  Minutes, not
seconds, so what it learns is kept on disk (`keep`, `kept`) as well as for the session (`remember`, which
the auto focus uses too): a Preview finds a focus once, and the Run after it, or the next session, does not
find it again.

torch is imported inside the functions: Spline 3D imports this module without it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

#: how far the spline reaches below the coverslip, into the glass, nm (module docstring)
BELOW_NM = 300.0
#: the auto focus stops once the edge of the heights is this close to the coverslip, nm
EDGE_TOLERANCE_NM = 15.0
#: passes on a drawn spline after the first on the bead spline
AUTO_PASSES = 4
#: the spots the auto focus fits, the brightest of those found, and the iterations each fit may take
AUTO_ROIS, AUTO_ITERATIONS = 4000, 30
#: the fraction of the lowest heights the edge is fitted on (`coverslip_edge`)
EDGE_WINDOW = 0.3
#: at most this many times its CRLB is a height blurred in the edge's fit: a pile-up on the coverslip is one
EDGE_MAX_BLUR = 3.0


def model_file(cal) -> Optional[Path]:
    """The uipsf model file a spline calibration was sampled from, if it names one and it is there."""
    name = (cal.parameters or {}).get("model_file")
    if not name or cal.source is None:
        return None
    path = Path(cal.source).with_name(str(name))
    return path if path.is_file() else None


_LOADED: dict = {}


def load(path):
    """The model file at ``path`` (`uipsf.load_model`), on the CPU, kept while it is unchanged: loading takes a
    second, and the form and every fit of a batch ask again."""
    try:
        import torch
        import uipsf
    except ImportError as error:
        raise ValueError(f"a focus in the sample draws the PSF with uipsf, which needs torch ({error}): "
                         f"pip install smappy-smlm[uipsf]") from error
    path = Path(path)
    key = (str(path), path.stat().st_mtime)
    if key not in _LOADED:
        _LOADED.clear()
        learned = uipsf.load_model(path, device=torch.device("cpu"), dtype=torch.float64)
        if learned.n_channels != 1:
            raise ValueError(f"{path.name} is a model of {learned.n_channels} channels: a focus in the sample is "
                             f"drawn for one channel")
        _LOADED[key] = learned
    return _LOADED[key]


def medium(learned) -> float:
    """The sample's index the model was learnt with: the beads' medium."""
    return float(learned.model.channels[0].psf.grid.optics.n_med)


def with_medium(learned, n_med: Optional[float]):
    """(model, shared) of ``learned`` with the sample's index ``n_med`` (None: the model's own).  Only the
    optics change: the pupil, the blur and every learnt value are the beads'."""
    if n_med is None or abs(float(n_med) - medium(learned)) < 1e-9:
        return learned.model, learned.shared
    import copy

    import torch
    from uipsf.export.model_file import build
    config = copy.deepcopy(learned.config)
    for channel in config.get("channels", [config]):
        channel["optics"]["n_med"] = float(n_med)
    return build(config, device=torch.device("cpu"), dtype=torch.float64), learned.shared


def draw(model, shared, focus_nm: float, heights_nm, roi_px: int) -> np.ndarray:
    """The PSF [N, Y, X] of emitters on the central pixel at ``heights_nm`` above the coverslip, the focus
    ``focus_nm`` into the sample as the objective was moved; the bead's size left out, the learnt blur kept.
    A negative height is in the glass: an emitter at the coverslip with the focus that much farther."""
    from uipsf.export import sample_psf
    h = np.asarray(heights_nm, float)
    xyz = np.zeros((len(h), 3))
    xyz[:, 2] = -float(focus_nm) + np.minimum(h, 0.0)
    psf = sample_psf(model, shared, xyz, roi_px, depth_nm=np.maximum(h, 0.0))
    return psf.double().numpy()


def in_focus_height(model, shared, focus_nm: float, span_nm: float, roi_px: int = 15) -> float:
    """The height whose PSF is sharpest (its brightest pixel highest) with the focus at ``focus_nm``: where
    the plane in focus lies above the coverslip, searched to ``span_nm`` beyond the paraxial guess."""
    optics = model.channels[0].psf.grid.optics
    guess = float(focus_nm) * optics.n_med / optics.n_imm
    h = np.arange(max(-BELOW_NM, guess - span_nm), guess + span_nm + 1, 20.0)
    return float(h[int(np.argmax(draw(model, shared, focus_nm, h, roi_px).max(axis=(1, 2))))])


def _roi_px(cal) -> int:
    """The region the calibration was sampled on: its record's, else its spline's cells plus one."""
    sampled = ((cal.parameters or {}).get("sampling") or {}).get("roi_size")
    return int(sampled) if sampled else int(cal.shape[1]) + 1


def depth_spline(cal, focus_nm: float, n_med: Optional[float] = None, learned=None):
    """The `SplineCalibration` of ``cal``'s model (`model_file`) for a sample whose focus is ``focus_nm``
    above the coverslip: z the height above the coverslip, as many planes ``cal.dz`` apart about the plane in
    focus as ``cal`` has about its own, at most `BELOW_NM` into the glass; normalised as the bead
    calibrations' (the brightest plane sums to 1).  ``learned`` is the model, if loaded already."""
    from ..io.calibration import SplineCalibration
    from .core import spline_coefficients
    if learned is None:
        path = model_file(cal)
        if path is None:
            raise ValueError(f"{Path(cal.source).name if cal.source else 'the calibration'} has no pupil model "
                             f"beside it: a focus in the sample needs a calibration made with the pupil (Zernike) "
                             f"method")
        learned = load(path)
    model, shared = with_medium(learned, n_med)
    dz, roi = float(cal.dz), _roi_px(cal)
    half = (cal.shape[0] - 1) * dz / 2
    focus = in_focus_height(model, shared, focus_nm, half)
    high = float(np.ceil((focus + half) / dz) * dz)          # planes on multiples of dz
    n = int(np.ceil((high - max(focus - half, -BELOW_NM)) / dz))
    heights = high - np.arange(n + 1) * dz                    # plane 0 the highest, as in a bead stack
    psf = np.maximum(draw(model, shared, focus_nm, heights, roi), 0)
    psf /= float(psf.sum(axis=(1, 2)).max())
    psf = psf.astype(np.float32)
    parameters = dict(cal.parameters or {})
    parameters.update({
        "sample_depth": {"focus_nm": float(focus_nm), "n_med": medium(learned) if n_med is None else float(n_med),
                         "in_focus_height_nm": focus, "heights_nm": [float(heights[-1]), float(high)]},
        "z_reference": "the coverslip: z is the height above it",
        "z_convention": "emitter_nm = -(index-z0)*dz"})
    return SplineCalibration(spline_coefficients(psf), dz, high / dz, x0=(roi - 1) / 2, psf=psf, em_on=cal.em_on,
                             pixelsize_um=cal.pixelsize_um, source=cal.source, parameters=parameters)


_DRAWN: dict = {}


def at_depth(cal, focus_nm: float, n_med: Optional[float] = None, learned=None):
    """`depth_spline`, kept for the last few (calibration, focus, index): a form's Preview and every file of a
    batch ask for the same one, and a spline takes a second to draw and fit."""
    path = model_file(cal) if learned is None else None
    key = (str(cal.source), id(cal) if cal.source is None else None, path.stat().st_mtime if path else id(learned),
           round(float(focus_nm), 1), None if n_med is None else round(float(n_med), 4))
    if key not in _DRAWN:
        if len(_DRAWN) >= 4:
            _DRAWN.pop(next(iter(_DRAWN)))
        _DRAWN[key] = depth_spline(cal, focus_nm, n_med, learned)
    return _DRAWN[key]


def in_focus(cal) -> float:
    """The height of the plane in focus of a drawn spline, 0 for any other: where a fit starts in z."""
    return float(((cal.parameters or {}).get("sample_depth") or {}).get("in_focus_height_nm", 0.0))


# ------------------------------------------------------------------ the auto focus

def brightest(images: np.ndarray, n: int = AUTO_ROIS) -> np.ndarray:
    """The indices of the ``n`` spots with the most photons above their region's edge, the brightest
    first.  Photons in the region, not the filtered peak detection ranks by: a defocused molecule has as
    many with a lower peak, and the lowest molecules are the defocused ones when the focus is deep."""
    images = np.asarray(images, float)
    edge = np.concatenate([images[:, 0], images[:, -1], images[:, 1:-1, 0], images[:, 1:-1, -1]], axis=1)
    signal = images.sum(axis=(1, 2)) - np.median(edge, axis=1) * images[0].size
    return np.argsort(-signal, kind="stable")[:n]


def fitted_heights(cal, images: np.ndarray, iterations: int = AUTO_ITERATIONS, n_threads: int = 0):
    """(z, z_err) in nm of the fits of ``images`` (photons, with the read noise's variance added) on ``cal``,
    only the good ones: finite, with a z inside the spline by more than a plane, and a z error at most
    three times the median."""
    from ..psf import SplinePSF
    model = SplinePSF(cal, z_start_nm=in_focus(cal))
    values = model.unpack(model.fit(images, iterations=iterations, n_threads=n_threads))
    z, err = np.asarray(values["z_nm"], float), np.asarray(values["z_err_nm"], float)
    index = cal.z_nm_to_index(z)
    good = np.isfinite(z) & np.isfinite(err) & (err > 0) & (index > 1) & (index < cal.shape[0] - 2)
    if good.any():
        good &= err <= 3 * np.median(err[good])
    return z[good], err[good]


@dataclass
class AutoDepth:
    focus_nm: float
    edge_sd_nm: float
    spike: float                        # the fraction of the edge's window on the coverslip
    n: int                              # heights the edge was fitted on
    passes: list = field(default_factory=list)       # (focus, edge shift) per pass on a drawn spline
    messages: list = field(default_factory=list)     # what must not be missed: the depth is uncertain
    notes: list = field(default_factory=list)        # what is worth knowing: no pile-up on the coverslip
    heights: np.ndarray = None          # of the pass kept, nm above the coverslip, and their CRLBs
    errors: np.ndarray = None
    method: str = "auto"
    spline: object = None               # in situ: the spline of the learnt model at its focus
    blur_nm: Optional[float] = None     # in situ: the learnt blur


def _next_focus(passes, paraxial: float) -> float:
    """The next pass's focus from the passes' (focus, edge shift) so far: between the two closest that put the
    edge on either side of the coverslip, where there are such, else a secant step whose slope is held
    within a factor of 2 of the paraxial one (on heights without a pile-up the edge scatters by tens of nm
    between passes, and a free secant then stepped hundreds of nm past the coverslip)."""
    above = [(f, s) for f, s in passes if s > 0]
    below = [(f, s) for f, s in passes if s < 0]
    if above and below:
        (fa, sa), (fb, sb) = min(above, key=lambda p: p[1]), max(below, key=lambda p: p[1])
        return max(0.0, fa + (fb - fa) * sa / (sa - sb))
    slope = paraxial
    if len(passes) > 1 and passes[-1][0] != passes[-2][0]:
        measured = (passes[-1][1] - passes[-2][1]) / (passes[-1][0] - passes[-2][0])
        slope = float(np.clip(measured, paraxial / 2, paraxial * 2))
    return max(0.0, passes[-1][0] - passes[-1][1] / slope)


def auto_focus(cal, images: np.ndarray, n_med: Optional[float] = None, iterations: int = AUTO_ITERATIONS,
               n_threads: int = 0, report: Callable[[str], None] = lambda text: None, learned=None) -> AutoDepth:
    """The focus, nm above the coverslip, at which the lowest of ``images`` (ROIs in photons, with the read
    noise's variance added) lie on the coverslip (module docstring).  ``cal`` is the bead spline, ``learned``
    its model if loaded already (`model_file`)."""
    from uipsf.workflows.insitu_rounds import coverslip_edge
    learned = learned if learned is not None else load(model_file(cal))
    keep = brightest(images)
    images = np.ascontiguousarray(images[keep], np.float32)

    heights = []

    def edge_of(spline):
        z, err = fitted_heights(spline, images, iterations, n_threads)
        edge = coverslip_edge(z, err, EDGE_WINDOW, EDGE_MAX_BLUR)
        if edge is None or not np.isfinite(edge.sd_nm):
            raise ValueError(f"the focus cannot be found from these molecules: {len(z)} fitted well, "
                             f"no lower edge to their heights; type it")
        heights.append((z, err))
        return edge

    report("focus on auto: fitting the bead spline")
    first = edge_of(cal)
    optics = learned.model.channels[0].psf.grid.optics
    n = medium(learned) if n_med is None else float(n_med)
    paraxial = n / optics.n_imm                     # the edge's shift per nm of focus
    focus = max(0.0, -first.shift_nm)               # the bead spline's z is the objective's movement
    passes, edges = [], []
    for k in range(AUTO_PASSES):
        report(f"focus on auto: pass {k + 1}, focus {focus:.0f} nm")
        edges.append(edge_of(at_depth(cal, focus, n_med, learned)))
        passes.append((focus, edges[-1].shift_nm))
        if abs(edges[-1].shift_nm) <= max(EDGE_TOLERANCE_NM, edges[-1].sd_nm):
            break
        focus = _next_focus(passes, paraxial)
    best = int(np.argmin([abs(s) for _, s in passes]))
    edge = edges[best]
    out = AutoDepth(float(passes[best][0]), edge.sd_nm / paraxial, edge.spike, edge.n, passes,
                    heights=heights[best + 1][0], errors=heights[best + 1][1])
    if abs(edge.shift_nm) > max(EDGE_TOLERANCE_NM, edge.sd_nm):
        out.messages.append(f"the lowest localisations are {edge.shift_nm:.0f} nm from the coverslip after "
                            f"{len(passes)} passes: the depth is uncertain, type it if it is known")
    if edge.spike <= 0:
        out.notes.append("no molecules piled up on the coverslip: the lowest may lie above it, and the "
                            "depth is then at least this")
    return out


# ------------------------------------------------------------------ in situ, every aberration held

def insitu_focus(cal, paths, camera, n_med: Optional[float] = None, focus_nm: Optional[float] = None,
                 molecules: int = 1000, rounds: int = 2, progress=None, cancel=None) -> AutoDepth:
    """The focus learnt in situ from the movies at ``paths`` with every aberration held from ``cal``'s pupil
    model (`insitu_run.run_insitu` with *hold* every aberration): only the focus, the blur and the molecules
    are learnt, the bead model's own blur the start.  ``focus_nm`` is where it starts, typically the auto
    depth's; on None the in situ run estimates one.  The result's ``spline`` is the learnt model drawn as
    `depth_spline` draws a bead model, z the height above the coverslip.

    Fewer molecules and rounds than a calibration in situ (`InSituSettings`, `InSituModelSettings`): with
    nothing of the pupil to learn, a thousand molecules fix the focus and the blur, and the rounds stop once
    the focus stops moving."""
    from dataclasses import fields
    from types import SimpleNamespace

    from .insitu import InSituSettings
    from .insitu_run import InSituModelSettings, run_insitu
    from .pupil import BeadSettings, OpticsSettings, RunSettings
    path = model_file(cal)
    if path is None:
        raise ValueError("in situ with the beads' aberrations needs a calibration made with the pupil (Zernike) "
                         "method: its model is the start")
    learned = load(path)
    o = learned.model.channels[0].psf.grid.optics
    optics = OpticsSettings(NA=o.NA, wavelength_nm=o.wavelength_nm, n_immersion=o.n_imm,
                            n_medium=o.n_med if n_med is None else float(n_med), n_coverslip=o.n_cov,
                            coverslip_um=o.d_cov_nm / 1000, pixel_x_nm=o.pixel_size_nm[0],
                            pixel_y_nm=o.pixel_size_nm[1])
    known = {f.name for f in fields(BeadSettings)}
    model = BeadSettings(**{k: v for k, v in (learned.options or {}).items() if k in known})

    from .psf_methods import LearningProgress, duration
    bar, last = LearningProgress(), [""]

    def tell(entry):
        # the window's own reading of the entries: a round's line, or its learning's iterations, with
        # the time left once there is a guess; the round's line kept in front of its iterations
        if progress is None or not isinstance(entry, dict):
            return
        _, left, text = bar.update(entry)
        if "message" in entry:
            last[0] = entry["message"]
            text = ""
        progress("focus in situ: " + "; ".join(filter(None, [last[0], text]))
                 + ("" if left is None else f", about {duration(left)} left"))
    result = run_insitu(paths, optics, model,
                        InSituModelSettings(focus_nm=focus_nm, hold="all", max_rounds=rounds, spline_z="height"),
                        InSituSettings(molecules=molecules), run=RunSettings(start_from=str(path), save_model=False),
                        camera=camera, progress=tell, cancel=cancel)
    dtype = result.model.psf.grid.rho.dtype                  # the model's own, as it was learnt
    shared = {k: v.detach().cpu().to(dtype) for k, v in result.cal.shared().items()}
    focus = float(shared["geometry.focus"])
    drawn = SimpleNamespace(model=result.model, shared=shared, config=None)
    spline = depth_spline(cal, focus, learned=drawn)
    sigma = shared.get("psf.sigma")
    blur = None if sigma is None else float(sigma.mean())
    spline.parameters["sample_depth"].update(method="in situ, every aberration held", blur_nm=blur)
    last = result.run.rounds[-1]
    h = np.asarray(last.heights if last.heights is not None else [], float)
    err = np.asarray(last.heights_sd if last.heights_sd is not None else np.full(len(h), np.nan), float)
    edge = last.edge
    out = AutoDepth(focus, np.nan if edge is None else edge.sd_nm, 0.0 if edge is None else edge.spike,
                    len(last.used), heights=h - (0.0 if edge is None else edge.shift_nm), errors=err,
                    method="in situ", spline=spline, blur_nm=blur)
    out.notes += list(result.messages)
    out.cal, out.drawn = result.cal, drawn
    return out


# ------------------------------------------------------------------ what was found, kept

_FOUND: dict = {}


def found(key):
    """What `remember` kept under ``key`` in this session, or None: a Preview and the Run after it, or every
    Preview, find the focus once."""
    return _FOUND.get(key)


def remember(key, result: AutoDepth) -> AutoDepth:
    if len(_FOUND) >= 8:
        _FOUND.pop(next(iter(_FOUND)))
    _FOUND[key] = result
    return result


def kept_path(key) -> Path:
    """Where an in situ result is kept between sessions: the configuration directory, named by its key --
    not beside the data, where the fitters' *auto* would take it for a bead calibration."""
    import hashlib
    import json

    from ..config import config_dir
    name = hashlib.sha1(json.dumps(key, default=str).encode()).hexdigest()[:16]
    return config_dir() / "sample_depth" / f"{name}_3dcal.h5"


def keep(key, result: AutoDepth) -> Optional[Path]:
    """The in situ ``result`` saved (`kept_path`): its spline, its model beside it, the key and the heights in
    the spline's record, so `kept` gives it back without learning again.  Best effort: a cache."""
    try:
        from .pupil import save
        path = kept_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.spline.parameters["sample_depth"].update(key=_jsonable(key), heights=result.heights.tolist(),
                                                       errors=result.errors.tolist(), notes=result.notes,
                                                       n=result.n)
        save(path, result.cal, result.spline, overwrite=True)
        return path
    except Exception:
        return None


def kept(key) -> Optional[AutoDepth]:
    """An in situ result `keep` saved under ``key``, or None."""
    path = kept_path(key)
    if not path.is_file():
        return None
    try:
        from ..io.calibration import load_spline_calibration
        spline = load_spline_calibration(path)
        record = spline.parameters.get("sample_depth") or {}
        if record.get("key") != _jsonable(key):
            return None
        out = AutoDepth(float(record["focus_nm"]), np.nan, 0.0, int(record.get("n", 0)),
                        heights=np.asarray(record.get("heights", []), float),
                        errors=np.asarray(record.get("errors", []), float), method="in situ", spline=spline,
                        blur_nm=record.get("blur_nm"))
        out.notes = list(record.get("notes", [])) + [f"kept from an earlier run: {path}"]
        return out
    except Exception:
        return None


def _jsonable(value):
    import json
    return json.loads(json.dumps(value, default=str))
