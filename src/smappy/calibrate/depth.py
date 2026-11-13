"""A bead-learnt pupil model as the spline of a sample at depth: Spline 3D's *focus*.

A bead calibration is taken with the beads on the coverslip and the objective moved through them, so its
spline is the PSF of an emitter *at the coverslip*, its z the objective's movement.  An emitter in the
sample, with the focus some way into it, differs on both counts: the index mismatch between immersion
and sample adds a spherical aberration that grows with the depth, and the light beyond the critical
angle (SAF) reaches the objective only from emitters within a wavelength of the coverslip.  A bead
average cannot be moved to another depth; a pupil learnt from the beads (uipsf, `pupil`) can: its
model, inside the spline's file (`pupil.save`; an older one beside it, `pupil.model_path`), is drawn here for emitters at every height above
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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..params import param
from .medium import differs, for_drawing, index_field, medium_field, mismatch_text, settle

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
    """Where the uipsf model a spline calibration was sampled from is, if it has one: the calibration file
    itself, which carries it since November 2026 (`pupil.carries_model`), or the model file an older save
    put beside it and named (``parameters["model_file"]``).  `uipsf.load_model` reads either."""
    if cal.source is None:
        return None
    from .pupil import carries_model
    source = Path(cal.source)
    if source.suffix.lower() == ".h5" and carries_model(source):
        return source
    name = (cal.parameters or {}).get("model_file")
    if not name:
        return None
    path = source.with_name(str(name))
    return path if path.is_file() else None


def model_of(cal, channels: int = 1):
    """The pupil model a spline calibration was drawn from, loaded, or None: the model it carries
    (`model_file`), or the microscope's pupil it records (``parameters["pupil"]``, `microscope_psf`) --
    two places, one model, so that *focus* draws either alike."""
    path = model_file(cal)
    if path is not None:
        return load(path) if channels == 1 else load(path, channels)
    kept = (cal.parameters or {}).get("pupil")
    if kept:
        from .microscope_psf import learned
        return learned(kept)
    return None


def has_model(cal) -> bool:
    """Whether ``cal`` has a pupil model to draw at depth, without loading it (torch need not be there)."""
    return model_file(cal) is not None or bool((cal.parameters or {}).get("pupil"))


_LOADED: dict = {}


def load(path, channels: int = 1):
    """The model file at ``path`` (`uipsf.load_model`), on the CPU, kept while it is unchanged: loading takes a
    second, and the form and every fit of a batch ask again.  ``channels``: as many as the caller draws, one
    but for a dual calibration (`dual_depth_calibration`)."""
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
        _LOADED[key] = learned
    learned = _LOADED[key]
    if learned.n_channels != channels:
        raise ValueError(f"{path.name} is a model of {learned.n_channels} channel(s): a focus in the sample is "
                         f"drawn here for {channels}")
    return learned


def medium(learned) -> float:
    """The sample's index the model was learnt with: the beads' medium."""
    return float(learned.model.channels[0].psf.grid.optics.n_med)


def resolved_medium(learned, n_med) -> float:
    """The sample's index ``n_med`` stands for (`medium.for_drawing`): None the model's own, "matched" its
    immersion's, else the number."""
    if n_med is None:
        return medium(learned)
    if n_med == "matched":
        return float(learned.model.channels[0].psf.grid.optics.n_imm)
    return float(n_med)


def _index_key(n_med):
    return None if n_med is None else n_med if isinstance(n_med, str) else round(float(n_med), 4)


def with_medium(learned, n_med):
    """(model, shared) of ``learned`` with the sample's index ``n_med`` (None: the model's own; "matched": its
    immersion's).  Only the optics change: the pupil, the blur and every learnt value are the beads'."""
    if n_med is None or abs(resolved_medium(learned, n_med) - medium(learned)) < 1e-9:
        return learned.model, learned.shared
    n_med = resolved_medium(learned, n_med)
    import copy

    import torch
    from uipsf.export.model_file import build
    config = copy.deepcopy(learned.config)
    for channel in config.get("channels", [config]):
        channel["optics"]["n_med"] = float(n_med)
    return build(config, device=torch.device("cpu"), dtype=torch.float64), learned.shared


def draw(model, shared, focus_nm: float, heights_nm, roi_px: int, field_xy=None, channel: int = 0) -> np.ndarray:
    """The PSF [N, Y, X] of emitters on the central pixel at ``heights_nm`` above the coverslip, the focus
    ``focus_nm`` into the sample as the objective was moved; the bead's size left out, the learnt blur kept.
    A negative height is in the glass: an emitter at the coverslip with the focus that much farther.
    ``field_xy``: where on the chip, for a field-dependent model (its centre if None); ``channel`` of a model of
    several, its defocus offset and photon share included (`uipsf.export.sample_psf`)."""
    from uipsf.export import sample_psf
    h = np.asarray(heights_nm, float)
    xyz = np.zeros((len(h), 3))
    xyz[:, 2] = -float(focus_nm) + np.minimum(h, 0.0)
    psf = sample_psf(model, shared, xyz, roi_px, depth_nm=np.maximum(h, 0.0), field_xy=field_xy, channel=channel)
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
        learned = model_of(cal)
        if learned is None:
            raise ValueError(f"{Path(cal.source).name if cal.source else 'the calibration'} carries no pupil "
                             f"model: a focus in the sample needs a calibration made with the pupil (Zernike) "
                             f"method, or the microscope's pupil")
    model, shared = with_medium(learned, n_med)
    dz, roi = float(cal.dz), _roi_px(cal)
    half = (cal.shape[0] - 1) * dz / 2
    focus = in_focus_height(model, shared, focus_nm, half)
    high = float(np.ceil((focus + half) / dz) * dz)          # planes on multiples of dz
    n = int(np.ceil((high - max(focus - half, -BELOW_NM)) / dz))
    heights = high - np.arange(n + 1) * dz                    # plane 0 the highest, as in a bead stack
    psf = np.maximum(draw(model, shared, focus_nm, heights, roi), 0)
    scale = float(psf.sum(axis=(1, 2)).max())
    psf = (psf / scale).astype(np.float32)
    field = getattr(cal, "field_splines", None)
    if field is not None:
        # the same nodes, drawn at this depth and normalised by the centre's scale, so the blend stays one
        from ..field_splines import FieldSplines
        nodes = [np.maximum(draw(model, shared, focus_nm, heights, roi, field_xy=(float(x), float(y))), 0) / scale
                 for y in field.ys for x in field.xs]
        field = FieldSplines(np.stack([spline_coefficients(n.astype(np.float32)) for n in nodes]), field.xs, field.ys)
    parameters = dict(cal.parameters or {})
    parameters.update({
        "sample_depth": {"focus_nm": float(focus_nm), "n_med": resolved_medium(learned, n_med),
                         "in_focus_height_nm": focus, "heights_nm": [float(heights[-1]), float(high)]},
        "z_reference": "the coverslip: z is the height above it",
        "z_convention": "emitter_nm = -(index-z0)*dz"})
    return SplineCalibration(spline_coefficients(psf), dz, high / dz, x0=(roi - 1) / 2, psf=psf, em_on=cal.em_on,
                             pixelsize_um=cal.pixelsize_um, source=cal.source, parameters=parameters,
                             field_splines=field)


def dual_depth_calibration(dual, focus_nm: float, n_med: Optional[float] = None):
    """`depth_spline` for a dual-colour calibration whose two-channel pupil model it carries (`pupil.save`):
    both channels drawn for the focus ``focus_nm`` on one grid of heights, normalised jointly as
    `pupil.dual_calibration` normalises them (`core.positive_pair_models`), so each keeps its photon share;
    the calibration's map and split as they are.  z is the height above the coverslip, as `depth_spline`'s."""
    from .core import CalibrationSettings, positive_pair_models
    from .dual import DualColorCalibration
    main = dual.main
    learned = model_of(main, channels=2)
    if learned is None:
        raise ValueError(f"{Path(main.source).name if main.source else 'the calibration'} carries no pupil "
                         f"model: a focus in the sample needs a dual calibration made with the pupil (Zernike) "
                         f"method")
    model, shared = with_medium(learned, n_med)
    dz, roi = float(main.dz), _roi_px(main)
    half = (main.shape[0] - 1) * dz / 2
    focus = in_focus_height(model, shared, focus_nm, half)
    high = float(np.ceil((focus + half) / dz) * dz)
    n = int(np.ceil((high - max(focus - half, -BELOW_NM)) / dz))
    heights = high - np.arange(n + 1) * dz
    raw = np.stack([np.maximum(draw(model, shared, focus_nm, heights, roi, channel=c), 0) for c in range(2)])
    (first, second), _ = positive_pair_models(raw, dz, CalibrationSettings(roi_size=roi, smooth_z_nm=0.0,
                                                                           smooth_xy_px=0.0))
    record = {"focus_nm": float(focus_nm), "n_med": resolved_medium(learned, n_med),
              "in_focus_height_nm": focus, "heights_nm": [float(heights[-1]), float(high)], "method": "typed"}
    for drawn, old in ((first, main), (second, dual.secondary)):
        drawn.z0 = high / dz                       # plane 0 the highest: z the height above the coverslip
        drawn.source, drawn.em_on, drawn.pixelsize_um = old.source, old.em_on, old.pixelsize_um
        drawn.parameters = {**(old.parameters or {}), **(drawn.parameters or {}), "sample_depth": record,
                            "z_reference": "the coverslip: z is the height above it"}
    return DualColorCalibration(first, second, dual.transformation, dict(dual.geometry),
                                {**(dual.parameters or {}), "sample_depth": record})


_DUAL: dict = {}


def dual_at_depth(dual, path, focus_nm: float, n_med: Optional[float] = None):
    """`dual_depth_calibration`, kept for the last few (file, focus, index): the form's Preview and every file of
    a batch ask for the same pair, and drawing it takes seconds."""
    path = Path(path)
    key = (str(path), path.stat().st_mtime, round(float(focus_nm), 1), _index_key(n_med))
    if key not in _DUAL:
        if len(_DUAL) >= 4:
            _DUAL.pop(next(iter(_DUAL)))
        _DUAL[key] = dual_depth_calibration(dual, focus_nm, n_med)
    return _DUAL[key]


_DRAWN: dict = {}


def at_depth(cal, focus_nm: float, n_med: Optional[float] = None, learned=None):
    """`depth_spline`, kept for the last few (calibration, focus, index): a form's Preview and every file of a
    batch ask for the same one, and a spline takes a second to draw and fit."""
    path = model_file(cal) if learned is None else None
    # the microscope's pupil is named by what it records; a model file by its time
    kept = json.dumps((cal.parameters or {}).get("pupil"), sort_keys=True, default=str) if learned is None else None
    key = (str(cal.source), id(cal) if cal.source is None else None,
           path.stat().st_mtime if path else kept if kept != "null" else id(learned),
           round(float(focus_nm), 1), _index_key(n_med))
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


def in_beads_orientation(images: np.ndarray, cal, data_em_on) -> np.ndarray:
    """``images`` mirrored in x into the orientation of ``cal``'s beads where the data were taken with the other
    EM gain (`io.calibration.matched_to_em`, which mirrors the spline for the fit instead): the auto focus fits
    the bead spline and its drawings, all in the beads' orientation, and a diagonal astigmatism seen mirrored
    reads every height with the wrong sign.  Unknown on either side, as they are."""
    if getattr(cal, "em_on", None) is None or data_em_on is None or bool(cal.em_on) == bool(data_em_on):
        return images
    return np.ascontiguousarray(np.asarray(images)[..., ::-1])


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
    learned = learned if learned is not None else model_of(cal)
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
    n = resolved_medium(learned, n_med)
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
    learned = model_of(cal)
    if learned is None:
        raise ValueError("in situ with the beads' aberrations needs a calibration made with the pupil (Zernike) "
                         "method, or the microscope's pupil: its model is the start")
    o = learned.model.channels[0].psf.grid.optics
    # the objective's NA: a minimal pupil's grid reaches beyond it (`MinimalPupil`)
    na = float(getattr(learned.model.channels[0].psf.pupil, "na", o.NA))
    optics = OpticsSettings(NA=na, wavelength_nm=o.wavelength_nm, n_immersion=o.n_imm,
                            medium="typed", n_medium=resolved_medium(learned, n_med), n_coverslip=o.n_cov,
                            coverslip_um=o.d_cov_nm / 1000, pixel_x_nm=o.pixel_size_nm[0],
                            pixel_y_nm=o.pixel_size_nm[1])
    model = BeadSettings.from_options(learned.options)

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
                        InSituSettings(molecules=molecules),
                        run=RunSettings(start_from=str(path) if path is not None else "", save_model=False),
                        camera=camera, progress=tell, cancel=cancel,
                        # the microscope's pupil held, where there is no model file to start from
                        pupil=None if path is not None else cal.parameters["pupil"]["pupil"])
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
    """The in situ ``result`` saved (`kept_path`): its spline with its model inside, the key and the heights in
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


# ------------------------------------------------------------------ the sample, one part for every consumer

#: *focus*'s choices, Spline 3D's (`plugins.fit.FOCUS_CHOICES`) and the sample part's
SAMPLE_FOCUS_CHOICES = (("auto", "auto"), ("insitu", "in situ"), ("typed", "typed"),
                        ("calibrated", "as calibrated"))


@dataclass
class SampleSettings:
    """Where in the sample a pupil (Zernike) calibration is drawn: the part of everything else that takes a
    spline and is not Spline 3D -- the DECODE training, *Simulate* -- with Spline 3D's own fields, names and
    meanings (`plugins.fit.SplineModelSettings`), which keeps them in its *model* part so that saved settings
    stay valid.  Shown only for a calibration that carries a model (`sample_state`); a bead average has none.
    The bead's size is never drawn, nor the beads' damping of the supercritical ring (`VectorPSF.for_emitters`):
    the SAF comes from the emitter's depth alone, the magnitude being the model's as learnt."""
    from ..params import param as _param
    focus: str = _param("auto", label="focus", choices=SAMPLE_FOCUS_CHOICES,
                        help="where the focus is in the sample, for the PSF drawn there from the pupil model, z "
                             "then the height above the coverslip: auto, from the lowest localizations; in situ, "
                             "learnt from the molecules with the beads' aberrations; typed; or as calibrated, the "
                             "beads' PSF at the coverslip")
    focus_nm: float = param(0.0, label="focus depth", unit="nm", min=0.0,
                             help="how far into the sample the objective was moved from the coverslip; typed "
                                  "here, else found from the movie")
    # the sample's index, a choice with presets (`calibrate.medium`): the number keeps its old name
    medium: str = medium_field("the index the calibration was learnt with")
    sample_index: Optional[float] = index_field()

    def __post_init__(self):
        settle(self.medium, self.sample_index)      # refuses an unknown choice; the pair is read by `resolve`

    def n_med(self):
        """What `depth` is handed as the sample's index (`medium.for_drawing`)."""
        return for_drawing(self.medium, self.sample_index)


def sample_state(cal, sample, prefix: str = "sample.", found: Optional[AutoDepth] = None):
    """(shown, active, hints) of a `SampleSettings` part at ``prefix`` for the calibration ``cal`` (loaded, or
    None): the part only where ``cal`` carries a model, the depth typed only on *typed*, what *auto* and *in
    situ* resolve to where ``found`` says.  Never loads the model: the form asks with every change."""
    has = cal is not None and has_model(cal)
    drawn = has and sample.focus != "calibrated"
    shown = {prefix + "focus": has, prefix + "focus_nm": drawn, prefix + "medium": drawn,
             prefix + "sample_index": drawn}
    active = {prefix + "focus_nm": sample.focus == "typed"}
    hints = {}
    if drawn and sample.focus in ("auto", "insitu"):
        word = "auto" if sample.focus == "auto" else "in situ"
        hints[prefix + "focus_nm"] = (f"{word}: {found.focus_nm:.0f} nm" if found is not None
                                      else f"{word}: found from the movie")
    if drawn:
        hints[prefix + "medium"] = "as calibrated"
        hints[prefix + "sample_index"] = stands_for(cal, sample.medium)
    return shown, active, hints


def stands_for(cal, medium_choice: str):
    """What the sample part's number shows in grey for ``medium_choice`` (`calibrate.medium`): the index it
    stands for with ``cal``, or a word where that is not known without loading the model."""
    from .medium import resolve
    learnt = learnt_index(cal)
    value = resolve("" if medium_choice == "typed" else medium_choice, None, learnt, learnt_immersion(cal))
    return ("the immersion's" if value == "matched" else "as calibrated" if value is None else value)


def learnt_index(cal) -> Optional[float]:
    """The sample's index ``cal``'s model was learnt with: recorded in its parameters (`pupil.learnt_index`),
    else the model's own where it is loaded already (it is after a first drawing) -- loading it for a form
    would take a second, and torch; None where neither says."""
    params = getattr(cal, "parameters", None) or {}
    if params.get("sample_index") is not None:
        return float(params["sample_index"])
    if (params.get("pupil") or {}).get("n_medium") is not None:      # the microscope's pupil
        return float(params["pupil"]["n_medium"])
    path = model_file(cal)
    learned = _LOADED.get((str(path), path.stat().st_mtime)) if path is not None else None
    return medium(learned) if learned is not None else None


def learnt_immersion(cal) -> Optional[float]:
    """The immersion's index ``cal``'s model was learnt with, where its parameters record it."""
    value = (getattr(cal, "parameters", None) or {}).get("n_immersion")
    return None if value is None else float(value)


def index_notice(cal, medium_choice: str, number) -> Optional[str]:
    """A line for the form when the PSF is drawn for another sample index than ``cal`` was learnt with
    (`medium.mismatch_text`); None when they agree or either is unknown."""
    from .medium import resolve
    learnt = learnt_index(cal)
    drawn = resolve(medium_choice, number, learnt, learnt_immersion(cal))
    if drawn == "matched" or not differs(drawn, learnt):
        return None
    return mismatch_text(float(drawn), learnt)


def for_sample(cal, sample, *, images=None, paths=None, camera=None, report=lambda text: None,
               warn=None, cancel=None):
    """``(spline, found)``: ``cal`` drawn for the sample ``sample`` says (`SampleSettings`), and the
    `AutoDepth` that found its focus, or None.  The one way a consumer other than Spline 3D resolves the part:

    * a calibration without a model, or *as calibrated*: ``cal`` itself;
    * *typed*: drawn at the typed depth (`at_depth`);
    * *auto*: the focus from ``images`` -- spots as the fit cuts them, photons with the read noise's variance
      added (`plugins.fit.sample_rois`), in the data's orientation, ``camera`` saying its EM gain
      (`in_beads_orientation`) -- by `auto_focus`, then drawn there;
    * *in situ*: learnt from the movies at ``paths`` with ``camera`` (`insitu_focus`), started where *auto*
      puts it when there are ``images``.

    The spline records how its focus was found (``parameters["sample_depth"]["method"]``).  Without torch,
    *auto* is said through ``warn`` and falls back to the calibration as it is, as Spline 3D's does: the
    default must not stop work that ran before; the others refuse."""
    if cal is None or not has_model(cal) or sample.focus == "calibrated":
        return cal, None
    try:
        import torch  # noqa: F401 -- only asked whether it is there
    except ImportError:
        if sample.focus == "auto":
            (warn or report)("focus on auto needs torch to draw the PSF in the sample (pip install "
                             "smappy-smlm[uipsf]): the calibration as it is, z above the beads' focus")
            return cal, None
        raise ValueError("a focus in the sample draws the PSF with uipsf, which needs torch: pip install "
                         "smappy-smlm[uipsf]")
    n_med = sample.n_med()
    found = None
    if sample.focus == "typed":
        spline = at_depth(cal, sample.focus_nm, n_med)
    elif sample.focus == "auto":
        if images is None or not len(images):
            raise ValueError("focus on auto needs spots from the movie: none were found; type the focus")
        found = auto_focus(cal, in_beads_orientation(images, cal, getattr(camera, "em_on", None)), n_med,
                           report=report)
        spline = at_depth(cal, found.focus_nm, n_med)
    elif sample.focus == "insitu":
        if not paths or camera is None:
            raise ValueError("a focus in situ is learnt from the movie: choose it")
        start = None
        if images is not None and len(images):
            try:
                start = auto_focus(cal, in_beads_orientation(images, cal, getattr(camera, "em_on", None)), n_med,
                                   report=report).focus_nm
            except ValueError:
                pass
        found = insitu_focus(cal, paths, camera, n_med, start, progress=report, cancel=cancel)
        spline = found.spline
    else:
        raise ValueError(f"focus must be one of {[k for k, _ in SAMPLE_FOCUS_CHOICES]}, not {sample.focus!r}")
    if found is not None:
        for message in found.messages:
            (warn or report)(f"focus {found.method}: {message}")
    # a copy: `at_depth` keeps the spline it hands out, and a typed and an auto focus share it
    import copy
    spline = copy.copy(spline)
    spline.parameters = dict(spline.parameters or {})
    record = dict(spline.parameters.get("sample_depth") or {})
    record.setdefault("method", {"typed": "typed", "auto": "auto", "insitu": "in situ"}[sample.focus])
    spline.parameters["sample_depth"] = record
    return spline, found


def z_bounds(cal) -> tuple:
    """(lowest, highest) z in nm a calibration's spline holds, by its convention emitter = -(index - z0) dz:
    the plane 0 the highest.  A drawn spline's z is the height above the coverslip and is not symmetric
    about 0; a bead spline's is about its focus."""
    nz = cal.shape[0]
    return float(-(nz - 1 - cal.z0) * cal.dz), float(cal.z0 * cal.dz)


def describe_sample(cal) -> str:
    """One line on where a spline is drawn for: '' for one at the coverslip (a bead calibration)."""
    record = (getattr(cal, "parameters", None) or {}).get("sample_depth") or {}
    return describe_record(record)


def describe_record(record: dict) -> str:
    if not record or record.get("focus_nm") is None:
        return ""
    text = f"focus {float(record['focus_nm']):.0f} nm into the sample"
    if record.get("method"):
        text += f" ({record['method']})"
    if record.get("n_med") is not None:
        text += f", n = {float(record['n_med']):g}"
    return text + ", z the height above the coverslip"
