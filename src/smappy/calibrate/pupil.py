"""Bead stacks to what uipsf learns a pupil from: smappy reads, uipsf fits.

uipsf (`src/uipsf/`, uiPSF v2) takes arrays: ROIs in photons with their z,
noise, saturation and chip positions (a `uipsf.Batch`).  Everything before
that is smappy's and exists once (`docs/uipsf/plan.md` §7.5):

* the stacks and their z from `input.read_bead_stacks`, as the bead
  calibration reads them;
* the camera from the camera database, as the fitter resolves it
  (`camera_for`), and photons by the one `camera.to_photons`; the read
  noise and the EM excess from `CameraMetadata` too, so the likelihood's
  noise is the fitter's;
* bead centres by `core.detect_beads` and, for a split chip, the pairs, the
  layout and the starting transform by `dual.collect_dual_beads` and
  `fit_dual_transform`, the dual calibration's own.

What differs from the bead average, and why:

* **Raw ROIs, not normalised volumes.**  The bead average subtracts each
  bead's background, divides by its brightness and aligns it; a likelihood
  fit needs the counts as the camera recorded them, so each ROI is cut
  around the detection's whole pixel and converted to photons, nothing
  more.  The fit places each bead itself.
* **Saturated pixels are masked, not the bead dropped.**  Of the bead
  average's brightness verdict only *too dim* is applied, before learning:
  on the Zenodo beads 22 of the 84 detections were faint specks, 0.13-0.27
  of the median bead where every real one was 0.43 or more, whose fits are
  noise and which uipsf's rejection then threw out one by one.  *Too bright*
  is left to the fit (a doublet shows in its deviance; a bright single bead
  is the best there is).  The verdict is relative to the other beads; a pair
  of two channels is judged by its light in both together, so a ratiometric
  pair's dim half does not drop it.  With *brightness range* on auto nothing
  is dropped.
* **The first slice is dropped** by default: on the Zenodo stacks it is 22 %
  brighter than the rest, shared by all beads (`docs/uipsf/architecture.md`,
  M3), and uiPSF v1 drops it too.  Stacks of different lengths are cut to
  the shortest about their middles, as `collect_beads` does; each stack's
  nominal focus is the middle of the slices kept.

torch is imported inside the functions: this module is imported by the
calibration window without torch installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import numpy as np

from ..plugins import param
from .core import (CalibrationSettings, brightness_centre, classify_brightness, doubles, find_peaks,
                   judge_shapes, measure_peaks, saturation_of, select_beads)
from .input import BeadStack, discover_acquisitions, read_bead_stacks

KEEP = slice(1, None)            # every slice but the first (module docstring)

#: the coverslip an objective is corrected for unless its microscope says otherwise (`coverslip_design_um`), µm
DESIGN_COVERSLIP_UM = 170.0

#: the order a calibration learns up to unless told: n_max 6, with 8 the held-out check (plan §10)
AUTO_ORDER = 6

MAGNITUDES = (("minimal", "minimal (cos θ^γ)"), ("free", "Zernike"), ("symmetric", "symmetric Zernike"),
              ("fixed", "fixed (apodisation)"))
BLURS = (("free", "free"), ("round", "round"), ("fixed", "fixed"))
#: the stage's drift through a stack (`uipsf.model.geometry`): none, a lateral one growing linearly through
#: each stack, or a free one per slice
DRIFTS = (("none", "none"), ("linear", "linear per stack"), ("free", "free per slice"))


@dataclass
class PSFModelSettings:
    """The PSF model's choices, one section wherever a model is learnt -- beads in one or two channels, in situ,
    4Pi -- and what uipsf's builder reads (`uipsf.model.build`): the pupil's order and magnitude, the blur.
    Each workflow keeps its own fields beside it (`BeadSettings`, `InSituModelSettings`), and the form hides
    what one does not read (`model_shown`)."""
    zernike_order: int | None = param(
        None, label="Zernike order", min=2, max=12,
        help=f"the highest radial order of the pupil's modes; auto: {AUTO_ORDER}")
    # the minimal model by default (the user's, November 2026, on M5's beads: `uipsf.model.MinimalPupil`)
    magnitude: str = param(
        "minimal", label="pupil magnitude", choices=MAGNITUDES,
        help="minimal: cos θ^γ, the ring beyond the critical angle and the NA, a few physical numbers; "
             "Zernike: every mode up to the order; symmetric: the rotationally symmetric ones; "
             "fixed: uniform, the apodisation only")
    # round by default everywhere, in situ too (the user's, November 2026): free took up astigmatism the pupil
    # should hold; to revisit for in situ once it is optimised.  Under *more*: learnt round is right almost
    # always, and the output page reports what it came to
    blur: str = param("round", label="blur", choices=BLURS, advanced=True,
                      help="a Gaussian blur of the image beyond the pixel's own, learnt (round: one σ for x "
                           "and y) or held at its value")
    blur_nm: float = param(20.0, label="blur σ", unit="nm", min=0.0, advanced=True,
                           help="the blur beyond the pixel's: its start when learnt, its value when fixed")
    # the pixel's blur under the learnt one: the model is drawn at pixel centres (`VectorPSF.pixel_blur`), the
    # camera integrates over the pixel.  Not in the form (the user's, November 2026: no case for off); kept
    # for scripts that reproduce models learnt without it
    pixel_blur: bool = param(True, label="pixel blur", hidden=True,
                             help="the pixel's own blur, its size / √12, under the learnt one, so that σ is "
                                  "what blurs the image beyond the pixel")
    # the minimal magnitude's supercritical-angle ring damped by a learnt exp(−2πκh) (`MinimalPupil`): the
    # beads' ring is weaker than a point's on the coverslip; a spline for single molecules is drawn without it,
    # and in situ never learns it
    saf_damping: bool = param(True, label="SAF damping", advanced=True,
                              help="minimal magnitude: learn a damping of the supercritical-angle ring, the "
                                   "light beyond the critical angle; off: the ring of a point on the coverslip")
    # the same ring on a Zernike magnitude (`uipsf.model.build.make_pupil`).  Not in the form (the user's, November
    # 2026): on M5's beads h traded with the high radial modes and the spline drawn without it compressed z by
    # up to 17 %; kept for scripts that study it (benchmarks/uipsf/real/m5_psf_model.py)
    zernike_ring: bool = param(False, label="SAF damping on Zernikes", hidden=True,
                               help="a Zernike magnitude's ring damped as a minimal one's, with SAF damping on")
    # the aperture's NA learnt (`MinimalPupil.learn_na`): an option, not the default (the user's, November 2026) --
    # the 4Pi beads wanted 0.97 of the objective's NA and one objective's 0.98, which is not yet understood
    learn_na: bool = param(False, label="learn NA", advanced=True,
                           help="minimal magnitude: learn the aperture's NA, an effective one at most 5 % above "
                                "the objective's; off: the objective's NA as typed")
    # the sample's index learnt (`uipsf.VectorPSF.learn_n_med`): a diagnostic, off by default -- a buffer's index is
    # measured only when it is matched (docs/4pi.md §24)
    learn_n_med: bool = param(False, label="learn sample index", advanced=True,
                              help="learn the sample's refractive index from the microscope's *sample index*, "
                                   "a diagnostic; off: as typed")

    def __post_init__(self):
        if self.magnitude not in dict(MAGNITUDES):
            raise ValueError(f"magnitude must be one of {[k for k, _ in MAGNITUDES]}, got {self.magnitude!r}")
        if self.blur not in dict(BLURS):
            raise ValueError(f"blur must be one of {[k for k, _ in BLURS]}, got {self.blur!r}")

    @property
    def order(self) -> int:
        return AUTO_ORDER if self.zernike_order is None else int(self.zernike_order)


@dataclass
class BeadSettings:
    """The choices a bead calibration exposes, one channel or two; the instrument comes as `Optics`."""
    psf: PSFModelSettings = param(default_factory=PSFModelSettings, label="PSF model")
    bead_diameter_nm: float = param(0.0, label="bead diameter", unit="nm", min=0.0,
                                    help="the beads' size, taken out of the model; 0: point-like")
    # docs/uipsf/field-dependence.md: astigmatism and coma varying over the chip, as a polynomial each
    field_dependence: bool = param(False, label="field dependence",
                                   help="learn astigmatism and coma that vary over the field of view; the spline "
                                        "is then one per point of a grid over the chip, blended per spot")
    # linear by default (November 2026): on M5's sixteen sets it gave the PSF a free drift per frame gives, coma
    # within 0.5 nm, σ 0.1 nm, γ 0.01, where none biased the coma by up to 5 nm
    drift: str = param("linear", label="drift", choices=DRIFTS, advanced=True,
                       help="a lateral drift of the stage through each stack, learnt: linear per stack, or "
                            "free per slice")
    # offered for beads only through a coverslip of another index (`learns_tilt`), always in situ
    fit_tilt: bool = param(False, label="learn coverslip tilt", advanced=True,
                           help="learn the coverslip's tilt against the optical axis, mostly a coma: in situ, "
                                "or beads seen through a coverslip of another index than the immersion's")
    rejection_k: float = param(3.0, label="rejection (k)", min=1.0, advanced=True,
                               help="robust standard deviations above the median at which a bead is "
                                    "rejected")
    heldout: float = param(0.0, label="held out", min=0.0, max=0.5, advanced=True,
                           help="the fraction of beads kept out of learning and localised for the "
                                "held-out report")
    # uipsf's M7: at most 50 nm apart loses phase precision only as the CRLB says, for a fraction of the time
    max_dz_nm: float = param(50.0, label="largest z step", unit="nm", min=1.0, advanced=True,
                             help="learn from every n-th slice, n the most whose step stays within this")
    # ±1.5 µm by default (November 2026): the far slices of a ±2 µm stack are mostly background
    z_range_nm: Optional[float] = param(1500.0, label="learn z range", unit="nm", min=100.0, advanced=True,
                                        help="± about focus learnt from, a shorter stack whole; auto: the whole "
                                             "stack")

    def __post_init__(self):
        if self.drift not in dict(DRIFTS):
            raise ValueError(f"drift must be one of {[k for k, _ in DRIFTS]}, got {self.drift!r}")

    @classmethod
    def from_options(cls, options: dict) -> "BeadSettings":
        """The settings a model file kept (`Calibration.options`, written by `asdict`: the PSF model nested)."""
        from ..plugins import settings_from
        options = dict(options or {})
        options.setdefault("z_range_nm", None)      # a model of before the range learnt from the whole stack
        return settings_from(cls, options)

    @property
    def order(self) -> int:
        return self.psf.order


def model_shown(method: str, magnitude: str = "minimal", dual: bool = False, layered: bool = False,
                ring: bool = True) -> dict:
    """Which of `BeadSettings`' fields a method's form shows (`FormState.shown`, dotted for the PSF part):
    ``method`` "beads", "insitu" or "molecules"; ``magnitude`` the chosen one; ``dual`` two channels; ``layered`` a
    coverslip of another index than the immersion's; ``ring`` whether the sample's index is below the NA, so
    that there is supercritical light to damp.

    The sample's index is learnt on request in every method (*learn sample index*, a diagnostic).
    In situ decides the magnitude by its *hold* (a bead model's γ held, or γ learnt) and never learns the
    SAF damping, the bead's size or the stage's drift, which are the beads'.  *learn NA* is offered in every
    method for a minimal magnitude; in situ holding a bead model's γ holds its learnt NA too.  Field
    dependence is for one channel of beads, for now (docs/uipsf/field-dependence.md).  The tilt: always in
    situ, where molecules above the glass see it; for beads only through a coverslip of another index.
    Molecules on the coverslip in z sweeps choose the magnitude as beads do, but have no SAF damping of their
    own -- their common height damps the ring physically (`uipsf.model.CoverslipStackGeometry`) -- no bead,
    no drift (each molecule is one frame), no field dependence and no tilt (a point at the glass sees none
    through oil), and nothing held out."""
    if method == "molecules":
        return {"psf.magnitude": True, "psf.saf_damping": False, "psf.learn_na": magnitude == "minimal",
                "psf.learn_n_med": True, "bead_diameter_nm": False, "drift": False, "field_dependence": False,
                "fit_tilt": False, "heldout": False, "max_dz_nm": False, "z_range_nm": False}
    beads = method != "insitu"
    return {"psf.magnitude": beads, "psf.saf_damping": beads and magnitude == "minimal" and ring,
            "psf.learn_na": magnitude == "minimal", "psf.learn_n_med": True,
            "bead_diameter_nm": beads, "drift": beads, "field_dependence": beads and not dual,
            "fit_tilt": not beads or layered, "heldout": True, "max_dz_nm": beads, "z_range_nm": beads}


@dataclass
class BeadROIs:
    """What uipsf learns from, and where it came from."""
    batch: object                        # uipsf.Batch [E, C, K, Y, X], photons
    noise: object                        # uipsf.ShiftedPoisson with the camera's excess noise
    camera: object                       # CameraMetadata, as resolved
    pixel_size_nm: tuple                 # (x, y) on the sample
    records: list                        # per emitter: stack, x_px, y_px (full image) and per channel
    sources: list                        # per stack: path, axes, roi, shape, z_nm
    transform: Optional[np.ndarray] = None   # secondary -> main chip pixels, H22 = 1: the start (two channels)
    geometry: Optional[dict] = None          # the split, as `DualColorCalibration.geometry`
    points: Optional[np.ndarray] = None      # [E, C, 2] sub-pixel chip positions of a pair (two channels)
    messages: list = field(default_factory=list)


def read_stacks(inputs, dz_nm=None) -> list:
    """`BeadStack`s from paths (acquisitions, discovered as the bead calibration does) or as given."""
    if isinstance(inputs, (BeadStack, str, Path)):
        inputs = [inputs]
    inputs = list(inputs)
    if not inputs:
        raise ValueError("select bead acquisitions")
    if isinstance(inputs[0], BeadStack):
        return inputs
    return [st for p in discover_acquisitions(inputs) for st in read_bead_stacks(p, dz_nm)]


def camera_for(path, camera_settings=None):
    """The camera of the acquisition at ``path`` as a fit of it would resolve it (camera database, the
    file's metadata, then ``camera_settings``' fields): complete, or a ValueError naming what is
    missing."""
    from ..io.tiff import open_stack
    from ..plugins.fit import CameraSettings
    settings = camera_settings if camera_settings is not None else CameraSettings()
    return settings.resolve(open_stack(path))


def _microscope_choices():
    from ..microscope_db import choices
    return choices()


def _camera_choices():
    from ..plugins.fit import _camera_choices as cameras
    return cameras()


def main_half(rois) -> Optional[str]:
    """The half of the chip the dual calibration made its main channel, or None for one channel."""
    return (getattr(rois, "geometry", None) or {}).get("main_channel")


def pixel_size_nm(camera) -> tuple:
    p = camera.pixelsize_um
    if p is None:
        raise ValueError("the camera has no pixel size: add it to the camera database or give it")
    if isinstance(p, (list, tuple, np.ndarray)):
        return float(p[0]) * 1000, float(p[-1]) * 1000
    return float(p) * 1000, float(p) * 1000


def _saturation(stack, saturation_adu) -> Optional[float]:
    if saturation_adu is not None:
        return float(saturation_adu)
    dtype = getattr(stack.images, "dtype", None) or np.asarray(stack.images).dtype
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else None


def _cut(stacks, centres, roi_size, slices, camera, saturation_adu=None):
    """ROIs around ``centres`` [(stack, x, y), ...] (whole pixels of the full image): photons [E, K, Y, X],
    saturation [E, K, Y, X] and objective z [E, K], every stack cut to the same slices."""
    from ..camera import to_photons
    half = roi_size // 2
    planes = [np.arange(len(st.z_nm))[slices] for st in stacks]
    nz = min(len(p) for p in planes)
    planes = [p[(len(p) - nz) // 2:(len(p) - nz) // 2 + nz] for p in planes]
    for s, x, y in centres:
        st = stacks[s]
        if not (half <= y < st.images.shape[-2] - half and half <= x < st.images.shape[-1] - half):
            raise ValueError(f"bead at ({x}, {y}) in {st.source}: its ROI leaves the image")
    data, saturated, z = [None] * len(centres), [None] * len(centres), [None] * len(centres)
    # stack by stack, each read once for all its beads and released before the next: a stack from the
    # files (`input.Planes`) is read whole by indexing, and once per bead it was read 40 times over
    for s in sorted({c[0] for c in centres}):
        st, kept = stacks[s], planes[s]
        images = np.asarray(st.images[kept])
        limit = _saturation(st, saturation_adu)
        objective = np.asarray(st.z_nm, float)[kept]
        for i, (_, x, y) in enumerate(centres):
            if centres[i][0] != s:
                continue
            adu = images[:, y - half:y - half + roi_size, x - half:x - half + roi_size]
            saturated[i] = adu >= limit if limit is not None else np.zeros(adu.shape, bool)
            data[i] = to_photons(adu, camera)
            z[i] = objective - objective.mean()         # about the middle: the nominal focus
        del images
    return np.stack(data), np.stack(saturated), np.stack(z)


def _batch(data, saturated, z, camera, field_xy, stack_id, roi_xy=None):
    """``data`` [E, C, K, Y, X] photons into a `uipsf.Batch` with the camera's noise."""
    import torch

    from uipsf import Batch, ShiftedPoisson
    z = torch.tensor(z[0] if np.allclose(z, z[0], atol=1e-3) else z, dtype=torch.float32)
    batch = Batch(data=torch.tensor(data, dtype=torch.float32), z_nm=Batch.z_from_objective(z),
                  s=float(camera.readout_variance), mask=~torch.tensor(saturated),
                  field_xy=torch.tensor(np.asarray(field_xy, float), dtype=torch.float32),
                  stack_id=torch.tensor(np.asarray(stack_id, int)),
                  roi_xy=None if roi_xy is None else torch.tensor(np.asarray(roi_xy, float), dtype=torch.float32))
    return batch, ShiftedPoisson(float(camera.excess_noise))


def _sources(stacks):
    return [{"path": str(st.source), "axes": st.axes, "roi": None if st.roi is None else list(st.roi),
             "shape": list(st.images.shape), "z_nm": np.asarray(st.z_nm, float).tolist()} for st in stacks]


def _unit(h) -> np.ndarray:
    """A projective map scaled to H₂₂ = 1, as uipsf and `DualColorCalibration` read it; smappy's fits leave
    the scale free."""
    h = np.asarray(h, float)
    return h / h[2, 2]


def _bright_enough(brightness, saturated, brightness_range) -> list:
    """The indices not *too dim*: at least the median of the unsaturated ones divided by
    √``brightness_range``, the bead average's lower bound (`core.classify_brightness`); all with no range."""
    brightness = np.asarray(brightness, float)
    unsaturated = ~np.asarray(saturated, bool)
    if brightness_range is None or not unsaturated.any():
        return list(range(len(brightness)))
    lower = float(np.median(brightness[unsaturated])) / np.sqrt(brightness_range)
    return [i for i, b in enumerate(brightness) if b >= lower]


def _origin(st) -> np.ndarray:
    return np.array(st.roi[:2] if st.roi is not None else (0, 0), float)


def rois_at(stacks, centres, camera, roi_size: int, slices=KEEP, saturation_adu=None) -> BeadROIs:
    """One channel, ROIs around given ``centres`` [(stack, x, y), ...], whole pixels of each stack's image:
    beads chosen elsewhere (by hand, or another program's list).  Saturation from the images' integer
    type unless ``saturation_adu``."""
    centres = [(int(s), int(x), int(y)) for s, x, y in centres]
    if not centres:
        raise ValueError("no beads to cut")
    data, saturated, z = _cut(stacks, centres, roi_size, slices, camera, saturation_adu)
    field_xy = [_origin(stacks[s]) + (x, y) for s, x, y in centres]
    batch, noise = _batch(data[:, None], saturated[:, None], z, camera, field_xy, [c[0] for c in centres])
    records = [{"id": i, "stack": s, "x_px": x, "y_px": y} for i, (s, x, y) in enumerate(centres)]
    return BeadROIs(batch, noise, camera, pixel_size_nm(camera), records, _sources(stacks))


def single_rois(stacks, camera, settings: CalibrationSettings | None = None, slices=KEEP) -> BeadROIs:
    """One channel: the beads the bead average would find (`core.find_stack_beads`) less the *too dim*
    ones (module docstring), a `settings.roi_size` ROI each, with the same records: position,
    background, brightness and its verdict, saturation."""
    settings = settings or CalibrationSettings()
    settings.validate()
    channels = {str(st.axes.get("channel", 0)) for st in stacks}
    if len(channels) > 1:
        raise ValueError("the acquisitions hold several channels; calibrate them one at a time")
    candidates = []
    for st in stacks:                       # one stack in memory at a time (`input.Planes`)
        images = np.asarray(st.images, np.float32)
        yx, _, heights, merged = find_peaks(images, settings)
        candidates.append(measure_peaks(images, yx, settings, saturation_of(st.images, settings), heights,
                                        merged))
        del images
    # isolated, over all stacks at once; a double by its shape is marked to be left out (`judge_shapes`)
    found = judge_shapes(select_beads(candidates, settings), settings)
    if not found:
        raise ValueError("no isolated beads found: review detection and the bead box")
    # specks out before the verdict, so a crowd of them cannot drag its median down to their level
    _, speck = brightness_centre(found, settings.brightness_range)
    beads = [r for r in found if speck is None or r["saturated"] or r["brightness_adu"] >= speck]
    classify_brightness(beads, settings)
    kept = [r for r in beads if r["brightness_reason"] != "too dim"]
    if not kept:
        raise ValueError("every bead found is too dim beside the others: review detection and brightness range")
    rois = rois_at(stacks, [(r["stack"], r["x_px"], r["y_px"]) for r in kept], camera, settings.roi_size, slices,
                   settings.saturation_adu)
    rois.records = [dict(r, id=i) for i, r in enumerate(kept)]
    rois.messages.append(f"{len(found) - len(kept)} of {len(found)} spots too dim beside the others: not beads")
    return rois


def dual_rois(stacks, camera, settings=None, slices=KEEP) -> BeadROIs:
    """Two channels on one chip: the pairs `collect_dual_beads` finds, each channel's ROI cut around its
    own detection in the camera's orientation (no flip: the transform carries a mirror), the start
    transform from `fit_dual_transform`."""
    from .dual import DualColorSettings, collect_dual_beads, fit_dual_transform
    settings = settings or DualColorSettings()
    beads = collect_dual_beads(stacks, settings)
    # the transform from every pair, as the dual calibration fits it; the pairs to learn from without the
    # ones too dim, judged by the pair's light in both channels together: a ratiometric pair sends little
    # to one channel, and judging each channel apart dropped real pairs for their dim half
    fitted = fit_dual_transform(beads.secondary_points, beads.main_points, beads.settings.registration)
    keep = _bright_enough([sum(part["brightness_adu"] for part in r["channel_records"]) for r in beads.records],
                          [any(part["saturated"] for part in r["channel_records"]) for r in beads.records],
                          settings.brightness_range)
    if not keep:
        raise ValueError("every pair is too dim in a channel beside the others: review detection and brightness "
                         "range")
    pairs = [beads.records[i] for i in keep]
    for i, r in zip(keep, pairs):
        r["pair_brightness_adu"] = sum(part["brightness_adu"] for part in r["channel_records"])
    per_channel = []
    for ch in (0, 1):
        per_channel.append([(r["channel_records"][ch]["stack"],) + tuple(int(round(v)) for v in
                            r["channel_records"][ch]["full_image_xy"]) for r in pairs])
    cuts = [_cut(stacks, centres, settings.roi_size, slices, camera, settings.saturation_adu)
            for centres in per_channel]
    data = np.stack([c[0] for c in cuts], 1)
    saturated = np.stack([c[1] for c in cuts], 1)
    roi_xy = np.stack([[_origin(stacks[s]) + (x, y) for s, x, y in centres] for centres in per_channel], 1)
    batch, noise = _batch(data, saturated, cuts[0][2], camera, roi_xy[:, 0], [c[0] for c in per_channel[0]],
                          roi_xy)
    records = [{"id": i, "stack": m[0], "x_px": m[1], "y_px": m[2],
                "channels": [{"x_px": m[1], "y_px": m[2]}, {"x_px": sc[1], "y_px": sc[2]}]}
               for i, (m, sc) in enumerate(zip(*per_channel))]
    messages = [f"{int((~fitted.accepted).sum())} of {len(fitted.accepted)} pairs off the starting transform",
                f"{len(beads.records) - len(keep)} of {len(beads.records)} pairs too dim, both channels "
                f"together: not used"]
    return BeadROIs(batch, noise, camera, pixel_size_nm(camera), records, beads.sources,
                    _unit(fitted.transformation), dict(beads.geometry),
                    np.stack([beads.main_points[keep], beads.secondary_points[keep]], 1), messages)


def bead_rois(inputs, camera=None, dual: bool = False, settings=None, slices=KEEP, camera_settings=None,
              dz_nm=None) -> BeadROIs:
    """`BeadROIs` from bead acquisitions (paths) or `BeadStack`s: one channel, or a split chip with
    ``dual``.  ``camera`` (a `CameraMetadata`) defaults to the first acquisition's, as a fit of it
    would resolve it (`camera_for`)."""
    stacks = read_stacks(inputs, dz_nm if dz_nm is not None else getattr(settings, "dz_nm", None))
    if camera is None:
        paths = [Path(i) for i in ([inputs] if isinstance(inputs, (str, Path)) else inputs)
                 if isinstance(i, (str, Path))]
        if not paths:
            raise ValueError("give the camera for stacks that are not files")
        camera = camera_for(discover_acquisitions(paths)[0], camera_settings)
    return (dual_rois if dual else single_rois)(stacks, camera, settings, slices)


# ------------------------------------------------------------------ the spline from a learnt model

#: each pixel of the spline's knots sampled as the model was learnt (None: its own sub-samples, one by default).
#: Not integrated more finely: learning on pixel centres puts the pixel's width into the learnt blur (σ 35 nm
#: learnt as 47.3 on beads drawn with their pixels integrated, √(35² + 110²/12) = 47.4), and integrating
#: that model again counted the pixel twice -- Spline 3D's z slope 0.992, 6 nm rms off uipsf's own
#: localisation of the same ROIs, against 0.999 and 2.8 nm at the model's own sampling
#: (docs/uipsf/architecture.md, "M6: the chain's acceptance").
UPSAMPLE = None


def z_grid(dz_nm: float, z_range_nm: float) -> tuple[np.ndarray, float]:
    """(emitter z of each plane, z0): planes ``dz_nm`` apart through ±``z_range_nm``, in the spline's
    convention z = −(index − z0)·dz, so plane 0 is the highest emitter, as in a bead stack."""
    half = int(np.ceil(abs(z_range_nm) / dz_nm))
    return -(np.arange(2 * half + 1) - half) * float(dz_nm), float(half)


def sample(learned, dz_nm: float, z_range_nm: float, roi_size: int = 27, upsample: int | None = UPSAMPLE,
           channel: int = 0, field_xy=None, depth_nm: float | None = None) -> np.ndarray:
    """The learnt PSF of ``channel`` as a voxel stack [z, y, x] for a point emitter on the central pixel,
    each pixel as the model sees it (`UPSAMPLE`): what a bead average is, without its noise.  ``learned`` is a uipsf
    `Calibration` or `LearnedPSF`; the bead's size is left out, the learnt blur kept, the channel's
    photon share and defocus offset included (`uipsf.export.sample_psf`).  ``depth_nm``, the emitter's height
    above the glass (molecules on the coverslip, `molecule_stack_run`); a point at the glass without."""
    z, _ = z_grid(dz_nm, z_range_nm)
    xyz = np.c_[np.zeros((len(z), 2)), z]
    where = {} if field_xy is None else dict(field_xy=field_xy)      # a field-dependent model: where on the chip
    if depth_nm is not None:
        where["depth_nm"] = float(depth_nm)
    return learned.psf(xyz, roi_px=roi_size, upsample=upsample, channel=channel, **where).double().numpy()


def _zernikes(learned) -> list:
    """Every channel's Zernikes, (Noll, coefficient, CRLB), for the calibration's record."""
    from uipsf.diagnostics import zernikes
    out = []
    for c in range(learned.model.n_channels):
        entry = {}
        for which in ("phase", "magnitude"):
            modes, values, crlb = zernikes(learned, which, c)
            entry[which] = [[int(j), float(v), float(e)] for j, v, e in zip(modes, values, crlb, strict=True)]
        out.append(entry)
    return out


def _upsample(learned, upsample) -> int:
    """The sub-samples per pixel axis a sampling used: ``upsample``, or the model's own."""
    return int(learned.model.channels[0].psf.upsample if upsample is None else upsample)


def _record(learned, dz_nm, z_range_nm, roi_size, upsample) -> dict:
    options = getattr(learned, "options", None)
    if options is not None and not isinstance(options, dict):
        from dataclasses import asdict
        options = asdict(options)
    from uipsf.diagnostics import coverslip_tilt
    tilt = coverslip_tilt(learned)
    extra = {} if tilt is None else {"coverslip_tilt_deg": [[float(v), float(e)] for v, e in zip(*tilt, strict=True)]}
    return {"method": "uipsf2_pupil", "zernikes": _zernikes(learned), "options": options or {}, **extra,
            "sampling": {"dz_nm": float(dz_nm), "z_range_nm": float(z_range_nm), "roi_size": int(roi_size),
                         "upsample": _upsample(learned, upsample)},
            "z_convention": "emitter_nm = -(index-z0)*dz", "lateral_unit": "pixel",
            "z_reference": "the model's focus: the emitter at z = 0"}


def spline_calibration(learned, dz_nm: float, z_range_nm: float, roi_size: int = 27, upsample: int | None = UPSAMPLE,
                       camera=None, depth_nm: float | None = None):
    """smappy's `SplineCalibration` of a one-channel uipsf model, normalised as the bead average's
    (`core._single_model`: clipped at zero, the brightest plane summing to one, no smoothing, since a
    model has no noise to smooth), so that Spline 3D reads either alike.  ``camera`` records its EM
    state and pixel size, as the bead average does.  ``depth_nm`` as `sample`'s."""
    from ..io.calibration import SplineCalibration
    from .core import spline_coefficients
    if learned.model.n_channels != 1:
        raise ValueError("a model of several channels makes a dual calibration (`dual_calibration`)")
    _, z0 = z_grid(dz_nm, z_range_nm)
    psf = np.maximum(sample(learned, dz_nm, z_range_nm, roi_size, upsample, depth_nm=depth_nm), 0)
    scale = float(psf.sum(axis=(1, 2)).max())
    psf /= scale
    field = field_splines(learned, dz_nm, z_range_nm, roi_size, upsample, scale)
    parameters = dict(_record(learned, dz_nm, z_range_nm, roi_size, upsample),
                      normalization="maximum plane sum = 1")
    psf = psf.astype(np.float32)
    if getattr(camera, "microscope", None):
        parameters["microscope"] = dict(camera.microscope)
    if field is not None:
        # ``psf`` is the field's centre, the middle of the beads' span and so of the grid: the spline a
        # reader that ignores the field fits with
        parameters["field_grid"] = list(field.shape)
        parameters["field_blend_error"] = float(field.blend_error)
        from ..field_splines import limit_warning
        warning = limit_warning(field.shape[0], field.blend_error)
        if warning is not None:
            # kept with the calibration, so the window's summary and a later reader see it too
            parameters["field_warning"] = warning
            import warnings
            warnings.warn(warning, stacklevel=2)
    return SplineCalibration(spline_coefficients(psf), float(dz_nm), z0, x0=(roi_size - 1) / 2, psf=psf,
                             em_on=None if camera is None else bool(camera.is_em),
                             pixelsize_um=None if camera is None or camera.pixelsize_um is None
                             else np.atleast_1d(np.asarray(camera.pixelsize_um, float)),
                             parameters=parameters, field_splines=field)


def field_extent(learned):
    """((x, y) lowest, (x, y) highest) chip pixels of the beads a field-dependent model was learnt on: where
    its field is known, and so the span of its spline grid; None without a field or bead positions."""
    if all(getattr(ch, "field", None) is None for ch in learned.model.channels):
        return None
    if callable(getattr(learned, "beads_field_xy", None)):
        xy = learned.beads_field_xy()
    else:                                       # a loaded model file: its bead table
        xy = [[b["field_x"], b["field_y"]] for b in learned.beads
              if b.get("state") == "used" and "field_x" in b]
    if not xy:
        return None
    xy = np.asarray(xy, float)
    return xy.min(0), xy.max(0)


def field_splines(learned, dz_nm: float, z_range_nm: float, roi_size: int, upsample, scale: float):
    """The node splines of a field-dependent model (`smappy.field_splines`), on the smallest grid over the
    beads' span whose blend keeps to `field_splines.TOLERANCE`, every node normalised by the centre's
    ``scale`` so that the blend is one; None without a field, or when one spline is enough."""
    from ..field_splines import FieldSplines, choose_grid, grid_nodes
    from .core import spline_coefficients
    extent = field_extent(learned)
    if extent is None:
        return None
    lo, hi = extent
    cache = {}

    def psf_at(x, y):
        key = (round(float(x), 3), round(float(y), 3))
        if key not in cache:
            cache[key] = np.maximum(sample(learned, dz_nm, z_range_nm, roi_size, upsample, field_xy=key), 0) / scale
        return cache[key]

    n, error = choose_grid(psf_at, lo, hi)
    if n == 1:
        return None
    xs, ys = grid_nodes(lo[0], hi[0], n), grid_nodes(lo[1], hi[1], n)
    coeff = np.stack([spline_coefficients(psf_at(x, y).astype(np.float32)) for y in ys for x in xs])
    return FieldSplines(coeff, xs, ys, blend_error=error)


def _transform(learned) -> np.ndarray:
    if callable(getattr(learned, "transform", None)):
        return np.asarray(learned.transform(1), float)
    from uipsf.model.channels import unpack_transform
    return _unit(unpack_transform(learned.shared()["ch1.transform"].detach().cpu().double()).numpy())


def dual_calibration(learned, dz_nm: float, z_range_nm: float, roi_size: int = 27,
                     upsample: int | None = UPSAMPLE, depth_nm: float | None = None):
    """smappy's `DualColorCalibration` of a two-channel uipsf model: both channels sampled on one z grid
    (a biplane offset is in the PSFs), normalised jointly by `positive_pair_models`, so each channel's
    `photon_normalization` is its learnt photon share; the model's transform and the split it was
    learnt on.  Spline 3D 2C reads it as one of its own (tests/uipsf_tests/test_smappy_dual.py).  ``depth_nm``
    as `sample`'s."""
    from .core import positive_pair_models
    from .dual import DualColorCalibration
    if learned.model.n_channels != 2:
        raise ValueError("a dual calibration needs a model of two channels")
    if learned.geometry is None:
        raise ValueError("the model does not say how the chip was split; learn it from dual_rois")
    raw = np.stack([sample(learned, dz_nm, z_range_nm, roi_size, upsample, c, depth_nm=depth_nm) for c in range(2)])
    settings = CalibrationSettings(roi_size=roi_size, smooth_z_nm=0.0, smooth_xy_px=0.0)
    (main, secondary), _ = positive_pair_models(raw, dz_nm, settings)
    record = _record(learned, dz_nm, z_range_nm, roi_size, upsample)
    for model in (main, secondary):
        model.parameters.update(record)
    return DualColorCalibration(main, secondary, _transform(learned), dict(learned.geometry), dict(record))


def objective_pupil(cal, source: str = "") -> dict:
    """What a learnt model says about the objective, to keep with the microscope (`microscope_db.
    keep_pupil`): the main channel's PSF as it describes itself (`uipsf.VectorPSF.describe`) -- its kind,
    every phase mode in nm of wavefront, so that it holds at another emission wavelength, γ (a minimal
    pupil) or the magnitude's Zernikes, σ and whether it is beyond the pixel, the optics -- and where it came
    from.  In situ holds from it (`uipsf.model.build.held_from`) and starts from it."""
    import datetime as _dt
    model = cal.model
    psf = model.channels[0].psf if hasattr(model, "channels") else model.psf
    # every mode, the held ones among them (`ZernikePupil.coefficients`): a pupil held from the microscope's
    # and kept again is the whole pupil, not its learnt part
    return dict(psf.describe(cal.shared()), source=str(source), date=_dt.date.today().isoformat())


#: The group of a calibration file its model is kept in: uipsf's `export.model_file.EMBEDDED`, written out
#: here so that reading whether a calibration has one needs neither uipsf nor torch.
MODEL_GROUP = "uipsf"


def model_path(path) -> Path:
    """Where an older calibration's model file is beside it: ``x_3dcal.h5`` -> ``x_3dcal.uipsf.h5``.  A
    calibration saved since November 2026 carries its model (`MODEL_GROUP`); this is read for the others."""
    path = Path(path)
    return path.with_name(path.stem + ".uipsf.h5")


def carries_model(path) -> bool:
    """Whether the calibration at ``path`` carries its model (`save`), read with h5py alone."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            group = f.get(MODEL_GROUP)
            return isinstance(group, h5py.Group) and group.attrs.get("format") == "uipsf-model"
    except (OSError, ValueError):
        return False


def save(path, cal, calibration, overwrite: bool = False) -> Path:
    """The spline (or dual) ``calibration`` at ``path`` with the uipsf ``cal`` (a `Calibration`) inside it, in
    the group `MODEL_GROUP`: the model re-samples the spline on another grid, at another depth, without
    learning again.  One file, so the spline cannot travel without what it was drawn from -- a model file
    beside it, as saved until November 2026, was lost whenever the spline alone was copied.  A model file
    left beside it by an earlier save is removed: it would describe another run.  Returns the path."""
    import uipsf

    from ..io.calibration import save_spline_calibration
    from .dual import DualColorCalibration, save_dual_color_calibration
    path = Path(path)
    if isinstance(calibration, DualColorCalibration):
        for m in (calibration.main, calibration.secondary):
            m.parameters.pop("model_file", None)
        calibration.parameters.pop("model_file", None)
        save_dual_color_calibration(path, calibration, overwrite=overwrite)
    else:
        calibration.parameters.pop("model_file", None)
        save_spline_calibration(path, calibration, overwrite=overwrite)
    uipsf.save_model(cal, path, group=MODEL_GROUP)
    beside = model_path(path)
    if beside.exists():
        beside.unlink()
    return path


# ------------------------------------------------------------------ a whole run: what the window does

#: the sample's index on *auto* for one objective (`OpticsSettings.sample_index`)
WATER = 1.33


@dataclass
class OpticsSettings:
    """The microscope.  Each field on *auto* takes the microscope database's value for the file's
    microscope (`microscope_db`, carried in `CameraMetadata.microscope`): its NA, its immersion, the
    emission of each arm from the filter it was set to.  A value typed beats it.  There is no default for
    the NA and the emission: without a microscope that gives them, a run refuses to start.

    Shown as the microscope control (`gui.microscope_form`): one line, the values in a window,
    locked until unlocked.  The microscope and the camera are chosen here for stacks nothing
    recognises; the camera's numbers here win over the camera database's, as a fitter's do."""
    microscope: str = param("", label="microscope", choices=lambda: _microscope_choices(),
                            help="auto: recognised from the stacks' metadata, else the last one used")
    camera: str = param("", label="camera", choices=lambda: _camera_choices(),
                        help="auto: identified from the stacks' metadata, else the microscope's")
    NA: Optional[float] = param(None, label="NA", min=0.1, max=1.7,
                                help="the objective's numerical aperture; auto: the microscope's (required "
                                     "without one)")
    wavelength_nm: Optional[float] = param(None, label="emission", unit="nm", min=300.0, max=1200.0,
                                           help="the beads' emission wavelength; auto: from the microscope's "
                                                "filter (required without one)")
    wavelength2_nm: Optional[float] = param(None, label="emission, 2nd", unit="nm", min=300.0, max=1200.0,
                                            help="dual colour: the secondary channel's; auto: the "
                                                 "microscope's other arm, or the same without one")
    n_immersion: Optional[float] = param(None, label="immersion index", min=1.0, max=2.0,
                                         help="the immersion medium's refractive index; auto: the "
                                              "microscope's, else oil's 1.518")
    # auto keeps each mode's own assumption: water for one objective, index-matched for 4Pi (`fourpi_beads.learn`);
    # typed, it is every mode's, and *learn sample index* starts from it
    n_medium: Optional[float] = param(None, label="sample index", min=1.0, max=2.0, advanced=True,
                                      help="the beads' medium: water 1.33, air 1.0, a mounting medium up to "
                                           "the immersion's (index matched); auto: water's, the immersion's for "
                                           "4Pi")
    n_coverslip: Optional[float] = param(None, label="coverslip index", min=1.0, max=2.0, advanced=True,
                                         help="auto: the microscope's for its objective, else the immersion's")
    # the thickness is the sample's, not the objective's: the design thickness is the microscope's
    # (`coverslip_design_um`, 170 µm without one); either matters only with a coverslip of another index
    coverslip_um: Optional[float] = param(None, label="coverslip thickness", unit="µm", min=0.0, max=1000.0,
                                          advanced=True,
                                          help="the coverslip's thickness; auto: the objective's design "
                                               "thickness.  Only with a coverslip of another index")
    pixel_x_nm: Optional[float] = param(None, label="pixel x", unit="nm", min=1.0,
                                        help="on the sample; auto: the microscope's for its 3D lens, or "
                                             "the camera's")
    pixel_y_nm: Optional[float] = param(None, label="pixel y", unit="nm", min=1.0,
                                        help="auto: as x, or the microscope's")
    conversion: Optional[float] = param(None, unit="e-/ADU", min=0,
                                        help="photoelectrons per count; auto: the camera database's")
    offset: Optional[float] = param(None, unit="ADU", help="the count of a pixel that saw no light")
    emgain: Optional[float] = param(None, label="EM gain", min=0, help="auto: from the stacks")

    def camera_overrides(self) -> dict:
        """The camera's numbers typed here, for `io.tiff.camera_metadata`: the pixel too, in the
        camera's µm -- typed, it is the user's word, and the beads' boxes are cut and their positions
        measured with the camera's pixel (`pixel_size_nm`), which without it was the file's or none:
        a stack with no pixel in its metadata or the camera database could not be calibrated with
        the pixel typed here (2026-11-12)."""
        out = {k: v for k, v in (("conversion", self.conversion), ("offset", self.offset),
                                 ("emgain", self.emgain)) if v is not None}
        if self.pixel_x_nm is not None:
            out["pixelsize_um"] = self.pixel_x_nm / 1000
            out["pixelsize_y_um"] = (self.pixel_y_nm or self.pixel_x_nm) / 1000
        return out

    def resolved(self, camera, channels: int = 1, main: Optional[str] = None) -> dict:
        """The values a run uses: ``NA``, ``n_immersion``, ``wavelengths`` (one per channel, None where
        nothing gives one) and ``pixel`` (x, y), each typed or from the microscope and the camera.

        Dual colour: the main channel is the arm on ``main``'s half of the chip (``left``, ``right``,
        ``upper``, ``lower``, the dual calibration's ``main_channel``); on *auto* (None) the
        microscope's first arm, the transmitted one, which is its main channel
        (`dual.resolve_arrangement`), and the left or the upper where no arm has a side.  Single colour: the microscope's first arm, the transmitted
        one, over the whole chip -- nothing records whether the splitting dichroic was in."""
        scope = getattr(camera, "microscope", None) or {}
        values = scope.get("values") or {}
        arms = list(scope.get("channels") or [])
        if channels > 1 and arms:
            sides = [arm.get("side") for arm in arms]
            if main is None:
                # the entry's main channel: its first arm's side.  It took the left or the upper
                # half, which on M2 (transmitted right) gave the main channel the reflected arm's
                # wavelength (2026-11)
                main = sides[0] or next((s for s in ("left", "upper") if any(
                    x in (s, {"left": "right", "upper": "lower"}[s]) for x in sides)), None)
            if main in sides:
                arms.sort(key=lambda arm: arm.get("side") != main)
        found = [arm.get("wavelength_nm") for arm in arms] or [values.get("wavelength_nm")]
        waves = [self.wavelength_nm or found[0]]
        if channels > 1:
            # an arm the microscope has but cannot name a wavelength for (an empty slider behind a
            # dichroic it does not record) is missing, not the first's: only a microscope with one
            # arm, or none known, means the same light on both halves
            second = found[1] if len(found) > 1 else waves[0]
            waves.append(self.wavelength2_nm or second)
        try:
            pixel = pixel_size_nm(camera)
        except (ValueError, AttributeError):         # no pixel, or no stacks yet
            pixel = None
        if self.pixel_x_nm is not None:
            pixel = (self.pixel_x_nm, self.pixel_y_nm or self.pixel_x_nm)
        n_immersion = self.n_immersion or values.get("n_immersion") or 1.518
        design = float(values.get("coverslip_design_um") or DESIGN_COVERSLIP_UM)
        return dict(NA=self.NA or values.get("NA"), n_immersion=n_immersion,
                    n_coverslip=self.n_coverslip or values.get("n_coverslip") or n_immersion,
                    coverslip_um=self.coverslip_um if self.coverslip_um is not None else design,
                    coverslip_design_um=design,
                    wavelengths=waves, pixel=pixel, microscope=scope.get("name"))

    def missing(self, camera, channels: int = 1, main: Optional[str] = None) -> list:
        """What a run would lack, in words: empty when it can start."""
        found = self.resolved(camera, channels, main)
        out = [] if found["NA"] else ["NA"]
        out += [("emission" if i == 0 else "emission, 2nd") for i, w in enumerate(found["wavelengths"]) if not w]
        return out

    def optics(self, camera, channels: int = 1, main: Optional[str] = None) -> list:
        """One `uipsf.Optics` per channel; a ValueError naming what is missing."""
        missing = self.missing(camera, channels, main)
        if missing:
            scope = getattr(camera, "microscope", None)
            why = (f"{scope['name']} does not give it for this file" if scope
                   else "no microscope in the database recognises this file")
            raise ValueError(f"give the microscope's {' and '.join(missing)}: {why}")
        from uipsf import Optics
        found = self.resolved(camera, channels, main)
        if found["pixel"] is None:
            pixel_size_nm(camera)                    # its error names what is missing
        return [Optics(NA=found["NA"], wavelength_nm=w, n_imm=found["n_immersion"], n_med=self.sample_index(),
                       pixel_size_nm=found["pixel"], n_cov=found["n_coverslip"],
                       d_cov_nm=found["coverslip_um"] * 1000, d_cov_design_nm=found["coverslip_design_um"] * 1000)
                for w in found["wavelengths"]]

    def sample_index(self, fourpi: bool = False, n_immersion: Optional[float] = None) -> float:
        """The sample's index: as typed, else water's (`WATER`), or for 4Pi the immersion's ``n_immersion``, its
        sample taken as index-matched."""
        if self.n_medium is not None:
            return float(self.n_medium)
        return float(n_immersion) if fourpi and n_immersion else WATER

    def coverslip_matters(self, camera) -> bool:
        """Whether the coverslip is a layer of its own here, an index other than the immersion's (a water or
        silicone objective): only then do its thickness and its tilt change what beads on it look like
        (`uipsf.Optics.coverslip_matters`)."""
        found = self.resolved(camera)
        return found["n_coverslip"] != found["n_immersion"]

    def hints(self, camera, channels: int = 1, main: Optional[str] = None) -> dict:
        """What the form shows greyed in its *auto* fields for this camera: `SettingsForm.set_hints`."""
        found = self.resolved(camera, channels, main)
        out = {"NA": found["NA"], "n_immersion": found["n_immersion"], "wavelength_nm": found["wavelengths"][0],
               "n_coverslip": found["n_coverslip"], "coverslip_um": found["coverslip_um"]}
        if channels > 1:
            out["wavelength2_nm"] = found["wavelengths"][1]
        if found["pixel"] is not None:
            out.update(pixel_x_nm=found["pixel"][0], pixel_y_nm=found["pixel"][1])
        return {k: v for k, v in out.items() if v is not None}


@dataclass
class SplineSettings:
    """How the learnt model is sampled into the spline the fitter reads."""
    dz_nm: Optional[float] = param(None, label="spline z step", unit="nm", min=1.0,
                                   help="auto: the stacks' step")
    z_range_nm: Optional[float] = param(None, label="spline z range", unit="nm", min=50.0,
                                        help="± about focus; auto: the range every used bead covers")


MODEL_FILTER = "uipsf model or PSF calibration (*.h5);;All files (*)"


@dataclass
class RunSettings:
    """What a run starts from and leaves behind."""
    start_from: str = param("", label="start from", kind="open_file", file_filter=MODEL_FILTER,
                            help="a previous model of this microscope (its calibration, or an older "
                                 ".uipsf.h5): the start, and the reference the Zernikes are drawn against")
    save_model: bool = param(True, label="save model",
                             help="save the calibration and its model beside the stacks; off: figures only")


def load_start(path):
    """The uipsf model at ``path``: a calibration that carries its model, a model file, or a calibration an
    older save put a model file beside."""
    import uipsf
    path = Path(path)
    beside = model_path(path)
    if not carries_model(path) and not path.name.endswith(".uipsf.h5") and beside.exists():
        path = beside
    try:
        return uipsf.load_model(path)
    except ValueError as error:
        raise ValueError(f"{path.name} is not a uipsf model and carries none, and no model file is beside "
                         f"it") from error


def default_save_path(paths, dual: bool) -> Path:
    """The bead calibration's naming (`core.calibration_save_defaults`), never over an existing file."""
    from .core import calibration_save_defaults
    folder, name = calibration_save_defaults(paths, dual=dual)
    path, n = Path(folder) / f"{name}.h5", 2
    while path.exists() or model_path(path).exists():
        path, n = Path(folder) / f"{name}_{n}.h5", n + 1
    return path


@dataclass
class PupilRun:
    """A run's result: the ROIs smappy cut, the learnt calibration, and what was saved."""
    rois: BeadROIs
    cal: object                          # uipsf Calibration
    start: object = None                 # the LearnedPSF it started from, or None
    calibration: object = None           # the SplineCalibration or DualColorCalibration, when saved
    saved: Optional[Path] = None
    dual: bool = False


#: a learning run costs about this many forward passes of one bead per bead and per parameter, each timed as
#: `estimate_seconds` times one: 1.4 to 2.3 on ten simulated beads at Zernike orders 3 to 8 (7 to 16 s a run,
#: 4 x86 cores).  The Jacobians, one forward-mode pass per parameter, are most of a run
PASSES_PER_PARAMETER = 2.0
#: forward passes timed, the median kept: one alone varied from 4.3 to 7.3 ms
TIMED_PASSES = 3


def field_centre(batch) -> tuple | None:
    """Where a field-dependent model's field is centred: the middle of the beads' chip positions (their
    span's, not their mean, so that a cluster of beads does not move it)."""
    if batch.field_xy is None:
        return None
    xy = batch.field_xy.detach().cpu().double().numpy()
    return tuple(float(v) for v in (xy.min(0) + xy.max(0)) / 2)


def learns_tilt(model: BeadSettings, optics) -> BeadSettings:
    """``model`` as a bead run learns it with ``optics`` (`uipsf.Optics`, one per channel): *learn coverslip
    tilt* dropped where the coverslip is the immersion (oil): beads on it see no tilt at all, and the window
    hides the choice there (`psf_window`), so one left ticked from another microscope must not cost a fit."""
    from dataclasses import replace
    if model.fit_tilt and not any(o.coverslip_matters for o in optics):
        return replace(model, fit_tilt=False)
    return model


def _learnt_model(rois: BeadROIs, optics: OpticsSettings, model: BeadSettings):
    """The model a run on ``rois`` learns (`calibrate_beads`' `bead_model`), for the estimates before it."""
    from uipsf.workflows.calibrate import bead_model, learning_batch
    batch = learning_batch(rois.batch, model)          # its z reach sizes the pupil grid, as in a run
    channels = batch.data.shape[1]
    found = optics.optics(rois.camera, channels, main_half(rois))
    return bead_model(found, learns_tilt(model, found), tuple(batch.data.shape[-2:]),
                      [None, rois.transform] if channels == 2 else None, float(batch.z_nm.abs().max()),
                      field_centre_px=field_centre(batch))


def estimate_memory(rois: BeadROIs, optics: OpticsSettings, model: BeadSettings):
    """What learning ``rois`` will hold at most, before it runs: a `uipsf.solve.MemoryPlan` -- the beads
    computed at once (as many as fit in the memory the system has free, `uipsf.solve.problem`), the most one
    such chunk holds above what is in use now, measured on one bead of this model (`uipsf.memory`), and
    what is free.  M6's 19 beads of 100 slices: chunks of 4, 8.5 GB expected, a run's peak 8.6 GB above the
    3.7 GB it started from."""
    from uipsf.device import on_model_device
    from uipsf.noise import ShiftedPoisson
    from uipsf.solve import memory_plan
    from uipsf.workflows.calibrate import learning_batch
    learnt = _learnt_model(rois, optics, model)
    return memory_plan(learnt, ShiftedPoisson(), learning_batch(on_model_device(rois.batch, learnt), model))


def estimate_seconds(rois: BeadROIs, optics: OpticsSettings, model: BeadSettings) -> float:
    """About how long learning ``rois`` will take, seconds, in well under a second: the model's forward pass
    on one bead, timed (`TIMED_PASSES`), times the beads, the parameters and `PASSES_PER_PARAMETER`.  A
    figure to decide by before an afternoon's run -- within a factor of two -- not a promise: the iterations
    a fit needs depend on the beads.  Timing one Gauss-Newton matrix instead was closer (4.4 to 5.1 matrices
    a run) but cost 1.5 to 3 s on these ten beads, more on many."""
    import time

    import torch
    from uipsf.device import on_model_device
    from uipsf.params import Params
    from uipsf.workflows.calibrate import learning_batch
    batch = rois.batch
    learnt = _learnt_model(rois, optics, model)
    one = learning_batch(on_model_device(batch.subset(torch.arange(1)), learnt), model)
    start = Params(learnt.specs(one), 1, dtype=one.data.dtype, device=one.data.device)
    values = start.physical()
    times = []
    with torch.no_grad():
        for k in range(TIMED_PASSES + 1):            # the first pass warms the kernels up: not counted
            t = time.perf_counter()
            learnt(values, one).sum().item()         # .item(): waits for a GPU to finish
            if k:
                times.append(time.perf_counter() - t)
    shared = sum(int(torch.as_tensor(v).numel()) for n, v in values.items() if start.specs[n].scope == "shared")
    per_bead = sum(int(torch.as_tensor(v).numel()) for n, v in values.items() if start.specs[n].scope != "shared")
    return float(np.median(times)) * PASSES_PER_PARAMETER * batch.n_emitters * (shared + per_bead)


def run_calibration(paths, dual: bool, optics: OpticsSettings, model: BeadSettings, detection=None,
                    spline: SplineSettings | None = None, run: RunSettings | None = None, camera=None,
                    previous: PupilRun | None = None, decisions: dict | None = None, save_to=None,
                    progress=None, cancel=None, rois: BeadROIs | None = None) -> PupilRun:
    """Bead files to a learnt pupil model and, with *save model*, its calibration on disk: the PSF
    calibration window's whole run, for a script too.

    ``previous`` (the window's *Calibrate* again with the same beads and settings): the same ROIs, its fit as the warm start, ``decisions``
    ({bead: "include" | "exclude"}) applied; None leaves out the beads marked doubles (`core.judge_shapes`),
    as the window marks them, and {} none.  ``camera`` defaults to the first acquisition's, as a fit
    would resolve it.  ``save_to`` defaults to the bead calibration's name beside the stacks.  ``rois``,
    the beads found already (the window's *Detect beads*), skips finding them again."""
    from uipsf.workflows import calibrate_beads
    spline = spline or SplineSettings()
    run = run or RunSettings()
    if previous is not None:
        rois, start, warm = previous.rois, previous.start, previous.cal
    else:
        report = progress or (lambda entry: None)
        if rois is None:
            report({"message": "reading the stacks and finding the beads"})
            rois = bead_rois(paths, camera, dual=dual, settings=detection)
        start = load_start(run.start_from) if run.start_from else None
        warm = None
    if decisions is None and previous is None:
        decisions = {i: "exclude" for i in doubles(rois.records)} or None
    channels = 2 if dual else 1
    found = optics.optics(rois.camera, channels, main_half(rois))
    cal = calibrate_beads(rois.batch, found, learns_tilt(model, found),
                          transforms=[None, rois.transform] if dual else None, geometry=rois.geometry,
                          start=start, noise=rois.noise, decisions=decisions, warm_start=warm,
                          progress=progress, cancel=cancel)
    result = PupilRun(rois, cal, start, dual=dual)
    if run.save_model and cal.result.stop_reason != "cancelled":
        dz = spline.dz_nm or float(rois.batch.z_step_nm)
        z_range = spline.z_range_nm or _auto_z_range(cal, rois.batch)
        roi = int(rois.batch.data.shape[-1])
        result.calibration = (dual_calibration(cal, dz, z_range, roi) if dual
                              else spline_calibration(cal, dz, z_range, roi, camera=rois.camera))
        if dual:
            from ..io.calibration import keep_microscope
            for channel in (result.calibration.main, result.calibration.secondary):
                keep_microscope(channel, rois.camera)
        # a warm run (``previous``) replaces its own run's files; a new run never overwrites
        again = previous.saved if previous is not None else None
        path = Path(save_to) if save_to else again or default_save_path(paths, dual)
        result.saved = save(path, cal, result.calibration, overwrite=bool(save_to or again))
    return result


def half_split_uncertainty(run: PupilRun, optics: OpticsSettings, splits: int = 3, seed: int = 0,
                           progress=None, cancel=None):
    """The window's *Uncertainty from halves*: ``run``'s calibration learnt again on halves of its used
    beads, ``splits`` complementary pairs, each from the run's own start -- the same ROIs, noise, transforms
    and settings (`uipsf.diagnostics.half_splits`, which says why).  ``optics`` are the settings the run was
    made with.  None if cancelled; kept on the calibration (``cal.halves``), so a save writes it."""
    from uipsf.diagnostics import half_splits
    rois, cal = run.rois, run.cal
    halves = half_splits(rois.batch, optics.optics(rois.camera, 2 if run.dual else 1, main_half(rois)), cal.options,
                         cal.beads.used, splits=splits, seed=seed,
                         transforms=[None, rois.transform] if run.dual else None, geometry=rois.geometry,
                         start=run.start, noise=rois.noise, progress=progress, cancel=cancel)
    if halves is not None:
        cal.halves = halves
    return halves


# ------------------------------------------------------------------ fit quality: the beads, refitted

def covered_z_nm(cal, batch) -> tuple[float, float]:
    """(lowest, highest) emitter z that every used bead's stack reaches, nm.  A bead sits at its own height,
    so its slices cover the nominal range shifted by it; at the ends only some beads remain, and a model
    there was learnt from those few.  So the spline on *auto*, *Model vs data* and the fit quality keep to
    the range all of them cover, as the bead average crops its average (`core.build_calibration`)."""
    import torch
    from uipsf.device import on_model_device
    from uipsf.workflows.calibrate import learning_batch
    used = list(cal.beads.used)
    sub = learning_batch(on_model_device(batch, cal.model), cal.options).subset(torch.tensor(used))
    with torch.no_grad():
        p = {k: v.to(sub.data.device) for k, v in cal.result.params.physical().items()}
        z = cal.model.geometry(p, sub)[..., 2].cpu().double().numpy()            # [E, K]
    return float(np.max(z.min(axis=1))), float(np.min(z.max(axis=1)))


def _auto_z_range(cal, batch) -> float:
    """± the spline reaches on *auto*: the covered range (`covered_z_nm`), the nearer end about focus."""
    low, high = covered_z_nm(cal, batch)
    return max(min(-low, high), float(batch.z_step_nm))


def calibration_of(run: PupilRun):
    """The run's spline (or dual) calibration: the saved one, else sampled now as saving would."""
    if run.calibration is None:
        rois = run.rois
        dz, z_range = float(rois.batch.z_step_nm), _auto_z_range(run.cal, rois.batch)
        roi = int(rois.batch.data.shape[-1])
        run.calibration = (dual_calibration(run.cal, dz, z_range, roi) if run.dual
                           else spline_calibration(run.cal, dz, z_range, roi, camera=rois.camera))
    return run.calibration


def _best_of(fit, starts, keys=("theta", "crlb", "logl", "iterations")):
    """``fit(z_start_nm)`` from each start, per ROI the best likelihood kept: an ambiguous PSF shows as
    jumps rather than as whichever side one start fell to."""
    best = None
    for z in starts:
        result = fit(float(z))
        if best is None:
            best = result
            continue
        keep = np.isfinite(result.logl) & (~np.isfinite(best.logl) | (result.logl > best.logl))
        for key in keys:
            getattr(best, key)[keep] = getattr(result, key)[keep]
    return best


def _centred(error, bead_id, valid) -> np.ndarray:
    """``error`` less each bead's median over its valid slices: its height above the coverslip is not
    known, its curve's distortion is what is left."""
    centred = error.copy()
    for e in np.unique(bead_id):
        use = (bead_id == e) & valid
        if use.any():
            centred[bead_id == e] -= np.nanmedian(error[use])
    return centred


def _pair_link(calibration, roi_xy, roisize: int) -> np.ndarray:
    """The global fit's link [E, 2, 2, 5] for each pair's two ROIs, as the two-colour fitters build it
    (`multichannel.build_link`): each ROI was cut around its own channel's detection (`dual_rois`), on
    the chip, so the secondary's offset is where the transform puts the main ROI's centre, less its own
    centre, and its factor the transform's local scale -- -1 along a mirror -- acting about the ROI's
    centre.  The photon factor is one: the secondary spline already carries the beads' split."""
    from ..dualfit import _linearise
    main, second = np.asarray(roi_xy[:, 0], float), np.asarray(roi_xy[:, 1], float)
    link = np.zeros((len(main), 2, 2, 5), np.float32)
    link[:, 1] = 1.0
    link[:, 0, 1, :2] = calibration.to_secondary(main) - second
    at = np.trunc(main)                                         # the scale is read at the ROI's pixel
    fx, fy = _linearise(calibration.to_secondary, at[:, 0], at[:, 1])
    half = (int(roisize) - 1) / 2.0
    link[:, 1, 1, 0], link[:, 1, 1, 1] = fx, fy
    link[:, 0, 1, 0] += (1.0 - fx) * half
    link[:, 0, 1, 1] += (1.0 - fy) * half
    return link


def fit_quality(run: PupilRun, z_starts: int = 5, iterations: int = 75) -> dict:
    """Every slice of every bead fitted with the run's spline by SMAPpy's own fitter: what a
    localisation of data with this calibration will see, as the bead average's fit quality
    (`validation.fit_bead_diagnostics`) shows it, and SMAP's bead calibration before it.

    All beads, used or not, so that a rejected bead's curve shows why.  The fit box is the bead box less
    two pixels a side, for the spot's lateral shift; the read noise is added as the shifted Poisson does
    (`uipsf.workflows.spline`).  Each slice is started from ``z_starts`` heights and the best likelihood
    kept, so an ambiguous PSF shows as jumps.  The *centred* error removes each bead's own offset (its
    height above the coverslip is not known), leaving the curve's distortion.

    One channel: Spline 3D.  Two: Spline 3D 2C's global fit (`GlobalSplinePSF`), one x, y and z per pair
    linked through the transform, the photons free per channel, as `validation.fit_paired_bead_diagnostics`
    refits the bead average's pairs: a fit of each channel on its own half of the calibration missed what
    the data will meet, a transform that puts the partner in the wrong place or splines whose z disagree.
    One z per pair, so one curve per bead, and the colour ratio beside it.
    """
    from ..psf import GlobalSplinePSF, SplinePSF
    calibration = calibration_of(run)
    batch = run.rois.batch
    data = batch.data.detach().cpu().double().numpy()                 # [E, C, K, Y, X] photons
    E, C, K, Y, X = data.shape
    s = batch.s
    s = float(s) if not hasattr(s, "dim") or s.dim() == 0 else s.expand_as(batch.data).cpu().numpy()
    data = np.maximum(data + s, 0)
    nominal = batch.z_nm.detach().cpu().double().numpy()
    nominal = np.broadcast_to(nominal, (E, K)) if nominal.ndim == 1 else nominal
    out = {"bead_id": np.repeat(np.arange(E), K), "frame": np.tile(np.arange(K), E),
           "nominal_z_nm": nominal.reshape(-1)}
    box = np.ascontiguousarray(data[..., 2:Y - 2, 2:X - 2].transpose(0, 2, 1, 3, 4).reshape(E * K, C, Y - 4, X - 4),
                               np.float32)                            # [E·K, C, y, x]
    if C == 1:
        spline = calibration
        best = _best_of(lambda z: SplinePSF(spline, z_start_nm=z).fit(np.ascontiguousarray(box[:, 0]),
                                                                       iterations=iterations),
                        spline.z_index_to_nm(np.linspace(1, spline.shape[0] - 2, z_starts)))
        z_index, photons = best.theta[:, 4], best.theta[:, 2]
    else:
        from ..dualfit import LINK_XYZ
        spline = calibration.main
        model = GlobalSplinePSF((calibration.main, calibration.secondary), LINK_XYZ)
        link = np.repeat(_pair_link(calibration, batch.roi_xy.detach().cpu().double().numpy(), Y - 4), K, axis=0)

        def fit(z):
            model.z_start_nm = z
            return model.fit(box, link=link, iterations=iterations)
        best = _best_of(fit, spline.z_index_to_nm(np.linspace(1, spline.shape[0] - 2, z_starts)))
        values = model.unpack(best, link)
        z_index, photons = best.theta[:, model.slot(4)], values["photons"]
    valid = np.isfinite(best.logl) & np.isfinite(best.theta).all(axis=1)
    # the slices of a bead beyond the spline's range (a bead high or low on the coverslip reaches past the
    # range the others cover, `covered_z_nm`) are not fitted: the fit could only stop at the spline's end
    reach = min(abs(float(spline.z_index_to_nm(0))), abs(float(spline.z_index_to_nm(spline.shape[0] - 1))))
    found = spline.z_index_to_nm(z_index) - out["nominal_z_nm"]
    for e in range(E):
        mine = out["bead_id"] == e
        central = mine & valid & (np.abs(out["nominal_z_nm"]) < reach / 2)
        offset = float(np.median(found[central])) if central.any() else 0.0
        valid[mine & (np.abs(out["nominal_z_nm"] + offset) > reach)] = False
    fitted = np.where(valid, spline.z_index_to_nm(z_index), np.nan)
    out.update({"fitted_z_nm": fitted,
                "centred_error_nm": _centred(fitted - out["nominal_z_nm"], out["bead_id"], valid),
                "photons": photons, "fit_valid": valid,
                "at_z_boundary": (z_index < 1) | (z_index > spline.shape[0] - 2)})
    if C == 2:
        out["ratio"] = np.where(valid, values["ratio"], np.nan)
    out["state"] = np.repeat([row.state for row in run.cal.beads.rows], K)
    return out
