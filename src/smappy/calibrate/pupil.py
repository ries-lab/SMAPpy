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
* **Saturated pixels are masked, not the bead dropped.**  The brightness
  pre-filter (*too dim*, *too bright*) is not applied either: uipsf judges
  every bead by its fit (`workflows/rejection.py`).  For two channels the
  dual pairing still uses brightness to vote for the offset between halves,
  as the dual calibration does.
* **The first slice is dropped** by default: on the Zenodo stacks it is 22 %
  brighter than the rest, shared by all beads (`docs/uipsf/architecture.md`,
  M3), and uiPSF v1 drops it too.  Stacks of different lengths are cut to
  the shortest about their middles, as `collect_beads` does; each stack's
  nominal focus is the middle of the slices kept.

torch is imported inside the functions: this module is imported by the
calibration window without torch installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..plugins import param
from .core import CalibrationSettings, classify_brightness, find_stack_beads, saturation_of
from .input import BeadStack, discover_acquisitions, read_bead_stacks

KEEP = slice(1, None)            # every slice but the first (module docstring)

#: the order a calibration learns up to unless told: n_max 6, with 8 the held-out check (plan §10)
AUTO_ORDER = 6

MAGNITUDES = (("free", "free"), ("symmetric", "symmetric"), ("fixed", "fixed (apodisation)"))
BLURS = (("free", "free"), ("fixed", "fixed"))


@dataclass
class BeadSettings:
    """The choices a bead calibration exposes; the instrument comes as `Optics`."""
    zernike_order: int | None = param(
        None, label="Zernike order", min=2, max=12,
        help=f"the highest radial order of the pupil's modes; auto: {AUTO_ORDER}")
    magnitude: str = param(
        "free", label="pupil magnitude", choices=MAGNITUDES,
        help="free: every mode up to the order; symmetric: the rotationally symmetric ones; "
             "fixed: uniform, the apodisation only")
    blur: str = param("free", label="blur", choices=BLURS,
                      help="an extra Gaussian blur of the image, learnt or held at its value")
    blur_nm: float = param(30.0, label="blur σ", unit="nm", min=0.0,
                           help="its start when free, its value when fixed")
    bead_diameter_nm: float = param(0.0, label="bead diameter", unit="nm", min=0.0,
                                    help="the beads' size, taken out of the model; 0: point-like")
    rejection_k: float = param(3.0, label="rejection (k)", min=1.0, advanced=True,
                               help="robust standard deviations above the median at which a bead is "
                                    "rejected")
    heldout: float = param(0.0, label="held out", min=0.0, max=0.5, advanced=True,
                           help="the fraction of beads kept out of learning and localised for the "
                                "held-out report")

    def __post_init__(self):
        if self.magnitude not in dict(MAGNITUDES):
            raise ValueError(f"magnitude must be one of {[k for k, _ in MAGNITUDES]}, got {self.magnitude!r}")
        if self.blur not in dict(BLURS):
            raise ValueError(f"blur must be one of {[k for k, _ in BLURS]}, got {self.blur!r}")

    @property
    def order(self) -> int:
        return AUTO_ORDER if self.zernike_order is None else int(self.zernike_order)


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
    if np.issubdtype(np.asarray(stack.images).dtype, np.integer):
        return float(np.iinfo(np.asarray(stack.images).dtype).max)
    return None


def _cut(stacks, centres, roi_size, slices, camera, saturation_adu=None):
    """ROIs around ``centres`` [(stack, x, y), ...] (whole pixels of the full image): photons [E, K, Y, X],
    saturation [E, K, Y, X] and objective z [E, K], every stack cut to the same slices."""
    from ..camera import to_photons
    half = roi_size // 2
    planes = [np.arange(len(st.z_nm))[slices] for st in stacks]
    nz = min(len(p) for p in planes)
    planes = [p[(len(p) - nz) // 2:(len(p) - nz) // 2 + nz] for p in planes]
    data, saturated, z = [], [], []
    for s, x, y in centres:
        st, kept = stacks[s], planes[s]
        if not (half <= y < st.images.shape[-2] - half and half <= x < st.images.shape[-1] - half):
            raise ValueError(f"bead at ({x}, {y}) in {st.source}: its ROI leaves the image")
        adu = np.asarray(st.images[kept, y - half:y - half + roi_size, x - half:x - half + roi_size])
        limit = _saturation(st, saturation_adu)
        saturated.append(adu >= limit if limit is not None else np.zeros(adu.shape, bool))
        data.append(to_photons(adu, camera))
        objective = np.asarray(st.z_nm, float)[kept]
        z.append(objective - objective.mean())          # about the middle: the nominal focus
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
    """One channel: the beads the bead average would find (`core.find_stack_beads`), a `settings.roi_size`
    ROI each, with the same records: position, background, brightness and its verdict, saturation.  The
    verdict is shown, not applied (module docstring)."""
    settings = settings or CalibrationSettings()
    settings.validate()
    channels = {str(st.axes.get("channel", 0)) for st in stacks}
    if len(channels) > 1:
        raise ValueError("the acquisitions hold several channels; calibrate them one at a time")
    found = []
    for s, st in enumerate(stacks):
        records, _ = find_stack_beads(st.images, settings, saturation_of(st.images, settings))
        found += [dict(r, stack=s) for r in records]
    if not found:
        raise ValueError("no isolated beads found: review detection and the bead box")
    classify_brightness(found, settings)
    rois = rois_at(stacks, [(r["stack"], r["x_px"], r["y_px"]) for r in found], camera, settings.roi_size, slices,
                   settings.saturation_adu)
    rois.records = [dict(r, id=i) for i, r in enumerate(found)]
    return rois


def dual_rois(stacks, camera, settings=None, slices=KEEP) -> BeadROIs:
    """Two channels on one chip: the pairs `collect_dual_beads` finds, each channel's ROI cut around its
    own detection in the camera's orientation (no flip: the transform carries a mirror), the start
    transform from `fit_dual_transform`."""
    from .dual import DualColorSettings, collect_dual_beads, fit_dual_transform
    settings = settings or DualColorSettings()
    beads = collect_dual_beads(stacks, settings)
    fitted = fit_dual_transform(beads.secondary_points, beads.main_points, beads.settings.registration)
    per_channel = []
    for ch in (0, 1):
        per_channel.append([(r["channel_records"][ch]["stack"],) + tuple(int(round(v)) for v in
                            r["channel_records"][ch]["full_image_xy"]) for r in beads.records])
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
    messages = [f"{int((~fitted.accepted).sum())} of {len(fitted.accepted)} pairs off the starting transform"]
    return BeadROIs(batch, noise, camera, pixel_size_nm(camera), records, beads.sources,
                    _unit(fitted.transformation), dict(beads.geometry),
                    np.stack([beads.main_points, beads.secondary_points], 1), messages)


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

#: each pixel of the spline's knots integrated over this many sub-samples per axis: 4 × 4 keeps the sampling's
#: own error below the spline's (docs/uipsf/architecture.md, "How finely SMAPpy's spline must sample")
UPSAMPLE = 4


def z_grid(dz_nm: float, z_range_nm: float) -> tuple[np.ndarray, float]:
    """(emitter z of each plane, z0): planes ``dz_nm`` apart through ±``z_range_nm``, in the spline's
    convention z = −(index − z0)·dz, so plane 0 is the highest emitter, as in a bead stack."""
    half = int(np.ceil(abs(z_range_nm) / dz_nm))
    return -(np.arange(2 * half + 1) - half) * float(dz_nm), float(half)


def sample(learned, dz_nm: float, z_range_nm: float, roi_size: int = 27, upsample: int = UPSAMPLE,
           channel: int = 0) -> np.ndarray:
    """The learnt PSF of ``channel`` as a voxel stack [z, y, x] for a point emitter on the central pixel,
    each pixel integrated: what a bead average is, without its noise.  ``learned`` is a uipsf
    `Calibration` or `LearnedPSF`; the bead's size is left out, the learnt blur kept, the channel's
    photon share and defocus offset included (`uipsf.export.sample_psf`)."""
    z, _ = z_grid(dz_nm, z_range_nm)
    xyz = np.c_[np.zeros((len(z), 2)), z]
    return learned.psf(xyz, roi_px=roi_size, upsample=upsample, channel=channel).double().numpy()


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


def _record(learned, dz_nm, z_range_nm, roi_size, upsample) -> dict:
    options = getattr(learned, "options", None)
    if options is not None and not isinstance(options, dict):
        from dataclasses import asdict
        options = asdict(options)
    return {"method": "uipsf2_pupil", "zernikes": _zernikes(learned), "options": options or {},
            "sampling": {"dz_nm": float(dz_nm), "z_range_nm": float(z_range_nm), "roi_size": int(roi_size),
                         "upsample": int(upsample)},
            "z_convention": "emitter_nm = -(index-z0)*dz", "lateral_unit": "pixel",
            "z_reference": "the model's focus: the emitter at z = 0"}


def spline_calibration(learned, dz_nm: float, z_range_nm: float, roi_size: int = 27, upsample: int = UPSAMPLE,
                       camera=None):
    """smappy's `SplineCalibration` of a one-channel uipsf model, normalised as the bead average's
    (`core._single_model`: clipped at zero, the brightest plane summing to one, no smoothing, since a
    model has no noise to smooth), so that Spline 3D reads either alike.  ``camera`` records its EM
    state and pixel size, as the bead average does."""
    from ..io.calibration import SplineCalibration
    from .core import spline_coefficients
    if learned.model.n_channels != 1:
        raise ValueError("a model of several channels makes a dual calibration (`dual_calibration`)")
    _, z0 = z_grid(dz_nm, z_range_nm)
    psf = np.maximum(sample(learned, dz_nm, z_range_nm, roi_size, upsample), 0)
    psf /= float(psf.sum(axis=(1, 2)).max())
    parameters = dict(_record(learned, dz_nm, z_range_nm, roi_size, upsample),
                      normalization="maximum plane sum = 1")
    psf = psf.astype(np.float32)
    return SplineCalibration(spline_coefficients(psf), float(dz_nm), z0, x0=(roi_size - 1) / 2, psf=psf,
                             em_on=None if camera is None else bool(camera.is_em),
                             pixelsize_um=None if camera is None or camera.pixelsize_um is None
                             else np.atleast_1d(np.asarray(camera.pixelsize_um, float)),
                             parameters=parameters)


def _transform(learned) -> np.ndarray:
    if callable(getattr(learned, "transform", None)):
        return np.asarray(learned.transform(1), float)
    from uipsf.model.channels import unpack_transform
    return _unit(unpack_transform(learned.shared()["ch1.transform"].detach().cpu().double()).numpy())


def dual_calibration(learned, dz_nm: float, z_range_nm: float, roi_size: int = 27, upsample: int = UPSAMPLE):
    """smappy's `DualColorCalibration` of a two-channel uipsf model: both channels sampled on one z grid
    (a biplane offset is in the PSFs), normalised jointly by `positive_pair_models`, so each channel's
    `photon_normalization` is its learnt photon share; the model's transform and the split it was
    learnt on.  Spline 3D 2C reads it as one of its own (tests/uipsf_tests/test_smappy_dual.py)."""
    from .core import positive_pair_models
    from .dual import DualColorCalibration
    if learned.model.n_channels != 2:
        raise ValueError("a dual calibration needs a model of two channels")
    if learned.geometry is None:
        raise ValueError("the model does not say how the chip was split; learn it from dual_rois")
    raw = np.stack([sample(learned, dz_nm, z_range_nm, roi_size, upsample, c) for c in range(2)])
    settings = CalibrationSettings(roi_size=roi_size, smooth_z_nm=0.0, smooth_xy_px=0.0)
    (main, secondary), _ = positive_pair_models(raw, dz_nm, settings)
    record = _record(learned, dz_nm, z_range_nm, roi_size, upsample)
    for model in (main, secondary):
        model.parameters.update(record)
    return DualColorCalibration(main, secondary, _transform(learned), dict(learned.geometry), dict(record))


def model_path(path) -> Path:
    """Where the model file goes beside a calibration: ``x_3dcal.h5`` -> ``x_3dcal.uipsf.h5``."""
    path = Path(path)
    return path.with_name(path.stem + ".uipsf.h5")


def save(path, cal, calibration, overwrite: bool = False) -> tuple[Path, Path]:
    """The spline (or dual) ``calibration`` at ``path`` and the uipsf ``cal`` (a `Calibration`) beside it
    (`model_path`), which re-samples it on another grid without learning again; the calibration
    records the model file's name.  Returns both paths."""
    import uipsf

    from ..io.calibration import save_spline_calibration
    from .dual import DualColorCalibration, save_dual_color_calibration
    path, beside = Path(path), model_path(path)
    if beside.exists() and not overwrite:
        raise FileExistsError(f"{beside} exists")
    if isinstance(calibration, DualColorCalibration):
        for m in (calibration.main, calibration.secondary):
            m.parameters["model_file"] = beside.name
        calibration.parameters["model_file"] = beside.name
        save_dual_color_calibration(path, calibration, overwrite=overwrite)
    else:
        calibration.parameters["model_file"] = beside.name
        save_spline_calibration(path, calibration, overwrite=overwrite)
    uipsf.save_model(cal, beside)
    return path, beside


# ------------------------------------------------------------------ a whole run: what the window does

@dataclass
class OpticsSettings:
    """The microscope, typed in until the microscope database supplies it (`docs/uipsf/plan.md` §7.5).
    No defaults for what must be right: the NA and the wavelength start at 0, "not given", and a run
    refuses them so (0 rather than None, which the form would offer as *auto*)."""
    NA: float = param(0.0, label="NA", min=0.0, max=1.7,
                      help="the objective's numerical aperture; required (0: not given)")
    wavelength_nm: float = param(0.0, label="emission", unit="nm", min=0.0, max=1200.0,
                                 help="the beads' emission wavelength; required (0: not given)")
    wavelength2_nm: Optional[float] = param(None, label="emission, 2nd", unit="nm", min=300.0, max=1200.0,
                                            help="dual colour: the secondary channel's; auto: the same")
    n_immersion: float = param(1.518, label="immersion index", min=1.0, max=2.0,
                               help="the immersion medium's refractive index (oil 1.518)")
    n_medium: float = param(1.33, label="sample index", min=1.0, max=2.0, advanced=True,
                            help="the beads' medium: water 1.33, air 1.0, a mounting medium up to "
                                 "the immersion's (index matched)")
    n_coverslip: Optional[float] = param(None, label="coverslip index", min=1.0, max=2.0, advanced=True,
                                         help="auto: the immersion's")
    pixel_x_nm: Optional[float] = param(None, label="pixel x", unit="nm", min=1.0, advanced=True,
                                        help="on the sample; auto: the camera's")
    pixel_y_nm: Optional[float] = param(None, label="pixel y", unit="nm", min=1.0, advanced=True,
                                        help="auto: as x, or the camera's")

    def optics(self, camera, channels: int = 1) -> list:
        """One `uipsf.Optics` per channel; a ValueError naming what is missing."""
        missing = [name for name, v in (("NA", self.NA), ("emission", self.wavelength_nm)) if not v]
        if missing:
            raise ValueError(f"give the microscope's {' and '.join(missing)}: there is no default for them")
        from uipsf import Optics
        px, py = pixel_size_nm(camera) if self.pixel_x_nm is None else (self.pixel_x_nm,
                                                                         self.pixel_y_nm or self.pixel_x_nm)
        waves = [self.wavelength_nm, self.wavelength2_nm or self.wavelength_nm][:channels]
        return [Optics(NA=self.NA, wavelength_nm=w, n_imm=self.n_immersion, n_med=self.n_medium,
                       pixel_size_nm=(px, py), n_cov=self.n_coverslip) for w in waves]


@dataclass
class SplineSettings:
    """How the learnt model is sampled into the spline the fitter reads."""
    dz_nm: Optional[float] = param(None, label="spline z step", unit="nm", min=1.0,
                                   help="auto: the stacks' step")
    z_range_nm: Optional[float] = param(None, label="spline z range", unit="nm", min=50.0,
                                        help="± about focus; auto: the stacks' range")


MODEL_FILTER = "uipsf model or PSF calibration (*.h5);;All files (*)"


@dataclass
class RunSettings:
    """What a run starts from and leaves behind."""
    start_from: str = param("", label="start from", kind="open_file", file_filter=MODEL_FILTER,
                            help="a previous model of this microscope (its .uipsf.h5, or the calibration "
                                 "beside it): the start, and the reference the Zernikes are drawn against")
    save_model: bool = param(True, label="save model",
                             help="save the calibration and its model beside the stacks; off: figures only")


def load_start(path):
    """The uipsf model at ``path``: a model file, or the calibration a model file was saved beside."""
    import uipsf
    path = Path(path)
    beside = model_path(path)
    if not path.name.endswith(".uipsf.h5") and beside.exists():
        path = beside
    try:
        return uipsf.load_model(path)
    except ValueError as error:
        raise ValueError(f"{path.name} is not a uipsf model, and no model file is beside it") from error


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


def run_calibration(paths, dual: bool, optics: OpticsSettings, model: BeadSettings, detection=None,
                    spline: SplineSettings | None = None, run: RunSettings | None = None, camera=None,
                    previous: PupilRun | None = None, decisions: dict | None = None, save_to=None,
                    progress=None, cancel=None) -> PupilRun:
    """Bead files to a learnt pupil model and, with *save model*, its calibration on disk: the PSF
    calibration window's whole run, for a script too.

    ``previous`` (the window's *Recalculate*): the same ROIs, its fit as the warm start, ``decisions``
    ({bead: "include" | "exclude"}) applied.  ``camera`` defaults to the first acquisition's, as a fit
    would resolve it.  ``save_to`` defaults to the bead calibration's name beside the stacks."""
    from uipsf.workflows import calibrate_beads
    spline = spline or SplineSettings()
    run = run or RunSettings()
    if previous is not None:
        rois, start, warm = previous.rois, previous.start, previous.cal
    else:
        report = progress or (lambda entry: None)
        report({"message": "reading the stacks and finding the beads"})
        rois = bead_rois(paths, camera, dual=dual, settings=detection)
        start = load_start(run.start_from) if run.start_from else None
        warm = None
    channels = 2 if dual else 1
    cal = calibrate_beads(rois.batch, optics.optics(rois.camera, channels), model,
                          transforms=[None, rois.transform] if dual else None, geometry=rois.geometry,
                          start=start, noise=rois.noise, decisions=decisions, warm_start=warm,
                          progress=progress, cancel=cancel)
    result = PupilRun(rois, cal, start, dual=dual)
    if run.save_model and cal.result.stop_reason != "cancelled":
        z = rois.batch.z_nm
        dz = spline.dz_nm or float(rois.batch.z_step_nm)
        z_range = spline.z_range_nm or float(z.abs().max())
        roi = int(rois.batch.data.shape[-1])
        result.calibration = (dual_calibration(cal, dz, z_range, roi) if dual
                              else spline_calibration(cal, dz, z_range, roi, camera=rois.camera))
        # *Recalculate* replaces its own run's files; a new run never overwrites
        again = previous.saved if previous is not None else None
        path = Path(save_to) if save_to else again or default_save_path(paths, dual)
        result.saved, _ = save(path, cal, result.calibration, overwrite=bool(save_to or again))
    return result
