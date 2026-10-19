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

from .core import CalibrationSettings, detect_beads
from .input import BeadStack, discover_acquisitions, read_bead_stacks

KEEP = slice(1, None)            # every slice but the first (module docstring)


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
    """One channel: every isolated bead `detect_beads` finds, a `settings.roi_size` ROI each."""
    settings = settings or CalibrationSettings()
    settings.validate()
    channels = {str(st.axes.get("channel", 0)) for st in stacks}
    if len(channels) > 1:
        raise ValueError("the acquisitions hold several channels; calibrate them one at a time")
    centres = []
    for s, st in enumerate(stacks):
        yx, _ = detect_beads(np.asarray(st.images, np.float32), settings)
        centres += [(s, int(x), int(y)) for y, x in yx]
    if not centres:
        raise ValueError("no isolated beads found: review detection and the bead box")
    return rois_at(stacks, centres, camera, settings.roi_size, slices, settings.saturation_adu)


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
