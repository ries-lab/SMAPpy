"""4Pi bead stacks into a learned 4Pi calibration: read, find the beads, cut each channel, learn, save.

The Ries-lab 4Pi's bead scans (`io.quadrants`) hold each channel's quadrant as a (z, mirror step, y, x)
stack.  This module turns a set of them into what uiPSF's 4Pi learning takes (`uipsf.workflows.fourpi`,
docs/4pi.md, D7) and the result into SMAPpy's `FourPiCalibration`.  No Qt: the PSF calibration window
calls it as a script does.

* **Photons** are (ADU − offset) · conversion, the camera's own numbers.
* **The beads** are found on the light of all channels, slices and mirror steps summed: the
  interference cancels across channels a quarter turn apart, so the sum is a plain bead.  The finder is
  the bead calibration's (`core.detect_beads`).
* **Each channel's ROI** is centred on the bead's centroid in that channel's own summed image, to the
  pixel: the quadrants are registered to about a pixel by the microscope, and the learning refines the
  maps between them (``ch<c>.transform``).
* **z** is minus the stage position from the middle slice (``Z Positions.txt``, else the step given):
  the stage moves the sample, so a bead rises through the focus as the stage rises, and SMAPpy's z
  falls -- the direction uiPSF v1's model of the same beads gives them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass
class FourPiBeads:
    """The beads of a set of 4Pi stacks: photons (bead, channel, slice, mirror step, y, x), each channel's
    ROI centre on its quadrant (bead, channel, (x, y)), the slices' z (nm), and where each came from."""
    data: np.ndarray
    roi_xy: np.ndarray
    z_nm: np.ndarray
    records: list = field(default_factory=list)

    def batch(self, read_noise_var: float = 1.0):
        """The uiPSF batch: observations z-major (slice, then mirror step), each slice's z repeated."""
        import torch
        import uipsf
        E, C, J, M, Y, X = self.data.shape
        return uipsf.Batch(data=torch.as_tensor(self.data.reshape(E, C, J * M, Y, X), dtype=torch.float32),
                           z_nm=torch.as_tensor(np.repeat(self.z_nm, M), dtype=torch.float32),
                           s=float(read_noise_var), roi_xy=torch.as_tensor(self.roi_xy, dtype=torch.float32))


def stage_z(path, n_slices: int, step_nm: Optional[float]) -> np.ndarray:
    """z of each slice, minus the stage from the middle slice: ``step_nm`` per slice, or the median step of
    the scan's ``Z Positions.txt`` (µm) beside it.  The median, not the positions: the example scans
    repeat their first reading (54.3205 µm twice, then steps of 50 ± 1 nm), the stage read before it
    moved, and taken as they are the first two slices would be one."""
    from ..io.quadrants import read_z_positions
    if step_nm is None:
        positions = read_z_positions(Path(path).parent)
        if positions is None or len(positions) != n_slices:
            raise ValueError(f"{Path(path).name}: no Z Positions.txt with {n_slices} positions; give the z step")
        step_nm = float(np.median(np.diff(positions))) * 1000
    return -(np.arange(n_slices) - (n_slices - 1) // 2) * float(step_nm)


def find_beads(paths: Sequence, conversion: float, offset: float, roi_size: int = 21, step_nm=None,
               settings=None, max_beads: Optional[int] = None) -> FourPiBeads:
    """Every isolated bead in the 4Pi stacks ``paths`` (module docstring), cut to ``roi_size`` pixels in
    every channel.  ``settings`` the bead finder's (`core.CalibrationSettings`), its box set to
    ``roi_size``."""
    from dataclasses import replace

    from ..io.quadrants import read_quadrant_mat
    from .core import CalibrationSettings, detect_beads
    settings = replace(settings or CalibrationSettings(), roi_size=int(roi_size))
    half = roi_size // 2
    data, roi_xy, records, z_ref = [], [], [], None
    for k, path in enumerate(paths):
        stack = (read_quadrant_mat(path).astype(np.float32) - float(offset)) * float(conversion)
        C, J, M, H, W = stack.shape
        z = stage_z(path, J, step_nm)
        if z_ref is None:
            z_ref = z
        elif len(z) != len(z_ref) or np.max(np.abs(z - z_ref)) > 5:
            raise ValueError(f"{Path(path).name}: its slices are not the other stacks' ({len(z)} slices)")
        summed = stack.sum(0).reshape(J * M, H, W)
        found, _ = detect_beads(summed, settings)
        per_channel = stack.sum((1, 2))                                        # (C, H, W)
        for y0, x0 in found:
            centres = [_centroid(per_channel[c], y0, x0, half) for c in range(C)]
            if any(not (half <= cy < H - half and half <= cx < W - half) for cy, cx in centres):
                continue
            data.append(np.stack([stack[c, :, :, cy - half:cy + half + 1, cx - half:cx + half + 1]
                                  for c, (cy, cx) in enumerate(centres)]))
            roi_xy.append([(cx, cy) for cy, cx in centres])
            records.append({"stack": k, "path": str(path), "x_px": int(x0), "y_px": int(y0)})
            if max_beads is not None and len(data) >= max_beads:
                break
        if max_beads is not None and len(data) >= max_beads:
            break
    if not data:
        raise ValueError("no isolated beads were found in the 4Pi stacks")
    return FourPiBeads(np.stack(data), np.asarray(roi_xy, float), z_ref, records)


def _centroid(image, y0, x0, half):
    """The pixel nearest the background-free centroid of ``image`` in the box around (y0, x0)."""
    box = image[max(y0 - half, 0):y0 + half + 1, max(x0 - half, 0):x0 + half + 1].astype(float)
    w = np.clip(box - np.median(box), 0, None)
    if w.sum() <= 0:
        return y0, x0
    yy, xx = np.mgrid[:box.shape[0], :box.shape[1]]
    return (int(round((w * yy).sum() / w.sum())) + max(y0 - half, 0),
            int(round((w * xx).sum() / w.sum())) + max(x0 - half, 0))


def learn(beads: FourPiBeads, NA: float, wavelength_nm: float, n_imm: float, pixel_size_nm,
          steps_rad=(2.0, 0.0, -2.0), n_max: int = 6, read_noise_var: float = 1.0, bead_radius_nm: float = 0.0,
          photons_per_observation: bool = True, device=None, progress=None, cancel=None):
    """uiPSF's 4Pi model learnt from ``beads`` (`uipsf.workflows.fourpi.learn_fourpi`): a `FourPiFit`.
    ``steps_rad`` the deformable mirrors' steps as set; which way round they act is found from the data.
    A free photon count per observation by default: on the example beads it took the mean deviance from
    1.16 to 1.06, the largest term found (docs/4pi.md §8)."""
    import uipsf
    from uipsf.model import PerObservation
    from uipsf.model.fourpi import FourPiModel
    from uipsf.workflows.fourpi import learn_fourpi
    px = (pixel_size_nm, pixel_size_nm) if np.isscalar(pixel_size_nm) else tuple(pixel_size_nm)
    optics = uipsf.Optics(NA=NA, wavelength_nm=wavelength_nm, n_imm=n_imm, n_med=n_imm, pixel_size_nm=px)
    grid = uipsf.PupilGrid.build(optics, 64, device=device)
    batch = beads.batch(read_noise_var)
    C, M = beads.data.shape[1], beads.data.shape[3]
    model = FourPiModel(grid, batch.data.shape[-2:], n_channels=C, n_phases=M, n_max=n_max, steps_rad=steps_rad,
                        bead_radius_nm=bead_radius_nm, intensity=PerObservation() if photons_per_observation else None)
    return learn_fourpi(model, batch, progress=progress, cancel=cancel)


def save(fit, path, z_range_nm: float = 600.0, dz_nm: float = 25.0, roi_px: Optional[int] = None,
         overwrite: bool = False):
    """The learnt model as a `FourPiCalibration` file (`fourpi.save_fourpi_calibration`); the calibration."""
    from uipsf.workflows.fourpi import calibration

    from ..fourpi import save_fourpi_calibration
    cal = calibration(fit, z_range_nm=z_range_nm, dz_nm=dz_nm, roi_px=roi_px)
    save_fourpi_calibration(path, cal, overwrite=overwrite)
    return cal
