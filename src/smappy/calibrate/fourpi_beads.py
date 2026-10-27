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

from ..plugins import param


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
         overwrite: bool = False, layout=None):
    """The learnt model as a `FourPiCalibration` file (`fourpi.save_fourpi_calibration`); the calibration.
    ``layout`` (a `channels.ChannelLayout`) is the geometry its maps are between, kept with it."""
    from uipsf.workflows.fourpi import calibration

    from ..fourpi import save_fourpi_calibration
    cal = calibration(fit, z_range_nm=z_range_nm, dz_nm=dz_nm, roi_px=roi_px)
    if layout is not None:
        cal.layout = layout.to_dict()
    save_fourpi_calibration(path, cal, overwrite=overwrite)
    return cal


# ---------------------------------------------------------------- any scan, through the channel layer
#
# The PSF calibration window's *4Pi beads*.  A scan comes in as the fitter reads its data: through the
# channel layer (`channels`), whatever wrote it -- quadrant .mat scans with their mirror steps, or a
# DCIMG z stack of one phase per slice, the four images on one chip.  The beads are found on channel 0's
# image summed over the scan, the channels registered from the same scans (`channels.register_channels`)
# so that every channel's ROI is cut where the bead is, and those ROIs are what uiPSF learns from; it
# refines the maps itself.


LAYOUTS = (("file", "as the file lays them out"), ("microscope", "the microscope's"),
           ("typed", "typed corners"), ("yale", "4Pi-SMS (Bewersdorf lab)"))


@dataclass
class FourPiBeadSettings:
    """The *4Pi beads* section of the PSF calibration window."""
    layout: str = param("file", label="channels", choices=LAYOUTS,
                        help="where the phase channels are: the quadrant files' own layout, the microscope "
                             "database's, typed, or the 4Pi-SMS design")
    x_corners: str = param("0, 200, 400, 600", label="x corners", unit="px",
                           help="each channel's left edge on the chip, comma-separated")
    y_corners: str = param("0, 0, 0, 0", label="y corners", unit="px",
                           help="each channel's top edge on the chip")
    width: int = param(200, label="width", unit="px", min=8, help="one channel's width")
    height: int = param(200, label="height", unit="px", min=8, beside=True, help="one channel's height")
    mirrored: str = param("", label="mirrored in x",
                          help="the channels mirrored left-right, by number from 1; empty: none")
    z_step_nm: float = param(0.0, label="z step", unit="nm", min=0.0,
                             help="the stage's step between slices; 0: from the scan's own record")
    mirror_steps: str = param("2, 0, -2", label="mirror steps", unit="rad",
                              help="a scan's phase steps per slice, as set; a scan of one phase per slice "
                                   "has none")
    roi_size: int = param(21, label="box", unit="px", min=9, help="the box each bead is cut to, per channel")
    n_max: int = param(6, label="Zernike order", min=2, max=10, advanced=True,
                       help="the highest Zernike order of each objective's pupil")
    photons_per_observation: bool = param(True, label="photons per frame", advanced=True,
                                          help="a bead's photon count free in every frame, as its light varies")
    model_z_range_nm: float = param(600.0, label="model z range", unit="nm", min=100.0, advanced=True,
                                    help="+- this: the z range the saved model covers")
    model_dz_nm: float = param(20.0, label="model z step", unit="nm", min=5.0, advanced=True,
                               help="the saved model's z step")


def scan_layout(settings: FourPiBeadSettings, source, camera=None):
    """The layout of a scan's frames: the source's own, the microscope's for ``camera``, typed or 4Pi-SMS,
    the last three on the chip and put on the camera's ROI."""
    from ..channels import YALE_4PI, from_corners
    roi = camera.roi_offset if camera is not None else (0, 0)
    if settings.layout == "file":
        layout = getattr(source, "layout", None)
        if layout is None:
            raise ValueError("this file does not lay out its channels itself (only the quadrant .mat files "
                             "do): use the microscope's channels, or type them")
        return layout
    if settings.layout == "yale":
        return YALE_4PI.on_camera(roi)
    if settings.layout == "microscope":
        from ..microscope_db import database
        scope = (getattr(camera, "microscope", None) or {}).get("name") if camera is not None else None
        entry = database().get(scope) if scope else None
        layout = entry.channel_layout(getattr(camera, "camera_name", "") or "") if entry else None
        if layout is None:
            raise ValueError("the microscope database gives no channel rectangles for these stacks' "
                             "microscope: choose one that has them, or type the corners")
        return layout.on_camera(roi)
    numbers = lambda text: [int(round(float(w))) for w in str(text).replace(";", ",").split(",") if w.strip()]
    x, y = numbers(settings.x_corners), numbers(settings.y_corners)
    if len(x) != len(y) or not 1 <= len(x) <= 4:
        raise ValueError("give one x and one y corner per channel, one to four of them")
    mirrored = set(numbers(settings.mirrored))
    return from_corners(x, y, settings.width, settings.height,
                        flip_x=[c + 1 in mirrored for c in range(len(x))]).on_camera(roi)


def scan_step_nm(path, given: float = 0.0) -> float:
    """The stage's step between slices: ``given``, or the scan's own record -- the median step of a
    ``Z Positions.txt`` beside quadrant files (`stage_z`), or the ``Step Size (um)`` of the ``.txt`` a
    4Pi-SMS DCIMG scan's folder holds."""
    if given:
        return float(given)
    from ..io.quadrants import read_z_positions
    folder = Path(path).parent
    positions = read_z_positions(folder)
    if positions is not None and len(positions) > 1:
        return float(np.median(np.diff(positions))) * 1000
    import re
    for text in sorted(folder.glob("*.txt")) + sorted(folder.parent.glob("*.txt")):
        found = re.search(r"Step Size \(um\):\s*([0-9.]+)", text.read_text(encoding="utf-8", errors="replace"))
        if found:
            return float(found.group(1)) * 1000
    raise ValueError(f"{Path(path).name}: no record of the z step beside it; give the z step")


def read_scan(path, settings: FourPiBeadSettings, camera):
    """One scan in photons as (channel, slice, mirror step, y, x), and its layout.  Quadrant files keep
    their mirror steps; any other scan is one phase per slice, its frames the slices."""
    from ..camera import to_photons
    from ..channels import ChannelSource
    from ..io.quadrants import is_quadrant_mat, read_quadrant_mat
    from ..io.tiff import open_stack
    if is_quadrant_mat(path) and settings.layout == "file":
        from ..channels import side_by_side
        raw = read_quadrant_mat(path)                                          # (C, J, M, y, x)
        C, J, M, H, W = raw.shape
        layout = side_by_side(C, W, H, name="quadrant files")
        return to_photons(raw, camera), layout
    base = open_stack(path)
    layout = scan_layout(settings, base, camera)
    source = ChannelSource.wrap(base, layout)
    frames = np.concatenate([b for _, b in source.frames(chunk=200)])         # (J, C, h, w)
    return to_photons(frames, camera).transpose(1, 0, 2, 3)[:, :, None], layout


def scan_beads(paths, settings: FourPiBeadSettings, camera, progress=None):
    """Every bead of the scans ``paths`` in every channel (`FourPiBeads`), with the layout and the
    registration the ROIs were cut by.  A series of files (one DCIMG per slice) is read once, through
    any of its files."""
    from ..channels import bead_positions, register_channels
    tell = progress or (lambda text: None)
    scans, seen = [], set()
    for path in paths:
        from ..io.tiff import open_stack
        key = tuple(str(p) for p in getattr(open_stack(path), "files", [path]))
        if key in seen:
            continue
        seen.add(key)
        tell(f"reading {Path(path).name} ({len(scans) + 1})")
        stack, layout = read_scan(path, settings, camera)
        scans.append((path, stack, layout))
    layout = scans[0][2]
    if any(s[2].size != layout.size or s[2].n_channels != layout.n_channels for s in scans):
        raise ValueError("the scans' channels differ in size or number")
    projections = [s[1].sum(axis=(1, 2)) for s in scans]                      # (C, h, w) each
    tell("registering the channels")
    registration = register_channels(projections)
    layout = layout.with_maps(registration.maps)
    half = int(settings.roi_size) // 2
    data, roi_xy, records = [], [], []
    steps = scans[0][1].shape[2]
    for k, (path, stack, own) in enumerate(scans):
        C, J, M, H, W = stack.shape
        if M != steps:
            raise ValueError("the scans' mirror steps differ")
        for p0 in bead_positions(projections[k][0]):
            centres = [np.rint(p0 if c == 0 else layout.to_channel(c, [p0])[0]).astype(int) for c in range(C)]
            if any(not (half <= cx < W - half and half <= cy < H - half) for cx, cy in centres):
                continue
            data.append(np.stack([stack[c, :, :, cy - half:cy + half + 1, cx - half:cx + half + 1]
                                  for c, (cx, cy) in enumerate(centres)]))
            roi_xy.append([(cx, cy) for cx, cy in centres])
            x_frame, y_frame = own.on_frame(0, [p0])[0]
            records.append({"stack": k, "path": str(path), "x_px": int(round(x_frame)),
                            "y_px": int(round(y_frame)),
                            "brightness": float(stack[0, :, :, centres[0][1], centres[0][0]].max())})
    if not data:
        raise ValueError("no bead lies far enough from every channel's edge for its box")
    J = scans[0][1].shape[1]
    z_nm = -(np.arange(J) - (J - 1) // 2) * scan_step_nm(scans[0][0], settings.z_step_nm)
    return FourPiBeads(np.stack(data), np.asarray(roi_xy, float), z_nm, records), layout, registration


def steps_of(settings: FourPiBeadSettings, n_steps: int) -> tuple:
    """The mirror steps the learning starts from: as set for a scan with ``n_steps`` of them, (0,) for
    one phase per slice."""
    if n_steps == 1:
        return (0.0,)
    steps = tuple(float(w) for w in str(settings.mirror_steps).replace(";", ",").split(",") if w.strip())
    if len(steps) != n_steps:
        raise ValueError(f"the scans have {n_steps} mirror steps per slice; give {n_steps} under mirror steps")
    return steps


# ---------------------------------------------------------------- after a calibration

def bead_fits(beads: FourPiBeads, cal, box: int = 15) -> dict:
    """Every frame of every bead fitted with the saved model (`psf.Global4PiPSF`), each channel's box at
    its own pixel and the rest of the model's map in the link: z and the phase against the stage, and per
    bead the slope of z on the stage and its scatter about it over the middle half of the scan."""
    from ..fourpi import link
    from ..psf import Global4PiPSF
    E, C, J, M, H, W = beads.data.shape
    o = (H - box) // 2
    rois = beads.data[..., o:o + box, o:o + box].transpose(0, 2, 3, 1, 4, 5).reshape(-1, C, box, box)
    maps = cal.maps()
    dx, dy = np.zeros((E, C)), np.zeros((E, C))
    if maps is not None:
        from ..channels import ChannelLayout
        layout = ChannelLayout([(0, 0, W, H)] * C, maps=maps)
        for c in range(1, C):
            there = layout.to_channel(c, beads.roi_xy[:, 0])
            dx[:, c], dy[:, c] = (there - beads.roi_xy[:, c]).T
    per = J * M
    lk = link(len(rois), C, dx=np.repeat(dx, per, 0), dy=np.repeat(dy, per, 0))
    model = Global4PiPSF(cal)
    values = model.unpack(model.fit(np.ascontiguousarray(rois, np.float32), lk, iterations=100), lk)
    z = values["z_nm"].reshape(E, J, M).astype(float)
    # beyond the model's z range a fit is held at its edge (`Global4PiPSF.unpack`): no z at all
    z[values["z_err_nm"].reshape(E, J, M) >= cal.shape[0] * cal.dz] = np.nan
    phase = values["phase_rad"].reshape(E, J, M)
    middle = slice(J // 4, 3 * J // 4)
    slope, scatter, phase_slope = {}, {}, {}
    for b in range(E):
        zz, st = z[b, middle].ravel(), np.repeat(beads.z_nm[middle], M)
        ok = np.isfinite(zz)
        if ok.sum() > 3:
            p = np.polyfit(st[ok], zz[ok], 1)
            slope[b], scatter[b] = float(p[0]), float(np.std(zz[ok] - np.polyval(p, st[ok])))
        # z from the phase, unwrapped along the scan, against the stage: the model's period against the
        # data's -- off by as much as the emission wavelength typed is off
        from_phase = np.unwrap(phase[b, middle, 0]) / cal.phase_per_nm
        phase_slope[b] = float(np.polyfit(beads.z_nm[middle], from_phase, 1)[0])
    return {"z": z, "phase": phase, "stage": beads.z_nm, "slope": slope, "scatter": scatter,
            "phase_slope": phase_slope, "phase_per_nm": float(cal.phase_per_nm)}


def draw_model(figure, cal) -> str:
    """The saved model: each channel's incoherent part through focus (x against z) and its modulation
    |A, B| / I at the centre against z."""
    V = cal.volumes                                   # (C, 3, z, y, x)
    C, _, Z, H, W = V.shape
    axes = figure.subplots(1, C + 1)
    z = cal.z_index_to_nm(np.arange(Z))
    for c in range(C):
        axes[c].imshow(V[c, 0, :, H // 2, :], aspect="auto", cmap="magma",
                       extent=(-W / 2, W / 2, z[-1], z[0]))
        axes[c].set_title(f"channel {c + 1}: I", fontsize="small")
        axes[c].set_xlabel("x (px)")
    axes[0].set_ylabel("z (nm)")
    for c in range(C):
        i, a, b = V[c, 0, :, H // 2, W // 2], V[c, 1, :, H // 2, W // 2], V[c, 2, :, H // 2, W // 2]
        axes[C].plot(z, np.hypot(a, b) / np.maximum(i, 1e-12), label=f"{c + 1}")
    axes[C].set_xlabel("z (nm)")
    axes[C].set_ylabel("modulation at the centre")
    axes[C].set_ylim(0, 1.05)
    axes[C].legend(frameon=False, fontsize="small")
    return f"fringe period {cal.period_nm:.1f} nm, the model's own"


def draw_against_stage(figure, fits: dict, selected=(), excluded=()) -> str:
    """z of every bead against the stage: the envelope's (left) and the phase's, unwrapped (right), each
    less the stage; the selected beads drawn darker, the excluded ones not at all."""
    left, right = figure.subplots(1, 2)
    stage = fits["stage"]
    E, J, M = fits["z"].shape
    for b in range(E):
        if b in excluded:
            continue
        strong = not selected or b in selected
        kw = dict(lw=1.5 if strong else 0.7, alpha=1.0 if strong else 0.35)
        left.plot(stage, np.nanmean(fits["z"][b], axis=1) - stage, **kw)
        phase = np.unwrap(fits["phase"][b, :, 0])
        from_phase = phase / fits["phase_per_nm"]
        offset = np.nanmedian(from_phase - stage)
        right.plot(stage, from_phase - stage - offset, **kw)
    for ax, what in ((left, "envelope z - stage (nm)"), (right, "z from the phase - stage (nm)")):
        ax.set_xlabel("stage z (nm)")
        ax.set_ylabel(what)
        ax.axhline(0, color="0.5", lw=0.5)
    scatter = [v for k, v in fits["scatter"].items() if k not in excluded]
    slopes = [v for k, v in fits.get("phase_slope", {}).items() if k not in excluded]
    if not scatter:
        return ""
    return (f"the envelope's z scatters {np.median(scatter):.1f} nm about the stage (median); z from the "
            f"phase runs at {np.median(slopes):.3f} nm per nm of stage -- far from 1, check the emission "
            "wavelength")
