"""Thin standalone readers: a Micro-Manager z stack to a `BeadStack` (plan §5.7).

SMAPpy reads Micro-Manager and NDTiff stacks, takes z from the metadata and
converts to photons itself; inside SMAPpy uipsf gets arrays, never files.
This reader is for scripts and the CLI, and does the minimum the same way:
- z from each plane's ``ZPositionUm`` when they are strictly monotonic,
  else the summary's ``z-step_um``; planes sorted to increasing objective
  position, which is the order `Batch.z_from_objective` expects;
- the camera is the caller's: conversion, offset, read noise and EM gain
  as a `noise.Camera`.  Inside SMAPpy its camera database supplies them;
  standalone they are typed in, from the camera's data sheet or SMAPpy's
  database (docs/uipsf/data.md has the Zenodo ones);
- saturation from the integer type's full scale, before any conversion.
One file, one channel, one position; anything else is SMAPpy's to split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ..noise import Camera


@dataclass
class BeadStack:
    """One z stack in photons: what SMAPpy's calibration window hands over (plan §5.7)."""
    photons: np.ndarray                    # [K, Y, X], float32
    z_objective_nm: np.ndarray             # [K], increasing
    read_noise_var: float = 0.0            # s of the shifted Poisson (noise.Camera.read_noise_var)
    chip_origin: tuple[int, int] = (0, 0)  # camera ROI offset (x, y): field coordinates
    saturated: np.ndarray | None = None    # [K, Y, X] bool
    source: str = "array"
    camera: Camera | None = None            # the conversion used, for the report
    excess: float = 1.0                     # 2 with EM gain

    @classmethod
    def from_adu(cls, adu, z_objective_nm, camera, chip_origin=(0, 0), source="array"):
        """Converted with a `noise.Camera`, saturation read off the integer type first."""
        adu = np.asarray(adu)
        saturated = adu >= np.iinfo(adu.dtype).max if np.issubdtype(adu.dtype, np.integer) else None
        photons = camera.to_photons(adu.astype(np.float32)).numpy()
        return cls(photons, np.asarray(z_objective_nm, float), camera.read_noise_var, tuple(chip_origin),
                   saturated, source, camera, camera.excess)

    @property
    def z_step_nm(self) -> float:
        return float(np.median(np.diff(self.z_objective_nm)))

    def mirrored(self, axis: str | None) -> BeadStack:
        """The stack flipped along ``axis`` ("x": left–right, "y": up–down; None: unchanged).

        **Legacy.**  An EMCCD reads out its EM port in the opposite direction
        from its normal port, so beads taken with EM off and data with EM on
        are mirror images of each other.  Mirroring the beads makes the PSF
        model match the data it will fit; v1 did the same with ``swapxy``.
        Beads and data taken with the same settings, as is usual now, need
        none.  The chip origin is kept, so field coordinates are those of the
        mirrored image within this stack; a field-dependent model fitted on
        mirrored stacks applies to data in the same frame.
        """
        if axis is None:
            return self
        if axis not in ("x", "y"):
            raise ValueError("mirror axis must be 'x', 'y' or None")
        dim = -1 if axis == "x" else -2

        def flip(a):
            return None if a is None else np.flip(a, dim).copy()
        return BeadStack(flip(self.photons), self.z_objective_nm, self.read_noise_var, self.chip_origin,
                         flip(self.saturated), f"{self.source} (mirrored in {axis})", self.camera, self.excess)


def read_tags(path) -> tuple[dict, list[dict]]:
    """(summary + first plane's tags, every plane's tags) of a Micro-Manager TIFF."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        summary = dict((tf.micromanager_metadata or {}).get("Summary", {}))
        planes = []
        for page in tf.pages:
            tag = page.tags.get("MicroManagerMetadata")
            md = tag.value if tag is not None else {}
            planes.append(json.loads(md) if isinstance(md, str) else dict(md or {}))
    tags = {str(k): v for k, v in {**summary, **(planes[0] if planes else {})}.items()
            if isinstance(v, (str, int, float, bool, list, tuple))}
    return tags, planes


def objective_positions(planes: list[dict], summary_tags: dict, n: int, dz_nm=None) -> np.ndarray:
    """Objective positions of the planes as stored, nm; raises when nothing gives a spacing."""
    if dz_nm is not None:
        return np.arange(n) * float(dz_nm)
    zp = [p.get("ZPositionUm") for p in planes]
    if len(zp) == n and all(v is not None for v in zp):
        z = np.asarray(zp, float) * 1000
        d = np.diff(z)
        if np.all(d > 0) or np.all(d < 0):
            return z
    step = summary_tags.get("z-step_um", summary_tags.get("z_step_um"))
    if step is None or not float(step):
        raise ValueError("no usable z spacing in the metadata; give dz_nm")
    return np.arange(n) * float(step) * 1000


def read_stack(path, camera: Camera, dz_nm=None, mirror: str | None = None) -> BeadStack:
    """A z stack from a Micro-Manager TIFF, in photons with ``camera``, by increasing objective position.

    ``mirror`` ("x" or "y") flips the stack to match data taken with the
    other EM port (`BeadStack.mirrored`); off by default.
    """
    import tifffile
    adu = tifffile.imread(path)
    if adu.ndim != 3:
        raise ValueError(f"{path}: expected one z stack [Z, Y, X], got shape {adu.shape}")
    tags, planes = read_tags(path)
    z = objective_positions(planes, tags, adu.shape[0], dz_nm)
    order = np.argsort(z)
    roi = _roi(tags.get("ROI"))
    origin = (roi[0], roi[1]) if roi else (0, 0)
    return BeadStack.from_adu(adu[order], z[order], camera, origin, str(path)).mirrored(mirror)


def _roi(value):
    """Micro-Manager's camera ROI, "x-y-w-h" or a list, as four ints; None if it is not one."""
    if value is None:
        return None
    parts = str(value).replace("[", " ").replace("]", " ").replace(",", " ").replace("-", " ").split()
    try:
        out = [int(float(p)) for p in parts]
    except ValueError:
        return None
    return out if len(out) == 4 else None


__all__ = ["BeadStack", "read_stack", "read_tags", "objective_positions"]
