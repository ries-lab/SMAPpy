"""Simulated beads imaged at several stage positions: what *Pixel size from stage* measures.

A slide of beads is moved by the stage along a raster (serpentine by default,
as a stage scan usually goes) and imaged once at each position.  The camera
sees the sample turned, mirrored and scaled by its pixel (`pixel_size.camera_matrix`),
and the stage arrives where it was sent plus half its ``backlash`` in the
direction it last moved along each axis -- the hysteresis the measurement
fits.  The truth is kept beside the spots: which bead each is
(``bead_true``), the matrix, and where the stage really was.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class StageScan:
    """Spots per frame in camera pixels, the stage as commanded, and the truth."""
    stage_um: np.ndarray                # (n_frames, 2) as commanded, what the file records
    stage_true_um: np.ndarray           # where the stage really was: with its backlash and noise
    frame: np.ndarray                   # per spot
    x_px: np.ndarray
    y_px: np.ndarray
    photons: np.ndarray
    bead_true: np.ndarray
    matrix: np.ndarray                  # pixels per micrometre: offset = matrix @ stage
    pixel_nm: Tuple[float, float]
    angle_deg: float
    mirrored: bool
    backlash_nm: Tuple[float, float]
    shape: Tuple[int, int]              # (height, width) of a frame, pixels
    frames: Optional[np.ndarray] = None  # (n_frames, height, width) ADU, when drawn

    @property
    def n_frames(self) -> int:
        return len(self.stage_um)

    @property
    def xy_px(self) -> np.ndarray:
        return np.column_stack([self.x_px, self.y_px])


def raster(n_x: int, n_y: int, step_um: Tuple[float, float], serpentine: bool = True
           ) -> np.ndarray:
    """(n_x n_y, 2) stage positions row by row, every other row backwards when ``serpentine``."""
    out = []
    for row in range(n_y):
        columns = range(n_x) if not serpentine or row % 2 == 0 else range(n_x - 1, -1, -1)
        out += [(c * step_um[0], row * step_um[1]) for c in columns]
    return np.asarray(out, float)


def simulate_stage_scan(n_x: int = 6, n_y: int = 5, step_um=(8.0, 8.0), *,
                        pixel_nm=(117.0, 113.0), angle_deg: float = 1.5,
                        mirrored: bool = False, backlash_nm=(0.0, 0.0),
                        stage_noise_nm: float = 5.0, shape=(128, 128),
                        density_per_um2: float = 0.01, precision_px: float = 0.03,
                        p_missed: float = 0.05, serpentine: bool = True,
                        stage_um: Optional[np.ndarray] = None, images: bool = False,
                        photons: float = 20000.0, background: float = 50.0,
                        sigma_px: float = 1.3, seed: int = 0) -> StageScan:
    """Beads at random over the region the scan sees, at ``density_per_um2``, imaged at each stage
    position of a raster of ``n_x`` by ``n_y`` steps of ``step_um`` (or at ``stage_um``, (n, 2)).

    Each frame holds the beads that fall on its ``shape`` (height, width), localized with
    ``precision_px`` and missed at ``p_missed``.  ``images`` also draws the frames, Gaussian spots
    of ``sigma_px`` on ``background`` photons with shot and read noise (`camera._draw`,
    `camera._camera`: conversion 1, offset 100)."""
    from ..pixel_size import approach_directions, camera_matrix
    rng = np.random.default_rng([seed, 31])
    stage = raster(n_x, n_y, step_um, serpentine) if stage_um is None \
        else np.asarray(stage_um, float)
    directions = approach_directions(stage)
    true = stage + 0.5e-3 * np.asarray(backlash_nm, float) * directions \
        + rng.normal(0.0, stage_noise_nm * 1e-3, stage.shape)
    matrix = camera_matrix(tuple(pixel_nm), angle_deg, mirrored)
    height, width = shape
    centre = np.array([width / 2.0, height / 2.0])
    # where the field sees the sample at stage 0: the camera's corners, back into micrometres
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], float) - centre
    field = corners @ np.linalg.inv(matrix).T
    lo = field.min(axis=0) - true.max(axis=0) - 1.0
    hi = field.max(axis=0) - true.min(axis=0) + 1.0
    n_beads = rng.poisson(density_per_um2 * np.prod(hi - lo))
    beads = rng.uniform(lo, hi, (n_beads, 2))
    frames, x, y, bead = [], [], [], []
    for f, s in enumerate(true):
        where = (beads + s) @ matrix.T + centre
        inside = np.all((where >= 2.0) & (where <= np.array([width, height]) - 3.0), axis=1)
        inside &= rng.random(n_beads) >= p_missed
        k = np.flatnonzero(inside)
        frames.append(np.full(len(k), f))
        x.append(where[k, 0])
        y.append(where[k, 1])
        bead.append(k)
    frame = np.concatenate(frames).astype(np.int64)
    n = len(frame)
    x = np.concatenate(x) + rng.normal(0.0, precision_px, n)
    y = np.concatenate(y) + rng.normal(0.0, precision_px, n)
    scan = StageScan(stage, true, frame, x, y, rng.poisson(photons, n).astype(float),
                     np.concatenate(bead).astype(np.int64), matrix, tuple(pixel_nm),
                     float(angle_deg), bool(mirrored), tuple(backlash_nm), (height, width))
    if images:
        from .camera import _camera, _draw
        image = np.full((len(stage), height, width), float(background))
        exact_x, exact_y = x, y
        widths = np.full(n, sigma_px * np.sqrt(2.0))
        _draw(image, frame, np.column_stack([exact_x, exact_y]), scan.photons, widths, widths)
        scan.frames = _camera(image, rng, conversion=1.0, offset=100.0, read_noise=1.5)
    return scan


def write_tiff(scan: StageScan, path, x_tag: str = "XPositionUm",
               y_tag: str = "YPositionUm", tags: Optional[dict] = None) -> Path:
    """The drawn frames as a Micro-Manager-like TIFF: each page with its stage position in its
    per-frame metadata (``MicroManagerMetadata``), as `smappy.frametags` reads it, and ``tags``
    -- the devices' states a microscope entry recognises -- beside it."""
    import tifffile
    if scan.frames is None:
        raise ValueError("simulate the scan with images=True to write it")
    path = Path(path)
    with tifffile.TiffWriter(path) as writer:
        for f, image in enumerate(scan.frames):
            md = json.dumps(dict(tags or {}, FrameIndex=f,
                                 **{x_tag: float(scan.stage_um[f, 0]),
                                    y_tag: float(scan.stage_um[f, 1])}))
            writer.write(image, metadata=None,
                         extratags=[(51123, "s", len(md) + 1, md, False)])
    return path
