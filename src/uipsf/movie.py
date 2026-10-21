"""Simulated blinking movies, imaged with a uipsf model: the test data of in situ learning (plan §9, M5).

In situ learning starts from the camera's frames: molecules blink at random places and heights, overlap,
sit under an uneven illumination and on a background.  Detection, selection and the rounds over the
movie (M5's steps 3–5) are tested on movies made here, whose truth is known and whose spots are the
forward model's own: a spot cut from a frame around its pixel is the μ the fit computes for that emitter
(`tests/uipsf_tests/test_movie.py`), so a later test measures the pipeline, not a mismatch of models.  One
difference is the fit's: its region blurs circularly (`optics.kernels`), so at the edges of an 11-pixel
region it is 0.2–0.6 % of the peak off the movie, which has no wrap-around.

**What is simulated.**
- *Fluorophores*, uniform over the field (whole frame), at heights uniform in ``heights_nm`` above the
  coverslip; a fraction ``on_coverslip`` exactly at the coverslip, the edge the focus is pinned by
  (plan §9, M5).
- *Blinking*: SMAPpy's model (`smappy.simulate.kinetics`): on, off and bleached in continuous time, the
  photons of a frame by the time the molecule was on in it, so the first and last frame of a blink are
  dimmer, as in real data.  ``blinking.photons`` is per blink.
- *Illumination*: a Gaussian of full width at half maximum ``illumination_fwhm_px`` about the field's
  centre (None: flat) scales the molecules' photons, not the background, as a laser does.
- *Spots*: the model's PSF (main channel) with its learnt pupil, blur and focus, `InSituGeometry`'s
  defocus, in a region of ``spot_px`` pixels about the spot's pixel, added into the frame; light outside
  that region is lost, as the fit loses it outside its region.
- *Camera*: a flat background of ``background`` photons per pixel, Poisson noise, Gaussian read noise of
  variance ``read_noise_var`` (photons²).  Frames are in photons; SMAPpy's camera conversion is the
  loader's business, not the simulator's.

**Positions** are chip pixels, zero-based centres (`docs/uipsf/conventions.md`): pixel i covers
[i − ½, i + ½].  In the truth, x and y are nm from pixel 0's centre and h is the height above the
coverslip.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .model import VectorPSF
from .model.channels import view
from .model.forward import _split


@dataclass
class MovieSettings:
    size_px: tuple[int, int] = (64, 64)          # (Y, X)
    n_frames: int = 200
    fluorophores: int = 300
    heights_nm: tuple[float, float] = (0.0, 1500.0)
    on_coverslip: float = 0.0                   # fraction of fluorophores at h = 0
    blinking: object = None                     # smappy.simulate.BlinkingSettings; None: its defaults
    illumination_fwhm_px: float | None = None
    background: float = 10.0                    # photons per pixel and frame
    read_noise_var: float = 0.0                 # photons²
    spot_px: int = 25                           # the region each spot is rendered in
    seed: int = 0


@dataclass
class MovieTruth:
    """One row per molecule and frame it is on in, sorted by frame."""
    frame: np.ndarray            # [M] int
    fluorophore: np.ndarray      # [M] which molecule
    blink: np.ndarray            # [M] which of its blinks: a blink's frames are one linked emitter
    x_nm: np.ndarray             # [M] from pixel 0's centre
    y_nm: np.ndarray
    h_nm: np.ndarray             # [M] height above the coverslip
    photons: np.ndarray          # [M] expected photons in that frame, illumination included

    def __len__(self) -> int:
        return len(self.frame)

    def rows(self, index) -> MovieTruth:
        return MovieTruth(*(getattr(self, f)[index] for f in self.__dataclass_fields__))


@dataclass
class Movie:
    frames: np.ndarray           # [T, Y, X] photons, with noise
    expected: np.ndarray         # [T, Y, X] the noise-free mean
    truth: MovieTruth
    pixel_size_nm: tuple[float, float]
    settings: MovieSettings = field(default_factory=MovieSettings)


def illumination(settings: MovieSettings, x_px, y_px) -> np.ndarray:
    """The relative excitation at chip pixels (``x_px``, ``y_px``): 1 at the field's centre."""
    if settings.illumination_fwhm_px is None:
        return np.ones(np.broadcast(x_px, y_px).shape)
    Y, X = settings.size_px
    s = settings.illumination_fwhm_px / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-((np.asarray(x_px) - (X - 1) / 2) ** 2 + (np.asarray(y_px) - (Y - 1) / 2) ** 2) / (2 * s * s))


def simulate_movie(model, shared: dict, settings: MovieSettings | None = None) -> Movie:
    """A blinking movie seen through ``model`` (an `InSituGeometry` model) with ``shared`` parameters,
    physical values: the pupil, blur and focus to image with (module docstring)."""
    from smappy.simulate import BlinkingSettings, blink, emission
    s = settings or MovieSettings()
    rng = np.random.default_rng(s.seed)
    Y, X = s.size_px
    px, py = model.pixel_size_nm
    n = int(s.fluorophores)
    xy_px = np.stack([rng.uniform(-0.5, X - 0.5, n), rng.uniform(-0.5, Y - 0.5, n)], 1)
    h = rng.uniform(*s.heights_nm, n)
    h[rng.random(n) < s.on_coverslip] = 0.0
    blinks = blink(n, s.n_frames, s.blinking or BlinkingSettings(), rng)
    em = emission(blinks, s.n_frames)
    owner = em.owner
    photons = em.photons * illumination(s, xy_px[owner, 0], xy_px[owner, 1])
    truth = MovieTruth(em.frame.astype(np.int64), owner, em.blink, xy_px[owner, 0] * px, xy_px[owner, 1] * py,
                       h[owner], photons)
    expected = np.full((s.n_frames, Y, X), float(s.background))
    render(model, shared, truth, expected, s.spot_px)
    frames = rng.poisson(expected).astype(np.float64)
    if s.read_noise_var > 0:
        frames += rng.normal(0.0, np.sqrt(s.read_noise_var), frames.shape)
    return Movie(frames.astype(np.float32), expected.astype(np.float32), truth, (px, py), s)


def render(model, shared: dict, truth: MovieTruth, image: np.ndarray, spot_px: int, chunk: int = 512) -> None:
    """Add every row of ``truth`` into ``image`` [T, Y, X], in place: the model's PSF in a region of
    ``spot_px`` pixels about the spot's pixel, times its photons."""
    base = model.psf
    psf = VectorPSF(base.grid, base.pupil, (spot_px, spot_px), upsample=base.upsample, kind=base.kind,
                    analyser=base.analyser)
    dev, dt = base.grid.rho.device, base.grid.rho.dtype
    p = view({k: torch.as_tensor(v, device=dev) for k, v in shared.items()}, 0)
    px, py = model.pixel_size_nm
    half = spot_px // 2
    T, Y, X = image.shape
    for a in range(0, len(truth), chunk):
        rows = truth.rows(slice(a, a + chunk))
        cx, cy = rows.x_nm / px, rows.y_nm / py
        ix, iy = np.rint(cx).astype(int), np.rint(cy).astype(int)
        # x, y from the region's centre pixel; the geometry gives the defocus and the height
        xyz = torch.as_tensor(np.stack([(cx - ix) * px, (cy - iy) * py, rows.h_nm], 1), dtype=dt, device=dev)
        q = dict(p, **{"emitter.xyz": xyz})
        batch = _FrameBatch(xyz.new_zeros(1))
        pos, depth = _split(model.geometry(q, batch))
        with torch.no_grad():
            spots = (psf(p, pos, depth)[:, 0] * torch.as_tensor(rows.photons, dtype=dt, device=dev)[:, None, None])
        spots = spots.cpu().double().numpy()
        for k in range(len(rows)):
            x0, y0 = ix[k] - half, iy[k] - half
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + spot_px, X), min(y0 + spot_px, Y)
            if xb > xa and yb > ya:
                image[rows.frame[k], ya:yb, xa:xb] += spots[k, ya - y0:yb - y0, xa - x0:xb - x0]


@dataclass
class _FrameBatch:
    """What a geometry reads of a batch: one observation, at the nominal focus."""
    z_nm: torch.Tensor


def close_pairs(truth: MovieTruth, distance_nm: float) -> np.ndarray:
    """[M] bool: rows with another molecule on in the same frame closer than ``distance_nm`` laterally."""
    out = np.zeros(len(truth), bool)
    for f in np.unique(truth.frame):
        idx = np.flatnonzero(truth.frame == f)
        if len(idx) < 2:
            continue
        d = np.hypot(truth.x_nm[idx, None] - truth.x_nm[None, idx], truth.y_nm[idx, None] - truth.y_nm[None, idx])
        np.fill_diagonal(d, np.inf)
        out[idx] = d.min(1) < distance_nm
    return out
