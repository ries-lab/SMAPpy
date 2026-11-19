"""Training samples drawn on the device, fresh for every batch: frames as the camera and the fitter
make them, and the molecules in the middle frame as the target.

**Emitters** blink as DECODE draws them (Speiser et al. 2021): each switches on at a time uniform
over the window and ``3 lifetime`` before and after it, stays on for an exponential time of mean
``lifetime_frames``, and gives ``flux * overlap`` photons to each frame its on-time overlaps.  Their
number is Poisson with the mean that puts ``density_per_40px`` on in 40 x 40 pixels of a frame on average --
``rate (lifetime + 1)`` per frame, the 1 for an on-time cut by frame boundaries.  Position uniform
over the frame and ``margin_px`` beyond it (pixel centres at whole numbers), z uniform, flux normal
cut at ``photons_min``.

**Frames**: a background uniform per frame -- or, with ``background_jitter`` at 0 or above, uniform per
movie and jittered by that fraction from frame to frame -- the spots (`render.SplineRenderer`, the calibration as
the camera reads it), then the camera -- Poisson, the EM register (a gamma of shape the electrons,
the excess noise), read noise, rounding to ADU -- and back to photons by the fitter's own factor
(`CameraModel.adu_to_photons`), so the network is trained on what it will be given.

**Targets**: the molecules on in the middle frame, at least ``target_photons_min`` photons in it,
as padded tensors with a mask: columns, rows (pixels), z (nm), photons in that frame; and that
frame's background.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .render import SplineRenderer
from .settings import CameraModel, SimulationSettings


@dataclass
class Batch:
    frames: torch.Tensor          # (b, window, rows, columns), photons as the fitter converts them
    xyz: torch.Tensor             # (b, k, 3): column, row (pixels), z (nm) of the middle frame's molecules
    photons: torch.Tensor         # (b, k): their photons in that frame
    mask: torch.Tensor            # (b, k) bool: which of the k are molecules
    background: torch.Tensor      # (b,): the middle frame's background, photons a pixel
    photons_window: torch.Tensor | None = None     # (b, k, window): the molecules' photons in every frame
    background_window: torch.Tensor | None = None  # (b, window): every frame's background
    # with ``movie_windows`` W > 1: the movies the windows were cut from, (ceil(b / W), window + W - 1,
    # rows, columns), which the network reads whole (`network.TwoStageUNet.forward`'s ``windows``)
    movie: torch.Tensor | None = None
    movie_windows: int = 1


def _uniform(lo, hi, n, g, device):
    return lo + (hi - lo) * torch.rand(n, generator=g, device=device)


def emitters(sim: SimulationSettings, batch: int, g: torch.Generator, device, frames: int | None = None):
    """Each sample's molecules: sample index, x, y, z, flux, and the photons each of its ``frames``
    frames (the window, unless given) gets, ``(n, frames)``.  With ``lifetime_max_frames`` above
    ``lifetime_frames`` each sample has its own mean on-time, log-uniform between them, and the
    molecules per frame stay ``density_per_40px`` whatever it is."""
    T = sim.window if frames is None else frames
    life = lo_life = sim.lifetime_frames
    hi_life = max(sim.lifetime_max_frames, lo_life)
    if hi_life > lo_life:      # drawn only then, so a fixed on-time draws exactly as before
        life = lo_life * (hi_life / lo_life) ** torch.rand(batch, generator=g, device=device)
    lo, hi = -3 * hi_life, T + 3 * hi_life
    area = ((sim.frame_px + 2 * sim.margin_px) / 40.0) ** 2           # the drawn area, in 40 x 40 pixels
    rate = sim.density_per_40px * area / (life + 1)
    rate = torch.full((batch,), rate, device=device) if not torch.is_tensor(rate) else rate
    counts = torch.poisson(rate * (hi - lo), generator=g).long()
    sample = torch.repeat_interleave(torch.arange(batch, device=device), counts)
    n = len(sample)
    start = _uniform(lo, hi, n, g, device)
    mean_on = life[sample] if torch.is_tensor(life) else life
    on = -mean_on * torch.log1p(-torch.rand(n, generator=g, device=device))      # exponential, mean life
    end = start + on
    k = torch.arange(T, device=device, dtype=start.dtype)
    overlap = (torch.minimum(end[:, None], k + 1) - torch.maximum(start[:, None], k)).clamp_min(0)
    edge = -0.5 - sim.margin_px, sim.frame_px - 0.5 + sim.margin_px
    x = _uniform(*edge, n, g, device)
    y = _uniform(*edge, n, g, device)
    z = _uniform(*sim.z_nm, n, g, device)
    flux = (sim.photons_mean + sim.photons_sd * torch.randn(n, generator=g, device=device)).clamp_min(
        max(sim.photons_min, 1e-8))
    return sample, x, y, z, flux, flux[:, None] * overlap


def camera(photons: torch.Tensor, cam: CameraModel, g: torch.Generator) -> torch.Tensor:
    """Expected photons a pixel to what the fitter hands the network: ADU drawn, back to photons."""
    electrons = torch.poisson(photons * cam.qe, generator=g)
    if cam.em_gain:
        # the sum of n exponential multiplications is a gamma of shape n and scale the gain
        shape = electrons.clamp_min(1e-6)
        electrons = torch._standard_gamma(shape) * cam.em_gain * (electrons > 0)
    read = cam.read_noise_e * torch.randn(photons.shape, generator=g, device=photons.device)
    adu = torch.round((electrons + read) / cam.conversion + cam.offset).clamp(0, 65535)
    return (adu - cam.offset) * cam.adu_to_photons


def draw_batch(renderer: SplineRenderer, sim: SimulationSettings, cam: CameraModel, batch: int,
               g: torch.Generator) -> Batch:
    """One batch of fresh samples on the renderer's device: movies of ``window + movie_windows - 1``
    frames, each cut into ``movie_windows`` windows shifted by a frame (one window a movie by default,
    every sample its own)."""
    device = renderer.device
    T, s = sim.window, sim.frame_px
    W = max(1, int(sim.movie_windows))
    n, L = -(-batch // W), T + W - 1
    sample, x, y, z, _, per_frame = emitters(sim, n, g, device, frames=L)
    if sim.background_jitter < 0:
        bg = _uniform(*sim.background, n * L, g, device).view(n, L)
    else:
        level = _uniform(*sim.background, n, g, device)[:, None]
        bg = (level * (1 + sim.background_jitter * torch.randn((n, L), generator=g, device=device))).clamp_min(0)
    image = bg[:, :, None, None].expand(n, L, s, s).contiguous()
    on = per_frame > 0
    m, f = on.nonzero(as_tuple=True)
    renderer.draw(image.view(n * L, s, s), sample[m] * L + f, x[m], y[m], z[m], per_frame[m, f])
    movie = camera(image, cam, g)
    idx = torch.arange(W, device=device)[:, None] + torch.arange(T, device=device)[None]     # (W, T)
    frames = movie[:, idx].reshape(n * W, T, s, s)[:batch]
    bgw = bg[:, idx].reshape(n * W, T)[:batch]
    # each window's middle frame's molecules
    mid = T // 2
    inside = (x >= -0.5) & (x < s - 0.5) & (y >= -0.5) & (y < s - 0.5)   # a target is in the frame
    keep = (per_frame[:, mid:mid + W] > max(sim.target_photons_min, 0.0)) & inside[:, None]     # (molecules, W)
    i, w = keep.nonzero(as_tuple=True)
    ks = sample[i] * W + w
    order = torch.argsort(ks, stable=True)          # grouped by sample; one window a movie: already so
    i, w, ks = i[order], w[order], ks[order]
    use = ks < batch
    i, w, ks = i[use], w[use], ks[use]
    kx, ky, kz, kp = x[i], y[i], z[i], per_frame[i, w + mid]
    counts = torch.bincount(ks, minlength=batch)
    k = int(counts.max()) if len(ks) else 1
    rank = torch.arange(len(ks), device=device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    xyz = torch.zeros((batch, max(k, 1), 3), device=device)
    photons = torch.zeros((batch, max(k, 1)), device=device)
    mask = torch.zeros((batch, max(k, 1)), dtype=torch.bool, device=device)
    xyz[ks, rank] = torch.stack([kx, ky, kz], 1)
    photons[ks, rank] = kp
    window = torch.zeros((batch, max(k, 1), T), device=device)
    window[ks, rank] = per_frame[i[:, None], idx[w]]
    mask[ks, rank] = True
    return Batch(frames, xyz, photons, mask, bgw[:, mid], window, bgw, movie if W > 1 else None, W)
