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

**Uneven backgrounds** (``background_uneven``): a share of the movies has its background multiplied by
a smooth field, fixed over the movie (`background_modulation`); their target is then the map.

**Targets**: the molecules on in the middle frame, at least ``target_photons_min`` photons in it,
as padded tensors with a mask: columns, rows (pixels), z (nm), photons in that frame; and that
frame's background.

**Two channels** (`draw_batch_2c`, `docs/decode/plex-plan.md` §16): crops of the halves' overlap at
random places (`channels.Grid`), so that the network sees every part of the registration; channel 0's
spot at the molecule's position, channel 1's at that plus the registration's remainder ``d``, each
from its own spline; a background per channel and frame; the camera on both; and the crop's offset
maps beside the frames.  Biplane's photons are one count, each plane's share in its spline; colour's
a total split by a share in channel 1 drawn per molecule (`settings.ChannelSettings`).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .render import SplineRenderer
from .settings import CameraModel, SimulationSettings


@dataclass
class Batch:
    # (b, window, rows, columns), photons as the fitter converts them; two channels (b, 2, window, ...)
    frames: torch.Tensor
    xyz: torch.Tensor             # (b, k, 3): column, row (pixels), z (nm) of the middle frame's molecules
    photons: torch.Tensor         # (b, k): their photons in that frame; colour (b, k, 2), each channel's
    mask: torch.Tensor            # (b, k) bool: which of the k are molecules
    background: torch.Tensor      # (b,): the middle frame's background, photons a pixel; two channels (b, 2)
    photons_window: torch.Tensor | None = None     # (b, k, window): the molecules' photons in every frame
    background_window: torch.Tensor | None = None  # (b, window): every frame's background
    # with ``movie_windows`` W > 1: the movies the windows were cut from, (ceil(b / W), window + W - 1,
    # rows, columns), which the network reads whole (`network.TwoStageUNet.forward`'s ``windows``)
    movie: torch.Tensor | None = None
    movie_windows: int = 1
    aux: torch.Tensor | None = None               # two channels: the offset maps, (b, 2, rows, columns)
    # an uneven background (``SimulationSettings.background_uneven``): the middle frame's per pixel,
    # (b, C, rows, columns); None: flat, ``background`` everywhere
    background_map: torch.Tensor | None = None
    # two channels: each target's channel-1 spot relative to its channel-0 one (the registration's
    # remainder), pixels, (b, k, 2) -- what the test set's bound needs (`bounds.drawn_crlb`)
    shift: torch.Tensor | None = None


def _uniform(lo, hi, n, g, device):
    return lo + (hi - lo) * torch.rand(n, generator=g, device=device)


def smooth_fields(n: int, s: int, g: torch.Generator, device, shared: torch.Tensor | None = None,
                  own: float = 0.3) -> torch.Tensor:
    """(n, s, s) smooth random fields in [0, 1]: noise on a coarse grid of 2 to 6 knots a side (one
    draw a batch), bicubic up to the frame, so it varies over 7 to 40 pixels of a 40-pixel frame.
    With ``shared`` (n, s, s), the field is that and ``own`` of a field of its own (a second channel
    sees mostly the same light, `background_modulation`)."""
    import torch.nn.functional as F
    k = int(torch.randint(2, 7, (1,), generator=g, device=device))
    coarse = torch.randn((n, 1, k + 2, k + 2), generator=g, device=device)
    f = F.interpolate(coarse, size=(s + 2 * s // k, s + 2 * s // k), mode="bicubic", align_corners=True)
    a = s // k
    f = f[:, 0, a:a + s, a:a + s]
    if shared is not None:
        f = (1 - own) * shared + own * _unit(f)
    return _unit(f)


def _unit(f: torch.Tensor) -> torch.Tensor:
    lo = f.amin((1, 2), keepdim=True)
    return (f - lo) / (f.amax((1, 2), keepdim=True) - lo).clamp_min(1e-6)


def background_modulation(sim: SimulationSettings, n: int, channels: int, s: int, g: torch.Generator,
                          device) -> torch.Tensor | None:
    """(n, channels, s, s): the factor each movie's background is multiplied by, 1 + c (2 f - 1) with f a
    smooth field (`smooth_fields`) and c uniform up to ``background_contrast``, for a share
    ``background_uneven`` of the movies and 1 elsewhere; None when the share is 0.  The channels share
    most of the field's shape (the same sample's light), each its own contrast."""
    if sim.background_uneven <= 0:
        return None
    first = smooth_fields(n, s, g, device)
    fields = [first] + [smooth_fields(n, s, g, device, shared=first) for _ in range(channels - 1)]
    uneven = torch.rand((n, 1, 1), generator=g, device=device) < sim.background_uneven     # a movie's, both channels
    out = []
    for f in fields:
        c = sim.background_contrast * torch.rand((n, 1, 1), generator=g, device=device)
        out.append(torch.where(uneven, 1 + c * (2 * f - 1), torch.ones_like(f)))
    return torch.stack(out, 1)


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
    if sim.photons_range is not None:
        flux = _uniform(*sim.photons_range, n, g, device)
    else:
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
    mod = background_modulation(sim, n, 1, s, g, device)                         # (n, 1, s, s) or None
    image = bg[:, :, None, None].expand(n, L, s, s).contiguous()
    if mod is not None:
        image = image * mod
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
    bg_map = None if mod is None else mod.repeat_interleave(W, 0)[:batch] * bgw[:, mid, None, None, None]
    return Batch(frames, xyz, photons, mask, bgw[:, mid], window, bgw, movie if W > 1 else None, W,
                 background_map=bg_map)


def draw_batch_2c(renderers, grid, sim: SimulationSettings, cam: CameraModel, channels, batch: int,
                  g: torch.Generator) -> Batch:
    """One batch of fresh two-channel samples (the module's docstring): ``renderers`` the two channels'
    (`render.SplineRenderer`), ``grid`` the halves' remainder (`channels.Grid`), ``channels`` the
    `settings.ChannelSettings`.  Movies of ``window + movie_windows - 1`` frames, each its own crop,
    cut into ``movie_windows`` windows shifted by a frame, as `draw_batch` (one window a movie by
    default, every sample its own): a movie's windows share its crop, offset maps and molecules, so
    drawing costs about 1 / window of what it does a window at a time.  ``aux`` is per window, as with
    one window a movie (`train.run` reads one a movie)."""
    device = renderers[0].device
    T, s = sim.window, sim.frame_px
    W = max(1, int(sim.movie_windows))
    n, L = -(-batch // W), T + W - 1
    sample, x, y, z, _, per_frame = emitters(sim, n, g, device, frames=L)
    r0, c0 = grid.origins(n, s, g)
    aux = grid.maps(r0, c0, s)                                                    # (n, 2, s, s)
    d = grid.at(sample, x, y, r0, c0)                                             # (m, 2)
    ranges = channels.background or [sim.background] * 2
    bg = torch.stack([_uniform(*ranges[c], n * L, g, device).view(n, L) for c in range(2)], 1)   # (n, 2, L)
    if channels.kind == "colour" and channels.dyes:
        dyes = torch.as_tensor(channels.dyes, dtype=per_frame.dtype, device=device)
        which = torch.randint(0, len(dyes), (len(sample),), generator=g, device=device)
        share = (dyes[which] + channels.share_sd * torch.randn(len(sample), generator=g, device=device)).clamp(
            *channels.share)
        per_channel = [per_frame * (1 - share[:, None]), per_frame * share[:, None]]
    elif channels.kind == "colour":
        share = _uniform(*channels.share, len(sample), g, device)
        per_channel = [per_frame * (1 - share[:, None]), per_frame * share[:, None]]
    elif channels.unlinked:
        share = _uniform(*channels.share, len(sample), g, device)
        per_channel = [per_frame * 2 * (1 - share[:, None]), per_frame * 2 * share[:, None]]
    else:
        per_channel = [per_frame, per_frame]
    mod = background_modulation(sim, n, 2, s, g, device)                         # (n, 2, s, s) or None
    image = bg[..., None, None].expand(n, 2, L, s, s).contiguous()
    if mod is not None:
        image = image * mod[:, :, None]
    for c, (renderer, px, py) in enumerate(zip(renderers, (x, x + d[:, 0]), (y, y + d[:, 1]))):
        on = per_channel[c] > 0
        m, f = on.nonzero(as_tuple=True)
        view = image[:, c].contiguous()
        renderer.draw(view.view(n * L, s, s), sample[m] * L + f, px[m], py[m], z[m], per_channel[c][m, f])
        image[:, c] = view
    movie = camera(image, cam, g)                                                 # (n, 2, L, s, s)
    idx = torch.arange(W, device=device)[:, None] + torch.arange(T, device=device)[None]     # (W, T)
    frames = movie[:, :, idx].transpose(1, 2).reshape(n * W, 2, T, s, s)[:batch]
    bgw = bg[:, :, idx].transpose(1, 2).reshape(n * W, 2, T)[:batch]
    mid = T // 2
    inside = (x >= -0.5) & (x < s - 0.5) & (y >= -0.5) & (y < s - 0.5)
    keep = (per_frame[:, mid:mid + W] > max(sim.target_photons_min, 0.0)) & inside[:, None]     # (molecules, W)
    i, w = keep.nonzero(as_tuple=True)
    ks = sample[i] * W + w
    order = torch.argsort(ks, stable=True)
    i, w, ks = i[order], w[order], ks[order]
    use = ks < batch
    i, w, ks = i[use], w[use], ks[use]
    counts = torch.bincount(ks, minlength=batch)
    k = max(int(counts.max()) if len(ks) else 1, 1)
    rank = torch.arange(len(ks), device=device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    xyz = torch.zeros((batch, k, 3), device=device)
    mask = torch.zeros((batch, k), dtype=torch.bool, device=device)
    xyz[ks, rank] = torch.stack([x[i], y[i], z[i]], 1)
    mask[ks, rank] = True
    window = torch.zeros((batch, k, T), device=device)
    window[ks, rank] = per_frame[i[:, None], idx[w]]
    if channels.photons == 2:
        photons = torch.zeros((batch, k, 2), device=device)
        photons[ks, rank] = torch.stack([per_channel[0][i, w + mid], per_channel[1][i, w + mid]], 1)
    else:
        photons = torch.zeros((batch, k), device=device)
        photons[ks, rank] = per_frame[i, w + mid]
    shift = torch.zeros((batch, k, 2), device=device)
    shift[ks, rank] = d[i].to(shift.dtype)
    aux_w = aux.repeat_interleave(W, 0)[:batch]
    bg_map = None if mod is None else mod.repeat_interleave(W, 0)[:batch] * bgw[:, :, mid, None, None]
    return Batch(frames, xyz, photons, mask, bgw[:, :, mid], window, bgw, movie if W > 1 else None, W, aux=aux_w,
                 background_map=bg_map, shift=shift)
