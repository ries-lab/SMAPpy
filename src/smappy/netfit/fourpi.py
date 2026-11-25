"""netfit on a 4Pi microscope's phase channels (docs/4pi.md §30): the channels on one grid, training
batches drawn with the IAB model, and the bound a molecule's x, y, z and phase have.

**The grid** is `channels.TwoChannels`' for C channels: channel 0's pixels, every other channel moved
onto it by a whole-pixel shift ``s_c`` (the registration's median offset over the overlap), so that a
molecule at ``p`` in channel 0 is at ``p + d_c(p)`` in channel c's shifted image, ``d_c`` the
registration's remainder -- two offset maps a channel beyond the first, read by the network beside the
frames (six for four channels).  Only where every channel holds the camera's pixels is read.  The
layout and the maps are the 4Pi model's own (`FourPiCalibration.layout`, `maps`), as Spline 4Pi takes
them on *auto*.

**A batch** (`draw_batch`) is `simulate.draw_batch_2c`'s with four channels and a phase: molecules
uniform in a crop of the overlap, each drawn in every channel with `render_4pi.IABRenderer` where the
registration puts it; its phase in each frame ``phase_per_nm z + phase0(t)``, the cavity's ``phase0``
uniform on the circle at the movie's start, then drifting (``cavity_rate_rad``, ``cavity_walk_rad``);
a background per channel and frame; the camera on each.  The target is the molecule's photons (all
channels together, as Spline 4Pi's shared N), x, y, z -- the envelope's -- and the phase in the middle
frame, wrapped: what the fitter reports as ``phase_rad``.  z from the phase needs the cavity's phase,
which only many molecules over time give (*4Pi z*), so the network does not learn it.

**The bound** (`crlb`) is the Fisher information of the IAB model in all channels at once, x, y, N, z
and the phase shared and a background per channel free, as `psf.Global4PiPSF` fits it; the
derivatives by central differences in float64 with each square held where it was drawn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .simulate import Batch, _uniform, background_modulation, camera, emitters


@dataclass
class FourChannels:
    """A 4Pi model's channels on channel 0's grid, for frames whose corner is at ``corner`` on the chip."""
    layout: object                 # `channels.ChannelLayout`, regions in frame pixels, with its maps
    shifts: tuple                  # per channel (columns, rows): its own pixels are channel 0's plus this, plus d
    box: tuple                     # (row0, row1, col0, col1) of channel 0 where every channel is read

    @classmethod
    def from_layout(cls, layout) -> FourChannels:
        (w0, h0) = layout.sizes[0]
        rows, cols = np.mgrid[0:h0, 0:w0]
        xy = np.c_[cols.ravel(), rows.ravel()].astype(float)
        shifts, r0, r1, c0, c1 = [(0, 0)], 0, h0, 0, w0
        for c in range(1, layout.n_channels):
            w, h = layout.sizes[c]
            off = layout.to_channel(c, xy) - xy
            inside = ((off[:, 0] + xy[:, 0] >= 0) & (off[:, 0] + xy[:, 0] <= w - 1)
                      & (off[:, 1] + xy[:, 1] >= 0) & (off[:, 1] + xy[:, 1] <= h - 1))
            if not inside.any():
                raise ValueError(f"channel {c} does not overlap channel 0")
            sx, sy = (int(np.rint(np.median(off[inside, k]))) for k in (0, 1))
            shifts.append((sx, sy))
            r0, r1, c0, c1 = max(r0, -sy), min(r1, h - sy), max(c0, -sx), min(c1, w - sx)
        if r1 <= r0 or c1 <= c0:
            raise ValueError("the channels have no region in common")
        return cls(layout, tuple(shifts), (r0, r1, c0, c1))

    @classmethod
    def from_calibration(cls, cal) -> FourChannels:
        """From a `fourpi.FourPiCalibration` that keeps its layout and maps."""
        from ..channels import ChannelLayout
        if not cal.layout:
            raise ValueError("the 4Pi model keeps no channel layout: learn it from bead scans in this program")
        layout = ChannelLayout.from_dict(cal.layout)
        maps = cal.maps()
        if maps is not None:
            layout = layout.with_maps(maps)
        return cls.from_layout(layout)

    @property
    def n_channels(self) -> int:
        return self.layout.n_channels

    @property
    def size(self) -> tuple:
        r0, r1, c0, c1 = self.box
        return r1 - r0, c1 - c0

    def remainder(self, xy) -> np.ndarray:
        """``d_c`` of channel-0 positions ``xy`` (n, 2), every channel beyond the first: (n, 2 (C - 1)),
        x then y of channel 1, then of channel 2, ..."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        out = [self.layout.to_channel(c, xy) - xy - np.asarray(self.shifts[c], float)
               for c in range(1, self.n_channels)]
        return np.concatenate(out, 1)

    def offsets(self, box=None) -> np.ndarray:
        """``d`` at the centre of each pixel of ``box`` (the whole box by default): (2 (C - 1), rows,
        columns), float32 -- what `channels.Grid` draws crops from."""
        r0, r1, c0, c1 = box or self.box
        rows, cols = np.mgrid[r0:r1, c0:c1]
        d = self.remainder(np.c_[cols.ravel(), rows.ravel()])
        return d.T.reshape(-1, r1 - r0, c1 - c0).astype(np.float32)

    def cut(self, frames, box=None) -> np.ndarray:
        """Frames (n, H, W) as the camera wrote them -> (n, C, rows, columns) on channel 0's grid."""
        r0, r1, c0, c1 = box or self.box
        each = self.layout.cut(frames)                                      # (n, C, h, w), the flips done
        return np.stack([each[:, c, r0 + sy:r1 + sy, c0 + sx:c1 + sx]
                         for c, (sx, sy) in enumerate(self.shifts)], 1)

    def on_frame(self, xy) -> np.ndarray:
        """Channel-0 positions back on the frame the camera wrote."""
        return self.layout.on_frame(0, xy)

    def to_dict(self) -> dict:
        return {"shifts": [list(s) for s in self.shifts], "box": list(self.box)}


def cavity(n: int, L: int, rate: float, walk: float, g, device) -> torch.Tensor:
    """(n, L): each movie's cavity phase in each frame -- uniform on the circle at its start, a drift
    uniform in +-``rate`` a frame and a random walk of ``walk`` a frame."""
    start = _uniform(-np.pi, np.pi, n, g, device)[:, None]
    slope = _uniform(-rate, rate, n, g, device)[:, None]
    t = torch.arange(L, device=device, dtype=start.dtype)[None]
    steps = walk * torch.randn((n, L), generator=g, device=device)
    steps[:, 0] = 0
    return start + slope * t + torch.cumsum(steps, 1)


def wrap(phase):
    return torch.remainder(phase + np.pi, 2 * np.pi) - np.pi


def draw_batch(renderer, grid, sim, cam, channels, batch: int, g) -> Batch:
    """One batch of fresh 4Pi samples (the module's docstring): ``renderer`` a `render_4pi.IABRenderer`,
    ``grid`` a `channels.Grid` over `FourChannels`, ``channels`` the `settings.ChannelSettings`.  Movies of
    ``window + movie_windows - 1`` frames, each its own crop, cut into ``movie_windows`` windows, as
    `simulate.draw_batch_2c`.  Frames (b, C, window, s, s), ``phase`` (b, k) the targets' phase in the
    middle frame, ``shift`` (b, k, 2 (C - 1)) each target's remainder in the other channels."""
    device = renderer.device
    C = renderer.n_channels
    T, s = sim.window, sim.frame_px
    W = max(1, int(sim.movie_windows))
    n, L = -(-batch // W), T + W - 1
    sample, x, y, z, _, per_frame = emitters(sim, n, g, device, frames=L)
    r0, c0 = grid.origins(n, s, g)
    aux = grid.maps(r0, c0, s)                                                    # (n, 2 (C - 1), s, s)
    d = grid.at(sample, x, y, r0, c0)                                             # (m, 2 (C - 1))
    X = torch.cat([x[:, None], x[:, None] + d[:, 0::2]], 1)                      # (m, C)
    Y = torch.cat([y[:, None], y[:, None] + d[:, 1::2]], 1)
    phase0 = cavity(n, L, channels.cavity_rate_rad, channels.cavity_walk_rad, g, device)
    phase = renderer.phase_of(z[:, None], phase0[sample])                         # (m, L)
    ranges = channels.background or [sim.background] * C
    if len(ranges) != C:
        raise ValueError(f"{len(ranges)} background ranges for {C} channels")
    bg = torch.stack([_uniform(*ranges[c], n * L, g, device).view(n, L) for c in range(C)], 1)   # (n, C, L)
    mod = background_modulation(sim, n, C, s, g, device)                         # (n, C, s, s) or None
    image = bg.transpose(1, 2)[..., None, None].expand(n, L, C, s, s).contiguous()
    if mod is not None:
        image = image * mod[:, None]
    m, f = (per_frame > 0).nonzero(as_tuple=True)
    renderer.draw(image.view(n * L, C, s, s), sample[m] * L + f, X[m], Y[m], z[m], phase[m, f], per_frame[m, f])
    movie = camera(image, cam, g).transpose(1, 2)                                 # (n, C, L, s, s)
    idx = torch.arange(W, device=device)[:, None] + torch.arange(T, device=device)[None]     # (W, T)
    frames = movie[:, :, idx].transpose(1, 2).reshape(n * W, C, T, s, s)[:batch]
    bgw = bg[:, :, idx].transpose(1, 2).reshape(n * W, C, T)[:batch]
    mid = T // 2
    inside = (x >= -0.5) & (x < s - 0.5) & (y >= -0.5) & (y < s - 0.5)
    keep = (per_frame[:, mid:mid + W] > max(sim.target_photons_min, 0.0)) & inside[:, None]
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
    photons = torch.zeros((batch, k), device=device)
    photons[ks, rank] = per_frame[i, w + mid]
    target_phase = torch.zeros((batch, k), device=device)
    target_phase[ks, rank] = wrap(phase[i, w + mid]).to(target_phase.dtype)
    shift = torch.zeros((batch, k, 2 * (C - 1)), device=device)
    shift[ks, rank] = d[i].to(shift.dtype)
    aux_w = aux.repeat_interleave(W, 0)[:batch]
    bg_map = None if mod is None else mod.repeat_interleave(W, 0)[:batch] * bgw[:, :, mid, None, None]
    return Batch(frames, xyz, photons, mask, bgw[:, :, mid], window, bgw, movie if W > 1 else None, W, aux=aux_w,
                 background_map=bg_map, shift=shift, phase=target_phase)


# the bound's parameters: x, y (pixels), N, z (nm), phase (rad), then one background per channel
_STEPS = (1e-3, 1e-3, None, 0.05, 1e-3)


def crlb(cal, x, y, shift, z_nm, phase, photons, background, em: bool = False) -> np.ndarray:
    """(n, 4): the bound of x, y (pixels), z (nm) and the phase (rad) of molecules at channel-0
    positions ``x``, ``y`` with each other channel's remainder ``shift`` (n, 2 (C - 1)), ``z_nm``,
    ``phase``, ``photons`` (all channels together) and ``background`` (n, C) -- every channel's spot in
    its own square, x, y, N, z and the phase shared, a background per channel free (the module's
    docstring).  ``em``: the EM register's excess noise halves the information."""
    from .render_4pi import IABRenderer
    r = IABRenderer(cal, dtype=torch.float64)
    t = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64)
    x, y, z, ph, N = t(x), t(y), t(z_nm), t(phase), t(photons)
    sh = t(shift).reshape(len(x), -1)
    X = torch.cat([x[:, None], x[:, None] + sh[:, 0::2]], 1)
    Y = torch.cat([y[:, None], y[:, None] + sh[:, 1::2]], 1)
    bg = t(background).reshape(len(x), -1)
    C = r.n_channels
    base, ox, oy = r.spots(X, Y, z, ph, clamp=False)
    corner = (ox, oy)

    def mu_of(dx=0.0, dy=0.0, dz=0.0, dp=0.0):
        sp, _, _ = r.spots(X + dx, Y + dy, z + dz, ph + dp, corner=corner, clamp=False)
        return sp

    derivs = []
    for k, step in enumerate(_STEPS):
        if k == 2:                                  # photons: the unit spot itself
            derivs.append(base)
            continue
        kw = [{"dx": step}, {"dy": step}, None, {"dz": step}, {"dp": step}][k]
        minus = {key: -v for key, v in kw.items()}
        derivs.append(N[:, None, None, None] * (mu_of(**kw) - mu_of(**minus)) / (2 * step))
    for c in range(C):                              # each channel's background
        one = torch.zeros_like(base)
        one[:, c] = 1.0
        derivs.append(one)
    J = torch.stack(derivs, -1).reshape(len(x), -1, len(derivs))      # (n, pixels, params)
    mu = (N[:, None, None, None] * base.clamp_min(0) + bg[:, :, None, None]).reshape(len(x), -1).clamp_min(1e-9)
    info = torch.einsum("npi,npj,np->nij", J, J, 1 / mu) / (2.0 if em else 1.0)
    cov = torch.linalg.pinv(info)
    var = torch.diagonal(cov, dim1=1, dim2=2)[:, [0, 1, 3, 4]]
    return torch.sqrt(var.clamp_min(0)).numpy()


def unwrap_known(phase, z_env_nm, phase0, phase_per_nm: float) -> np.ndarray:
    """z from the phase, on the fringe nearest the envelope's z, with the cavity's phase ``phase0`` known
    (a simulation's): ``(phase - phase0 + 2 pi n) / phase_per_nm`` for the n that puts it nearest
    ``z_env_nm``.  On real data *4Pi z* estimates phase0 from the localizations (`smappy.fourpi.unwrap`)."""
    base = np.angle(np.exp(1j * (np.asarray(phase) - phase0))) / phase_per_nm
    period = 2 * np.pi / abs(phase_per_nm)
    return base + np.round((np.asarray(z_env_nm) - base) / period) * period


def phase_scores(truth: dict, found: dict, pixel_nm, phase_per_nm: float | None = None) -> dict:
    """The phase of the molecules found against the truth's: matched as `metrics.evaluate` matches them,
    the circular RMSE (rad) and, with the truth's ``crlb_phase``, the robust width of the error over its
    bound, over all and near the focus (|z| < `metrics.CORE_Z_NM`).  With ``phase_per_nm``: z unwrapped
    from the phase with the cavity's phase known (`unwrap_known`) -- the share on a wrong fringe, and the
    RMSE of z where the fringe is right, the 4Pi z the network's phase and envelope give together."""
    from .metrics import CORE_Z_NM, PHOTONS_MIN, _below_limit, match, robust_width
    if "phase" not in found or "phase" not in truth:
        return {}
    ti, fi = match(truth["frame"], truth["xyz"], found["frame"], found["xyz"], pixel_nm)
    truth2, ti, fi, _ = _below_limit(truth, ti, fi, PHOTONS_MIN)
    if len(ti) < 10:
        return {}
    err = np.angle(np.exp(1j * (found["phase"][fi] - truth2["phase"][ti])))
    z = truth2["xyz"][ti, 2]
    core = np.abs(z) < CORE_Z_NM
    out = {"rmse_phase_rad": float(np.sqrt(np.mean(err ** 2))),
           "phase_flip_share": float(np.mean(np.abs(err) > np.pi / 2))}
    if core.sum() >= 50:
        out["core_rmse_phase_rad"] = float(np.sqrt(np.mean(err[core] ** 2)))
    if truth2.get("crlb_phase") is not None:
        b = truth2["crlb_phase"][ti]
        out["crlb_width_phase"] = robust_width(err / b)
        if core.sum() >= 50:
            out["core_crlb_width_phase"] = robust_width(err[core] / b[core])
    if found.get("phase_err") is not None:
        out["claim_width_phase"] = robust_width(err / found["phase_err"][fi])
    if phase_per_nm:
        phase0 = truth2["phase"][ti] - phase_per_nm * z
        zp = unwrap_known(found["phase"][fi], found["xyz"][fi, 2], phase0, phase_per_nm)
        period = 2 * np.pi / abs(phase_per_nm)
        wrong = np.abs(zp - z) > period / 2
        out["wrong_fringe_share"] = float(np.mean(wrong))
        if (~wrong).sum() >= 10:
            out["rmse_z_phase_nm"] = float(np.sqrt(np.mean((zp - z)[~wrong] ** 2)))
        if core.sum() >= 50:
            out["core_wrong_fringe_share"] = float(np.mean(wrong[core]))
            if (core & ~wrong).sum() >= 10:
                out["core_rmse_z_phase_nm"] = float(np.sqrt(np.mean((zp - z)[core & ~wrong] ** 2)))
    return out
