"""Starting values for a 4Pi bead fit, from the data alone (docs/4pi.md, D7).

The envelope's start is `start.rough_start`'s on the channels and mirror steps summed: across four
channels a quarter turn apart the interference cancels, and what is left is a plain bead stack.

The phases come from the light of each bead, channel and slice at the M mirror steps, n_m = A + B cos(θ +
s_m): with the steps as set this is linear in (A, B cos θ, B sin θ), exactly determined by three steps.
- **The channels' offsets** δ_c are the B-weighted circular mean of θ_c − θ_0 over beads and slices.
- **The steps' sign.**  The mirror steps are known as set, not which way round they enter the model's
  phase: that depends on which arm the mirror is in.  Reversing them reverses every θ, which three steps
  cannot tell apart (both fit three numbers exactly), but it reverses the fringe's direction too, and
  the model's is fixed: θ falls as z grows (`fourpi.py`).  So the steps are turned round when the data's
  θ grows with z.  A start with them the wrong way round stopped in a wrong minimum on simulated beads.
- **One phase per stack** (no mirror steps, M = 1: the stacks a 4Pi takes with its cavity left alone):
  nothing to demodulate per slice, so the offsets come from the fringe along z instead.  Each channel's
  light less the channels' mean (which takes out the incoherent part: the channels are spread round the
  circle) is projected on e^{iKz}, K the fringe's spatial frequency where that projection is strongest
  over periods of 200–340 nm, and δ_c is the phase of channel c's projection against channel 0's.  With no
  steps there is no direction to find.
- The beads' own phases φ_e are left to `workflows.fourpi`, which tries a circle of them on the
  starting model: the integrated fringe's phase at the focus depends on the pupils, which the data alone
  do not give.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .start import _one_channel


def _stack(batch, M):
    """The data as [E, C, J, M, Y, X] float64 on the CPU, and the slices' z [J] (or [E, J])."""
    d = batch.data.detach().cpu().double()
    E, C, K, Y, X = d.shape
    z = batch.z_nm.detach().cpu().double()
    z = z.reshape(K // M, M)[:, 0] if z.dim() == 1 else z.reshape(E, K // M, M)[..., 0]
    return d.reshape(E, C, K // M, M, Y, X), z


def phases_from_light(n: np.ndarray, steps: np.ndarray):
    """(θ, B) from the light ``n`` [..., M] at mirror steps ``steps`` [M]: n = A + B cos(θ + s)."""
    X = np.stack([np.ones_like(steps), np.cos(steps), -np.sin(steps)], 1)      # [M, 3]
    coef = n @ np.linalg.pinv(X).T                                             # [..., 3]
    return np.arctan2(coef[..., 2], coef[..., 1]), np.hypot(coef[..., 1], coef[..., 2])


def channel_phases(batch, n_phases: int, steps, background=None):
    """θ [E, C, J] and B [E, C, J] of every bead, channel and slice, for mirror ``steps`` [M]."""
    d, _ = _stack(batch, n_phases)
    bg = 0.0 if background is None else torch.as_tensor(background).double()[:, :, None, None, None, None]
    n = (d - bg).sum((-2, -1)).numpy()                                          # [E, C, J, M]
    return phases_from_light(n, np.asarray(steps, float))


def fringe_slope(theta, B, z) -> float:
    """dθ/dz in rad per nm: the B-weighted phase advance between neighbouring slices, per nm of z."""
    q = B * np.exp(1j * theta)                                                 # [E, C, J]
    z = np.asarray(z, float)
    dz = np.diff(z, axis=-1)
    dz = dz[:, None, :] if dz.ndim == 2 else dz                                 # [(E, 1,) J − 1]
    adv = np.conj(q[..., :-1]) * q[..., 1:]
    adv = np.where(dz > 0, adv, np.conj(adv))                                  # every advance towards +z
    return float(np.angle(adv.sum()) / np.median(np.abs(dz)))


def fringe_along_z(n: np.ndarray, z) -> tuple[np.ndarray, float]:
    """(δ_c [C − 1], dθ/dz) from each bead's and channel's light ``n`` [E, C, J] along one stack per bead at
    one phase (module docstring)."""
    z = np.asarray(z, float)
    r = n - n.mean(1, keepdims=True)
    r = r / np.abs(n.mean((1, 2), keepdims=True)).clip(min=1e-12)          # every bead counts alike
    zz = z if z.ndim == 2 else z[None]                                          # [1|E, J]
    best = None
    for period in np.linspace(200.0, 340.0, 141):
        p = (r * np.exp(1j * 2 * math.pi / period * zz)[:, None, :]).sum(-1)   # [E, C]
        power = float((np.abs(p) ** 2).sum())
        if best is None or power > best[0]:
            best = (power, period, p)
    _, period, p = best
    deltas = np.angle((p * np.conj(p[:, :1])).sum(0))[1:]
    return deltas, -2 * math.pi / period


def transforms_from_centroids(batch, background) -> list[np.ndarray] | None:
    """Each channel c ≥ 1's affine map onto channel 0's chip pixels, from the beads' centroids in the
    summed stacks; None without ROI positions or with fewer than three beads (a translation alone is
    taken from one or two)."""
    if batch.roi_xy is None:
        return None
    d = batch.data.detach().cpu().double()
    E, C, K, Y, X = d.shape
    img = (d - torch.as_tensor(background).double()[:, :, None, None, None]).sum(2).clamp(min=0)
    img = img / img.sum((-2, -1), keepdim=True).clamp(min=1e-12)
    xs = torch.arange(X, dtype=img.dtype) - (X - 1) / 2
    ys = torch.arange(Y, dtype=img.dtype) - (Y - 1) / 2
    cx, cy = (img.sum(-2) * xs).sum(-1), (img.sum(-1) * ys).sum(-1)              # [E, C]
    chip = batch.roi_xy.detach().cpu().double() + torch.stack([cx, cy], -1)       # [E, C, 2]
    out = []
    for c in range(1, C):
        src, dst = chip[:, c].numpy(), chip[:, 0].numpy()
        H = np.eye(3)
        if E >= 3:
            A = np.c_[src, np.ones(E)]
            H[:2] = np.linalg.lstsq(A, dst, rcond=None)[0].T
        else:
            H[:2, 2] = np.mean(dst - src, 0)
        out.append(H)
    return out


def rough_start(batch, model) -> dict:
    """xyz, photons and background per bead, the channels' offsets and transforms, and the mirror
    steps the right way round (module docstring)."""
    M, C = model.n_phases, model.n_channels
    d, z = _stack(batch, M)
    E = d.shape[0]
    summed = d.sum(1).mean(2)                                                  # [E, J, Y, X]
    xyz, photons, _ = _one_channel(summed, z, model.pixel_size_nm)
    background = torch.stack([_one_channel(d[:, c].flatten(1, 2), batch.z_nm.cpu().double(),
                                           model.pixel_size_nm)[2] for c in range(C)], 1)
    steps = np.r_[0.0, model.steps_init.double().numpy()]
    flipped = False
    if M == 1:
        n = (d - background.double()[:, :, None, None, None, None]).sum((-2, -1))[..., 0].numpy()
        deltas, slope = fringe_along_z(n, z.numpy())
        dt = batch.data.dtype
        return {"xyz": xyz.to(dt), "photons": photons.to(dt), "background": background.to(dt),
                "phase": torch.zeros(E, dtype=dt), "steps": torch.zeros(0, dtype=dt), "steps_flipped": False,
                "deltas": torch.as_tensor(deltas, dtype=dt), "transforms": transforms_from_centroids(batch, background),
                "fringe_slope": slope, "period_nm": 2 * math.pi / abs(slope)}
    theta, B = channel_phases(batch, M, steps, background)
    if fringe_slope(theta, B, z.numpy()) > 0:
        steps, flipped = -steps, True
        theta, B = channel_phases(batch, M, steps, background)
    slope = fringe_slope(theta, B, z.numpy())
    rel = np.exp(1j * (theta - theta[:, :1])) * B * B[:, :1]
    deltas = np.angle(rel.sum((0, 2)))[1:]
    dt = batch.data.dtype
    return {"xyz": xyz.to(dt), "photons": photons.to(dt), "background": background.to(dt),
            "phase": torch.zeros(E, dtype=dt), "steps": torch.as_tensor(steps[1:], dtype=dt),
            "steps_flipped": flipped, "deltas": torch.as_tensor(deltas, dtype=dt),
            "transforms": transforms_from_centroids(batch, background),
            "fringe_slope": slope, "period_nm": 2 * math.pi / abs(slope)}
