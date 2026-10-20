"""How well each bead fits the model: the numbers bead rejection decides on (plan §5.7).

The scores come from the learning fit itself, with no fit of their own:
every bead in a fit already has its stack fit on that fit's pupil (decided
October 2026; a second localisation on the same pupil cost 180 s on
Zenodo's 84 beads and reproduced what the fit held).

- **deviance**: the mean per pixel of the bead's stack, ≈ 1 when the bead
  looks like the model, higher for a double bead, a dirty one, a jump (a
  300 nm stage jump halfway through a stack: 1.42 against 0.99–1.08 for
  the good beads, tests/uipsf_tests/outliers.py);
- **photons**: |log(N / median N)|, the brightness against the others;
- **xy offset**: the distance of the bead from its ROI's centre, pixels;
- **masked**: the fraction of its pixels excluded (saturated).

With several channels a bead is a pair (or more), and it is scored as one:
the deviance over all its channels' pixels, its total photons, and the
larger of its offsets from the centres of its ROIs, each channel's where
the transform puts it.  A pair whose secondary ROI was cut around the
wrong spot shows there.

v1's z bias (``msez``: each slice localised alone against the stack fit)
is no longer a score: on Zenodo it rejected 2 of 27 beads on its own, the
others for their photons or deviance too.  The single-slice fits it needs
(`slice_fits`, or SMAPpy's spline fitter, `spline.SplineFrames`) remain for
the held-out report, a diagnostic.  Per-slice consistency (a stage jump, a
z bias) is SMAPpy's to show: it fits every bead with the exported model's
spline.  Whether a fit-based rejection does better is for the model tests
(plan §10).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import torch

from ..device import on_model_device
from ..model import Bleaching, ForwardModel
from ..params import Params
from ..solve import BlockLM, Problem


@dataclass
class BeadScores:
    deviance: np.ndarray          # [E] mean deviance per pixel of the stack fit
    photons: np.ndarray           # [E] |log(N / median N)|
    xy_offset_px: np.ndarray      # [E]
    masked: np.ndarray            # [E] fraction of excluded pixels
    fit: Params | None            # the fit the scores come from

    NAMES = ("deviance", "photons", "xy_offset_px", "masked")

    def row(self, e: int) -> dict:
        return {n: float(getattr(self, n)[e]) for n in self.NAMES}


def shared_values(params: Params) -> dict:
    return {n: v for n, v in params.physical().items() if params.specs[n].scope == "shared"}


def localise(model, shared: dict, batch, noise, start: dict | None = None, solver=None):
    """Every bead's stack fit on a fixed pupil: (`FitResult`, `Problem`).

    `BlockLM` by default: with the pupil fixed every bead is its own problem, and one shared damping let
    one hard bead hold them all back (Zenodo, 84 beads: 100 iterations without converging, 829 s)."""
    batch = on_model_device(batch, model)
    values = dict(shared)
    if start:
        values.update(start)
    specs = model.specs(batch)
    p = Params(specs, batch.n_emitters, dtype=batch.data.dtype, device=batch.data.device, values=values)
    p = p.fix(*[s.name for s in specs if s.scope == "shared"])
    problem = Problem(model, noise, batch)
    return (solver or BlockLM()).solve(problem, p), problem


def _single_frames(model):
    """The same PSFs for single frames: no bead kernel (it needs a stack), photons without a curve."""
    channels = []
    for ch in model.channels:
        ch = copy.copy(ch)
        ch.psf = copy.copy(ch.psf)
        ch.psf.bead_radius_nm = 0.0
        channels.append(ch)
    return ForwardModel(channels, intensity=Bleaching(order=0))


def slice_fits(model, shared: dict, batch, noise, stack_fit: Params):
    """Each slice localised alone: (xyz [E, K, 3], its CRLB [E, K, 3], where the stack fit puts it), nm."""
    batch = on_model_device(batch, model)
    E, C, K, Y, X = batch.data.shape
    xyz = model.geometry(stack_fit.physical(), batch).detach()            # [E, K, 3]
    frames = replace(batch, data=batch.data.transpose(1, 2).reshape(E * K, C, 1, Y, X),
                     z_nm=torch.zeros(1, dtype=batch.data.dtype, device=batch.data.device),
                     mask=None if batch.mask is None else
                     batch.mask.expand_as(batch.data).transpose(1, 2).reshape(E * K, C, 1, Y, X),
                     s=batch.s if not isinstance(batch.s, torch.Tensor) or batch.s.dim() == 0 else
                     batch.s.expand_as(batch.data).transpose(1, 2).reshape(E * K, C, 1, Y, X),
                     field_xy=None, stack_id=None,
                     roi_xy=None if batch.roi_xy is None else batch.roi_xy.repeat_interleave(K, 0))
    phys = stack_fit.physical()
    N = model.intensity(phys, batch).detach().reshape(-1)
    bg = phys["emitter.background"].detach().repeat_interleave(K, 0)
    single = _single_frames(model)
    # The pupil and blur carry over; per-slice stage terms do not (one frame has one slice).
    shapes = {s.name: s.shape for s in single.specs(frames) if s.scope == "shared"}
    shared = {k: v for k, v in shared.items() if k in shapes and tuple(v.shape) == tuple(shapes[k])}
    fit, _ = localise(single, shared, frames, noise,
                      start={"emitter.xyz": xyz.reshape(-1, 3), "emitter.photons": N.clamp(min=1.0),
                             "emitter.background": bg.clamp(min=1e-3)})
    found = fit.params.physical()["emitter.xyz"].detach().reshape(E, K, 3).cpu().double().numpy()
    crlb = fit.std["emitter.xyz"].reshape(E, K, 3).cpu().double().numpy()
    return found, crlb, xyz.cpu().double().numpy()


def score_beads(model, params: Params, batch, noise) -> BeadScores:
    """Every bead of ``batch`` scored from its fit ``params`` (module docstring)."""
    batch = on_model_device(batch, model)
    D = Problem(model, noise, batch).deviance(params)
    # D is on the CPU (`Problem.deviance`); so is everything below: copy, then cast (MPS has no float64).
    mask = torch.ones_like(D, dtype=torch.bool) if batch.mask is None else batch.mask.cpu().expand_as(D)
    deviance = ((D * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp(min=1)).double().numpy()
    masked = 1 - mask.flatten(1).double().mean(1).numpy()
    phys = params.physical()
    # a spot with no photons: huge, not inf
    N = np.maximum(phys["emitter.photons"].detach().cpu().double().numpy(), 1e-3)
    photons = np.abs(np.log(N / np.median(N)))
    with torch.no_grad():
        xyz = phys["emitter.xyz"][:, None]                       # the bead's own position, without drift
        offsets = []
        for c, ch in enumerate(model.channels):
            xy = model.positions(phys, batch, xyz, c)[:, 0, :2].detach().cpu().double().numpy()
            px, py = ch.pixel_size_nm
            offsets.append(np.hypot(xy[:, 0] / px, xy[:, 1] / py))
    return BeadScores(deviance, photons, np.max(offsets, 0), masked, params)


def unscored(n: int) -> BeadScores:
    """NaN scores for ``n`` beads that were in no fit (held out, excluded): never judged."""
    nan = np.full(n, np.nan)
    return BeadScores(nan, nan.copy(), nan.copy(), nan.copy(), None)


def merged(E: int, parts) -> BeadScores:
    """One `BeadScores` of ``E`` beads from (bead indices, scores) parts; the rest unscored."""
    out = unscored(E)
    for idx, sc in parts:
        idx = np.asarray(idx, int)
        for n in BeadScores.NAMES:
            getattr(out, n)[idx] = getattr(sc, n)
    return out
