"""Held-out beads: how a learned pupil localises beads it has not seen (plan §8).

The beads kept out of learning are localised on the learned pupil, every
shared parameter fixed: first each whole stack (its position and photons),
then every slice alone.  Where the stack fit puts the bead in slice k is
where the slice fit should find it, if the model is right and the stage
moved as recorded; the difference per slice, over the held-out beads, is
the calibration's **z bias** curve (and the lateral one), the measure a
3D fitter on this model will inherit.  Its spread over beads against the
slices' CRLBs says whether the CRLB is honest.

The stack fit itself is part of the check: a pupil that only fits its own
beads shows here as a held-out deviance above the training one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..device import on_model_device
from ..workflows.scoring import localise, slice_fits


@dataclass
class HeldOutReport:
    beads: list[int]
    deviance: float                 # mean per pixel of the held-out stack fits
    z_nm: np.ndarray                # [K] where the slices are, mean over beads of the stack fit's z
    z_bias_nm: np.ndarray           # [K] median over beads of (slice fit − stack fit)
    z_spread_nm: np.ndarray         # [K] sd over beads (held-out beads pass the rejection rules first)
    z_crlb_nm: np.ndarray           # [K] median CRLB of the slice fits
    xy_bias_nm: np.ndarray          # [K, 2]

    def central(self, fraction: float = 0.6) -> np.ndarray:
        """Slices within ``fraction`` of the largest |z|: where z is well determined."""
        return np.abs(self.z_nm) <= fraction * np.abs(self.z_nm).max()

    def summary(self) -> dict:
        c = self.central()
        return dict(beads=len(self.beads), deviance=self.deviance,
                    z_bias_rms_nm=float(np.sqrt(np.mean(self.z_bias_nm[c] ** 2))),
                    z_spread_over_crlb=float(np.median(self.z_spread_nm[c] / self.z_crlb_nm[c])),
                    xy_bias_rms_nm=float(np.sqrt(np.mean(self.xy_bias_nm[c] ** 2))))


def heldout_report(model, shared: dict, batch, noise, beads=None, localiser=None) -> HeldOutReport:
    """Localise ``batch``'s beads (the held-out ones) on the learned ``shared`` parameters."""
    batch = on_model_device(batch, model)
    fit, problem = localise(model, shared, batch, noise)
    D = problem.deviance(fit.params)
    mask = torch.ones_like(D, dtype=torch.bool) if batch.mask is None else batch.mask.expand_as(D)
    deviance = float((D * mask).sum() / mask.sum())
    found, crlb, stack = (localiser or slice_fits)(model, shared, batch, noise, fit.params)
    diff = found - stack                                            # [E, K, 3]
    # A plain sd: a robust one (1.4826·MAD) halved the spread of 4 matched beads.  Bad beads are kept
    # out by judging the held-out beads with the rejection rules (workflows/rejection.py) instead.
    spread = diff[..., 2].std(0, ddof=1) if len(diff) > 1 else np.full(diff.shape[1], np.nan)
    return HeldOutReport(list(range(batch.n_emitters)) if beads is None else list(beads), deviance,
                         stack[..., 2].mean(0), np.median(diff[..., 2], 0), spread,
                         np.median(crlb[..., 2], 0), np.median(diff[..., :2], 0))
