"""Does the camera's conversion fit the data?  A photon-transfer check on a fit's residuals.

With the right camera numbers the residuals d − μ of a matched fit have
variance F·μ + s per pixel (photons, F the excess factor, s the read-noise
variance in photons²): slope 1 and intercept s once divided by F.  The
check bins the pixels by μ, takes the variance of d − μ per bin, and fits
var = a + b·μ over the low-signal bins only, where model mismatch, which
grows as μ², adds least.  Then:
- b ≈ F: the conversion is right.  b > F: the data vary more than their
  photons allow, so the conversion is too high by about b/F (or mismatch
  remains in the bins used); b < F: too low.
- a is the read-noise variance the data show, in photons²; compare it with
  the s the fit assumed.

Why it exists: the Zenodo `1ch_40nm_bead` stacks were fitted by v1 with
0.22 e⁻/ADU where SMAPpy's database says 4.7, and SMAPpy's default read
noise of 1 e⁻ (an sCMOS value) is ~21 e⁻ on that EMCCD's normal port; both
were invisible in the fit's PSF and obvious here (docs/uipsf/architecture.md, M3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class NoiseTransfer:
    slope: float                 # b: 1 (× excess) when the conversion is right
    intercept: float             # a, photons²: the read-noise variance the data show
    mu: np.ndarray               # bin centres, photons
    variance: np.ndarray         # residual variance per bin, photons²
    used: np.ndarray             # bins in the fit

    @property
    def read_noise_photons(self) -> float:
        return float(np.sqrt(max(self.intercept, 0.0)))

    def conversion_factor(self, excess: float = 1.0) -> float:
        """By how much to divide the conversion so that the slope becomes ``excess``."""
        return self.slope / excess


def noise_transfer(data, mu, mask=None, bins: int = 30, low: float = 0.6) -> NoiseTransfer:
    """``data`` and ``mu`` in photons, any shape; the fit uses the lowest ``low`` fraction of the bins."""
    d = torch.as_tensor(data).double().flatten()
    m = torch.as_tensor(mu).double().flatten()
    if mask is not None:
        keep = torch.as_tensor(mask).flatten()
        d, m = d[keep], m[keep]
    d, m = d.numpy(), m.numpy()
    edges = np.unique(np.quantile(m, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, len(edges) - 2)
    centres, var = [], []
    for i in range(len(edges) - 1):
        sel = idx == i
        if sel.sum() > 50:
            centres.append(m[sel].mean())
            var.append((d[sel] - m[sel]).var())
    centres, var = np.array(centres), np.array(var)
    used = np.arange(len(centres)) < max(3, int(low * len(centres)))
    b, a = np.polyfit(centres[used], var[used], 1)
    return NoiseTransfer(float(b), float(a), centres, var, used)
