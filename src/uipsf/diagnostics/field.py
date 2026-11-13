"""How much a field-dependent model's aberrations vary over the beads' field, and whether that is real.

The report a calibration with *field dependence* ticked shows (`docs/uipsf/field-dependence.md`): per varying
mode the largest change across the beads' positions, in rad and nm of path, and one number for whether the
field is there at all -- a Wald test of the field's coefficients against zero, ½χ² in nats, comparable with
the likelihood-ratio probes' gains (`probes.py`): c·C⁻¹·c/2 is the objective's drop to second order, C the
coefficients' CRLB covariance from the fit, so it costs no refit.  Flagged beyond the χ² quantile of ``p``
(10⁻⁶, the probes' default).  Below it the field is noise: learn without it.

Too few beads do not determine the field at all: 4 beads for the 14 coefficients of a linear field of
seven modes left a covariance with negative eigenvalues (the fit's Hessian singular, November 2026, a
customer's dual-free run), whose square roots stopped the report, and the window's *State* with it.  Then
the report says the field is not determined (``determined``), with no gain and no uncertainties.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FieldReport:
    gain: float                 # ½ c·C⁻¹·c, nats
    dof: int                    # the field's coefficients
    threshold: float            # ½ χ²(p, dof)
    significant: bool
    changes: dict               # {Noll: (largest change across the beads, rad; its CRLB, rad)}
    wavelength_nm: float
    span_um: float              # the beads' span over the chip, the larger side
    n_beads: int = 0
    determined: bool = True     # False: the coefficients' covariance is not positive definite (docstring)

    def summary(self) -> str:
        nm = 1 / (2 * math.pi) * self.wavelength_nm
        if not self.determined:
            return (f"field dependence is not determined by {self.n_beads} beads ({self.dof} coefficients): "
                    "learn without it, or with more beads spread over the field")
        parts = ", ".join(f"Noll {j} {d:+.3f} rad ({d * nm:+.0f} nm)" for j, (d, _) in self.changes.items())
        verdict = ("field dependence is significant" if self.significant
                   else "field dependence is not significant: learn without it")
        return (f"{verdict} ({self.gain:.0f} nats over {self.dof} coefficients, threshold {self.threshold:.0f}); "
                f"largest changes over the beads' {self.span_um:.0f} µm: {parts}")


def field_report(cal, channel: int = 0, p: float = 1e-6) -> FieldReport | None:
    """The `FieldReport` of a learnt `Calibration` ``cal``; None when its channel has no field."""
    import torch
    from scipy.stats import chi2

    from ..model.channels import prefix
    ch = cal.model.channels[channel]
    field = getattr(ch, "field", None)
    if field is None:
        return None
    name = prefix(channel) + "field.phase"
    res = cal.result
    lo, hi = next((a, b) for n, a, b in res.params.layout("shared") if n == name)
    cov = res.covariance_shared[lo:hi, lo:hi].detach().cpu().double().numpy()
    c = res.params.physical()[name].detach().cpu().double().numpy()
    dof = len(c)
    threshold = 0.5 * float(chi2.isf(p, dof))
    xy = cal.beads_field_xy() if hasattr(cal, "beads_field_xy") else None
    if xy is None:
        raise ValueError("the calibration does not keep its beads' chip positions")
    xy = torch.as_tensor(np.asarray(xy, float))
    determined = bool(np.isfinite(cov).all())
    if determined:
        eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
        determined = bool(eig.min() > 1e-12 * max(eig.max(), 0.0))
    gain = 0.5 * float(c @ np.linalg.solve(cov, c)) if determined else math.nan
    basis = field.basis(xy).numpy()                                        # [E, T]
    changes = {}
    for j, _ in field.terms:
        rows = [i for i, (m, _, _) in enumerate(field.layout) if m == j]
        values = basis[:, rows] @ c[rows]                                  # [E] change at each bead
        k = int(np.argmax(np.abs(values)))
        g = basis[k, rows]
        err = math.sqrt(max(0.0, float(g @ cov[np.ix_(rows, rows)] @ g))) if determined else math.nan
        changes[j] = (float(values[k]), err)
    span = (xy.max(0).values - xy.min(0).values).numpy() * np.asarray(field.pixel_size_nm) / 1000.0
    return FieldReport(gain, dof, threshold, determined and gain > threshold, changes,
                       float(ch.psf.grid.optics.wavelength_nm), float(span.max()), int(len(xy)), determined)
