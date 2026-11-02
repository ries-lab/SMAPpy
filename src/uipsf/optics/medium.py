"""The sample's maps as functions of its refractive index: what a learnt n_med (``optics.n_med``) needs.

`PupilGrid.build` computes everything that depends on the sample's index once, in float64 on the CPU, for
the typed n_med: k_z,med, cos θ_med, the Fresnel factors and the apodisation, and from them the dipole
fields (`optics.vectorial`).  That stays the path of every model that holds n_med -- the default -- and costs
nothing.  A model that learns n_med (a diagnostic: a buffer's index is measured only when it is matched,
docs/4pi.md §23) recomputes them here from a tensor, on the device and in its dtype, so that they are
differentiable in n_med:

- **the smoothed root.**  cos θ_med = √(1 − (ρ·NA/n_med)²) has a kink at the critical angle, where a sample
  of the pupil crosses from propagating to evanescent light as n_med moves; its derivative there is infinite.
  `layers._root`'s smoothing, √(a + iε·exp(−a²/ε²)) with ε a 64th of how far a moves between neighbouring
  samples at the aperture's edge, keeps it finite: the same choice for the same kink (a tilt moves the
  critical ring as n_med does).  More than 3ε from zero it is the exact root.
- **the apodisation inside the Fresnel factor.**  The grid's t_p, t_s carry 2·n_med·cos θ_med and the
  apodisation √cos θ_imm / cos θ_med divides it out again; at the critical angle that is 0/0.  Here the
  products are formed with the cos θ_med cancelled, finite everywhere -- the same numbers elsewhere
  (`tests/uipsf_tests/test_medium.py`).

The fields carry the aperture, as `VectorPSF.fields` does.  The immersion's and the coverslip's maps do not
depend on n_med and are the grid's.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from .layers import _root
from .vectorial import _vector_fields


def eps(grid) -> float:
    """The smoothing of the critical angle's root: a 64th of how far 1 − (ρ·NA/n)² moves between two samples
    at the aperture's edge, for n near the NA (`layers._tilted`'s, in its units)."""
    return 4.0 / grid.size / 64


def maps(grid, n_med: torch.Tensor) -> SimpleNamespace:
    """(n, cos_med, sin_med, kz_med, kappa), [P, P]: the sample's maps at the index ``n_med`` (a 0-d tensor).
    ``kz_med`` complex, cycles/nm; ``kappa`` its imaginary part, how fast the supercritical light decays in
    the sample (`MinimalPupil`'s ring)."""
    o = grid.optics
    n = n_med.to(grid.dtype)
    rho = grid.rho.real if grid.rho.is_complex() else grid.rho
    s = rho * o.NA                                          # n·sin θ, the same in every layer
    cos_med = _root(1 - (s / n) ** 2, eps(grid))            # complex, +i branch beyond the critical angle
    kz_med = (n / o.wavelength_nm) * cos_med
    return SimpleNamespace(n=n, cos_med=cos_med, sin_med=(s / n).to(cos_med.dtype), kz_med=kz_med,
                           kappa=kz_med.imag)


def fields(grid, n_med: torch.Tensor, kind: str = "vector", analyser: str | None = None) -> torch.Tensor:
    """[F, P, P] complex: the aperture × apodisation × dipole fields of `VectorPSF.fields` at ``n_med``
    (``kind`` "vector" or "scalar", ``analyser`` None, "x" or "y" as `optics.vectorial.dipole_fields`)."""
    m = maps(grid, n_med)
    o = grid.optics
    n = m.n.to(m.cos_med.dtype)
    n_cov, n_imm = o.n_cov, o.n_imm
    cos_cov, cos_imm = grid.cos_cov, grid.cos_imm
    # the second interface's transmission (coverslip -> immersion) and √cos θ_imm do not depend on n_med
    root_imm = torch.sqrt(cos_imm)
    t2p = 2 * n_cov * cos_cov / (n_cov * cos_imm + n_imm * cos_cov) * root_imm
    t2s = 2 * n_cov * cos_cov / (n_cov * cos_cov + n_imm * cos_imm) * root_imm
    # t_p·apodisation and t_s·apodisation with cos θ_med cancelled (module docstring)
    tp = 2 * n / (n * cos_cov + n_cov * m.cos_med) * t2p
    ts = 2 * n / (n * m.cos_med + n_cov * cos_cov) * t2s
    if kind == "scalar":                    # the mean of the two (`vectorial.scalar_field`), apodised
        out = ((tp + ts) / 2)[None]
    else:
        out = _vector_fields(grid, tp, ts, m.cos_med, m.sin_med, analyser)
    return grid.aperture * out
