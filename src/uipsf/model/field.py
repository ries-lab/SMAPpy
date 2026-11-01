"""The coefficient field: pupil coefficients that depend on where the emitter is on the chip.

The pupil's own coefficients (``pupil.phase``) are the aberrations at the field's centre; a field adds to
some of them a function of each emitter's chip position (``Batch.field_xy``), so every emitter gets its own
pupil, [E, n] coefficients (`VectorPSF` takes one pupil per emitter).  Study and choices:
`docs/uipsf/field-dependence.md`.

**Which modes, which orders.**  Nodal aberration theory (Thompson; Droste et al. 2024, which fits it to SMLM
data) gives the field dependence of a nominally rotationally symmetric objective with small misalignments:
astigmatism quadratic in the field position H (a·H² + b·H + c), coma linear (a·H + c), the rest constant;
tilt and defocus over the field are a distortion and a focal surface, coordinate maps, not PSF shapes.  So
`FieldPolynomial` lets astigmatism (Noll 5, 6) vary up to second order and coma (Noll 7, 8) to first; the
constants are the pupil's own.

**Why a polynomial per mode, not NAT's coupled form.**  NAT ties the two astigmatism modes together (a
real a on H² is astigmatism oriented along the field direction), which holds only in the frame where the
pupil's and the field's axes agree in orientation and handedness; a mirror in the detection path conjugates
one against the other.  A polynomial of each mode in (u, v) holds NAT's terms in any frame for 14
parameters instead of 6 -- few for beads at their photon counts (M5's fields show linear astigmatism only,
§7 of the study).  NAT's constraints are for when in situ data need fewer parameters.

**Units.**  u, v are the sample-plane distance from ``centre_px`` in units of ``scale_um`` (default 50 µm),
so a coefficient is the mode's change in radians at that distance; ``centre_px`` is a configuration, not a
parameter: it trades exactly with the constants and the linear terms.
"""
from __future__ import annotations

import torch

from ..params import ParamSpec

#: (Noll mode, highest total order in u, v): astigmatism quadratic, coma linear (module docstring)
DEFAULT_TERMS = ((5, 2), (6, 2), (7, 1), (8, 1))


def monomials(order: int) -> list[tuple[int, int]]:
    """The powers (i, j) of u^i v^j with 1 ≤ i + j ≤ ``order``, by total order: u, v, u², uv, v², …"""
    return [(n - j, j) for n in range(1, order + 1) for j in range(n + 1)]


class FieldPolynomial:
    def __init__(self, centre_px, pixel_size_nm, scale_um: float = 50.0, terms=DEFAULT_TERMS):
        """``centre_px`` (x, y) the field's centre in chip pixels; ``pixel_size_nm`` (x, y) on the sample;
        ``terms`` the modes that vary and the order of each."""
        self.centre_px = tuple(float(c) for c in centre_px)
        self.pixel_size_nm = tuple(float(s) for s in pixel_size_nm)
        self.scale_um = float(scale_um)
        self.terms = tuple((int(j), int(order)) for j, order in terms)
        # one row per coefficient: (Noll mode, i, j)
        self.layout = [(j, i, k) for j, order in self.terms for i, k in monomials(order)]

    def specs(self) -> list[ParamSpec]:
        return [ParamSpec("field.phase", (len(self.layout),))]

    def coordinates(self, field_xy: torch.Tensor) -> torch.Tensor:
        """(u, v) [E, 2] of chip positions ``field_xy`` [E, 2] (pixels)."""
        c = field_xy.new_tensor(self.centre_px)
        s = field_xy.new_tensor(self.pixel_size_nm) / (1000.0 * self.scale_um)
        return (field_xy - c) * s

    def basis(self, field_xy: torch.Tensor) -> torch.Tensor:
        """[E, T]: each coefficient's monomial at each emitter."""
        uv = self.coordinates(field_xy)
        u, v = uv[:, 0:1], uv[:, 1:2]
        powers = torch.tensor([[i, k] for _, i, k in self.layout], dtype=uv.dtype, device=uv.device)
        return u ** powers[:, 0] * v ** powers[:, 1]

    def offsets(self, p: dict, field_xy: torch.Tensor) -> dict[int, torch.Tensor]:
        """{Noll mode: [E] change of its coefficient} at each emitter."""
        terms = self.basis(field_xy) * p["field.phase"]                      # [E, T]
        modes = torch.tensor([j for j, _, _ in self.layout], device=terms.device)
        return {j: terms[:, modes == j].sum(1) for j, _ in self.terms}

    def apply(self, p: dict, pupil, field_xy) -> dict:
        """``p`` with ``pupil.phase`` one row per emitter [E, n]: the centre's plus this field's change at
        each emitter's position.  Every varying mode must be learnt by ``pupil`` (`ZernikePupil.phase_modes`):
        a held one would have no centre value to vary about."""
        if field_xy is None:
            raise ValueError("a field-dependent model needs each emitter's chip position (Batch.field_xy)")
        missing = [j for j, _ in self.terms if j not in pupil.phase_modes]
        if missing:
            raise ValueError(f"the field varies Noll {missing}, which the pupil does not learn")
        base = p["pupil.phase"]
        field_xy = field_xy.to(device=base.device, dtype=base.dtype)
        delta = base.new_zeros(field_xy.shape[0], base.shape[-1])
        for j, d in self.offsets(p, field_xy).items():
            delta = delta.index_add(1, torch.tensor([pupil.phase_modes.index(j)], device=base.device), d[:, None])
        out = dict(p)
        out["pupil.phase"] = base + delta
        return out

    def config(self) -> dict:
        """What rebuilds it (`from_config`), for the model file."""
        return dict(kind="polynomial", centre_px=list(self.centre_px), pixel_size_nm=list(self.pixel_size_nm),
                    scale_um=self.scale_um, terms=[list(t) for t in self.terms])

    @classmethod
    def from_config(cls, cfg: dict) -> FieldPolynomial:
        return cls(cfg["centre_px"], cfg["pixel_size_nm"], cfg.get("scale_um", 50.0),
                   [tuple(t) for t in cfg.get("terms", DEFAULT_TERMS)])
