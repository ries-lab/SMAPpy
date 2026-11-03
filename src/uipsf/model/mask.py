"""Phase masks in the pupil: a known design, placed on the pupil by a few parameters (docs/uipsf/scanned-psfs.md §6).

A vortex, a top-hat, a double helix cannot be written as a few Zernikes: they have phase jumps, or need very high
orders.  So a mask is a pupil component of its own, multiplied into the pupil field (`VectorPSF.pupil_field`), and
the Zernikes learnt beside it are the system's aberrations.  The same component will serve the excitation beams of
MINFLUX and STED (the donuts), which is why the masks are the ones those need.

**The design and its placement.**  Each mask is an analytic phase Φ(u′, v′) in the mask's own coordinates, in
units of the objective's pupil radius (its NA), and is placed on the pupil by

    (u′, v′) = R(−θ) · ((u, v) − c) / s,     transmission  exp(i α Φ(u′, v′)),

the centre c, the rotation θ, the scale s and a phase factor α (a mask made for another wavelength does not delay
by exactly π).  Each placement parameter, and a mask's own (the top-hat's radius, the half-moon's step), is
**held at its design value by default and freed by name** (``free``): freed, it is a diagnostic of misalignment
rather than a model choice -- "the mask is 0.04 of the pupil radius off towards +x".  Only the placements a mask
can show are offered: a vortex looks the same at any scale and its rotation is a piston, so it has a centre and α
only; asking to free another is refused, rather than leaving a parameter the data cannot determine.

**Edges with derivatives.**  A step's position must have a derivative, or a freed centre never moves.  A
sampled step has none (moving it changes nothing until it crosses a sample), so every step is a logistic over
about a quarter of a sub-sample, and each pupil sample takes the mean of exp(iαΦ) over s × s sub-samples
(``supersample``, 4): the mean is what a pupil sample half covered by the step transmits, and the logistic gives
it a derivative.  The vortex's singularity needs nothing (the mean over sub-samples around it is ~0, as the field
there is).  Where the mean is below one, the transmission's magnitude is below one: the light a step scatters
out of the aperture's sampled field, which the PSF's energy normalisation then puts back.

**Phase wraps.**  A vortex's exp(iℓφ) is continuous for integer ℓ; with α ≠ 1 its wrap line (the −u′ axis)
becomes a real step, as an SLM's wrap is.  It is not smoothed: a vortex plate has no wrap, and an SLM's sits where
its pattern puts it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..optics.grid import PupilGrid
from ..optics.zernike import noll_to_nm
from ..params import Bounded, Identity, ParamSpec

#: the masks there are, by the name a spec gives
KINDS = ("vortex", "top-hat", "half-moon", "double helix", "zernike")
#: the placement parameters, in the order they are declared
PLACEMENTS = ("centre", "rotation", "scale", "alpha")


@dataclass
class MaskSpec:
    """A mask's design and which of its parameters are learnt: plain data, what a settings form or a model file
    holds (`make_mask`).  Lengths in units of the objective's pupil radius, angles in radians, phases in
    radians.  ``free`` names the parameters to learn: any of `PLACEMENTS` the mask offers and its own
    ("radius" for the top-hat, "delta" for the half-moon)."""
    kind: str = "vortex"
    charge: int = 1                       # vortex: the topological charge ℓ
    radius: float = 1 / math.sqrt(2) + 0.03   # top-hat: the π step's radius (SimuFLUX's, `library/pishift.py`)
    delta: float = 0.0                    # half-moon: the phase of the half without the π, moving the zero
    zones: int = 5                        # double helix: the number of annular zones
    zernike: dict = field(default_factory=dict)   # zernike: the design, {Noll index: rad}
    centre: tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    scale: float = 1.0
    alpha: float = 1.0
    free: tuple[str, ...] = ()
    supersample: int = 4


class PhaseMask:
    """A mask on a pupil grid: ``specs`` its parameters, ``transmission(p)`` its complex [P, P] map."""

    kind: str = ""
    placements: tuple[str, ...] = PLACEMENTS
    own: tuple[str, ...] = ()

    def __init__(self, grid: PupilGrid, spec: MaskSpec, aperture_na: float | None = None):
        """``aperture_na``: the objective's NA, the mask's unit of length (the grid's NA by default; a minimal
        pupil's grid reaches beyond it, `model.pupil.GRID_MARGIN`)."""
        offered = set(self.placements) | set(self.own)
        unknown = [n for n in spec.free if n not in offered]
        if unknown:
            raise ValueError(f"a {self.kind} mask cannot learn {', '.join(unknown)}: it offers "
                             f"{', '.join(sorted(offered))}")
        self.grid, self.spec = grid, spec
        self.aperture_na = float(grid.optics.NA if aperture_na is None else aperture_na)
        s = int(spec.supersample)
        P = grid.size
        d = 2.0 / P                                       # the grid's sample spacing in u, v (`PupilGrid`)
        offsets = ((torch.arange(s, dtype=grid.dtype) + 0.5) / s - 0.5) * d
        ou, ov = torch.meshgrid(offsets, offsets, indexing="xy")
        to_mask = grid.optics.NA / self.aperture_na       # grid units to the mask's (the objective's pupil radius)
        u = (grid.u[..., None] + ou.reshape(-1).to(grid.device)) * to_mask       # [P, P, s²]
        v = (grid.v[..., None] + ov.reshape(-1).to(grid.device)) * to_mask
        self.u, self.v = u, v
        #: a step's logistic scale: a quarter of a sub-sample, so it rises over about one (module docstring)
        self.edge = d / s * to_mask / 4

    def specs(self) -> list[ParamSpec]:
        s = self.spec
        out = []
        for name in self.placements:
            if name == "centre":
                out.append(ParamSpec("mask.centre", (2,), init=torch.tensor(s.centre, dtype=torch.float64),
                                     transform=Bounded(-0.5, 0.5), fixed="centre" not in s.free))
            elif name == "rotation":
                out.append(ParamSpec("mask.rotation", (), init=float(s.rotation), transform=Identity(),
                                     fixed="rotation" not in s.free))
            elif name == "scale":
                out.append(ParamSpec("mask.scale", (), init=float(s.scale), transform=Bounded(0.5, 2.0),
                                     fixed="scale" not in s.free))
            elif name == "alpha":
                out.append(ParamSpec("mask.alpha", (), init=float(s.alpha), transform=Bounded(0.0, 2.0),
                                     fixed="alpha" not in s.free))
        return out + self._own_specs()

    def _own_specs(self) -> list[ParamSpec]:
        return []

    def _value(self, p, name: str, default: float) -> torch.Tensor:
        key = f"mask.{name}"
        return p[key] if key in p else self.u.new_tensor(default)

    def coordinates(self, p) -> tuple[torch.Tensor, torch.Tensor]:
        """(u′, v′), [P, P, s²]: the sub-samples in the mask's own coordinates."""
        s = self.spec
        c = p["mask.centre"] if "mask.centre" in p else self.u.new_tensor(s.centre)
        theta = self._value(p, "rotation", s.rotation)
        scale = self._value(p, "scale", s.scale)
        x, y = self.u - c[0], self.v - c[1]
        cos, sin = torch.cos(theta), torch.sin(theta)
        return (cos * x + sin * y) / scale, (cos * y - sin * x) / scale

    def step(self, distance: torch.Tensor) -> torch.Tensor:
        """1 where ``distance`` > 0, 0 where < 0, a logistic over about one sub-sample between."""
        return torch.sigmoid(distance / self.edge)

    def phase(self, p, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Φ at the mask coordinates (u′, v′), before α."""
        raise NotImplementedError

    def transmission(self, p) -> torch.Tensor:
        """exp(iαΦ) averaged over each pupil sample's sub-samples: [P, P] complex."""
        u, v = self.coordinates(p)
        a = self._value(p, "alpha", self.spec.alpha) * self.phase(p, u, v)
        t = torch.complex(torch.cos(a), torch.sin(a)).mean(-1)
        return t.to(self.grid.kz_imm.dtype)

    def values(self, p) -> dict:
        """Every parameter's value, learnt or held, as plain numbers."""
        out = {}
        for sp in self.specs():
            v = p[sp.name] if sp.name in p else torch.as_tensor(sp.init)
            v = v.detach().cpu().double()
            out[sp.name.removeprefix("mask.")] = v.tolist() if v.dim() else float(v)
        return out

    def describe(self, p) -> dict:
        """The mask's hand-over description: its kind, design, every parameter's value and which were learnt."""
        return dict(config(self.spec), **{k: v for k, v in self.values(p).items()})


class Vortex(PhaseMask):
    """exp(iℓφ): the 2-D donut's mask, and the Vortex PSF's in emission (Hulleman et al. 2021).  Its scale
    changes nothing and its rotation is a piston, so only its centre and α are offered."""
    kind = "vortex"
    placements = ("centre", "alpha")

    def phase(self, p, u, v):
        return self.spec.charge * torch.atan2(v, u)


class TopHat(PhaseMask):
    """π inside a radius r, 0 outside: the 3-D donut's mask.  The radius is its own parameter (``"radius"``) and
    stands for the scale; a rotation changes nothing."""
    kind = "top-hat"
    placements = ("centre", "alpha")
    own = ("radius",)

    def _own_specs(self):
        return [ParamSpec("mask.radius", (), init=float(self.spec.radius), transform=Bounded(0.2, 0.98),
                          fixed="radius" not in self.spec.free)]

    def phase(self, p, u, v):
        r = self._value(p, "radius", self.spec.radius)
        return math.pi * self.step(r - torch.sqrt(u * u + v * v))


class HalfMoon(PhaseMask):
    """π on one half of the pupil (v′ < 0) and δ on the other (``"delta"``): the bilobed beam whose zero δ moves
    (SimuFLUX `library/halfmoon.py`, 1.4 nm per degree in Deguchi & Ries 2024).  A shift along the line
    changes nothing, but the centre is offered as a whole; scale changes nothing."""
    kind = "half-moon"
    placements = ("centre", "rotation", "alpha")
    own = ("delta",)

    def _own_specs(self):
        return [ParamSpec("mask.delta", (), init=float(self.spec.delta), transform=Identity(),
                          fixed="delta" not in self.spec.free)]

    def phase(self, p, u, v):
        delta = self._value(p, "delta", self.spec.delta)
        return math.pi * self.step(-v) + delta * self.step(v)


class DoubleHelix(PhaseMask):
    """Annular zones of vortices whose charge grows by 2 from zone to zone, the zones' edges at √(k/L): Prasad's
    rotating PSF with two lobes (Prasad 2013, Opt Lett 38:585).  An idealisation: a commercial double-helix plate
    is an optimised design, which this approximates; how well is for the data to say."""
    kind = "double helix"

    def phase(self, p, u, v):
        L = int(self.spec.zones)
        rho = torch.sqrt(u * u + v * v)
        charge = sum(2 * self.step(rho - math.sqrt(k / L)) for k in range(1, L))
        return charge * torch.atan2(v, u)


class ZernikeMask(PhaseMask):
    """A design given as Zernikes, {Noll: rad} in the mask's own coordinates (``zernike``): the tetrapod and the
    saddle point, smooth but of high order.  Unlike Zernikes learnt for the system, these move as one block: a
    freed centre, rotation or scale says how the plate sits."""
    kind = "zernike"

    def phase(self, p, u, v):
        rho = torch.sqrt(u * u + v * v)
        phi = torch.atan2(v, u)
        out = torch.zeros_like(rho)
        for j, c in self.spec.zernike.items():
            out = out + float(c) * zernike_at(int(j), rho, phi)
        return out


def zernike_at(j: int, rho: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Noll's Zernike j at (ρ, φ), torch and differentiable: `optics.zernike.zernike`'s polynomial, which works on
    numpy arrays for the grid's fixed bases."""
    n, m = noll_to_nm(j)
    a = abs(m)
    g = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * n + 2)
    r = torch.zeros_like(rho)
    for k in range((n - a) // 2 + 1):
        c = ((-1) ** k * math.factorial(n - k)
             / (math.factorial(k) * math.factorial((n + a) // 2 - k) * math.factorial((n - a) // 2 - k)))
        r = r + c * rho ** (n - 2 * k)
    return g * r * (torch.sin(a * phi) if m < 0 else torch.cos(a * phi))


_CLASSES = {c.kind: c for c in (Vortex, TopHat, HalfMoon, DoubleHelix, ZernikeMask)}


def make_mask(spec: MaskSpec | dict | None, grid: PupilGrid, aperture_na: float | None = None) -> PhaseMask | None:
    """The mask ``spec`` describes on ``grid`` (None for none, or a spec of kind "none")."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        spec = from_config(spec)
    if spec.kind in ("", "none"):
        return None
    if spec.kind not in _CLASSES:
        raise ValueError(f"no phase mask {spec.kind!r}; there are {', '.join(KINDS)}")
    return _CLASSES[spec.kind](grid, spec, aperture_na)


def config(spec: MaskSpec) -> dict:
    """``spec`` as plain data for a model file, only what its kind reads."""
    out = dict(kind=spec.kind, free=list(spec.free), supersample=int(spec.supersample))
    if spec.kind == "vortex":
        out["charge"] = int(spec.charge)
    elif spec.kind == "top-hat":
        out["radius"] = float(spec.radius)
    elif spec.kind == "half-moon":
        out["delta"] = float(spec.delta)
    elif spec.kind == "double helix":
        out["zones"] = int(spec.zones)
    elif spec.kind == "zernike":
        out["zernike"] = {str(j): float(v) for j, v in spec.zernike.items()}
    out.update(centre=[float(c) for c in spec.centre], rotation=float(spec.rotation), scale=float(spec.scale),
               alpha=float(spec.alpha))
    return out


def from_config(d: dict) -> MaskSpec:
    """The spec `config` wrote."""
    d = dict(d)
    if "zernike" in d:
        d["zernike"] = {int(j): float(v) for j, v in d["zernike"].items()}
    if "centre" in d:
        d["centre"] = tuple(float(c) for c in d["centre"])
    d["free"] = tuple(d.get("free", ()))
    known = MaskSpec.__dataclass_fields__
    return MaskSpec(**{k: v for k, v in d.items() if k in known})
