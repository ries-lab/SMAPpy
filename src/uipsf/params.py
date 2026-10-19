"""Parameters: what each component declares, and how the solver sees them.

A component lists its parameters as `ParamSpec`s.  Each has a scope --
*shared* by all emitters (pupil, blur) or one value per *emitter*
(position, photons) -- a transform that keeps it in bounds, an optional
prior in physical units, and a fixed flag.  The solver reads the block
structure from the specs alone: shared parameters form one dense block,
each emitter's its own (plan §5.3, §5.6).

Values are stored *unconstrained*; components only ever see the physical
values from `Params.physical()`.  Bounds by reparametrisation replace v1's
ramped penalties, and priors in physical units replace its loss weights;
there is no variable scaling anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F


class Transform:
    """Physical value = forward(unconstrained)."""

    def forward(self, x):
        return x

    def inverse(self, y):
        return y


class Identity(Transform):
    pass


class Magnitude(Transform):
    """|x|: for a parameter the model sees only squared, such as a Gaussian's σ.  Its sign is not
    identifiable, and a solver's step may cross zero (a cold and a warm fit gave σ = ±40.8 nm at the
    same objective, M7); reporting the magnitude makes the two the same answer.  At x = 0 the
    derivative is 0, as that of σ² already is."""

    def forward(self, x):
        return x.abs()

    def inverse(self, y):
        return torch.as_tensor(y).abs()


class Softplus(Transform):
    """Positive values with a gradient that does not vanish near zero, unlike exp."""

    def __init__(self, floor: float = 0.0):
        self.floor = floor

    def forward(self, x):
        return F.softplus(x) + self.floor

    def inverse(self, y):
        y = torch.as_tensor(y) - self.floor
        return y + torch.log(-torch.expm1(-y))


class Exp(Transform):
    def forward(self, x):
        return torch.exp(x)

    def inverse(self, y):
        return torch.log(torch.as_tensor(y))


class Bounded(Transform):
    """lo < value < hi through a logistic function."""

    def __init__(self, lo: float, hi: float):
        if not hi > lo:
            raise ValueError("Bounded needs hi > lo")
        self.lo, self.hi = lo, hi

    def forward(self, x):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(x)

    def inverse(self, y):
        t = (torch.as_tensor(y) - self.lo) / (self.hi - self.lo)
        return torch.log(t) - torch.log1p(-t)


@dataclass(frozen=True)
class Gaussian:
    """A Gaussian prior on the physical value: adds ½((value − mean)/sigma)² to the objective."""
    mean: float | torch.Tensor
    sigma: float | torch.Tensor

    def penalty(self, value: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.residual(value) ** 2).sum()

    def residual(self, value: torch.Tensor) -> torch.Tensor:
        """(value − mean)/sigma: the prior is a least-squares term, so it adds exactly to Gauss–Newton."""
        return (value - self.mean) / self.sigma


@dataclass(frozen=True)
class ParamSpec:
    name: str                                   # "pupil.phase", "emitter.xyz", "psf.sigma", …
    shape: tuple[int, ...]                      # without the emitter axis
    scope: Literal["shared", "emitter"] = "shared"
    init: float | torch.Tensor = 0.0            # physical value(s)
    transform: Transform = field(default_factory=Identity)
    prior: Gaussian | None = None
    fixed: bool = False

    @property
    def size(self) -> int:
        return math.prod(self.shape)


class Params:
    """Unconstrained values for a set of specs, packed into a shared vector and an emitter matrix."""

    def __init__(self, specs, n_emitters: int = 0, device=None, dtype=torch.float32, values=None):
        self.specs = {s.name: s for s in specs}
        if len(self.specs) != len(specs):
            raise ValueError("parameter names must be unique")
        self.n_emitters = int(n_emitters)
        self.values = {}
        values = values or {}
        for s in self.specs.values():
            shape = (self.n_emitters, *s.shape) if s.scope == "emitter" else s.shape
            phys = values.get(s.name, s.init)
            phys = torch.as_tensor(phys, dtype=dtype, device=device).broadcast_to(shape).clone()
            self.values[s.name] = s.transform.inverse(phys).to(dtype)

    def physical(self) -> dict[str, torch.Tensor]:
        return {k: self.specs[k].transform.forward(v) for k, v in self.values.items()}

    def prior_penalty(self) -> torch.Tensor:
        phys = self.physical()
        terms = [s.prior.penalty(phys[n]) for n, s in self.specs.items() if s.prior is not None]
        return torch.stack(terms).sum() if terms else torch.zeros(())

    def _free(self, scope):
        return [n for n, s in self.specs.items() if s.scope == scope and not s.fixed]

    def layout(self, scope: str) -> list[tuple[str, int, int]]:
        """(name, start, stop) of each free parameter in the packed block of ``scope``."""
        out, i = [], 0
        for n in self._free(scope):
            out.append((n, i, i + self.specs[n].size))
            i += self.specs[n].size
        return out

    def subset(self, index) -> Params:
        """The same parameters for some emitters only (a slice or index tensor): a solver's chunk."""
        new = self.replace()
        for n, s in self.specs.items():
            if s.scope == "emitter":
                new.values[n] = self.values[n][index]
        new.n_emitters = next((v.shape[0] for n, v in new.values.items()
                               if self.specs[n].scope == "emitter"), 0)
        return new

    def prior_residuals(self, scope: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The priors as weighted residuals r and dr/du, packed like the block of ``scope``.

        Free parameters without a prior get zeros.  The priors' share of the
        objective is ½Σr², of the Gauss–Newton Hessian diag((dr/du)²) and of
        the gradient (dr/du)·r: exact for a parameter without a transform.
        """
        dphys = self.derivative(scope)
        r, dr = torch.zeros_like(dphys), torch.zeros_like(dphys)
        phys = self.physical()
        for n, a, b in self.layout(scope):
            prior = self.specs[n].prior
            if prior is None:
                continue
            res = prior.residual(phys[n])
            sigma = torch.as_tensor(prior.sigma, dtype=res.dtype, device=res.device).broadcast_to(res.shape)
            if scope == "emitter":
                r[:, a:b] = res.reshape(self.n_emitters, -1)
                dr[:, a:b] = dphys[:, a:b] / sigma.reshape(self.n_emitters, -1)
            else:
                r[a:b] = res.reshape(-1)
                dr[a:b] = dphys[a:b] / sigma.reshape(-1)
        return r, dr

    def derivative(self, scope: str) -> torch.Tensor:
        """d(physical)/d(unconstrained) of each packed free parameter: [P_s] or [E, P_e].

        Every transform acts elementwise, so one forward-mode pass with a unit
        tangent gives the whole diagonal.  The solver needs it for the priors'
        Gauss–Newton terms and to turn covariances into physical units.
        """
        parts = []
        for n in self._free(scope):
            v = self.values[n]
            _, d = torch.func.jvp(self.specs[n].transform.forward, (v,), (torch.ones_like(v),))
            parts.append(d.reshape(self.n_emitters, -1) if scope == "emitter" else d.reshape(-1))
        if scope == "emitter":
            return torch.cat(parts, 1) if parts else torch.zeros(self.n_emitters, 0)
        return torch.cat(parts) if parts else torch.zeros(0)

    def shared_vector(self) -> torch.Tensor:
        """[P_s]: the free shared parameters, unconstrained, in declaration order."""
        parts = [self.values[n].reshape(-1) for n in self._free("shared")]
        return torch.cat(parts) if parts else torch.zeros(0)

    def emitter_matrix(self) -> torch.Tensor:
        """[E, P_e]: each emitter's free parameters, unconstrained."""
        parts = [self.values[n].reshape(self.n_emitters, -1) for n in self._free("emitter")]
        return torch.cat(parts, 1) if parts else torch.zeros(self.n_emitters, 0)

    def replace(self, shared: torch.Tensor | None = None, emitter: torch.Tensor | None = None) -> Params:
        """A copy with the free parameters taken from packed blocks (the inverse of the two above)."""
        new = object.__new__(Params)
        new.specs, new.n_emitters, new.values = self.specs, self.n_emitters, dict(self.values)
        if shared is not None:
            i = 0
            for n in self._free("shared"):
                k = self.specs[n].size
                new.values[n] = shared[i:i + k].reshape(self.specs[n].shape)
                i += k
        if emitter is not None:
            i = 0
            for n in self._free("emitter"):
                k = self.specs[n].size
                new.values[n] = emitter[:, i:i + k].reshape(self.n_emitters, *self.specs[n].shape)
                i += k
        return new

    def fix(self, *names) -> Params:
        return self._with_fixed(names, True)

    def free(self, *names) -> Params:
        return self._with_fixed(names, False)

    def _with_fixed(self, names, fixed):
        new = self.replace()
        new.specs = dict(self.specs)
        for n in names:
            s = self.specs[n]
            new.specs[n] = ParamSpec(s.name, s.shape, s.scope, s.init, s.transform, s.prior, fixed)
        return new
