"""The pupil function: a magnitude and a phase over the aperture.

Both are returned as real [P, P] maps; the PSF multiplies them with the
aperture, the apodisation and the dipole fields, and normalises the energy
(`VectorPSF.pupil_field`).  The magnitude is 1 + Σ c_j Z_j without a piston
term: its overall scale is fixed by that normalisation, which removes the
exact degeneracy between pupil scale and photons that v1 has (review §A).
It is not clipped at zero either; v1's clip had no gradient there.

**Held coefficients.**  A Zernike pupil may hold some modes at given values
(``held_phase``, ``held_magnitude``, {Noll index: value}): they are no
parameters, only a fixed map added to the learnt modes' sum, so a mode is
either learnt or held, never both; a minimal pupil holds its γ and h the same
way (``held_gamma``, ``held_ring_h``).  In situ holds the objective's magnitude --
a minimal model's γ, or a Zernike model's symmetric modes -- and the phase above
the learnt order from a bead model of the microscope (`model.build.held_from`):
learnt from the molecules, the magnitude trades with the focus, and the high
orders gain little over the beads' while scattering between molecule sets
(`docs/uipsf/architecture.md`, "In situ on matched M2 pairs").
"""
from __future__ import annotations

import math

import torch

from ..optics.grid import PupilGrid
from ..optics.zernike import noll_to_nm, zernike_basis, zernike_modes
from ..params import Bounded, ParamSpec


class ZernikePupil:
    def __init__(self, grid: PupilGrid, n_max: int = 6, magnitude_n_max: int | None = None,
                 phase_modes: list[int] | None = None, magnitude_modes: list[int] | None = None,
                 high_na: bool = False, held_phase: dict | None = None, held_magnitude: dict | None = None):
        """Phase modes default to Noll 5 … up to n_max; magnitude modes to Noll 2 … up to
        ``magnitude_n_max`` (default n_max).  ``magnitude_n_max=0`` gives a uniform magnitude.
        ``held_phase`` / ``held_magnitude`` hold modes at given values (module docstring); a held mode is
        taken out of the learnt ones."""
        self.grid, self.high_na = grid, high_na
        self.held_phase = {int(j): float(v) for j, v in (held_phase or {}).items()}
        self.held_magnitude = {int(j): float(v) for j, v in (held_magnitude or {}).items()}
        phase = phase_modes if phase_modes is not None else zernike_modes(n_max)
        self.phase_modes = [j for j in phase if j not in self.held_phase]
        mn = n_max if magnitude_n_max is None else magnitude_n_max
        magnitude = (magnitude_modes if magnitude_modes is not None
                     else zernike_modes(mn, exclude=("piston",)) if mn > 0 else [])
        self.magnitude_modes = [j for j in magnitude if j not in self.held_magnitude]
        self.phase_basis = zernike_basis(grid, self.phase_modes, high_na=high_na)
        self.magnitude_basis = zernike_basis(grid, self.magnitude_modes)
        self._held_maps()

    def _held_maps(self):
        """The held modes' fixed maps, [P, P] (0 where none is held)."""
        def fixed(held, high_na):
            if not held:
                return 0.0
            modes = list(held)
            basis = zernike_basis(self.grid, modes, high_na=high_na)
            c = torch.tensor([held[j] for j in modes], dtype=basis.dtype, device=basis.device)
            return torch.tensordot(c, basis, dims=([0], [0]))
        self.held_phase_map = fixed(self.held_phase, self.high_na)
        self.held_magnitude_map = fixed(self.held_magnitude, False)

    def coefficients(self, p: dict[str, torch.Tensor], which: str) -> dict[int, float]:
        """Every mode's coefficient of ``which`` ("phase" or "magnitude"), {Noll: value}: the learnt ones
        from ``p`` and the held ones, so a consumer sees the whole pupil."""
        held = self.held_phase if which == "phase" else self.held_magnitude
        modes = self.phase_modes if which == "phase" else self.magnitude_modes
        learnt = p[f"pupil.{which}"].detach().cpu().double().reshape(-1).tolist() if modes else []
        out = dict(zip(modes, learnt, strict=True))
        out.update(held)
        return dict(sorted(out.items()))

    def specs(self) -> list[ParamSpec]:
        return [ParamSpec("pupil.phase", (len(self.phase_modes),)),
                ParamSpec("pupil.magnitude", (len(self.magnitude_modes),))]

    def describe(self, p) -> dict:
        """The pupil's hand-over description, plain data (`model.build`, `docs/uipsf/architecture.md`, "One
        PSF model"): ``kind`` "zernike", every phase mode in nm of wavefront (so that it holds at another
        emission wavelength) and every magnitude mode, held ones included.  `MinimalPupil` gives γ instead."""
        own = float(self.grid.optics.wavelength_nm)
        out = {"kind": "zernike",
               "phase_nm": {str(j): round(v * own / (2 * math.pi), 3)
                            for j, v in self.coefficients(p, "phase").items()}}
        if self.magnitude_modes or self.held_magnitude:
            out["magnitude"] = {str(j): round(v, 5) for j, v in self.coefficients(p, "magnitude").items()}
        return out

    def truncated(self, n_max: int) -> ZernikePupil:
        """The same pupil with only the modes of radial order ≤ ``n_max``: a fit's first stage."""
        new = object.__new__(ZernikePupil)
        new.grid, new.high_na = self.grid, self.high_na
        # the held modes stay held at every stage, whatever their order
        new.held_phase, new.held_magnitude = self.held_phase, self.held_magnitude
        new.held_phase_map, new.held_magnitude_map = self.held_phase_map, self.held_magnitude_map
        keep_p = [i for i, j in enumerate(self.phase_modes) if noll_to_nm(j)[0] <= n_max]
        keep_m = [i for i, j in enumerate(self.magnitude_modes) if noll_to_nm(j)[0] <= n_max]
        new.phase_modes = [self.phase_modes[i] for i in keep_p]
        new.magnitude_modes = [self.magnitude_modes[i] for i in keep_m]
        new.phase_basis, new.magnitude_basis = self.phase_basis[keep_p], self.magnitude_basis[keep_m]
        return new

    def embed(self, coefficients: torch.Tensor, modes: list[int], which: str) -> torch.Tensor:
        """Coefficients of ``modes`` placed into this pupil's ``which`` ("phase" or "magnitude") modes,
        zeros elsewhere."""
        own = self.phase_modes if which == "phase" else self.magnitude_modes
        out = coefficients.new_zeros(len(own))
        for c, j in zip(coefficients, modes, strict=True):
            out[own.index(j)] = c
        return out

    def __call__(self, p: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """(magnitude, phase), each [..., P, P]; coefficients may carry leading (emitter) axes."""
        phase = torch.tensordot(p["pupil.phase"], self.phase_basis, dims=([-1], [0])) + self.held_phase_map
        magnitude = (1 + torch.tensordot(p["pupil.magnitude"], self.magnitude_basis, dims=([-1], [0]))
                     + self.held_magnitude_map)
        return magnitude, phase


#: how far beyond the objective's NA a minimal pupil's grid reaches: 1.51 for NA 1.50, M5's.  On a grid at the
#: nominal NA the grid's own edge cut light the beads wanted (`docs/uipsf/m5-media.md`); the aperture is then
#: the pupil's own cutoff, inside the grid
GRID_MARGIN = 1.51 / 1.50


def minimal_grid_optics(optics):
    """The optics a minimal pupil's grid is built on: the NA `GRID_MARGIN` beyond the objective's, short of
    the immersion index, which no NA reaches."""
    import dataclasses
    na = min(optics.NA * GRID_MARGIN, optics.n_imm - 0.005)
    return dataclasses.replace(optics, NA=na)


class MinimalPupil(ZernikePupil):
    """The minimal pupil (the user's model, decided November 2026 on M5's beads): the phase in Zernikes, the
    magnitude rotationally symmetric and physical, cos θ_imm^γ × the supercritical ring × the aperture.

    - **cos θ_imm^γ**, γ ≥ 0 learnt: the shape an apodisation error has, falling towards the edge.  It was
      0.32-0.41 on M5's 100 nm beads in both colours, media and with and without the 3D lens; 40 nm beads
      learnt 0.13-0.26 beside a larger blur, which is not settled (`benchmarks/uipsf/real/m5_consensus.py`).
    - **the ring**, only where the sample's index is below the NA (else there is no supercritical light):
      exp(−2πκh), κ = √((ρ·NA)² − n_med²)/λ, h ≥ 0 the emitter's height above the glass, which the bead
      model puts at the glass; one parameter, learnt with ``saf_damping`` (the default).  It belongs to the beads,
      as their size does: a spline for single molecules is drawn with h = 0 (`export.sample_psf`), their light
      beyond the critical angle following the physics of their own depth.
    - **the aperture**, held: a cutoff at the objective's ``na`` of the width of a pupil sample, on a grid
      reaching `GRID_MARGIN` beyond it (`minimal_grid_optics`), so the aperture is not the grid's edge.

    No m ≠ 0 magnitude: free, the asymmetric modes won nats on every M5 set, but held at the other sets'
    consensus they cost 2-36 k nats each -- they follow the acquisition, not the objective.  The parameters
    are physical, so a model is read by its numbers: γ (``pupil.gamma``) and h (``pupil.ring_h``, nm).
    Taken over from `benchmarks/uipsf/synthetic/radial_magnitude.EdgePupil` ``centre="cos", edge="hard",
    saf=True``, with the cutoff held; its rows are the studies'."""

    def __init__(self, grid: PupilGrid, n_max: int = 6, na: float | None = None, held_phase: dict | None = None,
                 phase_modes: list[int] | None = None, saf_damping: bool = True, held_gamma: float | None = None,
                 held_ring_h: float | None = None):
        """``held_gamma`` / ``held_ring_h`` hold γ and h at given values, as ``held_phase`` holds modes: no
        parameters then.  In situ holds a bead model's γ (`model.build.Held`): learnt from molecules the
        magnitude trades with the focus, as the symmetric Zernikes did."""
        super().__init__(grid, n_max=n_max, phase_modes=phase_modes, magnitude_modes=[], held_phase=held_phase)
        o = grid.optics
        self.na = float(o.NA / GRID_MARGIN if na is None else na)
        self.n_imm, self.n_med, self.wavelength_nm, self.grid_na = o.n_imm, o.n_med, o.wavelength_nm, o.NA
        # without ``saf_damping`` the ring is a point's at the glass: the physics alone, no height learnt
        self.saf_damping = bool(saf_damping)
        self.ring = self.saf_damping and o.n_med < self.na
        self.held_gamma = None if held_gamma is None else float(held_gamma)
        self.held_ring_h = None if held_ring_h is None else float(held_ring_h)
        rho = grid.rho.real if grid.rho.is_complex() else grid.rho
        self._extras(rho, grid.size)

    def _extras(self, rho, size):
        """The fixed maps: cos θ_imm, κ and the aperture."""
        s = rho * self.grid_na                                        # n·sin θ
        self.cos_imm = torch.sqrt(torch.clamp(1 - (s / self.n_imm) ** 2, min=1e-6))
        self.kappa = torch.sqrt(torch.clamp(s ** 2 - self.n_med ** 2, min=0.0)) / self.wavelength_nm
        # half a pupil sample at least, so the samples it crosses are partly lit (EdgePupil's `cut_w`)
        width = max(0.01, 1.0 / size)
        self.cutoff = torch.sigmoid((self.na / self.grid_na - rho) / width)

    def specs(self) -> list[ParamSpec]:
        extra = []
        if self.held_gamma is None:
            extra.append(ParamSpec("pupil.gamma", (), init=0.05, transform=Bounded(0.0, 2.0)))
        if self.ring and self.held_ring_h is None:
            extra.append(ParamSpec("pupil.ring_h", (), init=30.0, transform=Bounded(0.0, 400.0)))
        return super().specs() + extra

    def gamma(self, p) -> torch.Tensor:
        """γ: the learnt one, or the held one."""
        return p["pupil.gamma"] if self.held_gamma is None else self.grid.rho.real.new_tensor(self.held_gamma)

    def ring_h(self, p) -> torch.Tensor:
        """h, nm: the learnt one, or the held one."""
        return p["pupil.ring_h"] if self.held_ring_h is None else self.grid.rho.real.new_tensor(self.held_ring_h)

    def truncated(self, n_max: int) -> MinimalPupil:
        new = super().truncated(n_max)
        out = object.__new__(MinimalPupil)
        out.__dict__.update(new.__dict__)
        for k in ("na", "n_imm", "n_med", "wavelength_nm", "grid_na", "saf_damping", "ring", "held_gamma",
                  "held_ring_h", "cos_imm", "kappa", "cutoff"):
            setattr(out, k, getattr(self, k))
        return out

    def symmetric(self, p, cos_imm=None, kappa=None, cutoff=None) -> torch.Tensor:
        """The magnitude's factor, [..., P, P] (or at given maps' points, for a profile)."""
        cos_imm = self.cos_imm if cos_imm is None else cos_imm
        out = cos_imm ** self.gamma(p)[..., None, None]
        if self.ring:
            k = self.kappa if kappa is None else kappa
            out = out * torch.exp(-2 * math.pi * k * self.ring_h(p)[..., None, None])
        return out * (self.cutoff if cutoff is None else cutoff)

    def __call__(self, p):
        magnitude, phase = super().__call__(p)
        return magnitude * self.symmetric(p), phase

    def values(self, p) -> dict:
        """γ and, with a ring, h (nm): the magnitude's numbers, learnt or held."""
        out = {"gamma": float(self.gamma(p))}
        if self.ring:
            out["ring_h_nm"] = float(self.ring_h(p))
        return out

    def without_ring(self) -> MinimalPupil:
        """The same pupil for point emitters: no damping of the supercritical ring, which is the beads' (their
        dye above the glass); a point's light beyond the critical angle follows the physics of its own depth.
        A ring at h = 0 is exactly no ring, so the ring is switched off rather than h set to 0."""
        out = object.__new__(MinimalPupil)
        out.__dict__.update(self.__dict__)
        out.ring, out.held_ring_h = False, None
        return out

    def describe(self, p) -> dict:
        """The pupil's hand-over description (`ZernikePupil.describe`): kind "minimal", the phase in nm and γ.
        h is left out: it is the beads', and nothing downstream holds it."""
        return dict(super().describe(p), kind="minimal", gamma=round(float(self.gamma(p)), 4))


class PixelPupil:
    """A free magnitude and phase per pupil sample (v1's pupil model).  Many parameters:
    the solver treats it with L-BFGS (plan §5.6)."""

    def __init__(self, grid: PupilGrid):
        self.grid = grid

    def specs(self) -> list[ParamSpec]:
        P = self.grid.size
        return [ParamSpec("pupil.phase_map", (P, P)), ParamSpec("pupil.magnitude_map", (P, P), init=1.0)]

    def __call__(self, p):
        return p["pupil.magnitude_map"], p["pupil.phase_map"]
