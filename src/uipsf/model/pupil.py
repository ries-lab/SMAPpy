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
either learnt or held, never both; a minimal pupil holds its γ, and any pupil its
ring's h, the same way (``held_gamma``, ``held_ring_h``).  In situ holds the objective's magnitude --
a minimal model's γ, or a Zernike model's symmetric modes -- and the phase above
the learnt order from a bead model of the microscope (`model.build.held_from`):
learnt from the molecules, the magnitude trades with the focus, and the high
orders gain little over the beads' while scattering between molecule sets
(`docs/uipsf/architecture.md`, "In situ on matched M2 pairs").

**The supercritical ring** (``saf_damping``): exp(−2πκh) on the light beyond the
critical angle, h the beads' dye above the glass, which the bead model puts at the
glass.  As one factor it is the beads', and `without_ring` drops it.  A minimal
magnitude has it by default; a Zernike magnitude only on request (``zernike_ring``
in the builder), off by default: on M5's beads the learnt h ran to 120–160 nm as
it traded with the high radial modes, and the spline drawn without that h
compressed z by up to 17 % (`docs/uipsf/architecture.md`, "The supercritical ring
on a Zernike magnitude").
"""
from __future__ import annotations

import math

import torch

from ..optics import medium
from ..optics.grid import PupilGrid
from ..optics.zernike import noll_to_nm, zernike_basis, zernike_modes
from ..params import Bounded, ParamSpec


class ZernikePupil:
    def __init__(self, grid: PupilGrid, n_max: int = 6, magnitude_n_max: int | None = None,
                 phase_modes: list[int] | None = None, magnitude_modes: list[int] | None = None,
                 high_na: bool = False, held_phase: dict | None = None, held_magnitude: dict | None = None,
                 mask=None, saf_damping: bool = False, held_ring_h: float | None = None):
        """Phase modes default to Noll 5 … up to n_max; magnitude modes to Noll 2 … up to
        ``magnitude_n_max`` (default n_max).  ``magnitude_n_max=0`` gives a uniform magnitude.
        ``held_phase`` / ``held_magnitude`` hold modes at given values (module docstring); a held mode is
        taken out of the learnt ones.  ``mask``: a phase mask (`model.mask`), its transmission multiplied into
        the pupil field (`transmission`), the Zernikes learnt beside it being the system's aberrations.
        ``saf_damping``: the supercritical ring damped by a learnt height (module docstring), h held at
        ``held_ring_h`` if given."""
        self.grid, self.high_na = grid, high_na
        self.mask = mask
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
        self._ring_maps(saf_damping, held_ring_h, grid.optics.NA)

    def _ring_maps(self, saf_damping: bool, held_ring_h: float | None, na: float):
        """κ = √((ρ·NA_grid)² − n_med²)/λ, and whether there is a ring: only where the typed sample's index is
        below the aperture's ``na`` (else there is no supercritical light), decided once (`MinimalPupil`)."""
        o = self.grid.optics
        # without ``saf_damping`` the ring is a point's at the glass: the physics alone, no height learnt
        self.saf_damping = bool(saf_damping)
        self.ring = self.saf_damping and o.n_med < na
        self.held_ring_h = None if held_ring_h is None else float(held_ring_h)
        rho = self.grid.rho.real if self.grid.rho.is_complex() else self.grid.rho
        self.kappa = torch.sqrt(torch.clamp((rho * o.NA) ** 2 - o.n_med ** 2, min=0.0)) / o.wavelength_nm

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
                ParamSpec("pupil.magnitude", (len(self.magnitude_modes),))] + self._ring_specs() + self._mask_specs()

    def _ring_specs(self) -> list[ParamSpec]:
        if self.ring and self.held_ring_h is None:
            return [ParamSpec("pupil.ring_h", (), init=30.0, transform=Bounded(0.0, 400.0))]
        return []

    def ring_h(self, p) -> torch.Tensor:
        """h, nm: the learnt one, or the held one."""
        return p["pupil.ring_h"] if self.held_ring_h is None else self.grid.rho.real.new_tensor(self.held_ring_h)

    def ring_factor(self, p, kappa=None):
        """exp(−2πκh), [..., P, P] (or at given κ, for a profile); 1 without a ring.  With a learnt sample index
        (``optics.n_med``, `VectorPSF.learn_n_med`) κ follows it, through the critical angle's smoothed root
        (`optics.medium`), so it stays differentiable as n_med crosses a sample of the pupil -- but a ring the
        typed index did not give is not added, nor h with it, which the data could not fix while n_med stays
        above the NA."""
        if not self.ring:
            return 1.0
        if kappa is None:
            kappa = self.kappa if "optics.n_med" not in p else medium.maps(self.grid, p["optics.n_med"]).kappa
        return torch.exp(-2 * math.pi * kappa * self.ring_h(p)[..., None, None])

    def without_ring(self):
        """The same pupil for point emitters: no damping of the supercritical ring, which is the beads' (their
        dye above the glass); a point's light beyond the critical angle follows the physics of its own depth.
        A ring at h = 0 is exactly no ring, so the ring is switched off rather than h set to 0."""
        out = object.__new__(type(self))
        out.__dict__.update(self.__dict__)
        out.ring, out.held_ring_h = False, None
        return out

    def values(self, p) -> dict:
        """The magnitude's physical numbers: h (nm) with a ring."""
        return {"ring_h_nm": float(self.ring_h(p))} if self.ring else {}

    def _mask_specs(self) -> list[ParamSpec]:
        return self.mask.specs() if self.mask is not None else []

    def transmission(self, p) -> torch.Tensor | None:
        """The phase mask's complex transmission, [P, P], or None without a mask.  Apart from (magnitude,
        phase): a mask's mean over a pupil sample is complex, its phase undefined where it is ~0 (a vortex's
        centre), so it is multiplied into the field rather than added to the phase (`VectorPSF.pupil_field`)."""
        return self.mask.transmission(p) if self.mask is not None else None

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
        if self.mask is not None:
            out["mask"] = self.mask.describe(p)
        return out

    def truncated(self, n_max: int) -> ZernikePupil:
        """The same pupil with only the modes of radial order ≤ ``n_max``: a fit's first stage."""
        new = object.__new__(ZernikePupil)
        new.grid, new.high_na, new.mask = self.grid, self.high_na, self.mask
        # the held modes stay held at every stage, whatever their order
        new.held_phase, new.held_magnitude = self.held_phase, self.held_magnitude
        new.held_phase_map, new.held_magnitude_map = self.held_phase_map, self.held_magnitude_map
        for k in ("saf_damping", "ring", "held_ring_h", "kappa"):
            setattr(new, k, getattr(self, k))
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

    def zernike_maps(self, p: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """The Zernikes' (magnitude, phase) alone, without the ring."""
        phase = torch.tensordot(p["pupil.phase"], self.phase_basis, dims=([-1], [0])) + self.held_phase_map
        magnitude = (1 + torch.tensordot(p["pupil.magnitude"], self.magnitude_basis, dims=([-1], [0]))
                     + self.held_magnitude_map)
        return magnitude, phase

    def __call__(self, p: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """(magnitude, phase), each [..., P, P]; coefficients may carry leading (emitter) axes."""
        magnitude, phase = self.zernike_maps(p)
        return magnitude * self.ring_factor(p), phase


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
      model puts at the glass; one parameter, learnt with ``saf_damping`` (the default).  Whether there is a
      ring is decided when the pupil is built, from the typed n_med: with a learnt one (``optics.n_med``,
      `VectorPSF.learn_n_med`) κ follows it, through the critical angle's smoothed root (`optics.medium`), so
      it stays differentiable as n_med crosses a sample of the pupil -- but a ring the typed index did not
      give is not added, nor h with it, which the data could not fix while n_med stays above the NA.  It
      belongs to the beads, as their size does: a spline for single molecules is drawn with h = 0
      (`export.sample_psf`), their light beyond the critical angle following the physics of their own depth.
    - **the aperture**: a cutoff at the objective's ``na`` of the width of a pupil sample, on a grid
      reaching `GRID_MARGIN` beyond it (`minimal_grid_optics`), so the aperture is not the grid's edge.  Held
      by default; with ``learn_na`` the NA is a parameter, ``pupil.na``: the 4Pi beads of Zenodo wanted an
      effective NA of 1.31-1.32 behind objectives of 1.35, one objective's beads 1.40 behind 1.43 -- 9 700
      nats on the 4Pi beads (docs/4pi.md §23).  Why the aperture is smaller is not settled (a medium of
      lower index than assumed would cut it at n_med), so it is an option, not the default.

    No m ≠ 0 magnitude: free, the asymmetric modes won nats on every M5 set, but held at the other sets'
    consensus they cost 2-36 k nats each -- they follow the acquisition, not the objective.  The parameters
    are physical, so a model is read by its numbers: γ (``pupil.gamma``) and h (``pupil.ring_h``, nm).
    Taken over from `benchmarks/uipsf/synthetic/radial_magnitude.EdgePupil` ``centre="cos", edge="hard",
    saf=True``, with the cutoff held; its rows are the studies'."""

    def __init__(self, grid: PupilGrid, n_max: int = 6, na: float | None = None, held_phase: dict | None = None,
                 phase_modes: list[int] | None = None, saf_damping: bool = True, held_gamma: float | None = None,
                 held_ring_h: float | None = None, learn_na: bool = False, mask=None):
        """``held_gamma`` / ``held_ring_h`` hold γ and h at given values, as ``held_phase`` holds modes: no
        parameters then.  In situ holds a bead model's γ (`model.build.Held`): learnt from molecules the
        magnitude trades with the focus, as the symmetric Zernikes did.  ``learn_na`` learns the aperture's NA
        from ``na`` (`aperture`)."""
        super().__init__(grid, n_max=n_max, phase_modes=phase_modes, magnitude_modes=[], held_phase=held_phase,
                         mask=mask)
        o = grid.optics
        self.na = float(o.NA / GRID_MARGIN if na is None else na)
        self.n_imm, self.n_med, self.wavelength_nm, self.grid_na = o.n_imm, o.n_med, o.wavelength_nm, o.NA
        # the ring against the objective's NA, not the grid's beyond it
        self._ring_maps(saf_damping, held_ring_h, self.na)
        self.held_gamma = None if held_gamma is None else float(held_gamma)
        self.learn_na = bool(learn_na)
        rho = grid.rho.real if grid.rho.is_complex() else grid.rho
        self._extras(rho, grid.size)

    def _extras(self, rho, size):
        """The fixed maps: cos θ_imm and the aperture (κ is the ring's, `ZernikePupil._ring_maps`)."""
        s = rho * self.grid_na                                        # n·sin θ
        self.cos_imm = torch.sqrt(torch.clamp(1 - (s / self.n_imm) ** 2, min=1e-6))
        # half a pupil sample at least, so the samples it crosses are partly lit (EdgePupil's `cut_w`)
        width = max(0.01, 1.0 / size)
        self.cutoff = torch.sigmoid((self.na / self.grid_na - rho) / width)

    def _cutoff(self, na):
        """The aperture at ``na`` (a tensor, [...]): the held one's sigmoid, [..., P, P]."""
        rho = self.grid.rho.real if self.grid.rho.is_complex() else self.grid.rho
        width = max(0.01, 1.0 / self.grid.size)
        return torch.sigmoid((na[..., None, None] / self.grid_na - rho) / width)

    def specs(self) -> list[ParamSpec]:
        extra = []
        if self.held_gamma is None:
            extra.append(ParamSpec("pupil.gamma", (), init=0.05, transform=Bounded(0.0, 2.0)))
        if self.learn_na:
            # started at the typed NA, inside its bounds: 5 % above it, so that the data can say it is right
            extra.append(ParamSpec("pupil.na", (), init=self.na,
                                   transform=Bounded(0.8 * self.na, min(1.05 * self.na, self.grid_na))))
        return super().specs() + extra

    def gamma(self, p) -> torch.Tensor:
        """γ: the learnt one, or the held one."""
        return p["pupil.gamma"] if self.held_gamma is None else self.grid.rho.real.new_tensor(self.held_gamma)

    def aperture(self, p) -> torch.Tensor:
        """The aperture's NA: the learnt one, or the objective's."""
        return p["pupil.na"] if self.learn_na else self.grid.rho.real.new_tensor(self.na)

    def truncated(self, n_max: int) -> MinimalPupil:
        new = super().truncated(n_max)
        out = object.__new__(MinimalPupil)
        out.__dict__.update(new.__dict__)
        for k in ("na", "n_imm", "n_med", "wavelength_nm", "grid_na", "held_gamma", "learn_na", "cos_imm",
                  "cutoff"):
            setattr(out, k, getattr(self, k))
        return out

    def symmetric(self, p, cos_imm=None, kappa=None, cutoff=None) -> torch.Tensor:
        """The magnitude's factor, [..., P, P] (or at given maps' points, for a profile)."""
        cos_imm = self.cos_imm if cos_imm is None else cos_imm
        out = cos_imm ** self.gamma(p)[..., None, None] * self.ring_factor(p, kappa)
        if cutoff is None:
            cutoff = self._cutoff(p["pupil.na"]) if self.learn_na else self.cutoff
        return out * cutoff

    def __call__(self, p):
        magnitude, phase = self.zernike_maps(p)
        return magnitude * self.symmetric(p), phase

    def values(self, p) -> dict:
        """γ and, with a ring, h (nm): the magnitude's numbers, learnt or held."""
        out = {"gamma": float(self.gamma(p)), **super().values(p)}
        if self.learn_na:
            out["na"] = float(self.aperture(p))
        return out

    def describe(self, p) -> dict:
        """The pupil's hand-over description (`ZernikePupil.describe`): kind "minimal", the phase in nm, γ and
        a learnt NA (``na``), which a model holding γ holds too (`model.build.held_from`): the objective's, as γ
        is.  h is left out: it is the beads', and nothing downstream holds it."""
        out = dict(super().describe(p), kind="minimal", gamma=round(float(self.gamma(p)), 4))
        if self.learn_na:
            out["na"] = round(float(self.aperture(p)), 4)
        return out


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
