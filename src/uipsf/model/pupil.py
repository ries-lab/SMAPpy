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
either learnt or held, never both.  In situ holds the objective's symmetric
magnitude and the phase above the learnt order from a bead model of the
microscope: learnt from the molecules, the symmetric magnitude trades with the
focus, and the high orders gain little over the beads' while scattering between
molecule sets (`docs/uipsf/architecture.md`, "In situ on matched M2 pairs").
"""
from __future__ import annotations

import torch

from ..optics.grid import PupilGrid
from ..optics.zernike import noll_to_nm, zernike_basis, zernike_modes
from ..params import ParamSpec


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
