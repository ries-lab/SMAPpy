"""The microscope, as the forward model sees it.

Lengths are nanometres throughout (`docs/uipsf/conventions.md`).  The pixel size is
given for x and y separately: cameras behind a cylindrical lens or an
anamorphic relay have non-square pixels (the paper's M2 data: 127 × 116 nm).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Optics:
    NA: float
    wavelength_nm: float
    n_imm: float                     # immersion medium (oil: 1.518)
    n_med: float                     # sample medium (water: 1.33)
    pixel_size_nm: tuple[float, float]   # (x, y) on the sample
    n_cov: float | None = None       # coverslip; None means the same as the immersion

    def __post_init__(self):
        if self.n_cov is None:
            object.__setattr__(self, "n_cov", self.n_imm)
        if not 0 < self.NA < self.n_imm:
            raise ValueError(f"NA must be between 0 and the immersion index {self.n_imm}, got {self.NA}")
        if self.wavelength_nm <= 0 or min(self.pixel_size_nm) <= 0:
            raise ValueError("wavelength and pixel size must be positive (nanometres)")
        object.__setattr__(self, "pixel_size_nm", tuple(float(p) for p in self.pixel_size_nm))

    @property
    def k_max(self) -> float:
        """The pupil's edge as a spatial frequency, NA/λ in cycles per nm."""
        return self.NA / self.wavelength_nm


def choose_pupil_size(optics: Optics, z_max_nm: float, max_phase_step: float = math.pi / 2,
                      minimum: int = 64, multiple: int = 16) -> int:
    """The number of pupil samples across the aperture that keeps defocus from aliasing.

    The defocus phase 2π·k_z·z is steepest at the pupil's edge, where
    dk_z/dρ = NA²/(λ·sqrt(n² − NA²)) per unit of normalised radius ρ.  Over a
    pupil sample (Δρ = 2/P) the phase may step by at most ``max_phase_step``
    at the largest |z|; π aliases outright (review §A: at NA 1.43 and 64
    samples that happens near |z| ≈ 2.5–3 µm), so π/2 keeps a margin.  The
    result is rounded up to a multiple of 16 and is never below 64, uiPSF's
    fixed size, so the matrix DFT keeps sizes that suit the hardware.
    """
    n = optics.n_imm
    slope = optics.NA ** 2 / (optics.wavelength_nm * math.sqrt(n * n - optics.NA ** 2))
    needed = 2 * math.pi * abs(z_max_nm) * slope * 2 / max_phase_step
    size = max(minimum, math.ceil(needed))
    return int(math.ceil(size / multiple) * multiple)
