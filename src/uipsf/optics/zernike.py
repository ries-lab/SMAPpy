"""Zernike polynomials on the pupil grid.

Noll's single index j (1 = piston, 2/3 = tilt x/y, 4 = defocus, 5/6 =
astigmatism sin 2φ / cos 2φ, …) and his normalisation: each polynomial has
unit RMS over the unit disc, so a coefficient is the RMS wavefront in
radians.  The radial polynomials and the ordering are uiPSF's (`genZern1`),
so coefficients compare one to one with uiPSF v1 (`docs/uipsf/conventions.md`).

Piston, tilt and defocus are left out of the phase by default: piston is
invisible, tilt is the emitter's lateral position and defocus its z (brief,
decision 5).  With ``high_na=True`` the remaining modes are also made
orthogonal, over the aperture, to the exact high-NA defocus k_z and to
tilt, so no mode can mimic a shift of the emitter (Booth, Neil & Wilson
1998): at high NA the paraxial Z4 is not the defocus the vectorial model
produces, and Z11 (spherical) mixes with depth.
"""
from __future__ import annotations

from math import factorial

import numpy as np
import torch

from .grid import PupilGrid

PISTON, TILT_X, TILT_Y, DEFOCUS = 1, 2, 3, 4
_NAMED = {"piston": (PISTON,), "tilt": (TILT_X, TILT_Y), "defocus": (DEFOCUS,)}


def noll_to_nm(j: int) -> tuple[int, int]:
    """Radial order n and signed azimuthal order m of Noll index j (m < 0: sin, m > 0: cos)."""
    if j < 1:
        raise ValueError("Noll indices start at 1")
    n = int(np.ceil((-3 + np.sqrt(1 + 8 * j)) / 2))
    m = j - n * (n + 1) // 2 - 1
    if n % 2 != m % 2:
        m += 1
    return n, (-m if j % 2 else m)


# Short names by |m|, so a bar chart of coefficients reads without a table: astigmatism, coma, trefoil, ...
_FAMILY = {0: "Sp", 1: "Co", 2: "As", 3: "Tr", 4: "Qu", 5: "Pe", 6: "He", 7: "Hp", 8: "Oc"}
#: what the abbreviations of `zernike_name` stand for, for a figure's legend
ZERNIKE_NAMES = ("Pi piston, T tilt, Df defocus, Sp spherical, Co coma, As astigmatism, Tr trefoil, "
                 "Qu quadrafoil, Pe pentafoil, He hexafoil, Hp heptafoil, Oc octafoil; x: cos, y: sin; "
                 "2, 3: the second, third order")


def zernike_name(j: int) -> str:
    """Noll index j's short name: "Df" defocus, "Asx" / "Asy" the astigmatisms (cos 2φ / sin 2φ), "Cox",
    "Sp", "2Asx" the second astigmatism (n = 4), "2Sp" the second spherical (n = 6) and so on (`ZERNIKE_NAMES`).

    The suffix is the azimuthal term, x for cos mφ and y for sin mφ: the axes of a tilt or a coma, and for
    astigmatism 0°/90° against ±45°.  The prefix counts the family's radial orders from its first."""
    n, m = noll_to_nm(j)
    a = abs(m)
    if n == 0:
        return "Pi"
    if n == 1:
        return "Tx" if m > 0 else "Ty"
    if n == 2 and m == 0:
        return "Df"
    family = _FAMILY.get(a, f"m{a}")
    first = {0: 4, 1: 3}.get(a, a)                   # the family's lowest radial order
    k = (n - first) // 2 + 1
    return f"{k if k > 1 else ''}{family}{'' if m == 0 else ('x' if m > 0 else 'y')}"


#: the aberrations a calibration is read by, (n, |m|) -> name: what an astigmatic 3D PSF is made of, and what
#: a misaligned or mismatched objective adds
KEY_ABERRATIONS = {(2, 2): "astigmatism", (3, 1): "coma 1", (5, 1): "coma 2", (4, 0): "spherical 1",
                   (6, 0): "spherical 2"}


def zernike_family(j: int) -> str | None:
    """Which of `KEY_ABERRATIONS` Noll index j belongs to, or None."""
    n, m = noll_to_nm(j)
    return KEY_ABERRATIONS.get((n, abs(m)))


def nm_to_noll(n: int, m: int) -> int:
    j = n * (n + 1) // 2 + 1 + max(0, abs(m) - 1)
    if (m > 0 and n % 4 >= 2) or (m < 0 and n % 4 <= 1):
        j += 1
    return j


def n_modes(n_max: int) -> int:
    """How many Noll modes have radial order up to n_max."""
    return (n_max + 1) * (n_max + 2) // 2


def zernike_modes(n_max: int, exclude=("piston", "tilt", "defocus")) -> list[int]:
    """Noll indices up to radial order n_max, without the named or numbered modes in ``exclude``."""
    drop = set()
    for e in exclude:
        drop.update(_NAMED[e] if isinstance(e, str) else (int(e),))
    return [j for j in range(1, n_modes(n_max) + 1) if j not in drop]



def symmetric_modes(n_max: int) -> list[int]:
    """The rotationally symmetric modes (m = 0) of radial order 2 … n_max: Noll 4, 11, 22 up to 6.  A
    pupil's magnitude in these is the apodisation and transmission of an objective, which have no
    direction; a 4Pi model's magnitude free in every mode found three optima on uiPSF v1's beads by
    where it started, in these one (docs/4pi.md §20)."""
    return [j for j in zernike_modes(n_max, exclude=("piston",)) if noll_to_nm(j)[1] == 0]

def zernike(j: int, rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Noll-normalised Zernike polynomial j at (rho, phi); not cut to the disc."""
    n, m = noll_to_nm(j)
    a = abs(m)
    g = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * n + 2)
    r = np.zeros_like(rho, dtype=float)
    for k in range((n - a) // 2 + 1):
        r += ((-1) ** k * factorial(n - k)
              / (factorial(k) * factorial((n + a) // 2 - k) * factorial((n - a) // 2 - k))) * rho ** (n - 2 * k)
    return g * r * (np.sin(a * phi) if m < 0 else np.cos(a * phi))


def zernike_basis(grid: PupilGrid, modes: list[int], high_na: bool = False) -> torch.Tensor:
    """The modes sampled on the grid, [K, P, P] real, in the grid's dtype and device."""
    rho = grid.rho.detach().cpu().double().numpy()
    phi = grid.phi.detach().cpu().double().numpy()
    Z = np.stack([zernike(j, rho, phi) for j in modes]) if modes else np.zeros((0,) + rho.shape)
    if high_na:
        kz = np.real(grid.kz_imm.detach().cpu().numpy()).astype(float)
        against = np.stack([np.ones_like(rho), grid.u.cpu().double().numpy(),
                            grid.v.cpu().double().numpy(), kz])
        Z = orthogonalise(Z, against, grid.aperture.cpu().double().numpy())
    return torch.as_tensor(Z, dtype=grid.dtype, device=grid.device)


def orthogonalise(Z: np.ndarray, against: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove from each mode its projection on ``against`` over the mask, and restore unit RMS.

    The projection is a least-squares fit on the discrete aperture, so the
    result is orthogonal on the samples the model actually uses.
    """
    inside = mask > 0
    A = against[:, inside].T                                 # [n_samples, n_against]
    out = np.zeros_like(Z)
    for i, z in enumerate(Z):
        coef, *_ = np.linalg.lstsq(A, z[inside], rcond=None)
        resid = z[inside] - A @ coef
        rms = np.sqrt(np.mean(resid ** 2))
        out[i][inside] = resid / rms if rms > 0 else 0
    return out
