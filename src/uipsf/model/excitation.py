"""The excitation beam: a laser focused by the objective into the sample (S2 of docs/uipsf/scanned-psfs.md).

A confocal, STED or MINFLUX microscope images by moving a focused beam, so its PSF is the product of an
excitation and a detection; this is the excitation's focal field, E(r) for r near the focus.

**Reciprocity.**  A beam with Jones vector (e_x, e_y) at the back aperture has the angular spectrum
A(k)·[e_x h_x(k) + e_y h_y(k)] in the sample, where h_x, h_y are the dipole fields the detection model already
builds (`optics.vectorial`): ``h_x``'s three rows, the x analyser's response to the x, y and z dipole, read the
other way are the (E_x, E_y, E_z) focal field of x-polarised input (scanned-psfs.md §3).  With the detection's
apodisation √cos θ_imm / cos θ_med and its Fresnel factors (sample → immersion) this is exact up to one constant:
the factors for light going in, immersion → sample, are the detection's times n_imm cos θ_imm / (n_med cos θ_med)
at every interface, and the focusing's aplanatic factor is 1/√cos θ_imm, so the product is n_imm / n_med times the
detection's -- which the normalisation takes up.  The field is E(r) = Σ_k a(k) J(k) exp(2πi k·r) Δk², the beam
travelling towards +z, the emitter's height above the focus (`docs/uipsf/conventions.md`).

**What sits in the pupil.**
- *the objective* -- the pupil's magnitude and phase (`ZernikePupil`, `MinimalPupil`), the same components as
  detection's.  Its magnitude is rescaled to keep a flat beam's power through the aperture, as detection's is: a
  Zernike magnitude's overall scale would otherwise trade with the photons;
- *a phase mask* (`model.mask`: the vortex, top-hat and half-moon of the donuts).  Not rescaled: what a mask's
  steps scatter out of the aperture is lost from the focus, as it is in the microscope;
- *the input beam* -- a Gaussian of 1/e field radius ``fill`` (in units of the aperture's radius) centred at
  ``centre``, normalised to unit total power so that a beam overfilling the aperture loses what falls outside;
  ``fill=None`` is a flat beam over the aperture;
- *the polarisation* -- (ψ, χ), the orientation and the ellipticity of the polarisation ellipse, J =
  R(ψ)·(cos χ, i sin χ): χ = 0 linear along ψ, χ = ±π/4 circular.  Or radial or azimuthal, J(φ).

Each beam parameter is held at its spec's value unless named in ``free``, as a mask's placement is: learnt, it is
a diagnostic (a decentred beam, an imperfect λ/4).

**Normalisation.**  I = |E|² / (Δk² Σ_aperture |h_x|²): a flat, x-polarised beam through an aberration-free
objective integrates to one over the focal plane (Parseval; supercritical light, here evanescent in a sample of
lower index, is not counted).  A Gaussian then integrates to the fraction of its power that passes the aperture,
and beams that share a laser keep their relative heights: a donut's peak against a Gaussian's is physical, which
is what lets MINFLUX's patterns share one brightness (scanned-psfs.md §3, "Normalisation").

**The intensity an emitter sees.**  A freely rotating dye absorbs |E_x|² + |E_y|² + |E_z|²; a fixed dipole |μ·E|²;
a fluorescent or gold bead behind an open detector Σ_d w_d |E_d|² (``weights``).  The weights' derivation from the
detection model, the pinhole and the scan are S3-S5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..optics.grid import PupilGrid
from ..optics.mdft import MatrixDFT
from ..optics.vectorial import dipole_fields
from ..params import Bounded, Identity, ParamSpec

#: the polarisations there are, by the name a spec gives: the uniform ones set (ψ, χ), the others are maps
POLARISATIONS = {"x": (0.0, 0.0), "y": (math.pi / 2, 0.0), "linear": None, "circular": (0.0, math.pi / 4),
                 "elliptical": None, "radial": None, "azimuthal": None}
#: the beam's parameters that can be learnt, by the name ``free`` gives
FREEABLE = ("angle", "ellipticity", "fill", "centre")


@dataclass
class BeamSpec:
    """The input beam, plain data: its polarisation, its fill of the aperture and which parameters are learnt.

    ``polarisation`` one of `POLARISATIONS`: "x", "y" and "circular" set (ψ, χ); "linear" and "elliptical" take
    ``angle`` (ψ) and ``ellipticity`` (χ) as given; ``handedness`` ±1 flips a circular one.  ``fill`` the
    Gaussian's 1/e field radius in units of the aperture radius (SimuFLUX's 7 mm waist on a 6.5 mm aperture is
    2.15), None for a flat beam; ``centre`` its centre on the pupil, in the same units."""
    polarisation: str = "circular"
    handedness: int = 1
    angle: float = 0.0
    ellipticity: float = 0.0
    fill: float | None = 7.0 / 3.25
    centre: tuple[float, float] = (0.0, 0.0)
    free: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.polarisation not in POLARISATIONS:
            raise ValueError(f"polarisation must be one of {sorted(POLARISATIONS)}, got {self.polarisation!r}")
        if self.handedness not in (1, -1):
            raise ValueError("handedness must be 1 or -1")
        unknown = set(self.free) - set(FREEABLE)
        if unknown:
            raise ValueError(f"a beam can learn {FREEABLE}, not {sorted(unknown)}")
        if self.polarisation in ("radial", "azimuthal") and {"angle", "ellipticity"} & set(self.free):
            raise ValueError(f"a {self.polarisation} beam has no uniform polarisation to learn")
        if self.fill is None and {"fill", "centre"} & set(self.free):
            raise ValueError("a flat beam has no fill or centre to learn: give it a fill")

    def ellipse(self) -> tuple[float, float]:
        """(ψ, χ), the polarisation ellipse this spec starts at."""
        fixed = POLARISATIONS[self.polarisation]
        psi, chi = fixed if fixed is not None else (self.angle, self.ellipticity)
        return psi, chi * self.handedness


class ExcitationBeam:
    def __init__(self, grid: PupilGrid, pupil, beam: BeamSpec | None = None):
        """``grid`` at the excitation's wavelength; ``pupil`` the objective's (`ZernikePupil`, `MinimalPupil`,
        with a ``mask`` for a donut); ``beam`` the input beam (`BeamSpec`, circular and SimuFLUX's fill by
        default)."""
        self.grid, self.pupil = grid, pupil
        self.beam = beam if beam is not None else BeamSpec()
        apod = grid.aperture * grid.apodisation
        self.hx = apod * dipole_fields(grid, "x")                    # [3, P, P]: (E_x, E_y, E_z) of x input
        self.hy = apod * dipole_fields(grid, "y")
        weight = (self.hx.abs() ** 2 + self.hy.abs() ** 2).sum(0)    # a pupil sample's power, either input
        self.weight = weight / weight.sum()
        self.reference = float((self.hx.abs() ** 2).sum())            # Σ_aperture |h_x|², the normalisation
        if self.beam.polarisation in ("radial", "azimuthal"):
            c, s = torch.cos(grid.phi), torch.sin(grid.phi)
            j = torch.stack([c, s]) if self.beam.polarisation == "radial" else torch.stack([-s, c])
            self.jones_map = j.to(self.hx.dtype)
        else:
            self.jones_map = None

    def specs(self) -> list[ParamSpec]:
        b = self.beam
        out = list(self.pupil.specs())
        if self.jones_map is None:
            psi, chi = b.ellipse()
            out += [ParamSpec("beam.angle", (), init=psi, transform=Identity(), fixed="angle" not in b.free),
                    # χ beyond ±π/4 is the other handedness at ψ + π/2: bounded a little past it, so that a
                    # circular start is not on the bound
                    ParamSpec("beam.ellipticity", (), init=chi, transform=Bounded(-1.0, 1.0),
                              fixed="ellipticity" not in b.free)]
        if b.fill is not None:
            out += [ParamSpec("beam.fill", (), init=float(b.fill), transform=Bounded(0.2, 20.0),
                              fixed="fill" not in b.free),
                    ParamSpec("beam.centre", (2,), init=torch.tensor(b.centre, dtype=torch.float64),
                              transform=Bounded(-0.5, 0.5), fixed="centre" not in b.free)]
        return out

    def jones(self, p) -> torch.Tensor:
        """[2, P, P] complex (or [2, 1, 1] for a uniform polarisation): the input's (e_x, e_y)."""
        if self.jones_map is not None:
            return self.jones_map
        psi, chi = p["beam.angle"], p["beam.ellipticity"]
        a, b = torch.cos(chi).to(self.hx.dtype), 1j * torch.sin(chi).to(self.hx.dtype)
        c, s = torch.cos(psi).to(self.hx.dtype), torch.sin(psi).to(self.hx.dtype)
        return torch.stack([c * a - s * b, s * a + c * b])[:, None, None]

    def amplitude(self, p) -> torch.Tensor:
        """[P, P] real: the input beam's field on the pupil, of unit total power in units where the aperture's
        area is π (a flat beam: 1 everywhere)."""
        if self.beam.fill is None:
            return torch.ones_like(self.grid.u)
        w, c = p["beam.fill"], p["beam.centre"]
        r2 = (self.grid.u - c[0]) ** 2 + (self.grid.v - c[1]) ** 2
        # ∫ exp(−2r²/w²) d²r = π w²/2 over the whole plane, so √2/w makes its power π, a flat aperture's
        return math.sqrt(2) / w * torch.exp(-r2 / w ** 2)

    def objective(self, p) -> torch.Tensor:
        """[P, P] complex: the objective's pupil (magnitude rescaled to keep a flat beam's power) and its mask."""
        magnitude, phase = self.pupil(p)
        scale = torch.sqrt(1.0 / (self.weight * magnitude ** 2).sum((-2, -1), keepdim=True))
        A = torch.polar(torch.ones_like(phase), phase) * (magnitude * scale)
        t = self.pupil.transmission(p) if hasattr(self.pupil, "transmission") else None
        return A if t is None else A * t

    def spectrum(self, p) -> torch.Tensor:
        """[3, P, P] complex: the angular spectrum a(k)·J(k) of (E_x, E_y, E_z), before the defocus."""
        a = self.objective(p) * self.amplitude(p)
        J = self.jones(p)
        return a * (J[0] * self.hx + J[1] * self.hy)

    def field(self, p, shape: tuple[int, int], spacing_nm: tuple[float, float], z_nm, depth_nm=None,
              offset_nm=None) -> torch.Tensor:
        """[K, 3, Y, X] complex: (E_x, E_y, E_z) on a grid of ``shape`` (Y, X) samples ``spacing_nm`` (x, y)
        apart, centred on the beam's axis, at heights ``z_nm`` [K] above the focus -- the objective moved, z
        entering through the immersion's k_z as a bead stack's does -- and, with ``depth_nm`` [K], at that depth
        in the sample through the sample's k_z.  ``offset_nm`` (x, y) moves the beam.  In units whose square is
        `intensity`'s."""
        grid = self.grid
        z = torch.as_tensor(z_nm, dtype=grid.dtype, device=grid.device).reshape(-1, 1, 1, 1)
        wave = torch.exp(2j * math.pi * grid.kz_imm * z.to(grid.kz_imm.dtype))
        if depth_nm is not None:
            d = torch.as_tensor(depth_nm, dtype=grid.dtype, device=grid.device).reshape(-1, 1, 1, 1)
            wave = wave * torch.exp(2j * math.pi * grid.kz_med * d.to(grid.kz_med.dtype))
        dft = MatrixDFT(grid, shape, spacing_nm)
        x0 = y0 = None
        if offset_nm is not None:
            x0, y0 = (torch.as_tensor(float(v), dtype=grid.dtype, device=grid.device) for v in offset_nm)
        E = dft(self.spectrum(p) * wave, x0, y0)
        return E / math.sqrt(self.reference * grid.dk ** 2)

    def intensity(self, p, shape, spacing_nm, z_nm, depth_nm=None, offset_nm=None, *, dipole=None,
                  weights=None) -> torch.Tensor:
        """[K, Y, X]: the excitation an emitter absorbs, per nm² of a flat beam's unit power (module docstring).
        A freely rotating dye by default; ``dipole`` (3,) a fixed one, |μ·E|²; ``weights`` (3,) a bead's
        Σ_d w_d |E_d|²."""
        E = self.field(p, shape, spacing_nm, z_nm, depth_nm, offset_nm)
        if dipole is not None:
            mu = torch.as_tensor(dipole, dtype=E.dtype, device=E.device).reshape(1, 3, 1, 1)
            return ((mu * E).sum(1)).abs() ** 2
        I = E.real ** 2 + E.imag ** 2
        if weights is not None:
            I = I * torch.as_tensor(weights, dtype=I.dtype, device=I.device).reshape(1, 3, 1, 1)
        return I.sum(1)
