"""One PSF model, built one way: a description of the choices in, a pupil and a PSF out.

Every workflow -- beads in one or two channels, in situ, 4Pi -- used to build its pupil and PSF itself,
and each kept its own copy of the decisions that come with them: the grid a little beyond the NA for the
minimal pupil (`GRID_MARGIN`), the pixel's blur under the learnt one, the supercritical ring's damping.
Those copies drifted (in situ handed a minimal model's pupil on as a flat magnitude; a spline and a movie
dropped the pixel's blur).  Here they are made once:

- **the spec** is SMAPpy's `PSFModelSettings` (`smappy.calibrate.pupil`, torch-free, the form's section) or
  anything with its attributes: ``order``, ``magnitude`` (minimal, free, symmetric, fixed), ``saf_damping``,
  ``blur`` (free, round, fixed), ``blur_nm``, ``pixel_blur`` and ``learn_n_med`` (the sample's index learnt);
- `pupil_grid` owns the grid (and its margin), `make_pupil` the pupil, `make_psf` the PSF;
- **the hand-over** between models is the description a learnt PSF gives of itself
  (`VectorPSF.describe`, plain data, kept with a microscope by SMAPpy): ``kind``, the phase in nm, γ (a
  minimal pupil) or the magnitude's Zernikes, σ and whether it is beyond the pixel.  `held_from` turns it into
  what another model holds (`Held`), `start_from` into where it starts.

A point emitter's view of a learnt PSF is `VectorPSF.for_emitters`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..optics import PupilGrid, choose_pupil_size
from ..optics.zernike import noll_to_nm, zernike_modes
from .psf import VectorPSF
from .pupil import MinimalPupil, ZernikePupil, minimal_grid_optics

#: γ held in situ as a last resort, when neither a start model nor the microscope gives one (SMAPpy's
#: `microscope_db.gamma_of`, which warns then): the middle of what M5's 100 nm beads learnt, 0.32–0.41 over two
#: colours, two media and with and without the 3D lens.  Not a property of objectives in general -- M2's beads
#: learnt 0.08 and 0.16, and on M2's January movie 0.36 moved the learnt astigmatism by 0.24 rad
#: (`docs/uipsf/architecture.md`, "In situ on the minimal model").
NOMINAL_GAMMA = 0.36


@dataclass
class Held:
    """What a model holds from another's description (`held_from`): phase modes in radians of this model's
    wavelength, a minimal pupil's γ and learnt NA, or Zernike magnitude modes; ``source`` says where they came
    from."""
    phase: dict[int, float] = field(default_factory=dict)
    gamma: float | None = None
    na: float | None = None
    magnitude: dict[int, float] = field(default_factory=dict)
    source: str = ""

    def __bool__(self) -> bool:
        return bool(self.phase or self.magnitude or self.gamma is not None)

    @property
    def kind(self) -> str:
        """The magnitude a model holding this needs: "minimal" for γ, else "fixed" (Zernike modes held)."""
        return "minimal" if self.gamma is not None else "fixed"


def magnitude_modes(n_max: int, kind: str) -> list[int]:
    """The Noll modes of a Zernike pupil's magnitude for ``kind``: no piston, which the PSF's energy
    normalisation fixes (`model.pupil`); none for a minimal or fixed magnitude."""
    every = zernike_modes(n_max, exclude=("piston",))
    if kind == "free":
        return every
    if kind == "symmetric":
        return [j for j in every if noll_to_nm(j)[1] == 0]
    if kind in ("fixed", "minimal"):
        return []                      # the minimal pupil's magnitude is its own (`MinimalPupil`)
    raise ValueError(f"no magnitude choice {kind!r}")


def pupil_grid(optics, spec, z_max_nm: float = 0.0, *, size: int | None = None, device=None,
               dtype=torch.float32) -> PupilGrid:
    """The pupil grid for ``spec``: a minimal pupil's reaches `GRID_MARGIN` beyond the objective's NA
    (`minimal_grid_optics`), a Zernike pupil's ends at it.  ``size`` samples, or as many as keep defocus up
    to ``z_max_nm`` from aliasing (`choose_pupil_size`, never below 64)."""
    g = minimal_grid_optics(optics) if spec.magnitude == "minimal" else optics
    return PupilGrid.build(g, size or choose_pupil_size(g, z_max_nm), device=device, dtype=dtype)


def make_pupil(optics, spec, held: Held | None = None, *, grid: PupilGrid | None = None,
               magnitude_override: list[int] | None = None, z_max_nm: float = 0.0, device=None,
               dtype=torch.float32):
    """The pupil ``spec`` describes for the objective ``optics`` (its NA the aperture), on ``grid`` or a new
    `pupil_grid`.  ``held`` holds phase modes and, by its kind, γ (and a learnt NA) or magnitude modes; a spec
    whose magnitude cannot hold them is refused, rather than the held part silently dropped.
    ``magnitude_override`` replaces a Zernike magnitude's modes (4Pi's typed ones)."""
    held = held or Held()
    grid = grid if grid is not None else pupil_grid(optics, spec, z_max_nm, device=device, dtype=dtype)
    n_max = int(spec.order)
    if spec.magnitude == "minimal":
        if held.magnitude:
            raise ValueError("a minimal pupil cannot hold Zernike magnitude modes: hold a minimal model's γ")
        return MinimalPupil(grid, n_max=n_max, na=optics.NA if held.na is None else held.na,
                            held_phase=held.phase, saf_damping=spec.saf_damping, held_gamma=held.gamma,
                            learn_na=bool(getattr(spec, "learn_na", False)) and held.na is None)
    if held.gamma is not None:
        raise ValueError("a Zernike pupil cannot hold a minimal model's γ: choose the minimal magnitude")
    modes = magnitude_override if magnitude_override is not None else magnitude_modes(n_max, spec.magnitude)
    return ZernikePupil(grid, n_max=n_max, magnitude_modes=modes, held_phase=held.phase,
                        held_magnitude=held.magnitude)


def make_psf(optics, spec, roi_shape, *, held: Held | None = None, bead_radius_nm: float = 0.0,
             upsample: int = 1, kind: str = "vector", analyser: str | None = "unpolarised",
             grid: PupilGrid | None = None, pupil=None, magnitude_override: list[int] | None = None,
             z_max_nm: float = 0.0, device=None, dtype=torch.float32) -> VectorPSF:
    """The PSF ``spec`` describes: `make_pupil`'s pupil (or ``pupil``), the blur learnt round, free or held at
    ``spec.blur_nm``, the pixel's own blur under it with ``spec.pixel_blur``, the sample's index learnt from the
    optics' with ``spec.learn_n_med``."""
    if pupil is None:
        pupil = make_pupil(optics, spec, held, grid=grid, magnitude_override=magnitude_override, z_max_nm=z_max_nm,
                           device=device, dtype=dtype)
    return VectorPSF(pupil.grid, pupil, tuple(roi_shape), upsample=upsample, kind=kind, analyser=analyser,
                     bead_radius_nm=bead_radius_nm, sigma_nm=(spec.blur_nm, spec.blur_nm),
                     fit_sigma=spec.blur != "fixed", round_sigma=spec.blur == "round", pixel_blur=spec.pixel_blur,
                     learn_n_med=bool(getattr(spec, "learn_n_med", False)))


def held_from(description: dict | None, order: int, wavelength_nm: float, every: bool = False,
              source: str = "") -> Held:
    """What a model holds from ``description`` (`VectorPSF.describe`): the phase above ``order`` and the
    magnitude -- a minimal pupil's γ, or a Zernike pupil's rotationally symmetric modes.  ``every``: the whole
    pupil, every phase and magnitude mode.  The phase goes through nm of wavefront, so a pupil learnt at
    another emission holds at ``wavelength_nm``.  Empty without a description.

    Why these (`docs/uipsf/architecture.md`, "In situ on matched M2 pairs"): the magnitude is the objective's
    transmission, the same over days and colours, and learnt from molecules it trades with the focus; the
    phase above order 6 gains little over the beads' and scatters between molecule sets."""
    if not description:
        return Held()
    phase = {int(j): 2 * math.pi * float(nm) / float(wavelength_nm)
             for j, nm in (description.get("phase_nm") or {}).items() if every or noll_to_nm(int(j))[0] > order}
    if description.get("kind") == "minimal":
        na = description.get("na")
        return Held(phase, gamma=float(description["gamma"]), na=None if na is None else float(na), source=source)
    magnitude = {int(j): float(v) for j, v in (description.get("magnitude") or {}).items()
                 if int(j) > 1 and (every or noll_to_nm(int(j))[1] == 0)}
    return Held(phase, magnitude=magnitude, source=source)


def start_from(pupil, description: dict | None, wavelength_nm: float) -> dict:
    """Start values of ``pupil``'s learnt parameters from ``description``: every phase mode it learns from
    the description's nm, γ where it learns one, the magnitude's Zernikes where both have them."""
    dtype = pupil.grid.rho.real.dtype
    out = {"pupil.phase": torch.zeros(len(pupil.phase_modes), dtype=dtype)}
    if not description:
        return out
    phase_nm = description.get("phase_nm") or {}
    for i, j in enumerate(pupil.phase_modes):
        if str(j) in phase_nm:
            out["pupil.phase"][i] = 2 * math.pi * float(phase_nm[str(j)]) / float(wavelength_nm)
    if isinstance(pupil, MinimalPupil):
        if pupil.held_gamma is None and description.get("gamma") is not None:
            out["pupil.gamma"] = torch.tensor(float(description["gamma"]), dtype=dtype)
        if pupil.learn_na and description.get("na") is not None:
            out["pupil.na"] = torch.tensor(float(description["na"]), dtype=dtype)
    elif pupil.magnitude_modes and description.get("magnitude"):
        out["pupil.magnitude"] = torch.tensor([float(description["magnitude"].get(str(j), 0.0))
                                               for j in pupil.magnitude_modes], dtype=dtype)
    return out
