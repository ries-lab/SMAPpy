"""Learning a pupil from single molecules in the sample: the fit, on regions already cut (plan §9, M5).

This is M5's core, before detection: given each emitter's region (one frame, or the K frames of a linked
blink), learn the pupil, every emitter's position and the focus, with v2's fit, and say whether the data
agree with the sign of the aberrations the start gave.  Detection, selection and the rounds over the
whole movie come on top of it.

**Where the emitters start.**  In a bead stack every bead passes through focus, so its brightest slice
places it (`model.start`); an emitter seen once has no such slice.  It is localised on the starting
pupil from several heights instead, the best of them kept (`localise_from_heights`): with an
astigmatic start each emitter's shape says on which side of the focus it is, and the fit takes it
from there.  The heights are spread about the height in focus, f·n_med/n_imm for a focus f (the
paraxial focal shift, v1's `estzoffset`).

**The fit is the beads'** (`learn_beads`, with ``start_stage1``): low orders first with the starting
pupil, rejection by the beads' rules, then every mode.  The shared parameters are the pupil, the blur
and the focus, which has a loose prior (`model.InSituGeometry`).

**The twin test.**  For the even aberrations (astigmatism, spherical) a pupil and its twin,
A'(k) = A*(−k), make the same images with z mirrored: for a scalar field with matched indices
|E'(r, −z)|² = |E(r, z)|² exactly.  In Zernikes the twin is c_j → −(−1)^m c_j in the phase and
c_j → (−1)^m c_j in the magnitude, so astigmatism and spherical flip and coma does not.  In situ nothing
outside the data sets the sign, except what breaks the symmetry: the index mismatch (k_z,med against
k_z,imm, growing with depth), supercritical light, and the coverslip.  So after the fit the twin of the
learnt pupil is fitted too, every emitter localised on it again, and the difference of the two optima
says which sign the data prefer.  The sign the start gave is kept: explicit beats the data, and a
clear preference for the twin is a warning, never a silent flip (decided October 2026).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from ..device import on_model_device
from ..model.channels import prefix
from ..model.pupil import ZernikePupil
from ..model.start import rough_start
from ..noise import ShiftedPoisson
from ..optics.zernike import noll_to_nm
from ..params import Params
from ..solve import GaussNewtonLBFGS, Problem
from .beads import Calibration, learn_beads
from .rejection import RejectionSettings
from .scoring import localise, shared_values

# The starting heights: every 150 nm over ±900 nm about the height in focus.  An astigmatic PSF changes
# its shape over ~300 nm; a start within half of that of the truth converged in the tests.
HEIGHT_SPAN_NM, HEIGHT_STEP_NM = 900.0, 150.0
# At most this many emitters are computed at once.  A localisation needs 6–9 MB per emitter on Apple's GPU (a
# pupil of 64 samples, 19-pixel regions): the whole pool of a dense movie, 5000 emitters in one chunk, took
# 38 GB and swapped for twenty minutes.  256 is ~2 GB.  Bead stacks keep `Problem`'s own chunks.
INSITU_CHUNK = 256
# A difference of optima above this many nats is a clear preference for one sign; below it the data do not
# decide.  Both fits have the same parameters, so it is a log likelihood ratio, and 10 is strong evidence.
TWIN_DECISIVE_NATS = 10.0


def in_focus_height(model, focus_nm: float) -> float:
    """The height above the coverslip that a focus ``focus_nm`` images sharply, paraxially: f·n_med/n_imm."""
    optics = model.psf.grid.optics
    return float(focus_nm) * optics.n_med / optics.n_imm


def start_heights(model, focus_nm: float) -> np.ndarray:
    """The heights `localise_from_heights` tries: about the height in focus, never below the coverslip."""
    h = in_focus_height(model, focus_nm) + np.arange(-HEIGHT_SPAN_NM, HEIGHT_SPAN_NM + 1, HEIGHT_STEP_NM)
    return np.unique(np.maximum(h, 0.0))


def localise_from_heights(model, shared: dict, batch, noise, heights) -> tuple[dict, torch.Tensor]:
    """Every emitter localised on the fixed ``shared`` parameters from each height in ``heights``, the best
    kept: (its start, `Batch.start`'s keys, and its objective [E]).  A linked blink (K frames) is one
    emitter: one position from all its frames, its photons per frame with `PerObservation` ([E, K])."""
    heights = np.asarray(heights, float)
    return localise_from_starts(model, shared, batch, noise, np.broadcast_to(heights, (batch.n_emitters,
                                                                                       len(heights))))


def localise_from_starts(model, shared: dict, batch, noise, starts) -> tuple[dict, torch.Tensor]:
    """`localise_from_heights` with each emitter's own starting heights, ``starts`` [E, S] (NaN: none)."""
    batch = on_model_device(batch, model)
    rough = rough_start(batch, model.pixel_size_nm)
    starts = np.asarray(starts, float)
    best, found = None, None
    for k in range(starts.shape[1]):
        column = starts[:, k]
        if not np.isfinite(column).any():
            continue
        xyz = rough["xyz"].clone()
        xyz[:, 2] = torch.as_tensor(np.where(np.isfinite(column), column, np.nanmean(column)), dtype=xyz.dtype,
                                    device=xyz.device)
        # Through the batch's start, so that every intensity model starts its own parameters from it.
        fit, problem = localise(model, shared, replace(batch, start=dict(rough, xyz=xyz)), noise,
                                chunk_size=INSITU_CHUNK)
        obj = problem.objectives(fit.params)
        obj = torch.where(torch.as_tensor(np.isfinite(column)), obj, torch.full_like(obj, float("inf")))
        phys = fit.params.physical()
        photons = phys["emitter.photons"] if "emitter.photons" in phys else phys["emitter.photons_per_slice"]
        here = {"xyz": phys["emitter.xyz"].detach(), "photons": photons.detach(),
                "background": phys["emitter.background"].detach()}
        if best is None:
            best, found = obj, here
            continue
        better = obj < best
        best = torch.where(better, obj, best)
        on = better.to(here["xyz"].device)
        found = {k_: torch.where(on.reshape(-1, *([1] * (v.dim() - 1))), here[k_], v) for k_, v in found.items()}
    return found, best


#: `width_starts`: the model's widths every this many nm, how many best matches each emitter starts from, and
#: how far apart two of them must be
WIDTH_STEP_NM, WIDTH_STARTS, WIDTH_SEPARATION_NM = 25.0, 2, 250.0


def width_curve(model, shared: dict, heights, roi: int, photons: float = 3000.0, background: float = 10.0):
    """(σx², σy²) [H] in px², of SMAPpy's elliptical Gaussian fitted to the model's own spot at ``heights``
    (noise-free, ``photons`` on ``background`` per pixel): what an emitter's widths say of its height on
    this model."""
    from smappy.calibrate.insitu import gaussian

    from ..export import sample_heights
    spots = sample_heights(model, shared, heights, roi).double().numpy() * photons + background
    g = gaussian(spots.astype(np.float32))
    return g["sigma_x_pix"] ** 2, g["sigma_y_pix"] ** 2


def width_starts(model, shared: dict, widths, roi: int, span_nm: float | None = None) -> np.ndarray:
    """Each emitter's starting heights from its Gaussian widths ``widths`` [E, 2] (σx², σy², px², as
    detection measured them): the `WIDTH_STARTS` heights at least `WIDTH_SEPARATION_NM` apart whose
    widths on the model are nearest, and the height in focus; [E, WIDTH_STARTS + 1].

    Localising from ten to twenty heights each was most of a run's time; the widths say where an emitter
    is to within the model's error, and an astigmatic spot's two widths tell above from below focus.  The
    second match catches what the model gets wrong while it is still rough, the height in focus a spot
    the Gaussian misread."""
    focus = float(shared["geometry.focus"])
    centre = in_focus_height(model, focus)
    span = HEIGHT_SPAN_NM if span_nm is None else float(span_nm)
    grid = np.arange(max(0.0, centre - span), centre + span + 1, WIDTH_STEP_NM)
    cx, cy = width_curve(model, shared, grid, roi)
    w = np.asarray(widths, float)
    ok = np.isfinite(cx) & np.isfinite(cy)
    grid, cx, cy = grid[ok], cx[ok], cy[ok]
    # relative distance in each width: a width is measured to a fraction of itself
    d = ((w[:, None, 0] - cx[None]) / cx[None]) ** 2 + ((w[:, None, 1] - cy[None]) / cy[None]) ** 2
    out = np.full((len(w), WIDTH_STARTS + 1), np.nan)
    out[:, -1] = centre
    for e in range(len(w)):
        if not np.isfinite(d[e]).any():
            continue
        order = np.argsort(d[e])
        taken = []
        for i in order:
            if all(abs(grid[i] - grid[j]) >= WIDTH_SEPARATION_NM for j in taken):
                taken.append(i)
            if len(taken) == WIDTH_STARTS:
                break
        out[e, :len(taken)] = grid[taken]
    return out


def twin(model, shared: dict) -> dict:
    """The twin of the pupils in ``shared``, A'(k) = A*(−k) (module docstring): phase c_j → −(−1)^m c_j,
    magnitude c_j → (−1)^m c_j; everything else unchanged."""
    out = dict(shared)
    for c, ch in enumerate(model.channels):
        pupil = ch.psf.pupil
        if not isinstance(pupil, ZernikePupil):
            raise ValueError("the twin test needs a Zernike pupil")
        pre = prefix(c)
        for which, modes, sign in (("phase", pupil.phase_modes, -1.0), ("magnitude", pupil.magnitude_modes, 1.0)):
            name = pre + "pupil." + which
            if name in shared:
                v = torch.as_tensor(shared[name])
                parity = v.new_tensor([(-1.0) ** abs(noll_to_nm(j)[1]) for j in modes])
                out[name] = sign * parity * v
    return out


@dataclass
class TwinTest:
    """The learnt pupil against its twin, each fitted to the same emitters."""
    objective: float               # the learnt fit's optimum (½ deviance + priors)
    twin_objective: float          # the twin's
    twin: object                   # the twin's FitResult

    @property
    def delta_nats(self) -> float:
        """How much worse the twin fits: positive when the data agree with the learnt sign."""
        return self.twin_objective - self.objective

    @property
    def verdict(self) -> str:
        """"agree", "prefer twin" (the data clearly want the other sign) or "undecided"."""
        if self.delta_nats >= TWIN_DECISIVE_NATS:
            return "agree"
        if self.delta_nats <= -TWIN_DECISIVE_NATS:
            return "prefer twin"
        return "undecided"


@dataclass
class InSituCalibration:
    calibration: Calibration       # the fit, the emitters' table (`BeadTable`, one row per emitter)
    start_objectives: torch.Tensor  # [E] each emitter's optimum on the starting pupil
    twin: TwinTest | None = None

    @property
    def warnings(self) -> list[str]:
        if self.twin is not None and self.twin.verdict == "prefer twin":
            return [f"the data prefer the mirrored pupil by {-self.twin.delta_nats:.0f} nats: the sign of "
                    "astigmatism and spherical aberration may be the other one"]
        return []


def learn_insitu(model, batch, start: dict | None = None, noise=None, heights=None,
                 rejection: RejectionSettings | None = None, first_order: int | None = 4, solver=None,
                 chunk_size: int | None = None, twin_test: bool = True, progress=None,
                 cancel=None, located: dict | None = None) -> InSituCalibration:
    """Learn a pupil from emitters in the sample (module docstring).

    ``model`` has an `InSituGeometry`; ``batch`` holds each emitter's region, [E, 1, K, Y, X].  ``start``,
    physical shared values (a bead model of the microscope, `workflows.calibrate.start_values`, or
    Zernikes typed in), sets the pupil the emitters are first localised on and stage 1 starts from: it is
    what gives the even aberrations their sign.  ``heights`` defaults to `start_heights`.  ``located``,
    each emitter's start already localised on ``start`` (`localise_from_heights`'s), skips that step: the
    rounds have it (`insitu_rounds`).
    """
    noise = noise if noise is not None else ShiftedPoisson()
    solver = solver if solver is not None else GaussNewtonLBFGS()
    # Masked pixels are a neighbour's, masked on purpose and capped when the units were chosen; the beads'
    # rule (1 %, for saturation) rejected every masked unit of a dense movie
    rejection = rejection if rejection is not None else RejectionSettings(max_masked=1.0)
    batch = on_model_device(batch, model)
    dtype, device = batch.data.dtype, batch.data.device
    base = Params(model.specs(replace(batch, start=_placeholder(batch))), batch.n_emitters, dtype=dtype,
                  device=device)
    shared = shared_values(base)
    for n, v in (start or {}).items():
        if n in shared and tuple(torch.as_tensor(v).shape) == tuple(shared[n].shape):
            shared[n] = torch.as_tensor(v, dtype=dtype, device=device)
    if heights is None:
        heights = start_heights(model, float(shared["geometry.focus"]))
    if located is None:
        found, objectives = localise_from_heights(model, shared, batch, noise, heights)
    else:
        found, objectives = located, torch.full((batch.n_emitters,), float("nan"), dtype=torch.float64)
    located = replace(batch, start=found)
    cal = learn_beads(model, located, noise, rejection=rejection, first_order=first_order, solver=solver,
                      chunk_size=chunk_size or INSITU_CHUNK, progress=progress, cancel=cancel, start=shared,
                      start_stage1=True)
    result = InSituCalibration(cal, objectives)
    if twin_test and cal.result.stop_reason != "cancelled":
        result.twin = twin_fit(model, cal, located, noise, heights, solver, chunk_size or INSITU_CHUNK)
    return result


def twin_fit(model, cal: Calibration, batch, noise, heights, solver=None, chunk_size=INSITU_CHUNK,
             starts=None) -> TwinTest:
    """The twin of ``cal``'s pupil fitted to its used emitters, each localised on the twin first.

    Converged to the solver's own tolerance, not stage 1's looser one: stopped early, the twin's optimum is
    too high, and a start of the wrong sign then passed as agreeing with the data.  ``starts`` [E, S], per
    emitter of ``batch``, replaces the common ``heights`` (`width_starts` on the twin)."""
    solver = solver if solver is not None else GaussNewtonLBFGS()
    sub = on_model_device(batch, model).subset(torch.tensor(cal.beads.used))
    mirrored = twin(model, cal.shared())
    if starts is not None:          # per emitter of ``batch``, [E, S]
        found, _ = localise_from_starts(model, mirrored, sub, noise, np.asarray(starts)[np.asarray(cal.beads.used)])
    else:
        found, _ = localise_from_heights(model, mirrored, sub, noise, heights)
    sub = replace(sub, start=found)
    p = Params(model.specs(sub), sub.n_emitters, dtype=sub.data.dtype, device=sub.data.device, values=mirrored)
    fit = solver.solve(Problem(model, noise, sub, chunk_size), p)
    return TwinTest(cal.result.objective, fit.objective, fit)


def _placeholder(batch) -> dict:
    """A start of the right shapes for building the specs before any emitter is localised."""
    E = batch.n_emitters
    z = batch.data.new_zeros(E)
    return {"xyz": torch.stack([z, z, z], -1), "photons": z + 1, "background": z + 1}
