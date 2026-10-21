"""In situ learning in rounds: localise every molecule on the model, choose again, learn again (plan §9, M5).

The first round chooses by a Gaussian's width (`smappy.calibrate.insitu`), which knows nothing of the PSF:
its z proxy is approximate and a pair can pass its shape test.  Once there is a model, every unit the
detection kept is localised on it, and the next round chooses on what the model says: its height for
the z bins, its photons for the brightness window, its deviance for the shape (a pair fits badly; judged
one-sided, only a fit worse than the bin's is out).  It then learns again from the previous model, the
emitters started where the localisation put them.  uiPSF v1 repeated too (``repeat``), but chose again
only among what its first round had kept; here the whole pool is chosen from every round.

**Which molecules the rounds look at.**  Not the whole pool: at most a few times *per bin* from each bin of
the Gaussian's z proxy (`_capped_pool`), the same ones every round.

**Where each molecule starts.**  Localising every molecule from ten to twenty heights was most of a run's
time.  Each starts instead where the model's own widths match its Gaussian's (`insitu.width_starts`: the
two best matches and the height in focus) and, from the second round, where it was found last.

**Stopping.**  When the Zernikes and the focus moved since the last round by no more than their noise,
rms over them below ``stop_sd`` (1) standard errors of the change, and the edge sits at 0 within its
error; or after ``max_rounds``.  The twin test runs once, on the last round.

**The coverslip edge** (``coverslip``, on by default: the sample touches the coverslip).  The focus and
the heights are nearly degenerate: shifting every height by δh and the focus by δh·n_imm/n_med changes
the images only through the index mismatch (`docs/uipsf/architecture.md`, "M5, step 1"), so the focus is
known only to ~30–110 nm from the fit.  But no molecule is below the coverslip, and the lowest are on it:
the localised heights have a lower edge, which must be at 0.  `coverslip_edge` fits it by maximum
likelihood on the lowest ``edge_window`` (30 %) of the heights -- a spike at the edge (molecules on the
coverslip) and a flat density above it (molecules in the sample), shifted by s and blurred by each height's
CRLB -- so it works with a pile-up on the coverslip or without one.  The edge's shift s then becomes the
next round's prior on the focus, N(f − s·n_imm/n_med, σ_s·n_imm/n_med), and every height starts s lower:
the coverslip pins the common offset the fit alone cannot.  The factor, n_imm/n_med paraxially, is far
off at high NA, so it is measured: the lowest molecules localised again with the focus moved
(`_measured_ratio`), and from the second correction on a secant step on the rounds' (focus, edge)
pairs (`_focus_per_height`).  This automates v1's ``backgroundROI``, a region the user marked as
lying on the coverslip.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from ..model import InSituGeometry
from ..noise import ShiftedPoisson
from .insitu import (
    INSITU_CHUNK,
    InSituCalibration,
    in_focus_height,
    learn_insitu,
    localise_from_heights,
    localise_from_starts,
    start_heights,
    twin,
    twin_fit,
    width_starts,
)
from .scoring import localise, score_beads

# The edge's outliers are at most 5 %, and a spike must gain 6 nats.  With 20 % and Akaike's 2 the fit
# explained flat heights by a spike 80 nm above their edge and everything below it as outliers, once in
# five simulated sets; a true pile-up of a fifth of the molecules gains hundreds of nats.
MAX_OUTLIERS, SPIKE_NATS = 0.05, 6.0


@dataclass
class RoundSettings:
    max_rounds: int = 4
    coverslip: bool = True             # the sample touches the coverslip: its lowest molecules are on it
    edge_window: float = 0.3           # the fraction of lowest heights the edge is fitted on
    edge_max_sd_nm: float = 150.0      # an edge less certain than this sets no prior
    focus_floor_nm: float = 5.0        # the focus prior is never tighter than this
    stop_sd: float = 1.0               # stop when the rms change is below this many standard errors
    heights_step_nm: float = 200.0
    max_xy_offset_px: float = 3.0      # a localisation farther from its region's centre is not chosen
    pool_per_bin: float | None = 3.0   # the units localised each round: at most this times *per bin* per z bin
    z_range_nm: float | None = None    # only units within this of the height in focus are chosen (None: all)


@dataclass
class Edge:
    shift_nm: float                    # where the heights' lower edge is: the coverslip is at 0
    sd_nm: float
    spike: float                       # the fraction of the window's molecules on the coverslip
    n: int                             # heights in the window


def coverslip_edge(h, sd, window: float = 0.3) -> Edge | None:
    """The lower edge of heights ``h`` with CRLBs ``sd`` (module docstring): None with too few heights.

    The heights in the lowest ``window`` are modelled as a flat density above the edge s, optionally a
    spike on it, each height blurred by κ times its CRLB, plus a fraction ε of outliers uniform over the
    window (at most 5 %).  κ ≥ 1 because a localisation scatters more than its CRLB once the model is not exact (20 nm
    against 17 on simulated movies), and the outliers because a few molecules land hundreds of nanometres
    below the coverslip; without either, the fit explained that tail by moving the edge down 75 nm.  The
    spike is kept only if it gains more than `SPIKE_NATS`: without molecules on the coverslip its weight
    sits at its bound, 0, where the fit with it put the edge 10–28 nm too high on simulated heights."""
    from scipy.optimize import minimize
    from scipy.special import expit
    from scipy.stats import norm
    h, sd = np.asarray(h, float), np.asarray(sd, float)
    ok = np.isfinite(h) & np.isfinite(sd) & (sd > 0)
    h, sd = h[ok], sd[ok]
    if len(h) < 20:
        return None
    U = np.quantile(h, window)
    x, s0_i = h[h < U], sd[h < U]
    L = U - x.min()

    def unpack(p, spike):
        s, kappa, eps = p[0], 1 + np.exp(np.clip(p[1], -30, 5)), MAX_OUTLIERS * expit(p[2])
        return s, kappa, eps, (expit(p[3]) if spike else 0.0)

    def nll(p, spike=True):
        s, kappa, eps, w = unpack(p, spike)
        s_i = kappa * s0_i
        a = (U - s) / s_i
        flat = norm.cdf((x - s) / s_i) / np.maximum((U - s) * norm.cdf(a) + s_i * norm.pdf(a), 1e-300)
        dens = (1 - w) * flat
        if spike:
            dens = dens + w * norm.pdf(x, s, s_i) / np.maximum(norm.cdf(a), 1e-300)
        return -np.sum(np.log(np.maximum((1 - eps) * dens + eps / L, 1e-300)))

    opts = dict(xatol=0.02, fatol=1e-6, maxiter=8000, maxfev=8000)
    starts = np.quantile(x, [0.05, 0.3])
    flat = min((minimize(lambda p: nll(p, False), [s0, -1.0, -2.0], method="Nelder-Mead", options=opts)
                for s0 in starts), key=lambda r: r.fun)
    both = min((minimize(nll, [s0, -1.0, -2.0, 0.0], method="Nelder-Mead", options=opts) for s0 in starts),
               key=lambda r: r.fun)
    spiked = flat.fun - both.fun > SPIKE_NATS
    best, f = (both, nll) if spiked else (flat, lambda p: nll(p, False))
    w = unpack(best.x, spiked)[3]
    return Edge(float(best.x[0]), _sd_of_first(f, best.x, float(np.median(s0_i))), float(w), len(x))


def _sd_of_first(f, x, scale: float) -> float:
    """The standard error of the first parameter: the inverse Hessian of ``f`` at ``x``, by differences,
    over the parameters not at a bound.  The others are transformed (logit, log) and one at its bound
    (|value| > 8: an outlier fraction or κ − 1 of ~0) is flat there and left out, as it would make the
    Hessian singular."""
    x = np.asarray(x, float)
    keep = [0] + [i for i in range(1, len(x)) if abs(x[i]) < 8]
    n = len(keep)
    steps = np.array([max(1.0, scale / 10)] + [0.05] * (n - 1))
    H = np.empty((n, n))
    for a, i in enumerate(keep):
        for b, j in enumerate(keep):
            ei, ej = np.zeros_like(x), np.zeros_like(x)
            ei[i], ej[j] = steps[a], steps[b]
            H[a, b] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * steps[a] * steps[b])
    try:
        v = float(np.linalg.inv(H)[0, 0])
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.sqrt(v)) if v > 0 else float("inf")


@dataclass
class Round:
    index: int
    used: np.ndarray                   # the units learnt from (`InSituROIs` indices)
    calibration: InSituCalibration
    state: list                        # every unit's state in this round's selection
    edge: Edge | None = None           # the coverslip edge found before this round's learning
    focus_prior: tuple | None = None   # (mean, sd) nm, from the edge
    moved: float = float("inf")        # the rms change since the last round, in standard errors (`_moved`)
    heights: np.ndarray | None = None  # every kept unit's height localised before this round (the edge's data)
    heights_sd: np.ndarray | None = None


@dataclass
class InSituRun:
    rounds: list = field(default_factory=list)
    model: object = None               # the model of the last round, its focus prior the edge's

    @property
    def final(self) -> InSituCalibration:
        return self.rounds[-1].calibration


def localise_units(model, shared: dict, rois, ids, noise, heights=None, starts=None) -> dict:
    """Every unit ``ids`` of ``rois`` localised on the model's ``shared`` values, from ``heights`` (common to
    all) or ``starts`` [len(ids), S] (each its own): its start (`Batch.start` keys), its height and CRLB,
    photons and mean deviance per pixel."""
    batch, _ = rois.batch(ids=ids)
    batch = batch.to(dtype=model.psf.grid.rho.dtype)
    if starts is not None:
        found, _ = localise_from_starts(model, shared, batch, noise, starts)
    else:
        found, _ = localise_from_heights(model, shared, batch, noise, heights)
    fit, _ = localise(model, shared, replace(batch, start=found), noise, chunk_size=INSITU_CHUNK)
    phys = fit.params.physical()
    photons = phys["emitter.photons"] if "emitter.photons" in phys else phys["emitter.photons_per_slice"].mean(1)
    xy = phys["emitter.xyz"][:, :2].detach().cpu().double().numpy() / np.asarray(model.pixel_size_nm)
    return {"start": found, "h": phys["emitter.xyz"][:, 2].detach().cpu().double().numpy(),
            "xy_offset_px": np.hypot(xy[:, 0], xy[:, 1]),
            "sd_h": fit.std["emitter.xyz"][:, 2].cpu().double().numpy(),
            "photons": photons.detach().cpu().double().numpy(),
            "deviance": score_beads(model, fit.params, batch, noise, INSITU_CHUNK).deviance}


def _focus_per_height(history, measured: float) -> float:
    """How far the focus moves per nanometre the coverslip's molecules move: by the last two rounds'
    (focus, edge) when their foci differ, a secant step, else ``measured`` (`_measured_ratio`)."""
    if len(history) >= 2:
        (f0, s0), (f1, s1) = history[-2], history[-1]
        if abs(f1 - f0) > 10 and abs(s1 - s0) > 1:
            return float(np.clip((f1 - f0) / (s1 - s0), 0.25 * measured, 4 * measured))
    return measured


def _measured_ratio(model, shared: dict, rois, ids, start: dict, noise, paraxial: float,
                    delta_nm: float = 50.0) -> float:
    """Focus per height of the molecules ``ids`` (the lowest, by the edge's window): localised again with
    the focus ``delta_nm`` higher, from where they were found, the median of how far they moved.  The
    paraxial n_imm/n_med is far off at high NA: on a simulated movie (NA 1.43, oil into water) a focus
    120 nm too high raised the molecules on the coverslip by 65 nm, not the 106 nm it predicts, and the
    first correction made only ~60 % of the way."""
    batch, _ = rois.batch(ids=ids)
    batch = replace(batch.to(dtype=model.psf.grid.rho.dtype), start=start)
    h = []
    for f in (0.0, delta_nm):
        q = dict(shared, **{"geometry.focus": shared["geometry.focus"] + f})
        fit, _ = localise(model, q, batch, noise, chunk_size=INSITU_CHUNK)
        h.append(fit.params.physical()["emitter.xyz"][:, 2].detach().cpu().double().numpy())
    moved = float(np.median(h[1] - h[0]))
    return float(np.clip(delta_nm / moved, 0.5 * paraxial, 4 * paraxial)) if moved > 1 else paraxial


def _with_focus_prior(model, mean: float, sd: float):
    m = copy.copy(model)
    m.geometry = InSituGeometry(mean, sd, model.geometry.fit_focus)
    return m


def _moved(new: dict, old: dict, std_new: dict, std_old: dict) -> float:
    """How far the Zernikes and the focus moved since the last round: the rms over them of each change
    in units of its standard error, √(σ_new² + σ_old²).  Not the largest: each round learns from other
    molecules, so the largest of some twenty changes is ~3 even when nothing moves but the noise."""
    z = []
    for n in ("pupil.phase", "pupil.magnitude", "geometry.focus"):
        if n in new and n in old and n in std_new and n in std_old:
            d = (torch.as_tensor(new[n]) - torch.as_tensor(old[n])).cpu().double().reshape(-1)
            sd = (std_new[n].cpu().double() ** 2 + std_old[n].cpu().double() ** 2).sqrt().reshape(-1)
            z.append(d / sd.clamp(min=1e-12))
    z = torch.cat(z) if z else torch.zeros(0)
    return float(z.pow(2).mean().sqrt()) if z.numel() else 0.0


def calibrate_insitu(model, rois, start: dict | None = None, noise=None, settings: RoundSettings | None = None,
                     selection=None, progress=None, cancel=None) -> InSituRun:
    """Learn from ``rois`` (`smappy.calibrate.insitu.InSituROIs`) in rounds (module docstring).

    ``selection`` is the `InSituSettings` the rounds choose by (default ``rois``' own: the first round's);
    ``start`` the shared values to start from, which give the sign (`learn_insitu`).  ``progress`` gets
    every learning iteration's entry (`learn_insitu`) and a ``{"message", "round"}`` before each round;
    ``cancel`` (an Event) stops after the current iteration, the twin test skipped."""
    from smappy.calibrate.insitu import InSituSettings, select
    s = settings or RoundSettings()
    sel = selection or InSituSettings()
    noise = noise if noise is not None else ShiftedPoisson()
    dtype = model.psf.grid.rho.dtype
    optics = model.psf.grid.optics
    run = InSituRun()
    ids = rois.ids("used")
    batch, _ = rois.batch(ids=ids)
    batch = batch.to(dtype=dtype)
    roi = int(rois.units.regions.shape[-1])
    tell = progress or (lambda entry: None)
    tell({"message": f"round 1: localising {len(ids)} molecules from their widths", "round": 1})
    first = full_start(model, batch, start)
    located, _ = localise_from_starts(model, first, batch, noise, width_starts(model, first, _widths(rois, ids), roi))
    tell({"message": f"round 1: learning from {len(ids)} molecules", "round": 1})
    cal = learn_insitu(model, batch, start=start, noise=noise, twin_test=False, progress=progress,
                       cancel=cancel, located=located)
    run.rounds.append(Round(1, ids, cal, list(rois.state)))
    if _stopped(cal):
        return run
    pool = _capped_pool(rois, sel, s.pool_per_bin)
    current = model
    history = []                       # (focus, edge shift) of every round that found the edge
    last_height: dict = {}             # each unit's height where it was last found, a start for the next round
    measured = None                    # focus per height of the coverslip's molecules (`_measured_ratio`)
    for index in range(2, s.max_rounds + 1):
        tell({"message": f"round {index}: localising every molecule on the model", "round": index})
        shared = cal.calibration.shared()
        focus = float(shared["geometry.focus"])
        # each unit from where the model's widths put it, and from where it was found last
        starts = width_starts(current, shared, _widths(rois, pool), roi)
        previous = np.array([last_height.get(int(u), np.nan) for u in pool])
        loc = localise_units(current, shared, rois, pool, noise, starts=np.c_[starts, previous])
        last_height.update({int(u): float(h) for u, h in zip(pool, loc["h"], strict=True)})
        edge, prior = None, None
        shift = 0.0
        if s.coverslip:
            edge = coverslip_edge(loc["h"], loc["sd_h"], s.edge_window)
            if edge is not None and edge.sd_nm <= s.edge_max_sd_nm:
                shift = edge.shift_nm
                history.append((focus, shift))
                if measured is None:
                    low = np.flatnonzero(loc["h"] < np.quantile(loc["h"], s.edge_window))
                    measured = _measured_ratio(current, shared, rois, pool[low],
                                               {k: v[torch.as_tensor(low)] for k, v in loc["start"].items()}, noise,
                                               optics.n_imm / optics.n_med)
                ratio = _focus_per_height(history, measured)
                prior = (focus - shift * ratio, max(edge.sd_nm * ratio, s.focus_floor_nm))
                current = _with_focus_prior(model, *prior)
                shared = dict(shared, **{"geometry.focus": torch.as_tensor(prior[0], dtype=dtype)})
        U = len(rois.units.reason)
        h, N, dev = (np.full(U, np.nan) for _ in range(3))
        h[pool], N[pool], dev[pool] = loc["h"] - shift, loc["photons"], loc["deviance"]
        # A localisation that ended off its region's centre went to a neighbour or into the background; it can
        # still fit well, so the deviance does not catch it.  In a dense movie (Zenodo tubulin) such units were
        # most of round 2's choice, and learning then rejected or lost them.
        rejected = [None] * U
        centre = in_focus_height(current, float(shared["geometry.focus"]))
        for k, u in enumerate(pool):
            if not loc["xy_offset_px"][k] <= s.max_xy_offset_px:
                rejected[u] = "off centre"
            elif s.z_range_nm is not None and abs(loc["h"][k] - shift - centre) > s.z_range_nm:
                rejected[u] = "z range"
        state, _, _ = select(rois.units, sel, proxy=h, photons=N, shape=dev, rejected=rejected, one_sided=True)
        used = np.flatnonzero(np.array([st == "used" for st in state]))
        where = {int(u): k for k, u in enumerate(pool)}
        rows = torch.tensor([where[int(u)] for u in used])
        located = {k: v[rows] for k, v in loc["start"].items()}
        located["xyz"] = located["xyz"].clone()
        located["xyz"][:, 2] -= shift
        batch, _ = rois.batch(ids=used)
        tell({"message": f"round {index}: learning from {len(used)} molecules", "round": index})
        new = learn_insitu(current, batch.to(dtype=dtype), start=shared, noise=noise, located=located,
                           twin_test=False, progress=progress, cancel=cancel)
        moved = _moved(new.calibration.shared(), cal.calibration.shared(), new.calibration.result.std,
                       cal.calibration.result.std)
        run.rounds.append(Round(index, used, new, state, edge, prior, moved, loc["h"], loc["sd_h"]))
        cal = new
        if _stopped(cal):
            return run
        if moved < s.stop_sd and (edge is None or abs(edge.shift_nm) <= max(edge.sd_nm, s.focus_floor_nm)):
            break
    last = run.rounds[-1]
    tell({"message": "the twin test: the mirrored pupil fitted to the same molecules"})
    batch, _ = rois.batch(ids=last.used)
    located = replace(batch.to(dtype=dtype), start=None)
    learnt = cal.calibration.shared()
    heights = start_heights(current, float(learnt["geometry.focus"]))
    cal.twin = twin_fit(current, cal.calibration, located, noise, heights,
                        starts=width_starts(current, twin(current, learnt), _widths(rois, last.used), roi))
    run.model = current
    return run


def _capped_pool(rois, sel, per_bin_factor: float) -> np.ndarray:
    """The units each round localises: every one detection kept, but at most ``per_bin_factor`` times *per
    bin* from each bin of the Gaussian's z proxy, drawn at random once, so the rounds see the same ones.
    Localising the whole pool was 80 % of a run (Zenodo tubulin: ~5000 units, 280–350 s a round), when a
    round draws only *per bin* from each bin; a few times that leaves the draw its choice and the
    coverslip edge enough heights."""
    from smappy.calibrate.insitu import z_proxy
    pool = np.flatnonzero(np.array([r is None for r in rois.units.reason]))
    if per_bin_factor is None or not len(pool):
        return pool
    cap = max(1, int(round(per_bin_factor * sel.per_bin)))
    proxy = z_proxy(rois.units.gauss, sel.proxy)[pool]
    finite = np.isfinite(proxy)
    lo, hi = np.quantile(proxy[finite], [0.01, 0.99]) if finite.any() else (0.0, 1.0)
    bins = np.clip(np.digitize(proxy, np.linspace(lo, hi, sel.bins + 1)) - 1, 0, sel.bins - 1)
    rng = np.random.default_rng(sel.seed + 1)
    keep = []
    for b in range(sel.bins):
        members = pool[(bins == b) & finite]
        keep.append(rng.permutation(members)[:cap])
    return np.sort(np.concatenate(keep)) if keep else pool


def _widths(rois, ids) -> np.ndarray:
    """[len(ids), 2] the units' Gaussian widths σx², σy² (px²) as detection measured them."""
    g = rois.units.gauss
    ids = np.asarray(ids, int)
    return np.c_[g["sigma_x_pix"][ids] ** 2, g["sigma_y_pix"][ids] ** 2]


def full_start(model, batch, start: dict | None) -> dict:
    """Every shared value of ``model`` for ``batch``, ``start``'s where it gives them (`learn_insitu`'s)."""
    from ..params import Params
    base = Params(model.specs(replace(batch, start=_blank(batch))), batch.n_emitters, dtype=model.psf.grid.rho.dtype,
                  device=model.psf.grid.rho.device)
    shared = {n: v for n, v in base.physical().items() if base.specs[n].scope == "shared"}
    for n, v in (start or {}).items():
        if n in shared and tuple(torch.as_tensor(v).shape) == tuple(shared[n].shape):
            shared[n] = torch.as_tensor(v, dtype=shared[n].dtype, device=shared[n].device)
    return shared


def _stopped(cal: InSituCalibration) -> bool:
    return cal.calibration.result.stop_reason == "cancelled"


#: the heights the focus is looked for over when it is estimated from the data (`focus_from_data`), the step
#: between the heights each molecule is started from, and at most how many molecules: localising every chosen
#: one from 21 heights took over 5 min on one CPU core, and the edge needs only a few hundred
FOCUS_SEARCH_NM, FOCUS_SEARCH_STEP_NM, FOCUS_SEARCH_MOLECULES = (0.0, 3000.0), 250.0, 200


def focus_from_data(model, rois, start: dict | None = None, noise=None, guess_nm: float = 1000.0,
                    window: float = 0.3, molecules: int = FOCUS_SEARCH_MOLECULES) -> tuple[float, Edge]:
    """The focus as the coverslip edge puts it, before any learning: (focus nm, the edge).

    The units the first round would learn from are localised on the start pupil (a bead model or the
    astigmatism typed in) with the focus at ``guess_nm``, from heights all over `FOCUS_SEARCH_NM` (at most
    `FOCUS_SEARCH_MOLECULES` of them, drawn at random); the
    lower edge of their heights is the coverslip (`coverslip_edge`), and the focus is moved by its shift
    times the focus per height, measured (`_measured_ratio`).  The start pupil is rough, so this is a
    start: the rounds and their edge refine it.  It needs molecules on or near the coverslip."""
    noise = noise if noise is not None else ShiftedPoisson()
    optics = model.psf.grid.optics
    ids = rois.ids("used")
    if len(ids) > molecules:
        ids = np.sort(np.random.default_rng(0).choice(ids, molecules, replace=False))
    batch, _ = rois.batch(ids=ids)
    from ..params import Params
    base = Params(model.specs(replace(batch, start=_blank(batch))), batch.n_emitters,
                  dtype=model.psf.grid.rho.dtype)
    shared = {n: v for n, v in base.physical().items() if base.specs[n].scope == "shared"}
    shared.update({n: torch.as_tensor(v, dtype=shared[n].dtype) for n, v in (start or {}).items()
                   if n in shared and tuple(torch.as_tensor(v).shape) == tuple(shared[n].shape)})
    shared["geometry.focus"] = torch.as_tensor(float(guess_nm), dtype=shared["geometry.focus"].dtype)
    heights = np.arange(FOCUS_SEARCH_NM[0], FOCUS_SEARCH_NM[1] + 1, FOCUS_SEARCH_STEP_NM)
    loc = localise_units(model, shared, rois, ids, noise, heights)
    edge = coverslip_edge(loc["h"], loc["sd_h"], window)
    if edge is None or not np.isfinite(edge.sd_nm):
        raise ValueError("the focus cannot be estimated from these molecules: too few, or no lower edge to "
                         "their heights; give it")
    low = np.flatnonzero(loc["h"] < np.quantile(loc["h"], window))
    lowest = {k: v[torch.as_tensor(low)] for k, v in loc["start"].items()}
    ratio = _measured_ratio(model, shared, rois, ids[low], lowest, noise, optics.n_imm / optics.n_med)
    return float(guess_nm - edge.shift_nm * ratio), edge


def _blank(batch) -> dict:
    z = batch.data.new_zeros(batch.n_emitters)
    return {"xyz": torch.stack([z, z, z], -1), "photons": z + 1, "background": z + 1}
