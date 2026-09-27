"""The ground truth both simulators draw from, and the localizations a fit of
it would have given.

`ground_truth` runs the shared stages -- structure, labelling, blinking,
drift -- and is what the camera frames are drawn from as well, so the same
settings and seed give the same molecules either way.  `simulate` turns it
into a localization table without drawing a pixel:

* the photons of a frame are Poisson around what the fluorophore emitted in
  it, and a localization below the detection limit is not found;
* the lateral precision is the Mortensen et al. (Nat. Methods 2010) bound for
  a Gaussian spot of the PSF's width on pixels of this size over this
  background -- the form SMAP and most fitters report -- times sqrt(2) for an
  EMCCD's excess noise, and z is a fixed multiple of it.  The error that is
  added is drawn from that same precision, so the table's ``xy_err_nm`` is
  honest, which is what the precision plugins test against;
* two emitters on in one frame closer than the separation were one spot to
  the fitter: either both are removed, or they come out as one localization
  at their photon-weighted mean with their photons added, which is what a
  single-emitter fit of an unresolved pair converges to.  Removal keeps the
  table clean; averaging keeps the artefact, for a test of what it does.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional

import numpy as np

from ..locs import Localizations
from .kinetics import Blinks, Emission, Fluorophores, blink, blink_offsets, emission, label
from .settings import SimulationSettings
from .structure import load_structure


@dataclass
class GroundTruth:
    settings: SimulationSettings
    name: str                        # the structure's
    n_labels: int
    fluorophores: Fluorophores
    blinks: Blinks
    emission: Emission
    drift_nm: Optional[np.ndarray]   # (n_frames, 3), or None
    poses: Optional[dict] = None     # where each copy was put and how it was turned

    def positions(self) -> np.ndarray:
        """Where each emitting row was, drift included: (len(emission), 3)."""
        xyz = self.fluorophores.xyz[self.emission.owner]
        if self.blinks.offset is not None and self.emission.blink is not None:
            xyz = xyz + self.blinks.offset[self.emission.blink]
        if self.drift_nm is not None:
            xyz = xyz + self.drift_nm[self.emission.frame]
        return xyz


def settings_with(settings: Optional[SimulationSettings] = None, **values) -> SimulationSettings:
    return replace(settings or SimulationSettings(), **values)


def ground_truth(settings: SimulationSettings) -> GroundTruth:
    rng = np.random.default_rng(settings.seed)
    structure = load_structure(settings.structure.source())
    labels = structure.sample(rng)
    fluorophores = label(labels, settings.labelling, rng)
    blinks = blink(len(fluorophores), settings.n_frames, settings.blinking, rng)
    blinks.offset = blink_offsets(len(blinks), settings.labelling.linkage_free_nm, rng)
    emitted = emission(blinks, settings.n_frames)
    drift = drift_trace(settings.n_frames, rng) if settings.drift else None
    return GroundTruth(settings, structure.name, len(labels), fluorophores, blinks,
                       emitted, drift, labels.poses)


def drift_trace(n_frames: int, rng) -> np.ndarray:
    """A smooth random walk plus a slow linear creep of ~100 nm, (n, 3) nm."""
    walk = np.cumsum(rng.normal(0, 0.6, (n_frames, 3)), axis=0)
    walk -= walk[0]
    creep = np.linspace(0, 1, n_frames)[:, None] * np.array([80.0, -50.0, 30.0])
    return walk + creep


def mortensen(photons, background, sigma_nm, pixelsize_nm, emccd: bool = False):
    """The lateral precision per axis, nm, of a Gaussian spot fitted by
    maximum likelihood: Mortensen et al. 2010, eq. 5, with the pixel's own
    blur.  ``background`` in photons per pixel.

        var = sa^2 / N / (1 + int_0^1 ln t / (1 + t / tau) dt),
        sa^2 = sigma^2 + a^2 / 12,    tau = 2 pi sa^2 b / (N a^2)

    Not their eq. 6, the ``16/9 + 8 pi sa^2 b / (N a^2)`` that SMAP's
    ``MortensenCRLB`` uses: that is the error of an *unweighted least-squares*
    fit, up to 16/9 of this variance at low background.  Against SMAPpy's own
    fitter on simulated spots this one agrees to 1-2% from 200 to 5000
    photons; that one was 10-23% too pessimistic.
    """
    n = np.maximum(np.asarray(photons, float), 1.0)
    sa2 = np.asarray(sigma_nm, float) ** 2 + pixelsize_nm ** 2 / 12.0
    tau = 2 * np.pi * sa2 * np.asarray(background, float) / (n * pixelsize_nm ** 2)
    variance = sa2 / n / (1.0 + _log_integral(tau))
    return np.sqrt(variance * (2.0 if emccd else 1.0))


# Gauss-Legendre on u in (0, 1) with t = u^2: the integrand's logarithmic
# singularity at t = 0 becomes 4 u ln u, smooth enough that 48 nodes agree
# with adaptive quadrature to 2e-7 over tau from 1e-4 to 1e3
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(48)
_U = 0.5 * (_NODES + 1.0)
_W = 0.5 * _WEIGHTS


def _log_integral(tau):
    """``int_0^1 ln t / (1 + t / tau) dt``, elementwise: 0 with no background
    (tau = 0, so the variance is sa^2 / N), towards -1 as the background
    swamps the spot."""
    tau = np.asarray(tau, float)
    t = _U ** 2
    values = (_W * 4.0 * _U * np.log(_U)
              / (1.0 + t / np.maximum(tau, 1e-300)[..., None])).sum(axis=-1)
    return np.where(tau > 0, values, 0.0)


def spot_sigma(z_nm, optics) -> np.ndarray:
    """The PSF's width at ``z_nm``: the mean of x and y for an astigmatic one."""
    from .camera import astigmatic_sigmas
    z = np.asarray(z_nm, float)
    if not optics.astigmatism:
        return np.full(z.shape, optics.sigma_nm)
    sx, sy = astigmatic_sigmas(z, optics.sigma_nm, optics.focal_offset_nm, optics.depth_nm)
    return np.sqrt(sx * sy)


def close_groups(x, y, frame, distance: float) -> np.ndarray:
    """A group index per row: rows on in one frame within ``distance`` of each
    other (transitively) share one."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    n = len(frame)
    if n == 0 or distance <= 0:
        return np.arange(n)
    pairs = []
    edges = np.flatnonzero(np.diff(frame) != 0) + 1          # frame is sorted
    for block in np.split(np.arange(n), edges):
        if block.size < 2:
            continue
        p = cKDTree(np.column_stack([x[block], y[block]])).query_pairs(
            distance, output_type="ndarray")
        if p.size:
            pairs.append(block[p])
    if not pairs:
        return np.arange(n)
    p = np.vstack(pairs)
    graph = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def unresolvable(x, y, frame, min_separation: float) -> np.ndarray:
    """True where another emitter is active in the same frame within a PSF."""
    order = np.argsort(frame, kind="stable")
    group = np.empty(len(frame), np.int64)
    group[order] = close_groups(np.asarray(x)[order], np.asarray(y)[order],
                                np.asarray(frame)[order], min_separation)
    return np.bincount(group, minlength=len(frame))[group] > 1


def simulate(settings: Optional[SimulationSettings] = None, **values) -> Localizations:
    """A localization table: `SimulationSettings`, or its top-level fields as
    keywords (``simulate(n_frames=8000, seed=3, drift=True)``)."""
    settings = settings_with(settings, **values)
    truth = ground_truth(settings)
    return localizations(truth)


def localizations(truth: GroundTruth) -> Localizations:
    settings = truth.settings
    out, optics = settings.localizations, settings.optics
    rng = np.random.default_rng([settings.seed, 1])
    em = truth.emission
    photons = rng.poisson(em.photons).astype(np.float64)
    found = photons >= max(out.min_photons, 1.0)
    frame, owner, photons = em.frame[found], em.owner[found], photons[found]
    xyz = truth.positions()[found]
    background = np.round(draw_background(settings, len(frame), rng), 3)

    group = close_groups(xyz[:, 0], xyz[:, 1], frame, out.min_separation_nm)
    size = np.bincount(group, minlength=len(frame))[group]
    n_close = int((size > 1).sum())
    n_merged = None
    if out.close == "remove":
        keep = size == 1
        frame, owner, photons, xyz, background = (
            frame[keep], owner[keep], photons[keep], xyz[keep], background[keep])
    elif out.close == "average":
        frame, owner, photons, xyz, background, n_merged = _average(
            group, frame, owner, photons, xyz, background)
    else:
        raise ValueError(f"close emitters {out.close!r}: remove or average")

    n = len(frame)
    sigma = spot_sigma(xyz[:, 2], optics)
    prec = mortensen(photons, background, sigma, optics.pixelsize_nm, out.emccd)
    prec_z = out.z_factor * prec
    noisy = xyz + np.column_stack([rng.normal(0, prec), rng.normal(0, prec),
                                   rng.normal(0, prec_z)])
    fl = truth.fluorophores
    columns = {
        "frame": frame.astype(np.int64),
        "x_nm": noisy[:, 0].astype(np.float32), "y_nm": noisy[:, 1].astype(np.float32),
        "z_nm": noisy[:, 2].astype(np.float32),
        "photons": photons.astype(np.float32), "background": background.astype(np.float32),
        "xy_err_nm": prec.astype(np.float32), "z_err_nm": prec_z.astype(np.float32),
        # a fitted width scatters about the true one by about the precision
        "sigma_nm": (sigma + rng.normal(0, prec)).astype(np.float32),
        # the ground truth: which fluorophore, of which dye, in which copy
        "emitter": owner.astype(np.int32), "dye": fl.dye[owner].astype(np.int32),
        "copy": fl.copy[owner].astype(np.int32),
    }
    if n_merged is not None:
        columns["n_merged"] = n_merged.astype(np.int32)
    metadata = _metadata(truth, "smappy blinking structure")
    metadata.update({"n_localizations": n, "close_emitters": out.close,
                     "min_separation_nm": out.min_separation_nm,
                     "n_close": n_close,
                     # the name the plugin's text and older scripts read
                     "n_unresolvable_dropped": n_close if out.close == "remove" else 0})
    return Localizations(columns, metadata)


def draw_background(settings: SimulationSettings, n: int, rng) -> np.ndarray:
    from .kinetics import draw
    return draw(settings.background, settings.background_std, n, rng)


def _average(group, frame, owner, photons, xyz, background):
    """One row per group: positions weighted by photons, photons added, the
    brightest member's identity."""
    unique, inverse = np.unique(group, return_inverse=True)
    m = len(unique)
    total = np.bincount(inverse, weights=photons, minlength=m)
    w = photons / total[inverse]
    mean = np.column_stack([np.bincount(inverse, weights=w * xyz[:, k], minlength=m)
                            for k in range(3)])
    count = np.bincount(inverse, minlength=m)
    bg = np.bincount(inverse, weights=background, minlength=m) / count
    # the brightest member of each group, by sorting on (group, photons)
    order = np.lexsort((-photons, inverse))
    head = order[np.concatenate([[0], np.flatnonzero(np.diff(inverse[order])) + 1])]
    rows = np.argsort(frame[head], kind="stable")
    return (frame[head][rows], owner[head][rows], total[rows], mean[rows], bg[rows],
            count[rows])


def _metadata(truth: GroundTruth, what: str) -> dict:
    s = truth.settings
    return {"units": "nm", "simulation": what, "seed": s.seed,
            "structure": s.structure.source(), "structure_name": truth.name,
            "n_labels": truth.n_labels, "n_emitters": len(truth.fluorophores),
            "n_blinks": len(truth.blinks), "n_frames": s.n_frames,
            "off_time_frames": round(truth.blinks.off_time, 3),
            "simulation_settings": asdict(s),
            # a copy's position and turn, by its `copy` number: what a model
            # fitted to one site should give back
            "copies": truth.poses,
            "drift_truth": (truth.drift_nm.round(3).tolist()
                            if truth.drift_nm is not None else None)}
