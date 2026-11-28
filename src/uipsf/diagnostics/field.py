"""How much a field-dependent model's aberrations vary over the beads' field, and whether that is real.

The report a calibration with *field dependence* ticked shows (`docs/uipsf/field-dependence.md`): per varying
mode the largest change across the beads' positions, in rad and nm of path, and one number for whether the
field is there at all -- a Wald test of the field's coefficients against zero, ½χ² in nats, comparable with
the likelihood-ratio probes' gains (`probes.py`): c·C⁻¹·c/2 is the objective's drop to second order, C the
coefficients' CRLB covariance from the fit, so it costs no refit.  Flagged beyond the χ² quantile of ``p``
(10⁻⁶, the probes' default).  Below it the field is noise: learn without it.

Too few beads do not determine the field at all: 4 beads for the 14 coefficients of a linear field of
seven modes left a covariance with negative eigenvalues (the fit's Hessian singular, November 2026, a
customer's dual-free run), whose square roots stopped the report, and the window's *State* with it.  Then
the report says the field is not determined (``determined``), with no gain and no uncertainties.

Over a small field the test is not asked.  An imperfect model's misfit that varies with position -- with
the brightness, under a Gaussian illumination -- is what a polynomial of the position takes up there: an in
situ run on a simulated 8 µm movie found 31 nats against a threshold of 27, the field "changing" by 0.4 rad of
astigmatism across it.  Below `MIN_SPAN_UM` the report says the field cannot be told and calls it not
significant.  M5's beads, whose linear astigmatism of 0.09 rad per half field is real, span 25 µm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: the emitters' span (µm, the larger side) below which a field is not judged (module docstring)
MIN_SPAN_UM = 20.0


@dataclass
class FieldReport:
    gain: float                 # ½ c·C⁻¹·c, nats
    dof: int                    # the field's coefficients
    threshold: float            # ½ χ²(p, dof)
    significant: bool
    changes: dict               # {Noll: (largest change across the beads, rad; its CRLB, rad)}
    wavelength_nm: float
    span_um: float              # the beads' span over the chip, the larger side
    n_beads: int = 0
    determined: bool = True     # False: the coefficients' covariance is not positive definite (docstring)
    unit: str = "beads"         # what the emitters are, for the text: "molecules" from a sample's

    def summary(self) -> str:
        nm = 1 / (2 * math.pi) * self.wavelength_nm
        if self.span_um < MIN_SPAN_UM:
            return (f"field dependence cannot be told over only {self.span_um:.0f} µm of {self.unit}: learn without "
                    f"it, or from {self.unit} spread over at least {MIN_SPAN_UM:.0f} µm")
        if not self.determined:
            return (f"field dependence is not determined by {self.n_beads} {self.unit} ({self.dof} coefficients): "
                    f"learn without it, or with more {self.unit} spread over the field")
        parts = ", ".join(f"Noll {j} {d:+.3f} rad ({d * nm:+.0f} nm)" for j, (d, _) in self.changes.items())
        verdict = ("field dependence is significant" if self.significant
                   else "field dependence is not significant: learn without it")
        return (f"{verdict} ({self.gain:.0f} nats over {self.dof} coefficients, threshold {self.threshold:.0f}); "
                f"largest changes over the {self.unit}' {self.span_um:.0f} µm: {parts}")


def field_report(cal, channel: int = 0, p: float = 1e-6) -> FieldReport | None:
    """The `FieldReport` of a learnt `Calibration` ``cal``; None when its channel has no field."""
    import torch
    from scipy.stats import chi2

    from ..model.channels import prefix
    ch = cal.model.channels[channel]
    field = getattr(ch, "field", None)
    if field is None:
        return None
    name = prefix(channel) + "field.phase"
    res = cal.result
    lo, hi = next((a, b) for n, a, b in res.params.layout("shared") if n == name)
    cov = res.covariance_shared[lo:hi, lo:hi].detach().cpu().double().numpy()
    c = res.params.physical()[name].detach().cpu().double().numpy()
    dof = len(c)
    threshold = 0.5 * float(chi2.isf(p, dof))
    xy = cal.beads_field_xy() if hasattr(cal, "beads_field_xy") else None
    if xy is None:
        raise ValueError("the calibration does not keep its beads' chip positions")
    xy = torch.as_tensor(np.asarray(xy, float))
    determined = bool(np.isfinite(cov).all())
    if determined:
        eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
        determined = bool(eig.min() > 1e-12 * max(eig.max(), 0.0))
    gain = 0.5 * float(c @ np.linalg.solve(cov, c)) if determined else math.nan
    basis = field.basis(xy).numpy()                                        # [E, T]
    changes = {}
    for j, _ in field.terms:
        rows = [i for i, (m, _, _) in enumerate(field.layout) if m == j]
        values = basis[:, rows] @ c[rows]                                  # [E] change at each bead
        k = int(np.argmax(np.abs(values)))
        g = basis[k, rows]
        err = math.sqrt(max(0.0, float(g @ cov[np.ix_(rows, rows)] @ g))) if determined else math.nan
        changes[j] = (float(values[k]), err)
    span = (xy.max(0).values - xy.min(0).values).numpy() * np.asarray(field.pixel_size_nm) / 1000.0
    from ..model.geometry import BeadStackGeometry
    unit = "beads" if isinstance(cal.model.geometry, BeadStackGeometry) else "molecules"
    span_um = float(span.max())
    return FieldReport(gain, dof, threshold, determined and gain > threshold and span_um >= MIN_SPAN_UM, changes,
                       float(ch.psf.grid.optics.wavelength_nm), span_um, int(len(xy)), determined, unit)


# ------------------------------------------------------------------ the aberration maps

#: the map's grid, points along the longer side of the emitters' span
MAP_POINTS = 25
#: what *show* offers first: the families that vary most, at most this many, each at least `MOST_SHARE` of
#: the largest variation (a family barely varying is noise beside one that does)
MOST, MOST_SHARE = 3, 0.25
SHOW_MOST, SHOW_ALL = "varying most", "all varying"


@dataclass
class FieldMaps:
    """A field-dependent model's aberrations over the emitters' span: every phase mode's coefficient and the
    pupil's total RMS wavefront at each point of a grid of chip positions (`field_maps`)."""
    x_px: np.ndarray            # [X] chip pixels
    y_px: np.ndarray            # [Y]
    modes: list                 # Noll indices, learnt and held, in Noll order
    coefficients: np.ndarray    # [Y, X, M] rad
    rms: np.ndarray             # [Y, X] rad: the RMS of the pupil's phase over the aperture, about its mean
    varying: list               # the Noll modes the field varies (`FieldPolynomial.terms`)
    points: np.ndarray          # [E, 2] the emitters' chip positions
    wavelength_nm: float

    def of(self, j: int) -> np.ndarray:
        """[Y, X] the coefficient of Noll ``j`` (rad)."""
        return self.coefficients[..., self.modes.index(j)]


@dataclass
class MapItem:
    """One panel of the maps: a pair of modes of one family, drawn as a magnitude and an orientation
    (``pair`` = (cos mode, sin mode)), or a single mode as a map."""
    label: str
    modes: tuple                # (cos j, sin j) for a pair, (j,) for one mode
    m: int                      # the azimuthal order: 1 an arrow, 2 an axis; 0 a single map
    variation: float            # rad: the largest departure over the grid from the field's mean


def emitter_xy(source, channel: int = 0) -> np.ndarray | None:
    """[E, 2] the used emitters' chip positions of ``source``: a `Calibration` (`beads_field_xy`) or a
    `LearnedPSF` (its bead table's ``field_x``, ``field_y``); None where it does not keep them."""
    if hasattr(source, "beads_field_xy"):
        xy = source.beads_field_xy()
    else:
        xy = [(b["field_x"], b["field_y"]) for b in getattr(source, "beads", None) or []
              if b.get("state") == "used" and "field_x" in b]
    return None if not xy else np.asarray(xy, float).reshape(-1, 2)


def field_maps(source, channel: int = 0, xy=None, points: int = MAP_POINTS) -> FieldMaps | None:
    """The `FieldMaps` of ``source`` (a `Calibration` or a `LearnedPSF`) over the bounding box of ``xy``
    ([E, 2] chip pixels; default its own emitters', `emitter_xy`), ``points`` along its longer side; None
    where the channel has no field or no emitter positions are known.

    Only over the emitters' span: the field is a polynomial, and beyond the data it extrapolates freely."""
    import torch

    from ..model.channels import view
    ch = source.model.channels[channel]
    field = getattr(ch, "field", None)
    xy = emitter_xy(source, channel) if xy is None else np.asarray(xy, float).reshape(-1, 2)
    if field is None or xy is None or not len(xy):
        return None
    lo, hi = xy.min(0), xy.max(0)
    span = np.maximum(hi - lo, 1.0)
    step = float(span.max()) / (points - 1)
    nx, ny = (max(2, int(round(s / step)) + 1) for s in span)
    gx, gy = np.linspace(lo[0], hi[0], nx), np.linspace(lo[1], hi[1], ny)
    X, Y = np.meshgrid(gx, gy)
    where = torch.as_tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float64)
    shared = source.shared() if callable(getattr(source, "shared", None)) else source.shared
    psf = ch.psf
    pupil = psf.pupil
    with torch.no_grad():
        p = {k: torch.as_tensor(v).detach().cpu().double() for k, v in view(shared, channel).items()}
        p = field.apply(p, pupil, where)
        learnt = p["pupil.phase"]                                                     # [E, n]
        basis = pupil.phase_basis.detach().cpu().double()
        phase = torch.tensordot(learnt, basis, dims=([-1], [0]))
        held = getattr(pupil, "held_phase_map", None)
        if held is not None:
            phase = phase + torch.as_tensor(held).detach().cpu().double()
        inside = (pupil.grid.aperture.detach().cpu().double() > 0).reshape(-1)
        flat = phase.reshape(phase.shape[0], -1)[:, inside]
        rms = (flat - flat.mean(1, keepdim=True)).pow(2).mean(1).sqrt()
    modes = list(pupil.phase_modes) + sorted(pupil.held_phase)
    values = np.concatenate([learnt.numpy(), np.tile([pupil.held_phase[j] for j in sorted(pupil.held_phase)],
                                                     (len(where), 1))], 1)
    order = np.argsort(modes)
    return FieldMaps(gx, gy, [int(modes[i]) for i in order], values[:, order].reshape(ny, nx, -1),
                     rms.numpy().reshape(ny, nx), [int(j) for j, _ in field.terms], xy,
                     float(psf.grid.optics.wavelength_nm))


def _family(n: int, m: int) -> str:
    names = {1: "coma", 2: "astigmatism", 3: "trefoil", 4: "quadrafoil"}
    first = {1: 3, 2: 2, 3: 3, 4: 4}
    k = (n - first[m]) // 2
    return ("" if k == 0 else f"{k + 1}{'nd' if k == 1 else 'rd' if k == 2 else 'th'} ") + names[m]


def map_items(maps: FieldMaps) -> list[MapItem]:
    """The panels the maps can show, every family of modes the field varies: a sine and a cosine mode of one
    radial and azimuthal order (|m| = 1, coma; 2, astigmatism) as one pair, drawn as an arrow or an axis;
    anything else as one map.  Largest variation first."""
    from ..optics.zernike import noll_to_nm, zernike_name
    varying = [j for j in maps.varying if j in maps.modes]
    nm = {j: noll_to_nm(j) for j in maps.modes}
    items, done = [], set()
    for j in varying:
        if j in done:
            continue
        n, m = nm[j]
        partner = next((k for k in maps.modes if k != j and nm[k][0] == n and nm[k][1] == -m), None)
        if m != 0 and abs(m) <= 2 and partner is not None:
            cos_j, sin_j = (j, partner) if m > 0 else (partner, j)
            c, s = maps.of(cos_j), maps.of(sin_j)
            dev = np.hypot(c - c.mean(), s - s.mean())
            items.append(MapItem(f"{_family(n, abs(m))} (Noll {min(j, partner)}, {max(j, partner)})",
                                 (cos_j, sin_j), abs(m), float(dev.max())))
            done |= {j, partner}
        else:
            c = maps.of(j)
            items.append(MapItem(f"{zernike_name(j)} (Noll {j})", (j,), 0, float(np.abs(c - c.mean()).max())))
            done.add(j)
    return sorted(items, key=lambda i: -i.variation)


def show_choices(maps: FieldMaps | None) -> tuple:
    """What the page's *show* offers: `SHOW_MOST` (first, the default), `SHOW_ALL`, then each family."""
    items = map_items(maps) if maps is not None else []
    return (SHOW_MOST, SHOW_ALL) + tuple(i.label for i in items)


def chosen_items(maps: FieldMaps, show: str | None = None) -> list[MapItem]:
    """The items ``show`` (one of `show_choices`) picks: the families varying most (`MOST`, `MOST_SHARE`),
    all of them, or the one named."""
    items = map_items(maps)
    show = show or SHOW_MOST
    if show == SHOW_ALL:
        return items
    if show == SHOW_MOST:
        top = items[0].variation if items else 0.0
        return [i for i in items if i.variation >= MOST_SHARE * top][:MOST]
    return [i for i in items if i.label == show]


def pair_vector(maps: FieldMaps, item: MapItem) -> tuple[np.ndarray, np.ndarray]:
    """([Y, X] magnitude, rad; [Y, X] orientation, radians from the chip's x axis towards its y) of a pair:
    c_cos cos(mφ) + c_sin sin(mφ) = A cos(m(φ − θ)), A = √(c_cos² + c_sin²), θ = atan2(c_sin, c_cos) / m.
    The pupil's φ is measured from its x towards its y (`optics.grid`: atan2(v, u)), the chip's axes; for
    astigmatism θ is the axis along which the wavefront is advanced, for coma the direction of its tail."""
    c, s = maps.of(item.modes[0]), maps.of(item.modes[1])
    return np.hypot(c, s), np.arctan2(s, c) / item.m
