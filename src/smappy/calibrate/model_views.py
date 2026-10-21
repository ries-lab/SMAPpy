"""The calibration against the beads it came from: data and model side by side, on one contrast.

A PSF calibration is judged by looking at it next to the beads: a model that
misses the rings, a coma the beads do not have, a bead that is a double.  Both
methods of the PSF calibration window hand the same `ModelViews` to the same
pictures -- the *Model vs data* page (`draw_model_vs_data`) and the stack
browser -- so the bead average and the pupil model are compared alike.

* **Data**: the beads aligned onto the model and averaged; each bead on its
  own too, aligned the same way, so a slider can step through them.
* **Model**: what the calibration makes of the same beads, aligned the same
  way -- the residual is then the misfit only, never the alignment.

Both are photons per photon of the emitter, background removed, so data and
model share one colour scale.

**Bead average**: the data is its own average before smoothing (`raw_psf`),
the model the smoothed PSF the spline is made from (`calibration.psf`); a
bead is its stack shifted by the alignment's shifts and scaled as the average
scales it.  The model is the same for every bead.

**Pupil model**: each bead's model is the learnt forward model at that bead's
fitted position, photons and background (`Problem.expected`): the used beads
from the learning fit itself, the others localised on the final pupil.  The
alignment puts every slice where its emitter was -- z from the fitted
geometry (stage drift and z scale included), x and y from each channel's own
position (`ForwardModel.positions`, the transform included) -- by linear
interpolation onto the nominal slices, so the average is centred whatever
the beads' offsets.  On the slices learnt from (every n-th, `Batch.thinned`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: the z step the *Model vs data* page aims for between its planes: few enough to read, close enough to
#: follow an astigmatic PSF through focus
PLANE_STEP_NM = (100.0, 150.0)
#: at most this many planes on the page
MAX_PLANES = 9

SECTIONS = ("XY", "XZ", "YZ")
#: the background where a plane is blank: a bead's slices do not reach it
BLANK = (0.85, 0.85, 0.85, 1.0)


@dataclass
class ModelViews:
    """Data and model, per channel [z, y, x], on the planes at ``z_nm`` (the emitter's z of each)."""
    data: list                              # per channel: the aligned average of the used beads
    model: list                             # per channel: the model, aligned and averaged the same way
    z_nm: np.ndarray
    titles: list
    beads: dict = field(default_factory=dict)       # bead -> (data per channel, model per channel)
    states: dict = field(default_factory=dict)      # bead -> "used", "rejected", ...
    mirror_axis: Optional[int] = None

    def of(self, bead=None):
        """(data, model) per channel: the average, or the bead ``bead`` alone."""
        if bead is None or bead not in self.beads:
            return self.data, self.model
        return self.beads[bead]


# ------------------------------------------------------------------ the bead average

def _scaled(sample, reference):
    """``sample`` divided by its median ratio to ``reference`` over the reference's brightest quarter: the
    bead average's own amplitude match (`core.build_calibration`, normalize_and_score)."""
    bright = reference > np.quantile(reference, .75)
    valid = bright & np.isfinite(sample) & (np.abs(reference) > 1e-12)
    factor = np.median(sample[valid] / reference[valid]) if np.any(valid) else np.nan
    return sample / factor if np.isfinite(factor) and factor > 0 else sample


def average_views(result, dual: bool = False) -> ModelViews:
    """`ModelViews` of a bead average: one channel's `CalibrationResult`, or a dual one with ``dual``."""
    from .core import shift_volume
    if dual:
        r = result.registration
        data = [np.asarray(v, float) for v in r.channel_raw_psfs]
        model = [np.asarray(c.psf, float) for c in r.channel_calibrations]
        calibration, titles, mirror = r.channel_calibrations[0], ["main", "secondary"], \
            result.beads.geometry.get("mirror_axis_xy")
    else:
        r = result
        data, model = [np.asarray(r.raw_psf, float)], [np.asarray(r.calibration.psf, float)]
        calibration, titles, mirror = r.calibration, [""], None
    beads = r.beads
    p, crop = beads.settings.padding_px, r.z_crop_start
    nz = data[0].shape[0]
    views = ModelViews(data, model, calibration.z_index_to_nm(np.arange(nz)), titles, mirror_axis=mirror)
    volumes = np.asarray(beads.volumes)
    original = getattr(beads, "original_volumes", None)
    offsets = getattr(beads, "channel_offsets", None)
    for i in range(len(beads.records)):
        if not np.all(np.isfinite(r.shifts[i])):
            continue                                       # never aligned: nothing to show it as
        stacks = []
        for c in range(len(data)):
            if dual and original is not None and offsets is not None:
                v = shift_volume(original[i, c], r.shifts[i] + offsets[i, c])
            else:
                v = shift_volume(volumes[i, c] if dual else volumes[i], r.shifts[i])
            v = v[crop:v.shape[0] - crop, p:v.shape[1] - p, p:v.shape[2] - p]
            stacks.append(v)
        if stacks[0].shape != data[0].shape:
            continue
        # one factor per bead for both channels: the split of the light is part of what was learnt
        whole = _scaled(np.stack(stacks), np.stack(data))
        views.beads[i] = (list(whole), model)
        views.states[i] = "used" if r.accepted[i] else str(r.reasons[i])
    return views


# ------------------------------------------------------------------ the pupil model

def _onto(volume, z_of_slice, z_nm, shift_yx):
    """``volume`` [K, Y, X] resampled onto the planes ``z_nm``, its slices being at ``z_of_slice``, and
    shifted by ``shift_yx`` pixels; NaN where a plane lies outside the bead's own slices."""
    from scipy import ndimage
    order = np.argsort(z_of_slice)
    zs, vs = np.asarray(z_of_slice)[order], volume[order]
    out = np.full((len(z_nm),) + volume.shape[1:], np.nan)
    for k, z in enumerate(z_nm):
        if z < zs[0] or z > zs[-1]:
            continue
        j = int(np.clip(np.searchsorted(zs, z) - 1, 0, len(zs) - 2))
        w = (z - zs[j]) / max(zs[j + 1] - zs[j], 1e-9)
        plane = (1 - w) * vs[j] + w * vs[j + 1]
        out[k] = ndimage.shift(plane, shift_yx, order=1, mode="nearest")
    return out


def _bead_images(model, noise, batch, params):
    """What `pupil_views` aligns, numpy: (data, model) [E, C, K, Y, X] with the background removed and
    divided by the photons each channel receives, the emitter's z in each slice [E, K] and its (x, y) in
    each channel [E, C, K, 2], nm."""
    import torch

    from uipsf.solve import Problem
    problem = Problem(model, noise, batch)
    expected = problem.expected(params).double()
    b = batch.to(problem.device)
    with torch.no_grad():
        p = {k: v.to(problem.device) for k, v in params.physical().items()}
        xyz = model.geometry(p, b)                                         # [E, K, 3]
        N = model.intensity(p, b)                                          # [E, K]
        background = torch.as_tensor(model.background(p, b)).cpu().double()
        xy = torch.stack([model.positions(p, b, xyz, c)[..., :2] for c in range(model.n_channels)], 1)
        share = torch.stack([torch.as_tensor(model.share(p, c), device=N.device).expand(N.shape[0])
                             for c in range(model.n_channels)], 1)         # [E, C]
    scale = (share[:, :, None] * N[:, None]).clamp(min=1e-12).cpu().double()[..., None, None]
    data = batch.data.detach().cpu().double()
    return (((data - background) / scale).numpy(), ((expected - background) / scale).numpy(),
            xyz[..., 2].cpu().double().numpy(), xy.cpu().double().numpy())


def pupil_views(run) -> ModelViews:
    """`ModelViews` of a `pupil.PupilRun` (module docstring)."""
    import torch

    from uipsf.device import on_model_device
    from uipsf.workflows.beads import MAX_DZ_NM
    from uipsf.workflows.scoring import localise, shared_values
    cal, batch, noise = run.cal, run.rois.batch, run.rois.noise
    model = cal.model
    batch = on_model_device(batch, model).thinned(MAX_DZ_NM)
    used = list(cal.beads.used)
    parts = [(used, cal.result.params)]
    others = [r.bead for r in cal.beads.rows if r.bead not in used]
    if others:
        fit, _ = localise(model, shared_values(cal.result.params), batch.subset(torch.tensor(others)), noise)
        parts.append((others, fit.params))
    z_nm = batch.z_nm.detach().cpu().double().numpy()
    if z_nm.ndim > 1:
        z_nm = z_nm[0]
    z_nm = np.sort(z_nm)[::-1]                                  # as a spline's planes: the top first
    pixel = [model.channels[c].psf.grid.optics.pixel_size_nm for c in range(model.n_channels)]
    C = model.n_channels
    views = ModelViews([], [], z_nm, ["main", "secondary"][:C] if C > 1 else [""],
                       mirror_axis=None if run.rois.geometry is None else run.rois.geometry.get("mirror_axis_xy"))
    for ids, params in parts:
        data, expected, z_slice, xy = _bead_images(model, noise, batch.subset(torch.tensor(ids)), params)
        for j, bead in enumerate(ids):
            d, m = [], []
            for c in range(C):
                # the mean over the slices: the bead's place, the drift's wobble about it left in
                shift = (-float(np.mean(xy[j, c, :, 1])) / pixel[c][1], -float(np.mean(xy[j, c, :, 0])) / pixel[c][0])
                d.append(_onto(data[j, c], z_slice[j], z_nm, shift))
                m.append(_onto(expected[j, c], z_slice[j], z_nm, shift))
            views.beads[bead] = (d, m)
            views.states[bead] = cal.beads.rows[bead].state
    # only the planes every used bead reaches (`pupil.covered_z_nm`): at the ends a few beads alone would
    # stand for the average
    covered = np.all([np.isfinite(views.beads[b][0][0]).any(axis=(1, 2)) for b in used], axis=0)
    views.z_nm = z_nm[covered]
    for bead, (d, m) in views.beads.items():
        views.beads[bead] = ([v[covered] for v in d], [v[covered] for v in m])
    import warnings
    with warnings.catch_warnings():                             # a plane no bead reaches stays blank
        warnings.simplefilter("ignore", RuntimeWarning)
        for c in range(C):
            views.data.append(np.nanmean([views.beads[b][0][c] for b in used], axis=0))
            views.model.append(np.nanmean([views.beads[b][1][c] for b in used], axis=0))
    return views


# ------------------------------------------------------------------ the picture

def planes(z_nm, step_nm=None, count: int = MAX_PLANES) -> np.ndarray:
    """Indices of the planes to show: ``step_nm`` apart (auto: 100 nm, or 150 when that is too many for
    one row), symmetric about the plane nearest focus, at most ``count``."""
    z_nm = np.asarray(z_nm, float)
    dz = abs(float(np.median(np.diff(z_nm)))) if len(z_nm) > 1 else 1.0
    span = float(np.ptp(z_nm)) if len(z_nm) > 1 else 0.0
    if step_nm is None:
        step_nm = next((s for s in PLANE_STEP_NM if span / s + 1 <= count), span / max(count - 1, 1))
    every = max(1, int(round(step_nm / dz)))
    centre = int(np.argmin(np.abs(z_nm)))
    picked = sorted(set(range(centre % every, len(z_nm), every)))
    while len(picked) > count:
        picked = picked[1:-1]
    return np.asarray(picked, int)


def _section(volume, section):
    zc, yc, xc = (n // 2 for n in volume.shape)
    return {"XZ": volume[:, yc, :], "YZ": volume[:, :, xc]}[section]


def _rgba(image, residual: bool):
    """``image`` coloured on its own scale: 0 to its peak (inferno), or symmetric about 0 (RdBu) for a
    residual; NaN grey.  Each image on its own scale so that its shape shows whatever its brightness."""
    from matplotlib import colormaps
    image = np.asarray(image, float)
    peak = float(np.nanmax(np.abs(image))) if np.isfinite(image).any() else 0.0
    peak = peak or 1.0
    if residual:
        out = colormaps["RdBu_r"]((image / peak + 1) / 2)
    else:
        out = colormaps["inferno"](np.clip(image / peak, 0, 1))
    out[~np.isfinite(image)] = BLANK
    return out


def _cell(images, kinds, gap: int = 1):
    """One section as one picture: each channel its images side by side, ``gap`` white pixels apart, and
    the channels beside each other, four gaps apart, so that data and model are compared at a glance and
    a tall, narrow section leaves no width unused."""
    parts = []
    for channel in images:
        if parts:
            parts.append(np.ones((channel[0].shape[0], 4 * gap, 4)))
        for i, (image, kind) in enumerate(zip(channel, kinds)):
            if i:
                parts.append(np.ones((image.shape[0], gap, 4)))
            parts.append(_rgba(image, kind == "residual"))
    return np.concatenate(parts, 1)


def _colormap(name):
    """``name`` with NaN -- where a bead's slices do not reach -- drawn grey."""
    from matplotlib import colormaps
    return colormaps[name].with_extremes(bad=BLANK)


def residual_percent(data, model) -> float:
    """The largest |data − model| of a plane as a percentage of the plane's data peak: the number under
    each residual, as the stack browser gives it."""
    data = np.asarray(data, float)
    peak = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 0.0
    residual = data - np.asarray(model, float)
    worst = float(np.nanmax(np.abs(residual))) if np.isfinite(residual).any() else 0.0
    return 100.0 * worst / peak if peak else float("nan")


def draw_model_vs_data(figure, views: ModelViews, bead=None, section: str = "XY", step_nm=None,
                       average: bool = False) -> str:
    """XY planes ``step_nm`` apart (`planes`) as columns, and three rows almost touching: the data, the
    model and the residual (data − model), a set of three per channel, the channels side by side.  A plane's data and model share one contrast, the data's peak there, so a model
    too bright or too dim shows as well as one of the wrong shape; the residual is symmetric about 0 on its
    own scale, and under it how large it is, its largest |value| in % of the plane's data peak
    (`residual_percent`).  A side section (XZ, YZ) through the centre is one picture instead, data, model and
    residual side by side, the channels beside each other (`_cell`).  ``bead`` shows that bead alone unless ``average``.  Returns a status
    line."""
    if average:
        bead = None
    data, model = views.of(bead)
    which = "the average of the used beads" if bead is None or bead not in views.beads else \
        f"bead {bead} ({views.states.get(bead, '')})"
    kinds = ("data", "model", "residual")
    # the page's figure is drawn again on every choice: the rows' tight spacing, or the engine's own
    tight = section == "XY"
    engine = figure.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        engine.set(h_pad=0.005 if tight else 0.04167, w_pad=0.005 if tight else 0.04167,
                   hspace=0.0 if tight else 0.02, wspace=0.02)
    if section != "XY":
        # a side section stays one picture, data | model | residual side by side, each on its own contrast:
        # tall and narrow, its three read best next to each other
        ax = figure.subplots()
        cell = _cell([[_section(v, section) for v in (d, m, d - m)] for d, m in zip(data, model)], kinds)
        z = views.z_nm
        ax.imshow(cell, aspect="auto", extent=(0, cell.shape[1], z[-1], z[0]), interpolation="nearest")
        ax.set_xticks([])
        ax.set_ylabel("emitter z (nm)")
        legend = " | ".join(kinds) + (f", {views.titles[0]} left, {views.titles[1]} right" if len(data) > 1 else "") + \
            ", each on its own contrast"
        ax.set_title(f"{which}: {section} section through the centre\n{legend}", fontsize="small")
        return f"model vs data: {which}"
    picked = planes(views.z_nm, step_nm)
    columns = [(f"{views.z_nm[k] + 0.0:.0f} nm".replace("-0 ", "0 "), lambda v, k=k: v[k]) for k in picked]
    # two channels side by side, a panel each: stacked, their six rows of square pixels left the wide
    # figure mostly white
    panels = [figure] if len(data) == 1 else list(figure.subfigures(1, len(data), wspace=0.03))
    inferno, diverging = _colormap("inferno"), _colormap("RdBu_r")
    for c, (panel, d_volume, m_volume) in enumerate(zip(panels, data, model)):
        grid = panel.subplots(len(kinds), len(columns), squeeze=False)
        if len(data) > 1:
            panel.suptitle(views.titles[c], fontsize="small")
        for j, (title, cut) in enumerate(columns):
            d, m = np.asarray(cut(d_volume), float), np.asarray(cut(m_volume), float)
            peak = float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else 0.0
            r = d - m
            limit = float(np.nanmax(np.abs(r))) if np.isfinite(r).any() else 0.0
            images = ((d, dict(cmap=inferno, vmin=0, vmax=peak or 1.0)),
                      (m, dict(cmap=inferno, vmin=0, vmax=peak or 1.0)),
                      (r, dict(cmap=diverging, vmin=-(limit or 1.0), vmax=limit or 1.0)))
            for i, (image, options) in enumerate(images):
                ax = grid[i, j]
                ax.imshow(np.ma.masked_invalid(image), interpolation="nearest", **options)
                ax.set_xticks([])
                ax.set_yticks([])
                # square pixels leave spare height when the figure is wide: the three of a channel close
                # up on the model, the spare height around them rather than between them
                ax.set_anchor(("S", "C", "N")[i])
                if j == 0 and c == 0:
                    ax.set_ylabel(kinds[i], fontsize="small")
                if i == 0:
                    ax.set_title(title, fontsize="small")
                if i == 2:
                    ax.set_xlabel(f"±{residual_percent(d, m):.2g} %", fontsize="x-small", labelpad=1)
    figure.suptitle(f"{which}: XY planes; data and model on one contrast per plane; residual: largest "
                    "|data − model|, % of the plane's data peak", fontsize="small")
    return f"model vs data: {which}"
