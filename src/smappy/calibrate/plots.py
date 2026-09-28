"""The calibration window's figures, drawn onto a matplotlib `Figure`.

Each page is a function of the result, the beads the user has selected and
the ones they have excluded, and nothing else, so the window is only widgets
and a script or a test can draw the same page into a file.  A page draws and
returns; showing it (`canvas.draw_idle`) is the caller's.

The colours carry the same meaning on every page: green is a bead the PSF was
built from, orange (dual colour) one that only the transformation used, red
one that was rejected or excluded, gold what is selected.  A scatter artist
carries ``pair_ids``, the bead each point is, so a click on it can select that
row in the window's table.
"""
from __future__ import annotations

import numpy as np

from .dual import map_points


def stack_slice(volume, orientation, index):
    """One displayed plane out of a z,y,x stack."""
    if orientation == "XY":
        return volume[index]
    if orientation == "XZ":
        return volume[:, index, :]
    if orientation == "YZ":
        return volume[:, :, index]
    raise ValueError(f"unknown orientation {orientation!r}")


def residual_density(points):
    """Kernel density at each residual, for colouring a cloud by how crowded it is.

    Too few points, or points on a line, have no two-dimensional density;
    they are all given the same one rather than failing the page.
    """
    from scipy.stats import gaussian_kde
    points = np.asarray(points)
    if len(points) < 3 or np.linalg.matrix_rank(points-points.mean(0)) < 2:
        return np.ones(len(points))
    try:
        return gaussian_kde(points.T)(points.T)
    except np.linalg.LinAlgError:
        return np.ones(len(points))


def origin(result, stack):
    """Where acquisition ``stack`` sits on the camera chip, in pixels.

    Pooled plots put every acquisition in chip coordinates when the geometry
    says the ROIs are known; otherwise each starts at the origin.
    """
    beads = result.beads
    if beads.geometry["coordinate_system"] == "camera-chip":
        return np.array(beads.sources[stack]["roi"][:2])
    return np.array((0, 0))


def _sizes(axes, title=9):
    for ax in np.ravel(axes):
        ax.title.set_fontsize(title)
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)


# ------------------------------------------------------------ single channel
def draw_overview(figure, result, selected=(), excluded=()):
    """Single channel: the beads on their stack, brightness, the PSF, alignment."""
    r = result
    selected, excluded = set(selected), set(excluded)
    stack = r.beads.records[min(selected)]["stack"] if selected else 0
    figure.clear()
    axes = figure.subplots(2, 2)
    ax = axes[0, 0]
    im = r.beads.projections[stack]
    ax.imshow(im, cmap="gray", vmax=np.percentile(im, 99.5))
    for i, rec in enumerate(r.beads.records):
        if rec["stack"] == stack:
            good = r.accepted[i] and i not in excluded
            ax.plot(rec["x_px"], rec["y_px"], "o", mfc="none",
                    mec="gold" if i in selected else "lime" if good else "red",
                    mew=2.5 if i in selected else 1, ms=9 if i in selected else 6)
            ax.text(rec["x_px"]+2, rec["y_px"], str(i), color="yellow", fontsize=7)
    ax.set_title(f"Stack {stack+1}: bead IDs (green accepted, red rejected)", fontsize=9)
    psf = r.calibration.psf
    colors = ["green" if a and i not in excluded else "red"
              for i, a in enumerate(r.accepted)]
    xy_shift = np.hypot(r.shifts[:, 1], r.shifts[:, 2])
    brightness = np.array([rec["brightness_adu"] for rec in r.beads.records])
    axes[0, 1].scatter(xy_shift, brightness, c=colors)
    axes[0, 1].set(xlabel="Applied XY shift magnitude (pixels)", ylabel="Brightness (ADU)",
                   title="Brightness vs XY shift", yscale="log")
    if r.beads.records:
        lower = r.beads.records[0].get("brightness_lower_adu")
        upper = r.beads.records[0].get("brightness_upper_adu")
        if lower is not None and np.isfinite(lower):
            axes[0, 1].axhspan(lower, upper, color="green", alpha=.08)
    zmin, zmax = r.calibration.z_index_to_nm(np.array([psf.shape[0]-1, 0]))
    axes[1, 0].imshow(psf[:, psf.shape[1]//2, :], cmap="inferno", aspect="auto",
                      extent=(0, psf.shape[2]-1, zmin, zmax))
    axes[1, 0].set(xlabel="x (pixels)", ylabel="Emitter z (nm)", title="PSF xz")
    quality_ax = axes[1, 1]
    z_shift = r.shifts[:, 0]*r.calibration.dz
    quality_ax.scatter(z_shift, r.residuals, c=colors)
    quality_ax.set(xlabel="Applied z shift (nm)", ylabel="Relative shape residual",
                   title="Bead alignment")
    if selected:
        chosen = np.array(sorted(selected))
        axes[0, 1].scatter(xy_shift[chosen], brightness[chosen], s=100, facecolors="none",
                           edgecolors="gold", linewidths=2, zorder=5)
        quality_ax.scatter(z_shift[chosen], r.residuals[chosen], s=100, facecolors="none",
                           edgecolors="gold", linewidths=2, zorder=5)


def draw_fit_quality(figure, diagnostics, profiles, selected=()):
    """Single channel: the beads refitted, and their aligned midlines.

    Returns the one-line summary the window puts in its status bar.
    """
    import matplotlib.colors as mcolors
    d = diagnostics
    selected = set(selected)
    figure.clear()
    axes = figure.subplots(2, 2)
    bead_ids = sorted(set(d["bead_id"])-{-1})
    colors = []
    if bead_ids:
        hues = np.linspace(0, 1, len(bead_ids), endpoint=False)
        colors = mcolors.hsv_to_rgb(np.column_stack((hues, np.full(len(hues), .7),
                                                      np.full(len(hues), .8))))
    for color, bead_id in zip(colors, bead_ids):
        use = d["bead_id"] == bead_id
        order = np.argsort(d["expected_z_nm"][use])
        width = 2.5 if bead_id in selected else .8
        alpha = 1 if bead_id in selected else .75
        axes[0, 0].plot(d["expected_z_nm"][use][order], d["fitted_z_nm"][use][order],
                        color=color, linewidth=width, alpha=alpha)
        axes[0, 1].plot(d["expected_z_nm"][use][order], d["centered_error_nm"][use][order],
                        color=color, linewidth=width, alpha=alpha)
    average = d["bead_id"] == -1
    if np.any(average):
        order = np.argsort(d["expected_z_nm"][average])
        axes[0, 0].plot(d["expected_z_nm"][average][order], d["fitted_z_nm"][average][order],
                        color="black", linewidth=3, label="average bead stack")
        axes[0, 1].plot(d["expected_z_nm"][average][order],
                        d["centered_error_nm"][average][order], color="black", linewidth=3)
    limits = [min(d["expected_z_nm"]), max(d["expected_z_nm"])]
    axes[0, 0].plot(limits, limits, "k--")
    axes[0, 0].set(xlabel="Expected emitter z (nm)", ylabel="Fitted z (nm)",
                   title="In-sample bead refits")
    axes[0, 0].legend(loc="best", fontsize=7)
    axes[0, 1].set(xlabel="Expected emitter z (nm)",
                   ylabel="Error after bead offset removal (nm)", title="Centered fit error")
    profile_colors = {bead_id: color for bead_id, color in zip(bead_ids, colors)}
    for row, bead_id in enumerate(profiles["bead_id"]):
        color = profile_colors[int(bead_id)]
        width = 2.5 if bead_id in selected else .8
        alpha = 1 if bead_id in selected else .75
        axes[1, 0].plot(profiles["x_px"], profiles["x_profiles"][row], color=color,
                        linewidth=width, alpha=alpha)
        axes[1, 1].plot(profiles["z_nm"], profiles["z_profiles"][row], color=color,
                        linewidth=width, alpha=alpha)
    axes[1, 0].plot(profiles["x_px"], profiles["average_x"], color="black", linewidth=3,
                    label="average bead stack")
    axes[1, 1].plot(profiles["z_nm"], profiles["average_z"], color="black", linewidth=3)
    ylabel = "Intensity (calibration scale)"
    axes[1, 0].set(xlabel="x from center (pixels)", ylabel=ylabel,
                   title="Aligned bead lateral midlines")
    axes[1, 1].set(xlabel="Emitter z (nm)", ylabel=ylabel,
                   title="Aligned bead axial midlines")
    axes[1, 0].legend(loc="best", fontsize=7)
    for ax in axes[1]:
        ax.relim()
        ax.autoscale_view()
        ax.margins(x=.02, y=.05)
    individual = d["bead_id"] >= 0
    return (f"In-sample refit median absolute centered error: "
            f"{np.nanmedian(abs(d['centered_error_nm'][individual])):.1f} nm; "
            f"{np.count_nonzero(~d['fit_valid'][individual])} failed, "
            f"{np.count_nonzero(d['at_z_boundary'][individual])} at z boundary. "
            "This is a consistency check, not independent accuracy.")


# --------------------------------------------------------------- dual colour
def draw_pairs(figure, result, selected=(), excluded=()):
    """Dual colour: the pairs on their acquisition, and every pair on the main field."""
    r = result
    chosen, excluded = set(selected), set(excluded)
    si = r.beads.records[min(chosen)]["stack"] if chosen else 0
    figure.clear()
    axes = figure.subplots(1, 2)
    axes[0].imshow(r.beads.projections[si], cmap="gray",
                   vmax=np.percentile(r.beads.projections[si], 99.5))
    for i, rec in enumerate(r.beads.records):
        color = ("gold" if i in chosen else "red" if i in excluded else
                 "limegreen" if r.accepted[i] else
                 "darkorange" if r.transform_accepted[i] else "red")
        if rec["stack"] == si:
            xy = np.array([c["full_image_xy"] for c in rec["channel_records"]])
            line, = axes[0].plot(*xy.T, "o-", color=color, mfc="none", lw=.5, picker=5)
            line.pair_ids = np.array([i, i])
            axes[0].text(*xy[0], str(i), color=color, fontsize=7)
        artist = axes[1].scatter(*r.beads.main_points[i], c=color,
                                 s=45 if i in chosen else 20, picker=5)
        artist.pair_ids = np.array([i])
    for ch, ids in enumerate(r.beads.unmatched):
        for i in ids:
            rec = r.beads.channels[ch].records[i]
            xy = np.array(rec["full_image_xy"])
            if rec["stack"] == si:
                axes[0].plot(*xy, "+", color="cyan")
            if ch == 0:
                axes[1].plot(*(xy+origin(r, rec["stack"])), "+", color="dodgerblue", ms=9)
    axes[0].set_title(f"Acquisition {si+1}: paired / unpaired (cyan)")
    axes[1].set(title="Main FoV: PSF + transform (green), transform only (orange)\n"
                      "Rejected (red), unpaired (+)",
                xlabel="Main x (pixels)", ylabel="Main y (pixels)")
    h, w = r.beads.geometry["image_shape"]
    split = r.beads.geometry["split_position"]
    last = r.beads.settings.main_channel in ("right", "lower")
    horizontal = "right-left" in r.beads.settings.layout
    xlim = (split if last else 0, w if last else split) if horizontal else (0, w)
    ylim = (0, h) if horizontal else (split if last else 0, h if last else split)
    origins = np.array([origin(r, i) for i in range(len(r.beads.sources))])
    axes[1].set_xlim(origins[:, 0].min()+xlim[0], origins[:, 0].max()+xlim[1])
    axes[1].set_ylim(origins[:, 1].max()+ylim[1], origins[:, 1].min()+ylim[0])
    axes[1].set_aspect("equal")


def _mapped(result, excluded):
    """Every pair, whether the transformation used it, and its secondary mapped."""
    r = result
    ids = np.arange(len(r.beads.records))
    target = map_points(r.calibration.transformation, r.beads.secondary_points)
    good = np.array([r.transform_accepted[i] and i not in excluded for i in ids], dtype=bool)
    return ids, good, target


def draw_transformation(figure, result, selected=(), excluded=()):
    """Dual colour: the residuals of the two rounds of the transformation fit."""
    r = result
    excluded = set(excluded)
    ids, good, target = _mapped(r, excluded)
    delta = target-r.beads.main_points
    figure.clear()
    axes = figure.subplots(1, 2)
    for subset, accepted in ((ids[good], True), (ids[~good], False)):
        if not len(subset):
            continue
        if accepted:
            density = residual_density(delta[subset])
            order = np.argsort(density)
            subset = subset[order]
            artist = axes[0].scatter(*delta[subset].T, c=density[order], cmap="viridis",
                                     picker=5, label="Used (density)")
        else:
            artist = axes[0].scatter(*delta[subset].T, facecolors="none", edgecolors="red",
                                     picker=5, label="Rejected")
        artist.pair_ids = subset
    axes[0].axhline(0, color="gray", lw=.5)
    axes[0].axvline(0, color="gray", lw=.5)
    axes[0].legend(fontsize=8)
    spread = np.std(delta[ids[good]], axis=0, ddof=1) if good.sum() > 1 else [np.nan, np.nan]
    axes[0].set(title=f"Round 2 residuals\nσx={spread[0]:.3f}, σy={spread[1]:.3f} px",
                xlabel="dx (pixels)", ylabel="dy (pixels)")
    chosen = sorted(i for i in selected if 0 <= i < len(ids))
    if chosen:
        axes[0].scatter(*delta[chosen].T, s=110, facecolors="none", edgecolors="gold", lw=2)
    delta1 = r.transform_fit.round1_dxdy
    artist = axes[1].scatter(*delta1[ids].T, c=np.where(good, "green", "red"), picker=5)
    artist.pair_ids = ids
    limit = r.beads.settings.transform_axis_limit_px
    axes[1].plot([-limit, limit, limit, -limit, -limit],
                 [-limit, -limit, limit, limit, -limit], "k--", lw=.8)
    axes[1].axhline(0, color="gray", lw=.5)
    axes[1].axvline(0, color="gray", lw=.5)
    axes[1].set(title=f"Round 1 screening\n|dx|, |dy| ≤ {limit:g} px",
                xlabel="dx (pixels)", ylabel="dy (pixels)")
    if chosen:
        axes[1].scatter(*delta1[chosen].T, s=110, facecolors="none", edgecolors="gold", lw=2)
    # A physical display floor avoids autoscaling numerical roundoff to a
    # misleadingly huge cloud for effectively exact synthetic transforms.
    for ax, values in zip(axes, (delta[ids], delta1[ids])):
        extent = (max(limit*1.15, float(np.max(np.abs(values)))*1.1)
                  if len(values) else limit*1.15)
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent))
        ax.set_aspect("equal", adjustable="box")
        ax.title.set_fontsize(10)


def draw_field(figure, result, selected=(), excluded=()):
    """Dual colour: the two channels overlaid, and the shape residual across the field."""
    r = result
    excluded = set(excluded)
    ids, good, target = _mapped(r, excluded)
    figure.clear()
    axes = figure.subplots(1, 2)
    axes[0].scatter(*r.beads.main_points[ids].T, facecolors="none", edgecolors="black",
                    label="Main")
    artist = axes[0].scatter(*target[ids].T, marker="+", c=np.where(good, "green", "red"),
                             picker=5, label="Mapped ch. 2")
    artist.pair_ids = ids
    for ch, indices in enumerate(r.beads.unmatched):
        for i in indices:
            rec = r.beads.channels[ch].records[i]
            xy = np.array(rec["full_image_xy"])+origin(r, rec["stack"])
            if ch:
                xy = map_points(r.calibration.transformation, xy[None])[0]
            axes[0].plot(*xy, "x", color="magenta" if ch else "dodgerblue")
    axes[0].set(title="Overlay; unpaired ×\nMain (blue), secondary (magenta)",
                xlabel="Main x (pixels)", ylabel="Main y (pixels)")
    axes[0].legend(fontsize=7)
    valid = ids[np.isfinite(r.residuals[ids]) & np.array([
        r.beads.records[i]["brightness_accepted"] and i not in excluded for i in ids],
        dtype=bool)]
    if len(valid):
        artist = axes[1].scatter(*r.beads.main_points[valid].T, c=r.residuals[valid],
                                 cmap="magma", picker=5)
        artist.pair_ids = valid
        figure.colorbar(artist, ax=axes[1], label="Joint shape residual")
        psf_ids = valid[r.accepted[valid]]
        axes[1].scatter(*r.beads.main_points[psf_ids].T, s=65, facecolors="none",
                        edgecolors="limegreen", label="PSF accepted")
        axes[1].legend(fontsize=7)
    coverage = r.calibration.parameters["coverage_area_px2"]
    axes[1].set(title=f"Shape across FoV\nPooled hull (px²): transform "
                      f"{coverage['transformation']:.0f}\nPSF {coverage['psf']:.0f}",
                xlabel="Main x (pixels)", ylabel="Main y (pixels)")
    chosen = sorted(i for i in selected if 0 <= i < len(ids))
    for ax in axes:
        if chosen:
            ax.scatter(*r.beads.main_points[chosen].T, s=110, facecolors="none",
                       edgecolors="gold", lw=2)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.title.set_fontsize(10)


def draw_paired_fit_quality(figure, result, diagnostics, profiles, selected=()):
    """Dual colour: the pairs refitted with the global fit, and both channels' midlines."""
    d = diagnostics
    selected = set(selected)
    figure.clear()
    axes = figure.subplots(2, 4)
    for i in np.unique(d["bead_id"]):
        use = d["bead_id"] == i
        order = np.argsort(d["expected_z_nm"][use])
        style = (dict(color="black", lw=2.5, zorder=5) if i == -1 else
                 dict(lw=2 if i in selected else 0.7, alpha=0.8))
        for column, key in ((0, "fitted_z_nm"), (1, "centered_error_nm"), (2, "ratio")):
            axes[0, column].plot(d["expected_z_nm"][use][order], d[key][use][order], **style)
    limits = [float(np.min(d["expected_z_nm"])), float(np.max(d["expected_z_nm"]))]
    axes[0, 0].plot(limits, limits, "k--", lw=0.5)
    axes[0, 0].set(title="two-channel fit: z", xlabel="expected z (nm)",
                   ylabel="fitted z (nm)")
    axes[0, 1].axhline(0, color="gray", lw=0.5)
    axes[0, 1].set(title="centered z error", xlabel="expected z (nm)", ylabel="error (nm)")
    if result.calibration.parameters.get("secondary_main_brightness_ratio"):
        # the splitter's ratio is divided out in the link, so a correctly
        # fitted pair sits at one half, not at the raw brightness ratio
        axes[0, 2].axhline(0.5, color="gray", lw=0.5)
    # a fraction, so the axis is the whole fraction: left to itself
    # matplotlib would show a 1e-6 offset around a constant ratio and say
    # nothing about how well the colour actually separates
    axes[0, 2].set(title="fitted photon ratio", xlabel="expected z (nm)",
                   ylabel="secondary / total", ylim=(0, 1))
    error = d["centered_error_nm"][np.isfinite(d["centered_error_nm"]) & (d["bead_id"] >= 0)]
    if len(error):
        axes[0, 3].hist(error, bins=min(40, max(8, len(error)//10)), color="steelblue")
        axes[0, 3].axvline(0, color="gray", lw=0.5)
        axes[0, 3].set(title=f"z error: {np.std(error):.1f} nm rms",
                       xlabel="centered z error (nm)", ylabel="bead planes")
    else:
        axes[0, 3].set_axis_off()

    for ch, name in enumerate(("main", "secondary")):
        channel = profiles[ch]
        # the profiles come back in native camera orientation, so the model
        # has to be the native one too, not the mirrored twin the
        # registration worked in
        psf = (result.calibration.main if ch == 0 else result.calibration.secondary).psf
        models = (("x", psf[len(psf)//2, psf.shape[1]//2]),
                  ("z", psf[:, psf.shape[1]//2, psf.shape[2]//2]))
        for offset, (axis, model) in enumerate(models):
            ax = axes[1, 2*ch+offset]
            coord = channel[axis+("_px" if axis == "x" else "_nm")]
            for i, curve in zip(channel["bead_id"], channel[axis+"_profiles"]):
                ax.plot(coord, curve, lw=2 if i in selected else 0.6, alpha=0.6)
            ax.plot(coord, channel["average_"+axis], "k", lw=2, label="average")
            ax.plot(coord, model, "--", color="royalblue", lw=2, label="spline")
            ax.set(title=f"{name} {axis} profile",
                   xlabel=axis+(" (pixels)" if axis == "x" else " (nm)"),
                   ylabel="intensity (calibration scale)")
            ax.legend(fontsize=7)
            ax.margins(x=0.02, y=0.05)
    _sizes(axes)
    figure.suptitle("beads refitted with the two-channel global fit: "
                    "one shared z per pair; black is the averaged pair", fontsize=9)


# -------------------------------------------------------------------- both
def draw_beads(figure, result, selected=(), excluded=(), dual=False):
    """Every bead's brightness, shift, residual and correlation, pooled."""
    r = result
    selected, excluded = set(selected), set(excluded)
    ids = np.arange(len(r.beads.records))
    colors = np.array(["red" if i in excluded else "limegreen" if r.accepted[i]
                       else "darkorange" if dual and r.transform_accepted[i]
                       else "red" for i in ids])
    figure.clear()
    axes = figure.subplots(2, 2).ravel() if dual else figure.subplots(1, 2)
    xy = np.hypot(r.shifts[:, 1], r.shifts[:, 2])

    def scatter(ax, x, y, log=False):
        x, y = np.asarray(x), np.asarray(y)
        valid = np.isfinite(x) & np.isfinite(y)
        if log:
            valid &= y > 0
            ax.set_yscale("log")
        artist = ax.scatter(x[valid], y[valid], c=colors[valid], picker=5)
        artist.pair_ids = ids[valid]
        chosen = valid & np.array([i in selected for i in ids], dtype=bool)
        ax.scatter(x[chosen], y[chosen], s=100, facecolors="none",
                   edgecolors="gold", linewidths=2, zorder=5)

    for ch in range(2 if dual else 1):
        records = ([rec["channel_records"][ch] for rec in r.beads.records]
                   if dual else r.beads.records)
        scatter(axes[ch], xy, [rec["brightness_adu"] for rec in records], log=True)
        if records:
            lower, upper = (records[0].get(key)
                            for key in ("brightness_lower_adu", "brightness_upper_adu"))
            if lower is not None and upper is not None and np.isfinite([lower, upper]).all():
                axes[ch].axhspan(lower, upper, color="green", alpha=.08)
        name = ("Main" if ch == 0 else "Secondary")+" brightness" if dual else "Brightness"
        axes[ch].set(title=name+" vs XY shift", ylabel="Brightness (ADU)",
                     xlabel=("Shared" if dual else "Applied")+" XY shift (pixels)")
    ax = axes[2 if dual else 1]
    scatter(ax, r.shifts[:, 0]*r.calibration.dz, r.residuals)
    ax.set(title="Bead alignment", xlabel="Applied z shift (nm)",
           ylabel="Relative shape residual")
    if dual:
        scatter(axes[3], r.correlations, r.residuals)
        axes[3].set(title="Shape and correlation", xlabel="Joint correlation",
                    ylabel="Relative shape residual")
    _sizes(axes)
    figure.suptitle("All acquisitions · green: PSF" +
                    (" · orange: transformation only" if dual else "") +
                    " · red: rejected", fontsize=9)
