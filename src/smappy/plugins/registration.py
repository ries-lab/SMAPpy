"""Measuring the transformation between two channels from a table of localizations.

(The obvious name for this file is ``register.py``, and it must not be: every
plugin module does ``from . import register`` for the decorator, and importing
a submodule binds it on the package under its own name.  A sibling called
``register`` therefore replaces the decorator with a module the moment this
file is imported, and the next ``@register(...)`` anywhere raises "module
object is not callable" -- in whichever module happens to be imported second,
which is nowhere near the cause.)

The calibration step of the 2D two-colour workflow.  Fit a split-frame movie
with a plain Gaussian, ignoring the split, and every molecule appears twice in
the same frame; `smappy.calibrate.transform` turns that into the projective map
the paired fit needs, and this plugin is the place to look at what it found
before trusting it.

What the three panels answer, in the order one asks them:

* **Did it find the channels at all?**  The vote: a sharp peak means one
  constant offset explains thousands of pairs, and a smear means it did not.
* **Is the map right, and right everywhere?**  The residual scatter, which is
  SMAP's ``dxy`` panel, with its spread in pixels.
* **Is it right everywhere I care about?**  Where the pairs are.  A
  transformation fitted in one corner is extrapolation in the other three, and
  no residual will say so -- only the coverage map will.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..calibrate.transform import RegisterSettings, register_channels
from . import Context, ParamInfo, Plugin, Result, param, register

TRANSFORM_FILTER = "Channel transformation (*_2ct.h5 *.h5 *.hdf5)"


@dataclass
class RegisterLocsSettings:
    """The registration, plus where to put it."""
    registration: RegisterSettings = field(default_factory=RegisterSettings)
    save: bool = param(True, label="save transformation")
    path: str = param("", label="file", kind="save_file",
                      file_filter=TRANSFORM_FILTER,
                      help="auto: <table>_2ct.h5 beside the localizations")
    overwrite: bool = param(False, advanced=True)


def chip_pixels(locs, report=None):
    """The table's positions as chip pixels, whatever unit it was saved in.

    A transformation is a statement about the camera, so it lives in chip
    pixels -- `dualfit.which_channel` and `combine_peaks` both read it that
    way.  A fit writes nm by default and drops the pixel columns, so they are
    recovered from the pixel size the conversion recorded rather than demanded
    back from the user.
    """
    if "x_pix" in locs and "y_pix" in locs:
        return np.asarray(locs["x_pix"], float), np.asarray(locs["y_pix"], float)
    if "x_nm" not in locs or "y_nm" not in locs:
        raise ValueError("the table has no positions to register")
    pixelsize = locs.metadata.get("pixelsize_nm")
    if not pixelsize:
        raise ValueError("the table is in nm and does not record its pixel size, "
                         "so it cannot be put back into chip pixels; re-run the "
                         "fit with units 'pixel+nm'")
    if report is not None:
        report(f"converting nm back to pixels at {pixelsize:g} nm/pixel")
    return (np.asarray(locs["x_nm"], float) / pixelsize,
            np.asarray(locs["y_nm"], float) / pixelsize)


def source_geometry(locs):
    """The frame shape and camera ROI the table was fitted from, if recorded."""
    camera = (locs.metadata.get("camera") or {})
    roi = camera.get("roi")
    shape = locs.metadata.get("image_shape")
    if shape is None and roi is not None:
        shape = (int(roi[3]), int(roi[2]))
    return shape, (tuple(roi) if roi is not None else None)


def default_transform_path(locs, fallback: Path) -> Path:
    source = (locs.metadata.get("source")
              or (locs.metadata.get("provenance") or {}).get("source"))
    if source:
        stem = Path(source).name.split(".")[0]
        return Path(source).parent / f"{stem}_2ct.h5"
    return fallback


@register("Analysis/Register/Calibrate transform")
class CalibrateChannelTransform(Plugin):
    """Find the two channels in a single-channel fit, and register them."""

    description = ("Measure the transformation between the two halves of a "
                   "split frame from localizations fitted over the whole "
                   "frame: no initial shift, no magnification and no split "
                   "position needed.")
    Settings = RegisterLocsSettings
    preview_help = ("run the registration and draw it -- the vote, the "
                    "residual scatter and where the pairs are -- without "
                    "writing a file")
    params = {
        "registration.layout": ParamInfo(
            label="layout", choices=(("auto", "detect"), "right-left",
                                     "right-left mirrored", "up-down",
                                     "up-down mirrored")),
        "registration.main_channel": ParamInfo(
            label="main channel",
            choices=(("auto", "the left or upper half"), "left", "right",
                     "upper", "lower")),
        "registration.model": ParamInfo(
            label="transformation", choices=(
                ("projective", "projective (8 coefficients)"),
                ("polynomial", "polynomial, order 3 (20)")),
            help="projective extrapolates gracefully; the polynomial describes "
                 "a distorted field far better where the pairs reach, and is "
                 "refused when they do not reach far enough"),
        "registration.split_position": ParamInfo(
            label="split position", unit="px", min=1,
            help="auto: measured from the pairs, in chip coordinates"),
        "registration.coarse_tolerance_px": ParamInfo(
            label="coarse matching", unit="px", min=0.1,
            help="first pass; must cover the rotation between the channels -- "
                 "3 degrees over 512 px is 27 px"),
        "registration.fine_tolerance_px": ParamInfo(
            label="fine matching", unit="px", min=0,
            help="second pass over clean matches only, which is where the "
                 "accuracy comes from; 0 skips it"),
        "registration.vote_bin_px": ParamInfo(label="vote bin", unit="px",
                                              min=0.1, advanced=True),
        "registration.vote_smooth_px": ParamInfo(label="vote smoothing",
                                                 unit="px", min=0.1, advanced=True),
        "registration.min_pairs": ParamInfo(label="minimum pairs", min=4,
                                            advanced=True),
        "registration.max_pairs": ParamInfo(label="pairs used in the fit", min=4,
                                            advanced=True),
        "registration.reprojection_threshold_px": ParamInfo(
            label="outlier threshold", unit="px", min=0.01, advanced=True),
        "registration.transform_axis_limit_px": ParamInfo(
            label="dx/dy limit", unit="px", min=0.01, advanced=True),
    }
    main = ("registration", "save", "path")

    # ------------------------------------------------------------ the work
    def _register(self, ctx: Context, settings):
        locs = ctx.locs[ctx.selection.mask] if ctx.selection is not None else ctx.locs
        if not len(locs):
            raise ValueError("no localizations selected")
        x, y = chip_pixels(locs, ctx.report)
        shape, roi = source_geometry(locs)
        if shape is None:
            # nothing recorded the frame; the localizations bound it well
            # enough for a geometry that is only ever compared against itself
            shape = (int(np.ceil(y.max())) + 1, int(np.ceil(x.max())) + 1)
        if "frame" not in locs:
            raise ValueError("registration pairs within a frame, and the table "
                             "has no frame column")
        # the fit's own error bars, so a dim pair counts for what it knows
        precision = (np.asarray(locs["xy_err_pix"], float)
                     if "xy_err_pix" in locs else None)
        return register_channels(x, y, np.asarray(locs["frame"]), shape, roi,
                                 settings=settings.registration,
                                 progress=ctx.report, precision=precision), locs

    def run(self, ctx: Context, settings) -> Result:
        result, locs = self._register(ctx, settings)
        out = None
        if settings.save:
            out = (Path(settings.path) if settings.path
                   else default_transform_path(locs, Path.cwd() / "channels_2ct.h5"))
            result.save(out, overwrite=settings.overwrite)
        return Result(text=self._summary(result, out), plot=self._plot(result),
                      data={"transform": result.transform, "result": result,
                            "path": out}, settings=settings)

    def preview(self, ctx: Context, settings) -> Result:
        result, _ = self._register(ctx, settings)
        return Result(text=self._summary(result, None), plot=self._plot(result),
                      data={"transform": result.transform, "result": result},
                      settings=settings)

    # ------------------------------------------------------------- showing
    @staticmethod
    def _summary(result, path) -> str:
        p = result.transform.parameters
        g = result.geometry
        lines = [f"{g['layout']}, split at {g['split_position']} px, "
                 f"main channel {g['main_channel']}",
                 f"{p['n_inliers']:,} of {p['n_pairs']:,} pairs kept from "
                 f"{p['n_localizations']:,} localizations in {p['n_frames']:,} frames",
                 f"residual dx {p['residual_dx_px']:.3f} px, "
                 f"dy {p['residual_dy_px']:.3f} px "
                 f"({p['residual_centre_px']:.3f} in the middle of the paired "
                 f"region, {p['residual_edge_px']:.3f} at its edge)"]
        if path is not None:
            lines.append(f"saved to {path}")
        return "; ".join(lines)

    @staticmethod
    def _plot(result):
        def plot(ax) -> None:
            figure = ax.figure
            ax.remove()
            figure.set_size_inches(15, 4.6)
            figure.set_layout_engine("constrained")
            vote_ax, residual_ax, cover_ax = figure.subplots(1, 3)
            _draw_vote(vote_ax, result)
            _draw_residuals(residual_ax, result)
            _draw_coverage(cover_ax, result)
            for panel in (vote_ax, residual_ax, cover_ax):
                panel.title.set_fontsize(9)
                panel.tick_params(labelsize=8)
                panel.xaxis.label.set_fontsize(8)
                panel.yaxis.label.set_fontsize(8)
        return plot


def _draw_vote(ax, result) -> None:
    """The pair-difference histogram, with the winner marked.

    Drawn on a square-root scale: the peak is orders of magnitude above the
    background by design, and on a linear scale the background -- which is
    what one is judging the peak against -- would be invisible.
    """
    vote = result.vote
    ex, ey = vote.edges
    if vote.histogram.size <= 1:
        ax.set_title("no vote")
        return
    ax.imshow(np.sqrt(vote.histogram.T), origin="lower", aspect="auto",
              extent=(ex[0], ex[-1], ey[0], ey[-1]), cmap="magma")
    ax.plot(vote.peak[0], vote.peak[1], "o", mfc="none", mec="cyan", ms=14, mew=1.5)
    summed = {"none": "", "x": " (x summed)", "y": " (y summed)"}[vote.mode]
    ax.set(title=f"pair vote{summed}: contrast {vote.score:.0f} from "
                 f"{vote.n_vectors:,} pairs",
           xlabel="dx (px)" if vote.mode != "x" else "x1 + x2 (px)",
           ylabel="dy (px)" if vote.mode != "y" else "y1 + y2 (px)")


def _draw_residuals(ax, result) -> None:
    """Where each kept pair lands after the fit -- SMAP's ``dxy`` panel."""
    inliers = result.accepted
    dx, dy = result.residuals[inliers, 0], result.residuals[inliers, 1]
    out = result.residuals[~inliers]
    span = max(float(np.percentile(np.abs(np.r_[dx, dy]), 99.5)) * 3, 0.05)
    ax.hexbin(dx, dy, gridsize=45, extent=(-span, span, -span, span),
              cmap="viridis", mincnt=1, linewidths=0)
    if len(out):
        ax.plot(out[:, 0], out[:, 1], ".", color="crimson", ms=2, alpha=.5,
                label=f"{len(out):,} rejected")
        ax.legend(fontsize=7, loc="upper right")
    ax.axhline(0, color="w", lw=.6, alpha=.5)
    ax.axvline(0, color="w", lw=.6, alpha=.5)
    ax.set(xlim=(-span, span), ylim=(-span, span),
           title=f"residual: dx {np.std(dx):.3f}, dy {np.std(dy):.3f} px "
                 f"({inliers.sum():,} pairs)",
           xlabel="x reference - x mapped (px)",
           ylabel="y reference - y mapped (px)")


def _draw_coverage(ax, result) -> None:
    """Which parts of the reference channel actually constrained the fit.

    A projective map is only measured where there are pairs; everywhere else
    it is an extrapolation, however small the residual looks.
    """
    points = result.reference_points
    inliers = result.accepted
    step = max(len(points) // 4000, 1)
    ax.plot(points[~inliers][:, 0], points[~inliers][:, 1], ".", ms=2,
            color="crimson", alpha=.6, label="rejected")
    kept = points[inliers][::step]
    ax.plot(kept[:, 0], kept[:, 1], ".", ms=2, color="#1f77b4", label="kept")
    g = result.geometry
    height, width = g["image_shape"]
    roi = next((s["roi"] for s in g.get("sources", []) if s["roi"]), None)
    x0, y0 = (roi[0], roi[1]) if roi else (0, 0)
    split = g["split_position"]
    if "right-left" in g["layout"]:
        ax.axvline(split, color="k", lw=1, ls="--")
    else:
        ax.axhline(split, color="k", lw=1, ls="--")
    ax.set(xlim=(x0, x0 + width), ylim=(y0, y0 + height),
           title="pairs used, in the reference channel (dashed: the split)",
           xlabel="x (chip px)", ylabel="y (chip px)")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="upper right", markerscale=3)
