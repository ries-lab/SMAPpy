"""The prefit for a two-channel training: Spline 3D 2C on blocks of the movie, and what it says about
the two channels (docs/decode/gui-plan.md, step 4).

The one-channel prefit (`training.prefit`) fits Spline 3D and `decode_smlm.prefit.estimates` turns
the fits into the training's ranges.  Here the fitter is Spline 3D 2C (`plugins.fit.DualSplineFit`),
each block grouped into blinks as there, and the same estimates are taken from the molecules' total
photons and the main channel's background.  Beside them, what only two channels have:

* **the kind** -- biplane or two colours -- from the calibration: a biplane's two planes come into
  focus at different depths (Fig. 1c's 360 nm apart), a colour pair's at the same one (Fig. 2a's 60 nm
  apart, the chromatic shift), `kind_of`;
* **each channel's background range**, as the one-channel rule takes it (`decode_smlm.prefit.background_range`);
* **the brighter channel**, which the network reads as its grid (plex-plan §16, Jonas: the bright one
  main), `brighter`;
* **the dyes** (colour): the peaks of the molecules' shares in channel 1 (`dye_shares`), which Fig. 2a's
  showed at 0.785 and 0.92.

Spline 3D 2C fits the photons of the two channels apart for colour and linked for biplane, as the
network will treat them.  The registration is moved by ``moved`` (columns, rows) first, as the
fitter moves it (`fit_two.moved_calibration`): a splitter moved since the beads leaves the prefit
pairing the wrong spots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FOCUS_APART_NM = 200.0        # planes in focus this far apart are a biplane's
PEAK_SHARE = 0.15             # a dye's peak at least this share of the highest
PEAK_APART = 0.05             # two dyes' shares at least this far apart
MAX_DYES = 4


def is_dual(path) -> bool:
    """Whether ``path`` is a SMAPpy two-channel calibration, by its header alone."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return f.attrs.get("format") == "smappy-dual-color-calibration"
    except (OSError, TypeError):
        return False


def focus_nm(spline) -> float:
    """The depth at which a spline's PSF is brightest (nm, as its z)."""
    peak = np.asarray(spline.psf).reshape(len(spline.psf), -1).max(1)
    return float((np.argmax(peak) - spline.z0) * spline.dz)


def kind_of(cal) -> str:
    """"biplane" where the two channels' PSFs come into focus `FOCUS_APART_NM` or more apart, else "colour"."""
    return "biplane" if abs(focus_nm(cal.main) - focus_nm(cal.secondary)) >= FOCUS_APART_NM else "colour"


def brighter(photons_ch0, photons_ch1) -> int:
    """The calibration's channel (0 its main half, 1 the other) with more photons, at the median."""
    return int(np.median(photons_ch1) > np.median(photons_ch0))


def dye_shares(share, bins: int = 100, smooth_bins: float = 2.0) -> list:
    """The shares in channel 1 the molecules gather at: the peaks of their smoothed histogram, at least
    `PEAK_SHARE` of the highest and `PEAK_APART` apart, the highest `MAX_DYES`, in ascending order."""
    from scipy.ndimage import gaussian_filter1d
    share = np.asarray(share, float)
    share = share[np.isfinite(share)]
    if len(share) < 20:
        return []
    h, edges = np.histogram(np.clip(share, 0, 1), bins, (0, 1))
    h = gaussian_filter1d(h.astype(float), smooth_bins, mode="nearest")
    centre = (edges[:-1] + edges[1:]) / 2
    top = np.flatnonzero((h[1:-1] > h[:-2]) & (h[1:-1] >= h[2:])) + 1
    top = [i for i in sorted(top, key=lambda i: -h[i]) if h[i] >= PEAK_SHARE * h.max()]
    kept = []
    for i in top:
        if all(abs(centre[i] - centre[j]) >= PEAK_APART for j in kept):
            kept.append(i)
        if len(kept) == MAX_DYES:
            break
    return sorted(round(float(centre[i]), 3) for i in kept)


@dataclass
class TwoChannels:
    """What the prefit says about the two channels, in the calibration's order."""
    kind: str                                   # "biplane" or "colour"
    background: list                            # per channel (low, high), photons a pixel
    brighter: int                               # the calibration's channel with more photons
    shares: np.ndarray = field(repr=False, default=None)   # colour: each molecule's share in channel 1
    dyes: list = field(default_factory=list)    # colour: the shares the molecules gather at


def prefit_two(movie, camera_settings, calibration, kind: str = "auto", moved=(0.0, 0.0), *,
               n_blocks: int = 10, block_frames: int = 50, n_frames: int | None = None, report=lambda text: None):
    """Spline 3D 2C on blocks of ``movie`` (the module's docstring): ``(prefit, two)``, a
    `training.prefit.Prefit` with total photons and the main channel's background, and `TwoChannels`."""
    import tempfile

    from ..calibrate.dual import (
        load_dual_color_calibration,
        save_dual_color_calibration,
    )
    from ..group import GroupSettings, group
    from ..locs import Localizations
    from ..pipeline import FitSettings as PipelineFitSettings
    from ..plugins import Context
    from ..plugins.fit import (
        DualModelSettings,
        DualSplineFit,
        DualSplineFitSettings,
        FinishSettings,
        OutputSettings,
        SourceSettings,
    )
    from ..training.prefit import GROUP_DX_NM, Prefit, blocks
    from .fit_two import biplane_total, moved_calibration
    cal = load_dual_color_calibration(calibration)
    kind = kind_of(cal) if kind == "auto" else kind
    if n_frames is None:
        n_frames = _source(movie).n_frames
    path = Path(calibration)
    with tempfile.TemporaryDirectory() as folder:
        if any(moved):
            path = Path(folder) / "moved_2ct.h5"
            save_dual_color_calibration(path, moved_calibration(cal, moved))
        locs, grouped, iterations = Localizations(), Localizations(), 0
        for start, stop in blocks(n_frames, n_blocks, block_frames):
            report(f"Spline 3D 2C: frames {start} to {stop}")
            settings = DualSplineFitSettings(source=SourceSettings(path=str(movie), start=start, stop=stop),
                                             camera=camera_settings,
                                             model=DualModelSettings(calibration=str(path), link_photons=kind == "biplane"),
                                             output=OutputSettings(save=False),
                                             # the fits alone: no colours assigned, no drift corrected
                                             finish=FinishSettings(assign_colors=False, fourpi_z=False))
            iterations = settings.fit.iterations if hasattr(settings, "fit") else PipelineFitSettings().iterations
            block = DualSplineFit().run(Context(), settings).locs
            if block is None or len(block) == 0:
                continue
            if "photons_ch0" in block and "photons_ch1" in block:   # colour: the total the estimates read
                block.columns["photons"] = np.asarray(block["photons_ch0"]) + np.asarray(block["photons_ch1"])
            elif kind == "biplane":
                # Spline 3D 2C's linked count is the molecule's total over both planes (`psf.signal_integral`);
                # netfit draws each plane's PSF with one count, the planes' shares in the PSFs: the
                # total over their light (`fit_two.biplane_total`)
                block.columns["photons"] = np.asarray(block["photons"]) / biplane_total(cal)
            if "background_ch0" in block and "background" not in block:
                block.columns["background"] = np.asarray(block["background_ch0"])
            blinks, _ = group(block, GroupSettings(dx=GROUP_DX_NM))
            if len(locs):
                block.columns["group_id"] = np.asarray(block["group_id"]) + int(np.asarray(locs["group_id"]).max())
            locs.extend(block)
            grouped.extend(blinks)
    if len(locs) == 0:
        raise ValueError("the prefit found no molecules: check the camera, the frames and the move since the beads")
    from decode_smlm.prefit import background_range
    bg = [tuple(float(v) for v in background_range(np.asarray(locs[f"background_ch{c}"]))) for c in (0, 1)]
    if kind == "colour":
        p0, p1 = np.asarray(locs["photons_ch0"], float), np.asarray(locs["photons_ch1"], float)
        shares = p1 / np.maximum(p0 + p1, 1e-9)
        two = TwoChannels(kind, bg, brighter(p0, p1), shares, dye_shares(shares))
    else:
        two = TwoChannels(kind, bg, 0)          # a biplane's planes share each molecule's photons
    rows, cols = _frame_shape(movie)
    fitted = sum(b - a for a, b in blocks(n_frames, n_blocks, block_frames))
    return Prefit(localizations=locs, blinks=grouped, area_px=float(rows * cols / 2), n_frames=int(fitted),
                  max_iterations=int(iterations)), two


@dataclass
class GaussianTwo:
    """What a Gaussian two-channel prefit found beside `TwoChannels`: the registration Gaussian 2D 2C fitted
    with (a `ChannelTransform`), and each fit's widths in the two channels."""
    transform: object
    sigma_ch0: np.ndarray = field(repr=False, default=None)
    sigma_ch1: np.ndarray = field(repr=False, default=None)


def prefit_two_gaussian(movie, camera_settings, channels, registration, *, n_blocks: int = 10,
                        block_frames: int = 50, n_frames: int | None = None, report=lambda text: None):
    """Gaussian 2D 2C on blocks of ``movie``, for a training on Gaussians (`calibrate.gaussian_psf`): ``(prefit,
    two, gaussian)``, as `prefit_two` gives them (two colours: the widths say nothing of a biplane), and
    `GaussianTwo`.  ``channels`` is the fitters' *channels* part (the split), ``registration`` Gaussian 2D 2C's
    *channel registration*: from this movie -- measured once, before the blocks, on frames spread over it --
    or a file."""
    import tempfile
    from dataclasses import replace

    from ..calibrate.transform import save_channel_transform
    from ..group import GroupSettings, group
    from ..locs import Localizations
    from ..plugins import Context
    from ..plugins.fit import (DualGaussianFit, DualGaussianFitSettings, DualGaussianModelSettings, FinishSettings,
                               OutputSettings, SourceSettings)
    from ..training.prefit import GROUP_DX_NM, Prefit, blocks

    def settings_for(source, reg):
        return DualGaussianFitSettings(source=source, camera=camera_settings, channels=channels,
                                       model=DualGaussianModelSettings(registration=reg),
                                       output=OutputSettings(save=False),
                                       finish=FinishSettings(assign_colors=False, fourpi_z=False))
    fitter = DualGaussianFit()
    if registration.map == "movie":
        report("Gaussian 2D 2C: measuring the channels' registration on this movie")
        found = fitter.measured(Context(), settings_for(SourceSettings(path=str(movie)), registration))
        transform = found.transform
    else:
        transform = registration.load()
    if n_frames is None:
        n_frames = _source(movie).n_frames
    locs, grouped, iterations = Localizations(), Localizations(), 0
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "prefit_2ct.h5"
        save_channel_transform(path, transform)
        fixed = replace(registration, map="file", path=str(path))
        for start, stop in blocks(n_frames, n_blocks, block_frames):
            report(f"Gaussian 2D 2C: frames {start} to {stop}")
            settings = settings_for(SourceSettings(path=str(movie), start=start, stop=stop), fixed)
            iterations = settings.fit.iterations
            block = fitter.run(Context(), settings).locs
            if block is None or len(block) == 0:
                continue
            block.columns["photons"] = np.asarray(block["photons_ch0"]) + np.asarray(block["photons_ch1"])
            block.columns["background"] = np.asarray(block["background_ch0"])
            blinks, _ = group(block, GroupSettings(dx=GROUP_DX_NM))
            if len(locs):
                block.columns["group_id"] = np.asarray(block["group_id"]) + int(np.asarray(locs["group_id"]).max())
            locs.extend(block)
            grouped.extend(blinks)
    if len(locs) == 0:
        raise ValueError("the prefit found no molecules: check the camera, the frames and the channels")
    from decode_smlm.prefit import background_range
    bg = [tuple(float(v) for v in background_range(np.asarray(locs[f"background_ch{c}"]))) for c in (0, 1)]
    p0, p1 = np.asarray(locs["photons_ch0"], float), np.asarray(locs["photons_ch1"], float)
    shares = p1 / np.maximum(p0 + p1, 1e-9)
    two = TwoChannels("colour", bg, brighter(p0, p1), shares, dye_shares(shares))
    rows, cols = _frame_shape(movie)
    fitted = sum(b - a for a, b in blocks(n_frames, n_blocks, block_frames))
    prefit = Prefit(localizations=locs, blinks=grouped, area_px=float(rows * cols / 2), n_frames=int(fitted),
                    max_iterations=int(iterations))
    return prefit, two, GaussianTwo(transform, np.asarray(locs["sigma_nm_ch0"], float),
                                    np.asarray(locs["sigma_nm_ch1"], float))


def _source(movie):
    from ..plugins.fit import SourceSettings, _open
    return _open(SourceSettings(path=str(movie)), watch=False)


def _frame_shape(movie):
    _, first = next(iter(_source(movie).frames(chunk=1, start=0, stop=1)))
    return np.asarray(first).shape[-2:]
