"""4Pi in situ: a movie's own molecules, chosen from a Spline 4Pi fit and cut in every channel, for uiPSF's
in-situ 4Pi learning (`uipsf.workflows.fourpi_insitu`; docs/4pi.md §26).  No torch and no Qt here: the PSF
calibration window and a script call it alike.

* **The molecules** come from a fit of the movie with the bead model (*Spline 4Pi*): its table says where each
  molecule is, how bright, at which z.  `choose` keeps the bright, alone and in-range ones, spread over z by
  equal quotas in bins and over the movie at random, so that no height and no part of the movie dominates.
* **The boxes** are cut as the fitter cuts its own (`multichannel.channel_candidates`, `cut_channel_rois`):
  the molecule's position in channel 0's own pixels mapped into each channel by the layout, each box
  centred on the nearest pixel, in photons.  Each box's centre is uiPSF's ``roi_xy`` and the remainder in
  channel 0 its start x, y, so the model's maps start from the layout's -- the bead model's, with which the
  movie was fitted.
* **The backgrounds** start from each box's edge, per channel (the fitter's table has one for all).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..plugins import param


@dataclass
class FourPiInSituSettings:
    """Which molecules of the fit the in-situ learning takes, and how it learns (docs/4pi.md §26)."""
    molecules: int = param(3000, label="molecules", min=100,
                           help="how many molecules are learnt from, chosen over z and the movie")
    min_photons: float = param(0.0, label="min photons", min=0.0,
                               help="the dimmest molecule taken, all channels' photons; 0: the brighter half")
    roi_size: int = param(13, label="box", unit="px", min=9, help="each molecule's box per channel")
    z_margin_nm: float = param(100.0, label="z margin", unit="nm", min=0.0, advanced=True,
                               help="molecules this close to the model's z range's ends are left out")
    z_bins: int = param(8, label="z bins", min=1, advanced=True,
                        help="the molecules are spread over z in this many bins of equal quota")
    rounds: int = param(2, label="rounds", min=1, max=5, advanced=True,
                        help="localise, leave out the poor fits, learn: this many times")
    seed: int = param(0, label="seed", advanced=True, help="which molecules are drawn, where there are more")


def _column(locs, name):
    return np.asarray(locs[name], float)


def choose(locs, z_range_nm: float, settings: FourPiInSituSettings, frames=None) -> np.ndarray:
    """Indices of the molecules to learn from (module docstring): fitted to a finite position and phase,
    at least *min photons* (the median on 0), inside ±(``z_range_nm`` − *z margin*), with no other
    localization within a box in the same frame; then *molecules* of them spread over *z bins* equal bins.
    ``frames`` limits them to those frames (a set or a boolean function of the frame numbers)."""
    z = _column(locs, "z_env_nm" if "z_env_nm" in locs else "z_nm")
    x, y = _column(locs, "x_nm"), _column(locs, "y_nm")
    photons = _column(locs, "photons")
    frame = np.asarray(locs["frame"], np.int64)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(photons)
    if "phase_rad" in locs:
        ok &= np.isfinite(_column(locs, "phase_rad"))
    floor = settings.min_photons or float(np.median(photons[ok])) if ok.any() else 0.0
    ok &= photons >= floor
    limit = z_range_nm - settings.z_margin_nm
    ok &= np.abs(z) <= limit
    if frames is not None:
        ok &= frames(frame) if callable(frames) else np.isin(frame, list(frames))
    ok &= alone(x, y, frame, settings.roi_size * _pixel_nm(locs))
    index = np.flatnonzero(ok)
    rng = np.random.default_rng(settings.seed)
    if len(index) <= settings.molecules:
        return index
    edges = np.linspace(-limit, limit, settings.z_bins + 1)
    which = np.clip(np.digitize(z[index], edges) - 1, 0, settings.z_bins - 1)
    pools = [rng.permutation(index[which == b]) for b in range(settings.z_bins)]
    # equal quotas; what a thin bin cannot fill goes to the others
    sizes = np.array([len(p) for p in pools])
    quota, want = np.zeros(len(pools), int), settings.molecules
    while want > 0 and (quota < sizes).any():
        room = np.flatnonzero(quota < sizes)
        share = max(1, want // len(room))
        for b in room:
            take = min(share, sizes[b] - quota[b], want)
            quota[b] += take
            want -= take
    return np.sort(np.concatenate([p[:q] for p, q in zip(pools, quota)]))


def _pixel_nm(locs) -> float:
    px = (getattr(locs, "metadata", None) or {}).get("pixelsize_nm")
    if px is None:
        return 100.0
    return float(np.mean(px))


def alone(x, y, frame, distance) -> np.ndarray:
    """Which localizations have no other within ``distance`` (the same units) in their frame."""
    from scipy.spatial import cKDTree
    x, y, frame = np.asarray(x, float), np.asarray(y, float), np.asarray(frame)
    out = np.ones(len(x), bool)
    order = np.argsort(frame, kind="stable")
    bounds = np.flatnonzero(np.diff(frame[order])) + 1
    for group in np.split(order, bounds):
        if len(group) < 2:
            continue
        tree = cKDTree(np.c_[x[group], y[group]])
        n = np.array([len(v) for v in tree.query_ball_point(np.c_[x[group], y[group]], distance)])
        out[group[n > 1]] = False
    return out


@dataclass
class InSituMolecules:
    """The chosen molecules cut for learning: ``data`` (E, C, R, R) photons, ``roi_xy`` (E, C, 2) each box's
    centre in its channel's own pixels, ``start`` (xyz nm from channel 0's box centre, photons, backgrounds
    (E, C), phase), and where each came from: its ``frame`` and row ``index`` in the table."""
    data: np.ndarray
    roi_xy: np.ndarray
    start: dict
    frame: np.ndarray
    index: np.ndarray

    def __len__(self) -> int:
        return len(self.index)


def cut(source, layout, locs, index, camera, roi_size: int, pixel_nm, drift=None, progress=None) -> InSituMolecules:
    """The molecules ``index`` of the table ``locs`` cut from ``source`` (a `channels.ChannelSource`, or any
    source of whole frames, which is wrapped) in every channel of ``layout``, in photons by ``camera`` (module
    docstring).  ``pixel_nm`` (x, y); ``drift`` a `multichannel.ChannelDrift` the fit measured.  Molecules
    whose box leaves a channel are dropped."""
    from types import SimpleNamespace

    from ..camera import to_photons
    from ..channels import ChannelSource
    from ..multichannel import channel_candidates
    if not isinstance(source, ChannelSource):
        source = ChannelSource.wrap(source, layout.on_camera(camera.roi_offset, source.shape)
                                    if getattr(camera, "roi_offset", None) else layout)
    px = np.broadcast_to(np.asarray(pixel_nm, float), (2,))
    index = np.asarray(index)
    roi = np.asarray(getattr(camera, "roi_offset", (0, 0)) or (0, 0), float)
    chip = np.c_[_column(locs, "x_nm")[index] / px[0], _column(locs, "y_nm")[index] / px[1]] - roi
    own = source.layout.from_frame(0, chip)
    frame = np.asarray(locs["frame"], np.int64)[index]
    cands = SimpleNamespace(x=own[:, 0], y=own[:, 1], frame=frame)
    positions, residual, keep = channel_candidates(cands, source.layout, roi_size, drift)
    half = (roi_size - 1) // 2
    span = np.arange(-half, half + 1)
    C = source.layout.n_channels
    data = np.zeros((len(index), C, roi_size, roi_size), np.float32)
    order = np.argsort(frame, kind="stable")
    done = 0
    for f in np.unique(frame[keep]):
        rows = order[(frame[order] == f)]
        rows = rows[keep[rows]]
        image = to_photons(source.frame(int(f)), camera)                 # (C, h, w)
        for c in range(C):
            ys = positions[rows, c, 1][:, None] + span
            xs = positions[rows, c, 0][:, None] + span
            data[rows, c] = image[c][ys[:, :, None], xs[:, None, :]]
        done += len(rows)
        if progress is not None and done % 500 < len(rows):
            progress(f"cut {done} of {int(keep.sum())} molecules")
    edge = np.concatenate([data[..., 0, :], data[..., -1, :], data[..., 1:-1, 0], data[..., 1:-1, -1]], -1)
    background = np.median(edge, -1).clip(min=0.1)
    xyz = np.c_[residual[:, 0, 0] * px[0], residual[:, 0, 1] * px[1],
                _column(locs, "z_env_nm" if "z_env_nm" in locs else "z_nm")[index]]
    photons = _column(locs, "photons")[index]
    start = {"xyz": xyz[keep], "photons": photons[keep], "background": background[keep],
             "phase": np.zeros(int(keep.sum()))}
    return InSituMolecules(data[keep], positions[keep].astype(float), start, frame[keep], index[keep])
