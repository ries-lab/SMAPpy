"""What a fit of one emitter in two channels of one chip needs beyond the channel layer.

The fit itself is `multichannel.MultiChannelEngine` on `channels.ChannelLayout` (docs/4pi.md §9): the
halves are cut into two channel images, each searched alone and the peaks paired (`paired_candidates`),
each pair cut in both channels and fitted at once.  Until 2026-10 this module did that on the frame as
the camera delivered it -- one search over both halves, SMAP's ``PeakCombiner``, ROIs cut in chip
pixels; the layer gave the same numbers away from the seam (docs/4pi.md §9) and that path is gone.
What stays is what both paths had to agree on:

* **The link** (`LINK_XYZ`, `PARAMETERS`): which parameters the channels share.
* **The pairing tolerance** (`MATCH_PIXELS`) and the mutual-nearest match (`_match`), SMAP's
  ``matchlocs``, which the registration and the layer's pairing both use.
* **The split-frame geometry** (`split_axis`, `reference_is_first`, `which_channel`): how a
  registration records which half is which, read by the registration and the simulator.
* **The table** (`PairedROIs`, `paired_to_localizations`): the global fit's output as localizations.

Two points of the design hold on the layer unchanged.  **The candidate list is the union, not the
intersection**: a molecule seen in only one channel is still fitted, its partner ROI cut where the
registration puts it.  **Nothing is resampled**: each ROI is cut at an integer pixel and the sub-pixel
remainder goes to the fitter as an offset in its link, which keeps the Poisson likelihood true.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .columns import add_xy_err
from .detect import Candidates

# (x, y, photons, background, z): the fitter's parameter order everywhere
PARAMETERS = ("x", "y", "photons", "background", "z")
LINK_XYZ = (True, True, False, False, True)
"""x, y and z shared, photons and background free.

GlobLoc's ``shared_linkXYZ``: it buys the sqrt(2) in z from having both
channels while leaving the photon split free, which is what carries the colour
in a ratiometric two-colour experiment.  Linking everything is more precise
still and right for biplane, but then there is no colour to read.
"""
MATCH_PIXELS = 4.0
"""How far apart the same molecule may look in the two channels.

SMAP's ``matchlocs(..., pixelsize*4)``.  It is a candidate-pairing tolerance,
not a registration accuracy: the transformation is good to a fraction of a
pixel, but a dim peak's pixel maximum wanders.
"""


@dataclass
class PairedROIs:
    """ROIs of the same emitters in every channel, and how they are linked.

    ``images`` is ``(n, channels, roisize, roisize)`` and ``link`` is
    ``(n, 2, channels, 5)`` -- the offset then the factor of each parameter in
    each channel, ready for ``_fit3d.fit_cspline_global``.
    """

    images: np.ndarray
    link: np.ndarray
    candidates: Candidates          # the reference channel's integer position
    roisize: int
    residual: np.ndarray            # (n, channels, 2) sub-pixel dx, dy

    def __len__(self) -> int:
        return self.images.shape[0]

    @property
    def half(self) -> int:
        return (self.roisize - 1) // 2

    def to_image_x(self, x_in_roi: np.ndarray) -> np.ndarray:
        """A fitted x inside the reference ROI, back in image coordinates.

        The global x is the reference channel's ROI x, so its own sub-pixel
        residual has to be added back -- SMAP does this after the fit
        (``P(:,1) = P(:,1) + channelshift(1,1,:)``).  The pairing leaves
        that residual at zero for the reference channel, as ``PeakCombiner``
        does, so this is normally a no-op; it is here because the general form
        of the link allows it not to be.
        """
        return (np.asarray(x_in_roi, float) - self.half + self.candidates.x
                + self.residual[:, 0, 0])

    def to_image_y(self, y_in_roi: np.ndarray) -> np.ndarray:
        return (np.asarray(y_in_roi, float) - self.half + self.candidates.y
                + self.residual[:, 0, 1])


# ----------------------------------------------------------------- geometry
def split_axis(geometry: dict) -> int:
    """The image axis the two channels are laid out along: 0 = y, 1 = x."""
    return 1 if "right-left" in geometry["layout"] else 0


def reference_is_first(geometry: dict) -> bool:
    """Whether the main channel is the low half of the split axis."""
    return geometry["main_channel"] in ("left", "upper")


def which_channel(x: np.ndarray, y: np.ndarray, geometry: dict) -> np.ndarray:
    """True where a position falls in the main (reference) half of the chip.

    The registration's question about a chip position; ``PeakCombiner`` asked
    the transformation through ``getRef``, the geometry answers it directly.
    """
    along = x if split_axis(geometry) == 1 else y
    low = along < geometry["split_position"]
    return low if reference_is_first(geometry) else ~low


def _linearise(mapping, x: np.ndarray, y: np.ndarray,
               step: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """The local dx2/dx1 and dy2/dy1 of a map, by a finite difference.

    The link the fitter takes is diagonal -- a factor per coordinate, no cross
    term -- so this is the part of the transformation it can represent.  For a
    splitter the map is nearly a translation with a scale near one, and what is
    left over (the rotation) is small enough to sit inside the fit's residual,
    exactly as in GlobLoc.

    ``mapping`` is any callable from positions to positions, which is the whole
    reason this is a finite difference and not an analytic derivative: a
    projective map and a polynomial one are equally easy to differentiate this
    way, and the fitter never learns which it was given.
    """
    here = mapping(np.c_[x, y])
    along_x = mapping(np.c_[x + step, y])
    along_y = mapping(np.c_[x, y + step])
    return ((along_x[:, 0] - here[:, 0]) / step,
            (along_y[:, 1] - here[:, 1]) / step)


def _match(a: np.ndarray, b: np.ndarray, tolerance: float):
    """Mutually nearest pairs within ``tolerance`` -- SMAP's ``matchlocs``."""
    if not len(a) or not len(b):
        return np.empty(0, int), np.empty(0, int)
    distance, nearest = cKDTree(b).query(a)
    _, back = cKDTree(a).query(b)
    take = (distance <= tolerance) & (back[nearest] == np.arange(len(a)))
    return np.flatnonzero(take), nearest[take]


# ------------------------------------------------------------------ the table
def paired_to_localizations(result, pairs: PairedROIs, model, cam) -> "Localizations":
    """Raw global-fit output as a localization table, in camera pixels.

    `locs.fit_to_localizations` for the paired case: x and y belong to the
    emitter, while the photons and background may be one number or one per
    channel, and ``ratio`` -- the fraction of the photons in the last channel
    -- is the colour a ratiometric splitter measures.

    The model's fifth parameter follows the model: z for a spline, and for a
    Gaussian the width, which is per channel because the two halves of a
    splitter see different wavelengths.  Nothing above this line knows the
    difference -- pairing, cutting and the link are the same either way -- and
    this is the one place the two workflows part.
    """
    from .locs import Localizations

    p = model.unpack(result, pairs.link)
    excess = cam.excess_noise
    roi_x, roi_y = cam.roi_offset

    cols = {
        "frame": pairs.candidates.frame.astype(np.int64),
        "x_pix": pairs.to_image_x(p["x_roi"]) + roi_x,
        "y_pix": pairs.to_image_y(p["y_roi"]) + roi_y,
        "photons": p["photons"] * excess,
        "background": p["background"] * excess,
        "ratio": p["ratio"],
        "x_err_pix": p["x_err_pix"], "y_err_pix": p["y_err_pix"],
        "photons_err": p["photons_err"] * excess,
        "background_err": p["background_err"] * excess,
        "logl": result.logl,
        "logl_rel": result.logl * excess / (pairs.roisize ** 2 * model.n_channels),
        "peak_x_pix": pairs.candidates.x + roi_x,
        "peak_y_pix": pairs.candidates.y + roi_y,
        "iterations": result.iterations.astype(np.int32),
    }
    if model.is_3d:
        cols["z_nm"], cols["z_err_nm"] = p["z_nm"], p["z_err_nm"]
        # a 4Pi model's envelope z and its phase (`psf.Global4PiPSF`)
        for name in ("z_env_nm", "z_env_err_nm",
                     *(c for pair in getattr(model, "extra", ()) for c in pair)):
            if name in p:
                cols[name] = p[name]
    else:
        cols["sigma_pix"], cols["sigma_err_pix"] = p["sigma_pix"], p["sigma_err_pix"]
    for channel in range(model.n_channels):
        # the per-channel errors travel with the counts: a colour assignment
        # downstream weighs the split by how well each half was measured
        for name in (f"photons_ch{channel}", f"background_ch{channel}",
                     f"photons_err_ch{channel}", f"background_err_ch{channel}",
                     f"sigma_pix_ch{channel}", f"sigma_err_pix_ch{channel}"):
            if name in p:
                # a width is already in pixels; only the counts carry the gain
                scale = 1.0 if name.startswith("sigma") else excess
                cols[name] = p[name] * scale
    # Unlinked, each channel fitted its own position, and the partner's was
    # dropped: the table has one x, the reference's.  Kept as the difference
    # from where the link would have put it -- channel c at factor * x +
    # offset linked, x_c + offset free, so the offset cancels -- in the
    # reference's pixels.  That is what unlinking is for: the scatter of
    # these is how well the transformation holds, over the field.
    for channel in range(1, model.n_channels):
        for index, (name, column) in enumerate((("x_roi", "dx_pix"),
                                                ("y_roi", "dy_pix"))):
            other = p.get(f"{name}_ch{channel}")
            if other is None:
                continue
            factor = pairs.link[:, 1, channel, index].astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                cols[f"{column}_ch{channel}"] = other / factor - p[f"{name}_ch0"]
        if model.is_3d and f"z_nm_ch{channel}" in p:
            cols[f"dz_nm_ch{channel}"] = p[f"z_nm_ch{channel}"] - p["z_nm_ch0"]
    add_xy_err(cols)
    return Localizations({k: np.asarray(v) for k, v in cols.items()}, {})


def _concat_pairs(parts: List[PairedROIs]) -> PairedROIs:
    if len(parts) == 1:
        return parts[0]
    return PairedROIs(
        np.concatenate([p.images for p in parts]),
        np.concatenate([p.link for p in parts]),
        Candidates(*(np.concatenate([getattr(p.candidates, name) for p in parts])
                     for name in ("frame", "x", "y", "value"))),
        parts[0].roisize,
        np.concatenate([p.residual for p in parts]))
