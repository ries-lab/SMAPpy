"""Two channels on one chip, fitted as one emitter.

The split-frame companion to `pipeline.py`: candidates found anywhere on the
frame are combined into one list, each gets an ROI in *both* halves, and the
pair goes to the global fitter, which shares the parameters the caller marks.

This follows SMAP's ``fit_global_dualchannel`` workflow

    PeakFinder -> PeakCombiner -> RoiAdder -> RoiCutterWF -> MLE_global_spline

with `combine_peaks` standing for ``PeakCombiner`` and `cut_paired_rois` for
``RoiAdder`` + ``RoiCutterWF``.  Two points of that design are easy to get
wrong and are the reason this module exists:

* **The candidate list is the union, not the intersection.**  A molecule seen
  in only one channel is still fitted; its partner ROI is cut where the
  transformation says it should be, and the fit sees whatever is there --
  background, usually, which is the right answer for a molecule of the other
  colour.  Pairing only the peaks that both channels found would throw away
  most of a two-colour dataset.
* **Nothing is resampled.**  Each ROI is cut at an integer pixel, and the
  sub-pixel remainder of the transformed position is handed to the fitter as
  an offset in its link -- the same convention `calibrate/dual.py` uses for
  the beads.  Warping the image to align the channels would correlate the
  noise between pixels and break the Poisson likelihood the fitter assumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .calibrate.dual import DualColorCalibration
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
        (``P(:,1) = P(:,1) + channelshift(1,1,:)``).  `combine_peaks` leaves
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

    ``PeakCombiner`` asks the transformation the same question through
    ``getRef``; the split-frame geometry answers it directly.
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


def secondary_to_reference(x: np.ndarray, y: np.ndarray,
                           calibration: DualColorCalibration,
                           origin: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Positions on the secondary half, seen from the reference channel.

    The one mapping `combine_peaks` pairs peaks with, and what a preview
    draws the secondary peaks back-projected by -- so that when a projected
    peak does not land on its partner, the offset on screen is the offset the
    combiner works with, not a copy of it.  ``origin`` is the camera ROI's
    corner on the chip, since the transformation is in chip coordinates.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x):
        return np.empty((0, 2))
    ox, oy = origin
    return calibration.to_reference(np.c_[x + ox, y + oy]) - (ox, oy)


# ------------------------------------------------------------------ combining
def combine_peaks(candidates: Candidates, calibration: DualColorCalibration,
                  image_shape: Tuple[int, int], roisize: int,
                  origin: Tuple[float, float] = (0.0, 0.0),
                  match_pixels: float = MATCH_PIXELS):
    """Candidates from both halves of the chip, merged into one list per frame.

    Returns ``(reference, secondary, residual, counts)``: `Candidates` at
    integer pixels in each channel, ``(n, 2, 2)`` sub-pixel residuals (dx, dy
    per channel; zero for the reference), and how many peaks were merged and
    how many candidates fell off the frame -- two peaks becoming one pair is
    the normal case, not a loss, and the statistics must not confuse them.

    The steps are ``PeakCombiner``'s, in its order:

    1. split the maxima by which half of the chip they are on;
    2. map the secondary ones into reference coordinates;
    3. match within ``match_pixels``; merge a matched pair as a mean weighted
       by ``sqrt(photons)`` -- inverse-variance on the position, and what the
       4Pi branch of ``PeakCombiner`` uses (the two-channel branch squares the
       photons instead, with a comment in the SMAP source questioning it);
    4. the combined list is *every* candidate: matched pairs, and the peaks
       only one channel saw;
    5. round the reference position to a pixel, map that rounded position into
       the secondary channel, round again, and keep the remainder as the
       sub-pixel offset the fitter is given.
    """
    geometry = calibration.geometry
    height, width = image_shape
    half = (roisize - 1) // 2
    ox, oy = origin

    is_reference = which_channel(candidates.x + ox, candidates.y + oy, geometry)
    frames, xs, ys, values = [], [], [], []
    matched = 0

    for frame in np.unique(candidates.frame):
        on_frame = candidates.frame == frame
        ref = candidates[on_frame & is_reference]
        other = candidates[on_frame & ~is_reference]
        # the secondary peaks, seen from the reference channel
        mapped = secondary_to_reference(other.x, other.y, calibration, origin)
        here = np.c_[ref.x, ref.y].astype(float)

        weight_ref = np.sqrt(np.maximum(ref.value, 0.0))
        weight_other = np.sqrt(np.maximum(other.value, 0.0))
        a, b = _match(here, mapped, match_pixels)
        merged = ((here[a] * weight_ref[a, None] + mapped[b] * weight_other[b, None])
                  / np.maximum(weight_ref[a] + weight_other[b], 1e-12)[:, None])
        matched += len(a)
        lone_ref = np.setdiff1d(np.arange(len(here)), a)
        lone_other = np.setdiff1d(np.arange(len(mapped)), b)

        positions = np.vstack([here[lone_ref], mapped[lone_other], merged])
        strength = np.concatenate([ref.value[lone_ref], other.value[lone_other],
                                   ref.value[a] + other.value[b]])
        frames.append(np.full(len(positions), frame, np.int64))
        xs.append(positions[:, 0])
        ys.append(positions[:, 1])
        values.append(strength)

    if not frames:
        empty = Candidates.empty()
        return empty, empty, np.empty((0, 2, 2)), {"matched": 0, "dropped": 0}

    frame = np.concatenate(frames)
    x_ref = np.rint(np.concatenate(xs))
    y_ref = np.rint(np.concatenate(ys))
    value = np.concatenate(values).astype(np.float32)

    # the rounded reference pixel, mapped forward: where the partner ROI goes
    exact = calibration.to_secondary(np.c_[x_ref + ox, y_ref + oy]) - (ox, oy)
    x_sec, y_sec = np.rint(exact[:, 0]), np.rint(exact[:, 1])
    residual = np.zeros((len(frame), 2, 2))
    residual[:, 1, 0] = exact[:, 0] - x_sec
    residual[:, 1, 1] = exact[:, 1] - y_sec

    # both ROIs have to lie inside the frame; a border candidate is dropped, as
    # `cut_rois` drops one rather than shifting it inward and biasing the fit
    inside = np.ones(len(frame), bool)
    for cx, cy in ((x_ref, y_ref), (x_sec, y_sec)):
        inside &= ((cx >= half) & (cx < width - half)
                   & (cy >= half) & (cy < height - half))
    # and each ROI must stay on its own half, or the two would overlap
    inside &= which_channel(x_ref + ox, y_ref + oy, geometry)
    inside &= ~which_channel(x_sec + ox, y_sec + oy, geometry)

    def picked(x, y):
        return Candidates(frame[inside], x[inside].astype(np.int32),
                          y[inside].astype(np.int32), value[inside])

    counts = {"matched": int(matched), "dropped": int((~inside).sum())}
    return picked(x_ref, y_ref), picked(x_sec, y_sec), residual[inside], counts


def _match(a: np.ndarray, b: np.ndarray, tolerance: float):
    """Mutually nearest pairs within ``tolerance`` -- SMAP's ``matchlocs``."""
    if not len(a) or not len(b):
        return np.empty(0, int), np.empty(0, int)
    distance, nearest = cKDTree(b).query(a)
    _, back = cKDTree(a).query(b)
    take = (distance <= tolerance) & (back[nearest] == np.arange(len(a)))
    return np.flatnonzero(take), nearest[take]


# -------------------------------------------------------------------- cutting
def build_link(reference: Candidates, secondary: Candidates, residual: np.ndarray,
               calibration: DualColorCalibration,
               origin: Tuple[float, float] = (0.0, 0.0),
               photon_ratio: Optional[float] = None,
               roisize: Optional[int] = None) -> np.ndarray:
    """The ``(n, 2, 2, 5)`` link the global fitter takes.

    The fitter evaluates each channel at ``factor * global + offset``, in that
    channel's own ROI pixel coordinates.  The factors are the local scale of
    the transformation for x and y, and the splitter's photon ratio for the
    photon number, so that a shared photon parameter means the *total* as
    channel 0 sees it.  Everything else is offset 0, factor 1: the two
    calibrations are checked on load to share one z grid, so z needs neither.

    The x and y offsets are the sub-pixel residual *plus* ``(1 - factor) *
    half``: the scale acts about the ROI centre, where both ROIs were cut, not
    about the ROI's corner.  For a splitter that is nearly a translation the
    term is a hundredth of a pixel; for a *mirrored* splitter the factor is -1
    and without it the partner channel is evaluated at a negative coordinate,
    outside its ROI -- its photons fit to nothing and the colour is lost.
    ``roisize`` is required for that; without it the old corner-anchored form
    is kept, for callers that build a link by hand.
    """
    n = len(reference)
    link = np.zeros((n, 2, 2, 5), np.float32)
    link[:, 1] = 1.0                                   # factors default to one
    link[:, 0, :, 0] = residual[:, :, 0]               # x offsets
    link[:, 0, :, 1] = residual[:, :, 1]               # y offsets

    ox, oy = origin
    fx, fy = _linearise(calibration.to_secondary,
                        reference.x + ox, reference.y + oy)
    link[:, 1, 1, 0] = fx
    link[:, 1, 1, 1] = fy
    if roisize is not None:
        half = (int(roisize) - 1) / 2.0
        link[:, 0, 1, 0] += (1.0 - fx) * half
        link[:, 0, 1, 1] += (1.0 - fy) * half
    if photon_ratio is None:
        photon_ratio = calibration.parameters.get(
            "secondary_main_brightness_ratio", 1.0)
    link[:, 1, 1, 2] = photon_ratio
    return link


def cut_paired_rois(photons: np.ndarray, reference: Candidates, secondary: Candidates,
                    residual: np.ndarray, roisize: int,
                    calibration: DualColorCalibration, first_frame: int = 0,
                    origin: Tuple[float, float] = (0.0, 0.0),
                    photon_ratio: Optional[float] = None) -> PairedROIs:
    """Cut the pair of ROIs for every candidate out of the same split frame.

    No resampling: each ROI is taken at its own integer pixel and the
    remainder rides along in the link.
    """
    from .roi import _block_index

    photons = np.asarray(photons, np.float32)
    if photons.ndim == 2:
        photons = photons[None]
    half = (roisize - 1) // 2
    span = np.arange(-half, half + 1)
    index = _block_index(reference.frame, first_frame, len(photons))

    cut = []
    for channel in (reference, secondary):
        rows = channel.y[:, None] + span
        cols = channel.x[:, None] + span
        cut.append(photons[index[:, None, None], rows[:, :, None], cols[:, None, :]])
    images = np.ascontiguousarray(np.stack(cut, axis=1), np.float32)

    link = build_link(reference, secondary, residual, calibration, origin, photon_ratio,
                      roisize=roisize)
    return PairedROIs(images, link, reference, roisize, residual)


# --------------------------------------------------------------------- driving
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

    p = model.unpack(result)
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
    add_xy_err(cols)
    return Localizations({k: np.asarray(v) for k, v in cols.items()}, {})


@dataclass
class DualChannelEngine:
    """`LocalizationEngine` for a split frame: two ROIs per candidate, one fit.

    Frames go in whole -- both halves at once, as the camera delivers them --
    because that is how the peaks are found and combined.
    """

    camera: "CameraMetadata"
    finder: "PeakFinder"
    model: "PSFModel"          # GlobalSplinePSF or GlobalGaussianPSF
    calibration: DualColorCalibration
    settings: "FitSettings" = None
    photon_ratio: Optional[float] = None

    _pairs: List[PairedROIs] = field(default_factory=list, init=False)
    _buffered: int = field(default=0, init=False)
    stats: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        from .pipeline import FitSettings
        if self.settings is None:
            self.settings = FitSettings()
        self.stats = {"frames": 0, "candidates": 0, "pairs": 0, "matched": 0,
                      "localizations": 0, "dropped_at_border": 0, "rejected": 0,
                      "detect_seconds": 0.0, "fit_seconds": 0.0}

    def push(self, frames: np.ndarray, first_frame: int = 0):
        import time
        from .camera import to_photons

        started = time.perf_counter()
        photons = to_photons(frames, self.camera) / self.camera.excess_noise
        candidates, _ = self.finder(photons, first_frame=first_frame,
                                    n_threads=self.settings.n_threads)
        reference, secondary, residual, counts = combine_peaks(
            candidates, self.calibration, photons.shape[-2:], self.settings.roisize,
            origin=self.camera.roi_offset)
        pairs = cut_paired_rois(photons, reference, secondary, residual,
                                self.settings.roisize, self.calibration,
                                first_frame=first_frame,
                                origin=self.camera.roi_offset,
                                photon_ratio=self.photon_ratio)
        self.stats["detect_seconds"] += time.perf_counter() - started
        self.stats["frames"] += len(photons)
        self.stats["candidates"] += len(candidates)
        self.stats["matched"] += counts["matched"]      # seen in both channels
        self.stats["dropped_at_border"] += counts["dropped"]

        if len(pairs):
            self._pairs.append(pairs)
            self._buffered += len(pairs)
        if self._buffered >= self.settings.block_rois():
            return self.flush()
        return None

    def flush(self):
        import time
        from .locs import valid, to_nm

        if not self._pairs:
            return None
        pairs = _concat_pairs(self._pairs)
        self._pairs, self._buffered = [], 0

        started = time.perf_counter()
        result = self.model.fit(pairs.images, pairs.link,
                                iterations=self.settings.iterations,
                                n_threads=self.settings.n_threads)
        self.stats["fit_seconds"] += time.perf_counter() - started
        self.stats["pairs"] += len(pairs)

        locs = paired_to_localizations(result, pairs, self.model, self.camera)
        keep = valid(locs, self.settings.max_fit_distance)
        self.stats["rejected"] += int((~keep).sum())
        locs = locs[keep]
        self.stats["localizations"] += len(locs)

        if self.settings.output_unit != "pixel":
            self.camera.require("pixelsize_um")
            locs = to_nm(locs, self.camera.pixelsize_nm_xy,
                         keep_pixels=self.settings.output_unit == "pixel+nm")
        return locs

    def __str__(self) -> str:
        s = self.stats
        return (f"{s['localizations']} localizations from {s['frames']} split "
                f"frames ({s['detect_seconds']:.1f} s detection and pairing, "
                f"{s['fit_seconds']:.1f} s fitting)")


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
