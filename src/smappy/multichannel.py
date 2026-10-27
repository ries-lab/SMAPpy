"""One emitter seen in several channels of a frame, fitted as one: the engine.

`smappy.dualfit` does this for two halves of a chip; this is the same fit
for any `ChannelLayout` -- the four phase images of a 4Pi microscope, in
rectangles of one chip or in separate files side by side -- and the same
conventions: every channel's ROI is cut at its own integer pixel, nothing is
resampled for the fit, and the sub-pixel remainder of where the layout's map
puts the molecule rides in the fitter's link, with the map's local scale as
the link's factor (`dualfit.build_link`, whose reasons hold here unchanged).

What differs is where the candidates come from.  A 4Pi molecule's four
images differ only by the interference, which shifts its light between them,
so a molecule bright in one channel may be nearly dark in the next.  Summing
the channels -- mapped onto channel 0 for this, and only for this -- cancels
the interference (the four phases are a quarter turn apart) and gives every
molecule its whole signal, twice the signal-to-noise of one channel: one peak
per molecule, nothing to pair.  SMAP found peaks per channel and merged them
(``PeakCombiner``), which a dark channel defeats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .channels import ChannelLayout, detection_lookup, local_scale, onto_reference
from .detect import Candidates
from .dualfit import PairedROIs, _concat_pairs, paired_to_localizations


def detection_image(channels: np.ndarray, layout: ChannelLayout, lookups: Optional[list] = None,
                    resampling: str = "nearest") -> np.ndarray:
    """(n, C, h, w) channels -> (n, h, w): their sum on channel 0's pixels.

    Each channel is read onto channel 0's pixels by ``lookups`` (`channels.detection_lookup`, made here
    with ``resampling`` when not given) and added; the peak finder's own filter smooths the sum, so the
    pixels the nearest-pixel lookup skips or doubles -- a row or column where the scale differs from one,
    a step where a rotation crosses a rounding boundary -- cost no more than half a pixel of alignment in
    one channel of the four.
    """
    n, C, h, w = channels.shape
    total = np.array(channels[:, 0], np.float32)
    for c in range(1, C):
        if np.allclose(layout.maps[c], np.eye(3)):
            total += channels[:, c]
            continue
        lookup = lookups[c] if lookups is not None else detection_lookup(layout, c, (h, w), resampling)
        total += onto_reference(channels[:, c], lookup)
    return total


def channel_candidates(candidates: Candidates, layout: ChannelLayout, roisize: int,
                       drift: "Optional[ChannelDrift]" = None):
    """Where each candidate's ROI goes in every channel.

    Returns ``(positions, residual, keep)``: integer (x, y) per channel
    (n, C, 2), the sub-pixel remainder of the map (n, C, 2) -- zero for
    channel 0 -- and which candidates have every ROI inside its channel.
    ``drift`` moves each channel's position by its offset in the
    candidate's frame (`ChannelDrift`).
    """
    w, h = layout.size
    half = (roisize - 1) // 2
    ref = np.c_[candidates.x, candidates.y].astype(float)
    n, C = len(ref), layout.n_channels
    positions = np.zeros((n, C, 2), np.int64)
    residual = np.zeros((n, C, 2))
    keep = np.ones(n, bool)
    shift = None if drift is None else drift.at(candidates.frame)
    for c in range(C):
        exact = ref if c == 0 else layout.to_channel(c, ref)
        if shift is not None:
            exact = exact + shift[:, c]
        at = np.rint(exact)
        positions[:, c] = at
        residual[:, c] = exact - at
        keep &= ((at[:, 0] >= half) & (at[:, 0] < w - half)
                 & (at[:, 1] >= half) & (at[:, 1] < h - half))
    return positions, residual, keep


def build_link(candidates: Candidates, residual: np.ndarray, layout: ChannelLayout,
               roisize: int, n_parameters: int) -> np.ndarray:
    """The ``(n, 2, C, P)`` link: x and y offset by each channel's remainder
    and scaled by the map's local factor about the ROI centre, as in
    `dualfit.build_link`; every other parameter offset 0, factor 1 -- the
    channels' shares of the light are in their models."""
    n, C = len(candidates), layout.n_channels
    link = np.zeros((n, 2, C, n_parameters), np.float32)
    link[:, 1] = 1.0
    link[:, 0, :, 0] = residual[:, :, 0]
    link[:, 0, :, 1] = residual[:, :, 1]
    half = (int(roisize) - 1) / 2.0
    for c in range(1, C):
        fx, fy = local_scale(layout, c, candidates.x, candidates.y)
        link[:, 1, c, 0] = fx
        link[:, 1, c, 1] = fy
        link[:, 0, c, 0] += (1.0 - fx) * half
        link[:, 0, c, 1] += (1.0 - fy) * half
    return link


def cut_channel_rois(channels: np.ndarray, candidates: Candidates, positions: np.ndarray,
                     roisize: int, first_frame: int = 0) -> np.ndarray:
    """(n, C, roisize, roisize): every candidate's ROI in every channel, each
    at its own integer pixel."""
    from .fit_rois import _block_index
    half = (roisize - 1) // 2
    span = np.arange(-half, half + 1)
    index = _block_index(candidates.frame, first_frame, len(channels))
    out = []
    for c in range(channels.shape[1]):
        rows = positions[:, c, 1][:, None] + span
        cols = positions[:, c, 0][:, None] + span
        out.append(channels[index[:, None, None], c, rows[:, :, None], cols[:, None, :]])
    return np.ascontiguousarray(np.stack(out, axis=1), np.float32)


@dataclass
class ChannelDrift:
    """How far each channel's molecules are from where the layout's map puts
    them, over time: (blocks, C, 2) pixel offsets at the blocks' centre
    frames, interpolated between and held beyond.

    Measured by fitting with x and y free in every channel
    (`measure_channel_drift`).  It takes up a slow drift of the channels
    against each other -- the 4Pi cavity's optics move -- and a constant
    error of the registration alike, including the part a bead registration
    cannot see: where the model's PSF of each channel is centred.  SMAP's
    ``Channeldrift4Pi`` measured it once, on the first 3000 frames.
    """
    frames: np.ndarray
    offsets: np.ndarray

    def at(self, frame) -> np.ndarray:
        frame = np.asarray(frame, float)
        out = np.empty((len(frame),) + self.offsets.shape[1:])
        for c in range(self.offsets.shape[1]):
            for axis in range(2):
                out[:, c, axis] = np.interp(frame, self.frames, self.offsets[:, c, axis])
        return out

    def to_dict(self) -> dict:
        return {"frames": self.frames.tolist(), "offsets_px": self.offsets.tolist()}


def measure_channel_drift(locs, channels: int, pixelsize_nm, block_of=None,
                          min_count: int = 20) -> ChannelDrift:
    """Each channel's offset from channel 0, per block of frames, from a fit
    with x and y free per channel: the medians of its ``dx_nm_ch<c>`` and
    ``dy_nm_ch<c>`` (`dualfit.paired_to_localizations`), over the better
    half of the localizations by their lateral error.  ``block_of`` maps a
    frame to its block (default: all one block)."""
    frame = np.asarray(locs["frame"], float)
    block = np.zeros(len(frame), int) if block_of is None else np.asarray(block_of(frame), int)
    px, py = (pixelsize_nm, pixelsize_nm) if np.isscalar(pixelsize_nm) else pixelsize_nm
    err = np.asarray(locs["x_err_nm"], float)
    good = np.isfinite(err) & (err <= np.nanmedian(err))
    frames, offsets = [], []
    for b in np.unique(block):
        rows = good & (block == b)
        if rows.sum() < min_count:
            continue
        here = np.zeros((channels, 2))
        for c in range(1, channels):
            for axis, (name, scale) in enumerate(((f"dx_nm_ch{c}", px), (f"dy_nm_ch{c}", py))):
                v = np.asarray(locs[name], float)[rows]
                here[c, axis] = np.nanmedian(v) / scale
        frames.append(float(np.median(frame[rows])))
        offsets.append(here)
    if not frames:
        raise ValueError(f"fewer than {min_count} well-fitted localizations in every block: "
                         "nothing to measure the channels' drift by")
    order = np.argsort(frames)
    return ChannelDrift(np.asarray(frames)[order], np.asarray(offsets)[order])


@dataclass
class MultiChannelEngine:
    """`LocalizationEngine` for several channels: one ROI per channel and
    candidate, one fit.

    It takes blocks of channels, (n, C, h, w), as `channels.ChannelSource`
    reads them -- the geometry is that layer's -- and needs of the layout
    only its maps and its channels' size.  Positions come out in channel 0's
    own pixels: the sample's coordinates, whatever rectangle of whichever
    camera channel 0 was cut from.  The table keeps the layout, to draw them
    back on the raw frame.
    """

    camera: "CameraMetadata"
    finder: "PeakFinder"
    model: "PSFModel"              # Global4PiPSF, GlobalSplinePSF
    layout: ChannelLayout
    settings: "FitSettings" = None
    drift: Optional[ChannelDrift] = None

    _pairs: List[PairedROIs] = field(default_factory=list, init=False)
    _buffered: int = field(default=0, init=False)
    #: how the channels are put onto channel 0 to find the molecules (`channels.RESAMPLING`)
    resampling: str = "nearest"
    _lookups: Optional[list] = field(default=None, init=False)
    stats: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        from .pipeline import FitSettings
        if self.settings is None:
            self.settings = FitSettings()
        if self.model.n_channels != self.layout.n_channels:
            raise ValueError(f"the model has {self.model.n_channels} channels, the layout "
                             f"{self.layout.n_channels}")
        self.stats = {"frames": 0, "candidates": 0, "pairs": 0, "localizations": 0,
                      "dropped_at_border": 0, "rejected": 0,
                      "detect_seconds": 0.0, "fit_seconds": 0.0}

    def push(self, frames: np.ndarray, first_frame: int = 0):
        import time
        from .camera import to_photons

        started = time.perf_counter()
        frames = np.asarray(frames)
        w, h = self.layout.size
        if frames.ndim != 4 or frames.shape[1:] != (self.layout.n_channels, h, w):
            raise ValueError(f"expected blocks of {self.layout.n_channels} channels of {w} x {h} "
                             f"(a ChannelSource's), got {frames.shape}")
        channels = to_photons(frames, self.camera) / self.camera.excess_noise
        if self._lookups is None:
            h, w = channels.shape[-2:]
            self._lookups = [None] + [detection_lookup(self.layout, c, (h, w), self.resampling)
                                      for c in range(1, self.layout.n_channels)]
        summed = detection_image(channels, self.layout, self._lookups)
        candidates, _ = self.finder(summed, first_frame=first_frame,
                                    n_threads=self.settings.n_threads)
        roisize = self.settings.roisize
        positions, residual, keep = channel_candidates(candidates, self.layout, roisize,
                                                       self.drift)
        candidates, positions, residual = candidates[keep], positions[keep], residual[keep]
        images = cut_channel_rois(channels, candidates, positions, roisize, first_frame)
        link = build_link(candidates, residual, self.layout, roisize,
                          len(self.model.parameters))
        self.stats["detect_seconds"] += time.perf_counter() - started
        self.stats["frames"] += len(frames)
        self.stats["candidates"] += len(keep)
        self.stats["dropped_at_border"] += int((~keep).sum())
        if len(candidates):
            self._pairs.append(PairedROIs(images, link, candidates, roisize, residual))
            self._buffered += len(candidates)
        if self._buffered >= self.settings.block_rois():
            return self.flush()
        return None

    def flush(self):
        import time
        from .locs import to_nm, valid
        from .pipeline import with_readout, without_readout

        if not self._pairs:
            return None
        pairs = _concat_pairs(self._pairs)
        self._pairs, self._buffered = [], 0
        started = time.perf_counter()
        variance = self.camera.readout_variance
        result = self.model.fit(with_readout(pairs.images, variance), pairs.link,
                                iterations=self.settings.iterations,
                                n_threads=self.settings.n_threads)
        without_readout(result, self.model, variance)
        self.stats["fit_seconds"] += time.perf_counter() - started
        self.stats["pairs"] += len(pairs)

        locs = paired_to_localizations(result, pairs, self.model, self.camera)
        # in channel 0's own pixels: the camera ROI, which the table writer
        # adds for a chip, is the channel layer's business here
        roi_x, roi_y = self.camera.roi_offset
        for name, roi in (("x_pix", roi_x), ("y_pix", roi_y),
                          ("peak_x_pix", roi_x), ("peak_y_pix", roi_y)):
            locs.columns[name] = (np.asarray(locs[name], float) - roi).astype(np.float64)
        keep = valid(locs, self.settings.max_fit_distance)
        self.stats["rejected"] += int((~keep).sum())
        locs = locs[keep]
        self.stats["localizations"] += len(locs)
        cal = getattr(self.model, "calibration", None)
        if hasattr(cal, "phase_per_nm"):
            # what 4Pi z needs to turn the phase into z (`plugins/fourpi_z.py`)
            locs.metadata["fourpi"] = {"phase_per_nm": float(cal.phase_per_nm),
                                       "period_nm": float(cal.period_nm),
                                       "calibration": str(cal.source)}
        locs.metadata["channel_layout"] = self.layout.to_dict()
        if self.drift is not None:
            locs.metadata["channel_drift"] = self.drift.to_dict()
        return to_nm(locs, *self.camera.pixelsize_nm_or_default, keep_pixels=False)

    def __str__(self) -> str:
        s = self.stats
        return (f"{s['localizations']} localizations from {s['frames']} frames of "
                f"{self.layout.n_channels} channels ({s['detect_seconds']:.1f} s detection, "
                f"{s['fit_seconds']:.1f} s fitting)")
