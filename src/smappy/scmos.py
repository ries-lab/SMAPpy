"""sCMOS noise maps: each pixel's offset and read-noise variance, from dark frames.

An sCMOS reads every pixel through its own amplifier, so each pixel has its
own baseline and its own read noise -- typically 0.7 to 1.4 e- rms, with a
tail of "hot" and random-telegraph pixels at 5 to 30 e-.  The fit otherwise
uses one offset and one variance for the whole chip (`CameraMetadata.offset`,
`readout_variance`).  With a map, the offset is subtracted pixel by pixel and
the variance enters the likelihood pixel by pixel (Huang et al. 2013): the
noisy pixels are weighted down instead of pulling the fit.

**Calibration is a dark movie and nothing else.**  The shutter closed, the
readout mode and binning of the data, 1000 to 2000 frames: each pixel's mean is
its offset and its variance its read noise, both in ADU.  The variance of N
frames is known to sqrt(2 / (N - 1)), 4.5 % at 1000 frames -- plenty, since it
only sets the weights.  What is deliberately left out:

* **gain per pixel.**  The pixels' gains differ by 1 to 3 % on a modern sCMOS,
  and measuring one pixel's gain to 1 % from a photon-transfer curve takes
  about 20 000 frames per intensity; a quick calibration would add more error
  than it removes.  The camera's one conversion is used (Picasso offers a gain
  map; SMAP reads one and uses only its median, `makegainoffsetCMOS`).
* **dark current.**  At the exposures of single-molecule imaging (tens of ms)
  it is a small fraction of the read noise, so one dark movie at any short
  exposure stands for all of them.

**The map covers all or part of the chip** and keeps which part, ``region``
``(x, y, width, height)`` in chip pixels.  A file is fitted with the part of
the map under its own camera ROI (`NoiseMap.crop`), and refused when it reaches
outside the map: a pixel's noise is never guessed.

The map is kept in ADU so that a conversion corrected later corrects it too;
`NoiseMap.in_photons` converts it with the camera the fit uses.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

#: the fewest dark frames a map is made from: below this a pixel's variance is
#: known to worse than 15 %
MIN_FRAMES = 100
#: below this many frames the calibration says so (4.5 % at 1000)
GOOD_FRAMES = 1000
#: a pixel whose read noise is this many times the median is counted as hot
#: in the report; the fit needs no such cut, it weights every pixel by its own
HOT_FACTOR = 3.0

Region = Tuple[int, int, int, int]


@dataclass
class NoiseMap:
    """One pixel's offset and read-noise variance per pixel of ``region``."""

    offset: np.ndarray                 # (h, w) float32, ADU
    variance: np.ndarray               # (h, w) float32, ADU^2
    region: Region = (0, 0, 0, 0)      # (x, y, width, height) on the chip
    n_frames: int = 0
    #: what it was measured with: camera, readout mode, exposure, source, date
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.offset = np.asarray(self.offset, np.float32)
        self.variance = np.asarray(self.variance, np.float32)
        if self.offset.shape != self.variance.shape or self.offset.ndim != 2:
            raise ValueError("offset and variance must be two images of one shape")
        h, w = self.offset.shape
        x, y = int(self.region[0]), int(self.region[1])
        self.region = (x, y, w, h)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.offset.shape

    def read_noise_adu(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, 0.0))

    # ---------------------------------------------------------------- cropping
    def crop(self, roi: Optional[Sequence[int]], shape: Tuple[int, int]) -> "NoiseMap":
        """The part of the map under a file's frames.

        ``roi`` is the file's camera ROI (`CameraMetadata.roi`): where on the
        chip its frames start; ``shape`` its frames' ``(height, width)``.  A
        file that says nothing about its ROI is taken to start where the map
        does if it has the map's size, and at the chip's corner otherwise --
        the full-chip file of a full-chip map, or a file of the very ROI the
        map was taken with.  Anything that reaches outside the map is refused.
        """
        h, w = int(shape[0]), int(shape[1])
        mx, my, mw, mh = self.region
        if roi is None or len(roi) < 2:
            x, y = (mx, my) if (h, w) == (mh, mw) else (0, 0)
        else:
            x, y = int(roi[0]), int(roi[1])
        if x < mx or y < my or x + w > mx + mw or y + h > my + mh:
            raise ValueError(
                f"the frames cover chip pixels x {x}-{x + w - 1}, y {y}-{y + h - 1}, and the "
                f"noise map only x {mx}-{mx + mw - 1}, y {my}-{my + mh - 1}: calibrate with "
                f"dark frames that cover this camera ROI (or the whole chip)")
        rows = slice(y - my, y - my + h)
        cols = slice(x - mx, x - mx + w)
        return replace(self, offset=self.offset[rows, cols].copy(),
                       variance=self.variance[rows, cols].copy(), region=(x, y, w, h))

    def in_photons(self, adu_to_photons: float, excess_noise: float = 1.0) -> np.ndarray:
        """The variance in the fit's units: electrons squared, divided by the
        excess-noise factor as the counts are (`CameraMetadata.readout_variance`)."""
        scale = float(adu_to_photons) / float(excess_noise)
        return np.maximum(self.variance, 0.0).astype(np.float32) * np.float32(scale ** 2)

    # ------------------------------------------------------------------ a file
    def save(self, path) -> Path:
        import h5py
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["format"] = "smappy noise map"
            f.attrs["version"] = 1
            f.attrs["region"] = np.asarray(self.region, np.int64)
            f.attrs["n_frames"] = int(self.n_frames)
            for key, value in self.info.items():
                if value is not None:
                    f.attrs[f"info_{key}"] = value if not isinstance(value, dict) else str(value)
            f.create_dataset("offset_adu", data=self.offset, compression="gzip")
            f.create_dataset("variance_adu2", data=self.variance, compression="gzip")
        return path

    @classmethod
    def load(cls, path) -> "NoiseMap":
        import h5py
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"no noise map at {path}")
        with h5py.File(path, "r") as f:
            if f.attrs.get("format") != "smappy noise map":
                raise ValueError(f"{path.name} is not a smappy noise map")
            info = {k[5:]: _plain(v) for k, v in f.attrs.items() if k.startswith("info_")}
            return cls(offset=f["offset_adu"][()], variance=f["variance_adu2"][()],
                       region=tuple(int(v) for v in f.attrs["region"]),
                       n_frames=int(f.attrs.get("n_frames", 0)), info=info)


def _plain(value):
    """An HDF5 attribute as a Python value."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


_CACHE: Dict[Tuple[str, float], NoiseMap] = {}


def load_cached(path) -> NoiseMap:
    """`NoiseMap.load`, kept while the file is unchanged: a fit's every
    Preview and a batch's every file read the same map."""
    path = Path(path)
    key = (str(path.resolve()), path.stat().st_mtime)
    found = _CACHE.get(key)
    if found is None:
        _CACHE.clear()
        found = _CACHE[key] = NoiseMap.load(path)
    return found


# --------------------------------------------------------------- calibrating
@dataclass
class Moments:
    """Each pixel's running mean and sum of squared deviations (Welford's,
    combined block by block as Chan et al. combine two halves): one pass over
    the frames, in float64, so a dark movie of any length fits in memory as
    three images."""

    n: int = 0
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None

    def add(self, block: np.ndarray) -> None:
        block = np.asarray(block, np.float64)
        if block.ndim == 2:
            block = block[None]
        k = block.shape[0]
        if k == 0:
            return
        mean_b = block.mean(axis=0)
        m2_b = ((block - mean_b) ** 2).sum(axis=0)
        if self.mean is None:
            self.n, self.mean, self.m2 = k, mean_b, m2_b
            return
        if mean_b.shape != self.mean.shape:
            raise ValueError("frames of different sizes in one dark movie")
        n = self.n + k
        delta = mean_b - self.mean
        self.mean = self.mean + delta * (k / n)
        self.m2 = self.m2 + m2_b + delta ** 2 * (self.n * k / n)
        self.n = n

    def variance(self) -> np.ndarray:
        if self.n < 2:
            raise ValueError("a variance needs at least two frames")
        return self.m2 / (self.n - 1)


def dark_map(blocks: Iterable[np.ndarray], region: Optional[Sequence[int]] = None,
             info: Optional[Dict[str, Any]] = None, report=None,
             min_frames: int = MIN_FRAMES) -> NoiseMap:
    """The noise map of a dark movie, from its blocks of frames ``(n, h, w)``
    in ADU.  ``region``: where on the chip the frames are (the file's camera
    ROI), the chip's corner if None."""
    moments = Moments()
    for block in blocks:
        moments.add(block)
        if report is not None:
            report(f"{moments.n} dark frames read")
    if moments.n < min_frames:
        raise ValueError(f"{moments.n} dark frames: a noise map needs at least {min_frames}")
    x, y = (0, 0) if region is None or len(region) < 2 else (int(region[0]), int(region[1]))
    return NoiseMap(offset=moments.mean, variance=moments.variance(), region=(x, y, 0, 0),
                    n_frames=moments.n, info=dict(info or {}))


def summary(noise: NoiseMap, adu_to_photons: Optional[float] = None) -> Dict[str, float]:
    """The numbers a calibration reports: the offset's spread, the read
    noise's median and tail, the hot pixels -- in e- where the conversion is
    known, else in ADU."""
    scale = float(adu_to_photons) if adu_to_photons else 1.0
    noise_e = noise.read_noise_adu() * scale
    median = float(np.median(noise_e))
    return {
        "frames": float(noise.n_frames),
        "offset_median_adu": float(np.median(noise.offset)),
        "offset_spread_adu": float(np.percentile(noise.offset, 99) - np.percentile(noise.offset, 1)),
        "read_noise_median": median,
        "read_noise_mean_rms": float(np.sqrt(np.mean(noise_e ** 2))),
        "read_noise_p99": float(np.percentile(noise_e, 99)),
        "read_noise_max": float(noise_e.max()),
        "hot_fraction": float(np.mean(noise_e > HOT_FACTOR * median)) if median > 0 else 0.0,
        "variance_precision": float(np.sqrt(2.0 / max(noise.n_frames - 1, 1))),
    }


def default_path(camera: str, mode: str = "") -> Path:
    """Where a calibration keeps its map: the configuration directory's
    ``noise_maps``, one file per camera and readout mode, dated so that a new
    calibration never overwrites the one older fits name."""
    from . import config
    stem = "_".join(_safe(part) for part in (camera or "camera", mode) if part)
    day = _dt.date.today().strftime("%Y%m%d")
    return config.config_dir() / "noise_maps" / f"{stem}_{day}.h5"


def _safe(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-." else "_" for c in str(text))
    return "_".join(part for part in keep.split("_") if part)[:60]


# ------------------------------------------------------------------- fitting
@dataclass
class FitNoise:
    """A noise map ready for a file's frames: the offset in ADU, the variance
    in the fit's units, both the frames' shape (`NoiseMap.crop`)."""

    offset_adu: np.ndarray
    variance: np.ndarray
    path: str = ""

    def to_photons(self, frames: np.ndarray, adu_to_photons: float) -> np.ndarray:
        """`camera.to_photons` with the offset of each pixel."""
        out = np.asarray(frames, np.float32) - self.offset_adu
        out *= np.float32(adu_to_photons)
        return out


def for_frames(camera, shape: Tuple[int, int], path=None) -> FitNoise:
    """The noise map ``camera`` names (`CameraMetadata.noise_map`, or ``path``)
    cut to frames of ``shape`` at the camera's ROI, in the fit's units."""
    path = path or getattr(camera, "noise_map", None)
    if not path:
        raise ValueError(
            f"{getattr(camera, 'camera_name', None) or 'the camera'} has no noise map for "
            f"this readout mode: make one with Localize > Camera noise calibration, "
            f"or turn per-pixel noise off")
    if camera.is_em:
        raise ValueError("per-pixel noise is for an sCMOS: this file was taken with EM gain")
    noise = load_cached(path).crop(camera.roi, shape)
    return FitNoise(offset_adu=noise.offset,
                    variance=noise.in_photons(camera.adu_to_photons, camera.excess_noise),
                    path=str(path))


def cut(variance: np.ndarray, candidates, roisize: int) -> np.ndarray:
    """The variance of each fit ROI, cut at the candidates' positions as
    `fit_rois.cut_rois` cuts the frames (the candidates it kept)."""
    half = (roisize - 1) // 2
    span = np.arange(-half, half + 1)
    rows = np.asarray(candidates.y)[:, None] + span
    cols = np.asarray(candidates.x)[:, None] + span
    return np.ascontiguousarray(variance[rows[:, :, None], cols[:, None, :]], np.float32)


# ------------------------------------------------------------ several channels
def for_layout(camera, layout, frame_shape=None) -> FitNoise:
    """The noise map cut into the channels of ``layout`` as the frames are --
    each channel's rectangle, flipped as it is, the smaller ones padded
    (`channels.ChannelLayout.cut`) -- so that every channel's pixel has its own
    noise whatever mirror it went through: ``(C, h, w)`` offset and variance.

    ``layout`` is on the frames (the camera ROI taken off, as a
    `channels.ChannelSource` holds it); ``frame_shape`` the frames' (height,
    width) as the camera wrote them, the channels' extent if None.  A padded
    pixel has no offset and no variance, as it has no light."""
    if frame_shape is None:
        frame_shape = (max(y + h for _, y, _, h in layout.regions),
                       max(x + w for x, _, w, _ in layout.regions))
    whole = for_frames(camera, tuple(frame_shape)[-2:])
    return FitNoise(offset_adu=np.ascontiguousarray(layout.cut(whole.offset_adu[None])[0]),
                    variance=np.ascontiguousarray(layout.cut(whole.variance[None])[0]),
                    path=whole.path)


def channel(noise: FitNoise, index: int, size) -> FitNoise:
    """One channel of `for_layout`'s, at its own ``size`` (width, height): what a
    channel fitted on its own reads (`multichannel.SplitEngine`)."""
    w, h = size
    return FitNoise(offset_adu=np.ascontiguousarray(noise.offset_adu[index, :h, :w]),
                    variance=np.ascontiguousarray(noise.variance[index, :h, :w]),
                    path=noise.path)


def cut_channels(variance: np.ndarray, positions: np.ndarray, roisize: int) -> np.ndarray:
    """(n, C, roisize, roisize): the variance of every ROI in every channel,
    cut at each channel's own integer pixel as `multichannel.cut_channel_rois`
    cuts the frames.  ``variance`` (C, h, w), ``positions`` (n, C, 2) x, y."""
    half = (roisize - 1) // 2
    span = np.arange(-half, half + 1)
    out = []
    for c in range(variance.shape[0]):
        rows = positions[:, c, 1][:, None] + span
        cols = positions[:, c, 0][:, None] + span
        out.append(variance[c][rows[:, :, None], cols[:, None, :]])
    return np.ascontiguousarray(np.stack(out, axis=1), np.float32)
