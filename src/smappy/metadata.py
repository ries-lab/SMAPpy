"""Camera metadata needed to turn raw camera counts into photons.

Only what the fitting pipeline actually needs is kept.  Values can come from
three places, in increasing priority:

1. the image file itself (Micro-Manager / OME tags)
2. a camera preset (:mod:`smappy.io.cameras_mat`, SMAP's ``*_cameras.mat``)
3. explicit user input -- a dict or a YAML file

Missing required values raise a clear error rather than silently defaulting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

REQUIRED = ("conversion", "offset", "pixelsize_um")

#: a pixel size is one number or two.  Almost every microscope has square
#: pixels and says so with one; a camera whose x and y differ says both, and
#: one number then means the same in both directions rather than only in x.
PixelSize = Union[float, Sequence[float]]


def pixel_sizes(value: Optional[PixelSize]) -> Optional[Tuple[float, float]]:
    """``(x, y)`` from one number or two.  None stays None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    values = [float(v) for v in value]
    if not values:
        return None
    if len(values) == 1:
        return (values[0], values[0])
    return (values[0], values[1])


@dataclass
class CameraMetadata:
    """Camera parameters for ADU -> photon conversion and nm coordinates."""

    conversion: Optional[float] = None  # e- per ADU
    offset: Optional[float] = None  # camera baseline, ADU
    # effective pixel size in the sample: one number, or x and y
    pixelsize_um: Optional[PixelSize] = None
    em_on: Optional[bool] = None  # EM gain used (EMCCD); None = unspecified
    emgain: Optional[float] = None  # EM multiplication gain
    roi: Optional[Sequence[int]] = None  # (x, y, width, height) on the chip
    exposure_ms: Optional[float] = None
    camera_name: Optional[str] = None
    comment: str = ""

    # ---------------------------------------------------------------- factors
    @property
    def adu_to_photons(self) -> float:
        """Multiplicative factor applied to (counts - offset)."""
        self.require("conversion")
        if self.is_em:
            if not self.emgain:
                raise ValueError("em_on is set but emgain is zero or unset")
            return float(self.conversion) / float(self.emgain)
        return float(self.conversion)

    @property
    def is_em(self) -> bool:
        """Whether EM amplification was used (unspecified counts as off)."""
        return bool(self.em_on)

    @property
    def excess_noise(self) -> float:
        """EMCCD excess-noise factor.

        The fitter assumes pure Poisson noise.  EM amplification roughly
        doubles the variance, which is handled outside the fitter by dividing
        the photon counts by this factor before fitting and multiplying
        photons and background by it afterwards.
        """
        return 2.0 if self.is_em else 1.0

    @property
    def pixelsize_um_xy(self) -> Optional[Tuple[float, float]]:
        """The pixel size in x and y, in um; one number means both."""
        return pixel_sizes(self.pixelsize_um)

    @property
    def pixelsize_nm_xy(self) -> Optional[Tuple[float, float]]:
        """The pixel size in x and y, in nm -- what a fit converts with."""
        sizes = self.pixelsize_um_xy
        return None if sizes is None else (sizes[0] * 1000.0, sizes[1] * 1000.0)

    @property
    def square_pixels(self) -> bool:
        sizes = self.pixelsize_um_xy
        return sizes is None or sizes[0] == sizes[1]

    @property
    def roi_offset(self) -> tuple:
        """(x, y) chip offset of the image, used for absolute nm coordinates."""
        if self.roi is None:
            return (0, 0)
        return (int(self.roi[0]), int(self.roi[1]))

    # ---------------------------------------------------------------- merging
    def merged_with(self, other: "CameraMetadata") -> "CameraMetadata":
        """Return a copy where set values of ``other`` override ours."""
        upd = {f.name: getattr(other, f.name) for f in fields(other)
               if not _is_unset(getattr(other, f.name), f.name)}
        return replace(self, **upd)

    def require(self, *names: str) -> None:
        missing = [n for n in (names or REQUIRED) if getattr(self, n) is None]
        if missing:
            raise ValueError(
                "missing camera metadata: " + ", ".join(missing) +
                ". State it in the camera config file, pass it as an option "
                "(--pixelsize, --conversion, --offset), or give it directly: "
                "CameraMetadata(conversion=6.7, offset=398.6, pixelsize_um=0.127)"
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["roi"] = None if self.roi is None else [int(v) for v in self.roi]
        if self.pixelsize_um is not None and not isinstance(self.pixelsize_um,
                                                            (int, float)):
            sizes = self.pixelsize_um_xy
            d["pixelsize_um"] = (float(sizes[0]) if self.square_pixels
                                 else [float(sizes[0]), float(sizes[1])])
        return d

    # ------------------------------------------------------------ contructors
    @classmethod
    def from_dict(cls, d: dict) -> "CameraMetadata":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown camera metadata fields: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path) -> "CameraMetadata":
        import yaml
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        if "camera" in data:  # allow a nested section in a larger config
            data = data["camera"]
        return cls.from_dict(data)

    def to_yaml(self, path) -> None:
        import yaml
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def __str__(self) -> str:
        sizes = self.pixelsize_um_xy
        px = ("?" if sizes is None else f"{sizes[0]:g} um" if self.square_pixels
              else f"{sizes[0]:g} x {sizes[1]:g} um")
        em = (f"EM on, gain {self.emgain:g}" if self.is_em
              else ("EM off" if self.em_on is not None else "EM unspecified"))
        return (f"{self.camera_name or 'camera'}: conversion="
                f"{self.conversion} e-/ADU, offset={self.offset} ADU, "
                f"pixel {px}, {em}, roi={list(self.roi) if self.roi is not None else None}")


def _is_unset(value, name: str) -> bool:
    """Values that mean 'not specified' when merging.

    Every optional field defaults to ``None``, so a user config that omits
    ``em_on`` cannot accidentally switch EM off.
    """
    if value is None:
        return True
    if name == "comment" and value == "":
        return True
    return False


def load_camera_metadata(path) -> CameraMetadata:
    """Load user-supplied metadata from a YAML file."""
    return CameraMetadata.from_yaml(Path(path))
