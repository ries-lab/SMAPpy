"""Microscope profiles: what uiPSF needs to know about the instrument.

A PSF model is learnt from the optics outward -- numerical aperture,
refractive indices, the emission wavelength -- and a user calibrating beads on
Tuesday should not type those in again.  A profile is a small YAML file per
microscope, kept where the group shares it, with the values uiPSF's own
``config/systemtype/*.yaml`` files hold under names a user reads::

    name: M2, astigmatic, two colour
    NA: 1.43
    refractive_index: {immersion: 1.516, medium: 1.335, coverslip: 1.516}
    emission_wavelength_nm: 680
    pixelsize_um: [0.127, 0.116]     # optional: else from the camera
    roi_size_px: 25                  # the model's lateral size
    dual:                            # a split camera; omit for one channel
      layout: up-down mirrored       # as the bead calibration: smappy.calibrate.dual.LAYOUTS
      main_channel: upper
      split_position: null           # null: the middle
    insitu:                          # the pupil learning from blinking starts at
      zernike_index: [5]             # Noll index (5: astigmatism)
      zernike_coeff: [0.5]           # its amplitude, in radians
    uipsf: {}                        # any uiPSF parameter, nested as in its YAML

The camera -- gain, offset, EM gain -- is not here: it comes from the
acquisition and the camera database, as it does for fitting.  Pixel size is
both: the camera's file says it when the microscope software was told, and a
profile is the fallback for one that was not.

Profiles are looked for in ``$SMAPPY_MICROSCOPES`` (one or more directories,
separated as ``PATH`` is), ``<config dir>/microscopes`` and the shipped
``smappy/data/microscopes``; the first file of a name wins, so a group's
directory overrides what ships.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

SHIPPED = Path(__file__).resolve().parents[1] / "data" / "microscopes"


@dataclass
class Microscope:
    """One profile, with uiPSF's defaults for what it leaves out."""
    name: str = "default"
    NA: float = 1.43
    refractive_index: Dict[str, float] = field(default_factory=lambda: dict(
        immersion=1.516, medium=1.335, coverslip=1.516))
    emission_wavelength_nm: float = 680.0
    pixelsize_um: Optional[Any] = None
    roi_size_px: int = 25
    dual: Optional[Dict[str, Any]] = None
    insitu: Dict[str, Any] = field(default_factory=lambda: dict(
        zernike_index=[5], zernike_coeff=[0.5]))
    uipsf: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path) -> "Microscope":
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: a microscope profile is a YAML mapping")
        known = set(cls.__dataclass_fields__) - {"path"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}; a profile "
                             f"has {sorted(known)} (uiPSF's own go under 'uipsf')")
        data.setdefault("name", path.stem)
        profile = cls(**data, path=path)
        profile.validate()
        return profile

    def validate(self) -> None:
        if not 0 < self.NA < 2:
            raise ValueError(f"{self.name}: NA {self.NA} is not a numerical aperture")
        for key in ("immersion", "medium", "coverslip"):
            if key not in self.refractive_index:
                raise ValueError(f"{self.name}: refractive_index needs {key}")
        if self.dual is not None:
            from ..calibrate.dual import LAYOUTS
            layout = self.dual.get("layout", "up-down mirrored")
            if layout not in LAYOUTS:
                raise ValueError(f"{self.name}: dual layout must be one of {LAYOUTS}")

    @property
    def pixelsize_xy_um(self) -> Optional[Tuple[float, float]]:
        p = self.pixelsize_um
        if p is None:
            return None
        if isinstance(p, (list, tuple)):
            return float(p[0]), float(p[-1])
        return float(p), float(p)


def directories() -> List[Path]:
    from ..config import config_dir
    found = [Path(p).expanduser() for p in
             os.environ.get("SMAPPY_MICROSCOPES", "").split(os.pathsep) if p]
    return found + [config_dir() / "microscopes", SHIPPED]


def profiles() -> Dict[str, Path]:
    """Every profile there is, by file name, the first of a name winning."""
    out: Dict[str, Path] = {}
    for d in directories():
        if d.is_dir():
            for p in sorted(d.glob("*.yaml")):
                out.setdefault(p.stem, p)
    return out


def load(name_or_path: str) -> Microscope:
    """A profile by its name, or a YAML file by its path; '' is uiPSF's defaults."""
    if not name_or_path:
        return Microscope()
    found = profiles()
    if name_or_path in found:
        return Microscope.from_yaml(found[name_or_path])
    if Path(name_or_path).exists():
        return Microscope.from_yaml(name_or_path)
    raise ValueError(f"no microscope profile {name_or_path!r}; there are "
                     f"{sorted(found)} in {[str(d) for d in directories()]}")


def choices() -> List[Tuple[str, str]]:
    """For the plugin's drop-down: (name, the profile's own name)."""
    out = [("", "uiPSF defaults")]
    for stem, path in profiles().items():
        try:
            out.append((stem, Microscope.from_yaml(path).name))
        except (OSError, ValueError, yaml.YAMLError):
            out.append((stem, f"{stem} (unreadable)"))
    return out
