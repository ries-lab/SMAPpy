"""Saved csv conversions: how the columns of one program's text export are read.

SMAP's ``Loader_csvAndMore`` kept a *conversion structure* per program
(`settings/csvloaderconversion`): which header is x, which the frame, the
camera pixel when the positions are in pixels, a factor for the units.  A
lab reads the same few programs' exports again and again, so the mapping is
made once, saved under a name, and found again by the headers.

A conversion is a small YAML file: its ``headers`` (what a file must have
for it to apply), the ``mapping`` (header -> smappy column; anything not
named keeps its own name), the ``units`` of the lengths, the pixel size for
``px``, and ``first_frame``.  The shipped ones are in
``smappy/data/csv_conversions``; the user's are under the config directory
(`config.config_dir`) and win over a shipped one of the same name.  The
conversion that applies to a file is the one with the most headers, all
of which the file has: ThunderSTORM's "x [nm]", "y [nm]" and "frame" are
not enough to tell it from a program that copied its headers, but they are
enough to count its frames from 1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

SHIPPED = Path(__file__).resolve().parent.parent / "data" / "csv_conversions"
SUFFIX = ".yaml"


@dataclass
class Conversion:
    name: str
    headers: List[str] = field(default_factory=list)    # all of them in the file
    mapping: Dict[str, str] = field(default_factory=dict)
    units: str = "nm"
    pixelsize_nm: Optional[float] = None
    first_frame: int = 0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("name")
        return d


def directories() -> List[Path]:
    from ..config import config_dir
    return [config_dir() / "csv_conversions", SHIPPED]


def conversions() -> Dict[str, Conversion]:
    """Every saved conversion by name, the user's winning over a shipped one."""
    out: Dict[str, Conversion] = {}
    for d in directories():
        for path in sorted(d.glob(f"*{SUFFIX}")) if d.is_dir() else ():
            name = path.name[:-len(SUFFIX)]
            if name in out:
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                out[name] = Conversion(name=name, **data)
            except (OSError, TypeError, yaml.YAMLError):
                continue                    # a broken file costs itself only
    return out


def conversion_named(name: str) -> Conversion:
    found = conversions().get(name)
    if found is None:
        raise ValueError(f"no csv conversion called {name!r}; there are "
                         + (", ".join(conversions()) or "none"))
    return found


def conversion_for(headers: Sequence[str]) -> Optional[Conversion]:
    """The saved conversion whose headers the file has, the most specific."""
    have = {h.strip() for h in headers}
    fitting = [c for c in conversions().values() if c.headers and set(c.headers) <= have]
    return max(fitting, key=lambda c: len(c.headers), default=None)


def save_conversion(conversion: Conversion) -> Path:
    """Write ``conversion`` to the user's directory, replacing one of its name."""
    d = directories()[0]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{conversion.name}{SUFFIX}"
    path.write_text(yaml.safe_dump(conversion.to_dict(), sort_keys=False), encoding="utf-8")
    return path
