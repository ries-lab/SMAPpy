"""The microscope database: which microscope took this file, and its optics.

uiPSF needs the NA, the immersion index, the emission wavelength and the
pixel on the sample per axis, and Micro-Manager records none of them.  It
records every device's state instead -- ``Servos-Position1 = 28000``,
``Thorlabs Filter Wheel-State = 3`` -- and those states decide the rest: on
the Ries lab's M2 the cylindrical lens of the astigmatic path makes the pixel
127 x 116 nm, the filter wheel the wavelength.  So a microscope entry names
the few devices that matter, names their states, and lets each parameter
follow one of them.  `docs/microscope-database.md` is the design.

**One device per parameter, not one state per configuration.**  The camera
database recognises a readout mode by tags that must all match, because a
conversion depends on gain, rate and port together.  A microscope's numbers
do not: the pixel follows the lens and nothing else, the wavelength the
filter.  With a lens of two states, a wheel of six and a slider of four,
entering combinations would mean 48 of them.

**States are named, and parameters keyed by the names.**  A servo is
re-trimmed, a device renamed in Micro-Manager, an EMU configuration exported
again: the device's tags and values change, and every parameter that follows
it keeps its table, because ``"in"`` is still ``"in"``.  Numeric devices are
read by the nearest state, split at the midpoints, so a small re-trim needs
nothing at all; a change it does not absorb is saved as a new version of
the entry, the old one kept with the date it applied until.

**Nothing here overrides the file**, as in `camera_db`: what is resolved is
what the entry says, each value with a `Source`, and the file and the user
are laid over it where they meet a fit.  The microscopes are the
``microscopes`` list of a lab's ``microscopes.json``, beside its cameras.
"""
from __future__ import annotations

import copy
import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .camera_db import (SHIPPED, Parameter, Source, read_file, read_number,
                        user_file, write_section)

#: how a device's reading becomes a state
READS = ("exact", "nearest", "value")

#: what a parameter that follows a filter reads from its state's name
FROM_NAME = "from the name"

#: chip sides a channel of a split lands on
SIDES = ("left", "right", "top", "bottom")


# ------------------------------------------------------------------ devices

@dataclass
class Device:
    """One device of the microscope: where it is read, and its named states.

    `tags` are tried in order, so a device renamed in Micro-Manager keeps its
    entry (``ELL9 - Reflected Slider-Label``, before that ``Thorlabs
    ELL9-Label``).  `states` maps each state's name to what the tag reads in
    it.  ``exact`` compares text (an ELL9's ``Position 1``); ``nearest``
    takes the closest state's value, which splits between neighbours at the
    midpoint -- a servo sent to 28000 and read at 27900 is still in -- and,
    with a `tolerance`, rejects a reading too far from every state;
    ``value`` keeps a continuous position (a TIRF angle) as it is, with no
    state.  `emu` is the EMU control it was imported from, which a second
    import updates it by.
    """
    name: str
    tags: List[str] = field(default_factory=list)
    read: str = "exact"
    states: Dict[str, Any] = field(default_factory=dict)
    tolerance: Optional[float] = None
    emu: str = ""

    def tag_in(self, metadata: Dict[str, Any]) -> Optional[str]:
        return next((t for t in self.tags if t in metadata), None)

    def state(self, metadata: Dict[str, Any]) -> Tuple[Optional[str], Any, Source]:
        """``(state name, what the file read, Source)``; a name of None means
        the file says nothing this device knows, and the source says why."""
        tag = self.tag_in(metadata)
        if tag is None:
            return None, None, Source("missing", "no tag " + " or ".join(self.tags))
        raw = metadata[tag]
        said = f"{tag} = {raw}"
        if self.read == "value":
            return None, read_number(raw), Source("metadata", said)
        if self.read == "nearest":
            found = read_number(raw)
            numbers = [(name, read_number(v)) for name, v in self.states.items()]
            numbers = [(name, v) for name, v in numbers if v is not None]
            if found is None or not numbers:
                return None, raw, Source("missing", f"{said}: not a number")
            # min keeps the first of two states at the same distance -- and
            # of two states a configuration gave the same value
            name, value = min(numbers, key=lambda nv: abs(nv[1] - found))
            if self.tolerance is not None and abs(value - found) > self.tolerance:
                return None, raw, Source(
                    "missing", f"{said}: no state within {self.tolerance:g}")
            return name, raw, Source("metadata", said)
        wanted = str(raw).strip()
        for name, value in self.states.items():
            if str(value).strip() == wanted:
                return name, raw, Source("metadata", said)
        return None, raw, Source("missing", f"{said}: no state reads that")

    @classmethod
    def from_dict(cls, name: str, raw: Dict[str, Any]) -> "Device":
        tags = raw.get("tag", [])
        tags = [tags] if isinstance(tags, str) else [str(t) for t in tags]
        tolerance = raw.get("tolerance")
        return cls(name=name, tags=tags, read=str(raw.get("read", "exact")),
                   states=dict(raw.get("states") or {}),
                   tolerance=None if tolerance is None else float(tolerance),
                   emu=str(raw.get("emu", "")))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"tag": self.tags[0] if len(self.tags) == 1 else list(self.tags)}
        if self.read != "exact":
            out["read"] = self.read
        if self.tolerance is not None:
            out["tolerance"] = self.tolerance
        if self.states:
            out["states"] = dict(self.states)
        if self.emu:
            out["emu"] = self.emu
        return out


# --------------------------------------------------------------- parameters

def wavelength_from_name(name: Any) -> Optional[float]:
    """A band-pass filter's centre from its name: ``676/37`` -> 676.

    The names an EMU configuration gives the filter positions are already
    the data; only a name that is not a band (``665LP``, ``None``,
    ``Empty``) has to be typed.
    """
    found = re.match(r"\s*(\d{3,4}(?:\.\d+)?)\s*/\s*\d", str(name))
    return float(found.group(1)) if found else None


@dataclass
class Follows:
    """A parameter that follows one device: a value per state of it."""
    device: str
    values: Any = field(default_factory=dict)      # {state: value}, or FROM_NAME

    def resolve(self, states: Dict[str, Tuple[Optional[str], Any, Source]]):
        if self.device not in states:
            return None, Source("missing", f"no device {self.device}")
        state, _, source = states[self.device]
        if state is None:
            return None, Source("missing", f"{self.device}: {source.detail}")
        if self.values == FROM_NAME:
            value = wavelength_from_name(state)
            if value is None:
                return None, Source("missing",
                                    f"{self.device} is {state}: no wavelength in the name")
            return value, Source("device", f"{self.device} is {state}")
        if state not in self.values:
            return None, Source("missing", f"{self.device} is {state}: no value for it")
        return self.values[state], Source("device", f"{self.device} is {state}")


def parameter_from(raw: Any):
    """A `Follows` for ``{"device": ...}``, else the camera database's
    `Parameter`: a bare value is fixed, ``{"tag": ...}`` read from the file."""
    if isinstance(raw, dict) and "device" in raw:
        return Follows(device=str(raw["device"]), values=raw.get("values", {}))
    return Parameter.from_dict(raw)


def parameter_to(parameter) -> Any:
    if isinstance(parameter, Follows):
        return {"device": parameter.device, "values": parameter.values}
    if not parameter.tag and not parameter.state:
        return parameter.fixed                 # a bare value reads best
    return parameter.to_dict()


def resolve_parameter(parameter, metadata, states):
    if isinstance(parameter, Follows):
        return parameter.resolve(states)
    value, source = parameter.resolve(metadata)
    if source.kind == "fixed" and not parameter.tag:
        source = Source("fixed", "in the microscope's entry")
    return value, source


# ------------------------------------------------------------------ channels

@dataclass
class Channel:
    """One arm of a split: the filter device it follows and where it lands.

    `side` is a side of the chip, or a table from camera to side: a camera
    mounted turned puts the same arm on another side.  The first channel is
    the one that applies alone -- with the splitting dichroic out the light
    goes the transmitted way, over the whole chip.
    """
    name: str
    filter: str = ""
    side: Any = ""

    def side_for(self, camera: str) -> str:
        if isinstance(self.side, dict):
            return str(self.side.get(camera, ""))
        return str(self.side or "")

    @classmethod
    def from_dict(cls, name: str, raw: Dict[str, Any]) -> "Channel":
        return cls(name=name, filter=str(raw.get("filter", "")), side=raw.get("side", ""))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"filter": self.filter}
        if self.side:
            out["side"] = self.side
        return out


# --------------------------------------------------------------- the entry

def acquisition_date(metadata: Dict[str, Any]) -> Optional[_dt.date]:
    """The day the file was taken, from Micro-Manager's ``StartTime``."""
    text = str(metadata.get("StartTime") or metadata.get("Time") or "")
    found = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", text)
    return _dt.date(*map(int, found.groups())) if found else None


def read_date(text: Any) -> Optional[_dt.date]:
    try:
        return _dt.date.fromisoformat(str(text)[:10])
    except ValueError:
        return None


#: the parts of an entry a version can hold, each a table by name
SECTIONS = ("devices", "parameters", "channels", "cameras")


@dataclass
class Microscope:
    """One microscope: how to recognise it, its devices, and its numbers.

    `identify` are tags that must all match, taken from devices that stay
    with the microscope (a laser's serial number), not from the camera,
    which moves between microscopes.  `cameras` are the ones it has had,
    by their name in the camera database, each with the parameters that
    depend on the pair -- the pixel, which the cylindrical lens changes and
    the camera's mounting turns -- overriding the microscope's.
    `versions` are what the entry said before it was last changed, each with
    the day it applied until (`as_of`).
    """
    name: str
    identify: Dict[str, str] = field(default_factory=dict)
    comment: str = ""
    devices: Dict[str, Device] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    channels: List[Channel] = field(default_factory=list)
    cameras: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    versions: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------ identifying
    def identifies(self, metadata: Dict[str, Any]) -> bool:
        return bool(self.identify) and all(
            str(metadata.get(tag, "")).strip() == str(value).strip()
            for tag, value in self.identify.items())

    # -------------------------------------------------------------- versions
    def as_of(self, day: Optional[_dt.date]) -> "Microscope":
        """The entry as it stood on `day`: the current one, with every
        version that was still in force then laid over it, newest first.

        A version holds only what changed, section by section and by name;
        a name set to None did not exist yet (a device added later).
        """
        if day is None or not self.versions:
            return self
        older = sorted((v for v in self.versions
                        if (read_date(v.get("until")) or _dt.date.min) > day),
                       key=lambda v: read_date(v.get("until")), reverse=True)
        if not older:
            return self
        raw = self.to_dict()
        for version in older:
            for section in SECTIONS:
                if section not in version:
                    continue
                table = raw.setdefault(section, {})
                for key, value in version[section].items():
                    if value is None:
                        table.pop(key, None)
                    else:
                        table[key] = copy.deepcopy(value)
        raw.pop("versions", None)
        return Microscope.from_dict(raw)

    def until(self, day: Optional[_dt.date]) -> Optional[str]:
        """The `until` of the version `day` falls in, or None for the
        current entry."""
        if day is None:
            return None
        dates = sorted(d for d in (read_date(v.get("until")) for v in self.versions)
                       if d is not None and d > day)
        return dates[0].isoformat() if dates else None

    # ------------------------------------------------------------- resolving
    def device_states(self, metadata: Dict[str, Any]):
        return {name: device.state(metadata) for name, device in self.devices.items()}

    def resolve(self, metadata: Dict[str, Any], camera: str = "",
                channels: int = 1) -> "MicroscopeResolution":
        """Every parameter of this entry for this file, with its source.

        `camera` is the camera database's name for the camera that took the
        file; its parameters override the microscope's.  `channels` is 1 or
        2, chosen by whoever asks: nothing records whether the splitting
        dichroic is in.
        """
        day = acquisition_date(metadata)
        entry = self.as_of(day)
        states = entry.device_states(metadata)
        parameters = dict(entry.parameters)
        camera_key = entry.camera_key(camera)
        if camera_key is not None:
            parameters.update(entry.cameras[camera_key])
        values: Dict[str, Any] = {}
        sources: Dict[str, Source] = {}
        for name, parameter in parameters.items():
            value, source = resolve_parameter(parameter_from(parameter), metadata, states)
            if camera_key is not None and name in entry.cameras[camera_key]:
                source = Source(source.kind, f"{source.detail}, with {camera_key}"
                                if source.detail else f"with {camera_key}")
            values[name] = value
            sources[name] = source
        resolved = MicroscopeResolution(
            microscope=self, version=self.until(day), camera=camera_key or camera,
            camera_listed=camera_key is not None or not camera,
            states=states, values=values, sources=sources, metadata=metadata)
        resolved.channels = entry.resolve_channels(resolved, metadata, states, channels,
                                                   camera_key or camera)
        return resolved

    def camera_key(self, camera: str) -> Optional[str]:
        """The entry's name for `camera`: by name, or by a former name of
        that camera in the camera database (an entry saved before a rename)."""
        if not camera:
            return None
        if camera in self.cameras:
            return camera
        from .camera_db import database
        found = database().get(camera)
        if found is None:
            return None
        return next((k for k in self.cameras if found.answers_to(k)), None)

    def resolve_channels(self, resolved, metadata, states, count: int, camera: str):
        """A wavelength per channel used.

        Dual: each arm's filter, on its side.  Single: the first arm, over
        the whole chip.  An arm whose filter gives no wavelength (``Empty``)
        takes the microscope's `wavelength_nm`, which follows the filter in
        the common path where there is one.
        """
        if not self.channels:
            return [ChannelResolution(name="", side="", wavelength_nm=resolved.values.get(
                "wavelength_nm"), source=resolved.sources.get(
                    "wavelength_nm", Source("missing", "no wavelength_nm")))]
        out = []
        for channel in self.channels[:max(1, count)]:
            value, source = (Follows(channel.filter, FROM_NAME).resolve(states)
                             if channel.filter else
                             (None, Source("missing", "no filter device")))
            if value is None and resolved.values.get("wavelength_nm") is not None:
                value = resolved.values["wavelength_nm"]
                source = Source(resolved.sources["wavelength_nm"].kind,
                                f"{source.detail}; the microscope's: "
                                f"{resolved.sources['wavelength_nm'].detail}")
            out.append(ChannelResolution(name=channel.name,
                                         side=channel.side_for(camera) if count > 1 else "",
                                         wavelength_nm=value, source=source))
        return out

    # ------------------------------------------------------------- as a file
    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Microscope":
        channels = raw.get("channels") or {}
        return cls(
            name=str(raw.get("name", "")),
            identify={str(k): str(v) for k, v in (raw.get("identify") or {}).items()},
            comment=str(raw.get("comment", "")),
            devices={str(k): Device.from_dict(str(k), v)
                     for k, v in (raw.get("devices") or {}).items()},
            parameters=dict(raw.get("parameters") or {}),
            channels=[Channel.from_dict(str(k), v) for k, v in channels.items()],
            cameras={str(k): dict(v or {}) for k, v in (raw.get("cameras") or {}).items()},
            versions=[dict(v) for v in (raw.get("versions") or [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name}
        if self.comment:
            out["comment"] = self.comment
        out["identify"] = dict(self.identify)
        out["devices"] = {k: d.to_dict() for k, d in self.devices.items()}
        out["parameters"] = {k: copy.deepcopy(v) for k, v in self.parameters.items()}
        if self.channels:
            out["channels"] = {c.name: c.to_dict() for c in self.channels}
        if self.cameras:
            out["cameras"] = copy.deepcopy(self.cameras)
        if self.versions:
            out["versions"] = copy.deepcopy(self.versions)
        return out


@dataclass
class ChannelResolution:
    name: str
    side: str
    wavelength_nm: Optional[float]
    source: Source


@dataclass
class MicroscopeResolution:
    """What a microscope entry made of one file.

    `states` is every device's state by name (with what the file read and
    from where): kept with the table, because a calibration is matched to
    data by it -- beads taken with the cylindrical lens in do not describe
    a movie taken with it out.
    """
    microscope: Optional[Microscope] = None
    version: Optional[str] = None
    camera: str = ""
    camera_listed: bool = True
    states: Dict[str, Tuple[Optional[str], Any, Source]] = field(default_factory=dict)
    values: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, Source] = field(default_factory=dict)
    channels: List[ChannelResolution] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.microscope.name if self.microscope else ""

    @property
    def pixel_nm(self) -> Optional[Tuple[float, float]]:
        value = self.values.get("pixel_nm")
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return float(value[0]), float(value[-1])
        return float(value), float(value)

    def state_names(self) -> Dict[str, Optional[str]]:
        """``{device: state}``, what is kept with a table."""
        return {name: state for name, (state, _, _) in self.states.items()}

    def warnings(self) -> List[str]:
        out = []
        if self.microscope is not None and not self.camera_listed:
            out.append(f"{self.camera} is not one of {self.name}'s cameras: "
                       f"the pixel is the camera database's")
        return out

    def rows(self) -> List[tuple]:
        """``(what, value, source)``, devices then parameters, for showing."""
        out = [(f"device: {name}", state if state is not None else raw, str(source))
               for name, (state, raw, source) in self.states.items()]
        out += [(name, self.values.get(name), str(self.sources.get(name, "")))
                for name in self.values]
        out += [(f"channel: {c.name or 'single'}"
                 + (f" ({c.side})" if c.side else ""), c.wavelength_nm, str(c.source))
                for c in self.channels]
        return out


# ---------------------------------------------------------------- database

class MicroscopeDatabase:
    """The microscopes known here: the shipped list, plus the user's own."""

    def __init__(self, microscopes: Sequence[Microscope] = (), sources: Sequence[Path] = ()):
        self.microscopes: List[Microscope] = list(microscopes)
        self.sources: List[Path] = list(sources)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], source: Optional[Path] = None):
        found = [Microscope.from_dict(m) for m in (raw.get("microscopes") or [])
                 if isinstance(m, dict) and m.get("name")]
        return cls(found, [source] if source else [])

    @classmethod
    def load(cls, path) -> "MicroscopeDatabase":
        path = Path(path)
        return cls.from_dict(read_file(path), path)

    def save(self, path) -> Path:
        """Write the microscopes into `path`, keeping its cameras."""
        return write_section(path, "microscopes", [m.to_dict() for m in self.microscopes])

    def __len__(self) -> int:
        return len(self.microscopes)

    def names(self) -> List[str]:
        return [m.name for m in self.microscopes]

    def get(self, name: str) -> Optional[Microscope]:
        return next((m for m in self.microscopes if m.name == name), None)

    def joined_with(self, other: "MicroscopeDatabase") -> "MicroscopeDatabase":
        """This database with `other`'s microscopes added, theirs winning by name."""
        kept = [m for m in self.microscopes if other.get(m.name) is None]
        return MicroscopeDatabase(list(other.microscopes) + kept,
                                  list(other.sources) + list(self.sources))

    def identify(self, metadata: Dict[str, Any]) -> Optional[Microscope]:
        return next((m for m in self.microscopes if m.identifies(metadata)), None)

    def resolve(self, metadata: Dict[str, Any], microscope: str = "", camera: str = "",
                channels: int = 1) -> MicroscopeResolution:
        """This file's microscope parameters.

        `microscope` names one instead of recognising it -- a microscope that
        cannot be recognised is chosen, never guessed.  `camera` is the camera
        database's name for the file's camera; without one it is looked up.
        """
        metadata = dict(metadata or {})
        found = self.get(microscope) if microscope else self.identify(metadata)
        if found is None:
            return MicroscopeResolution(metadata=metadata)
        if not camera:
            from .camera_db import database
            known = database().identify(metadata)
            camera = known.name if known is not None else ""
        return found.resolve(metadata, camera=camera, channels=channels)


# ------------------------------------------------------------------ the default

_cache: Optional[MicroscopeDatabase] = None
_revision = 0


def database(reload: bool = False) -> MicroscopeDatabase:
    """The shipped microscopes joined with the user's, cached."""
    global _cache, _revision
    if _cache is not None and not reload:
        return _cache
    _revision += 1
    found = MicroscopeDatabase()
    for path in (SHIPPED, user_file()):
        if not path.exists():
            continue
        try:
            found = found.joined_with(MicroscopeDatabase.load(path))
        except (OSError, ValueError) as error:
            import warnings
            warnings.warn(f"{path} could not be read: {error}", stacklevel=2)
    _cache = found
    return _cache


def revision() -> int:
    return _revision


# ------------------------------------------------------------------ saving

def changes(old: Microscope, new: Microscope) -> Dict[str, Dict[str, Any]]:
    """What `old` said that `new` does not, section by section and by name:
    a version.  A name `new` added is None in it -- it did not exist yet."""
    before, after = old.to_dict(), new.to_dict()
    out: Dict[str, Dict[str, Any]] = {}
    for section in SECTIONS:
        a, b = before.get(section) or {}, after.get(section) or {}
        diff = {k: copy.deepcopy(a.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
        if diff:
            out[section] = dict(sorted(diff.items()))
    return out


def save_microscope(microscope: Microscope, path=None, until: Any = None,
                    previous: Optional[Microscope] = None) -> Path:
    """Add this microscope to the user's file, or replace it there by name,
    keeping what the entry said before as a version.

    `previous` is the entry being changed -- the user's or the shipped one
    of the same name if not given.  What it said that the new entry does
    not is kept with `until`, the day it stopped applying: the EMU
    configuration's date for an import, else today.  An unchanged entry
    adds no version.
    """
    path = Path(path) if path is not None else user_file()
    mine = MicroscopeDatabase.load(path) if path.exists() else MicroscopeDatabase()
    if previous is None:
        previous = mine.get(microscope.name) or database().get(microscope.name)
    microscope = copy.deepcopy(microscope)
    if previous is not None:
        diff = changes(previous, microscope)
        if diff:
            day = until or _dt.date.today()
            day = day.isoformat() if isinstance(day, _dt.date) else str(day)[:10]
            microscope.versions = list(previous.versions) + [dict(until=day, **diff)]
        else:
            microscope.versions = list(previous.versions)
    for i, existing in enumerate(mine.microscopes):
        if existing.name == microscope.name:
            mine.microscopes[i] = microscope
            break
    else:
        mine.microscopes.append(microscope)
    mine.save(path)
    database(reload=True)
    return path
