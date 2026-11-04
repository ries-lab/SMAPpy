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

#: chip halves a channel of a split lands on, in the words of the dual-channel
#: calibration's ``main_channel`` (`calibrate.dual`), so the two meet
SIDES = ("left", "right", "upper", "lower")


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


def filter_wavelength(name: Any, filters: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """The emission a filter passes, nm: the entry's table first (``filters``: a typed
    ``wavelength_nm``, or the centre of a typed ``band_nm``), else the name (`wavelength_from_name`).
    The table is for a filter whose name is not its band -- a long pass, an empty slot behind the
    splitting dichroic -- whose effective band is typed once."""
    typed = (filters or {}).get(str(name))
    if isinstance(typed, dict):
        if typed.get("wavelength_nm") is not None:
            return float(typed["wavelength_nm"])
        band = typed.get("band_nm")
        if band and len(band) == 2 and None not in band:
            return (float(band[0]) + float(band[1])) / 2
    return wavelength_from_name(name)


@dataclass
class Follows:
    """A parameter that follows one device: a value per state of it."""
    device: str
    values: Any = field(default_factory=dict)      # {state: value}, or FROM_NAME

    def resolve(self, states: Dict[str, Tuple[Optional[str], Any, Source]],
                filters: Optional[Dict[str, Any]] = None):
        if self.device not in states:
            return None, Source("missing", f"no device {self.device}")
        state, _, source = states[self.device]
        if state is None:
            return None, Source("missing", f"{self.device}: {source.detail}")
        if self.values == FROM_NAME:
            value = filter_wavelength(state, filters)
            if value is None:
                return None, Source("missing",
                                    f"{self.device} is {state}: no wavelength in the name, "
                                    f"and no band typed for it")
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


def resolve_parameter(parameter, metadata, states, filters=None):
    if isinstance(parameter, Follows):
        return parameter.resolve(states, filters)
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

    `region` is where it lands exactly, for a split into more than two --
    the four phase images of a 4Pi microscope: ``[x, y, width, height]`` on
    the chip, in whole pixels, or a table from camera to that; `mirror` is
    ``"x"``, ``"y"`` or ``"xy"`` for an arm the optics flip.  That is the
    geometry, coarse and stable; the sub-pixel registration between the
    channels belongs to a calibration (`smappy.channels`).
    """
    name: str
    filter: str = ""
    side: Any = ""
    region: Any = None
    mirror: str = ""

    def side_for(self, camera: str) -> str:
        if isinstance(self.side, dict):
            return str(self.side.get(camera, ""))
        return str(self.side or "")

    def region_for(self, camera: str) -> Optional[List[int]]:
        region = self.region.get(camera) if isinstance(self.region, dict) else self.region
        return [int(v) for v in region] if region else None

    @classmethod
    def from_dict(cls, name: str, raw: Dict[str, Any]) -> "Channel":
        return cls(name=name, filter=str(raw.get("filter", "")), side=raw.get("side", ""),
                   region=raw.get("region"), mirror=str(raw.get("mirror", "")))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"filter": self.filter}
        if self.side:
            out["side"] = self.side
        if self.region:
            out["region"] = copy.deepcopy(self.region)
        if self.mirror:
            out["mirror"] = self.mirror
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


#: the parts of an entry a version can hold: tables by name, and the scalars
SECTIONS = ("devices", "parameters", "channels", "pupils", "objective", "filters")
SCALARS = ("camera", "main_channel", "split_position", "layout")

#: what the objective section holds: fixed for an entry (we do not swap objectives), and no
#: magnification -- the pixel on the sample is the camera's, measured, per state of the 3D lens
#: where it differs; the nominal magnification says little with relay optics behind it
OBJECTIVE = ("name", "NA", "immersion", "n_immersion")

#: the colours a kept PSF is told apart by (`colour_of`): an objective's aberrations change
#: with the colour, not between two far-red filters
COLOURS = ((500.0, "blue"), (570.0, "green"), (640.0, "red"), (float("inf"), "far red"))


def colour_of(wavelength_nm: Any) -> str:
    """``blue`` below 500 nm, ``green`` to 570, ``red`` to 640, ``far red`` above; "" for none."""
    try:
        value = float(wavelength_nm)
    except (TypeError, ValueError):
        return ""
    return next(name for edge, name in COLOURS if value < edge)


def same_camera(a: str, b: str) -> bool:
    """Whether two names are one camera of the camera database (a former name counts)."""
    if not a or not b:
        return False
    if a == b:
        return True
    from .camera_db import database
    found = database().get(a)
    return found is not None and found.answers_to(b)


@dataclass
class Microscope:
    """One microscope: how to recognise it, its devices, and its numbers.

    `identify` are tags that must all match, taken from devices that stay
    with the microscope (a laser's serial number), not from the camera,
    which moves between microscopes.  `camera` is the one it is set up with,
    by its name in the camera database: the pixel per state of the 3D lens
    and the sides of the split are for that camera, so a new camera is a
    reprogramming of the microscope, the setup before kept as a version.
    `versions` are what the entry said before it was last changed, each with
    the day it applied until (`as_of`).
    """
    name: str
    identify: Dict[str, str] = field(default_factory=dict)
    comment: str = ""
    devices: Dict[str, Device] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    channels: List[Channel] = field(default_factory=list)
    camera: str = ""
    versions: List[Dict[str, Any]] = field(default_factory=list)
    #: the objective's pupil, learnt from beads, by the state of the 3D lens (`pupil_key`): the PSF as a
    #: bead model describes itself (`uipsf.VectorPSF.describe`: kind, phase in nm, γ or the magnitude's
    #: Zernikes, σ), what in situ learning holds and starts from (`pupil_of`)
    pupils: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: the objective (`OBJECTIVE`): its name, NA, immersion and the immersion's index -- the NA
    #: and index uiPSF reads, where an older entry has them under `parameters`
    objective: Dict[str, Any] = field(default_factory=dict)
    #: filters whose name is not their band, by name: ``{"wavelength_nm": 690}`` or
    #: ``{"band_nm": [665, 720]}`` (`filter_wavelength`)
    filters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: the arm the others are registered onto, and whose PSF is a dual calibration's main; "" the
    #: first arm
    main_channel: str = ""
    #: where the two halves of a split meet on the chip, the first pixel of the second half along
    #: the split's axis -- a number, or a table from camera to one; None the middle of the frame as
    #: acquired, which is how the lab centres its camera ROI on the split
    split_position: Any = None
    #: how a two-arm split lies when the arms' sides are not known yet (``right-left mirrored``):
    #: the optics say it, which arm is on which side is read from data later; the arms' `side`s win
    layout: str = ""

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
            for scalar in SCALARS:
                if scalar in version:
                    raw[scalar] = version[scalar]
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

    def setups(self):
        """``(until, entry)`` of every setup this microscope has had, newest first: the current
        one (until None), then each version's as it stood."""
        yield None, self
        for day in sorted({d for d in (read_date(v.get("until")) for v in self.versions) if d},
                          reverse=True):
            yield day.isoformat(), self.as_of(day - _dt.timedelta(days=1))

    def with_camera(self, camera: str):
        """``(until, entry)`` of the newest setup with `camera`, or None."""
        return next(((until, entry) for until, entry in self.setups()
                     if same_camera(camera, entry.camera)), None)

    # ------------------------------------------------------------- resolving
    def followed(self) -> List[str]:
        """The devices a parameter or a channel follows: the ones whose state changes the PSF or
        its scale, which a calibration and its data must agree on.  A laser or a bright-field
        lamp is read and kept, and never compared."""
        names = {raw["device"] for raw in self.parameters.values()
                 if isinstance(raw, dict) and "device" in raw}
        names |= {c.filter for c in self.channels if c.filter}
        return [name for name in self.devices if name in names]

    def device_states(self, metadata: Dict[str, Any]):
        return {name: device.state(metadata) for name, device in self.devices.items()}

    def resolve(self, metadata: Dict[str, Any], camera: str = "",
                channels: int = 1) -> "MicroscopeResolution":
        """Every parameter of this entry for this file, with its source.

        The setup is the one of the file's day (`as_of`).  `camera` is the camera that took the
        file, as the camera database recognised it: when it is not that setup's, the file is
        read with the newest setup that had it -- a camera swapped before the entry was -- and
        a warning says so.  `channels` is 1 or 2, chosen by whoever asks: nothing records
        whether the splitting dichroic is in.
        """
        day = acquisition_date(metadata)
        entry, version, note = self.as_of(day), self.until(day), ""
        if camera and entry.camera and not same_camera(camera, entry.camera):
            dated, other = entry.camera, self.with_camera(camera)
            if other is not None:
                version, entry = other
                note = (f"taken with {camera}, not {dated}: read as {self.name} was set up "
                        f"with {camera}")
            else:
                note = (f"taken with {camera}, which {self.name} was never set up with: its pixel "
                        f"and its split may not fit")
        states = entry.device_states(metadata)
        values: Dict[str, Any] = {}
        sources: Dict[str, Source] = {}
        for name, parameter in entry.parameters.items():
            values[name], sources[name] = resolve_parameter(parameter_from(parameter), metadata,
                                                            states, entry.filters)
        # the objective's section wins over the same names in `parameters`, which older
        # entries used for them
        for name in OBJECTIVE:
            key = "objective" if name == "name" else name
            if entry.objective.get(name) not in (None, ""):
                values[key] = entry.objective[name]
                sources[key] = Source("fixed", "the entry's objective")
        used = camera or entry.camera
        resolved = MicroscopeResolution(
            microscope=self, version=version, camera=used, camera_note=note,
            states=states, values=values, sources=sources, metadata=metadata)
        resolved.channels = entry.resolve_channels(resolved, metadata, states, channels, used)
        return resolved

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
        for channel in self.ordered_channels()[:max(1, count)]:
            value, source = (Follows(channel.filter, FROM_NAME).resolve(states, self.filters)
                             if channel.filter else
                             (None, Source("missing", "no filter device")))
            if value is None and resolved.values.get("wavelength_nm") is not None:
                value = resolved.values["wavelength_nm"]
                source = Source(resolved.sources["wavelength_nm"].kind,
                                f"{source.detail}; the microscope's: "
                                f"{resolved.sources['wavelength_nm'].detail}")
            out.append(ChannelResolution(name=channel.name,
                                         side=channel.side_for(camera) if count > 1 else "",
                                         wavelength_nm=value, source=source,
                                         mirror=channel.mirror if count > 1 else ""))
        return out

    # ------------------------------------------------------------- the arrangement
    def ordered_channels(self) -> List[Channel]:
        """The arms, the main one first (`main_channel`), the rest as entered."""
        main = next((c for c in self.channels if c.name == self.main_channel), None)
        if main is None:
            return list(self.channels)
        return [main] + [c for c in self.channels if c is not main]

    def split_for(self, camera: str = "") -> Optional[float]:
        """The split's position on `camera`'s chip, or None for the middle of the frame."""
        position = self.split_position
        if isinstance(position, dict):
            position = position.get(camera or self.camera)
        return None if position in (None, "") else float(position)

    def arrangement(self, camera: str = "") -> Optional[Dict[str, Any]]:
        """How a two-arm split lies on `camera`'s chip, in the words of the chip settings
        (`calibrate.dual.RegisterSettings`): ``layout`` (``right-left`` or ``up-down``),
        ``mirrored`` (one arm flipped against the other, along the split's axis), the
        ``main_channel`` half (``left``, ``right``, ``upper``, ``lower``) and the
        ``split_position`` (None: the middle).  None for an entry without two arms on two sides --
        a 4Pi's regions are `channel_layout`'s."""
        used = camera or self.camera
        arms = self.ordered_channels()[:2]
        sides = [arm.side_for(used) for arm in arms]
        if len(arms) == 2 and set(sides) == {"left", "right"}:
            layout = "right-left"
        elif len(arms) == 2 and set(sides) == {"upper", "lower"}:
            layout = "up-down"
        elif self.layout:
            # the arrangement without the arms' sides: the main half the left or upper one, as a
            # dual calibration's *auto* takes it
            layout = self.layout.replace(" mirrored", "")
            mirrored = self.layout.endswith("mirrored")
            return {"layout": self.layout, "mirrored": mirrored,
                    "main_channel": "left" if layout == "right-left" else "upper",
                    "split_position": self.split_for(used)}
        else:
            return None
        mirrored = set(arms[0].mirror) != set(arms[1].mirror)
        return {"layout": layout + (" mirrored" if mirrored else ""), "mirrored": mirrored,
                "main_channel": sides[0], "split_position": self.split_for(used)}

    def split_layout(self, roi, camera: str = "", name: str = ""):
        """The two arms as a `channels.ChannelLayout` on the chip, for frames with camera ROI
        ``roi`` (x0, y0, width, height): the frame cut at the split -- the entry's position, or
        the frame's middle -- the main arm first, the other flipped where the optics flip it.
        None without an `arrangement`."""
        from .channels import split_halves
        found = self.arrangement(camera)
        if found is None:
            return None
        try:
            return split_halves(found, roi, name or self.name)
        except ValueError as error:
            raise ValueError(f"{self.name}'s {error}") from None

    def channel_layout(self, camera: str = "", name: str = ""):
        """The channels' rectangles on the chip as a `channels.ChannelLayout`,
        for every channel that has a `region` -- None when none has one.
        Chip coordinates: `ChannelLayout.on_camera` puts it on a ROI's
        frames."""
        from .channels import ChannelLayout
        used = camera or self.camera
        placed = [(c, c.region_for(used)) for c in self.channels]
        placed = [(c, r) for c, r in placed if r]
        if not placed:
            return None
        return ChannelLayout([tuple(r) for _, r in placed],
                             [("x" in c.mirror, "y" in c.mirror) for c, _ in placed],
                             name=name or self.name)

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
            camera=str(raw.get("camera", "")),
            versions=[dict(v) for v in (raw.get("versions") or [])],
            pupils={str(k): dict(v) for k, v in (raw.get("pupils") or {}).items()},
            objective=dict(raw.get("objective") or {}),
            filters={str(k): dict(v) for k, v in (raw.get("filters") or {}).items()},
            main_channel=str(raw.get("main_channel") or ""),
            split_position=copy.deepcopy(raw.get("split_position")),
            layout=str(raw.get("layout") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name}
        if self.comment:
            out["comment"] = self.comment
        out["identify"] = dict(self.identify)
        if self.objective:
            out["objective"] = dict(self.objective)
        out["devices"] = {k: d.to_dict() for k, d in self.devices.items()}
        out["parameters"] = {k: copy.deepcopy(v) for k, v in self.parameters.items()}
        if self.channels:
            out["channels"] = {c.name: c.to_dict() for c in self.channels}
        if self.main_channel:
            out["main_channel"] = self.main_channel
        if self.layout:
            out["layout"] = self.layout
        if self.split_position not in (None, ""):
            out["split_position"] = copy.deepcopy(self.split_position)
        if self.filters:
            out["filters"] = copy.deepcopy(self.filters)
        if self.camera:
            out["camera"] = self.camera
        if self.pupils:
            out["pupils"] = copy.deepcopy(self.pupils)
        if self.versions:
            out["versions"] = copy.deepcopy(self.versions)
        return out

    # ------------------------------------------------------------- the objective
    def lens_device(self) -> Optional[str]:
        """The device the PSF's shape follows -- the 3D lens: what ``modality`` or ``astigmatism_nm``
        follows -- or None for a microscope without one."""
        for name in ("modality", "astigmatism_nm"):
            raw = self.parameters.get(name)
            if isinstance(raw, dict) and "device" in raw:
                return str(raw["device"])
        return None

    def pupil_key(self, states: Dict[str, Any], wavelength_nm: Any = None) -> str:
        """Where a pupil learnt in ``states`` ({device: state}) at ``wavelength_nm`` is kept:
        ``3DA: on · far red``, ``3DA: on`` without a wavelength, ``all`` (``all · red``) for a
        microscope without a 3D lens.  By colour (`colour_of`), not by filter: the objective's
        aberrations move between green and far red, not between 676/37 and 700/100."""
        device = self.lens_device()
        key = "all" if device is None or states.get(device) is None else f"{device}: {states[device]}"
        colour = colour_of(wavelength_nm)
        return f"{key} \u00b7 {colour}" if colour else key

    def pupil_for(self, states: Dict[str, Any], wavelength_nm: Any = None) -> Optional[Dict[str, Any]]:
        """The pupil kept for ``states`` in this colour; else one kept for the state in no colour
        (before colours were told apart), then for every state.  Never another colour's: a red
        model holds the wrong aberrations for far red.  One kept before pupils described their
        kind (Zernike magnitude dicts, before November 2026) is none: no legacy, it is kept again
        from a new bead model (`keep_pupil`)."""
        keys = [self.pupil_key(states, wavelength_nm), self.pupil_key(states)]
        if colour_of(wavelength_nm):
            keys.append(f"all \u00b7 {colour_of(wavelength_nm)}")
        keys.append("all")
        found = next((self.pupils[k] for k in keys if k in self.pupils), None)
        return found if found and found.get("kind") else None


@dataclass
class ChannelResolution:
    name: str
    side: str
    wavelength_nm: Optional[float]
    source: Source
    mirror: str = ""


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
    camera_note: str = ""
    #: True when no microscope was recognised and the last one used stands in
    fallback: bool = False
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

    def summary(self) -> Dict[str, Any]:
        """What travels with the file's `CameraMetadata` into the table and
        a calibration: plain values, no sources.  ``channels`` are every arm
        with its side for this camera; single-channel data is the first."""
        values = {k: (list(v) if isinstance(v, tuple) else v)
                  for k, v in self.values.items() if v is not None}
        return {"name": self.name, "version": self.version, "camera": self.camera,
                "fallback": self.fallback,
                "states": self.state_names(), "values": values,
                "followed": self.microscope.followed() if self.microscope else [],
                "channels": [dict({"name": c.name, "side": c.side,
                                   "wavelength_nm": c.wavelength_nm},
                                  **({"mirror": c.mirror} if c.mirror else {}))
                             for c in self.channels]}

    def warnings(self) -> List[str]:
        return [self.camera_note] if self.microscope is not None and self.camera_note else []

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


def one_line(summary: Optional[Dict[str, Any]], short: bool = False) -> str:
    """A microscope summary (`MicroscopeResolution.summary`) in one line, for where a file is opened:
    *M2 · 3DA on · iXon 1 · 127 × 116 nm*; ``short``, the name and the modality alone
    (*M2, astigmatic*).  The switches a parameter follows; the filters are left to the window."""
    if not summary:
        return "no microscope recognised"
    values = summary.get("values") or {}
    if short:
        return ", ".join(str(v) for v in (summary.get("name"), values.get("modality")) if v)
    states = summary.get("states") or {}
    bits = [str(summary.get("name")) + (" (last used: none recognised)"
                                        if summary.get("fallback") else "")]
    for device in summary.get("followed") or []:
        state = states.get(device)
        # the switches (the 3D lens); the filters are in the window, the line stays short
        if state is None or wavelength_from_name(state) or state.lower() in ("none", "empty"):
            continue
        bits.append(f"{device} {state}")
    if summary.get("camera"):
        bits.append(str(summary["camera"]))
    pixel = values.get("pixel_nm")
    if isinstance(pixel, (list, tuple)) and len(pixel) == 2:
        bits.append(f"{pixel[0]:g} \u00d7 {pixel[1]:g} nm" if pixel[0] != pixel[1] else f"{pixel[0]:g} nm")
    return " \u00b7 ".join(bits)


def choices() -> List[Tuple[str, str]]:
    """``(value, label)`` for a microscope chooser: *auto*, then the database's."""
    try:
        names = database().names()
    except Exception:
        names = []
    return [("", "auto")] + [(name, name) for name in names]


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
                channels: int = 1, fallback: str = "") -> MicroscopeResolution:
        """This file's microscope parameters.

        `microscope` names one instead of recognising it -- a microscope that
        cannot be recognised is chosen, never guessed.  `camera` is the camera
        database's name for the file's camera; without one it is looked up.
        """
        metadata = dict(metadata or {})
        found = self.get(microscope) if microscope else self.identify(metadata)
        stand_in = found is None and bool(fallback) and self.get(fallback) is not None
        if stand_in:
            found = self.get(fallback)
        if found is None:
            return MicroscopeResolution(metadata=metadata)
        if not camera:
            from .camera_db import database
            known = database().identify(metadata)
            camera = known.name if known is not None else ""
        resolved = found.resolve(metadata, camera=camera, channels=channels)
        resolved.fallback = stand_in
        return resolved


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


def pupil_of(summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The objective's pupil for a file's microscope (a `MicroscopeResolution.summary`): the entry as
    it stood in the file's setup version, for the state its 3D lens was in; None when not kept."""
    if not summary or not summary.get("name"):
        return None
    entry = database().get(str(summary["name"]))
    if entry is None:
        return None
    until = read_date(summary.get("version")) if summary.get("version") else None
    if until is not None:
        entry = entry.as_of(until - _dt.timedelta(days=1))
    return entry.pupil_for(summary.get("states") or {}, emission_of(summary))


def _summary_of(microscope) -> Optional[Dict[str, Any]]:
    """A summary from a summary, a `MicroscopeResolution` or a name."""
    if microscope is None or microscope == "":
        return None
    if isinstance(microscope, MicroscopeResolution):
        return microscope.summary() if microscope.microscope is not None else None
    if isinstance(microscope, str):
        return {"name": microscope}
    return microscope if isinstance(microscope, dict) else None


def name_of(microscope) -> str:
    if isinstance(microscope, Microscope):
        return microscope.name
    found = _summary_of(microscope)
    return str(found.get("name") or "") if found else ""


def arrangement_of(microscope, camera: str = "") -> Optional[Dict[str, Any]]:
    """How the chip is split on the microscope a file was taken on (`Microscope.arrangement`):
    ``microscope`` a summary, a `MicroscopeResolution`, an entry's name or the entry itself; the
    setup of the file's version, the camera it names unless ``camera`` is given.  None when no
    entry says."""
    if isinstance(microscope, Microscope):
        return microscope.arrangement(camera)
    summary = _summary_of(microscope)
    if not summary or not summary.get("name"):
        return None
    entry = database().get(str(summary["name"]))
    if entry is None:
        return None
    until = read_date(summary.get("version")) if summary.get("version") else None
    if until is not None:
        entry = entry.as_of(until - _dt.timedelta(days=1))
    return entry.arrangement(camera or str(summary.get("camera") or ""))


def emission_of(summary: Optional[Dict[str, Any]]) -> Optional[float]:
    """A summary's emission, nm: the main arm's, else the microscope's ``wavelength_nm``."""
    if not summary:
        return None
    for channel in summary.get("channels") or []:
        if channel.get("wavelength_nm") is not None:
            return float(channel["wavelength_nm"])
    value = (summary.get("values") or {}).get("wavelength_nm")
    return None if value is None else float(value)


def gamma_of(summary: Optional[Dict[str, Any]],
             wavelength_nm: Optional[float] = None) -> Tuple[Optional[float], Optional[float], str]:
    """(the minimal pupil's γ for a file's microscope, the NA learnt with it or None, where they came from),
    or (None, None, "").

    γ is the objective's apodisation: it belongs to the microscope, perhaps a little to the colour, and to
    nothing else -- not the 3D lens, not the sample (the user's decision, November 2026).  So it is read,
    in order, from the entry's ``gamma`` parameter (a bare value, or one following the emission filter:
    a value per colour, resolved with the file like any parameter), else from the minimal pupils kept with
    the entry under any state of its 3D lens, the one whose emission is nearest ``wavelength_nm``, with the NA
    it learnt, if it did: γ and the aperture trade (on 4Pi beads 0.40 at NA 1.35, 0.24 at a learnt 1.31), so
    one is held with the other."""
    if not summary or not summary.get("name"):
        return None, None, ""
    name = str(summary["name"])
    value = (summary.get("values") or {}).get("gamma")
    if value is not None:
        return float(value), None, f"{name}'s entry"
    entry = database().get(name)
    if entry is None:
        return None, None, ""
    until = read_date(summary.get("version")) if summary.get("version") else None
    if until is not None:
        entry = entry.as_of(until - _dt.timedelta(days=1))
    kept = [p for p in entry.pupils.values() if p.get("kind") == "minimal" and p.get("gamma") is not None]
    if not kept:
        return None, None, ""
    if wavelength_nm is not None:
        kept.sort(key=lambda p: abs(float(p.get("wavelength_nm") or wavelength_nm) - float(wavelength_nm)))
    found = kept[0]
    at = f" at {float(found['wavelength_nm']):g} nm" if found.get("wavelength_nm") else ""
    na = found.get("na")
    return float(found["gamma"]), None if na is None else float(na), f"{name}'s kept pupil{at}"


def keep_pupil(summary: Dict[str, Any], pupil: Dict[str, Any], path=None) -> tuple:
    """Keep ``pupil`` with the microscope ``summary`` names, for the state of its 3D lens, in the
    user's file (`save_microscope`: the entry before kept as a dated version).  Returns (the key, the
    file).  Only for the current setup: a pupil of an older one would describe another camera."""
    if not summary or not summary.get("name"):
        raise ValueError("no microscope recognised for these beads: choose one first")
    if summary.get("version"):
        raise ValueError(f"these beads were taken with {summary['name']}'s setup until "
                         f"{summary['version']}, not its current one")
    entry = database().get(str(summary["name"]))
    if entry is None:
        raise ValueError(f"{summary['name']} is not in the microscope database")
    if not pupil.get("kind"):
        raise ValueError("a pupil to keep describes its kind (`uipsf.VectorPSF.describe`)")
    entry = copy.deepcopy(entry)
    key = entry.pupil_key(summary.get("states") or {},
                          pupil.get("wavelength_nm") or emission_of(summary))
    entry.pupils[key] = dict(pupil)
    return key, save_microscope(entry, path)


def revision() -> int:
    return _revision


# ------------------------------------------------------------------ saving

def changes(old: Microscope, new: Microscope) -> Dict[str, Dict[str, Any]]:
    """What `old` said that `new` does not, section by section and by name:
    a version.  A name `new` added is None in it -- it did not exist yet."""
    before, after = old.to_dict(), new.to_dict()
    out: Dict[str, Any] = {scalar: before.get(scalar, "") for scalar in SCALARS
                           if before.get(scalar, "") != after.get(scalar, "")}
    for section in SECTIONS:
        a, b = before.get(section) or {}, after.get(section) or {}
        diff = {k: copy.deepcopy(a.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
        if diff:
            out[section] = dict(sorted(diff.items()))
    return out


def save_microscope(microscope: Microscope, path=None, until: Any = None,
                    previous: Optional[Microscope] = None,
                    versions: Optional[List[Dict[str, Any]]] = None) -> Path:
    """Add this microscope to the user's file, or replace it there by name,
    keeping what the entry said before as a version.

    `previous` is the entry being changed -- the user's or the shipped one
    of the same name if not given.  What it said that the new entry does
    not is kept with `until`, the day it stopped applying: the EMU
    configuration's date for an import, else today.  An unchanged entry
    adds no version.  `versions` are the ones to keep, edited (the manager's
    *versions* tab: a day moved, one removed); None keeps the stored ones.
    """
    path = Path(path) if path is not None else user_file()
    mine = MicroscopeDatabase.load(path) if path.exists() else MicroscopeDatabase()
    if previous is None:
        previous = mine.get(microscope.name) or database().get(microscope.name)
    microscope = copy.deepcopy(microscope)
    if previous is not None:
        kept = list(previous.versions if versions is None else versions)
        diff = changes(previous, microscope)
        if diff:
            day = until or _dt.date.today()
            day = day.isoformat() if isinstance(day, _dt.date) else str(day)[:10]
            microscope.versions = kept + [dict(until=day, **diff)]
        else:
            microscope.versions = kept
    for i, existing in enumerate(mine.microscopes):
        if existing.name == microscope.name:
            mine.microscopes[i] = microscope
            break
    else:
        mine.microscopes.append(microscope)
    mine.save(path)
    database(reload=True)
    return path
