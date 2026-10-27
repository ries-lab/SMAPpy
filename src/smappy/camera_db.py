"""The camera database: which camera took this file, and what it was set to.

Micro-Manager records a great deal about an acquisition and not the two
numbers a fit needs most.  The e-/ADU conversion is never in the metadata; it
depends on the readout mode, which *is*.  An iXon reports no baseline at all,
and an EMCCD's conversion changes with the pre-amp gain, the readout rate and
the output amplifier.  So the numbers have to come from somewhere else, chosen
by what the metadata does say.

That is what this is: a list of cameras, each identified by one tag -- a serial
number, a camera ID -- and each carrying, per parameter, where its value comes
from.  Three sources:

* ``fixed``: the same whatever the acquisition (a pixel size, usually);
* ``metadata``: read this tag out of the file and interpret it this way --
  as a number, a yes or no, or through a table from what the tag reads to
  the value (``map``), with a default for a file without the tag;
* ``state``: it depends on the readout mode, and the mode is recognised by a
  set of tags that must all match.

SMAP keeps the same model in ``settings/*_cameras.mat``; this is that concept
in JSON, so that adding a camera is editing a file rather than a MATLAB
struct.  The Ries lab's file was ported once; smappy does not read SMAP's
files.  The cameras are the ``cameras`` list of a lab's ``microscopes.json``,
whose ``microscopes`` list is `microscope_db`'s: one file per lab, shipped
(``data/microscopes.json``) and the user's own.

**Nothing here overrides the file.**  A value the metadata gives is the value
used; the database supplies what the metadata does not carry, and what it
supplies is always attributable -- `Resolution.sources` says, for every
parameter, exactly where the number came from.  That is the whole point of
keeping it: a wrong conversion is invisible in the fit and obvious in the
table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .metadata import CameraMetadata

#: parameters that end up in `CameraMetadata`, and so reach the fit and the
#: table's ``metadata["camera"]``.  Anything else in the database is carried,
#: shown and ignored.  The exposure and the frame time are not read by the
#: fit; they are kept with the table, where everything in seconds reads them.
FITTING_PARAMETERS = ("conversion", "offset", "pixelsize_um", "em_on", "emgain",
                      "read_noise_e", "exposure_ms", "frame_time_ms", "roi")

#: names a parameter was stored under before, read as the current one: a
#: user's ``cameras.json`` written before the frame time had its name
RENAMED = {"frame_interval_ms": "frame_time_ms"}

DATA = Path(__file__).resolve().parent / "data"
SHIPPED = DATA / "microscopes.json"


# --------------------------------------------------------------- reading a tag

def read_number(text: str) -> Optional[float]:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def read_numbers(text: str) -> Optional[List[float]]:
    """``"0-0-512-512"``, ``"[0 0 512 512]"``, ``"0,0,512,512"`` -> four numbers."""
    cleaned = str(text)
    for character in "[]()," + "-":
        cleaned = cleaned.replace(character, " ")
    values = [read_number(part) for part in cleaned.split()]
    found = [v for v in values if v is not None]
    return found or None


def read_boolean(text: str) -> Optional[bool]:
    value = str(text).strip().lower()
    if value in ("true", "yes", "on", "1"):
        return True
    if value in ("false", "no", "off", "0"):
        return False
    number = read_number(text)
    return None if number is None else number > 0


def read_map(text: str, table: Any) -> Any:
    """The value `table` gives for what the tag reads, or None.

    For what depends on a device that is not the camera's readout mode: on
    the Ries lab's M2 the cylindrical lens of the astigmatic 3D path is in or
    out, and with it the pixel is 127 x 116 nm or 127 nm square.  A readout
    mode (`State`) could carry it too, but every mode would then be entered
    twice, once with the lens and once without.
    """
    if not isinstance(table, dict):
        return None
    wanted = str(text).strip()
    for key, value in table.items():
        if str(key).strip() == wanted:
            return value
    return None


READERS = {
    "text": lambda text, argument: str(text).strip(),
    "number": lambda text, argument: read_number(text),
    "numbers": lambda text, argument: read_numbers(text),
    "boolean": lambda text, argument: read_boolean(text),
    "positive": lambda text, argument: (None if read_number(text) is None
                                        else read_number(text) > 0),
    "equals": lambda text, argument: str(text).strip() == str(argument).strip(),
    "not_equals": lambda text, argument: str(text).strip() != str(argument).strip(),
    "map": read_map,
}


@dataclass
class Source:
    """Where one parameter's value came from, in words.

    Carried beside the value rather than derived later: by the time a fit is
    running, "offset = 196" and "offset = 196, because this file was taken
    with 10 MHz readout on gain 1" are the same number and not the same
    answer.
    """
    kind: str                      # "metadata", "state", "fixed", "missing"
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}" if self.detail else self.kind


@dataclass
class Parameter:
    """Where one camera parameter comes from.

    `fixed` is the value, or -- for one read from a tag -- the default for a
    file that does not carry the tag or reads a value the map does not have.
    """
    fixed: Any = None
    tag: str = ""
    read: str = "text"
    argument: Any = None
    state: bool = False

    @classmethod
    def from_dict(cls, raw: Any) -> "Parameter":
        if not isinstance(raw, dict):           # a bare value means "fixed"
            return cls(fixed=raw)
        if raw.get("state"):
            return cls(state=True)
        if "tag" in raw:
            return cls(tag=str(raw["tag"]), read=str(raw.get("read", "text")),
                       argument=raw.get("value"), fixed=raw.get("default"))
        return cls(fixed=raw.get("fixed"))

    def to_dict(self) -> Any:
        if self.state:
            return {"state": True}
        if self.tag:
            out: Dict[str, Any] = {"tag": self.tag, "read": self.read}
            if self.argument is not None:
                out["value"] = self.argument
            if self.fixed is not None:
                out["default"] = self.fixed
            return out
        return {"fixed": self.fixed}

    def resolve(self, metadata: Dict[str, Any], prefix: str = ""):
        """``(value, Source)``; a value of None means it could not be read."""
        if self.state:
            return None, Source("state")
        if self.tag:
            tag = expand(self.tag, prefix)
            if tag not in metadata:
                if self.fixed is not None:
                    return self.fixed, Source("fixed", f"the default: no tag {tag}")
                return None, Source("missing", f"no tag {tag}")
            reader = READERS.get(self.read, READERS["text"])
            value = reader(metadata[tag], self.argument)
            if value is None:
                if self.fixed is not None:
                    return self.fixed, Source(
                        "fixed", f"the default: {tag} is {metadata[tag]!r}")
                return None, Source("missing", f"{tag} is {metadata[tag]!r}")
            return value, Source("metadata", f"{tag} = {metadata[tag]}")
        return self.fixed, Source("fixed", "in the database")


def expand(text: str, prefix: str) -> str:
    """``"{prefix}-Gain"`` with the camera's device prefix filled in.

    A lab that renames its Micro-Manager device -- ``Andor`` to ``Camera-1``
    -- then edits one field instead of every tag of that camera.
    """
    return str(text).replace("{prefix}", prefix) if prefix else str(text)


@dataclass
class State:
    """One readout mode: the tags that identify it, and what they imply."""
    match: Dict[str, str] = field(default_factory=dict)
    values: Dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def matches(self, metadata: Dict[str, Any], prefix: str = "") -> bool:
        if not self.match:
            return False
        for tag, wanted in self.match.items():
            found = metadata.get(expand(tag, prefix))
            if found is None or str(found).strip() != str(wanted).strip():
                return False
        return True

    def describe(self, prefix: str = "") -> str:
        return self.name or ", ".join(f"{expand(k, prefix)} = {v}"
                                      for k, v in self.match.items())

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "State":
        return cls(match={str(k): str(v) for k, v in (raw.get("match") or {}).items()},
                   values=dict(raw.get("values") or {}),
                   name=str(raw.get("name", "")))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.name:
            out["name"] = self.name
        out["match"] = self.match
        out["values"] = self.values
        return out


@dataclass
class Camera:
    """One camera: how to recognise it, and where each of its numbers lives."""
    name: str
    identify: Optional[Parameter] = None      # None: chosen by hand only
    prefix: str = ""
    comment: str = ""
    #: names it went by before: a camera is named for itself (*iXon 1*), and
    #: settings, chains and user files saved under a microscope's name for it
    #: (*M1 Andor*) still find it
    aliases: List[str] = field(default_factory=list)
    parameters: Dict[str, Parameter] = field(default_factory=dict)
    states: List[State] = field(default_factory=list)

    # ------------------------------------------------------------ identifying
    def answers_to(self, name: str) -> bool:
        return bool(name) and (name == self.name or name in self.aliases)

    def identifies(self, metadata: Dict[str, Any]) -> bool:
        if self.identify is None or not self.identify.tag:
            return False
        tag = expand(self.identify.tag, self.prefix)
        found = metadata.get(tag)
        if found is None:
            return False
        wanted = str(self.identify.argument).strip()
        text = str(found).strip()
        if self.identify.read == "contains":
            return wanted in text
        return text == wanted

    def identified_by(self, metadata: Dict[str, Any]) -> str:
        if self.identify is None or not self.identify.tag:
            return "chosen by hand"
        tag = expand(self.identify.tag, self.prefix)
        return f"{tag} = {metadata.get(tag)}"

    def state_for(self, metadata: Dict[str, Any]) -> Optional[State]:
        for state in self.states:
            if state.matches(metadata, self.prefix):
                return state
        return None

    # ------------------------------------------------------------- resolving
    def resolve(self, metadata: Dict[str, Any]) -> "Resolution":
        """Every parameter this camera knows about, with where it came from."""
        state = self.state_for(metadata)
        values: Dict[str, Any] = {}
        sources: Dict[str, Source] = {}
        for name, parameter in self.parameters.items():
            value, source = parameter.resolve(metadata, self.prefix)
            if parameter.state:
                if state is None:
                    source = Source("missing", "the readout mode was not recognised")
                elif name in state.values:
                    value = state.values[name]
                    source = Source("state", state.describe(self.prefix))
                else:
                    source = Source("missing",
                                    f"{state.describe(self.prefix)} sets no {name}")
            values[name] = value
            sources[name] = source
        return Resolution(camera=self, state=state, values=values, sources=sources,
                          metadata=metadata)

    # ------------------------------------------------------------- as a file
    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Camera":
        identify = raw.get("identify")
        return cls(
            name=str(raw.get("name", "")),
            prefix=str(raw.get("prefix", "")),
            comment=str(raw.get("comment", "")),
            aliases=[str(a) for a in (raw.get("aliases") or [])],
            identify=(None if not identify else
                      Parameter(tag=str(identify.get("tag", "")),
                                read=str(identify.get("match", "exact")),
                                argument=identify.get("value", ""))),
            parameters={RENAMED.get(str(k), str(k)): Parameter.from_dict(v)
                        for k, v in (raw.get("parameters") or {}).items()},
            states=[State.from_dict(s) for s in (raw.get("states") or [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name}
        if self.aliases:
            out["aliases"] = list(self.aliases)
        if self.prefix:
            out["prefix"] = self.prefix
        if self.comment:
            out["comment"] = self.comment
        if self.identify is not None and self.identify.tag:
            out["identify"] = {"tag": self.identify.tag,
                               "value": self.identify.argument}
            if self.identify.read != "exact":
                out["identify"]["match"] = self.identify.read
        out["parameters"] = {k: v.to_dict() for k, v in self.parameters.items()}
        if self.states:
            out["states"] = [s.to_dict() for s in self.states]
        return out


@dataclass
class Resolution:
    """What a camera made of one file's metadata.

    `values` is every parameter the database describes, `sources` where each
    one came from, and `camera_metadata` the subset the fitter reads.  A
    parameter that could not be resolved is present with a value of None and a
    source that says why -- which is what the parameter view shows, and the
    reason a missing conversion is a sentence rather than a silent zero.
    """
    camera: Optional[Camera] = None
    state: Optional[State] = None
    values: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, Source] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: the `microscope_db.MicroscopeResolution` of the same file, if one was
    #: recognised: `io.tiff.resolve_camera` lays its pixel over the file's
    microscope: Any = None

    @property
    def camera_name(self) -> str:
        return self.camera.name if self.camera else ""

    @property
    def state_name(self) -> str:
        if self.state is None:
            return ""
        return self.state.describe(self.camera.prefix if self.camera else "")

    @property
    def missing(self) -> List[str]:
        """Parameters the fit needs that could not be resolved."""
        return [name for name in ("conversion", "offset", "pixelsize_um")
                if self.values.get(name) is None]

    def overlaid_with(self, values: Dict[str, Any], source: Source,
                      only_over: Sequence[str] = ()) -> "Resolution":
        """A copy with `values` on top, each credited to `source`.

        How "the metadata wins" is implemented: the database says what a
        parameter is *when the file does not say*, so a value the file does
        carry replaces it -- and the table then reads "metadata: PixelSizeUm =
        0.16" rather than quietly showing the database's number.

        `only_over` restricts the overlay to values that currently come from
        those kinds of source, which is what keeps a file's guess from
        replacing what that very file said.
        """
        merged = dict(self.values)
        sources = dict(self.sources)
        for name, value in values.items():
            if value is None:
                continue
            current = sources.get(name)
            if only_over and current is not None and current.kind not in only_over:
                continue
            merged[name] = value
            sources[name] = source if isinstance(source, Source) else Source(*source)
        return Resolution(camera=self.camera, state=self.state, values=merged,
                          sources=sources, metadata=self.metadata, microscope=self.microscope)

    def camera_metadata(self) -> CameraMetadata:
        """The part of this the fitter reads.

        Named after the database entry when there is one and after the
        Micro-Manager device otherwise, so a file from a camera nobody has
        entered is still labelled with what took it.
        """
        known = {k: v for k, v in self.values.items()
                 if k in FITTING_PARAMETERS and v is not None}
        if "roi" in known:
            # four whole pixels, however the tag wrote them
            known["roi"] = tuple(int(v) for v in known["roi"])
        name = self.camera_name or self.values.get("camera_name")
        microscope = self.microscope.summary() if self.microscope is not None else None
        return CameraMetadata(camera_name=name or None, microscope=microscope, **known)

    def rows(self) -> List[tuple]:
        """``(parameter, value, source, used for fitting)``, for showing."""
        return [(name, self.values.get(name), str(self.sources.get(name, "")),
                 name in FITTING_PARAMETERS)
                for name in sorted(self.values)]


class CameraDatabase:
    """The cameras known here: the shipped list, plus the user's own."""

    def __init__(self, cameras: Sequence[Camera] = (), sources: Sequence[Path] = ()):
        self.cameras: List[Camera] = list(cameras)
        self.sources: List[Path] = list(sources)

    # ------------------------------------------------------------- as a file
    @classmethod
    def from_dict(cls, raw: Dict[str, Any], source: Optional[Path] = None
                  ) -> "CameraDatabase":
        cameras = [Camera.from_dict(c) for c in (raw.get("cameras") or [])
                   if isinstance(c, dict) and c.get("name")]
        return cls(cameras, [source] if source else [])

    @classmethod
    def load(cls, path) -> "CameraDatabase":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), path)

    def to_dict(self) -> Dict[str, Any]:
        return {"version": 1, "cameras": [c.to_dict() for c in self.cameras]}

    def save(self, path) -> Path:
        """Write the cameras into `path`, keeping whatever else the file holds:
        a lab's ``microscopes.json`` carries its microscopes beside them."""
        return write_section(path, "cameras", self.to_dict()["cameras"])

    # --------------------------------------------------------------- reading
    def __len__(self) -> int:
        return len(self.cameras)

    def names(self) -> List[str]:
        return [c.name for c in self.cameras]

    def get(self, name: str) -> Optional[Camera]:
        """The camera of that name, or of that former name (`Camera.aliases`)."""
        return (next((c for c in self.cameras if c.name == name), None)
                or next((c for c in self.cameras if c.answers_to(name)), None))

    def joined_with(self, other: "CameraDatabase") -> "CameraDatabase":
        """This database, with `other`'s cameras added -- theirs winning.

        By name: a user file that defines "iXon 1" replaces the shipped one
        rather than shadowing it from somewhere in the middle of a list -- and
        so does one that defines it under a name it has since dropped, a copy
        saved as "M1 Andor" before the camera was renamed.
        """
        import copy
        theirs = [copy.deepcopy(c) for c in other.cameras]
        kept = []
        for camera in self.cameras:
            replacement = next((c for c in theirs if camera.answers_to(c.name)
                                or c.answers_to(camera.name)), None)
            if replacement is None:
                kept.append(camera)
                continue
            # the copy answers to every name the shipped camera did, so a
            # microscope that names "iXon 1" finds a user's "M1 Andor"
            for name in [camera.name] + camera.aliases:
                if name != replacement.name and name not in replacement.aliases:
                    replacement.aliases.append(name)
        return CameraDatabase(theirs + kept,
                              list(other.sources) + list(self.sources))

    def identify(self, metadata: Dict[str, Any]) -> Optional[Camera]:
        """The camera whose identifying tag this file carries, if any."""
        return next((c for c in self.cameras if c.identifies(metadata)), None)

    def resolve(self, metadata: Dict[str, Any], camera: str = "") -> Resolution:
        """Work out this file's camera parameters.

        `camera` names one to use instead of identifying it, which is what a
        file with no identifying tag needs -- a camera that cannot be
        recognised is chosen, never guessed.
        """
        metadata = dict(metadata or {})
        found = self.get(camera) if camera else self.identify(metadata)
        if found is None:
            return Resolution(metadata=metadata)
        return found.resolve(metadata)


# ------------------------------------------------------------------ the default

_cache: Optional[CameraDatabase] = None
_revision = 0


def user_file() -> Path:
    """Where a lab's own cameras live: ``<config dir>/microscopes.json``,
    beside its microscopes."""
    from . import config
    return config.config_dir() / "microscopes.json"


def legacy_user_file() -> Path:
    """``<config dir>/cameras.json``, where cameras were saved before they
    shared a file with the microscopes: still read, below `user_file`, and
    never written."""
    from . import config
    return config.config_dir() / "cameras.json"


def read_file(path) -> Dict[str, Any]:
    """A lab's file as a dict, or an empty one if there is none."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_section(path, key: str, entries: List[Dict[str, Any]]) -> Path:
    """Replace one list of a lab's file and keep the rest of it.

    The cameras and the microscopes share the file, and each database writes
    only its own list: saving a camera must not drop the microscopes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = read_file(path)
    data["version"] = 1
    data[key] = entries
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def database(reload: bool = False) -> CameraDatabase:
    """The shipped cameras joined with the user's, cached.

    Neither file has to exist: with no user file the shipped list stands, and
    with neither the database is empty and every parameter comes from the
    image metadata, as it did before there was a database.
    """
    global _cache, _revision
    if _cache is not None and not reload:
        return _cache
    _revision += 1
    shipped = CameraDatabase()
    if SHIPPED.exists():
        try:
            shipped = CameraDatabase.load(SHIPPED)
        except (OSError, ValueError):
            shipped = CameraDatabase()
    for path in (legacy_user_file(), user_file()):     # the newer one wins
        if path.exists():
            try:
                shipped = shipped.joined_with(CameraDatabase.load(path))
            except (OSError, ValueError) as error:
                import warnings
                warnings.warn(f"{path} could not be read: {error}", stacklevel=2)
    _cache = shipped
    return _cache


def revision() -> int:
    """How many times the database has been read: a cache keyed on it notices
    a camera that was added while the program ran."""
    return _revision


# ------------------------------------------------------------- a new entry
#
# What the camera editor starts from.  A file from a camera nobody has entered
# still says most of what the entry needs -- the device, its serial number,
# which tags carry the gain and the readout mode -- and only the conversion
# and the baseline have to be typed, from the camera's data sheet or its
# calibration.  So the draft guesses everything it can from the tags and
# leaves those two empty: a guessed conversion is exactly what the database
# exists to prevent.

#: device tags that name one camera, best first: a serial number survives
#: a renamed device, a model name does not tell two of a kind apart
IDENTIFYING = ("SerialNumber", "Serial Number", "CameraID", "Camera ID",
               "Camera", "CameraName", "Description")

#: device tags that set a readout mode, which the conversion and the baseline
#: depend on.  ``Gain`` is the pre-amp gain on a Photometrics camera, which
#: reports its EM gain as ``MultiplierGain``, and the EM gain itself on an
#: iXon, so it is a mode tag only on the first (`draft`).
MODE_TAGS = ("ReadoutMode", "ReadoutRate", "Readout Rate", "PixelReadoutRate",
             "Pre-Amp-Gain", "PreAmpGain", "Port", "Output_Amplifier",
             "Sensitivity/DynamicRange", "ScanMode", "ReadoutSpeed")

#: the parameters the editor shows, in its order; anything else an entry
#: carries is kept as it is
EDITED = ("conversion", "offset", "pixelsize_um", "em_on", "emgain",
          "read_noise_e", "exposure_ms", "frame_time_ms", "roi")


def device_of(metadata: Dict[str, Any]) -> str:
    """The Micro-Manager camera device that took this file, or ``""``."""
    return str(metadata.get("Core-Camera") or metadata.get("Camera") or "").strip()


def generalise(tag: str, prefix: str) -> str:
    """``Evolve512-Gain`` -> ``{prefix}-Gain``: the inverse of `expand`."""
    tag = str(tag)
    if prefix and tag.startswith(prefix + "-"):
        return "{prefix}" + tag[len(prefix):]
    return tag


def draft(metadata: Dict[str, Any], name: str = "") -> Camera:
    """A camera entry for the device that took this file, from its tags.

    Everything the tags can say is filled in: the device as the prefix, the
    serial number to recognise it by, the tags that carry the EM gain, the
    exposure and the ROI, the pixel size if the file has a calibrated one,
    and -- where the device reports a readout mode -- conversion and offset
    marked as depending on it.  Their *values* are left empty (`with_mode`
    sets them for this file's mode): the tags never carry them.
    """
    metadata = dict(metadata or {})
    device = device_of(metadata)
    prefix = device if device and any(str(k).startswith(device + "-")
                                      for k in metadata) else ""

    def has(key: str) -> bool:
        return bool(prefix) and f"{prefix}-{key}" in metadata

    identify = None
    for key in IDENTIFYING:
        if has(key) and str(metadata[f"{prefix}-{key}"]).strip():
            identify = Parameter(tag="{prefix}-" + key, read="exact",
                                 argument=str(metadata[f"{prefix}-{key}"]).strip())
            break

    parameters: Dict[str, Parameter] = {}
    # Photometrics reports the EM gain as MultiplierGain and the pre-amp gain
    # as Gain; Andor reports the EM gain as Gain
    gain_is_preamp = has("MultiplierGain") or has("Port")
    if has("Port"):
        parameters["em_on"] = Parameter(tag="{prefix}-Port", read="not_equals",
                                        argument="Normal")
    elif has("Output_Amplifier"):
        parameters["em_on"] = Parameter(tag="{prefix}-Output_Amplifier",
                                        read="not_equals", argument="Conventional")
    elif has("EMSwitch"):
        parameters["em_on"] = Parameter(tag="{prefix}-EMSwitch", read="equals",
                                        argument="On")
    else:
        parameters["em_on"] = Parameter(fixed=False)
    gain = "MultiplierGain" if gain_is_preamp else "Gain"
    if parameters["em_on"].tag and has(gain):
        parameters["emgain"] = Parameter(tag="{prefix}-" + gain, read="number")
    else:
        parameters["emgain"] = Parameter(fixed=1.0)

    modes = [key for key in MODE_TAGS if has(key)]
    if gain_is_preamp and has("Gain"):
        modes.append("Gain")
    for key in ("conversion", "offset"):
        parameters[key] = Parameter(state=True) if modes else Parameter(fixed=None)

    pixelsize = read_number(metadata.get("PixelSizeUm", ""))
    parameters["pixelsize_um"] = Parameter(fixed=pixelsize or None)
    # a typical sCMOS, said out loud so it can be changed; with EM gain the
    # fit ignores it (`CameraMetadata.readout_variance`)
    parameters["read_noise_e"] = Parameter(fixed=1.0)
    if "Exposure-ms" in metadata:
        parameters["exposure_ms"] = Parameter(tag="Exposure-ms", read="number")
    elif has("Exposure"):
        parameters["exposure_ms"] = Parameter(tag="{prefix}-Exposure", read="number")
    if has("ActualInterval-ms"):
        parameters["frame_time_ms"] = Parameter(tag="{prefix}-ActualInterval-ms",
                                                read="number")
    if "ROI" in metadata:
        parameters["roi"] = Parameter(tag="ROI", read="numbers")

    camera = Camera(name=name or device or "my camera", prefix=prefix,
                    identify=identify, parameters=parameters)
    if modes:
        camera = with_mode(camera, metadata, ["{prefix}-" + key for key in modes], {})
    return camera


def mode_tags(camera: Camera) -> List[str]:
    """Every tag any of this camera's readout modes is recognised by."""
    seen: List[str] = []
    for state in camera.states:
        seen += [tag for tag in state.match if tag not in seen]
    return seen


def with_mode(camera: Camera, metadata: Dict[str, Any], tags: Sequence[str],
              values: Dict[str, Any]) -> Camera:
    """This camera, with this file's readout mode recognised by `tags`.

    The mode is what those tags read in this file, and it carries `values`.
    It replaces the mode the file already matched -- correcting a conversion
    -- or one recognised by the same tags and values, and is added otherwise:
    a camera's other modes are never touched, because each of them was
    measured on its own.
    """
    import copy
    camera = copy.deepcopy(camera)
    match = {generalise(tag, camera.prefix):
             str(metadata.get(expand(tag, camera.prefix), "")).strip()
             for tag in tags}
    if not match:
        return camera              # a mode recognised by nothing matches nothing
    new = State(match=match, values={k: v for k, v in values.items()
                                     if v is not None})
    current = camera.state_for(metadata)
    for i, state in enumerate(camera.states):
        if state is current or state.match == match:
            new.name = state.name if state.match == match else ""
            camera.states[i] = new
            break
    else:
        camera.states.append(new)
    return camera


def problems(camera: Camera, metadata: Dict[str, Any],
             known: Optional[CameraDatabase] = None) -> List[str]:
    """What is wrong with this entry, for this file, in words.

    The editor shows these before it saves, and only a missing name stops
    it: an entry with no tag to be recognised by can still be chosen by hand,
    and one that leaves the conversion unknown is still a camera the fit can
    be told the rest of.  They say what the fit will still be missing.
    """
    out: List[str] = []
    if not camera.name.strip():
        out.append("the camera needs a name")
    if camera.identify is None or not camera.identify.tag:
        out.append("no tag to recognise it by: it can only be chosen by hand, "
                   "under camera in the Localize tab")
    elif metadata and not camera.identifies(metadata):
        out.append(f"this file does not match {camera.identified_by(metadata)}: "
                   f"the entry would not recognise it")
    if known is not None and camera.identify is not None and metadata:
        other = known.identify(metadata)
        if other is not None and other.name != camera.name:
            out.append(f"this file is already recognised as {other.name}, "
                       f"which comes first")
    if any(p.state for p in camera.parameters.values()) and metadata \
            and camera.state_for(metadata) is None:
        out.append("no readout mode recognises this file: tick the tags that "
                   "set it")
    resolved = camera.resolve(metadata) if metadata else None
    if resolved is not None:
        missing = [n for n in ("conversion", "offset", "pixelsize_um")
                   if resolved.values.get(n) is None]
        if missing:
            out.append("still unknown for this file: " + ", ".join(missing))
    return out


def save_camera(camera: Camera, path=None) -> Path:
    """Add this camera to the user's database, or replace it there by name.

    Always the user's file (`user_file`), never the shipped one: an update
    of smappy replaces the shipped list, and a camera of the lab's own must
    survive it.  A camera that has the name of a shipped one takes its place
    (`CameraDatabase.joined_with`), which is how a shipped entry is corrected
    or given another readout mode.  The database is read again afterwards, so
    the next file opened is recognised.
    """
    path = Path(path) if path is not None else user_file()
    mine = CameraDatabase.load(path) if path.exists() else CameraDatabase()
    for i, existing in enumerate(mine.cameras):
        if existing.name == camera.name or camera.answers_to(existing.name):
            mine.cameras[i] = camera
            break
    else:
        mine.cameras.append(camera)
    mine.save(path)
    database(reload=True)
    return path
