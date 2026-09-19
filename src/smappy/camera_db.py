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
* ``metadata``: read this tag out of the file and interpret it this way;
* ``state``: it depends on the readout mode, and the mode is recognised by a
  set of tags that must all match.

SMAP keeps the same model in ``settings/*_cameras.mat``; this is that concept
in JSON, so that adding a camera is editing a file rather than a MATLAB
struct.  `from_mat` ports one.

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

#: parameters that end up in `CameraMetadata` and so reach the fit.  Anything
#: else in the database is carried, shown and ignored by the fitter -- a frame
#: interval is worth knowing and is not something the fit reads.
FITTING_PARAMETERS = ("conversion", "offset", "pixelsize_um", "em_on", "emgain",
                      "exposure_ms", "roi")

DATA = Path(__file__).resolve().parent / "data"
SHIPPED = DATA / "cameras.json"


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


READERS = {
    "text": lambda text, argument: str(text).strip(),
    "number": lambda text, argument: read_number(text),
    "numbers": lambda text, argument: read_numbers(text),
    "boolean": lambda text, argument: read_boolean(text),
    "positive": lambda text, argument: (None if read_number(text) is None
                                        else read_number(text) > 0),
    "equals": lambda text, argument: str(text).strip() == str(argument).strip(),
    "not_equals": lambda text, argument: str(text).strip() != str(argument).strip(),
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
    """Where one camera parameter comes from."""
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
                       argument=raw.get("value"))
        return cls(fixed=raw.get("fixed"))

    def to_dict(self) -> Any:
        if self.state:
            return {"state": True}
        if self.tag:
            out: Dict[str, Any] = {"tag": self.tag, "read": self.read}
            if self.argument is not None:
                out["value"] = self.argument
            return out
        return {"fixed": self.fixed}

    def resolve(self, metadata: Dict[str, Any], prefix: str = ""):
        """``(value, Source)``; a value of None means it could not be read."""
        if self.state:
            return None, Source("state")
        if self.tag:
            tag = expand(self.tag, prefix)
            if tag not in metadata:
                return None, Source("missing", f"no tag {tag}")
            reader = READERS.get(self.read, READERS["text"])
            value = reader(metadata[tag], self.argument)
            if value is None:
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
    parameters: Dict[str, Parameter] = field(default_factory=dict)
    states: List[State] = field(default_factory=list)

    # ------------------------------------------------------------ identifying
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
            identify=(None if not identify else
                      Parameter(tag=str(identify.get("tag", "")),
                                read=str(identify.get("match", "exact")),
                                argument=identify.get("value", ""))),
            parameters={str(k): Parameter.from_dict(v)
                        for k, v in (raw.get("parameters") or {}).items()},
            states=[State.from_dict(s) for s in (raw.get("states") or [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name}
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
                          sources=sources, metadata=self.metadata)

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
        return CameraMetadata(camera_name=name or None, **known)

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
        return cls.from_dict(json.loads(path.read_text()), path)

    def to_dict(self) -> Dict[str, Any]:
        return {"version": 1, "cameras": [c.to_dict() for c in self.cameras]}

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    # --------------------------------------------------------------- reading
    def __len__(self) -> int:
        return len(self.cameras)

    def names(self) -> List[str]:
        return [c.name for c in self.cameras]

    def get(self, name: str) -> Optional[Camera]:
        return next((c for c in self.cameras if c.name == name), None)

    def joined_with(self, other: "CameraDatabase") -> "CameraDatabase":
        """This database, with `other`'s cameras added -- theirs winning.

        By name: a user file that defines "M1 Andor" replaces the shipped one
        rather than shadowing it from somewhere in the middle of a list.
        """
        cameras = [c for c in self.cameras if other.get(c.name) is None]
        return CameraDatabase(list(other.cameras) + cameras,
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


def user_file() -> Path:
    """Where a lab's own cameras live: ``<config dir>/cameras.json``."""
    from . import config
    return config.config_dir() / "cameras.json"


def database(reload: bool = False) -> CameraDatabase:
    """The shipped cameras joined with the user's, cached.

    Neither file has to exist: with no user file the shipped list stands, and
    with neither the database is empty and every parameter comes from the
    image metadata, as it did before there was a database.
    """
    global _cache
    if _cache is not None and not reload:
        return _cache
    shipped = CameraDatabase()
    if SHIPPED.exists():
        try:
            shipped = CameraDatabase.load(SHIPPED)
        except (OSError, ValueError):
            shipped = CameraDatabase()
    path = user_file()
    if path.exists():
        try:
            shipped = shipped.joined_with(CameraDatabase.load(path))
        except (OSError, ValueError) as error:
            import warnings
            warnings.warn(f"{path} could not be read: {error}", stacklevel=2)
    _cache = shipped
    return _cache
