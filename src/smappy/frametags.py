"""What the microscope said about each frame, and about the acquisition.

Micro-Manager writes the state of every device into each image's metadata --
415 keys per frame on our setup -- and almost all of it never changes.  The
few that do are the record of what happened during the acquisition: the piezo
position (`PIZStage-Position`, the focus lock's work and the first thing to
look at when a 3D measurement goes wrong), the length of the UV pulse
(`Laser Trigger-Duration0 (us)`), the focus lock's analog signals, the
camera's own timestamps.  SMAP kept a list of tag names per camera
(`imagemetadata`) and read only those; here every frame is compared with the
first and a tag is kept from the moment it differs, so nothing has to be
named in advance and a tag nobody thought of still turns up.

Keeping only the changed ones is what keeps it small: 93 000 frames with six
changing tags is 4 MB.  Two limits stop a microscope that writes something
new into every frame from growing it without end -- a text tag is dropped
once it has had more than `MAX_STATES` values (a UUID, a time stamp), and no
more than `MAX_TAGS` tags are kept.  A text tag with few values is a state --
a laser on or off, a filter wheel position -- and is kept as text.

The static part -- the summary and the first frame's device properties -- is
`acquisition`; it goes into the file once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

#: tags kept at most; the rest are counted and named in `dropped`
MAX_TAGS = 64
#: a text tag with more values than this is an identifier, not a state
MAX_STATES = 16
#: the key the table carries the tags under, in `Localizations.metadata`
KEY = "frame_tags"
#: the static metadata, in `Localizations.metadata`
ACQUISITION = "acquisition"

# Change every frame by construction and say nothing the frame number does not.
# `ElapsedTime-ms` and the camera's own clock are kept: a gap in them is a
# dropped frame, and they are what turns a frame into a time.
_IGNORED = {"UUID", "ImageNumber", "Frame", "FrameIndex", "FileName", "ReceivedTime",
            "TimeReceivedByCore", "Andor-ImageNumber"}


def flatten(md: dict, prefix: str = "") -> Iterator[Tuple[str, object]]:
    """A frame's metadata as flat ``Device-Property`` keys and scalar values.

    Micro-Manager 2 wraps user data as ``{"type": ..., "scalar": value}`` and
    NDTiff nests properties by device; both come out as the flat keys a
    Micro-Manager TIFF writes.  ``UserData`` keys are already device-prefixed
    (``Andor-ElapsedTime-ms(HW)``) and keep their own names.
    """
    for key, value in md.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            if "scalar" in value:
                value = value["scalar"]
            else:
                yield from flatten(value, "" if key == "UserData" else f"{name}-")
                continue
        if isinstance(value, (str, int, float, bool)):
            yield name, value


def _number(value) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class FrameTags:
    """Collects the tags that change, frame by frame, as frames are read.

    `add` is called by the readers with each frame's metadata, in frame
    order.  A tag's column is started the first time its value differs from
    the first frame's and filled in backwards with that value, so the
    columns are complete without the first frames having been kept.
    """

    def __init__(self):
        self.frames: List[int] = []
        self._first: Dict[str, object] = {}
        self._columns: Dict[str, list] = {}
        self._text: Dict[str, set] = {}       # distinct values of a text tag
        self._dropped: set = set()

    def __len__(self) -> int:
        return len(self.frames)

    def add(self, frame: int, md: Optional[dict]) -> None:
        if md is None:
            return
        values = dict(flatten(md))
        n = len(self.frames)
        self.frames.append(int(frame))
        if n == 0:
            self._first = {k: v for k, v in values.items() if k not in _IGNORED}
            return
        for name, column in self._columns.items():
            column.append(values.get(name, column[-1]))
        for name, first in self._first.items():
            if name in self._columns or name in self._dropped:
                continue
            value = values.get(name, first)
            if value != first:
                if len(self._columns) >= MAX_TAGS:
                    self._dropped.add(name)
                    continue
                self._columns[name] = [first] * n + [value]
        for name in list(self._text) + [k for k in self._columns
                                        if k not in self._text]:
            self._check_text(name)

    def _check_text(self, name: str) -> None:
        """Drop a text tag once it is clearly an identifier."""
        column = self._columns.get(name)
        if column is None:
            return
        if name not in self._text:
            if _number(self._first[name]) is not None:
                return                       # a number: never a text tag
            self._text[name] = set(map(str, column))
        else:
            self._text[name].add(str(column[-1]))
        if len(self._text[name]) > MAX_STATES:
            del self._columns[name], self._text[name]
            self._dropped.add(name)

    @property
    def dropped(self) -> List[str]:
        return sorted(self._dropped)

    def table(self) -> Dict[str, np.ndarray]:
        """``{"frame": frames, tag: values}``, numbers as float, states as text.

        Empty when no frame carried metadata.
        """
        if not self.frames:
            return {}
        out = {"frame": np.asarray(self.frames, np.int64)}
        for name, column in self._columns.items():
            if name in self._text:
                out[name] = np.asarray([str(v) for v in column], dtype=object)
                continue
            numbers = [_number(v) for v in column]
            out[name] = np.asarray([np.nan if v is None else v for v in numbers],
                                   np.float64)
        return out


def tags_of_page(page) -> Optional[dict]:
    """The per-plane Micro-Manager metadata of a tifffile page, if it has any."""
    tag = page.tags.get("MicroManagerMetadata")
    if tag is None or not isinstance(tag.value, dict):
        return None
    return tag.value


# ---------------------------------------------------------------- the static
def acquisition(source) -> dict:
    """What the microscope said once: the summary, the devices, the comment.

    The device properties are the first frame's, without the serial-port
    settings (``COM5-BaudRate`` and the like, a third of the keys on our
    setup and nothing anyone looks up), and without the summary's copy of
    the same properties.  The comment is Micro-Manager's ``comments.txt``,
    which is where the acquisition's note to itself ("50ms 638i30 UVi20")
    lives.
    """
    summary = {k: v for k, v in (getattr(source, "summary", None) or {}).items()
               if k not in ("InitialScopeData", "ScopeDataKeys")}
    devices = {k: v for k, v in flatten(getattr(source, "mm_metadata", None) or {})
               if not _is_port(k) and k not in _IGNORED}
    out = {}
    if summary:
        out["summary"] = summary
    if devices:
        out["devices"] = devices
    comment = _comment(source)
    if comment:
        out["comment"] = comment
    return out


def _is_port(key: str) -> bool:
    head = key.split("-", 1)[0]
    return head.startswith("COM") and head[3:].isdigit()


def _comment(source) -> str:
    """The text of ``comments.txt`` beside the images, or ''."""
    import json
    files = getattr(source, "files", None) or []
    if not files:
        return ""
    path = Path(files[0]).parent / "comments.txt"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    found = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "comments" and isinstance(v, dict) and "scalar" in v:
                    found.append(str(v["scalar"]))
                else:
                    walk(v)
    walk(data)
    return "\n".join(found).strip()
