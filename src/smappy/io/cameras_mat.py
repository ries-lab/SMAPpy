"""Reading SMAP's ``settings/*_cameras.mat`` into the camera database.

SMAP keeps its cameras in a MATLAB struct: a table of parameters per camera,
each row saying whether the value is fixed, read from a metadata tag, or
depends on the readout mode, plus a list of modes with the tags that identify
them.  That is the same model `smappy.camera_db` holds in JSON, so this is a
translation and not an interpretation -- the point of it is that a lab with a
``*_cameras.mat`` gets its cameras by converting the file once, instead of
typing eight cameras and thirty readout modes in again.

Two things are translated rather than copied.  SMAP's parameter names become
smappy's (``cam_pixelsize_um`` -> ``pixelsize_um``, ``EMon`` -> ``em_on``), and
its MATLAB expressions (``str2double(X)``, ``~strcmp(X,'Conventional')``)
become the database's named readers.  What is left over -- SMAP's own bookkeeping
rows, and the frame count and image size, which smappy reads from the file
itself -- is dropped, because a database is a place to look things up and not
an archive of another program's internals.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..camera_db import Camera, CameraDatabase, Parameter, State

#: SMAP's names for the things smappy also has.  Anything not here is dropped.
NAMES = {
    "EMon": "em_on",
    "emgain": "emgain",
    "conversion": "conversion",
    "offset": "offset",
    "cam_pixelsize_um": "pixelsize_um",
    "exposure": "exposure_ms",
    "roi": "roi",
    "timediff": "frame_interval_ms",
    "comment": "comment",
}

#: SMAP rows that describe SMAP, or that smappy reads from the image itself.
DROPPED = ("numberOfFrames", "Width", "Height", "roimode", "correctionfile",
           "imagemetadata")

#: SMAP's fallback entry, which it uses for any camera it does not recognise.
#: It carries whatever numbers someone once left in it, and using them for an
#: unknown camera is the guess this database exists not to make -- so it is
#: not converted, and an unrecognised camera is chosen by name instead.
FALLBACK = "Default"


def _text(value) -> str:
    """A MATLAB cell entry as a plain string ('' for empty)."""
    if isinstance(value, np.ndarray):
        return "" if value.size == 0 else str(value.reshape(-1)[0])
    return "" if value is None else str(value)


def _number(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def reader_for(expression: str) -> Dict[str, Any]:
    """One of SMAP's MATLAB expressions as a named reader.

    The five that the settings files actually use.  Anything else is read as
    text, which is what SMAP's own empty expression does.
    """
    expression = (expression or "").strip()
    if expression in ("str2double(X)", "str2num(X)"):
        return {"read": "numbers" if expression == "str2num(X)" else "number"}
    if expression == "str2double(X)>0":
        return {"read": "positive"}
    match = re.fullmatch(r"\(?(~?)strcmp\(X,'([^']*)'\)\)?", expression)
    if match:
        return {"read": "not_equals" if match.group(1) else "equals",
                "argument": match.group(2)}
    return {"read": "text"}


def _value(text: str, name: str):
    """A stored value as the type the parameter wants."""
    text = (text or "").strip()
    if text == "":
        return None
    if name == "pixelsize_um":
        # SMAP keeps x and y; so does a camera here, and one number means the
        # same in both directions -- which is what almost every entry says
        numbers = [n for n in (_number(part) for part in text.split()) if n]
        if not numbers:
            return None
        if len(numbers) == 1 or numbers[0] == numbers[1]:
            return numbers[0]
        return numbers[:2]
    if name in ("em_on",):
        from ..camera_db import read_boolean
        return read_boolean(text)
    if name in ("roi",):
        from ..camera_db import read_numbers
        return read_numbers(text)
    if name == "comment":
        return text
    number = _number(text)
    return text if number is None else number


#: SMAP reads these from the image rather than from a metadata key.  Only the
#: ROI has an equivalent here; the rest smappy takes from the file itself.
PSEUDO_TAGS = {"ROI direct": "ROI"}


def _prefix_of(camera_tags) -> str:
    """The Micro-Manager device name every tag of this camera starts with.

    Worth finding, because it is the one thing that changes when a lab renames
    its device: with a prefix the tags read ``{prefix}-Gain``, and the rename
    is one field rather than a dozen.  As many whole ``-``-separated segments
    as every tag shares, so ``Andor iXon X-9603-Gain`` and
    ``Andor iXon X-9603-Exposure`` give ``Andor iXon X-9603`` and not
    ``Andor iXon X``.
    """
    candidates = [t.split("-") for t in camera_tags if "-" in t]
    if not candidates:
        return ""
    shared = []
    for parts in zip(*candidates):
        if len(set(parts)) != 1:
            break
        shared.append(parts[0])
    # the last shared segment may be the start of the parameter's own name
    # ("Andor-ActualInterval-ms" next to "Andor-Exposure" shares only "Andor")
    return "-".join(shared[:-1]) if len(shared) == len(min(candidates, key=len)) \
        else "-".join(shared)


def _tag(tag: str, prefix: str) -> str:
    if prefix and tag.startswith(prefix + "-"):
        return "{prefix}-" + tag[len(prefix) + 1:]
    return tag


def to_database(path) -> CameraDatabase:
    """Every camera in a SMAP ``*_cameras.mat``, as a camera database."""
    import scipy.io                              # only when a .mat is read

    path = Path(path)
    mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    table = np.atleast_2d(mat["camtab"])
    raw_cameras = np.atleast_1d(mat["cameras"])

    cameras = []
    for row, raw in zip(table, raw_cameras):
        camera = _camera(row, raw)
        if camera.name == FALLBACK and camera.identify is None:
            continue
        cameras.append(camera)
    return CameraDatabase(cameras, [path])


def _camera(row, raw) -> Camera:
    name, id_tag, id_value = _text(row[0]), _text(row[1]), _text(row[2])
    rows = [[_text(cell) for cell in line] for line in np.atleast_2d(raw.par)]

    tags = [line[3] for line in rows if line[1] == "metadata" and line[3]]
    states_raw = [st for st in np.atleast_1d(getattr(raw, "state", []))
                  if hasattr(st, "_fieldnames")]
    for state in states_raw:
        tags += [_text(pair[0]) for pair in np.atleast_2d(state.defpar)
                 if _text(pair[0]) not in ("", "select")]
    if id_tag and id_tag != "select":
        tags.append(id_tag)
    prefix = _prefix_of(tags)

    parameters: Dict[str, Parameter] = {}
    comment = ""
    for line in rows:
        smap_name, mode, fixed, tag, _example, expression = line[:6]
        if smap_name in DROPPED or smap_name not in NAMES:
            continue
        key = NAMES[smap_name]
        tag = PSEUDO_TAGS.get(tag, tag)
        if mode == "metadata":
            # SMAP's editor writes "select" where nothing was chosen, and
            # "... direct" for what it reads off the image rather than a tag;
            # either way this camera has no metadata key for the parameter
            if not tag or tag == "select" or tag.endswith(" direct"):
                continue
            reader = reader_for(expression)
            parameters[key] = Parameter(tag=_tag(tag, prefix),
                                        read=reader["read"],
                                        argument=reader.get("argument"))
        elif mode == "state dependent":
            parameters[key] = Parameter(state=True)
        else:                                      # fix
            value = _value(fixed, key)
            if key == "comment":
                text_value = str(value or "")
                # SMAP's placeholder for a comment nobody wrote
                comment = "" if text_value == "settings not initialized" else text_value
                continue
            if value is not None:
                parameters[key] = Parameter(fixed=value)

    states = []
    for state in states_raw:
        match = {}
        for pair in np.atleast_2d(state.defpar):
            tag, wanted = _text(pair[0]), _text(pair[1])
            if tag and tag != "select" and wanted:
                match[_tag(tag, prefix)] = wanted
        if not match:                              # a state that matches nothing
            continue
        values = {}
        for pair in np.atleast_2d(state.par):
            smap_name, text = _text(pair[0]), _text(pair[1])
            key = NAMES.get(smap_name)
            # only what this camera says is state dependent: the rest of the
            # column is SMAP's editor filling every row of every state
            if key and parameters.get(key, Parameter()).state:
                value = _value(text, key)
                if value is not None:
                    values[key] = value
        if values:
            states.append(State(match=match, values=values))

    identify = None
    if id_tag and id_tag != "select" and id_value:
        identify = Parameter(tag=_tag(id_tag, prefix), read="exact",
                             argument=id_value)
    return Camera(name=name, prefix=prefix, comment=comment, identify=identify,
                  parameters=parameters, states=states)


def main(argv=None) -> int:
    """``python -m smappy.io.cameras_mat lab_cameras.mat cameras.json``."""
    import argparse

    from ..camera_db import user_file

    parser = argparse.ArgumentParser(
        description="Convert a SMAP *_cameras.mat into a camera database.")
    parser.add_argument("mat", help="the SMAP settings file to read")
    parser.add_argument("json", nargs="?", default=None,
                        help=f"where to write it (default: {user_file()})")
    args = parser.parse_args(argv)

    database = to_database(args.mat)
    path = database.save(args.json or user_file())
    print(f"{len(database)} cameras written to {path}")
    print(f"  (SMAP's '{FALLBACK}' camera is not converted: its numbers stand "
          f"for no camera in particular)")
    for camera in database.cameras:
        states = f", {len(camera.states)} readout modes" if camera.states else ""
        print(f"  {camera.name}{states}")
    return 0


if __name__ == "__main__":       # pragma: no cover - a convenience entry point
    raise SystemExit(main())
