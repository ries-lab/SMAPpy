"""A trained model's file: safetensors, written and read here, with numpy alone.

The format (https://github.com/huggingface/safetensors): an 8-byte little-endian length, a JSON
header of that length, the tensors' raw bytes.  The header names each tensor's dtype, shape and
byte range, and ``__metadata__`` holds strings -- here one, ``"smappy"``, a JSON document of what a
fitter needs to use the model: the method and its architecture, the input and output channels,
the scaling, the camera, the calibration's sha256, the training's ranges, the base model, the
epochs, the SMAPpy version (docs/decode/poc-plan.md §5).

Why not ``torch.save``: it is pickle, which runs code when loaded, and a model file is meant to
be passed around.  Why not the ``safetensors`` package: the format is a page of code, and this way
the GUI reads a model's header without torch or another dependency -- every time the fitter's
form changes (`read_metadata`).  The package is used only in a test, as a check that its readers
read these files.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

# safetensors' dtype names, those numpy has
_DTYPES = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32,
           "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}
_NAMES = {np.dtype(v): k for k, v in _DTYPES.items()}
MAX_HEADER = 100 * 2 ** 20


def write(path, tensors: dict, metadata: dict | None = None) -> Path:
    """Write ``tensors`` (name -> array) with ``metadata`` (JSON-able) to ``path``, atomically."""
    path = Path(path)
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        a = np.asarray(tensors[name])
        if not a.flags.c_contiguous:
            a = a.copy(order="C")              # not ascontiguousarray: it makes a 0-d tensor 1-d
        if a.dtype.byteorder == ">":
            a = a.astype(a.dtype.newbyteorder("<"))
        if a.dtype not in _NAMES:
            raise TypeError(f"{name}: a {a.dtype} tensor has no safetensors dtype")
        data = a.tobytes()
        header[name] = {"dtype": _NAMES[a.dtype], "shape": list(a.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    if metadata is not None:
        header["__metadata__"] = {"smappy": json.dumps(metadata, separators=(",", ":"))}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)                     # the tensors start 8-byte aligned
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for data in blobs:
            f.write(data)
    os.replace(tmp, path)
    return path


def _header(f) -> tuple[dict, int]:
    (n,) = struct.unpack("<Q", f.read(8))
    if n > MAX_HEADER:
        raise ValueError(f"{getattr(f, 'name', 'the file')}: a header of {n} bytes is not a safetensors file")
    return json.loads(f.read(n)), 8 + n


def read_metadata(path) -> dict:
    """The model's SMAPpy metadata, from the header alone: no tensor is read."""
    with open(path, "rb") as f:
        header, _ = _header(f)
    meta = header.get("__metadata__") or {}
    return json.loads(meta["smappy"]) if "smappy" in meta else {}


def read(path) -> tuple[dict, dict]:
    """``(tensors, metadata)``: name -> numpy array, and the SMAPpy metadata."""
    with open(path, "rb") as f:
        header, _ = _header(f)
        body = f.read()
    meta = header.pop("__metadata__", None) or {}
    tensors = {}
    for name, h in header.items():
        a, b = h["data_offsets"]
        dtype = np.dtype(_DTYPES[h["dtype"]]).newbyteorder("<")
        tensors[name] = np.frombuffer(body, dtype=dtype, count=(b - a) // dtype.itemsize, offset=a).reshape(h["shape"]).copy()
    return tensors, (json.loads(meta["smappy"]) if "smappy" in meta else {})
