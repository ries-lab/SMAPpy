"""Just enough of MATLAB's object store to read the objects in a v7.3 MAT-file.

SMAP keeps its ROI manager's sites as objects of a class
(``interfaces.SEsites``), and MATLAB writes an object not as a struct but as
a reference into one blob shared by the whole file, ``/#subsystem#/MCOS``.
h5py sees a ``uint32`` array where the object is.  Neither scipy nor h5py
reads further, and a package that does (mat-io) would be a dependency for
two dozen numbers per site, so this is the part of the format that is needed.

What is read, as MATLAB 9.x and later write it (layout version 2 to 4,
worked out by mat-io and checked here against a SMAP file of R2024b):

* An object reference is ``[0xDD000000, ndims, dims..., object ids...,
  class id]``: an array of objects, all of one class.
* ``/#subsystem#/MCOS`` is a cell array.  Its first cell is the metadata, a
  ``uint8`` blob: a version, the number of names, eight region offsets, the
  names (NUL-terminated), then the regions -- the classes (package and name,
  as 1-based indices into the names), the objects' properties saved by
  ``saveobj`` (region 2), the objects (class id and where their properties
  are) and their ordinary properties (region 4).  The cells from the third
  on are the properties' values.
* A property is three numbers: its name, a flag and a value -- the value is
  a name for flag 0, a cell of the store (``value + 2``) for flag 1, and the
  value itself, a logical, for flag 2.

Not read: the defaults of properties an object did not save (MATLAB stores
them in the last cells; a property left at its default is simply missing
here), dynamic properties, handles to functions, and the v5-v7 files, which
keep the same store in another container.  A value that cannot be read is
None, never an error, since the caller wants what *can* be read.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

OBJECT_MAGIC = 0xDD000000


class Store:
    """The object store of one open v7.3 file (an `h5py.File`)."""

    def __init__(self, f):
        self.f = f
        cells = [f[r] for r in f["#subsystem#/MCOS"][()].ravel()]
        self.cells = cells
        blob = np.asarray(cells[0][()]).ravel().astype(np.uint8).tobytes()
        version, n_names = np.frombuffer(blob, "<u4", 2, 0)
        if not 2 <= version <= 4:
            raise ValueError(f"MATLAB object store of version {version}, not 2 to 4")
        offsets = np.frombuffer(blob, "<u4", 8, 8)
        self.names = [s.decode("utf-8", "replace")
                      for s in blob[40:offsets[0]].split(b"\0")[:n_names]]

        def region(i):
            return np.frombuffer(blob[offsets[i]:offsets[i + 1]], "<u4")

        self.classes = region(0).reshape(-1, 4)
        self.objects = region(2).reshape(-1, 6)
        self.saved = _properties(region(1))
        self.plain = _properties(region(3))

    def name(self, index: int) -> str:
        return self.names[index - 1] if 0 < index <= len(self.names) else ""

    def class_name(self, class_id: int) -> str:
        package, name = self.classes[class_id][:2]
        return ".".join(n for n in (self.name(package), self.name(name)) if n)

    def object(self, object_id: int) -> Dict[str, Any]:
        """One object as a dict of its properties, its class under
        ``__class__``."""
        class_id, _, _, saved, plain, _ = self.objects[object_id]
        out: Dict[str, Any] = {"__class__": self.class_name(class_id)}
        for table, index in ((self.saved, saved), (self.plain, plain)):
            if 0 < index <= len(table):
                for name, flag, value in table[index - 1]:
                    out[self.name(name)] = self._property(flag, value)
        return out

    def _property(self, flag: int, value: int):
        if flag == 0:
            return self.name(value)
        if flag == 1:
            return self.value(self.cells[value + 2]) if value + 2 < len(self.cells) else None
        return bool(value)

    def value(self, item):
        """A MATLAB value as Python: a struct a dict, a cell a list, text a
        str (a char matrix a list of its rows), a number an array in
        MATLAB's own shape, an object a list of dicts (`object`), and what
        is not read None."""
        try:
            return self._value(item)
        except Exception:                 # what cannot be read is left out
            return None

    def _value(self, item):
        cls = item.attrs.get("MATLAB_class", b"")
        cls = cls.decode() if isinstance(cls, bytes) else str(cls)
        if not hasattr(item, "shape"):                     # a group
            return {k: self.value(item[k]) for k in item} if cls in ("struct", "") else None
        if item.attrs.get("MATLAB_empty", 0):
            return None
        data = item[()]
        if item.attrs.get("MATLAB_object_decode", 0) == 3:
            return self.objects_of(data)
        if cls == "char":
            rows = np.asarray(data).astype(np.uint16).T           # MATLAB's shape
            text = ["".join(map(chr, row)).rstrip("\0") for row in np.atleast_2d(rows)]
            return text[0] if len(text) == 1 else text
        if cls == "cell":
            return [self.value(self.f[r]) for r in np.asarray(data).T.ravel()]
        if cls in ("function_handle",) or data.dtype.kind not in "biuf":
            return None
        return np.asarray(data).T

    def objects_of(self, reference) -> List[Dict[str, Any]]:
        """The objects an object reference names, in MATLAB's order."""
        ref = np.asarray(reference).ravel().astype(np.int64)
        if not len(ref) or ref[0] != OBJECT_MAGIC:
            return []
        ndims = int(ref[1])
        ids = ref[2 + ndims:-1]
        return [self.object(int(i)) for i in ids]


def _properties(region: np.ndarray) -> List[np.ndarray]:
    """A property region: after a first, empty entry, per object a count
    and that many (name, flag, value) triples, padded to 8 bytes."""
    out = []
    p = 2
    while p < len(region):
        n = int(region[p])
        out.append(region[p + 1:p + 1 + 3 * n].reshape(-1, 3))
        p += 1 + 3 * n
        p += p % 2
    return out


def read_objects(f, path: str) -> Optional[List[Dict[str, Any]]]:
    """The objects at ``path`` of an open v7.3 file, or None when there are
    none or they cannot be read."""
    if path not in f or "#subsystem#/MCOS" not in f:
        return None
    try:
        return Store(f).value(f[path]) or None
    except Exception:
        return None
