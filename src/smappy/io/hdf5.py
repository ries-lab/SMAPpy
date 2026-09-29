"""Writing and reading localization tables as HDF5.

One file holds everything: the localization columns under ``/locs`` and the
full provenance -- camera metadata, every stage's settings, the calibration
used -- as JSON in the file attributes.

The writer appends block by block and flushes as it goes, so results are on
disk and readable while an acquisition is still running, and an interrupted run
keeps everything up to the last block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from ..locs import Localizations

FORMAT_VERSION = 1


class LocalizationWriter:
    """Append-only HDF5 writer for localizations.

    Usage::

        with LocalizationWriter("out.h5", metadata=...) as writer:
            writer.append(locs)
    """

    def __init__(self, path, metadata: Optional[Dict[str, object]] = None,
                 chunk: int = 8192, overwrite: bool = True,
                 compression: Optional[str] = "lzf"):
        self.path = Path(path)
        self.chunk = int(chunk)
        # lzf compresses about five times faster than gzip for ~15% more space,
        # which matters because this runs while frames are still being fitted
        self.compression = compression
        mode = "w" if overwrite else "w-"
        self._file = h5py.File(self.path, mode)
        self._group = self._file.create_group("locs")
        self._datasets: Dict[str, h5py.Dataset] = {}
        self._metadata: Dict[str, object] = {}
        self._n = 0
        self._file.attrs["format"] = "smappy-localizations"
        self._file.attrs["format_version"] = FORMAT_VERSION
        if metadata:
            self.set_metadata(metadata)

    # ------------------------------------------------------------------ writing
    def set_metadata(self, metadata: Dict[str, object]) -> None:
        """Store provenance as JSON; may be called again to add more."""
        self._metadata.update(metadata)
        self._write_metadata()

    def _write_metadata(self) -> None:
        self._file.attrs["metadata"] = json.dumps(self._metadata,
                                                  default=_json_default, indent=1)

    def append(self, locs: Localizations) -> None:
        if len(locs) == 0:
            return
        if not self._datasets:
            self._create(locs)
            # The table itself knows its units, so its metadata fills in what
            # the caller did not give -- but only that.  What the caller passed
            # is the *newer* of the two: the session's log has the run that
            # just finished on it and the table's copy is the one it was
            # loaded with, so updating over the caller here is how a file used
            # to come back with its previous history instead of its current.
            fill = {k: v for k, v in (locs.metadata or {}).items()
                    if k not in self._metadata}
            if fill:
                self.set_metadata(fill)
        elif set(self._datasets) != set(locs.keys()):
            raise ValueError(
                "columns changed between blocks: "
                f"{sorted(set(self._datasets) ^ set(locs.keys()))}")

        n = len(locs)
        for name, dataset in self._datasets.items():
            dataset.resize(self._n + n, axis=0)
            dataset[self._n:self._n + n] = locs[name]
        self._n += n
        self._file.attrs["n_localizations"] = self._n
        self._file.flush()

    def _create(self, locs: Localizations) -> None:
        for name in sorted(locs.keys()):
            column = np.asarray(locs[name])
            self._datasets[name] = self._group.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=column.dtype,
                chunks=(self.chunk,), compression=self.compression)

    # ------------------------------------------------------------------ closing
    def close(self) -> None:
        if self._file:
            self._file.attrs["n_localizations"] = self._n
            self._file.close()
            self._file = None

    def __enter__(self) -> "LocalizationWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return self._n


def save_localizations(path, locs: Localizations,
                       metadata: Optional[Dict[str, object]] = None) -> Path:
    """Write a complete table in one go."""
    with LocalizationWriter(path, metadata or locs.metadata) as writer:
        writer.append(locs)
    return Path(path)


# ------------------------------------------------------------------ GUI state
#
# The evaluation pipeline is provenance -- the columns of a site table mean
# nothing without knowing which evaluators produced them -- so it travels with
# the data, and the rest of the GUI state may come along so that reopening a
# file restores the session that made it.
#
# In a dataset and not in `f.attrs`: an HDF5 attribute is bounded by the object
# header, in practice about 64 kB, which a long pipeline plus a state snapshot
# can pass.  The failure would show up late and only on big projects.

GUI_GROUP = "gui"
GUI_STATE = "gui/state"


def save_gui_state(path, state: Optional[Dict[str, object]]) -> None:
    """Write (or clear) the GUI state of an existing localization file."""
    import h5py
    with h5py.File(path, "a") as f:
        if GUI_STATE in f:
            del f[GUI_STATE]
        if not state:
            return
        f.require_group(GUI_GROUP)
        f.create_dataset(GUI_STATE, data=json.dumps(state, default=_json_default),
                         dtype=h5py.string_dtype("utf-8"))


def load_gui_state(path) -> Optional[Dict[str, object]]:
    """The GUI state saved with a file, or None.  Never raises on a bad one."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            if GUI_STATE not in f:
                return None
            return json.loads(f[GUI_STATE][()])
    except (OSError, KeyError, ValueError):
        return None


# -------------------------------------------------------------- tool results
#
# What a tool worked out, kept so that it can be looked at again.  A drift
# correction subtracts a curve and the corrected table no longer says what the
# curve was; reopening the file a week later and pressing *Plot* should still
# draw it.  Only what a plugin's `keep` hands over is written -- a curve, a
# histogram, the few numbers behind a figure -- never the table itself.
#
# In a dataset of its own for the same reason as the GUI state: a per-frame
# drift curve is far past what an HDF5 attribute holds.

RESULTS_GROUP = "results"
RESULTS = "results/saved"


def save_results(path, results: Optional[Dict[str, object]]) -> None:
    """Write (or clear) the saved tool results of an existing file."""
    import h5py
    with h5py.File(path, "a") as f:
        if RESULTS in f:
            del f[RESULTS]
        if not results:
            return
        f.require_group(RESULTS_GROUP)
        f.create_dataset(RESULTS, data=json.dumps(results, default=_json_default),
                         dtype=h5py.string_dtype("utf-8"))


def load_results(path) -> Dict[str, object]:
    """What the tools that ran on this file left behind; {} if none.

    Never raises: a file from another program, a half-written one, or one
    written by a newer version must still open.
    """
    import h5py
    try:
        with h5py.File(path, "r") as f:
            if RESULTS not in f:
                return {}
            saved = json.loads(f[RESULTS][()])
    except (OSError, KeyError, ValueError, TypeError):
        return {}
    return saved if isinstance(saved, dict) else {}


# -------------------------------------------------------------------- images
#
# Pixel images kept with the table: the camera frames a fit kept
# (`smappy.rawframes`) and any image a layer showed when the file was saved.
# One group each under /images, the pixels in a dataset chunked by plane and
# the placement in its attributes, so that a plane is read without the rest.
#
# Written with mode "a", after the table: `save_localizations` starts the file
# again, and anything written before it would be gone.

IMAGES = "images"


def save_images(path, images) -> None:
    """Write (or clear) the images kept in an existing localization file.

    ``images`` are `ImageData`.  One whose pixels are still in *this* file
    (`StoredStack`) is read in full first, since its group is replaced.
    """
    images = list(images or ())
    arrays = [np.asarray(im.data, dtype=np.float32) for im in images]
    with h5py.File(path, "a") as f:
        if IMAGES in f:
            del f[IMAGES]
        if not images:
            return
        group = f.create_group(IMAGES)
        for n, (im, data) in enumerate(zip(images, arrays)):
            g = group.create_group(str(n))
            g.create_dataset("data", data=data, compression="lzf",
                             chunks=(1,) + data.shape[1:] if data.ndim == 3 else None)
            if im.frames is not None:
                g.create_dataset("frames", data=np.asarray(im.frames, np.int64))
            px, py = im.pixelsize_xy
            g.attrs.update({"name": im.name or "", "kind": im.kind,
                            "pixelsize": float(px), "pixelsize_y": float(py),
                            "x0": float(im.x0), "y0": float(im.y0),
                            "path": str(im.path or ""),
                            "metadata": json.dumps(im.metadata or {},
                                                   default=_json_default)})


class StoredStack:
    """The pixels of a kept image, read from the file a plane at a time.

    A fit keeps 51 frames, and on a full sCMOS chip that is 800 MB: reading
    them all whenever the file is opened would make opening a table cost what
    the table does not need.  A layer shows one plane, so one plane is read,
    and the last one read is kept.  ``np.asarray`` reads everything, which is
    what saving does.
    """

    def __init__(self, path, key: str, shape, dtype):
        self.path, self.key = str(path), key
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self._cached: Tuple = (None, None)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)) and self._cached[0] == int(index):
            return self._cached[1]
        with h5py.File(self.path, "r") as f:
            plane = f[self.key][index]
        if isinstance(index, (int, np.integer)):
            self._cached = (int(index), plane)
        return plane

    def __array__(self, dtype=None, copy=None):
        with h5py.File(self.path, "r") as f:
            data = f[self.key][()]
        return data if dtype is None else data.astype(dtype)


def load_images(path) -> List:
    """The images kept in a localization file, as `ImageData`; [] if none.

    Never raises: a file without them, or a damaged group, opens without.
    """
    from ..images import ImageData
    out = []
    try:
        with h5py.File(path, "r") as f:
            group = f.get(IMAGES)
            if not isinstance(group, h5py.Group):
                return []
            for key in sorted(group, key=lambda k: int(k) if k.isdigit() else 1e9):
                g = group[key]
                data = g["data"]
                a = g.attrs
                px, py = float(a.get("pixelsize", 1.0)), float(a.get("pixelsize_y", 0) or 0)
                out.append(ImageData(
                    data=StoredStack(path, f"{IMAGES}/{key}/data", data.shape, data.dtype),
                    pixelsize=px, pixelsize_y=py if py and py != px else None,
                    x0=float(a.get("x0", 0.0)), y0=float(a.get("y0", 0.0)),
                    name=str(a.get("name", "")), path=str(a.get("path", "")) or None,
                    kind=str(a.get("kind", "image")),
                    frames=g["frames"][()] if "frames" in g else None,
                    metadata=json.loads(a.get("metadata", "{}"))))
    except (OSError, KeyError, ValueError, TypeError):
        return out
    return out


def load_localizations(path) -> Localizations:
    """Read a table written by :class:`LocalizationWriter`."""
    with h5py.File(path, "r") as f:
        if not isinstance(f.get("locs"), h5py.Group):
            what = ("a table in one dataset, as Picasso writes"
                    if isinstance(f.get("locs"), h5py.Dataset) else "no /locs group")
            raise ValueError(f"{Path(path).name} is not a smappy localization "
                             f"file: it has {what}")
        columns = {name: f["locs"][name][()] for name in f["locs"]}
        metadata = {}
        if "metadata" in f.attrs:
            metadata = json.loads(f.attrs["metadata"])
    return Localizations(columns, metadata)


def _json_default(obj):
    """Make numpy values and paths JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
