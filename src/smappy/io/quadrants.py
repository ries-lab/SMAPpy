"""4Pi bead stacks cut into their quadrants, as MATLAB files.

The Ries-lab 4Pi's bead scans (zenodo 10823308, ``40nm_600``; uiPSF's example
data, ``4pi_40nm_bead``) are MATLAB v7.3 files with one variable per phase
channel, ``qd1`` .. ``qd4``, each a stack over z and the deformable mirrors'
phase steps, and a ``Z Positions.txt`` beside them.  MATLAB writes arrays
column-major, so h5py hands back the axes reversed: (z, phase, x, y).  They
are put the right way round here, (y, x) last, which uiPSF v1 did not do --
so a v1 model learned from these files is transposed against them
(`smappy.fourpi.load_uipsf_v1`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .tiff import ImageSource


def read_quadrant_mat(path) -> np.ndarray:
    """The quadrants of one file as (channel, z, phase, y, x), in the order of
    their names (``qd1``, ``qd2``, ...)."""
    import h5py
    with h5py.File(path, "r") as f:
        names = sorted((k for k in f if k.lower().startswith("qd")),
                       key=lambda k: int(k[2:]) if k[2:].isdigit() else k)
        if not names:
            raise ValueError(f"{Path(path).name} has no qd1..qd4 variables")
        stacks = [np.asarray(f[k][...]) for k in names]
    stacks = [s[None, None] if s.ndim == 2 else s for s in stacks]
    stacks = [s[:, None] if s.ndim == 3 else s for s in stacks]
    return np.stack([np.swapaxes(s, -1, -2) for s in stacks])


def read_z_positions(folder) -> np.ndarray:
    """The piezo positions of a bead scan in um, from its ``Z Positions.txt``,
    or None when there is none."""
    path = Path(folder) / "Z Positions.txt"
    if not path.exists():
        return None
    values = [float(w) for w in path.read_text(encoding="utf-8").replace(",", " ").split()
              if _is_number(w)]
    return np.asarray(values, float)


def _is_number(word: str) -> bool:
    try:
        float(word)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------- SMLM frames
#
# A 4Pi SMLM acquisition in this format is a series of files, ``..._00043_
# 00000.mat``, ``..._00044_00000.mat``, ..., each holding the next frames of
# every quadrant, (frames, x, y) as h5py reads them.  They are read as one
# acquisition whose frames are the quadrants side by side, left to right
# (`smappy.channels.side_by_side`), so that everything that handles a frame
# handles these.

def is_quadrant_mat(path) -> bool:
    """Whether ``path`` is a MATLAB v7.3 file of ``qd1`` .. variables."""
    path = Path(path)
    if path.suffix.lower() != ".mat" or not path.is_file():
        return False
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return any(k.lower().startswith("qd") for k in f)
    except OSError:
        return False


def quadrant_series(path) -> list:
    """The files of the acquisition ``path`` belongs to, in order."""
    return numbered_series(path)


def numbered_series(path) -> list:
    """The files in ``path``'s folder whose names differ from its only in
    their numbers, in numeric order: one acquisition written as many files
    (``..._00043_00000.mat``, ``..._00044_00000.mat``; ``Cell09_642_000_000
    .dcimg`` ... ``_060``)."""
    import re
    path = Path(path)
    pattern = re.sub(r"\d+", r"\\d+", re.escape(path.stem))
    found = [p for p in path.parent.glob("*" + path.suffix)
             if re.fullmatch(pattern, p.stem)]

    def key(p):
        return [int(n) for n in re.findall(r"\d+", p.stem)]
    return sorted(found, key=key) or [path]


def open_quadrant_frames(path):
    """The series ``path`` belongs to, as an `ImageSource` of side-by-side
    quadrants, with its layout as ``source.layout``."""
    import h5py
    from ..channels import side_by_side

    files = quadrant_series(path)
    counts = []
    for f in files:
        with h5py.File(f, "r") as h:
            names = _quadrant_names(h)
            shape = h[names[0]].shape
            if len(shape) < 3:
                raise ValueError(f"{f.name}: expected (frames, x, y) per quadrant, got {shape}")
            # a bead scan's (z, phase steps, x, y): its planes in order, as frames
            counts.append(int(np.prod(shape[:-2])))
    channels, (h, w) = len(names), (shape[-1], shape[-2])
    return QuadrantSource(files=files, shape=(h, channels * w), dtype=np.dtype(np.uint16),
                          n_frames=int(sum(counts)), n_frames_declared=None,
                          mm_metadata={}, summary={}, counts=counts, names=names,
                          layout=side_by_side(channels, w, h, name="quadrant files"))


def _planes(dataset, start: int, stop: int) -> np.ndarray:
    """Planes ``start:stop`` of a quadrant, its leading axes taken as one."""
    if dataset.ndim == 3:
        return dataset[start:stop]
    lead = dataset.shape[:-2]
    index = np.unravel_index(np.arange(start, stop), lead)
    return np.stack([dataset[tuple(int(i[k]) for i in index)] for k in range(stop - start)])


def _quadrant_names(h5) -> list:
    names = sorted((k for k in h5 if k.lower().startswith("qd")),
                   key=lambda k: int(k[2:]) if k[2:].isdigit() else k)
    if not names:
        raise ValueError("no qd1..qd4 variables")
    return names


@dataclass
class QuadrantSource(ImageSource):
    """Frames of a series of quadrant files, the quadrants side by side."""

    counts: list = field(default_factory=list)       # frames in each file
    names: list = field(default_factory=list)        # the quadrants' variables
    layout: object = None                            # a `ChannelLayout`

    def frames(self, chunk: int = 100, start: int = 0, stop: int = None):
        import h5py
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        first, at = 0, start
        pending = []
        for path, count in zip(self.files, self.counts):
            last = first + count
            if last > start and first < stop:
                with h5py.File(path, "r") as h:
                    a, b = max(start, first) - first, min(stop, last) - first
                    for s in range(a, b, chunk):
                        e = min(s + chunk, b)
                        parts = [np.swapaxes(_planes(h[n], s, e), -1, -2)
                                 for n in self.names]
                        pending.append(np.concatenate(parts, axis=-1))
                        # blocks of ``chunk`` frames, across the files' seams
                        while sum(len(p) for p in pending) >= chunk:
                            joined = np.concatenate(pending)
                            yield at, joined[:chunk]
                            at += chunk
                            pending = [joined[chunk:]] if len(joined) > chunk else []
            first = last
        if pending:
            yield at, np.concatenate(pending)

    def watch(self, chunk: int = 100, **kwargs):
        # a finished series: nothing is being written
        return self.frames(chunk=chunk, start=kwargs.get("start", 0), stop=kwargs.get("stop"))
