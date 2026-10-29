"""Acquisition discovery and explicit z-volume assembly for bead calibration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import tifffile

from ..io.tiff import _series_files
from ..io.ndtiff import open_ndtiff


class Planes:
    """A z-stack [z, y, x] of camera values that stays in its files: indexing reads the planes it selects,
    ``np.asarray`` the whole stack.

    Nothing needs every bead stack in memory at once -- detection takes one stack at a time, the fit only
    the ROIs cut from them -- yet `read_bead_stacks` used to read them whole, page list and stacked copy,
    and a calibration from full-chip stacks kept 80 GB of them (2026-10).  So a stack holds where its planes
    are, and is read when it is used: once to find its beads, once to cut them.

    A plane is ``(path, page)`` of a TIFF or ``(source, frame)`` of an NDTiff source (memory-mapped);
    `crop` narrows the window read from each, as the halves of a split chip are cut.
    """
    ndim = 3

    def __init__(self, planes, shape, dtype, window=None):
        self._planes = list(planes)
        self._window = window or (slice(0, shape[0]), slice(0, shape[1]))
        rows, cols = (len(range(*w.indices(n))) for w, n in zip(self._window, shape))
        self._full = tuple(shape)
        self.shape = (len(self._planes), rows, cols)
        self.dtype = np.dtype(dtype)

    def __len__(self):
        return self.shape[0]

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def crop(self, rows: slice, cols: slice) -> "Planes":
        """The same planes, read only within ``rows`` × ``cols`` of the current window."""
        def within(outer, inner, n):
            start, stop, _ = outer.indices(n)
            a, b, _ = inner.indices(stop - start)
            return slice(start + a, start + b)
        window = (within(self._window[0], rows, self._full[0]), within(self._window[1], cols, self._full[1]))
        return Planes(self._planes, self._full, self.dtype, window)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        z = np.arange(len(self))[key[0]]
        if np.ndim(z) == 0:
            return self._read([int(z)])[0][key[1:]]
        return self._read(z)[(slice(None),) + key[1:]]

    def __array__(self, dtype=None, copy=None):
        out = self._read(range(len(self)))
        return out if dtype is None else out.astype(dtype, copy=False)

    def _read(self, indices) -> np.ndarray:
        indices = [int(i) for i in indices]
        out = np.empty((len(indices),) + self.shape[1:], self.dtype)
        by_file = {}
        for k, i in enumerate(indices):
            where, page = self._planes[i]
            by_file.setdefault(id(where), (where, []))[1].append((k, page))
        for where, pages in by_file.values():
            if isinstance(where, (str, Path)):
                with tifffile.TiffFile(where) as tf:          # one opening per file, not per plane
                    for k, page in pages:
                        out[k] = tf.pages[page].asarray()[self._window]
            else:
                for k, frame in pages:
                    out[k] = where.frame(frame)[self._window]
        return out


@dataclass
class BeadStack:
    images: np.ndarray                 # z, y, x; original camera values (a `Planes` when read from files)
    z_nm: np.ndarray                    # increasing objective positions
    source: str = "array"
    axes: dict = field(default_factory=dict)  # channel, position, time, etc.
    roi: tuple | None = None  # camera x,y,width,height; None means unknown

    @property
    def dz(self):
        return float(np.median(np.diff(self.z_nm)))


def discover_acquisitions(paths, recursive=True):
    """Return canonical acquisition paths, deduplicating TIFF series and NDTiff.

    A directory may be an acquisition or a parent containing many acquisitions.
    Only TIFF images, NDTiff indexes and 4Pi data -- Hamamatsu DCIMG and
    MATLAB quadrant files, each numbered series one acquisition
    (`io.quadrants.numbered_series`) -- are candidates; outputs are not inputs.
    """
    from ..io.quadrants import is_quadrant_mat, numbered_series
    if isinstance(paths, (str, Path)):
        paths = [paths]
    found = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = [path]
        if path.is_dir():
            candidates = list(path.rglob('*') if recursive else path.iterdir())
        for p in candidates:
            if p.name == 'NDTiff.index':
                found.add(p.parent)
            elif p.is_file() and p.suffix.lower() in ('.tif', '.tiff'):
                if (p.parent / 'NDTiff.index').exists():
                    found.add(p.parent)
                else:
                    found.add(_series_files(p)[0].resolve())
            elif p.is_file() and (p.suffix.lower() == '.dcimg' or is_quadrant_mat(p)):
                found.add(numbered_series(p)[0].resolve())
    return sorted(found, key=lambda p: str(p))


def is_fourpi_scan(path) -> bool:
    """True for 4Pi data: a Hamamatsu DCIMG or a MATLAB quadrant file (`io.quadrants`), which only the
    4Pi mode reads; a folder when every acquisition in it is one."""
    from ..io.quadrants import is_quadrant_mat
    try:
        found = discover_acquisitions([path])
    except (OSError, ValueError):
        return False
    return bool(found) and all(p.suffix.lower() == '.dcimg' or is_quadrant_mat(p) for p in found)


def _plane_axes(md):
    # MM 1.x/2.x TIFF per-plane tags, and MM's newer coordinate dictionary.
    axes = dict(md.get('Axes', md.get('axes', {})))
    for axis, names in {'z': ('SliceIndex', 'Slice'),
                        'time': ('FrameIndex', 'Frame'),
                        'channel': ('ChannelIndex',),
                        'position': ('PositionIndex',)}.items():
        if axis not in axes:
            for name in names:
                if name in md:
                    axes[axis] = md[name]
                    break
    return axes


def _stack(planes):
    """[z, y, x] from planes as `read_bead_stacks` records them: arrays (stacked), or ``(where, index,
    shape, dtype)`` places in the files (a `Planes`, read when used)."""
    if not isinstance(planes[0], tuple):
        return np.stack(planes)
    shapes, dtypes = {p[2] for p in planes}, {np.dtype(p[3]) for p in planes}
    if len(shapes) != 1 or len(dtypes) != 1 or len(next(iter(shapes))) != 2:
        return np.empty((0,))                        # refused by the caller: not one monochrome z, y, x
    return Planes([p[:2] for p in planes], next(iter(shapes)), next(iter(dtypes)))


def _assemble(records, summary, source, dz_nm):
    if not records:
        raise ValueError(f'{source}: no image planes found')
    groups = {}
    for image, axes, md in records:
        group = {k: v for k, v in axes.items() if k != 'z'}
        key = json.dumps(group, sort_keys=True)
        groups.setdefault(key, []).append((image, axes.get('z'), md.get('ZPositionUm'), md.get('ROI', summary.get('ROI'))))
    stacks = []
    for key, planes in groups.items():
        axes = json.loads(key)
        if len(planes) < 4:
            raise ValueError(f'{source}: fewer than four z planes for {axes}; '
                             'time frames are not implicitly treated as z planes')
        zi = [p[1] for p in planes]
        zp = [p[2] for p in planes]
        indexed = all(v is not None for v in zi)
        if any(v is not None for v in zi) and not indexed:
            raise ValueError(f'{source}: incomplete z coordinates for {axes}')
        if indexed:
            zi = np.asarray(zi, float)
            if len(np.unique(zi)) != len(zi):
                raise ValueError(f'{source}: duplicate z planes for {axes}')
            order = np.argsort(zi)
            if not np.allclose(np.diff(zi[order]), 1):
                raise ValueError(f'{source}: missing/nonconsecutive z planes for {axes}')
        else:
            order = np.arange(len(planes))
        physical = all(v is not None for v in zp)
        if physical:
            z = np.asarray(zp, float)[order] * 1000
            d = np.diff(z)
            physical = np.isfinite(z).all() and (np.all(d > 0) or np.all(d < 0))
        if dz_nm is not None:
            if not np.isfinite(dz_nm) or dz_nm <= 0:
                raise ValueError('dz_nm must be positive and finite')
            # Preserve scan direction if objective positions say it was descending.
            direction = -1 if physical and z[-1] < z[0] else 1
            z = np.arange(len(planes)) * float(dz_nm) * direction
        elif not physical:
            step = summary.get('z-step_um', summary.get('z_step_um'))
            if not indexed or step is None or not np.isfinite(float(step)) or float(step) == 0:
                raise ValueError(f'{source}: no usable z spacing; supply dz_nm explicitly')
            z = np.arange(len(planes)) * float(step) * 1000
        if z[-1] < z[0]:
            order = order[::-1]
            z = z[::-1]
        spacing = np.diff(z)
        if not np.allclose(spacing, np.median(spacing), rtol=.02, atol=.1):
            raise ValueError(f'{source}: irregular objective z spacing; '
                             'resampling is not implicit (override dz_nm only if metadata is wrong)')
        # Preserve camera dtype so integer full-scale saturation remains known.
        # Numerical calibration converts to float32 after inspecting it.
        stack = _stack([planes[i][0] for i in order])
        if stack.ndim != 3 or (np.issubdtype(stack.dtype, np.floating) and not np.isfinite(stack).all()):
            raise ValueError(f'{source}: expected finite monochrome z,y,x images')
        rois = []
        for plane in planes:
            value = plane[3]
            if isinstance(value, str):
                import re
                value = re.split(r'[;,\s-]+', value.strip())
            try:
                roi = tuple(int(v) for v in value)
                if len(roi) != 4 or roi[2:] != (stack.shape[2], stack.shape[1]) or min(roi[:2]) < 0:
                    roi = None
            except (TypeError, ValueError):
                roi = None
            rois.append(roi)
        if len(set(rois)) > 1:
            raise ValueError(f'{source}: inconsistent ROI metadata within a z stack')
        stacks.append(BeadStack(stack, z, str(source), axes, rois[0]))
    return stacks


#: frames past which an acquisition that does not say what it is counts as a movie: a bead stack is a
#: few hundred planes, a blinking movie tens of thousands of frames
MOVIE_FRAMES = 2000


def acquisition_kind(path):
    """``("movie" | "stack" | None, frames)``: a blinking movie or a bead z-stack, judged without reading a
    frame.  By Micro-Manager's summary where there is one -- a z-stack has *Slices* above 1, a movie only
    *Frames* (M2's and M5's bead stacks: 101 slices, 1 frame; their movies 100 000 to 750 000 frames, 1
    slice) -- else by length.  None when it cannot tell, or the file cannot be opened.  So that a movie
    added to a bead method is said at once (`psf_window`), not after every page of it was read into
    memory and the stack assembly failed."""
    from ..io.tiff import open_stack
    try:
        source = open_stack(path)
    except Exception:
        return None, 0
    summary = getattr(source, "summary", None) or {}
    n = int(source.n_frames)
    try:
        slices, frames = int(summary.get("Slices") or 0), int(summary.get("Frames") or 0)
    except (TypeError, ValueError):
        slices, frames = 0, 0
    if slices > 1:
        return "stack", n
    if slices == 1 and frames > 1:
        return "movie", n
    return ("movie" if n > MOVIE_FRAMES else None), n


def read_bead_stacks(path, dz_nm=None):
    """Read all distinct channel/position/repeat z-stacks in one acquisition.

    Positive z follows increasing objective position; the fitter reports emitter
    z with the opposite sign. A supplied dz overrides spacing, not grouping.
    """
    path = Path(path)
    if is_fourpi_scan(path):
        # read as a TIFF it failed with "not a TIFF file: header=b'MATL'"
        raise ValueError(f"{path.name} is a 4Pi scan, four phase images per frame: choose the 4Pi mode "
                         "at the top of the window to calibrate it")
    records = []
    if path.is_dir() and (path / 'NDTiff.index').exists():
        source = open_ndtiff(path)
        for i, axes in enumerate(source.index.axes):
            with source.index.path(i).open('rb') as f:
                f.seek(int(source.index.md_offset[i]))
                md = json.loads(f.read(int(source.index.md_length[i])))
            frame = source.frame(i)                   # a view of the memory map: nothing is read
            records.append(((source, i, frame.shape, frame.dtype), axes, md))
        summary = source.summary
    else:
        summary = {}
        for f in _series_files(path):
            with tifffile.TiffFile(f) as tf:
                summary.update((tf.micromanager_metadata or {}).get('Summary', {}))
                for page in tf.pages:
                    tag = page.tags.get('MicroManagerMetadata')
                    md = tag.value if tag is not None else {}
                    if isinstance(md, str):
                        md = json.loads(md)
                    records.append(((f, page.index, page.shape, page.dtype), _plane_axes(md), md))
    return _assemble(records, summary, path, dz_nm)
