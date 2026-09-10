"""Acquisition discovery and explicit z-volume assembly for bead calibration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import tifffile

from ..io.tiff import _series_files
from ..io.ndtiff import open_ndtiff


@dataclass
class BeadStack:
    images: np.ndarray                 # z, y, x; original camera values
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
    Only TIFF images and NDTiff indexes are candidates; outputs are not inputs.
    """
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
    return sorted(found, key=lambda p: str(p))


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
        stack = np.stack([planes[i][0] for i in order])
        if stack.ndim != 3 or not np.isfinite(stack).all():
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


def read_bead_stacks(path, dz_nm=None):
    """Read all distinct channel/position/repeat z-stacks in one acquisition.

    Positive z follows increasing objective position; the fitter reports emitter
    z with the opposite sign. A supplied dz overrides spacing, not grouping.
    """
    path = Path(path)
    records = []
    if path.is_dir() and (path / 'NDTiff.index').exists():
        source = open_ndtiff(path)
        for i, axes in enumerate(source.index.axes):
            with source.index.path(i).open('rb') as f:
                f.seek(int(source.index.md_offset[i]))
                md = json.loads(f.read(int(source.index.md_length[i])))
            records.append((source.frame(i), axes, md))
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
                    records.append((page.asarray(), _plane_axes(md), md))
    return _assemble(records, summary, path, dz_nm)
