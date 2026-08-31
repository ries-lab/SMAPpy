"""The short way to call the pipeline.

Everything here is assembly: `fit` builds the camera, the PSF model and the peak
finder from plain values and hands them to :func:`smappy.pipeline.fit_stack`,
which is unchanged and still there for anyone who wants to hold the pieces
themselves.  The point is that an external program -- a microscope control
program, a notebook -- should not have to import from six modules and
rediscover the order they go in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from .detect import AbsoluteCutoff, DoGFilter, DynamicCutoff, GaussFilter, PeakFinder
from .locs import Localizations
from .metadata import CameraMetadata
from .pipeline import FitSettings, fit_stack, provenance
from .psf import GaussianPSF, PSFModel, SplinePSF

CameraLike = Union[CameraMetadata, dict, str, Path, None]


def fit(data, out=None, camera: CameraLike = None, calibration=None, *,
        units: str = "nm", presets=None, sigma: float = 1.2,
        cutoff: float = 1.7, filter: str = "dog", roisize: int = 13,
        max_fit_distance: Optional[float] = None, threads: int = 0,
        chunk: int = 200, frames: Optional[int] = None,
        settings: Optional[FitSettings] = None, progress=None,
        read_ahead: int = 2, collect: bool = True) -> Localizations:
    """Fit a dataset and, with ``out``, write it to HDF5.

    ``data`` is a path to an acquisition, an
    :class:`~smappy.io.tiff.ImageSource`, an array of frames ``(n, y, x)``, or
    any iterable of ``(first_frame, block)`` -- so images already in memory need
    no file.

    ``camera`` is a :class:`~smappy.metadata.CameraMetadata`, a dict of its
    fields, or the path of a YAML config, and overrides whatever the image
    metadata says.  ``presets`` is an optional SMAP ``*_cameras.mat``.  Images
    passed directly carry no metadata at all, so the camera must be complete --
    and coordinates are then relative to the image, unless ``roi`` says where on
    the chip it sat.

    ``calibration`` is a ``_3dcal.mat`` path (or a loaded calibration, or a
    ready :class:`~smappy.psf.PSFModel`); without one the fit is Gaussian and
    there is no z.

    The table is returned whether or not it is also written.  For a long
    acquisition, ``collect=False`` streams to ``out`` alone and returns an empty
    table -- the file is then the result, and memory does not grow with it.
    """
    from .io.hdf5 import LocalizationWriter

    source, blocks = _frames(data, chunk, frames)
    cam = _camera(camera, source, presets)
    model = _model(calibration, sigma, cam)
    finder = PeakFinder(DoGFilter(sigma) if filter == "dog" else GaussFilter(sigma),
                        DynamicCutoff(cutoff) if cutoff < 20 else AbsoluteCutoff(cutoff))
    settings = settings or FitSettings(roisize=roisize, output_unit=units,
                                       max_fit_distance=max_fit_distance,
                                       n_threads=threads)

    writer = None
    if out is not None:
        writer = LocalizationWriter(out)
        writer.set_metadata(provenance(cam, finder, model, settings,
                                       source=_source_name(data, source)))

    collected = Localizations()

    def sink(block: Localizations) -> None:
        if writer is not None:
            writer.append(block)
        if collect:
            if not collected.columns:
                collected.metadata.update(block.metadata)
            collected.extend(block)

    try:
        _, engine = fit_stack(blocks, cam, finder, model, settings, sink=sink,
                              progress=progress, read_ahead=read_ahead)
        if writer is not None:
            # what the run actually did, beside how it was set up
            writer.set_metadata({"stats": dict(engine.stats)})
    finally:
        if writer is not None:
            writer.close()

    collected.metadata["stats"] = dict(engine.stats)
    return collected


def view(locs, settings=None, display=None, block: bool = True,
         group_settings=None):
    """Open the viewer on a table or on a saved HDF5 file."""
    from .viewer import show
    return show(_localizations(locs), settings, display, block=block,
                group_settings=group_settings)


def load(path) -> Localizations:
    """Read a localization file written by smappy."""
    from .io.hdf5 import load_localizations
    return load_localizations(path)


# ------------------------------------------------------------------ assembling
def _frames(data, chunk: int, frames: Optional[int]
            ) -> Tuple[object, Iterable[Tuple[int, np.ndarray]]]:
    """(source or None, an iterable of ``(first_frame, block)``)."""
    if isinstance(data, (str, Path)):
        from .io.tiff import open_stack
        data = open_stack(data)
    if hasattr(data, "frames"):                      # an ImageSource
        stop = min(frames, data.n_frames) if frames else None
        return data, data.frames(chunk=chunk, stop=stop)
    if isinstance(data, np.ndarray):
        block = data[:frames] if frames else data
        if block.ndim == 2:                          # a single frame
            block = block[None]
        return None, [(0, block)]
    return None, data                                # already (index, block)


def _camera(camera: CameraLike, source, presets) -> CameraMetadata:
    if source is not None:
        from .io.tiff import camera_metadata
        return camera_metadata(source, presets, camera)
    if camera is None:
        raise ValueError(
            "images passed directly carry no metadata, so the camera has to be "
            "given: fit(..., camera={'conversion': 6.7, 'offset': 400, "
            "'pixelsize_um': 0.127})")
    cam = _as_camera(camera)
    cam.require()
    return cam


def _as_camera(camera: CameraLike) -> CameraMetadata:
    if isinstance(camera, CameraMetadata):
        return camera
    if isinstance(camera, dict):
        return CameraMetadata.from_dict(camera)
    return CameraMetadata.from_yaml(camera)


def _model(calibration, sigma: float, cam: CameraMetadata) -> PSFModel:
    if calibration is None:
        return GaussianPSF(sigma=sigma)
    if isinstance(calibration, PSFModel):
        return calibration
    if isinstance(calibration, (str, Path)):
        from .io.calibration import load_spline_calibration, warn_on_em_mismatch
        calibration = load_spline_calibration(calibration)
        warn_on_em_mismatch(calibration, cam.em_on)
    return SplinePSF(calibration)


def _localizations(locs) -> Localizations:
    if isinstance(locs, (str, Path)):
        return load(locs)
    return locs


def _source_name(data, source):
    if isinstance(data, (str, Path)):
        return data
    files = getattr(source, "files", None)
    return files[0] if files else None
