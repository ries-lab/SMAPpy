"""Fitting as plugins: raw frames -> localizations.

The fit is assembled from *parts* -- source, camera, detection, PSF model,
fit, output -- each a small settings dataclass, so different fitters share
them and a script builds exactly the same thing::

    from smappy.plugins.fit import GaussianFit, GaussianFitSettings, SourceSettings
    GaussianFit()(None, settings=GaussianFitSettings(
        source=SourceSettings(path="run.ome.tif"),
        camera=CameraSettings(conversion=6.7, offset=400, pixelsize_um=0.127)))

Every part is a field of the wrapper's settings, which the GUI shows as an
expandable section.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..detect import AbsoluteCutoff, DoGFilter, DynamicCutoff, GaussFilter, PeakFinder
from ..locs import Localizations
from ..metadata import CameraMetadata
from ..pipeline import FitSettings, fit_stack, provenance
from ..psf import GaussianPSF, SplinePSF
from . import ParamInfo, Plugin, Result, Selection, param, register

TIFF_FILTER = "Image stacks (*.tif *.tiff *.ome.tif);;All files (*)"


# ---------------------------------------------------------------------- parts
@dataclass
class SourceSettings:
    """Where the frames come from."""
    path: str = param("", label="file", kind="open_file", file_filter=TIFF_FILTER,
                      help="a Micro-Manager TIFF (any file of the series) or "
                           "an NDTiff directory")
    start: int = param(0, label="first frame", min=0, advanced=True)
    stop: Optional[int] = param(None, label="last frame", min=1, advanced=True,
                                help="auto: to the end")
    live: bool = param(False, label="live",
                       help="the file is still being written: fit what is there "
                            "and keep watching for new frames")
    live_timeout: float = param(30.0, label="stop after", unit="s idle", min=1,
                                advanced=True, help="live: give up after this long "
                                "without a new frame")
    chunk: int = param(200, label="frames per block", min=1, advanced=True)


@dataclass
class CameraSettings:
    """ADU -> photons and pixel -> nm.  Empty fields come from the file."""
    preset: str = param("", label="preset", choices=lambda: _preset_choices(),
                        help="a camera YAML; fills the fields below")
    conversion: Optional[float] = param(None, unit="e-/ADU", min=0)
    offset: Optional[float] = param(None, unit="ADU")
    pixelsize_um: Optional[float] = param(None, label="pixel size", unit="um", min=0)
    em_on: Optional[bool] = param(None, label="EM gain on")
    emgain: Optional[float] = param(None, label="EM gain", min=0)

    def overrides(self) -> Dict[str, Any]:
        """What the user set, to win over the file's metadata."""
        return {k: v for k, v in asdict(self).items()
                if k != "preset" and v is not None}

    def resolve(self, source) -> CameraMetadata:
        """The complete camera: the file's metadata under the user's values."""
        overrides = self.overrides()
        if self.preset:
            preset = asdict(CameraMetadata.from_yaml(self.preset))
            overrides = {**{k: v for k, v in preset.items() if v is not None}, **overrides}
        if source is not None:
            from ..io.tiff import camera_metadata
            return camera_metadata(source, overrides=overrides)
        cam = CameraMetadata.from_dict(overrides)
        cam.require()
        return cam


@dataclass
class DetectionSettings:
    """Finding candidates in the filtered image."""
    filter: str = param("dog", choices=(("dog", "difference of Gaussians"),
                                        ("gauss", "Gaussian")))
    sigma: float = param(1.2, label="filter sigma", unit="pix", min=0.1)
    cutoff_mode: str = param("dynamic", label="cutoff",
                             choices=(("dynamic", "dynamic (x noise)"),
                                      ("absolute", "absolute (photons)")))
    cutoff: float = param(1.7, label="cutoff value", min=0)

    def finder(self, n_threads: int = 0) -> PeakFinder:
        image_filter = (DoGFilter(self.sigma) if self.filter == "dog"
                        else GaussFilter(self.sigma))
        cutoff = (DynamicCutoff(self.cutoff) if self.cutoff_mode == "dynamic"
                  else AbsoluteCutoff(self.cutoff))
        return PeakFinder(image_filter, cutoff, n_threads=n_threads)


@dataclass
class GaussianModelSettings:
    """A free-width Gaussian: x, y, photons, background and sigma."""
    sigma: float = param(1.2, label="start sigma", unit="pix", min=0.1)
    elliptical: bool = param(False, help="fit sigma_x and sigma_y separately")

    def model(self) -> GaussianPSF:
        return GaussianPSF(sigma=self.sigma, elliptical=self.elliptical)


@dataclass
class SplineModelSettings:
    """An experimental PSF from a SMAP ``_3dcal.mat``: adds z."""
    calibration: str = param("", label="calibration", kind="open_file",
                             file_filter="SMAP calibration (*_3dcal.mat *.mat)")

    def model(self, camera: Optional[CameraMetadata] = None) -> SplinePSF:
        from ..io.calibration import load_spline_calibration, warn_on_em_mismatch
        if not self.calibration:
            raise ValueError("a spline fit needs a _3dcal.mat calibration file")
        calibration = load_spline_calibration(self.calibration)
        if camera is not None:
            warn_on_em_mismatch(calibration, camera.em_on)
        return SplinePSF(calibration)


@dataclass
class OutputSettings:
    save: bool = param(True, label="save HDF5")
    path: str = param("", label="file", kind="save_file",
                      file_filter="HDF5 (*.hdf5 *.h5)",
                      help="empty: next to the source, as <name>_locs.hdf5")

    def resolve(self, source_path: str) -> Optional[Path]:
        if not self.save:
            return None
        if self.path:
            return Path(self.path)
        src = Path(source_path)
        name = src.name.replace(".ome", "").rsplit(".", 1)[0] if src.is_file() else src.name
        return (src if src.is_dir() else src.parent) / f"{name}_locs.hdf5"


FIT_PARAMS = {
    "roisize": ParamInfo(label="ROI size", unit="pix", min=5),
    "iterations": ParamInfo(min=1, advanced=True),
    "max_block_rois": ParamInfo(label="ROIs per fit", min=100, advanced=True),
    "n_threads": ParamInfo(label="threads", min=0, help="0: one per core", advanced=True),
    "max_fit_distance": ParamInfo(label="max fit distance", unit="pix", advanced=True,
                                  help="reject fits that ran off; auto: keep all"),
    "output_unit": ParamInfo(label="units", choices=("nm", "pixel", "pixel+nm")),
}


def _preset_choices() -> List:
    """Camera YAMLs: ``~/.smappy/cameras`` and the checkout's ``examples``."""
    dirs = [Path.home() / ".smappy" / "cameras",
            Path(__file__).resolve().parents[3] / "examples"]
    extra = os.environ.get("SMAPPY_CAMERAS")
    if extra:
        dirs.insert(0, Path(extra))
    choices = [("", "-")]
    for d in dirs:
        if d.is_dir():
            choices += [(str(p), p.stem) for p in sorted(d.glob("*.yaml"))]
    return choices


# -------------------------------------------------------------------- wrappers
class _FitPlugin(Plugin):
    """What both fitters share: opening the source, streaming, saving."""

    def model(self, settings, camera: CameraMetadata):
        raise NotImplementedError

    def react(self, changed: str, settings) -> Optional[Dict[str, Any]]:
        if changed == "source.path" and settings.source.path:
            try:
                source = _open(settings.source, watch=False)
                from ..io.tiff import camera_metadata
                cam = camera_metadata(source, require=False)
            except Exception:
                return None
            return {f"camera.{k}": v for k, v in asdict(cam).items()
                    if k in ("conversion", "offset", "pixelsize_um", "em_on", "emgain")
                    and v is not None}
        if changed == "camera.preset" and settings.camera.preset:
            cam = CameraMetadata.from_yaml(settings.camera.preset)
            return {f"camera.{k}": v for k, v in asdict(cam).items()
                    if k in ("conversion", "offset", "pixelsize_um", "em_on", "emgain")
                    and v is not None}
        return None

    def run(self, locs, selection: Selection, settings, progress=None,
            stream=None) -> Result:
        from ..io.hdf5 import LocalizationWriter
        from ..live import camera_extent

        src = settings.source
        if not src.path:
            raise ValueError("choose a source file")
        source = _open(src, watch=src.live)
        camera = settings.camera.resolve(source)
        fit = settings.fit
        finder = settings.detection.finder(fit.n_threads)
        model = self.model(settings, camera)
        out = settings.output.resolve(src.path)

        if src.live:
            from ..io.watch import WatchSettings, watch_stack
            blocks = watch_stack(src.path, chunk=src.chunk, start=src.start, stop=src.stop,
                                 settings=WatchSettings(timeout=src.live_timeout))
        else:
            stop = min(src.stop, source.n_frames) if src.stop else None
            blocks = source.frames(chunk=src.chunk, start=src.start, stop=stop)

        if stream:
            stream("start", {"extent": camera_extent(camera, source.shape, fit.output_unit),
                             "path": out})
        writer = None
        if out is not None:
            writer = LocalizationWriter(out)
            writer.set_metadata(provenance(camera, finder, model, fit, source=src.path))
        collected = Localizations()

        def sink(block: Localizations) -> None:
            if writer is not None:
                writer.append(block)
            if not collected.columns:
                collected.metadata.update(block.metadata)
            collected.extend(block)
            if stream:
                stream("block", block)

        def report(engine) -> None:
            if progress:
                s = engine.stats
                progress(f"{s['frames']} frames, {s['localizations']} localizations")

        try:
            _, engine = fit_stack(blocks, camera, finder, model, fit, sink=sink,
                                  progress=report, read_ahead=2)
            if writer is not None:
                writer.set_metadata({"stats": dict(engine.stats)})
        finally:
            if writer is not None:
                writer.close()
        stats = dict(engine.stats)
        collected.metadata["stats"] = stats
        text = (f"{stats['localizations']} localizations from {stats['frames']} frames"
                + (f", saved to {out}" if out else ""))
        return Result(locs=collected.compact(), text=text,
                      data={"stats": stats, "path": out}, settings=settings)


def _open(src: SourceSettings, watch: bool):
    if watch:
        from ..io.watch import WatchSettings, open_growing_stack
        return open_growing_stack(src.path, WatchSettings(timeout=src.live_timeout))
    from ..io.tiff import open_stack
    return open_stack(src.path)


@dataclass
class GaussianFitSettings:
    source: SourceSettings = field(default_factory=SourceSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    model: GaussianModelSettings = field(default_factory=GaussianModelSettings)
    fit: FitSettings = field(default_factory=lambda: FitSettings(output_unit="nm"))
    output: OutputSettings = field(default_factory=OutputSettings)


@dataclass
class SplineFitSettings:
    source: SourceSettings = field(default_factory=SourceSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    model: SplineModelSettings = field(default_factory=SplineModelSettings)
    fit: FitSettings = field(default_factory=lambda: FitSettings(output_unit="nm"))
    output: OutputSettings = field(default_factory=OutputSettings)


@register("Localize/Gaussian 2D")
class GaussianFit(_FitPlugin):
    description = "Detect and fit with a free-width Gaussian PSF: x, y, photons, sigma."
    Settings = GaussianFitSettings
    params = {f"fit.{k}": v for k, v in FIT_PARAMS.items()}

    def model(self, settings, camera):
        return settings.model.model()


@register("Localize/Spline 3D")
class SplineFit(_FitPlugin):
    description = "Detect and fit with an experimental spline PSF from a _3dcal.mat: adds z."
    Settings = SplineFitSettings
    params = GaussianFit.params

    def model(self, settings, camera):
        return settings.model.model(camera)

