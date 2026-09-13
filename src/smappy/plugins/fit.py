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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..detect import AbsoluteCutoff, DoGFilter, DynamicCutoff, GaussFilter, PeakFinder
from ..locs import Localizations
from ..metadata import CameraMetadata
from ..pipeline import FitSettings, fit_stack, provenance
from ..psf import GaussianPSF, SplinePSF
from . import Context, ParamInfo, Plugin, Result, param, register

TIFF_FILTER = "Image stacks (*.tif *.tiff *.ome.tif);;All files (*)"


# ---------------------------------------------------------------------- parts
@dataclass
class SourceSettings:
    """Where the frames come from."""
    path: str = param("", label="file", kind="open_file", file_filter=TIFF_FILTER,
                      help="any file of the acquisition: a Micro-Manager TIFF "
                           "series, an NDTiff directory, or one image out of a "
                           "folder written one file per frame")
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
                      help="filled from the source: <acquisition>_locs.hdf5 "
                           "next to the folder the images are in")

    def resolve(self, source_path: str) -> Optional[Path]:
        if not self.save:
            return None
        if self.path:
            return Path(self.path)
        return default_output_path(source_path)


def default_output_path(source_path) -> Path:
    """Where a fit of ``source_path`` is saved unless told otherwise.

    An acquisition that is a folder -- one file per frame, an NDTiff dataset,
    or a Micro-Manager OME series in its own directory -- is named by that
    folder, and the result goes *next to* it rather than into it: the image
    folder stays images only, and ``img_000000000_Default_000`` or
    ``run_MMStack_Pos0`` would name nothing anyway.  A lone stack file is named
    after itself and saved beside it.
    """
    from ..io.ndtiff import is_ndtiff
    from ..io.singles import is_single_image_set

    src = Path(source_path)
    series = src.is_file() and src.name.lower().endswith((".ome.tif", ".ome.tiff"))
    if src.is_dir() or series or is_single_image_set(src) or is_ndtiff(src):
        folder = src if src.is_dir() else src.parent
        return folder.parent / f"{folder.name}_locs.hdf5"
    name = src.name.rsplit(".", 1)[0]
    return src.parent / f"{name}_locs.hdf5"


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

    def engine(self, settings, camera: CameraMetadata, finder, model):
        """What consumes the frames.  One ROI per candidate by default; the
        two-channel fitter overrides this with one pair per candidate."""
        from ..pipeline import LocalizationEngine
        return LocalizationEngine(camera, finder, model, settings.fit)

    CAMERA_FIELDS = ("conversion", "offset", "pixelsize_um", "em_on", "emgain")

    def _camera(self, settings) -> Optional[CameraMetadata]:
        """The camera as it stands: the file's metadata under the user's values.

        Cached on the source path, because this opens the acquisition and the
        GUI asks after every keystroke that changes a field.
        """
        path = settings.source.path
        if not path:
            return None
        key = (path, settings.camera.preset, tuple(
            getattr(settings.camera, name) for name in self.CAMERA_FIELDS))
        if getattr(self, "_camera_key", None) == key:
            return self._camera_value
        try:
            from ..io.tiff import camera_metadata
            source = _open(settings.source, watch=False)
            overrides = settings.camera.overrides()
            if settings.camera.preset:
                preset = asdict(CameraMetadata.from_yaml(settings.camera.preset))
                overrides = {**{k: v for k, v in preset.items() if v is not None},
                             **overrides}
            # require=False: a camera that is still missing a field has to
            # hint at the fields it does know, not refuse the lot
            camera = camera_metadata(source, overrides=overrides, require=False)
        except Exception:
            return None
        self._camera_key, self._camera_value = key, camera
        return camera

    def hints(self, settings) -> Optional[Dict[str, Any]]:
        """What the camera fields left on *auto* will actually be."""
        camera = self._camera(settings)
        if camera is None:
            return None
        return {f"camera.{name}": getattr(camera, name)
                for name in self.CAMERA_FIELDS}

    def react(self, changed: str, settings) -> Optional[Dict[str, Any]]:
        if changed == "source.path" and settings.source.path:
            updates: Dict[str, Any] = {}
            # the output follows the source, but only while it is ours: a path
            # the user typed is theirs and survives choosing another input
            default = str(default_output_path(settings.source.path))
            if settings.output.path in ("", getattr(self, "_default_output", None)):
                updates["output.path"] = default
                self._default_output = default
            try:
                source = _open(settings.source, watch=False)
                from ..io.tiff import camera_metadata
                cam = camera_metadata(source, require=False)
            except Exception:
                return updates or None
            updates.update({f"camera.{k}": v for k, v in asdict(cam).items()
                            if k in self.CAMERA_FIELDS and v is not None})
            return updates
        if changed == "camera.preset" and settings.camera.preset:
            cam = CameraMetadata.from_yaml(settings.camera.preset)
            return {f"camera.{k}": v for k, v in asdict(cam).items()
                    if k in self.CAMERA_FIELDS and v is not None}
        return None

    def preview(self, ctx: Context, settings, frame: int = 0) -> Result:
        """Detect and fit one frame, and draw what came out.

        Detection is the part a threshold is set by, so it never depends on
        the fit: a missing or unreadable calibration costs the fitted markers
        and earns a line saying why, while the candidates are still drawn over
        the frame they were found in.
        """
        from ..camera import to_photons

        src = settings.source
        if not src.path:
            raise ValueError("choose a source file")
        source = _open(src, watch=False)
        index = int(np.clip(frame, 0, max(source.n_frames-1, 0)))
        # not `resolve`: that insists on a complete camera, and a preview is
        # how a threshold gets set -- a missing pixel size must not stand in
        # the way of looking at one frame.  What detection needs (conversion
        # and offset) still fails loudly if it is missing.
        camera = self._camera(settings)
        if camera is None:
            raise ValueError(f"could not read the camera from {src.path}")
        camera.require("conversion", "offset")
        fit = settings.fit
        finder = settings.detection.finder(fit.n_threads)

        raw = source.frame(index)
        photons = to_photons(raw, camera)/camera.excess_noise
        candidates, filtered = finder(photons[None], first_frame=index)
        maxima = filtered[0][filtered[0] > np.percentile(filtered[0], 99)]
        cutoff = float(finder.cutoff(maxima)) if maxima.size else float("nan")

        locs, trouble = None, ""
        try:
            model = self.model(settings, camera)
            engine = self.engine(settings, camera, finder, model)
            if camera.pixelsize_um is None:
                # pixels, then: a preview is about detection and shape, and
                # nm coordinates would only fail on the missing pixel size
                engine.settings = replace(engine.settings, output_unit="pixel")
            engine.push(raw[None], first_frame=index)
            locs = engine.flush()
        except Exception as error:
            trouble = f"{type(error).__name__}: {error}"

        lines = [f"frame {index} of {source.n_frames}: {len(candidates)} candidates, "
                 f"cutoff {cutoff:.3g} on the filtered image"]
        if locs is not None:
            lines.append(f"{len(locs)} fitted")
        if trouble:
            lines.append("not fitted -- " + trouble)
        text = "; ".join(lines)

        unit = "nm" if (locs is not None and "x_nm" in locs.columns) else "pix"
        fitted = None
        if locs is not None and len(locs):
            scale = (1000.0*float(camera.pixelsize_um)
                     if unit == "nm" and camera.pixelsize_um else 1.0)
            names = ("x_nm", "y_nm") if unit == "nm" else ("x_pix", "y_pix")
            offset = camera.roi_offset if unit == "nm" else (0, 0)
            fitted = (np.asarray(locs[names[0]])/scale-offset[0],
                      np.asarray(locs[names[1]])/scale-offset[1])

        def plot(ax) -> None:
            """Two panels: what was seen, and what the finder saw."""
            figure = ax.figure
            ax.remove()
            figure.set_size_inches(10, 6)
            figure.set_layout_engine("constrained")
            left, right = figure.subplots(1, 2, sharex=True, sharey=True)
            left.imshow(photons, cmap="gray",
                        vmax=np.percentile(photons, 99.8) or None)
            left.plot(candidates.x, candidates.y, "o", mfc="none", mec="lime",
                      ms=9, mew=1, label=f"{len(candidates)} candidates")
            if fitted is not None:
                left.plot(*fitted, "+", color="gold", ms=7,
                          label=f"{len(fitted[0])} fitted")
            left.set(title=f"frame {index} (photons)")
            left.legend(fontsize=7, loc="upper right")
            # scaled to the cutoff, not to the brightest pixel: the question
            # this panel answers is how far the peaks sit above the threshold,
            # and a single bright molecule would otherwise flatten the rest
            top = float(cutoff*2) if np.isfinite(cutoff) and cutoff > 0 else None
            image = right.imshow(filtered[0], cmap="viridis", vmin=0, vmax=top)
            bar = figure.colorbar(image, ax=right, fraction=0.046)
            if top is not None:
                bar.ax.axhline(cutoff, color="red", lw=1.5)
            right.plot(candidates.x, candidates.y, ".", color="red", ms=3)
            right.set(title=f"{settings.detection.filter} filtered; red line at "
                            f"the cutoff, {cutoff:.3g}")
            for panel in (left, right):
                panel.title.set_fontsize(9)
                panel.tick_params(labelsize=8)
            if trouble:
                figure.suptitle("detection only -- " + trouble, fontsize=8,
                                color="#a05000")

        return Result(text=text, plot=plot, settings=settings,
                      data={"frame": index, "candidates": len(candidates),
                            "cutoff": cutoff, "locs": locs, "error": trouble})

    def run(self, ctx: Context, settings) -> Result:
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

        ctx.emit("start", {"extent": camera_extent(camera, source.shape, fit.output_unit),
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
            ctx.emit("block", block)

        def report(engine) -> None:
            s = engine.stats
            ctx.report(f"{s['frames']} frames, {s['localizations']} localizations")

        try:
            from ..pipeline import drive
            engine = self.engine(settings, camera, finder, model)
            drive(engine, blocks, sink=sink, progress=report, read_ahead=2)
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


@dataclass
class DualModelSettings:
    """A dual-colour calibration, and which parameters the channels share."""
    calibration: str = param("", label="calibration", kind="open_file",
                             file_filter="Dual-colour calibration (*.h5 *.hdf5)",
                             help="from Bead calibration in *Dual colour* mode")
    link_xy: bool = param(True, label="link x, y",
                          help="one position for both channels, through the "
                               "registration; unlink only to check the transform")
    link_z: bool = param(True, label="link z",
                         help="one z for both channels: this is the sqrt(2)")
    link_photons: bool = param(False, label="link photons",
                               help="off for two colours -- the photon ratio is "
                                    "what tells the dyes apart; on for biplane")
    link_background: bool = param(False, label="link background", advanced=True)
    photon_ratio: Optional[float] = param(None, label="photon ratio", min=0,
                                          advanced=True,
                                          help="secondary / main; auto: from the "
                                               "calibration's beads")

    def shared(self) -> tuple:
        return (self.link_xy, self.link_xy, self.link_photons,
                self.link_background, self.link_z)

    def load(self):
        from ..calibrate.dual import load_dual_color_calibration
        if not self.calibration:
            raise ValueError("a two-channel fit needs a dual-colour calibration")
        return load_dual_color_calibration(self.calibration)

    def model(self, calibration, camera: Optional[CameraMetadata] = None):
        from ..io.calibration import warn_on_em_mismatch
        from ..psf import GlobalSplinePSF
        if camera is not None:
            warn_on_em_mismatch(calibration.main, camera.em_on)
        return GlobalSplinePSF((calibration.main, calibration.secondary),
                               self.shared())


@dataclass
class DualSplineFitSettings:
    source: SourceSettings = field(default_factory=SourceSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    model: DualModelSettings = field(default_factory=DualModelSettings)
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


@register("Localize/Spline 3D 2C")
class DualSplineFit(_FitPlugin):
    """The 3D two-channel workflow: one emitter, both halves of a split frame.

    Candidates are found over the whole frame and combined, each gets an ROI
    in both channels, and the pair is fitted at once with the parameters the
    form links -- SMAP's ``fit_global_dualchannel``.  Beyond x, y and z it
    yields ``ratio``, the fraction of the photons in the secondary channel,
    which is what separates the two dyes.
    """

    description = ("Detect and fit both halves of a split frame as one emitter, "
                   "sharing x, y and z: adds the photon ratio that tells the "
                   "two colours apart.")
    Settings = DualSplineFitSettings
    params = GaussianFit.params

    def model(self, settings, camera):
        return settings.model.model(settings.model.load(), camera)

    def engine(self, settings, camera, finder, model):
        from ..dualfit import DualChannelEngine
        return DualChannelEngine(camera, finder, model, settings.model.load(),
                                 settings.fit, settings.model.photon_ratio)

