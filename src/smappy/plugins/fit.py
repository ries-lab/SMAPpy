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
import time
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..detect import AbsoluteCutoff, DoGFilter, DynamicCutoff, GaussFilter, PeakFinder
from ..drift import DriftSettings
from ..locs import Localizations
from ..metadata import CameraMetadata
from ..calibrate.transform import RegisterSettings
from ..pipeline import FitSettings, fit_stack, provenance
from ..psf import GaussianPSF, SplinePSF
from ..rcc import RCCSettings
from . import Context, ParamInfo, Plot, Plugin, Result, param, register
from .assign_colors import AssignColorSettings

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
    # the camera is normally recognised from the file's own tags; this is for
    # the file whose camera carries no tag to recognise it by
    camera: str = param("", label="camera", choices=lambda: _camera_choices(),
                        help="auto: identified from the file's metadata")
    preset: str = param("", label="preset", choices=lambda: _preset_choices(),
                        help="a camera YAML; fills the fields below")
    conversion: Optional[float] = param(None, unit="e-/ADU", min=0)
    offset: Optional[float] = param(None, unit="ADU")
    pixelsize_um: Optional[float] = param(None, label="pixel size", unit="um", min=0)
    pixelsize_y_um: Optional[float] = param(None, label="pixel size y", unit="um",
                                            min=0, advanced=True,
                                            help="auto: square pixels, the same "
                                                 "as in x")
    em_on: Optional[bool] = param(None, label="EM gain on")
    emgain: Optional[float] = param(None, label="EM gain", min=0)

    def overrides(self) -> Dict[str, Any]:
        """What the user set, to win over the file's metadata."""
        out = {k: v for k, v in asdict(self).items()
               if k not in ("preset", "camera", "pixelsize_y_um") and v is not None}
        if self.pixelsize_y_um and self.pixelsize_um:
            out["pixelsize_um"] = [self.pixelsize_um, self.pixelsize_y_um]
        return out

    def resolve(self, source) -> CameraMetadata:
        """The complete camera: the file's metadata under the user's values."""
        overrides = self.overrides()
        if self.preset:
            preset = asdict(CameraMetadata.from_yaml(self.preset))
            overrides = {**{k: v for k, v in preset.items() if v is not None}, **overrides}
        if source is not None:
            from ..io.tiff import camera_metadata
            return camera_metadata(source, overrides=overrides, camera=self.camera)
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


# ------------------------------------------------- once the last frame is in
#
# Two things a two-colour dataset needs before anybody looks at it, and
# neither can be done per block: a drift curve is measured *across* the
# acquisition, and the modes of the colour histogram are a property of the
# whole sample.  Running them at the end, from the fit itself, means the file
# on disk is the finished table rather than a raw fit somebody has to remember
# to correct -- and costs one pass over the localizations, not one per block.
@dataclass
class FinishSettings:
    """What is run over the finished table, before it is saved.

    Drift first, then colours: the correction moves positions and the
    assignment reads photons, so the order changes neither answer, but a
    colour histogram is the last thing one looks at and belongs beside the
    table it will be read from.

    Both are the shipped plugins, with their own settings -- the same code the
    Analysis tab runs, so a fit that finishes itself and a fit finished by hand
    afterwards give the same numbers.
    """
    assign_colors: bool = param(True, label="assign colours",
                                help="split the localizations by their photon "
                                     "ratio and write `channel` "
                                     "(Analysis/Dual-Color/AssignColors)")
    drift: str = param("none", label="drift correction",
                       choices=(("none", "none"), ("rcc", "RCC"),
                                ("comet", "COMET")),
                       help="estimate the drift from the finished table and "
                            "subtract it from every localization.  Off by "
                            "default: it is minutes of work on a dataset it "
                            "cannot see beforehand")
    colors: AssignColorSettings = param(default_factory=AssignColorSettings,
                                        label="colour assignment", advanced=True)
    rcc: RCCSettings = param(default_factory=RCCSettings, label="RCC drift",
                             advanced=True)
    comet: DriftSettings = param(default_factory=DriftSettings,
                                 label="COMET drift", advanced=True)

    def steps(self) -> List[tuple]:
        """``(plugin, settings)`` in the order they run; empty for none.

        The plugins are imported here rather than at the top of the module:
        their settings are needed to declare the fields above, the plugins
        themselves not until somebody runs one.
        """
        out = []
        if self.drift == "rcc":
            from .drift_rcc import RCCDrift
            out.append((RCCDrift(), self.rcc))
        elif self.drift == "comet":
            from .drift_comet import CometDrift
            out.append((CometDrift(), self.comet))
        if self.assign_colors:
            from .assign_colors import AssignColors
            out.append((AssignColors(), self.colors))
        return out


def finish_params() -> Dict[str, ParamInfo]:
    """The finishing plugins' own labels, under ``finish``.

    Read off the plugins rather than copied out: the fields are the same
    numbers with the same meaning, and a second vocabulary for them -- "time
    windows" here, ``n_timepoints`` in the Analysis tab -- would be a way of
    getting one of the two wrong.  `AssignColorSettings` declares its own with
    `param` and needs nothing.
    """
    from .drift_comet import CometDrift
    from .drift_rcc import RCCDrift
    out = {f"finish.comet.{name}": info for name, info in CometDrift.params.items()}
    out.update({f"finish.rcc.{name}": info for name, info in RCCDrift.params.items()})
    return out


@dataclass
class Finished:
    """The outcome of the finishing steps: what to save, and what to say."""
    locs: Localizations
    changed: bool = False               # is the streamed file now out of date?
    notes: List[str] = field(default_factory=list)
    plots: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)


def finish_localizations(locs: Localizations, settings: FinishSettings,
                         report=None) -> Finished:
    """Run the finishing plugins over a finished fit.

    A plain function over a table, so a script can finish a file the same way
    the Localize tab finishes a fit.

    A step that fails is a note and nothing more.  It has to be: these run
    when the frames are already fitted, and a colour histogram with one mode
    or a drift estimate on too few localizations must cost its own step, not
    the twenty minutes that produced the table.
    """
    finished = Finished(locs)
    for plugin, sub in settings.steps():
        label = plugin.name or plugin.path.rsplit("/", 1)[-1]
        if report:
            report(f"{label}: starting")
        try:
            result = plugin.run(Context(locs=finished.locs, progress=report), sub)
        except Exception as error:
            finished.notes.append(f"{label} failed -- "
                                  f"{type(error).__name__}: {error}")
            continue
        if result.locs is not None:
            finished.locs = result.locs
            finished.changed = True
            finished.history.append(
                {"time": datetime.now().isoformat(timespec="seconds"),
                 "what": plugin.path, "text": result.text,
                 "settings": asdict(sub) if is_dataclass(sub) else sub,
                 "changed": True})
        # the drift curve and the colour histogram are the whole reason for
        # looking at a finished fit, so they travel with the fit's own result
        for figure in result.figures():
            name = f"{label}: {figure.name}" if figure.name else label
            finished.plots[name] = figure
        finished.notes.append(f"{label}: {result.text.splitlines()[0]}"
                              if result.text else label)
    return finished


FIT_PARAMS = {
    "roisize": ParamInfo(label="ROI size", unit="pix", min=5),
    "iterations": ParamInfo(min=1, advanced=True),
    "max_block_rois": ParamInfo(label="ROIs per fit", min=100, advanced=True),
    "n_threads": ParamInfo(label="threads", min=0, help="0: one per core", advanced=True),
    "max_fit_distance": ParamInfo(label="max fit distance", unit="pix", advanced=True,
                                  help="reject fits that ran off; auto: keep all"),
    "output_unit": ParamInfo(label="units", choices=("nm", "pixel", "pixel+nm")),
}


def _camera_choices() -> List:
    """The cameras in the database, for a file that identifies none."""
    from ..camera_db import database
    try:
        names = database().names()
    except Exception:
        names = []
    return [("", "auto")] + [(name, name) for name in names]


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

    preview_help = ("run one frame and draw it: the detected candidates over "
                    "the image, and the filtered image the threshold acts on. "
                    "Nothing is saved and the session is not touched.")

    def model(self, settings, camera: CameraMetadata):
        raise NotImplementedError

    def engine(self, settings, camera: CameraMetadata, finder, model):
        """What consumes the frames.  One ROI per candidate by default; the
        two-channel fitter overrides this with one pair per candidate."""
        from ..pipeline import LocalizationEngine
        return LocalizationEngine(camera, finder, model, settings.fit)

    def finder(self, settings, camera: Optional[CameraMetadata] = None):
        """What finds the candidates.  A two-channel fitter overrides this to
        threshold each half of the split frame separately."""
        return settings.detection.finder(settings.fit.n_threads)

    @staticmethod
    def _split_for_detection(geometry, camera) -> Optional[tuple]:
        """``(axis, position)`` in frame coordinates, for a split frame.

        The geometry records the seam on the chip; detection works on the
        frame the camera actually delivered, so the ROI corner comes off.
        """
        from ..dualfit import split_axis
        axis = split_axis(geometry)              # 0 = rows (y), 1 = columns (x)
        offset = camera.roi_offset[1 - axis] if camera is not None else 0
        return (axis, float(geometry["split_position"]) - offset)

    CAMERA_FIELDS = ("conversion", "offset", "pixelsize_um", "em_on", "emgain")

    def resolution(self, settings):
        """Everything known about this acquisition's camera, and from where.

        The `Resolution` the camera database produced, with the file's own
        metadata and the user's settings on top of it -- which is what both
        the hints below and the parameter view read, so that what the view
        shows is what the fit will use and not a second opinion.

        Cached on the source path and the camera settings, because this opens
        the acquisition and the GUI asks after every keystroke.
        """
        path = settings.source.path
        if not path:
            return None
        key = (path, settings.camera.preset, settings.camera.camera, tuple(
            getattr(settings.camera, name) for name in self.CAMERA_FIELDS),
            settings.camera.pixelsize_y_um)
        if getattr(self, "_camera_key", None) == key:
            return self._camera_value
        try:
            from ..io.tiff import resolve_camera
            source = _open(settings.source, watch=False)
            overrides = settings.camera.overrides()
            if settings.camera.preset:
                preset = asdict(CameraMetadata.from_yaml(settings.camera.preset))
                overrides = {**{k: v for k, v in preset.items() if v is not None},
                             **overrides}
            resolved = resolve_camera(source, camera=settings.camera.camera,
                                      overrides=overrides)
        except Exception:
            return None
        self._camera_key, self._camera_value = key, resolved
        return resolved

    def _camera(self, settings) -> Optional[CameraMetadata]:
        """The camera as it stands: the file's metadata under the user's values.

        Never `require`d: a camera that is still missing a field has to hint
        at the fields it does know, not refuse the lot.
        """
        resolved = self.resolution(settings)
        return None if resolved is None else resolved.camera_metadata()

    def hints(self, settings) -> Optional[Dict[str, Any]]:
        """What the camera fields left on *auto* will actually be."""
        camera = self._camera(settings)
        if camera is None:
            return None
        hints = {f"camera.{name}": getattr(camera, name)
                 for name in self.CAMERA_FIELDS}
        sizes = camera.pixelsize_um_xy
        if sizes is not None:
            hints["camera.pixelsize_um"] = sizes[0]
            hints["camera.pixelsize_y_um"] = sizes[1]
        return hints

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

    def finish(self, ctx: Context, settings, locs: Localizations) -> Finished:
        """What is run over the finished table before it is saved.

        Nothing, unless the settings carry a `finish` part -- which is what a
        two-channel fit adds, and what a single-channel one has no use for.
        """
        wanted = getattr(settings, "finish", None)
        if wanted is None:
            return Finished(locs)
        return finish_localizations(locs, wanted, ctx.report)

    def back_projected(self, settings, camera: CameraMetadata, candidates):
        """Peaks from a second channel, drawn where the first channel sees them.

        None for a single-channel fitter.  The two-channel one returns the
        secondary half's peaks mapped into the reference channel, which is the
        cheapest check of a transformation there is: a projected peak that
        misses its partner shows the registration error directly.
        """
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
        finder = self.finder(settings, camera)

        raw = source.frame(index)
        photons = to_photons(raw, camera)/camera.excess_noise
        candidates, filtered = finder(photons[None], first_frame=index)
        maxima = filtered[0][filtered[0] > np.percentile(filtered[0], 99)]
        cutoff = float(finder.cutoff(maxima)) if maxima.size else float("nan")

        try:
            projected = self.back_projected(settings, camera, candidates)
        except Exception:
            projected = None      # no calibration: the fit will say so below

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
            # both columns are chip coordinates -- x_pix has the camera ROI
            # offset added when it is made, x_nm inherits it -- while the image
            # drawn is the ROI, so the offset comes off in either unit.  Taking
            # it off only for nm put every fit of a pixel-unit preview one ROI
            # offset away, onto the other half of a split frame.
            offset = camera.roi_offset
            fitted = (np.asarray(locs[names[0]])/scale-offset[0],
                      np.asarray(locs[names[1]])/scale-offset[1])

        def plot(figure) -> None:
            """Detection on the left, the fit on the right.

            Every peak the finder found, and for two channels the secondary
            peaks projected into the reference channel; the filtered image
            the cutoff acts on; and only what was actually fitted -- which for
            a two-channel fit is one position per emitter, in the reference
            channel, however many halves the peaks were found on.
            """
            peaks, filtered_ax, fits = figure.subplots(1, 3, sharex=True, sharey=True)
            shown = np.percentile(photons, 99.8) or None

            peaks.imshow(photons, cmap="gray", vmax=shown)
            peaks.plot(candidates.x, candidates.y, "o", mfc="none", mec="lime",
                       ms=9, mew=1, label=f"{len(candidates)} peaks")
            if projected is not None and len(projected[0]):
                peaks.plot(*projected, "+", color="cyan", ms=9, mew=1.5,
                           label=f"{len(projected[0])} channel 2 peaks, "
                                 "back-projected")
            peaks.set(title=f"frame {index}: peaks found")
            peaks.legend(fontsize=7, loc="upper right")

            # scaled to the cutoff, not to the brightest pixel: the question
            # this panel answers is how far the peaks sit above the threshold,
            # and a single bright molecule would otherwise flatten the rest
            top = float(cutoff*2) if np.isfinite(cutoff) and cutoff > 0 else None
            image = filtered_ax.imshow(filtered[0], cmap="viridis", vmin=0, vmax=top)
            bar = figure.colorbar(image, ax=filtered_ax, fraction=0.046)
            if top is not None:
                bar.ax.axhline(cutoff, color="red", lw=1.5)
            filtered_ax.plot(candidates.x, candidates.y, ".", color="red", ms=3)
            filtered_ax.set(title=f"{settings.detection.filter} filtered; red "
                                  f"line at the cutoff, {cutoff:.3g}")

            fits.imshow(photons, cmap="gray", vmax=shown)
            if fitted is not None:
                fits.plot(*fitted, "+", color="gold", ms=9, mew=1.5,
                          label=f"{len(fitted[0])} fitted")
                fits.legend(fontsize=7, loc="upper right")
            fits.set(title="fitted" if locs is not None else "not fitted")

            for panel in (peaks, filtered_ax, fits):
                panel.title.set_fontsize(9)
                panel.tick_params(labelsize=8)
            if trouble:
                figure.suptitle("detection only -- " + trouble, fontsize=8,
                                color="#a05000")

        return Result(text=text, settings=settings,
                      plot=Plot(draw=plot, panels=3, size=(15, 6)),
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
        finder = self.finder(settings, camera)
        model = self.model(settings, camera)
        out = settings.output.resolve(src.path)

        if src.live:
            from ..io.watch import WatchSettings, watch_stack
            blocks = watch_stack(src.path, chunk=src.chunk, start=src.start, stop=src.stop,
                                 settings=WatchSettings(timeout=src.live_timeout))
            total = None                 # a growing stack has no end to count to
        else:
            stop = min(src.stop, source.n_frames) if src.stop else None
            blocks = source.frames(chunk=src.chunk, start=src.start, stop=stop)
            total = max((stop if stop is not None else source.n_frames) - src.start, 0)

        ctx.emit("start", {"extent": camera_extent(camera, source.shape, fit.output_unit),
                           "path": out})
        record = provenance(camera, finder, model, fit, source=src.path)
        writer = None
        if out is not None:
            writer = LocalizationWriter(out)
            writer.set_metadata(record)
        collected = Localizations()

        def sink(block: Localizations) -> None:
            if writer is not None:
                writer.append(block)
            if not collected.columns:
                collected.metadata.update(block.metadata)
            collected.extend(block)
            ctx.emit("block", block)

        started = time.perf_counter()

        def report(engine) -> None:
            """Where it has got to, how fast, and how much longer.

            A fit runs for minutes and the only thing on screen was a count of
            frames; the rate is what says whether it is worth waiting for, and
            is the number one compares between machines and settings.
            """
            s = engine.stats
            elapsed = time.perf_counter() - started
            rate = s["frames"] / elapsed if elapsed > 0 else 0.0
            where = f"frame {s['frames']:,}" + (f" of {total:,}" if total else "")
            left = ""
            if total and rate > 0 and s["frames"] < total:
                seconds = int((total - s["frames"]) / rate)
                left = f", {seconds // 60}:{seconds % 60:02d} left"
            ctx.report(f"{where}  {rate:,.1f} frames/s{left}  "
                       f"{s['localizations']:,} localizations")

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
        seconds = time.perf_counter() - started
        rate = stats["frames"] / seconds if seconds > 0 else 0.0
        text = (f"{stats['localizations']} localizations from {stats['frames']} frames"
                f" in {seconds:.1f} s ({rate:,.1f} frames/s)"
                + (f", saved to {out}" if out else ""))

        finished = self.finish(ctx, settings, collected.compact())
        if finished.history:
            # the file says what was done to it, in the shape the session's
            # log uses, so reopening it shows these runs and their settings
            finished.locs.metadata["history"] = (
                list(collected.metadata.get("history") or []) + finished.history)
        if finished.changed and out is not None:
            # The streamed file is the raw fit -- written block by block,
            # which is what makes a crash cost only the last block -- and is
            # now one revision behind the table in hand.  It is rewritten
            # rather than corrected in place: the finishing steps add columns,
            # and an append-only writer cannot widen what it has written.
            ctx.report(f"writing the finished table to {out}")
            record["stats"] = stats
            record["history"] = finished.locs.metadata.get("history") or []
            from ..io.hdf5 import save_localizations
            save_localizations(out, finished.locs, record)
        if finished.notes:
            text = "\n".join([text] + finished.notes)
        return Result(locs=finished.locs, text=text, plots=finished.plots,
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
    finish: FinishSettings = param(default_factory=FinishSettings,
                                   label="after the fit")


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
    params = {**GaussianFit.params, **finish_params()}

    def model(self, settings, camera):
        return settings.model.model(settings.model.load(), camera)

    def back_projected(self, settings, camera, candidates):
        from ..dualfit import secondary_to_reference, which_channel
        calibration = settings.model.load()
        ox, oy = camera.roi_offset
        secondary = ~which_channel(candidates.x + ox, candidates.y + oy,
                                   calibration.geometry)
        mapped = secondary_to_reference(candidates.x[secondary],
                                        candidates.y[secondary],
                                        calibration, (ox, oy))
        return mapped[:, 0], mapped[:, 1]

    def finder(self, settings, camera=None):
        from dataclasses import replace as _replace
        base = super().finder(settings, camera)
        try:
            geometry = settings.model.load().geometry
        except Exception:
            return base                          # no calibration yet: preview only
        return _replace(base, split=self._split_for_detection(geometry, camera))

    def engine(self, settings, camera, finder, model):
        from ..dualfit import DualChannelEngine
        return DualChannelEngine(camera, finder, model, settings.model.load(),
                                 settings.fit, settings.model.photon_ratio)



# ----------------------------------------------------- two channels, no model
TRANSFORM_FILTER = ("Channel transformation (*_2ct.h5 *.h5 *.hdf5);;"
                    "All files (*)")


def _replace_split(finder, split, camera):
    """The peak finder, thresholding each half of the frame on its own."""
    from dataclasses import replace as _replace
    if split is None:
        return finder
    axis, chip = split
    offset = camera.roi_offset[1 - axis] if camera is not None else 0
    return _replace(finder, split=(axis, chip - offset))


def _dim_channel_count(parts, split) -> int:
    """How many localizations the *dimmer* half has, over the blocks so far.

    The dim channel is what limits the pairs -- a pair needs both -- so it is
    the thing to count, and counting the total instead would call a dataset
    finished on the strength of the bright half alone.
    """
    if split is None:
        return sum(len(p) for p in parts)
    axis, chip = split
    column = "x_pix" if axis == 1 else "y_pix"
    low = high = 0
    for part in parts:
        if column not in part:
            return sum(len(p) for p in parts)
        values = np.asarray(part[column], float)
        low += int((values < chip).sum())
        high += int((values >= chip).sum())
    return min(low, high)


def calibration_blocks(start: int, stop: int, frames: int, blocks: int,
                       skip: int = 0):
    """Which frames to measure the channel transformation on.

    Not the first `frames` of the movie.  A registration measured on the
    opening seconds describes the opening seconds: it sees whichever molecules
    happened to be blinking then, and in a sample that is not uniformly
    labelled that is a *part of the field*, which is exactly the thing a
    transformation must not be fitted on only part of.  Later frames also
    differ -- the bright population has bleached, the density has dropped, and
    any slow relative movement of the two channels has happened by then.

    So the same number of frames is taken as evenly spaced blocks across the
    whole movie.  Blocks rather than every n-th frame because reading is
    sequential and a stride would cost far more than it buys; the spread is
    what matters, not the interleaving.

    ``skip`` drops the opening frames, which are not single molecules at all:
    every fluorophore is still on, the frames are often saturated, and what a
    fit makes of them is a dense mat of spurious positions.  Measured on a
    real dataset, including the first hundred frames does not merely add noise
    -- the registration fails outright, because the vote is swamped and the
    projective fit that follows is handed collinear rubbish.  It is skipped
    rather than merely down-weighted because nothing there is usable.
    """
    start, stop = int(start), int(stop)
    if skip > 0 and stop - start > skip:
        start += int(skip)
    span = max(stop - start, 0)
    frames = max(int(frames), 1)
    if span <= frames:
        return [(start, stop)] if span else [(start, start + 1)]
    blocks = max(min(int(blocks), frames), 1)
    per = max(frames // blocks, 1)
    blocks = min(blocks, max(span // per, 1))
    # first block at the start, last one ending at the end, the rest between
    if blocks == 1:
        return [(start, start + per)]
    step = (span - per) / (blocks - 1)
    out = []
    for i in range(blocks):
        a = start + int(round(i * step))
        out.append((a, min(a + per, stop)))
    return out


@dataclass
class ChannelTransformSettings:
    """Where the second channel is.

    Either a transformation measured earlier -- by *Register/Calibrate
    transform*, or a dual-colour bead calibration, which carries one -- or one
    measured from this very movie: fit its first frames with a plain Gaussian
    over the whole frame, register the pairs, and go on to the real fit with
    the result.  The second is the usual case, because a ratiometric
    experiment images every molecule twice and so carries its own
    registration; a saved file is for when the splitter is known to be stable,
    or when the data are too sparse to register on.
    """
    path: str = param("", label="transformation", kind="open_file",
                      file_filter=TRANSFORM_FILTER,
                      help="from Register/Calibrate transform, or a dual-colour "
                           "bead calibration")
    calibrate: bool = param(False, label="calibrate from this movie",
                            help="ignore the file: fit the first frames with a "
                                 "plain Gaussian and register them")
    # A cap, not a target: what is wanted is enough *pairs*, and how many
    # frames that takes depends entirely on the sample and the dye.
    calibrate_frames: int = param(5000, label="at most", unit="frames", min=50)
    calibrate_locs: int = param(10000, label="localizations wanted", min=500,
                                help="in the dimmer of the two channels, which "
                                     "is what limits the pairs; blocks are read "
                                     "until this is reached or the frame cap is")
    calibrate_blocks: int = param(10, label="spread over", unit="blocks", min=1,
                                  help="the frames are taken as this many "
                                       "evenly spaced blocks across the whole "
                                       "movie, not as one run at the start")
    registration: RegisterSettings = field(default_factory=RegisterSettings)
    calibrate_skip: int = param(500, label="skip the first", unit="frames", min=0,
                                help="the opening frames of a movie are not "
                                     "single molecules: everything is on at "
                                     "once and often saturated, and fitting "
                                     "that gives a registration nothing to "
                                     "pair")
    save_calibration: bool = param(True, label="save it", advanced=True,
                                   help="write the measured transformation "
                                        "beside the output, as *_2ct.h5")

    def provisional_split(self, settings):
        """Roughly where the frame divides, before anything has measured it.

        The calibration pass has a chicken-and-egg problem: thresholding each
        half separately is what lets a dim channel be detected at all, and the
        seam is not known until the registration that needs those detections
        has run.  A rough answer settles it.  The threshold only needs the two
        populations kept apart, so a seam a few pixels off costs nothing but a
        thin band judged against the wrong half -- whereas pooling the two
        costs the dim channel outright.

        The chip is split down the middle of its longer axis, which is what a
        split-frame camera ROI looks like, unless the registration settings
        already say otherwise.  Returns ``(axis, chip position)``, axis 0 for
        rows and 1 for columns, or None when there is nothing to go on.
        """
        given = getattr(self, "registration", None)
        layout = getattr(given, "layout", "auto") if given else "auto"
        position = getattr(given, "split_position", None) if given else None
        roi = None
        try:
            camera = settings.camera.resolve(
                _open(replace(settings.source, live=False), watch=False))
            roi = camera.roi
        except Exception:
            return None
        if roi is None:
            return None
        x0, y0, width, height = roi
        if layout != "auto":
            axis = 1 if "right-left" in layout else 0
        else:
            axis = 1 if width > height else 0
        if position is not None:
            return (axis, float(position))
        origin, extent = ((x0, width) if axis == 1 else (y0, height))
        return (axis, origin + extent / 2.0)

    def load(self):
        from ..calibrate.transform import load_transform
        if not self.path:
            raise ValueError("a two-channel fit needs a transformation: choose a "
                             "file, or tick 'calibrate from this movie'")
        return load_transform(self.path)


@dataclass
class DualGaussianModelSettings:
    """A free-width Gaussian per channel, and which parameters they share."""
    sigma: float = param(1.2, label="start sigma", unit="pix", min=0.1)
    link_xy: bool = param(True, label="link x, y",
                          help="one position for both channels, through the "
                               "registration; unlink only to check the transform")
    link_photons: bool = param(False, label="link photons",
                               help="off for two colours -- the photon ratio is "
                                    "what tells the dyes apart")
    link_background: bool = param(False, label="link background", advanced=True)
    link_sigma: bool = param(False, label="link width", advanced=True,
                             help="off: each half finds its own width, which it "
                                  "should -- they see different wavelengths and "
                                  "rarely share a focus")
    photon_ratio: Optional[float] = param(None, label="photon ratio", min=0,
                                          advanced=True,
                                          help="secondary / main; auto: 1")

    def shared(self) -> tuple:
        return (self.link_xy, self.link_xy, self.link_photons,
                self.link_background, self.link_sigma)

    def model(self, channels: int = 2):
        from ..psf import GlobalGaussianPSF
        return GlobalGaussianPSF(sigma=self.sigma, shared=self.shared(),
                                 channels=channels)


@dataclass
class DualGaussianFitSettings:
    source: SourceSettings = field(default_factory=SourceSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    model: DualGaussianModelSettings = field(default_factory=DualGaussianModelSettings)
    transform: ChannelTransformSettings = field(default_factory=ChannelTransformSettings)
    fit: FitSettings = field(default_factory=lambda: FitSettings(output_unit="nm"))
    output: OutputSettings = field(default_factory=OutputSettings)
    finish: FinishSettings = param(default_factory=FinishSettings,
                                   label="after the fit")


@register("Localize/Gaussian 2D 2C")
class DualGaussianFit(_FitPlugin):
    """The 2D two-channel workflow: one emitter, both halves, no PSF model.

    `DualSplineFit` without the calibration.  Candidates are found over the
    whole frame and combined, each gets a ROI in both halves, and the pair is
    fitted at once with x and y shared through the registration -- so the two
    photon numbers that come back are one molecule's, split by colour.

    What it needs instead of a PSF model is the registration, and in a
    ratiometric experiment that can be had from the data: every molecule is
    already imaged twice in the same frame.  Ticking *calibrate from this
    movie* fits the first frames with a plain Gaussian, hands them to
    `smappy.calibrate.transform`, and uses what comes back -- SMAP's
    ``fit_dualcolor`` sequence (fit, ``RegisterLocs2``, refit) as one run.
    """

    description = ("Detect and fit both halves of a split frame as one emitter "
                   "with a Gaussian PSF, sharing x and y: adds the photon ratio "
                   "that tells the two colours apart.  No PSF calibration; the "
                   "registration can be measured from the movie itself.")
    Settings = DualGaussianFitSettings
    params = {**GaussianFit.params, **finish_params()}

    def model(self, settings, camera):
        return settings.model.model()

    # a transformation measured from this movie, kept so that `engine` and
    # `back_projected` need no file: writing one only to read it back would
    # make saving it a precondition of fitting rather than a convenience
    _measured = None

    def transform(self, settings):
        """The registration this fit will use: measured here, or off disk."""
        if self._measured is not None:
            return self._measured
        return settings.transform.load()

    def back_projected(self, settings, camera, candidates):
        from ..dualfit import secondary_to_reference, which_channel
        transform = self.transform(settings)
        ox, oy = camera.roi_offset
        secondary = ~which_channel(candidates.x + ox, candidates.y + oy,
                                   transform.geometry)
        mapped = secondary_to_reference(candidates.x[secondary],
                                        candidates.y[secondary],
                                        transform, (ox, oy))
        return mapped[:, 0], mapped[:, 1]

    def finder(self, settings, camera=None):
        """Each half thresholded on its own.

        The two halves are two detection channels; a dynamic cutoff pooled
        over both is set by whichever is brighter, and the dim one then loses
        the fainter partner of every pair -- which is most of the pairs a
        registration would have had, and most of the colour the fit is for.
        """
        from dataclasses import replace as _replace
        base = super().finder(settings, camera)
        try:
            geometry = self.transform(settings).geometry
        except Exception:
            return base                          # no transformation yet
        return _replace(base, split=self._split_for_detection(geometry, camera))

    def engine(self, settings, camera, finder, model):
        from ..dualfit import DualChannelEngine
        return DualChannelEngine(camera, finder, model, self.transform(settings),
                                 settings.fit, settings.model.photon_ratio)

    def preflight(self, ctx: Context, settings):
        if settings.transform.calibrate or settings.transform.path:
            return None
        return ("No transformation chosen.  Tick 'calibrate from this movie' to "
                "measure one from the first frames, or choose a file.  Start "
                "anyway?")

    def run(self, ctx: Context, settings) -> Result:
        self._measured = None
        if settings.transform.calibrate:
            self._measured = self._calibrate(ctx, settings)
        try:
            return super().run(ctx, settings)
        finally:
            self._measured = None

    # ------------------------------------------------------------ calibrate
    def _calibrate(self, ctx: Context, settings):
        """Fit the first frames plainly, register the pairs, save the result.

        Deliberately the *same* detection and camera settings as the real fit:
        the registration has to describe the peaks the fit will go on to pair,
        and a threshold set here and not there would register on a different
        population than it serves.
        """
        from ..calibrate.transform import register_channels

        from ..locs import concat

        src = settings.source
        provisional = settings.transform.provisional_split(settings)
        source = _open(replace(src, live=False), watch=False)
        camera = settings.camera.resolve(source)
        last = min(src.stop, source.n_frames) if src.stop else source.n_frames
        blocks = calibration_blocks(src.start, last,
                                    settings.transform.calibrate_frames,
                                    settings.transform.calibrate_blocks,
                                    settings.transform.calibrate_skip)
        ctx.report("calibrating the channel transformation")

        # a context of its own: the calibration pass must not push its blocks
        # into the live view the real fit is about to draw into
        quiet = Context(progress=lambda _: None)
        want = int(settings.transform.calibrate_locs)

        def fit_range(start, stop):
            plain = GaussianFitSettings(
                source=replace(src, start=start, stop=stop, live=False),
                camera=settings.camera, detection=settings.detection,
                model=GaussianModelSettings(sigma=settings.model.sigma),
                # pixels as well as nm: a transformation is a statement about
                # the camera, and is fitted in chip pixels
                fit=replace(settings.fit, output_unit="pixel+nm"),
                output=OutputSettings(save=False))
            # the provisional split only decides which half a localization is
            # in, for the threshold and for the count; the seam the fit will
            # actually use is measured by the registration afterwards
            plain_fit = GaussianFit()
            plain_fit.finder = lambda s, c, sp=provisional: _replace_split(
                s.detection.finder(s.fit.n_threads), sp, c)
            return plain_fit.run(quiet, plain).locs

        # One small block first, only to find out how dense this sample is.
        # Stopping early once the count is reached would be the obvious thing
        # and is wrong: on a bright dataset the very first block satisfies it,
        # and the frames then all come from one place -- which is exactly the
        # spreading this went to the trouble of arranging.  So the probe sets
        # how *many* frames are needed, and they are spread regardless.
        probe_start, probe_stop = blocks[0][0], min(blocks[0][0] + 100, last)
        probe = fit_range(probe_start, probe_stop)
        rate = (_dim_channel_count([probe], provisional)
                / max(probe_stop - probe_start, 1)) if probe is not None else 0
        if rate > 0:
            needed = int(min(max(want / rate, 100),
                             settings.transform.calibrate_frames))
            blocks = calibration_blocks(src.start, last, needed,
                                        settings.transform.calibrate_blocks,
                                        settings.transform.calibrate_skip)
        ctx.report(f"{rate:.0f} localizations per frame in the dimmer channel: "
                   f"taking {sum(b - a for a, b in blocks):,} frames in "
                   f"{len(blocks)} block(s) between {blocks[0][0]:,} and "
                   f"{blocks[-1][1]:,}")

        parts, used, dim = [], 0, 0
        for start, stop in blocks:
            part = fit_range(start, stop)
            used += stop - start
            if part is not None and len(part):
                parts.append(part)
        dim = _dim_channel_count(parts, provisional) if parts else 0
        if not parts:
            raise ValueError("the calibration pass found no localizations; check "
                             "the detection cutoff")
        locs = concat(parts) if len(parts) > 1 else parts[0]
        enough = "" if dim >= want else f" -- wanted {want:,}, the movie ran out"
        ctx.report(f"registering {len(locs):,} localizations from {used:,} frames "
                   f"({dim:,} in the dimmer channel{enough})")
        result = register_channels(
            np.asarray(locs["x_pix"], float), np.asarray(locs["y_pix"], float),
            np.asarray(locs["frame"]), source.shape[-2:],
            camera.roi, settings=settings.transform.registration,
            progress=ctx.report,
            precision=(np.asarray(locs["loc_precision_pix"], float)
                       if "loc_precision_pix" in locs else None))

        if settings.transform.save_calibration:
            out = settings.output.resolve(src.path)
            if out is None:
                # saving the localizations is off, but the transformation was
                # still asked for: it belongs beside the movie it describes,
                # not in whatever directory the process happens to be in
                out = default_output_path(src.path)
            path = out.with_name(out.name.rsplit(".", 1)[0] + "_2ct.h5")
            result.save(path, overwrite=True)
            ctx.report(f"transformation saved to {path}")
        p = result.transform.parameters
        ctx.report(f"registered on {p['n_inliers']:,} pairs, residual "
                   f"dx {p['residual_dx_px']:.3f} px, dy {p['residual_dy_px']:.3f} px")
        return result.transform
