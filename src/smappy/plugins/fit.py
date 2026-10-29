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

import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..detect import AbsoluteCutoff, DoGFilter, DynamicCutoff, GaussFilter, PeakFinder
from ..locs import Localizations
from ..metadata import DEFAULT_PIXELSIZE_NM, CameraMetadata
from ..calibrate.transform import RegisterSettings
from ..pipeline import FitSettings, fit_stack, provenance
from ..psf import GaussianPSF, SplinePSF
from . import (Context, FormState, Notice, ParamInfo, Plot, Plugin, PreflightChoice,
               PreflightQuestion, Result, ShapeEdit, param, register)
from .after_fit import AfterFitSettings, Finished, finish_localizations  # noqa: F401 (re-exported)

# a simulation's recipe opens as an acquisition too (`smappy.simulate.source`)
TIFF_FILTER = ("Image stacks (*.tif *.tiff *.ome.tif *.sim.yaml);;"
               "Simulations (*.sim.yaml);;All files (*)")
#: Spline 3D's *focus*: where the focus is in the sample, and how it is known -- the pupil model drawn
#: for it (`smappy.calibrate.depth`) -- or the beads' spline as it was calibrated, at the coverslip
FOCUS_CHOICES = (("auto", "auto"), ("insitu", "in situ"), ("typed", "typed"),
                 ("calibrated", "as calibrated"))
#: frames the auto focus reads, in blocks spread over the movie, before the fit
AUTO_FOCUS_BLOCKS, AUTO_FOCUS_BLOCK_FRAMES = 10, 50


# ---------------------------------------------------------------------- parts
@dataclass
class SourceSettings:
    """Where the frames come from."""
    path: str = param("", label="file", kind="open_file", file_filter=TIFF_FILTER,
                      help="any file of the acquisition: a Micro-Manager TIFF "
                           "series, an NDTiff directory, one image out of a "
                           "folder written one file per frame, or a "
                           "simulation (*.sim.yaml, File > Simulate)")
    start: int = param(0, label="first frame", min=0, advanced=True,
                       help="frames before this one are skipped")
    stop: Optional[int] = param(None, label="last frame", min=1, advanced=True,
                                help="auto: to the end")
    live: bool = param(False, label="live",
                       help="the file is still being written: fit what is there "
                            "and keep watching for new frames")
    live_timeout: float = param(30.0, label="stop after", unit="s idle", min=1,
                                advanced=True, help="live: give up after this long "
                                "without a new frame")
    chunk: int = param(200, label="frames per block", min=1, advanced=True,
                       help="frames read and detected at once; only memory "
                            "and speed depend on it")


@dataclass
class CameraSettings:
    """The microscope and its camera: ADU -> photons and pixel -> nm.  Empty fields come from
    the file, the camera database and the microscope database; shown as one line and a window
    (`gui.microscope_form`), the values locked until unlocked."""
    # recognised from the file's devices; chosen for a file no entry recognises, and else the
    # last one used stands in.  It brings its camera, and its pixel for the 3D lens's state
    microscope: str = param("", label="microscope", choices=lambda: _microscope_choices(),
                            help="auto: recognised from the file's metadata, else the last one "
                                 "used")
    # the camera is normally recognised from the file's own tags, else the microscope's; this
    # is for the file whose camera carries no tag to recognise it by
    camera: str = param("", label="camera", choices=lambda: _camera_choices(),
                        help="auto: identified from the file's metadata, else the microscope's")
    conversion: Optional[float] = param(None, unit="e-/ADU", min=0,
                                        help="photoelectrons per count; auto: "
                                             "from the file or the camera "
                                             "database")
    offset: Optional[float] = param(None, unit="ADU",
                                    help="the count of a pixel that saw no light")
    em_on: Optional[bool] = param(None, label="EM gain on",
                                  help="an EMCCD with its electron multiplication "
                                       "on: the counts are divided by the gain, "
                                       "and the noise is doubled")
    emgain: Optional[float] = param(None, label="EM gain", min=0,
                                    help="the EM gain the camera was set to")
    pixelsize_um: Optional[float] = param(None, label="pixel size", unit="µm", min=0,
                                          help="the pixel's size in the sample, "
                                               "after the magnification")
    pixelsize_y_um: Optional[float] = param(None, label="pixel size y", unit="µm",
                                            min=0,
                                            help="auto: square pixels, the same "
                                                 "as in x, or the microscope's")
    # the fit adds it to the likelihood without EM gain only (`CameraMetadata.
    # readout_variance`), so it is shown only there (`_FitPlugin.describe`)
    read_noise_e: Optional[float] = param(None, label="read noise", unit="e-",
                                          min=0,
                                          help="rms per pixel; auto: the camera's, else 1 "
                                               "(an sCMOS)")
    # metadata, not a setting: kept so a chain that set it still reads (hidden)
    frame_time_ms: Optional[float] = param(None, label="frame time", unit="ms",
                                           min=0, hidden=True,
                                           help="from one frame to the next, kept "
                                                "with the table; auto: from the "
                                                "file, else the exposure")

    def overrides(self) -> Dict[str, Any]:
        """What the user set, to win over the file's metadata."""
        out = {k: v for k, v in asdict(self).items()
               if k not in ("camera", "microscope", "pixelsize_y_um") and v is not None}
        if self.pixelsize_y_um and self.pixelsize_um:
            out["pixelsize_um"] = [self.pixelsize_um, self.pixelsize_y_um]
        return out

    def resolution(self, source):
        """Everything known about the camera behind ``source``, and from where.

        The one place these fields meet a file: the camera database, the
        file's own metadata, then the fields set here, each winning over the
        one before (`io.tiff.resolve_camera`).  The fitters, their parameter
        view and the camera editor all come through
        here, so that none of them can disagree with the fit about a camera.
        """
        from .. import config
        from ..io.tiff import resolve_camera
        resolved = resolve_camera(source, camera=self.camera, overrides=self.overrides(),
                                  microscope=self.microscope,
                                  fallback=str(config.get("last_microscope", "") or ""))
        scope = resolved.microscope
        if scope is not None and not scope.fallback and \
                config.get("last_microscope") != scope.name:
            config.set("last_microscope", scope.name)
        return resolved

    def resolve(self, source, require: bool = True) -> CameraMetadata:
        """The camera the fit uses: `resolution`'s values, or with no file
        these fields alone.  Complete unless ``require`` is False."""
        cam = (CameraMetadata.from_dict(self.overrides()) if source is None
               else self.resolution(source).camera_metadata())
        if require:
            cam.require()
        return cam


@dataclass
class DetectionSettings:
    """Finding candidates in the filtered image."""
    filter: str = param("dog", choices=(("dog", "difference of Gaussians"),
                                        ("gauss", "Gaussian")),
                        help="what the frame is smoothed with before maxima are "
                             "looked for; the difference of Gaussians also "
                             "removes a smooth background")
    sigma: float = param(1.2, label="filter sigma", unit="px", min=0.1,
                         help="the width of the smoothing: about the PSF's")
    cutoff_mode: str = param("dynamic", label="cutoff",
                             choices=(("dynamic", "dynamic (x noise)"),
                                      ("absolute", "absolute (photons)")),
                             help="dynamic: set per frame from the noise of the "
                                  "filtered image; absolute: a fixed value")
    cutoff: float = param(1.7, label="cutoff value", min=0,
                          help="lower finds dimmer molecules, and more noise")

    # what *cutoff value* becomes when *cutoff* is switched (`_FitPlugin.react`):
    # the two read the number on different scales, so 1.7 noise widths kept
    # as 1.7 photons would find every pixel
    CUTOFF_DEFAULTS = {"dynamic": 1.7, "absolute": 20.0}

    def finder(self, n_threads: int = 0) -> PeakFinder:
        image_filter = (DoGFilter(self.sigma) if self.filter == "dog"
                        else GaussFilter(self.sigma))
        cutoff = (DynamicCutoff(self.cutoff) if self.cutoff_mode == "dynamic"
                  else AbsoluteCutoff(self.cutoff))
        return PeakFinder(image_filter, cutoff, n_threads=n_threads)


def in_region(x, y, shapes) -> np.ndarray:
    """Which of the points ``x``, ``y`` the drawn region keeps: inside an
    *include* shape if there is any, and inside no *exclude* shape.

    SMAP's ``RoiAdder``, as shapes rather than a pixel mask: a shape is a
    polygon (a rectangle is four corners) in the frame's pixels, so it is
    kept in the settings, saved with a workspace or a chain, and means the
    same on a stack of any size.  The edge counts as inside
    (`regions.inside_polygon`), as for the ROIs.
    """
    from ..regions import inside_polygon
    x, y = np.asarray(x, float), np.asarray(y, float)
    included = [s["points"] for s in shapes or () if s.get("kind") == "include"]
    excluded = [s["points"] for s in shapes or () if s.get("kind") == "exclude"]
    keep = np.ones(x.shape, bool) if not included else np.zeros(x.shape, bool)
    for points in included:
        keep |= inside_polygon(x, y, points)
    for points in excluded:
        keep &= ~inside_polygon(x, y, points)
    return keep


class RegionFinder:
    """A peak finder that drops the candidates outside the drawn region.

    The region acts on the candidates, not on the image: the frame is
    filtered whole -- a dynamic cutoff is set from the noise of the whole
    frame, as without a region -- and only which peaks go on to be fitted
    changes.  Everything else is the finder's (`__getattr__`), so a preview
    still reads its cutoff.
    """

    def __init__(self, finder, shapes):
        self._finder = finder
        self.shapes = list(shapes)

    def __call__(self, photons, first_frame: int = 0, **kwargs):
        candidates, filtered = self._finder(photons, first_frame=first_frame, **kwargs)
        return candidates[in_region(candidates.x, candidates.y, self.shapes)], filtered

    def __getattr__(self, name):
        return getattr(self._finder, name)

    def __str__(self) -> str:
        return f"{self._finder} in {len(self.shapes)} drawn shapes"


#: how a shape of the region is drawn: what is fitted in, what is left out
REGION_STYLE = {"include": dict(color="#2ca02c", ls="-"),
                "exclude": dict(color="#d62728", ls="--")}


def draw_region(ax, shapes) -> None:
    """The region's shapes as outlines on a frame: green the parts fitted
    in, dashed red the parts left out."""
    for shape in shapes or ():
        points = np.asarray(shape.get("points") or [], float)
        if len(points) < 3:
            continue
        closed = np.vstack([points, points[:1]])
        style = REGION_STYLE.get(shape.get("kind"), REGION_STYLE["include"])
        ax.plot(closed[:, 0], closed[:, 1], lw=1.5, **style)


@dataclass
class FitRegionSettings:
    """Where in the frame to fit: shapes drawn on the preview's frame."""
    shapes: list = param(default_factory=list, label="drawn", kind="shapes",
                         help="drawn on the Preview's frame: candidates outside "
                              "the include shapes, or inside an exclude shape, "
                              "are not fitted")
    use: bool = param(True, label="use the region",
                      help="off fits the whole frame and keeps the shapes")

    def active(self) -> bool:
        return bool(self.use and self.shapes)

    def summary(self) -> str:
        """What the folded part's title says."""
        if not self.shapes:
            return "fit region: the whole frame"
        inside = sum(1 for s in self.shapes if s.get("kind") == "include")
        outside = len(self.shapes) - inside
        bits = ([f"{inside} to fit in"] if inside else []) + \
            ([f"{outside} left out"] if outside else [])
        return "fit region: " + ", ".join(bits) + ("" if self.use else " (off)")


@dataclass
class GaussianModelSettings:
    """A free-width Gaussian: x, y, photons, background and sigma."""
    sigma: float = param(1.2, label="start sigma", unit="px", min=0.1,
                         help="the PSF width the fit starts from; it is fitted")
    elliptical: bool = param(False, help="fit sigma_x and sigma_y separately")

    def model(self) -> GaussianPSF:
        return GaussianPSF(sigma=self.sigma, elliptical=self.elliptical)


@dataclass
class SplineModelSettings:
    """An experimental PSF from a bead calibration (.h5, or SMAP's ``_3dcal.mat``): adds z."""
    calibration: str = param("", label="calibration", kind="open_file",
                             file_filter="Bead calibrations or fitted files (*.h5 *.hdf5 *_3dcal.mat *.mat)",
                             help="empty: auto, the calibration listed for this file's microscope "
                                  "and state (the menu: all found); a bead calibration .h5 or a SMAP "
                                  "_3dcal.mat; or a fitted .h5, for the calibration it was fitted with")
    # shown only for a calibration with a pupil model beside it (`SplineFit._focus_form`); one
    # place in the sample, the focus, known three ways, or the beads' spline as it is
    focus: str = param("auto", label="focus", choices=FOCUS_CHOICES,
                       help="where the focus is in the sample, for the PSF drawn there from the pupil "
                            "model, z then the height above the coverslip: auto, from the lowest "
                            "localizations; in situ, learnt from the molecules with the beads' "
                            "aberrations; typed; or as calibrated, the beads' PSF at the coverslip")
    focus_nm: float = param(0.0, label="position", unit="nm", min=0.0,
                            help="how far into the sample the objective was moved from the "
                                 "coverslip; typed here, else found at Preview or Run")
    sample_index: Optional[float] = param(None, label="sample index", min=1.0, max=2.0,
                                          advanced=True,
                                          help="the sample's refractive index; auto: the beads' "
                                               "medium the model was learnt with")
    z_start_nm: float = param(0.0, label="start z", unit="nm", advanced=True,
                              help="where each fit starts in z, relative to "
                                   "the calibration's focal plane")

    def model(self, camera: Optional[CameraMetadata] = None,
              roisize: Optional[int] = None) -> SplinePSF:
        from ..io.calibration import (load_spline_calibration, warn_on_em_mismatch,
                                      warn_on_microscope_mismatch)
        if not self.calibration:
            raise ValueError("a spline fit needs a bead calibration file")
        calibration = load_spline_calibration(self.calibration)
        start = 0.0
        if self.focus == "typed":
            # a run resolves *auto* to typed first, and *in situ* to its own spline
            # (`SplineFit.with_focus`); a model without either is the beads' spline
            from ..calibrate.depth import at_depth, in_focus
            calibration = at_depth(calibration, self.focus_nm, self.sample_index)
            start = in_focus(calibration)
        elif self.focus == "calibrated":
            from ..calibrate.depth import in_focus
            start = in_focus(calibration)          # an in situ spline's plane in focus, else 0
        if camera is not None:
            warn_on_em_mismatch(calibration, camera.em_on)
            warn_on_microscope_mismatch(calibration, camera.microscope)
        if roisize is not None:
            # the ROI sits in the middle of the calibration grid, and a fit that
            # starts up to a pixel off centre reaches one further: any larger
            # and the model silently repeats its edge
            lateral = min(calibration.shape[1:])
            if roisize > lateral - 2:
                raise ValueError(
                    f"the fit box ({roisize} px) must be at least 2 px smaller than "
                    f"the calibration ({lateral} px across): make the fit box "
                    f"smaller, or the calibration's bead box larger")
        return SplinePSF(calibration, z_start_nm=start + float(self.z_start_nm))


@dataclass
class OutputSettings:
    save: bool = param(True, label="save HDF5",
                       help="write the table to a file as it is fitted")
    path: str = param("", label="file", kind="save_file",
                      file_filter="HDF5 (*.hdf5 *.h5)",
                      help="filled from the source: <acquisition>_locs.hdf5 "
                           "next to the folder the images are in")
    raw_frames: int = param(50, label="raw frames", min=0, advanced=True,
                            help="camera frames kept with the table, in "
                                 "photons, after their average; 0: none")
    show_tags: str = param("PIZStage", label="show image tags", advanced=True,
                           help="image tags drawn while the fit runs and when "
                                "it is done, names or parts of names; empty: "
                                "none (Analysis/Process/Image Tags)")

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
    if src.name.lower().endswith((".sim.yaml", ".sim.yml")):
        name = src.name.rsplit(".", 2)[0]            # run.sim.yaml -> run
    return src.parent / f"{name}_locs.hdf5"


# ------------------------------------------------- once the last frame is in
#
# What a finished fit wants -- z from the widths, the drift taken out, the
# colours put on, a lab's own chain -- cannot be done per block: a drift curve
# is measured *across* the acquisition, and the modes of the colour histogram
# are a property of the whole sample.  Running them at the end, from the fit
# itself, means the file on disk is the finished table rather than a raw fit
# somebody has to remember to correct.  The steps, their order and their
# reasons are `after_fit.py`'s; the names below are kept for scripts.
FinishSettings = AfterFitSettings


def _camera_choices() -> List:
    """The cameras in the database, for a file that identifies none."""
    from ..camera_db import database
    try:
        names = database().names()
    except Exception:
        names = []
    return [("", "auto")] + [(name, name) for name in names]


def _microscope_choices() -> List:
    from ..microscope_db import choices
    return choices()


def camera_resolution(settings):
    """The camera behind a fitter's settings: the database, the file, the
    user's fields -- or None when the source cannot be opened.

    A function of the settings alone, so that the camera editor can be opened
    on what a fitter is set to fit without the fitter that is set to fit it.
    """
    try:
        return settings.camera.resolution(_open(settings.source, watch=False))
    except Exception:
        return None


# -------------------------------------------------------------------- wrappers
class _FitPlugin(Plugin):
    """What every fitter shares: opening the source, the camera, detection,
    streaming, saving.  The four fitters are thin subclasses that say which
    model they fit and, for two channels (`_TwoChannelFit`), where the
    registration comes from."""

    changes_locs = True

    preview_help = ("run one frame and draw it: the detected candidates over "
                    "the image, and the filtered image the threshold acts on. "
                    "Nothing is saved and the session is not touched.")

    def model(self, settings, camera: CameraMetadata):
        raise NotImplementedError

    # ---------------------------------------------------------- the calibration on auto
    #: "single" or "dual" for a fitter that reads a bead calibration (`model.calibration`): then an
    #: empty field is *auto* (`calibration_list`), and a fitted file stands for its calibration
    calibration_kind: Optional[str] = None
    #: two channels' photons in the table: what *after the fit*'s colours need
    two_channel = False
    #: a 4Pi interference phase in the table: what *after the fit*'s 4Pi z needs
    phase = False
    #: the kind of calibration a simulated recipe's (`simulate.source.recipe_calibration`) this
    #: fitter takes, where its model settings do not say; and what it sets beside it
    calibration_kind: Optional[str] = None
    calibration_extras: Dict[str, Any] = {}
    # *after the fit*'s button: its steps on the open table, without fitting again
    part_runs = (("finish", "Run these now", "Analysis/After fit/After the fit"),)

    def found_calibrations(self, settings) -> list:
        """``[(entry, matching)]`` for this file (`calibration_list.find`): the data's microscope,
        its devices' states, its camera and EM gain, and the day it was taken.  Kept per file and
        state of the list, since the form asks with every change."""
        from .. import calibration_list
        from ..microscope_db import acquisition_date
        path = settings.source.path
        listed = calibration_list.list_path()
        stamp = listed.stat().st_mtime if listed.exists() else 0
        key = (path, stamp, settings.camera.microscope, settings.camera.camera)
        if getattr(self, "_found_key", None) == key:
            return self._found_value
        scope, em_on, day = None, None, None
        try:
            resolved = self.resolution(settings)
        except Exception:
            resolved = None
        if resolved is not None:
            camera = resolved.camera_metadata()
            scope, em_on = camera.microscope, camera.em_on
            if resolved.microscope is not None:
                day = acquisition_date(resolved.microscope.metadata or {})
        found = calibration_list.find(scope, em_on, self.calibration_kind == "dual", path, day)
        self._found_key, self._found_value = key, found
        return found

    def with_calibration(self, settings, ctx: Optional[Context] = None):
        """``settings`` with the calibration this run fits with written in: *auto*'s pick for an
        empty field, the calibration a fitted file was fitted with for one of those.  The settings
        handed back (and so the history and the file) name the file used, never "auto"."""
        if self.calibration_kind is None or not settings.source.path:
            return settings
        from .. import calibration_list
        chosen = settings.model.calibration
        tell = ctx.report if ctx is not None else (lambda text: None)
        if chosen:
            recorded = calibration_list.recorded_calibration(chosen)
            if recorded is None:
                return settings
            if not Path(recorded).is_file():
                raise ValueError(f"{Path(chosen).name} was fitted with {recorded}, which is not "
                                 f"there any more")
            tell(f"calibration: {Path(recorded).name}, as {Path(chosen).name} was fitted with")
            return replace(settings, model=replace(settings.model, calibration=recorded))
        found = self.found_calibrations(settings)
        if not found or not found[0][1]:
            kind = "dual-colour" if self.calibration_kind == "dual" else "bead"
            others = (f"; {len(found)} other{'s' if len(found) != 1 else ''} listed, for "
                      f"another microscope or state (choose one from the menu)" if found else "")
            raise ValueError(f"no {kind} calibration listed for this file's microscope and state"
                             f"{others}: make one with PSF calibration, or choose a file")
        entry = found[0][0]
        tell(f"calibration on auto: {entry.label()}")
        return replace(settings, model=replace(settings.model, calibration=entry.path))

    def with_focus(self, ctx: Context, settings, source, camera, finder):
        """``settings`` with a focus in the sample on *auto* or *in situ* found and written in; as
        given for a fitter without one (only Spline 3D has it)."""
        return settings

    def focus_plots(self, settings, camera) -> dict:
        """A Preview's figures of the focus found (`with_focus`), by name; none for a fitter
        without one."""
        return {}

    def _calibration_form(self, settings, state) -> None:
        """What the form shows for *auto*: the pick as the field's hint, every one found in its menu."""
        if self.calibration_kind is None or not settings.source.path:
            return
        found = self.found_calibrations(settings)
        state.choices["model.calibration"] = [
            (e.path, e.label() if ok else f"{e.label()} -- not this microscope or state")
            for e, ok in found]
        state.hints["model.calibration"] = (f"auto: {found[0][0].label()}" if found and found[0][1]
                                            else "auto: none listed for this microscope and state")

    def _model_telling(self, ctx: Context, settings, camera: CameraMetadata):
        """`model`, with what it warned about said where the user reads it.

        Loading a calibration can warn -- the beads were taken with EM gain
        and the data without, say, so the model may be mirrored against the
        data -- and a Python warning goes to a console a GUI user never sees.
        It goes to the run as a warning as well (`Context.warn`), which the
        panel keeps in view and the run puts in its text and its file.
        """
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = self.model(settings, camera)
        for warning in caught:
            ctx.warn(str(warning.message))
            warnings.warn_explicit(warning.message, warning.category,
                                   warning.filename, warning.lineno)
        return model

    def open_source(self, settings, watch: bool = False, stop_event=None):
        """What the frames are read from: the file as the camera wrote it.  A
        fitter of several channels reads them through the channel layer
        instead (`channels.ChannelSource`), and everything after sees
        channels."""
        return _open(settings.source, watch=watch, stop_event=stop_event)

    def watch_blocks(self, source, path, **options):
        """A live fit's blocks of frames (`smappy.io.watch.watch_stack`'s
        stream): read from the path, whatever opened it.  A fitter whose
        source reads the frames its own way (the channel layer) reads them
        from the source."""
        from ..io.watch import watch_stack
        return watch_stack(path, **options)

    def view_camera(self, camera: CameraMetadata) -> CameraMetadata:
        """The camera of the space positions are reported in: the chip's,
        its ROI included.  A channel layer reports them in channel 0's own
        pixels, where the ROI has already been taken off."""
        return camera

    def engine(self, settings, camera: CameraMetadata, finder, model):
        """What consumes the frames.  One ROI per candidate by default; the
        multi-channel fitters override this with one ROI per channel."""
        from ..pipeline import LocalizationEngine
        return LocalizationEngine(camera, finder, model, settings.fit)

    def finder(self, settings, camera: Optional[CameraMetadata] = None):
        """What finds the candidates, in each channel on its own for a
        multi-channel fitter."""
        return settings.detection.finder(settings.fit.n_threads)

    def region_finder(self, settings, camera: Optional[CameraMetadata] = None):
        """`finder`, keeping only the candidates in the drawn region
        (`FitRegionSettings`) when there is one."""
        finder = self.finder(settings, camera)
        region = getattr(settings, "region", None)
        if region is not None and region.active():
            return RegionFinder(finder, list(region.shapes))
        return finder

    CAMERA_FIELDS = ("conversion", "offset", "pixelsize_um", "em_on", "emgain")

    def resolution(self, settings):
        """Everything known about this acquisition's camera, and from where.

        The `Resolution` the camera database produced, with the file's own
        metadata and the user's settings on top of it -- which is what both
        the hints below and the parameter view read, so that what the view
        shows is what the fit will use and not a second opinion.

        Cached on the source path, the camera settings and the database's
        revision (a camera added in the editor is seen at once), because this
        opens the acquisition and the GUI asks after every keystroke.
        """
        from ..camera_db import revision
        from ..microscope_db import revision as microscope_revision
        path = settings.source.path
        if not path:
            return None
        key = (path, settings.camera.camera, settings.camera.microscope, tuple(
            getattr(settings.camera, name) for name in self.CAMERA_FIELDS),
            settings.camera.pixelsize_y_um, revision(), microscope_revision())
        if getattr(self, "_camera_key", None) == key:
            return self._camera_value
        resolved = camera_resolution(settings)
        if resolved is None:
            return None
        self._camera_key, self._camera_value = key, resolved
        return resolved

    def describe(self, ctx: Optional[Context], settings) -> FormState:
        """The database's cameras, read again (one may have been added); what
        the camera fields on *auto* will be; and a file whose camera the
        database does not know.  Best effort: working these out opens the
        file, which may be missing, half-written or not an image at all, and
        none of that is worth an error in a field still being typed."""
        state = FormState(choices={"camera.camera": _camera_choices(),
                                   "camera.microscope": _microscope_choices()})
        try:
            state.hints = self._hints(settings) or {}
        except Exception:
            pass
        try:
            state.notice = self._notice(settings)
        except Exception:
            pass
        try:
            self._calibration_form(settings, state)
        except Exception:
            pass
        try:
            state.titles = {"camera": self._camera_title(settings)}
        except Exception:
            pass
        try:
            details, shown = self._microscope_details(settings)
            if details is not None:
                state.hints["camera"] = details
                state.shown.update(shown)
        except Exception:
            pass
        region = getattr(settings, "region", None)
        if region is not None:
            state.titles["region"] = region.summary()
            state.active["region.use"] = bool(region.shapes)
        finish = getattr(settings, "finish", None)
        if finish is not None:
            # a step only where this fitter's table can take it: colours after
            # two channels, z after an elliptical Gaussian
            from .after_fit import shown
            model = getattr(settings, "model", None)
            can = {"two_channel": self.two_channel,
                   "widths": bool(getattr(model, "elliptical", False)),
                   "phase": self.phase}
            state.shown.update({f"finish.{name}": on
                                for name, on in shown(finish, **can).items()})
            state.titles["finish"] = finish.summary(**can)
            from ..drift import merge_values
            dx, dt = merge_values(ctx.session if ctx is not None else None)
            for part in ("rcc", "comet"):
                state.hints[f"finish.{part}.group_dx_nm"] = dx
                state.hints[f"finish.{part}.group_dt"] = dt
        return state

    def _notice(self, settings) -> Optional[Notice]:
        """A file whose camera the database does not know, said while it is
        chosen, with the way to add it.

        Said even when the fields are all typed in, for a file that names its
        camera: the numbers are then right for this file, and the next file
        from the same camera will need them typed again.  Not for a
        simulation, which says exactly what its camera is, nor for a bare
        TIFF whose numbers are typed in, which has nothing to be recognised
        by.
        """
        from ..io.tiff import SIMULATED
        resolved = self.resolution(settings)
        if resolved is None:
            return None
        origin = (resolved.sources or {}).get("camera_name")
        if origin is not None and (origin.kind, origin.detail) == SIMULATED:
            return None
        missing = [{"pixelsize_um": "pixel size"}.get(name, name)
                   for name in resolved.missing]           # as the form says
        if resolved.camera is None:
            if settings.camera.camera:
                return None            # a name the database no longer has
            device = resolved.values.get("camera_name")
            if not device and not missing:
                return None            # a bare TIFF, its numbers typed in
            text = (f"camera {device!r} is not in the camera database"
                    if device else "this file names no camera")
            text += (f": {', '.join(missing)} must be typed in below."
                     if missing else ".")
            if "pixel size" in missing:
                text += (f" Without a pixel size the nm positions use an "
                         f"assumed {DEFAULT_PIXELSIZE_NM:g} nm.")
            return Notice(text, "Add camera...", "camera editor")
        modal = [name for name, parameter in resolved.camera.parameters.items()
                 if parameter.state]
        if resolved.state is None and modal:
            unknown = [name for name in modal if name in missing]
            text = f"{resolved.camera_name}: this readout mode is not in the camera database"
            text += (f", so {', '.join(unknown)} "
                     f"{'is' if len(unknown) == 1 else 'are'} not known." if unknown
                     else "; the values below are used.")
            return Notice(text, "Add readout mode...", "camera editor")
        return None

    def _camera_title(self, settings) -> str:
        """The microscope part's title while it is folded: the microscope, its state, its camera
        and its pixel, in one line (`microscope_db.one_line`)."""
        from ..microscope_db import one_line
        resolved = self.resolution(settings)
        if resolved is None:
            return "microscope: no file"
        camera = resolved.camera_metadata()
        if camera.microscope:
            return one_line(camera.microscope)
        bits = [str(resolved.camera_name or resolved.values.get("camera_name") or "")]
        if camera.conversion is not None:
            bits.append(f"{camera.conversion:.3g} e-/ADU")
        sizes = camera.pixelsize_um_xy
        if sizes is not None:
            bits.append(f"{1000 * sizes[0]:.4g} nm" if sizes[0] == sizes[1]
                        else f"{1000 * sizes[0]:.4g}\u00d7{1000 * sizes[1]:.4g} nm")
        return "no microscope recognised \u00b7 " + " \u00b7 ".join(b for b in bits if b)

    def _microscope_details(self, settings):
        """``(details, shown)`` for the microscope control: the line, its status, the values the
        fit uses in a sentence and each one's source; and which rows the fit shows -- the read
        noise only without EM gain, where the fit uses it."""
        resolved = self.resolution(settings)
        if resolved is None:
            return None, {}                    # no file: the control says so by itself
        camera = resolved.camera_metadata()
        title = self._camera_title(settings)
        scope = resolved.microscope
        status = ("missing" if resolved.missing else
                  "fallback" if scope is not None and scope.fallback else
                  "unrecognised" if scope is None else "ok")
        sizes = camera.pixelsize_um_xy
        bits = []
        if camera.conversion is not None:
            bits.append(f"{camera.conversion:g} e-/ADU")
        if camera.offset is not None:
            bits.append(f"offset {camera.offset:g}")
        if camera.is_em:
            bits.append(f"EM gain {camera.emgain:g}" if camera.emgain else "EM gain ?")
        if sizes is not None:
            bits.append(f"pixel {1000 * sizes[0]:.4g} nm" if sizes[0] == sizes[1]
                        else f"pixel {1000 * sizes[0]:.4g}\u00d7{1000 * sizes[1]:.4g} nm")
        if resolved.missing:
            bits.append("missing: " + ", ".join(resolved.missing))
        sources = {name: str(resolved.sources[key]) for name, key in (
            ("conversion", "conversion"), ("offset", "offset"), ("em_on", "em_on"),
            ("emgain", "emgain"), ("pixelsize_um", "pixelsize_um"),
            ("read_noise_e", "read_noise_e")) if key in resolved.sources}
        if scope is not None:
            sources["microscope"] = ("the last one used: nothing in the file recognises one"
                                     if scope.fallback else f"recognised: {scope.name}")
            for warning in scope.warnings():
                sources["camera"] = warning
        shown = {"camera.read_noise_e": not camera.is_em,
                 "camera.emgain": camera.is_em}
        return {"line": title, "status": status, "values": ", ".join(bits),
                "sources": sources}, shown

    def unknown_camera(self, settings) -> Optional[str]:
        """The question at Run for a camera that is in no database and
        whose numbers nobody typed, or None.

        The notice (`_notice`) says it while the file is chosen; this asks
        once more at Run, because a fit on a guessed gain and offset gives
        photon counts and precisions that are wrong by the factor of the
        guess, and nothing in the table says so afterwards.  Only while a
        value the fit needs comes from nowhere -- not from the database, the
        file or the form: once typed, or once the camera is added, no
        question.  Never for a simulation, which knows its camera.
        """
        from ..io.tiff import SIMULATED
        if settings.camera.camera:
            return None
        resolved = self.resolution(settings)
        if resolved is None or resolved.camera is not None:
            return None
        origin = (resolved.sources or {}).get("camera_name")
        if origin is not None and (origin.kind, origin.detail) == SIMULATED:
            return None
        if not resolved.missing:
            return None
        device = resolved.values.get("camera_name")
        which = f"This camera ({device})" if device else "This camera"
        return (f"{which} is not in the camera database: its gain and offset "
                "are guesses. Add it to the camera manager?\n\n"
                "Or cancel and type them into the camera fields.")

    def preflight(self, ctx: Context, settings):
        """A camera the database does not know, whose numbers are guesses:
        the editor that adds it, or Cancel (the default) to type them in."""
        text = self.unknown_camera(settings)
        if text is None:
            return None
        return PreflightQuestion(text, run_label="", choices=(
            PreflightChoice("Open camera editor", settings=settings,
                            window="camera editor",
                            help="add the camera, with its conversion and "
                                 "offset, so every file from it is read right"),))

    def _camera(self, settings) -> Optional[CameraMetadata]:
        """The camera as it stands: the file's metadata under the user's values.

        Never `require`d: a camera that is still missing a field has to hint
        at the fields it does know, not refuse the lot.
        """
        resolved = self.resolution(settings)
        return None if resolved is None else resolved.camera_metadata()

    def _hints(self, settings) -> Optional[Dict[str, Any]]:
        """What the camera fields left on *auto* will actually be."""
        camera = self._camera(settings)
        if camera is None:
            return None
        hints = {f"camera.{name}": getattr(camera, name)
                 for name in self.CAMERA_FIELDS}
        # the camera row on *auto* says which camera it found, as the values do
        hints["camera.camera"] = camera.camera_name or None
        sizes = camera.pixelsize_um_xy
        if sizes is not None:
            hints["camera.pixelsize_um"] = sizes[0]
            hints["camera.pixelsize_y_um"] = sizes[1]
        return hints

    def react(self, changed: str, settings) -> Optional[Dict[str, Any]]:
        if changed.endswith("cutoff_mode"):
            # the number is on the other mode's scale: its default instead
            prefix = changed[:-len("cutoff_mode")]
            part = settings
            for name in filter(None, prefix.split(".")):
                part = getattr(part, name)
            value = DetectionSettings.CUTOFF_DEFAULTS.get(part.cutoff_mode)
            return None if value is None else {prefix + "cutoff": value}
        if changed == "source.path" and settings.source.path:
            updates: Dict[str, Any] = {}
            # the output follows the source, but only while it is ours: a path
            # the user typed is theirs and survives choosing another input
            default = str(default_output_path(settings.source.path))
            if settings.output.path in ("", getattr(self, "_default_output", None)):
                updates["output.path"] = default
                self._default_output = default
            # a simulated split camera knows the map between its halves and
            # writes it beside its recipe: a two-colour fitter takes it, as a
            # file picked by hand, unless one was chosen
            transform = getattr(settings, "transform", None)
            if transform is not None and hasattr(transform, "path"):
                from ..simulate.source import recipe_transform
                written = recipe_transform(settings.source.path)
                if written is not None and transform.path in (
                        "", getattr(self, "_default_transform", None)):
                    updates["transform.path"] = str(written)
                    updates["transform.calibrate"] = False
                    self._default_transform = str(written)
            # a recipe drawn with a bead calibration: a spline fitter fits
            # with that calibration, unless one was chosen
            model = getattr(settings, "model", None)
            kind = self.calibration_kind or (
                "single" if isinstance(model, SplineModelSettings)
                else "dual" if isinstance(model, DualModelSettings) else None)
            if kind is not None:
                from ..simulate.source import recipe_calibration
                drawn = recipe_calibration(settings.source.path)
                if drawn is not None and drawn[0] == kind and model.calibration in (
                        "", getattr(self, "_default_calibration", None)):
                    updates["model.calibration"] = drawn[1]
                    updates.update(self.calibration_extras)
                    self._default_calibration = drawn[1]
            # The camera is *not* copied in: a field on auto shows the file's
            # value as a grey hint (`describe`) and stays auto, so the run reads
            # the file it is given.  Copying turned the file's numbers into the
            # user's -- they then overrode a later *camera* choice, and the
            # next file inherited the last one's values.
            return updates or None
        return None

    def finish(self, ctx: Context, settings, locs: Localizations) -> Finished:
        """What is run over the finished table before it is saved: the
        *after the fit* part (`after_fit.py`), nothing without one."""
        wanted = getattr(settings, "finish", None)
        if wanted is None:
            return Finished(locs)
        from .after_fit import with_session_merge
        return finish_localizations(locs, with_session_merge(wanted, ctx.session), ctx.report)

    def preview(self, ctx: Context, settings, frame: int = 0) -> Result:
        """Detect and fit one frame, and draw what came out.

        Detection is the part a threshold is set by, so it never depends on
        the fit: a missing or unreadable calibration costs the fitted markers
        and earns a line saying why, while the candidates are still drawn over
        the frame they were found in.
        """
        from ..camera import to_photons

        try:
            settings = self.with_calibration(settings)
        except ValueError:
            pass                       # the model says why, below, and the candidates still show
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
        finder = self.region_finder(settings, camera)

        raw = source.frame(index)
        photons = to_photons(raw, camera)/camera.excess_noise
        candidates, filtered = finder(photons[None], first_frame=index)
        # what detection applied to this frame, from the finder's own code
        cutoff = float(finder.cutoffs(filtered)[0][0])
        said = f"{cutoff:.3g}"

        region = getattr(settings, "region", None)
        shapes = list(getattr(region, "shapes", []) or [])

        locs, trouble = None, ""
        # the focus in the sample found first, as a run finds it, and drawn beside the frame
        fitting, extra = settings, {}
        try:
            fitting = self.with_focus(ctx, settings, source, camera, finder)
            extra = self.focus_plots(settings, camera)
        except Exception as error:
            trouble = f"focus: {error}"
        try:
            model = self._model_telling(ctx, fitting, camera)
            engine = self.engine(fitting, camera, finder, model)
            engine.push(raw[None], first_frame=index)
            locs = engine.flush()
        except Exception as error:
            trouble = "; ".join(filter(None, [trouble, f"{type(error).__name__}: {error}"]))

        lines = [f"frame {index} of {source.n_frames}: {len(candidates)} candidates, "
                 f"cutoff {said} on the filtered image"]
        if locs is not None:
            lines.append(f"{len(locs)} fitted")
        if trouble:
            lines.append("not fitted -- " + trouble)
        text = "; ".join(lines)

        fitted = None
        if locs is not None and len(locs):
            # drawn over the frame, so in its pixels: a chip coordinate (the
            # camera ROI offset added when it is made) while the image drawn
            # is the ROI, so the offset comes off
            from ..locs import in_pixels
            offset = camera.roi_offset
            x, y = in_pixels(locs, "x_pix", "y_pix")
            fitted = (x - offset[0], y - offset[1])

        def plot(figure) -> None:
            """Detection on the left, the fit on the right: every peak the
            finder found, the filtered image the cutoff acts on, and only what
            was actually fitted."""
            peaks, filtered_ax, fits = figure.subplots(1, 3, sharex=True, sharey=True)
            shown = np.percentile(photons, 99.8) or None

            peaks.imshow(photons, cmap="gray", vmax=shown)
            peaks.plot(candidates.x, candidates.y, "o", mfc="none", mec="lime",
                       ms=9, mew=1, label=f"{len(candidates)} peaks")
            peaks.set(title=f"frame {index}: peaks found")
            peaks.legend(fontsize=7, loc="upper right")
            draw_region(peaks, shapes)

            # scaled to the cutoff, not to the brightest pixel: the question
            # this panel answers is how far the peaks sit above the threshold,
            # and a single bright molecule would otherwise flatten the rest
            top = 2*cutoff if np.isfinite(cutoff) and cutoff > 0 else None
            image = filtered_ax.imshow(filtered[0], cmap="viridis", vmin=0, vmax=top)
            bar = figure.colorbar(image, ax=filtered_ax, fraction=0.046)
            if top is not None:
                bar.ax.axhline(cutoff, color="red", lw=1.5)
            filtered_ax.plot(candidates.x, candidates.y, ".", color="red", ms=3)
            filtered_ax.set(title=f"{settings.detection.filter} filtered; "
                                  f"red line at the cutoff, {said}")

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

        # the frame panel takes the fit region, drawn with the mouse (the
        # window's tools, `ShapeEdit`), written back into *fit region*
        edit = (ShapeEdit("region.shapes", panel=0, shapes=shapes)
                if getattr(settings, "region", None) is not None else None)
        return Result(text=text, settings=settings, plots=extra,
                      plot=Plot(draw=plot, panels=3, size=(15, 6), edit=edit),
                      data={"frame": index, "candidates": len(candidates),
                            "cutoff": cutoff, "locs": locs, "error": trouble})

    def run(self, ctx: Context, settings) -> Result:
        from ..io.hdf5 import LocalizationWriter
        from ..live import camera_extent

        settings = self.with_calibration(settings, ctx)
        src = settings.source
        if not src.path:
            raise ValueError("choose a source file")
        source = self.open_source(settings, watch=src.live, stop_event=ctx.stop)
        camera = settings.camera.resolve(source)
        fit = settings.fit
        finder = self.region_finder(settings, camera)
        settings = self.with_focus(ctx, settings, source, camera, finder)
        model = self._model_telling(ctx, settings, camera)
        out = settings.output.resolve(src.path)

        # what the microscope recorded per frame, read with the pixels
        from ..frametags import FrameTags, acquisition
        tags = FrameTags()
        behind = [0]              # frames written and not yet read: the lag
        if src.live:
            from ..io.watch import WatchSettings
            # given `ctx.writer_finished`, the writer's word ends the watch
            # and *stop after* is never consulted (`smappy.io.watch`)
            def on_behind(frames: int) -> None:
                behind[0] = frames
            blocks = self.watch_blocks(source, src.path, chunk=src.chunk, start=src.start,
                                       stop=src.stop,
                                       settings=WatchSettings(timeout=src.live_timeout,
                                                              poll=LIVE_POLL_SECONDS),
                                       tags=tags, stop_event=ctx.stop,
                                       finished=ctx.writer_finished, idle_blocks=True,
                                       on_behind=on_behind)
            total = None                 # a growing stack has no end to count to
        else:
            stop = min(src.stop, source.n_frames) if src.stop else None
            source.tags = tags
            blocks = _until(source.frames(chunk=src.chunk, start=src.start, stop=stop),
                            ctx.stop)
            total = max((stop if stop is not None else source.n_frames) - src.start, 0)
        # the frames kept with the table: counted as they go past, so the
        # stack is read once, for the fit
        from ..rawframes import RawFrameKeeper
        keeper = RawFrameKeeper(settings.output.raw_frames, src.start,
                                None if src.live else src.start + total)
        blocks = keeper.watch(blocks)

        if camera.pixelsize_um is None:
            ctx.warn(f"no pixel size: x_nm and y_nm use an assumed "
                     f"{DEFAULT_PIXELSIZE_NM:g} nm per pixel, which the table "
                     f"records, so they divide back into camera pixels exactly")
        ctx.emit("start", {"extent": camera_extent(self.view_camera(camera), source.shape),
                           "path": out})
        record = provenance(camera, finder, model, fit, source=src.path)
        if self.calibration_kind is not None and "calibration" not in record:
            # a two-channel model is two splines and records neither: the file it came from
            record["calibration"] = {"file": str(settings.model.calibration)}
        record["acquisition"] = acquisition(source)
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
        warned = [False]

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
            lag = ""
            if behind[0] and rate > 0:
                seconds = behind[0] / rate
                lag = f"  {behind[0]:,} frames behind"
                if seconds > LIVE_BEHIND_WARN_SECONDS and not warned[0]:
                    warned[0] = True
                    ctx.warn(f"the fit cannot keep up with the camera: "
                             f"{behind[0]:,} frames (about {seconds:.0f} s) "
                             f"behind, at {rate:,.0f} frames/s; what is shown "
                             f"lags the acquisition, and catches up once it "
                             f"slows or ends")
            ctx.report(f"{where}  {rate:,.1f} frames/s{left}  "
                       f"{s['localizations']:,} localizations{lag}")
            # the same as numbers, for a program following the fit
            # (smappy-microclaw's metrics): frames are counted from frame 0
            ctx.emit("fitted", {"frames": src.start + s["frames"],
                                "localizations": s["localizations"],
                                "behind": behind[0]})

        tags_shown = [time.perf_counter(), -1]      # when, and at how many frames

        def live_tags() -> None:
            """The image tags so far, every `LIVE_TAGS_SECONDS`: into the live
            table's metadata (``("tags", table)``), and drawn -- the same
            figures as at the end (`image_tags.plots`) -- when *show image
            tags* names some (``("plots", {name: draw})``).

            A fit of 90 000 frames runs for many minutes, and a focus lock
            that has run into its limit is what one wants to see while it can
            still be done something about, not after.
            """
            now = time.perf_counter()
            if now - tags_shown[0] < LIVE_TAGS_SECONDS or len(tags) == tags_shown[1]:
                return
            tags_shown[:] = [now, len(tags)]
            table = tags.table()
            if len(table) < 2:                     # nothing has changed yet
                return
            ctx.emit("tags", table)
            if settings.output.show_tags:
                from .image_tags import names, plots as tag_plots
                shown = names(table, settings.output.show_tags)
                if shown:
                    frames = (np.asarray(collected["frame"])
                              if "frame" in collected else None)
                    ctx.emit("plots", tag_plots(table, shown, frames))

        def progress(engine) -> None:
            report(engine)
            live_tags()

        try:
            from ..pipeline import drive
            engine = self.engine(settings, camera, finder, model)
            drive(engine, blocks, sink=sink, progress=progress, read_ahead=2,
                  flush_seconds=0.0 if src.live else None)
            record["frame_tags"] = tags.table()
            # kept with the localizations, so a file that was fitted against
            # a mismatched calibration says so whoever opens it
            record["warnings"] = list(ctx.warnings)
            if writer is not None:
                writer.set_metadata({"stats": dict(engine.stats),
                                     "frame_tags": record["frame_tags"],
                                     "acquisition": record["acquisition"]}
                                    | ({"warnings": record["warnings"]}
                                       if record["warnings"] else {}))
        finally:
            if writer is not None:
                writer.close()
        stats = dict(engine.stats)
        collected.metadata["stats"] = stats
        for key in ("frame_tags", "acquisition", "warnings"):
            if record[key]:
                collected.metadata[key] = record[key]
        seconds = time.perf_counter() - started
        rate = stats["frames"] / seconds if seconds > 0 else 0.0
        stopped = ctx.stop is not None and ctx.stop.is_set()
        text = (f"{stats['localizations']} localizations from {stats['frames']} frames"
                f" in {seconds:.1f} s ({rate:,.1f} frames/s)"
                + (", stopped early" if stopped else "")
                + (f", saved to {out}" if out else ""))

        # A stopped run keeps the raw fit and finishes nothing: whoever asked
        # wants the file now (MicroClaw allows ten seconds), and a drift
        # correction of part of an acquisition is not one anybody asked for.
        finished = (Finished(collected.compact()) if stopped
                    else self.finish(ctx, settings, collected.compact()))
        # the table in hand says what it was fitted from, as its file does: a
        # comparison with a simulation's truth finds the recipe by it
        finished.locs.metadata.setdefault("source", str(src.path))
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
        raw = keeper.image(self.view_camera(camera), source=str(src.path),
                           name=f"{Path(str(src.path)).name}: raw frames")
        if raw is not None and out is not None:
            # last, after any rewrite above: that one starts the file again
            from ..io.hdf5 import save_images
            save_images(out, [raw])
        if finished.notes:
            text = "\n".join([text] + finished.notes)
        # in the text, so they are in the session's history and the file's
        # with it, beside the run they belong to
        if ctx.warnings:
            text = "\n".join([text] + [f"warning: {w}" for w in ctx.warnings])
        plots = dict(finished.plots)
        if record["frame_tags"] and settings.output.show_tags:
            from .image_tags import names, plots as tag_plots, summary
            shown = names(record["frame_tags"], settings.output.show_tags)
            plots.update(tag_plots(record["frame_tags"], shown,
                                   np.asarray(finished.locs["frame"])))
            if shown:
                text = "\n".join([text, summary(record["frame_tags"], shown)])
        return Result(locs=finished.locs, text=text, plots=plots,
                      # the file holds this very table: written as it came,
                      # and again after the finishing steps if they changed it
                      data={"stats": stats},
                      written=None if out is None else str(out), saved=out is not None,
                      stopped=stopped, images=[raw] if raw is not None else [],
                      settings=settings)


# how often a live fit looks for new frames.  Each block read is fitted at once
# (`drive`'s ``flush_seconds=0``) rather than when thousands of boxes have
# gathered or every 5 s, as before: the time from a frame written to its
# localizations was 2.8-3.3 s at the median and up to 6.7 s, which is too slow
# for anything that acts on the result.  A fitter that falls behind is handed
# up to ``chunk`` frames at a time, so it catches up in large blocks.
LIVE_POLL_SECONDS = 0.05
# a live fit this far behind the camera says so, once
LIVE_BEHIND_WARN_SECONDS = 5.0
# how often a running fit hands on its image tags and redraws their figures:
# drawing them is a few tens of milliseconds on the GUI thread, and a focus
# lock drifts over minutes, not seconds
LIVE_TAGS_SECONDS = 5.0


def _until(blocks, stop_event):
    """``blocks`` until ``stop_event`` is set; all of them without one."""
    for item in blocks:
        if stop_event is not None and stop_event.is_set():
            return
        yield item


def _open(src: SourceSettings, watch: bool, stop_event=None):
    if watch:
        from ..io.watch import WatchSettings, open_growing_stack
        return open_growing_stack(src.path, WatchSettings(timeout=src.live_timeout),
                                  stop_event=stop_event)
    from ..io.tiff import open_stack
    return open_stack(src.path)


# ------------------------------------------------------------------ the forms
#
# Every fitter's form is the same parts in the same order -- the frames, the
# camera, detection, the PSF model, the fit, the output -- and only the model
# is each fitter's own.  The parts are declared once, in halves rather than in
# one base: a dataclass lays out its bases' fields first, the most basic base
# first, so a part that sits *between* two of them -- the 2D two-colour fit's
# *transform* -- needs a half on either side.  A fitter that declares `model`
# again keeps it where the first half put it.  Saved workspaces and chains
# name the fields by these paths (``fit.roisize``), so they are the contract.
@dataclass
class _Reading:
    """The first half of every fitter's form: what is read and found."""
    source: SourceSettings = field(default_factory=SourceSettings)
    # folded, its title a one-line summary (`_FitPlugin._camera_title`): the
    # camera is read far more often than it is set
    camera: CameraSettings = param(default_factory=CameraSettings, label="microscope",
                                   collapsed=True, kind="microscope")
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    # folded, its title what is drawn: most fits use the whole frame
    region: FitRegionSettings = param(default_factory=FitRegionSettings,
                                      label="fit region", collapsed=True)
    model: Any = None                    # each fitter's own part


@dataclass
class _Writing:
    """The second half: how it is fitted and where it goes."""
    fit: FitSettings = field(default_factory=FitSettings)
    output: OutputSettings = field(default_factory=OutputSettings)


@dataclass
class _Finishing:
    """What every fit does once its last frame is in (`after_fit.py`); folded,
    its title the steps that are on."""
    finish: FinishSettings = param(default_factory=FinishSettings,
                                   label="after the fit", collapsed=True)


@dataclass
class GaussianFitSettings(_Finishing, _Writing, _Reading):
    model: GaussianModelSettings = field(default_factory=GaussianModelSettings)


@dataclass
class SplineFitSettings(_Finishing, _Writing, _Reading):
    model: SplineModelSettings = field(default_factory=SplineModelSettings)


# The links two channels can share, declared once for both two-channel
# models.  Fields made by functions rather than inherited, for the reason the
# forms are in halves: each model shows its own first field (the calibration,
# the start sigma) above them, and its own fifth link where it belongs.
def _link_xy():
    return param(True, label="link x, y",
                 help="one position for both channels, through the "
                      "registration; unlink only to check the transform")


def _link_photons(more: str = ""):
    return param(False, label="link photons",
                 help="off for two colours -- the photon ratio is what tells "
                      "the dyes apart" + more)


def _link_background():
    return param(False, label="link background", advanced=True,
                 help="one background for both halves; off: each half fits "
                      "its own")


def _photon_ratio(auto: str):
    return param(None, label="photon ratio", min=0, advanced=True,
                 help="secondary / main, used when the photons are linked; "
                      "auto: " + auto)


class _LinkedChannels:
    """What the two-channel models share beyond their fields."""

    # the field that links the model's own fifth parameter: z, or the width
    FIFTH_LINK = ""

    def shared(self) -> tuple:
        """Which of x, y, photons, background and the fifth parameter both
        channels fit as one."""
        return (self.link_xy, self.link_xy, self.link_photons,
                self.link_background, getattr(self, self.FIFTH_LINK))


@dataclass
class DualModelSettings(_LinkedChannels):
    """A dual-colour calibration, and which parameters the channels share."""
    FIFTH_LINK = "link_z"
    calibration: str = param("", label="calibration", kind="open_file",
                             file_filter="Dual-colour calibrations or fitted files (*.h5 *.hdf5)",
                             help="empty: auto, the dual-colour calibration listed for this file's "
                                  "microscope and state (the menu: all found); one from PSF "
                                  "calibration in Dual colour mode; or a fitted .h5, for the one it "
                                  "was fitted with")
    link_xy: bool = _link_xy()
    link_z: bool = param(True, label="link z",
                         help="one z for both channels: this is the sqrt(2)")
    link_photons: bool = _link_photons("; on for biplane")
    link_background: bool = _link_background()
    photon_ratio: Optional[float] = _photon_ratio(
        "the split the calibration's two PSFs carry")

    def load(self):
        from ..calibrate.dual import load_dual_color_calibration
        if not self.calibration:
            raise ValueError("a two-channel fit needs a dual-colour calibration")
        return load_dual_color_calibration(self.calibration)

    def model(self, calibration, camera: Optional[CameraMetadata] = None):
        from ..io.calibration import warn_on_em_mismatch, warn_on_microscope_mismatch
        from ..psf import GlobalSplinePSF
        if camera is not None:
            warn_on_em_mismatch(calibration.main, camera.em_on)
            warn_on_microscope_mismatch(calibration.main, camera.microscope)
        # the secondary as the channel layer cuts it, flipped back along a mirror
        return GlobalSplinePSF(calibration.layer_splines(), self.shared())


@dataclass
class _SplineTransform:
    """Spline 3D 2C's registration, where the 2D form has its own (`_Transform`); the class is defined
    below, with the 2D form's: the calibration's PSFs, another registration."""
    transform: SplineTransformSettings = field(default_factory=lambda: SplineTransformSettings())


@dataclass
class DualSplineFitSettings(_Finishing, _Writing, _SplineTransform, _Reading):
    model: DualModelSettings = field(default_factory=DualModelSettings)


# --------------------------------------------------------------- the fitters
@register("Localize/Gaussian 2D")
class GaussianFit(_FitPlugin):
    description = "Detect and fit with a free-width Gaussian PSF: x, y, photons, sigma."
    Settings = GaussianFitSettings
    # 2: the camera's read noise in the likelihood (1 e- without EM gain)
    # 3: nm columns only; the pixel ones (`x_pix`, ...) are no longer written
    # 4: the dynamic cutoff's quantiles nearest-rank, as SMAP takes them
    version = "4"

    def model(self, settings, camera):
        return settings.model.model()


@register("Localize/Spline 3D")
class SplineFit(_FitPlugin):
    description = ("Detect and fit with an experimental spline PSF from a bead "
                   "calibration: adds z.")
    Settings = SplineFitSettings
    # 2: the camera's read noise in the likelihood (1 e- without EM gain)
    # 3: a SMAP calibration is x-flipped only if SMAP flipped its beads
    #    (`EMon`), no longer for any non-zero `emmirror` ("no mirror" is 2)
    # 4: nm columns only; the pixel ones (`x_pix`, ...) are no longer written
    # 5: the dynamic cutoff's quantiles nearest-rank, as SMAP takes them
    # 6: *focus* on auto by default: a pupil (Zernike) calibration is drawn for the focus in the
    #    sample, z the height above the coverslip (`calibrate.depth`)
    version = "6"
    calibration_kind = "single"

    def model(self, settings, camera):
        return settings.model.model(camera, roisize=settings.fit.roisize)

    def model_file(self, settings):
        """The pupil model beside the calibration as it stands (*auto* resolved), or None: what
        *focus* draws from.  Kept per calibration, since the form asks with every change."""
        from ..calibrate.depth import model_file
        try:
            path = self.with_calibration(settings).model.calibration
        except Exception:
            return None
        if not path or not Path(path).is_file():
            return None
        key = (str(path), Path(path).stat().st_mtime)
        if getattr(self, "_model_file_key", None) != key:
            from ..io.calibration import load_spline_calibration
            try:
                found = model_file(load_spline_calibration(path))
            except Exception:
                found = None
            self._model_file_key, self._model_file_value = key, found
        return self._model_file_value

    def describe(self, ctx: Optional[Context], settings) -> FormState:
        state = super().describe(ctx, settings)
        try:
            self._focus_form(settings, state)
        except Exception:
            pass
        return state

    def _focus_form(self, settings, state) -> None:
        """*focus* and its rows only for a calibration with a pupil model beside it: a bead average
        has no model to draw at another place.  *position* is typed only on *typed*; on *auto* and
        *in situ* it shows what the last Preview or Run found for this file, if it found it."""
        from ..calibrate import depth
        m = settings.model
        has = self.model_file(settings) is not None
        shown = has or m.focus in ("typed", "insitu")
        state.shown["model.focus"] = shown
        state.shown["model.focus_nm"] = shown and m.focus != "calibrated"
        state.shown["model.sample_index"] = shown and m.focus != "calibrated"
        state.active["model.focus_nm"] = m.focus == "typed"
        if m.focus in ("auto", "insitu"):
            found = depth.found(self._focus_key(settings))
            word = "auto" if m.focus == "auto" else "in situ"
            state.hints["model.focus_nm"] = (f"{word}: {found.focus_nm:.0f} nm, found before"
                                             if found is not None else f"{word}: found at Preview or Run")
        if shown and has and m.focus != "calibrated":
            try:
                state.hints["model.sample_index"] = (
                    f"auto: {depth.medium(depth.load(self.model_file(settings))):g}")
            except Exception:
                pass

    def _focus_key(self, settings) -> tuple:
        """What a focus found depends on: the movie and its range, the calibration, the sample's index,
        and -- for *auto*, which fits the movie's spots -- the detection, the fit box and the camera.  A
        Preview finds it once for these, and the Run after it uses it."""
        try:
            settings = self.with_calibration(settings)
        except Exception:
            pass
        m, src = settings.model, settings.source

        def stamp(path):
            try:
                return (str(path), Path(path).stat().st_mtime)
            except OSError:
                return (str(path), None)
        key = [m.focus, stamp(src.path), stamp(m.calibration), m.sample_index]
        if m.focus == "auto":
            key += [src.start, src.stop, repr(settings.detection), settings.fit.roisize,
                    repr(getattr(settings, "region", None)), repr(settings.camera)]
        return tuple(key)

    def with_focus(self, ctx: Context, settings, source, camera, finder):
        """*focus* on *auto* or *in situ* found before the fit and written in, so that the history and
        the file name what was fitted with, never "auto", as the calibration on auto is named: *auto*
        as the position, typed (`calibrate.depth.auto_focus`); *in situ* as the spline it learnt, kept
        in the configuration directory (`calibrate.depth.keep`), as calibrated.  Kept for the session
        (`calibrate.depth.remember`), so a Preview finds it once and the Run after it does not again.
        A calibration without a pupil model has nothing to draw: *auto* fits it as calibrated."""
        from ..calibrate import depth
        m = settings.model
        if m.focus not in ("auto", "insitu"):
            return settings
        from ..io.calibration import load_spline_calibration
        cal = load_spline_calibration(m.calibration)
        if depth.model_file(cal) is None:
            if m.focus == "auto":
                return replace(settings, model=replace(m, focus="calibrated"))
            raise ValueError("in situ needs a calibration made with the pupil (Zernike) method: its "
                             "aberrations are held")
        try:
            import torch  # noqa: F401 -- only asked whether it is there
        except ImportError:
            if m.focus == "auto":
                # the default must not stop a fit that worked before: said, and fitted as calibrated
                ctx.warn("focus on auto needs torch to draw the PSF in the sample (pip install "
                         "smappy-smlm[uipsf]): fitted as calibrated, z above the beads' focus")
                return replace(settings, model=replace(m, focus="calibrated"))
            raise ValueError("a focus in situ draws the PSF with uipsf, which needs torch: pip install "
                             "smappy-smlm[uipsf]")
        if settings.source.live:
            raise ValueError(f"a focus {'on auto' if m.focus == 'auto' else 'in situ'} needs the frames "
                             f"before the fit: type it for a live fit")
        found = self.found_focus(ctx, settings, source, camera, finder, cal)
        if m.focus == "auto":
            return replace(settings, model=replace(m, focus="typed", focus_nm=round(found.focus_nm, 1)))
        path = depth.kept_path(self._focus_key(settings))
        return replace(settings, model=replace(m, focus="calibrated", calibration=str(path),
                                               focus_nm=round(found.focus_nm, 1)))

    def focus_plots(self, settings, camera) -> dict:
        from ..calibrate import depth
        if settings.model.focus not in ("auto", "insitu"):
            return {}
        found = depth.found(self._focus_key(settings))
        return {} if found is None or found.heights is None else {"focus": focus_plot(found, settings)}

    def found_focus(self, ctx: Context, settings, source, camera, finder, cal):
        """The focus *auto* or *in situ* finds (`calibrate.depth.AutoDepth`), from this session's or
        an earlier one's, or found now with what it does said as it goes."""
        from ..calibrate import depth
        m = settings.model
        key = self._focus_key(settings)
        found = depth.found(key) or (depth.kept(key) if m.focus == "insitu" else None)
        if found is not None:
            ctx.report(f"focus {found.method}: {found.focus_nm:.0f} nm above the coverslip, as found before")
            return depth.remember(key, found)
        if m.focus == "auto":
            images = sample_rois(source, camera, finder, settings.fit.roisize, settings.source)
            if not len(images):
                raise ValueError("no spots found in the frames the focus is looked for in: review "
                                 "detection, or type the focus")
            found = depth.auto_focus(cal, images, m.sample_index, n_threads=settings.fit.n_threads,
                                     report=ctx.report)
            ctx.report(f"focus on auto: {found.focus_nm:.0f} nm above the coverslip, from the lowest "
                       f"of {found.n} localizations, {len(found.passes)} passes")
        else:
            # in situ starts where auto puts it: seconds, where its own estimate takes minutes
            start = None
            try:
                start = self.found_focus(ctx, replace(settings, model=replace(m, focus="auto")),
                                         source, camera, finder, cal).focus_nm
            except ValueError:
                pass
            found = depth.insitu_focus(cal, [settings.source.path], camera, m.sample_index, start,
                                       progress=ctx.report, cancel=ctx.stop)
            if depth.keep(key, found) is None:
                raise ValueError("the focus learnt in situ could not be kept for the fit: see the "
                                 "configuration directory's sample_depth folder")
            ctx.report(f"focus in situ: {found.focus_nm:.0f} nm above the coverslip, blur "
                       f"{found.blur_nm:.0f} nm, from {found.n} molecules")
        for note in found.notes:
            ctx.report(f"focus {found.method}: {note}")
        for message in found.messages:
            ctx.warn(f"focus {found.method}: {message}")
        return depth.remember(key, found)


def focus_plot(found, settings):
    """The heights the focus was found from, the coverslip at 0 and the plane in focus marked: a
    Preview's figure for *focus* on *auto* or *in situ*."""
    def draw(ax) -> None:
        h = np.asarray(found.heights, float)
        h = h[np.isfinite(h)]
        if len(h):
            low, high = np.quantile(h, [0.001, 0.999])
            ax.hist(h, bins=np.arange(min(low, -50.0), high + 25.0, 25.0), color="#4c72b0")
        ax.axvline(0.0, color="black", lw=1.5, label="coverslip")
        spline = getattr(found, "spline", None)
        record = ((spline.parameters or {}).get("sample_depth") or {}) if spline is not None else {}
        focus_height = record.get("in_focus_height_nm")
        if focus_height is None and found.method == "auto":
            from ..calibrate.depth import at_depth, in_focus
            from ..io.calibration import load_spline_calibration
            try:
                focus_height = in_focus(at_depth(load_spline_calibration(settings.model.calibration),
                                                 found.focus_nm, settings.model.sample_index))
            except Exception:
                focus_height = None
        if focus_height is not None:
            ax.axvline(focus_height, color="#c44e52", ls="--", lw=1.2, label="plane in focus")
        title = f"focus {found.method}: {found.focus_nm:.0f} nm"
        if found.blur_nm is not None:
            title += f", blur {found.blur_nm:.0f} nm"
        ax.set(xlabel="height above the coverslip (nm)", ylabel="localizations", title=title)
        ax.legend(frameon=False, fontsize=8)
    return draw


def sample_rois(source, camera, finder, roisize: int, src) -> np.ndarray:
    """The spots of `AUTO_FOCUS_BLOCKS` blocks of frames spread over the source's range, as the fit
    cuts them: photons with the read noise's variance added (`pipeline.with_readout`)."""
    from ..camera import to_photons
    from ..fit_rois import cut_rois
    from ..pipeline import with_readout
    stop = min(src.stop, source.n_frames) if src.stop else source.n_frames
    length = max(stop - src.start, 0)
    block = min(AUTO_FOCUS_BLOCK_FRAMES, length)
    if block == 0:
        return np.zeros((0, roisize, roisize), np.float32)
    starts = []
    for first in np.linspace(src.start, stop - block, AUTO_FOCUS_BLOCKS).astype(int):
        if not starts or first >= starts[-1] + block:        # a short movie is read once, not twice
            starts.append(int(first))
    parts = []
    for first in starts:
        frames = np.concatenate([b for _, b in source.frames(chunk=block, start=int(first),
                                                              stop=int(first) + block)])
        photons = to_photons(frames, camera) / camera.excess_noise
        candidates, _ = finder(photons, first_frame=int(first))
        parts.append(cut_rois(photons, candidates, roisize, first_frame=int(first)).images)
    images = np.concatenate(parts) if parts else np.zeros((0, roisize, roisize), np.float32)
    return with_readout(images, camera.readout_variance)


class _TwoChannelFit(_FitPlugin):
    """What both two-channel fitters share: one emitter, both halves.

    On the channel layer (`channels`, docs/4pi.md §9): the registration's
    halves are cut into two channel images, each searched with its own
    cutoff, the peaks paired (`multichannel.paired_candidates`); each pair
    gets a ROI in both channels and is fitted at once with the parameters the
    form links (`multichannel.MultiChannelEngine`).  The two differ
    only in where the registration comes from -- a dual-colour bead
    calibration, which brings the two PSFs with it, or a transformation alone
    -- so that is all each of them says (`read_registration`).

    The registration is read once per run or preview and handed to the
    channel layout, the model and the photon factors, which would each read
    the file otherwise.  It is kept with the path
    it came from, so that a preview with another file chosen meanwhile never
    gets a run's.
    """

    _loaded = None             # (path, registration) of the run in progress
    two_channel = True

    def registration_path(self, settings) -> str:
        raise NotImplementedError

    def read_registration(self, settings):
        raise NotImplementedError

    def measure(self, ctx: Context, settings):
        """A registration measured from the movie before the fit, or None."""
        return None

    def registration(self, settings):
        """The registration: the one this run read, or off disk."""
        loaded = self._loaded
        if loaded is not None and loaded[0] == self.registration_path(settings):
            return loaded[1]
        return self.read_registration(settings)

    def _reading_once(self, settings, call, measured=None):
        """``call()`` with the registration read once beforehand.

        A file that cannot be read is not an error here: the steps that need
        it then try themselves and fail as they did, so a preview still draws
        the candidates and says why it could not fit them.
        """
        key = self.registration_path(settings)
        if measured is not None:
            self._loaded = (key, measured)
        else:
            try:
                self._loaded = (key, self.read_registration(settings))
            except Exception:
                self._loaded = None
        try:
            return call()
        finally:
            self._loaded = None

    def run(self, ctx: Context, settings) -> Result:
        settings = self.with_calibration(settings, ctx)
        measured = self.measure(ctx, settings)
        return self._reading_once(
            settings, lambda: super(_TwoChannelFit, self).run(ctx, settings),
            measured)

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
                model=GaussianModelSettings(sigma=self._start_sigma(settings)),
                # a transformation is a statement about the camera, and is
                # fitted in chip pixels: the nm columns over the pixel size
                fit=settings.fit,
                output=OutputSettings(save=False, raw_frames=0))
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
        # in chip pixels, which the table records the size of
        from ..locs import in_pixels
        x, y = in_pixels(locs, "x_pix", "y_pix")
        result = register_channels(
            x, y, np.asarray(locs["frame"]), source.shape[-2:],
            camera.roi, settings=settings.transform.registration,
            progress=ctx.report,
            precision=(in_pixels(locs, "xy_err_pix")
                       if "xy_err_nm" in locs else None))

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

    def _start_sigma(self, settings) -> float:
        """The Gaussian width the calibration pass's plain fit starts from, px."""
        return 1.3

    # ----------------------------------------------------------- the channel layer
    def channel_layout(self, settings, base, camera: Optional[CameraMetadata] = None):
        """The two channels on this movie's frames (`calibrate.dual.ChannelTransform.layout`): the
        registration's halves on the chip, put on the movie's camera ROI, which may cut into them -- a
        calibration from one ROI applies to a movie from another, its pixels keeping their place in
        the halves (`ChannelLayout.on_camera`)."""
        import warnings
        registration = self.registration(settings)
        geometry = registration.geometry
        shape = tuple(base.shape[-2:])
        if geometry.get("coordinate_system") == "roi-local":
            if shape != tuple(geometry["image_shape"]):
                raise ValueError("ROI-local calibration requires the original bead image dimensions")
            roi = (0, 0)
        elif camera is None or camera.roi is None:
            warnings.warn("the movie has no camera ROI: a full-chip transformation is applied as if "
                          "the frames began at the chip's corner", stacklevel=2)
            roi = (0, 0)
        else:
            roi = camera.roi_offset
        return registration.layout().on_camera(roi, shape, cut_into=True)

    def open_source(self, settings, watch: bool = False, stop_event=None):
        """The movie read as its two channels (`channels.ChannelSource`): from here on nothing sees
        the halves of the chip."""
        from ..channels import ChannelSource
        base = _open(settings.source, watch=watch, stop_event=stop_event)
        camera = self._camera(settings)
        return ChannelSource.wrap(base, self.channel_layout(settings, base, camera))

    def watch_blocks(self, source, path, **options):
        return source.watch(**options)

    def view_camera(self, camera: CameraMetadata) -> CameraMetadata:
        # positions are in channel 0's own pixels, where the camera ROI is behind the channel layer
        return replace(camera, roi=None)

    def region_finder(self, settings, camera: Optional[CameraMetadata] = None):
        # each channel is searched alone; the region, drawn on channel 0, picks among the pairs
        # (`engine`), so a molecule only the other channel saw is judged where it lies
        return self.finder(settings, camera)

    def photon_factors(self, settings) -> tuple:
        """The link's photon factor per channel (`multichannel.build_link`): the splitter's photon
        ratio over the split the secondary's PSF already carries -- one, for a spline pair and no
        ratio given; for a registration alone the beads' measured ratio, or one.

        A dual bead calibration's secondary spline carries the beads' split (`psf.photon_scales`):
        multiplying by the ratio again counted it twice.  The beads' measured ratio is the fallback
        only for a registration alone, which has no PSF to carry one; it reads a few per cent high
        against the splines' (0.274 for 0.252 at a fifth of the light)."""
        registration = self.registration(settings)
        ratio = settings.model.photon_ratio
        scale = 1.0
        if hasattr(registration, "secondary"):
            from ..psf import photon_scales
            scale = float(photon_scales(registration.layer_splines())[1])
            if ratio is None:
                ratio = scale
        if ratio is None:
            ratio = registration.parameters.get("secondary_main_brightness_ratio", 1.0)
        return (1.0, float(ratio) / scale)

    def engine(self, settings, camera, finder, model, source=None):
        from ..multichannel import MultiChannelEngine
        source = source if source is not None else self.open_source(settings)
        region = getattr(settings, "region", None)
        select = None
        if region is not None and region.active():
            shapes = list(region.shapes)
            select = lambda x, y: in_region(x, y, shapes)      # noqa: E731
        return MultiChannelEngine(camera, finder, model, source.layout, settings.fit,
                                  detection="pairs", photon_factors=self.photon_factors(settings),
                                  select=select)

    def preview(self, ctx: Context, settings, frame: int = 0) -> Result:
        try:
            settings = self.with_calibration(settings)
        except ValueError:
            pass
        return self._reading_once(settings, lambda: self._layer_preview(ctx, settings, frame))

    def _layer_preview(self, ctx: Context, settings, frame: int) -> Result:
        """One frame: each channel with the molecules found in it, with its own cutoff; on channel 1
        -- where a fit region is drawn, in the pixels positions are reported in -- the second
        channel's peaks put there by the registration, the cheapest check of it there is; and the
        fitted positions in both.  Without a registration, the frame as the camera wrote it."""
        from ..camera import to_photons
        from . import ShapeEdit
        src = settings.source
        if not src.path:
            raise ValueError("choose a source file")
        try:
            source = self.open_source(settings)
        except Exception as error:
            result = super().preview(ctx, settings, frame)
            result.text = f"{result.text}; the channels not cut -- {error}"
            return result
        layout = source.layout
        index = int(np.clip(frame, 0, max(source.n_frames - 1, 0)))
        camera = self._camera(settings)
        if camera is None:
            raise ValueError(f"could not read the camera from {src.path}")
        camera.require("conversion", "offset")
        raw = source.frame(index)
        channels = to_photons(raw[None], camera) / camera.excess_noise
        finder = self.finder(settings, camera)
        found, cutoffs = [], []
        for c, (w, h) in enumerate(layout.sizes):
            candidates, filtered = finder(np.ascontiguousarray(channels[:, c, :h, :w]), first_frame=index)
            found.append(candidates)
            cutoffs.append(float(np.max(finder.cutoffs(filtered)[0])))
        projected = layout.to_reference(1, np.c_[found[1].x, found[1].y].astype(float))
        text = (f"frame {index}: " + ", ".join(f"{len(f)} in channel {c + 1} (cutoff {k:.3g})"
                                                for c, (f, k) in enumerate(zip(found, cutoffs))))
        fitted, trouble = None, ""
        try:
            model = self.model(settings, camera)
            engine = self.engine(settings, camera, finder, model, source)
            engine.push(raw[None], index)
            fitted = engine.flush()
            text += f"; {engine.stats['matched']} paired, {0 if fitted is None else len(fitted)} fitted"
        except Exception as error:
            trouble = str(error)
            text += f"; not fitted -- {error}"
        (px, py), _ = camera.pixelsize_nm_or_default
        ox, oy = layout.origins[0]
        shapes = list(settings.region.shapes)

        def draw(figure):
            panels = figure.subplots(1, 2)
            for c, ax in enumerate(panels):
                w, h = layout.sizes[c]
                ax.imshow(channels[0, c, :h, :w], cmap="gray")
                ax.set_title(f"channel {c + 1}")
                ax.plot(found[c].x, found[c].y, "o", mfc="none", color="C1", ms=8)
                if fitted is not None and len(fitted):
                    xy = np.c_[np.asarray(fitted["x_nm"]) / px - ox, np.asarray(fitted["y_nm"]) / py - oy]
                    there = xy if c == 0 else layout.to_channel(c, xy)
                    ax.plot(there[:, 0], there[:, 1], "+", color="C3", ms=8)
                ax.set_xticks([])
                ax.set_yticks([])
            panels[0].plot(projected[:, 0], projected[:, 1], "x", color="C0", ms=6)
            draw_region(panels[0], shapes)
        return Result(text=text, settings=settings,
                      plot=Plot(draw, panels=2, size=(7, 3.6),
                                edit=ShapeEdit("region.shapes", panel=0, shapes=shapes)),
                      data={"frame": index, "candidates": sum(len(f) for f in found),
                            "cutoff": tuple(cutoffs), "projected": projected,
                            "locs": fitted, "error": trouble})


@register("Localize/Spline 3D 2C")
class DualSplineFit(_TwoChannelFit):
    """The 3D two-channel workflow: one emitter, both halves of a split frame.

    Candidates are found in each channel and paired, each gets an ROI in both
    channels, and the pair is fitted at once with the parameters the form
    links -- SMAP's ``fit_global_dualchannel``.  Beyond x, y and z it
    yields ``ratio``, the fraction of the photons in the secondary channel,
    which is what separates the two dyes.
    """

    description = ("Detect and fit both halves of a split frame as one emitter, "
                   "sharing x, y and z: adds the photon ratio that tells the "
                   "two colours apart.")
    # 2: photons and ratio in photons, not in the beads' split; linked photons
    #    count that split once
    # 3: the PSFs normalised together; linked photons are the total
    # 4: a calibration's PSFs clipped at zero, not lifted: the background is
    #    the background (the version is the calibration's as much as the fit's)
    # 5: the camera's read noise in the likelihood; the calibration's photon
    #    shares from the light before clipping
    # 6: nm columns only; the pixel ones (`x_pix`, `dx_pix_ch1`, ...) are no
    #    longer written
    # 7: the dynamic cutoff's quantiles nearest-rank, as SMAP takes them
    # 8: a fit region drawn on one half applies to both
    # 9: on the channel layer (docs/4pi.md §9): positions in the main channel's own pixels from its
    #    corner on the chip, each channel searched alone, a box never across the seam
    version = "9"
    Settings = DualSplineFitSettings
    calibration_kind = "dual"

    def registration_path(self, settings) -> str:
        return f"{settings.model.calibration}|{settings.transform.path}"

    def read_registration(self, settings):
        """The dual-colour calibration: both PSFs, and its registration or the file's in place of it."""
        from ..calibrate.dual import with_registration
        calibration = settings.model.load()
        if settings.transform.path:
            return with_registration(calibration, settings.transform.load())
        return calibration

    def measure(self, ctx: Context, settings):
        """The calibration's PSFs with the registration measured from this movie, when asked: the
        splitter moves between bead calibrations, the PSFs hardly.  Measured with the calibration's
        layout and main channel, so that each PSF stays on the half it was made from
        (`dual.with_registration` refuses one that does not)."""
        if not settings.transform.calibrate:
            return None
        from ..calibrate.dual import with_registration
        calibration = settings.model.load()
        g = calibration.geometry
        given = settings.transform.registration
        fixed = replace(settings, transform=replace(settings.transform, registration=replace(
            given, layout=g.get("layout", given.layout), main_channel=g.get("main_channel",
                                                                           given.main_channel))))
        return with_registration(calibration, self._calibrate(ctx, fixed))

    def model(self, settings, camera):
        return settings.model.model(self.registration(settings), camera)


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
    from ..locs import in_pixels
    column = "x_pix" if axis == 1 else "y_pix"
    low = high = 0
    for part in parts:
        try:
            values = in_pixels(part, column)
        except (KeyError, ValueError):
            return sum(len(p) for p in parts)
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
    calibrate_frames: int = param(5000, label="at most", unit="frames", min=50,
                                  help="the most frames the calibration pass "
                                       "fits, however few localizations they "
                                       "give")
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
class DualGaussianModelSettings(_LinkedChannels):
    """A free-width Gaussian per channel, and which parameters they share."""
    FIFTH_LINK = "link_sigma"
    sigma: float = param(1.2, label="start sigma", unit="px", min=0.1,
                         help="the PSF width both halves' fits start from; "
                              "each is fitted")
    link_xy: bool = _link_xy()
    link_photons: bool = _link_photons()
    link_background: bool = _link_background()
    link_sigma: bool = param(False, label="link width", advanced=True,
                             help="off: each half finds its own width, which it "
                                  "should -- they see different wavelengths and "
                                  "rarely share a focus")
    photon_ratio: Optional[float] = _photon_ratio(
        "the beads' ratio if the transformation is a bead calibration, "
        "otherwise 1")

    def model(self, channels: int = 2):
        from ..psf import GlobalGaussianPSF
        return GlobalGaussianPSF(sigma=self.sigma, shared=self.shared(),
                                 channels=channels)


@dataclass
class SplineTransformSettings(ChannelTransformSettings):
    """Spline 3D 2C's registration: the calibration's own, a file's, or one measured from the movie;
    the PSFs are always the calibration's."""
    path: str = param("", label="transformation", kind="open_file", file_filter=TRANSFORM_FILTER,
                      help="empty: the calibration's own; a _2ct.h5 to use instead, with the "
                           "calibration's PSFs")
    calibrate: bool = param(False, label="measure from this movie",
                            help="keep the calibration's PSFs and measure where the second channel is "
                                 "from this movie: for a splitter moved since the beads")


@dataclass
class _Transform:
    """The part only the 2D two-colour form has, between its two halves."""
    transform: ChannelTransformSettings = field(default_factory=ChannelTransformSettings)


@dataclass
class DualGaussianFitSettings(_Finishing, _Writing, _Transform, _Reading):
    model: DualGaussianModelSettings = field(default_factory=DualGaussianModelSettings)


@register("Localize/Gaussian 2D 2C")
class DualGaussianFit(_TwoChannelFit):
    """The 2D two-channel workflow: one emitter, both halves, no PSF model.

    `DualSplineFit` without the calibration.  Candidates are found in each
    channel and paired, each gets a ROI in both channels, and the pair is
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
    # 2: linked photons are the total over both halves, not the main half's
    # 3: the camera's read noise in the likelihood (1 e- without EM gain)
    # 4: nm columns only; the pixel ones (`x_pix`, `sigma_pix_ch0`, ...) are
    #    no longer written
    # 5: the dynamic cutoff's quantiles nearest-rank, as SMAP takes them
    # 6: a fit region drawn on one half applies to both
    # 7: on the channel layer, as Spline 3D 2C's 9
    version = "7"
    Settings = DualGaussianFitSettings

    def model(self, settings, camera):
        return settings.model.model()

    def registration_path(self, settings) -> str:
        return settings.transform.path

    def read_registration(self, settings):
        """A transformation off disk, or the one in a dual-colour calibration."""
        return settings.transform.load()

    def measure(self, ctx: Context, settings):
        """The transformation measured from this movie, when that is asked for.

        Handed to the run rather than saved and read back: writing a file
        only to read it would make saving it a precondition of fitting rather
        than a convenience.
        """
        return self._calibrate(ctx, settings) if settings.transform.calibrate else None

    def _start_sigma(self, settings) -> float:
        return float(settings.model.sigma)

    def preflight(self, ctx: Context, settings):
        """No plain "start anyway": without a transformation the fit cannot
        pair the channels and fails at once, so the one useful answer besides
        cancelling (and choosing a file) is to measure it from the movie.
        A camera to add comes first (`_FitPlugin.preflight`)."""
        camera = super().preflight(ctx, settings)
        if camera is not None:
            return camera
        if settings.transform.calibrate or settings.transform.path:
            return None
        from dataclasses import replace as _replace
        calibrated = _replace(settings, transform=_replace(settings.transform,
                                                           calibrate=True))
        return PreflightQuestion(
            "No transformation chosen: the two channels cannot be paired "
            "without one.  Measure it from the first frames of this movie, or "
            "cancel and choose a file.",
            choices=(PreflightChoice("Calibrate from this movie", calibrated),),
            run_label="")
