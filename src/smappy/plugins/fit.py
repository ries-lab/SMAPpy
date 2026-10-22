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
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..detect import AbsoluteCutoff, DoGFilter, DynamicCutoff, GaussFilter, PeakFinder
from ..drift import DriftSettings
from ..locs import Localizations
from ..metadata import DEFAULT_PIXELSIZE_NM, CameraMetadata
from ..calibrate.transform import RegisterSettings
from ..pipeline import FitSettings, fit_stack, provenance
from ..psf import GaussianPSF, SplinePSF
from ..rcc import RCCSettings
from . import (Context, FormState, Notice, ParamInfo, Plot, Plugin, PreflightChoice,
               PreflightQuestion, Result, ShapeEdit, param, register)
from .assign_colors import AssignColorSettings

# a simulation's recipe opens as an acquisition too (`smappy.simulate.source`)
TIFF_FILTER = ("Image stacks (*.tif *.tiff *.ome.tif *.sim.yaml);;"
               "Simulations (*.sim.yaml);;All files (*)")


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
    """ADU -> photons and pixel -> nm.  Empty fields come from the file."""
    # the camera is normally recognised from the file's own tags; this is for
    # the file whose camera carries no tag to recognise it by
    camera: str = param("", label="camera", choices=lambda: _camera_choices(),
                        help="auto: identified from the file's metadata")
    # recognised from the file's devices as the camera is; chosen for a file
    # no entry recognises.  Its pixel for the 3D lens's state beats the file's
    microscope: str = param("", label="microscope", choices=lambda: _microscope_choices(),
                            advanced=True,
                            help="auto: identified from the file's metadata; it sets the pixel "
                                 "per state of its 3D lens")
    conversion: Optional[float] = param(None, unit="e-/ADU", min=0,
                                        help="photoelectrons per count; auto: "
                                             "from the file or the camera "
                                             "database")
    offset: Optional[float] = param(None, unit="ADU",
                                    help="the count of a pixel that saw no light")
    pixelsize_um: Optional[float] = param(None, label="pixel size", unit="µm", min=0,
                                          help="the pixel's size in the sample, "
                                               "after the magnification")
    pixelsize_y_um: Optional[float] = param(None, label="pixel size y", unit="µm",
                                            min=0, advanced=True,
                                            help="auto: square pixels, the same "
                                                 "as in x")
    em_on: Optional[bool] = param(None, label="EM gain on",
                                  help="an EMCCD with its electron multiplication "
                                       "on: the counts are divided by the gain, "
                                       "and the noise is doubled")
    emgain: Optional[float] = param(None, label="EM gain", min=0,
                                    help="the EM gain the camera was set to")
    read_noise_e: Optional[float] = param(None, label="read noise", unit="e-",
                                          min=0, advanced=True,
                                          help="rms per pixel; auto: 1 without "
                                               "EM gain (an sCMOS), 0 with it")
    frame_time_ms: Optional[float] = param(None, label="frame time", unit="ms",
                                           min=0, advanced=True,
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
        from ..io.tiff import resolve_camera
        return resolve_camera(source, camera=self.camera, overrides=self.overrides(),
                              microscope=self.microscope)

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


def draw_region(ax, shapes, counterparts=()) -> None:
    """The region's shapes as outlines on a frame: green the parts fitted
    in, dashed red the parts left out; their counterparts in the other half
    of a split frame (`dualfit.region_counterparts`) thin and dotted."""
    for drawn, mirrored in ((shapes, False), (counterparts, True)):
        for shape in drawn or ():
            points = np.asarray(shape.get("points") or [], float)
            if len(points) < 3:
                continue
            closed = np.vstack([points, points[:1]])
            style = dict(REGION_STYLE.get(shape.get("kind"), REGION_STYLE["include"]))
            if mirrored:
                style["ls"] = ":"
            ax.plot(closed[:, 0], closed[:, 1], lw=1.0 if mirrored else 1.5, **style)


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
                             file_filter="Bead calibrations (*.h5 *.hdf5 *_3dcal.mat *.mat)",
                             help="the bead calibration of this microscope: an "
                                  ".h5 from the Localize tab's Bead calibration, or a "
                                  "SMAP _3dcal.mat")
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
        return SplinePSF(calibration, z_start_nm=float(self.z_start_nm))


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
                           help="image tags drawn when the fit is done, "
                                "names or parts of names; empty: none "
                                "(Analysis/Process/Image Tags)")

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
                                     "(Analysis/Dual-colour/Assign colours)")
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

    def engine(self, settings, camera: CameraMetadata, finder, model):
        """What consumes the frames.  One ROI per candidate by default; the
        two-channel fitter overrides this with one pair per candidate."""
        from ..pipeline import LocalizationEngine
        return LocalizationEngine(camera, finder, model, settings.fit)

    def finder(self, settings, camera: Optional[CameraMetadata] = None):
        """What finds the candidates.  A two-channel fitter overrides this to
        threshold each half of the split frame separately."""
        return settings.detection.finder(settings.fit.n_threads)

    def region_finder(self, settings, camera: Optional[CameraMetadata] = None):
        """`finder`, keeping only the candidates in the drawn region
        (`FitRegionSettings`) when there is one."""
        finder = self.finder(settings, camera)
        region = getattr(settings, "region", None)
        if region is not None and region.active():
            return RegionFinder(finder, list(region.shapes)
                                + self.region_counterparts(settings, camera,
                                                           region.shapes))
        return finder

    def region_counterparts(self, settings, camera, shapes) -> list:
        """The region's shapes as they fall on the frame's other half, for a
        split frame whose halves are one sample; none for one channel."""
        return []

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
            state.titles = {"camera": self._camera_title(settings)}
        except Exception:
            pass
        region = getattr(settings, "region", None)
        if region is not None:
            state.titles["region"] = region.summary()
            state.active["region.use"] = bool(region.shapes)
        if getattr(settings, "finish", None) is not None:
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
        """The camera part's title while it is folded: which camera, and the
        numbers that turn counts into photons and pixels into nm."""
        resolved = self.resolution(settings)
        camera = (resolved.camera_metadata() if resolved is not None
                  else settings.camera.resolve(None, require=False))
        bits = []
        name = ((resolved.camera_name or resolved.values.get("camera_name"))
                if resolved is not None else "")
        if name:
            bits.append(str(name))
        if camera.conversion is not None:
            bits.append(f"{camera.conversion:.3g} e-/ADU")
        sizes = camera.pixelsize_um_xy
        if sizes is not None:
            bits.append(f"{1000 * sizes[0]:.4g} nm" if sizes[0] == sizes[1]
                        else f"{1000 * sizes[0]:.4g}\u00d7{1000 * sizes[1]:.4g} nm")
        if camera.microscope:
            from ..microscope_db import one_line
            bits.append(one_line(camera.microscope, short=True))
        head = f"camera: {settings.camera.camera or 'auto'}"
        return head + (" \u2013 " + ", ".join(bits) if bits else "")

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
            # The camera is *not* copied in: a field on auto shows the file's
            # value as a grey hint (`describe`) and stays auto, so the run reads
            # the file it is given.  Copying turned the file's numbers into the
            # user's -- they then overrode a later *camera* choice, and the
            # next file inherited the last one's values.
            return updates or None
        return None

    def finish(self, ctx: Context, settings, locs: Localizations) -> Finished:
        """What is run over the finished table before it is saved.

        Nothing, unless the settings carry a `finish` part -- which is what a
        two-channel fit adds, and what a single-channel one has no use for.
        """
        wanted = getattr(settings, "finish", None)
        if wanted is None:
            return Finished(locs)
        # the drift steps merge blinks as the session does, as they would
        # when run from the Analysis tab (`drift.merge_from_session`)
        from ..drift import merge_from_session
        wanted = replace(wanted, rcc=merge_from_session(wanted.rcc, ctx.session),
                         comet=merge_from_session(wanted.comet, ctx.session))
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
        finder = self.region_finder(settings, camera)

        raw = source.frame(index)
        photons = to_photons(raw, camera)/camera.excess_noise
        candidates, filtered = finder(photons[None], first_frame=index)
        # what detection applied to this frame, from the finder's own code:
        # one number, or one per half of a split frame
        cutoffs = tuple(float(c) for c in finder.cutoffs(filtered)[0])
        split = getattr(finder, "split", None)
        halves = (None if len(cutoffs) == 1 else
                  ("top", "bottom") if split[0] == 0 else ("left", "right"))
        said = (f"{cutoffs[0]:.3g}" if halves is None else
                ", ".join(f"{c:.3g} {h}" for c, h in zip(cutoffs, halves)))
        top_cutoff = max(cutoffs)

        try:
            projected = self.back_projected(settings, camera, candidates)
        except Exception:
            projected = None      # no calibration: the fit will say so below

        # the region as drawn, and where it falls on the other half of a split
        # frame: read now, while this preview's registration is at hand
        region = getattr(settings, "region", None)
        shapes = list(getattr(region, "shapes", []) or [])
        counterparts = (self.region_counterparts(settings, camera, shapes)
                        if region is not None and region.active() else [])

        locs, trouble = None, ""
        try:
            model = self._model_telling(ctx, settings, camera)
            engine = self.engine(settings, camera, finder, model)
            engine.push(raw[None], first_frame=index)
            locs = engine.flush()
        except Exception as error:
            trouble = f"{type(error).__name__}: {error}"

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
            draw_region(peaks, shapes, counterparts)

            # scaled to the cutoff, not to the brightest pixel: the question
            # this panel answers is how far the peaks sit above the threshold,
            # and a single bright molecule would otherwise flatten the rest
            top = (2*top_cutoff if np.isfinite(top_cutoff) and top_cutoff > 0
                   else None)
            image = filtered_ax.imshow(filtered[0], cmap="viridis", vmin=0, vmax=top)
            bar = figure.colorbar(image, ax=filtered_ax, fraction=0.046)
            if top is not None:
                # a split frame: the first half's line solid, the second's dashed
                for value, style in zip(cutoffs, ("-", "--")):
                    bar.ax.axhline(value, color="red", lw=1.5, ls=style)
            filtered_ax.plot(candidates.x, candidates.y, ".", color="red", ms=3)
            if halves is None:
                line = f"red line at the cutoff, {said}"
            else:
                line = (f"cutoffs {cutoffs[0]:.3g} {halves[0]} (red), "
                        f"{cutoffs[1]:.3g} {halves[1]} (dashed)")
            filtered_ax.set(title=f"{settings.detection.filter} filtered; {line}")

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
        return Result(text=text, settings=settings,
                      plot=Plot(draw=plot, panels=3, size=(15, 6), edit=edit),
                      data={"frame": index, "candidates": len(candidates),
                            "cutoff": cutoffs[0] if halves is None else cutoffs,
                            "halves": halves, "locs": locs, "error": trouble})

    def run(self, ctx: Context, settings) -> Result:
        from ..io.hdf5 import LocalizationWriter
        from ..live import camera_extent

        src = settings.source
        if not src.path:
            raise ValueError("choose a source file")
        source = _open(src, watch=src.live, stop_event=ctx.stop)
        camera = settings.camera.resolve(source)
        fit = settings.fit
        finder = self.region_finder(settings, camera)
        model = self._model_telling(ctx, settings, camera)
        out = settings.output.resolve(src.path)

        # what the microscope recorded per frame, read with the pixels
        from ..frametags import FrameTags, acquisition
        tags = FrameTags()
        behind = [0]              # frames written and not yet read: the lag
        if src.live:
            from ..io.watch import WatchSettings, watch_stack
            # given `ctx.writer_finished`, the writer's word ends the watch
            # and *stop after* is never consulted (`smappy.io.watch`)
            def on_behind(frames: int) -> None:
                behind[0] = frames
            blocks = watch_stack(src.path, chunk=src.chunk, start=src.start, stop=src.stop,
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
        ctx.emit("start", {"extent": camera_extent(camera, source.shape),
                           "path": out})
        record = provenance(camera, finder, model, fit, source=src.path)
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

        try:
            from ..pipeline import drive
            engine = self.engine(settings, camera, finder, model)
            drive(engine, blocks, sink=sink, progress=report, read_ahead=2,
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
        raw = keeper.image(camera, source=str(src.path),
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
    camera: CameraSettings = param(default_factory=CameraSettings, label="camera",
                                   collapsed=True)
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
    """What a two-channel fit does once its last frame is in."""
    finish: FinishSettings = param(default_factory=FinishSettings,
                                   label="after the fit")


@dataclass
class GaussianFitSettings(_Writing, _Reading):
    model: GaussianModelSettings = field(default_factory=GaussianModelSettings)


@dataclass
class SplineFitSettings(_Writing, _Reading):
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
                             file_filter="Dual-colour calibration (*.h5 *.hdf5)",
                             help="from Bead calibration in *Dual colour* mode")
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
        return GlobalSplinePSF((calibration.main, calibration.secondary),
                               self.shared())


@dataclass
class DualSplineFitSettings(_Finishing, _Writing, _Reading):
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
    version = "5"

    def model(self, settings, camera):
        return settings.model.model(camera, roisize=settings.fit.roisize)


class _TwoChannelFit(_FitPlugin):
    """What both two-channel fitters share: one emitter, both halves.

    Candidates are found over the whole frame, each half thresholded on its
    own, and combined; each gets an ROI in both halves, and the pair is fitted
    at once with the parameters the form links (`dualfit`).  The two differ
    only in where the registration comes from -- a dual-colour bead
    calibration, which brings the two PSFs with it, or a transformation alone
    -- so that is all each of them says (`read_registration`).

    The registration is read once per run or preview and handed to the
    finder, the model, the engine and the back-projection, which each read the
    file before: three times a run, four a preview.  It is kept with the path
    it came from, so that a preview with another file chosen meanwhile never
    gets a run's.
    """

    _loaded = None             # (path, registration) of the run in progress

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
        measured = self.measure(ctx, settings)
        return self._reading_once(
            settings, lambda: super(_TwoChannelFit, self).run(ctx, settings),
            measured)

    def preview(self, ctx: Context, settings, frame: int = 0) -> Result:
        return self._reading_once(
            settings, lambda: super(_TwoChannelFit, self).preview(ctx, settings,
                                                                  frame))

    def region_counterparts(self, settings, camera, shapes) -> list:
        """A shape drawn on one half applies to both (`dualfit.region_counterparts`).
        Without a registration yet -- a preview before one is chosen -- the
        shapes act on the half they are drawn on."""
        from ..dualfit import region_counterparts
        try:
            registration = self.registration(settings)
        except Exception:
            return []
        origin = camera.roi_offset if camera is not None else (0, 0)
        return region_counterparts(shapes, registration, origin)

    def back_projected(self, settings, camera, candidates):
        from ..dualfit import secondary_to_reference, which_channel
        registration = self.registration(settings)
        ox, oy = camera.roi_offset
        secondary = ~which_channel(candidates.x + ox, candidates.y + oy,
                                   registration.geometry)
        mapped = secondary_to_reference(candidates.x[secondary],
                                        candidates.y[secondary],
                                        registration, (ox, oy))
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
            geometry = self.registration(settings).geometry
        except Exception:
            return base                          # none yet: preview only
        return _replace(base, split=self._split_for_detection(geometry, camera))

    def engine(self, settings, camera, finder, model):
        from ..dualfit import DualChannelEngine
        return DualChannelEngine(camera, finder, model, self.registration(settings),
                                 settings.fit, settings.model.photon_ratio)


@register("Localize/Spline 3D 2C")
class DualSplineFit(_TwoChannelFit):
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
    version = "8"
    Settings = DualSplineFitSettings

    def registration_path(self, settings) -> str:
        return settings.model.calibration

    def read_registration(self, settings):
        """The dual-colour calibration: the registration and both PSFs."""
        return settings.model.load()

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
class _Transform:
    """The part only the 2D two-colour form has, between its two halves."""
    transform: ChannelTransformSettings = field(default_factory=ChannelTransformSettings)


@dataclass
class DualGaussianFitSettings(_Finishing, _Writing, _Transform, _Reading):
    model: DualGaussianModelSettings = field(default_factory=DualGaussianModelSettings)


@register("Localize/Gaussian 2D 2C")
class DualGaussianFit(_TwoChannelFit):
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
    # 2: linked photons are the total over both halves, not the main half's
    # 3: the camera's read noise in the likelihood (1 e- without EM gain)
    # 4: nm columns only; the pixel ones (`x_pix`, `sigma_pix_ch0`, ...) are
    #    no longer written
    # 5: the dynamic cutoff's quantiles nearest-rank, as SMAP takes them
    # 6: a fit region drawn on one half applies to both
    version = "6"
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
