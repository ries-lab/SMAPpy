"""*Overlay channels*: a split camera's two halves put on top of each other, one dye in each.

The thin plugin around `smappy.channel_overlay`, whose docstring has the
method.  For two dyes that are each seen in one half of the chip only -- no
ratio to measure, nothing to pair -- fitted as *two channels* (Gaussian 2D or
Spline 3D, *channels* > *read*: *two channels*).  The secondary half's
localizations are moved onto the main half by a registration measured on
beads, which every half sees, and ``channel`` (the dye) is written from
``camera_channel``.

Where the registration comes from is one choice, so that the plugin serves
the first sample of a day and the fiftieth alike:

* a *transformation file* -- a ``_2ct.h5`` of *Calibrate transform*, of this
  plugin, or a dual-colour bead calibration's registration;
* *fitted beads* -- a bead localization file fitted as two channels,
  registered here as *Calibrate transform* does
  (`registration.register_table`);
* *bead movies* -- a bead movie or a folder of them, fitted here with
  Gaussian 2D on two channels, in the arrangement and with the camera the
  sample was read with, then registered.

The last two keep what they measured beside the beads (``<name>_2ct.h5``)
with the files and settings it came from, and use it again while those have
not changed (`channel_overlay.cached_transform`): a batch, or the *after the
fit* step of every fit of a day, fits and registers the beads once.

There is no correction of the bead map from the sample: one from crosstalk pairs
was tried and taken out (`channel_overlay`'s docstring says why).  A shift left
between the dyes is measured by *Register layers* on the two overlaid channels.

The step of every fitter's *after the fit* (`after_fit.py`) is this plugin,
run before drift correction: the drift is the sample's, the same in both
halves, and is best measured on the overlaid table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

from ..calibrate.dual import ArrangementSettings, RegisterSettings
from . import Context, FormState, Notice, Plugin, Result, param, register

SOURCES = (("file", "transformation file"), ("table", "fitted beads"),
           ("movies", "bead movies"))
TRANSFORM_FILTER = ("Channel transformation (*_2ct.h5 *.h5 *.hdf5);;All files (*)")
LOCS_FILTER = "Localizations (*.hdf5 *.h5 *.csv)"
NEEDS = ("needs a table fitted as two channels (camera_channel): fit with channels > read: "
         "two channels")


@dataclass
class BeadFitSettings:
    """How the bead movies are fitted: Gaussian 2D, two channels, the sample's camera."""
    cutoff: float = param(1.7, label="cutoff", min=0.1,
                          help="the detection threshold, in units of the filtered "
                               "frame's noise (the fitters' dynamic cutoff)")
    max_frames: Optional[int] = param(None, label="at most", unit="frames", min=1,
                                      help="frames fitted per bead movie; auto: all")


@dataclass
class OverlaySettings:
    source: str = param("file", label="transformation from", choices=SOURCES,
                        help="a saved registration, beads already fitted as two "
                             "channels, or bead movies to fit")
    path: str = param("", label="file", kind="open_file", file_filter=TRANSFORM_FILTER,
                      help="a _2ct.h5 (Calibrate transform, or this plugin's), or a "
                           "dual-colour bead calibration")
    beads_table: str = param("", label="fitted beads", kind="open_file",
                             file_filter=LOCS_FILTER,
                             help="bead localizations fitted with channels > read: "
                                  "two channels")
    beads: str = param("", label="bead movies", kind="dir",
                       help="a folder of bead movies, or one movie's path; "
                            "beads seen in both halves")
    bead_fit: BeadFitSettings = param(default_factory=BeadFitSettings, label="bead fit",
                                      collapsed=True)
    # where the halves lie, for the beads: the fitter's own when this is its step
    channels: ArrangementSettings = param(default_factory=ArrangementSettings,
                                          label="channels", collapsed=True)
    registration: RegisterSettings = param(default_factory=RegisterSettings,
                                           collapsed=True)
    save: bool = param(True, label="keep the registration",
                       help="write it beside the beads as <name>_2ct.h5 and use it again "
                            "while the beads and these settings are unchanged")
    # the camera the sample was read with, which the bead movies are read with too: the
    # fitter's, filled when this is its step (`fit._FitPlugin.finish`); empty: the table's record
    camera: dict = param(default_factory=dict, hidden=True,
                         help="CameraSettings values for the bead fit")


def shown(settings: OverlaySettings) -> dict:
    """The fields of the chosen source only."""
    beads = settings.source in ("table", "movies")
    return {"path": settings.source == "file", "beads_table": settings.source == "table",
            "beads": settings.source == "movies", "bead_fit": settings.source == "movies",
            "channels": beads, "registration": beads, "save": beads}


def with_fitter(settings: OverlaySettings, fitter_settings) -> OverlaySettings:
    """``settings`` with the fitter's camera and arrangement, which the beads are fitted in:
    what a fitter's *after the fit* hands its step."""
    camera = getattr(fitter_settings, "camera", None)
    channels = getattr(fitter_settings, "channels", None)
    out = settings
    if camera is not None and not settings.camera:
        values = {k: v for k, v in asdict(camera).items() if v not in (None, "")}
        out = replace(out, camera=values)
    # a split camera's *channels* part (`fit.ChannelsSettings`); not Spline 4Pi's four
    if all(hasattr(channels, name) for name in ("layout", "main_channel", "split_position")):
        out = replace(out, channels=ArrangementSettings(
            layout=channels.layout, main_channel=channels.main_channel,
            split_position=channels.split_position))
    return out


def registration_for(locs, settings: OverlaySettings, report=None, stop=None):
    """``(transform, how, measured, path)``: the registration ``settings`` name, a line saying
    where it came from, the `RegistrationResult` when it was measured now (else None), and the
    file it is kept in (None when it is kept nowhere)."""
    from ..calibrate.transform import load_transform
    from .. import channel_overlay as co
    if settings.source == "file":
        if not settings.path:
            raise ValueError("choose a transformation file (_2ct.h5)")
        return (load_transform(settings.path), f"from {Path(settings.path).name}", None,
                Path(settings.path))
    if settings.source == "table":
        if not settings.beads_table:
            raise ValueError("choose the fitted beads: a localization file fitted as two "
                             "channels")
        files = [Path(settings.beads_table)]
        key = co.bead_key(files, channels=settings.channels,
                          registration=settings.registration)
        where = co.transform_path(files[0])
    elif settings.source == "movies":
        if not settings.beads:
            raise ValueError("choose the bead movies: a folder of them, or one")
        files = co.bead_files(settings.beads)
        camera = co.camera_of(locs, settings.camera)
        key = co.bead_key(files, channels=settings.channels,
                          registration=settings.registration, fit=settings.bead_fit,
                          camera={k: v for k, v in asdict(camera).items() if v is not None})
        where = co.transform_path(settings.beads)
    else:
        raise ValueError(f"no source {settings.source!r}")
    if settings.save:
        cached = co.cached_transform(where, key)
        if cached is not None:
            return cached, f"from {where.name}, measured on these beads before", None, where
    if settings.source == "table":
        from ..io.formats import load
        beads, _ = load(files[0])
    else:
        beads = co.fit_bead_movies(files, camera, settings.channels,
                                   settings.bead_fit.cutoff, settings.bead_fit.max_frames,
                                   report, stop)
    from .registration import register_table
    measured = register_table(beads, settings.channels, settings.registration, report)
    measured.transform.parameters["beads"] = co._plain(key)
    how = (f"measured on {len(files)} bead "
           f"{'file' if settings.source == 'table' else 'movie'}{'s' * (len(files) > 1)}")
    if settings.save:
        measured.save(where, overwrite=True)
        how += f", kept as {where.name}"
    return measured.transform, how, measured, where if settings.save else None


@register("Analysis/After fit/Overlay channels")
class OverlayChannels(Plugin):
    """Put a split camera's secondary half onto its main half, from beads."""

    description = ("Move the localizations of a split camera's second half onto the first, by "
                   "a registration from a file, fitted beads or bead movies; channel is the half.")
    Settings = OverlaySettings
    version = "2"
    changes_locs = True
    # every localization moves, whatever the layer shows, and pairs are within a frame
    grouping = "ungrouped"

    def describe(self, ctx: Optional[Context], settings: OverlaySettings) -> FormState:
        state = FormState(shown=shown(settings))
        locs = getattr(ctx, "locs", None) if ctx is not None else None
        if locs is not None and len(locs) and "camera_channel" not in locs:
            state.notice = Notice(NEEDS)
        from ..calibrate.dual import main_channel_choices
        scope = (locs.metadata.get("camera") or {}).get("microscope") if locs is not None else None
        state.choices["channels.main_channel"] = main_channel_choices(settings.channels, scope)
        return state

    def run(self, ctx: Context, settings: OverlaySettings) -> Result:
        from .. import channel_overlay as co
        from .registration import registration_plot
        locs, _ = ctx.ungrouped()
        co.secondary_rows(locs)                      # refused here, before beads are fitted
        if co.is_overlaid(locs):
            raise ValueError("the secondary channel has already been put onto the main one")
        transform, how, measured, path = registration_for(locs, settings, ctx.report, ctx.stop)
        problem = co.check_layout(locs, transform)
        if problem:
            ctx.warn(problem)
        lines = [f"registration {how}"]
        plots = {}
        if measured is not None:
            p = measured.transform.parameters
            lines.append(f"{p['n_inliers']:,} bead pairs, residual dx "
                         f"{p['residual_dx_px']:.3f} px, dy {p['residual_dy_px']:.3f} px")
            plots["beads"] = registration_plot(measured)
        out = co.overlay_table(locs, transform, path=path)
        n = int(co.secondary_rows(locs).sum())
        lines.insert(0, f"{n:,} of {len(locs):,} localizations moved onto the main channel")
        return Result(locs=out, text="\n".join(lines), plots=plots, settings=settings,
                      data={"transform": transform, "measured": measured})

