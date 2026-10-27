"""*Spline 4Pi*: the fitter for 4Pi data, the IAB model across the phase channels.

A thin fitter on the shared `_FitPlugin` (the source, the camera, detection,
streaming, saving, *after the fit*), with its own model and engine:

* the model is `psf.Global4PiPSF` over a 4Pi model file (`smappy.fourpi`:
  this program's, or uiPSF v1's ``psfmodel_*_4pi.h5``);
* the engine is `multichannel.MultiChannelEngine`, which takes the frame
  apart into its channels by a `channels.ChannelLayout`, finds the molecules
  in the channels' sum and fits each in all of them at once.

The layout is where the channels are on the frame: as the file lays them out
(the quadrant files of `smappy.io.quadrants`, side by side), typed as SMAP's
4Pi settings give them (each corner, one size, which are mirrored in x), or
the 4Pi-SMS design of the Bewersdorf lab (`channels.YALE_4PI`).  The maps
between the channels come with the model -- uiPSF learns them with the PSF,
and the two belong together: a channel's PSF is centred where its map puts
the molecule.

*After the fit* runs *4Pi z* by default: the fit's z is the envelope's, and
the phase becomes z there (`plugins/fourpi_z.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import Context, FormState, Notice, Plot, Result, param, register
from .fit import (_FitPlugin, _Finishing, _Reading, _Writing, CameraMetadata)

MODEL_FILTER = "4Pi models (*.h5)"
REGISTRATION_FILTER = "Channel registrations (*.json)"
LAYOUTS = (("auto", "the file's, else the microscope's"), ("microscope", "the microscope's"),
           ("file", "as the file lays them out"), ("typed", "typed corners"),
           ("yale", "4Pi-SMS (Bewersdorf lab)"))
REGISTRATIONS = (("model", "the model's"), ("beads", "from beads"),
                 ("file", "from a registration file"), ("none", "none"))


@dataclass
class ChannelLayoutSettings:
    layout: str = param("auto", label="channels", choices=LAYOUTS,
                        help="where the phase channels are: the file's own layout, the "
                             "microscope's (the database), or typed")
    x_corners: str = param("0, 200, 400, 600", label="x corners", unit="px",
                           help="each channel's left edge on the chip, comma-separated")
    y_corners: str = param("0, 0, 0, 0", label="y corners", unit="px",
                           help="each channel's top edge on the chip")
    width: int = param(200, label="width", unit="px", min=8,
                       help="one channel's width")
    height: int = param(200, label="height", unit="px", min=8, beside=True,
                        help="one channel's height")
    mirrored: str = param("", label="mirrored in x",
                          help="the channels mirrored left-right, by number from 1, "
                               "comma-separated; empty: none")
    registration: str = param("model", label="registration", choices=REGISTRATIONS,
                              help="how the channels map onto the first: the maps kept "
                                   "with the model, measured on beads, a saved registration, "
                                   "or none")
    beads: str = param("", label="beads", kind="open_file",
                       help="a bead acquisition in this layout (a z stack), or a folder "
                            "of them; measured once, kept beside them")
    path: str = param("", label="registration file", kind="open_file",
                      file_filter=REGISTRATION_FILTER,
                      help="a channel registration (.json), as beads leave beside them")


@dataclass
class FourPiModelSettings:
    calibration: str = param("", label="4Pi model", kind="open_file",
                             file_filter=MODEL_FILTER,
                             help="a 4Pi model (.h5): this program's or uiPSF v1's")
    channels: ChannelLayoutSettings = param(default_factory=ChannelLayoutSettings,
                                            label="channels")
    link_xy: bool = param(True, label="link x, y",
                          help="one position for every channel; off measures each "
                               "channel's own, to check the registration")
    starts: int = param(2, label="z starts", min=1, max=6, advanced=True,
                        help="how many z planes of the phasor start are fitted from")
    phase_starts: int = param(1, label="phase starts", min=1, max=4, advanced=True,
                              help="from each z start, how many phases around the circle; 2 catches the "
                                   "few fits that stop in a worse optimum, at half the speed")
    channel_drift: bool = param(False, label="channel drift",
                                help="measure how the channels move against each other, "
                                     "from fits with x and y free per channel, and follow it")
    drift_blocks: int = param(10, label="drift blocks", min=1, advanced=True,
                              help="how many blocks of frames, spread over the acquisition, "
                                   "the drift is measured in")
    drift_frames: int = param(500, label="frames per block", unit="frames", min=10,
                              advanced=True, help="how many frames each block fits")


@dataclass
class FourPiFitSettings(_Finishing, _Writing, _Reading):
    model: FourPiModelSettings = field(default_factory=FourPiModelSettings)


def _numbers(text: str) -> list:
    return [int(round(float(w))) for w in str(text).replace(";", ",").split(",") if w.strip()]


_REGISTERED = {}       # (bead paths, layout) -> Registration: beads are read once a session


def registration_from_beads(path: str, layout):
    """The beads' registration for ``layout`` (`channels.register_channels`).

    Measured once: kept beside the beads (`channels.registration_path`) and
    read from there while it is newer than every bead file and was measured
    on the same rectangles.
    """
    from pathlib import Path
    from ..channels import (ChannelLayout, bead_projections, load_registration,
                            register_channels, registration_path, save_registration)
    if not path:
        raise ValueError("registration from beads: choose a bead acquisition")
    key = (str(path), tuple(layout.regions), tuple(layout.flips))
    if key in _REGISTERED:
        return _REGISTERED[key]
    kept = registration_path(path)
    beads = Path(path)
    newest = max((p.stat().st_mtime for p in ([beads] if beads.is_file() else beads.rglob("*"))
                  if p.is_file() and p != kept), default=0.0)
    if kept.exists() and kept.stat().st_mtime >= newest:
        saved = load_registration(kept)
        if saved.regions == layout.regions and saved.flips == layout.flips:
            _REGISTERED[key] = saved.maps
            return saved.maps
    reg = register_channels(bead_projections(path, layout))
    try:
        save_registration(kept, layout.with_maps(reg.maps), reg, source=str(path))
    except OSError:
        pass                         # a read-only bead folder: measured again next time
    _REGISTERED[key] = reg.maps
    return reg.maps


def microscope_layout(resolution):
    """The channels' rectangles the microscope database gives for this file's
    microscope, its setup on the file's day and its camera, or None."""
    from ..microscope_db import acquisition_date
    scope = getattr(resolution, "microscope", None)
    entry = getattr(scope, "microscope", None)
    if entry is None:
        return None
    entry = entry.as_of(acquisition_date(scope.metadata or {}))
    return entry.channel_layout(scope.camera)


def layout_of(settings: FourPiFitSettings, base=None, calibration=None, camera=None,
              resolution=None):
    """The channel layout on this acquisition's frames, with its maps.

    The geometry: the file's own (quadrant files), the microscope's
    (`microscope_db.Microscope.channel_layout`), typed, or 4Pi-SMS -- the
    last three on the chip, put on the camera ROI's frames.  The maps: the
    model's, the beads', a registration file's, or none.
    """
    from ..channels import YALE_4PI, from_corners, load_registration
    part = settings.model.channels
    own = getattr(base, "layout", None)
    roi = camera.roi_offset if camera is not None else (0, 0)
    choice = part.layout
    if choice == "auto":
        choice = "file" if own is not None else "microscope"
    if choice == "file":
        if own is None:
            raise ValueError("this file does not lay out its channels itself (only the "
                             "quadrant .mat files do): use the microscope's, or type them")
        layout = own
    elif choice == "microscope":
        chip = microscope_layout(resolution)
        if chip is None:
            raise ValueError("the microscope database gives no channel rectangles for this "
                             "file's microscope and camera: add them to its entry (a region "
                             "per channel), or type the corners")
        layout = chip.on_camera(roi)
    elif choice == "yale":
        layout = YALE_4PI.on_camera(roi)
    else:
        x, y = _numbers(part.x_corners), _numbers(part.y_corners)
        if len(x) != len(y) or not 1 <= len(x) <= 4:
            raise ValueError("give one x and one y corner per channel, one to four of them")
        mirrored = set(_numbers(part.mirrored))
        layout = from_corners(x, y, part.width, part.height,
                              flip_x=[c + 1 in mirrored for c in range(len(x))]).on_camera(roi)
    maps = None
    if part.registration == "beads":
        maps = registration_from_beads(part.beads, layout)
    elif part.registration == "file":
        if not part.path:
            raise ValueError("choose a channel registration file")
        saved = load_registration(part.path)
        if saved.size != layout.size or saved.n_channels != layout.n_channels:
            raise ValueError(f"the registration was measured on {saved.n_channels} channels "
                             f"of {saved.size}, this layout has {layout.n_channels} of "
                             f"{layout.size}")
        maps = saved.maps
    elif part.registration == "model" and calibration is not None:
        maps = calibration.maps()
        measured = getattr(calibration, "layout", None)
        if maps is not None and measured:
            w, h = measured["regions"][0][2:]
            if (w, h) != tuple(layout.size):
                raise ValueError(f"the model's registration is between channels of {w} x {h}, "
                                 f"this layout's are {layout.size[0]} x {layout.size[1]}: "
                                 "register from beads instead")
    if maps is not None:
        if len(maps) != layout.n_channels:
            raise ValueError(f"the registration has {len(maps)} channels, the layout "
                             f"{layout.n_channels}")
        layout = layout.with_maps(maps)
    return layout


@register("Localize/Spline 4Pi")
class FourPiFit(_FitPlugin):
    """Fit 4Pi data: each molecule in every phase channel at once, with the IAB spline model, for x, y, the envelope's z and the interference phase."""

    Settings = FourPiFitSettings
    version = "1"
    phase = True
    # a simulated 4Pi camera's model, with the maps it was drawn with
    calibration_kind = "fourpi"
    calibration_extras = {"model.channels.registration": "model"}

    def _calibration(self, settings):
        from ..fourpi import load_fourpi_calibration
        path = settings.model.calibration
        if not path:
            raise ValueError("choose a 4Pi model: this program's, or uiPSF v1's "
                             "psfmodel_*_4pi.h5")
        cached = getattr(self, "_cal", None)
        if cached is not None and cached[0] == path:
            return cached[1]
        cal = load_fourpi_calibration(path)
        self._cal = (path, cal)
        return cal

    def model(self, settings, camera: CameraMetadata):
        from ..psf import Global4PiPSF
        xy = bool(settings.model.link_xy)
        return Global4PiPSF(self._calibration(settings),
                            shared=(xy, xy, True, False, True, True),
                            n_starts=int(settings.model.starts),
                            phase_starts=int(settings.model.phase_starts))

    # ----------------------------------------------------------- the channel layer
    def channel_layout(self, settings, base=None):
        """The layout on this acquisition's frames, with its maps (`layout_of`)."""
        from .fit import _open
        base = base if base is not None else _open(settings.source, watch=False)
        try:
            calibration = self._calibration(settings)
        except Exception:
            calibration = None             # a preview without a model still shows the channels
        resolution = self.resolution(settings)
        camera = resolution.camera_metadata() if resolution is not None else None
        return layout_of(settings, base, calibration, camera, resolution)

    def open_source(self, settings, watch: bool = False, stop_event=None):
        """The acquisition read as channels (`channels.ChannelSource`): from
        here on nothing sees the camera's frames."""
        from ..channels import ChannelSource
        from .fit import _open
        if watch and settings.model.channel_drift:
            raise ValueError("the channels' drift is measured over the whole acquisition, "
                             "which a live fit does not have yet: untick channel drift")
        base = _open(settings.source, watch=watch, stop_event=stop_event)
        return ChannelSource.wrap(base, self.channel_layout(settings, base))

    def watch_blocks(self, source, path, **options):
        # the camera's frames as the channels, cut as the file grows (a
        # DCIMG followed by its frames' footers, `io.dcimg.DcimgSource.watch`)
        return source.watch(**options)

    def view_camera(self, camera: CameraMetadata) -> CameraMetadata:
        # positions are in channel 0's own pixels, where the camera ROI is
        # already behind the channel layer
        from dataclasses import replace
        return replace(camera, roi=None)

    def engine(self, settings, camera: CameraMetadata, finder, model):
        from ..multichannel import MultiChannelEngine
        source = self.open_source(settings)
        drift = (self.channel_drift(settings, camera, finder, source)
                 if settings.model.channel_drift else None)
        return MultiChannelEngine(camera, finder, model, source.layout, settings.fit, drift)

    def channel_drift(self, settings, camera, finder, source):
        """The channels' drift, from blocks of frames spread over the
        acquisition fitted with x and y free in every channel
        (`multichannel.measure_channel_drift`)."""
        from ..locs import concat
        from ..multichannel import MultiChannelEngine, measure_channel_drift
        from ..pipeline import drive
        from ..psf import Global4PiPSF
        src = settings.source
        stop = min(src.stop, source.n_frames) if src.stop else source.n_frames
        blocks = max(int(settings.model.drift_blocks), 1)
        length = max(int(settings.model.drift_frames), 1)
        starts = np.linspace(src.start, max(stop - length, src.start), blocks).astype(int)
        free = Global4PiPSF(self._calibration(settings), shared=(False, False, True, False,
                                                                 True, True),
                            n_starts=int(settings.model.starts))
        engine = MultiChannelEngine(camera, finder, free, source.layout, settings.fit)
        parts = []
        for first in starts:
            drive(engine, source.frames(chunk=src.chunk, start=int(first),
                                        stop=int(min(first + length, stop))),
                  sink=parts.append)
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            raise ValueError("channel drift: nothing was fitted in the drift blocks")
        locs = concat(parts)
        edges = np.r_[starts, np.inf]
        return measure_channel_drift(locs, source.n_channels,
                                     camera.pixelsize_nm_or_default[0],
                                     lambda f: np.searchsorted(edges, f, side="right") - 1)

    def describe(self, ctx: Optional[Context], settings) -> FormState:
        state = super().describe(ctx, settings)
        part = settings.model.channels
        typed = part.layout == "typed"
        state.shown.update({"model.channels.beads": part.registration == "beads",
                            "model.channels.path": part.registration == "file",
                            **{f"model.channels.{n}": typed
                               for n in ("x_corners", "y_corners", "width", "height",
                                         "mirrored")}})
        state.active.update({"model.drift_blocks": settings.model.channel_drift,
                             "model.drift_frames": settings.model.channel_drift})
        if not settings.model.link_xy:
            state.notice = Notice("x and y fitted per channel: the table's dx_nm_ch*, "
                                  "dy_nm_ch* are each channel's offset from where the "
                                  "registration puts it")
        return state

    def preview(self, ctx: Context, settings, frame: int = 0) -> Result:
        """One frame: the channels' sum with the molecules found in it -- the
        panel a fit region is drawn on, in channel 0's pixels as the
        molecules are -- and each channel with the positions fitted there."""
        from ..camera import to_photons
        from ..multichannel import MultiChannelEngine, detection_image
        from . import ShapeEdit
        src = settings.source
        if not src.path:
            raise ValueError("choose a source file")
        source = self.open_source(settings)
        layout = source.layout
        index = int(np.clip(frame, 0, max(source.n_frames - 1, 0)))
        camera = self._camera(settings)
        if camera is None:
            raise ValueError(f"could not read the camera from {src.path}")
        camera.require("conversion", "offset")
        raw = source.frame(index)
        channels = to_photons(raw[None], camera) / camera.excess_noise
        summed = detection_image(channels, layout)
        finder = self.region_finder(settings, camera)
        candidates, _ = finder(summed, first_frame=index)
        text = (f"frame {index}: {len(candidates)} molecules found on the sum of "
                f"{layout.n_channels} channels")
        fitted = None
        try:
            model = self.model(settings, camera)
        except Exception as error:
            text += f"; not fitted -- {error}"
        else:
            engine = MultiChannelEngine(camera, finder, model, layout, settings.fit)
            engine.push(raw[None], index)
            fitted = engine.flush()
            text += f", {0 if fitted is None else len(fitted)} fitted"
        (px, py), _ = camera.pixelsize_nm_or_default
        shapes = list(settings.region.shapes)

        def draw(figure):
            from .fit import draw_region
            panels = figure.subplots(1, layout.n_channels + 1)
            panels[0].imshow(summed[0], cmap="gray")
            panels[0].plot(candidates.x, candidates.y, "o", mfc="none", color="C1", ms=8)
            panels[0].set_title("sum")
            draw_region(panels[0], shapes)
            for c in range(layout.n_channels):
                ax = panels[c + 1]
                ax.imshow(channels[0, c], cmap="gray")
                ax.set_title(f"channel {c + 1}")
                if fitted is not None and len(fitted):
                    xy = np.c_[np.asarray(fitted["x_nm"]) / px, np.asarray(fitted["y_nm"]) / py]
                    there = xy if c == 0 else layout.to_channel(c, xy)
                    ax.plot(there[:, 0], there[:, 1], "+", color="C3", ms=8)
            for ax in panels:
                ax.set_xticks([])
                ax.set_yticks([])
        return Result(text=text, settings=settings,
                      plot=Plot(draw, panels=layout.n_channels + 1,
                                size=(3 * (layout.n_channels + 1), 3.2),
                                edit=ShapeEdit("region.shapes", panel=0, shapes=shapes)))
