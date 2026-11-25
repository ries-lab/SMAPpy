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

from ..channels import FourPiChannelsSettings
from . import Context, FormState, Notice, Plot, Result, param, register
from .fit import (_FitPlugin, _Finishing, _Reading, _Writing, CameraMetadata, channel_photons)

MODEL_FILTER = "4Pi models (*.h5)"
REGISTRATION_FILTER = "Channel registrations (*.json)"
REGISTRATIONS = (("model", "the model's"), ("beads", "from beads"),
                 ("file", "from a registration file"), ("none", "none"))


def maps_learnt_in_situ(calibration) -> bool:
    """Whether ``calibration`` is an in-situ model whose channel maps were learnt from the movie: its maps are
    then part of the model -- moved where the molecules put each channel, 0.14 px on the 2017 tubulin's
    second channel -- and a registration from a file or beads would silently put the bead fields' back
    (docs/4pi.md §34)."""
    parameters = getattr(calibration, "parameters", None) or {}
    if "in-situ" not in str(parameters.get("method", "")):
        return False
    free = parameters.get("free")
    return free is None or any(str(n).endswith(".transform") for n in free)


@dataclass
class FourPiRegistrationSettings:
    """How the channels map onto the first: the maps kept with the model, measured on beads, a
    saved registration, or none."""
    source: str = param("model", label="from", choices=REGISTRATIONS,
                        help="the maps kept with the model, measured on beads, a saved "
                             "registration, or none")
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
    # as the two-channel fitters' channel move (`channel_move`): over time a robust smoothing spline, or constant
    drift_over_time: bool = param(True, label="over time",
                                  help="follow the drift over the acquisition as a smooth curve; off: one "
                                       "offset per channel for the whole acquisition")
    registration: FourPiRegistrationSettings = param(
        default_factory=FourPiRegistrationSettings, label="registration", collapsed=True)


@dataclass
class FourPiFitSettings(_Finishing, _Writing, _Reading):
    # the *channels* part in its place, four rectangles where the others read a split
    channels: FourPiChannelsSettings = param(default_factory=FourPiChannelsSettings,
                                             label="channels", collapsed=True)
    model: FourPiModelSettings = field(default_factory=FourPiModelSettings)


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


def model_geometry(calibration, base=None):
    """On *auto* with no microscope entry: the channels' rectangles the model
    was made on -- the PSF window keeps its bead scans' layout with it -- on
    this file's frames, where they fit them; None where the model keeps none,
    it was cut by a camera ROI, or its rectangles do not fit.  Bead scans
    are taken on the chip the movie is: the 2017 tubulin data's model, with
    its 4Pi-SMS rectangles, fits Cell01 with nothing typed (docs/4pi.md §26)."""
    from ..channels import ChannelLayout
    record = getattr(calibration, "layout", None)
    if not record or record.get("origins") or record.get("name") == "quadrant files":
        return None
    geometry = ChannelLayout.from_dict({**record, "maps": None})
    shape = getattr(base, "shape", None)
    if shape is not None and not geometry.fits(tuple(shape)[-2:]):
        return None
    return geometry


def layout_of(settings: FourPiFitSettings, base=None, calibration=None, camera=None,
              resolution=None):
    """The channel layout on this acquisition's frames, with its maps.

    The geometry: the file's own (quadrant files), the microscope's
    (`microscope_db.Microscope.channel_layout`), typed, or 4Pi-SMS -- the
    last three on the chip, put on the camera ROI's frames -- and on *auto*
    without either of the first two, the model's (`model_geometry`).  The
    maps: the model's, the beads', a registration file's, or none.
    """
    from ..channels import fourpi_layout, load_registration
    part = settings.channels
    registration = settings.model.registration
    own = getattr(base, "layout", None)
    roi = camera.roi_offset if camera is not None else (0, 0)
    entry = microscope_layout(resolution) if part.choice(own) == "microscope" else None
    geometry = model_geometry(calibration, base) if entry is None and part.layout == "auto" and own is None \
        else None
    # the model's rectangles are on its scans' frames, as this file's are: not moved by a camera ROI
    layout = geometry if geometry is not None else fourpi_layout(part, own, roi, entry)
    maps = None
    if registration.source == "beads":
        maps = registration_from_beads(registration.beads, layout)
    elif registration.source == "file":
        if not registration.path:
            raise ValueError("choose a channel registration file")
        saved = load_registration(registration.path)
        if saved.size != layout.size or saved.n_channels != layout.n_channels:
            raise ValueError(f"the registration was measured on {saved.n_channels} channels "
                             f"of {saved.size}, this layout has {layout.n_channels} of "
                             f"{layout.size}")
        maps = saved.maps
    elif registration.source == "model" and calibration is not None:
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


def brightest_molecule(model, rois, camera, candidates: int = 300):
    """The brightest molecule of a block's ROIs (`MultiChannelEngine.rois`) that the fit describes,
    fitted again and drawn by the model: (data, model, the fit's numbers), each (C, h, w) in photons --
    for the Preview's *molecule* figure; None when no fit ended inside the model's z grid.

    Up to ``candidates`` ROIs are fitted (a frame has fewer).  Of the fits inside the model's z grid,
    the brightest is taken whose residual is no worse than the typical one -- a mean squared Pearson
    residual within 10 % of their median -- so a model that misses the data everywhere still shows it,
    but a box holding two molecules does not stand for the fit: the brightest boxes are the
    overlapping pairs, and on a simulation the brightest was two molecules fitted at the end of the
    grid.  One frame's fits: milliseconds."""
    from ..fourpi import molecule_model
    from ..pipeline import with_readout
    images = np.asarray(rois.images, float)[:candidates]
    link = np.ascontiguousarray(rois.link[:candidates])
    data = np.asarray(with_readout(images, camera.readout_variance), np.float32)
    theta = np.asarray(model.fit(data, link).theta, float)
    cal = model.calibration
    nz = cal.shape[0]
    z = theta[:, model.slot(4)]
    good = np.flatnonzero(np.all(np.isfinite(theta), axis=1) & (z > 0.5) & (z < nz - 1.5))
    if not len(good):
        return None
    fitted = molecule_model(cal, model, theta[good], link[good], rois.roisize)
    chi = np.mean((data[good] - fitted) ** 2 / np.clip(fitted, 1e-3, None), axis=(1, 2, 3))
    usual = np.flatnonzero(chi <= 1.1 * np.median(chi))
    pick = usual[np.argmax(theta[good[usual], model.slot(2)])]
    k, row = good[pick], theta[good[pick]]
    numbers = {"photons": float(row[model.slot(2)]),
               "z_nm": float(cal.z_index_to_nm(row[model.slot(4)])),
               "phase_rad": float(np.angle(np.exp(1j * row[model.slot(5)])))}
    return np.asarray(data[k], float), fitted[pick], numbers


def draw_molecule(figure, data, fitted, numbers):
    """One molecule in every channel: the data (top), the model the fit found (middle), on one grey
    scale per channel, and the residual (bottom) in units of the noise, (data - model) / sqrt(model),
    from -4 (blue) to 4 (red) -- a plain difference took its scale from the one brightest pixel.  A good
    fit leaves noise; a ring or a lobe in the residual is a model that does not match the PSF there,
    and a channel whose residual has the spot's shape has the wrong share of the light, or the wrong
    phase offset."""
    from ..fourpi import in_pi
    C = data.shape[0]
    axes = np.asarray(figure.subplots(3, C)).reshape(3, C)
    residual = (data - fitted) / np.sqrt(np.clip(fitted, 1e-3, None))
    limit = 4.0
    for c in range(C):
        lo, hi = float(min(data[c].min(), fitted[c].min())), float(max(data[c].max(), fitted[c].max()))
        axes[0, c].imshow(data[c], cmap="gray", vmin=lo, vmax=hi)
        axes[1, c].imshow(fitted[c], cmap="gray", vmin=lo, vmax=hi)
        axes[2, c].imshow(residual[c], cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[0, c].set_title(f"channel {c + 1}", fontsize="small")
    for row, name in enumerate(("data", "model", "residual")):
        axes[row, 0].set_ylabel(name)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    figure.suptitle(f"{numbers['photons']:.0f} photons, z {numbers['z_nm']:.0f} nm, "
                    f"phase {in_pi(numbers['phase_rad'])}", fontsize="small")


@register("Localize/Spline 4Pi")
class FourPiFit(_FitPlugin):
    """Fit 4Pi data: each molecule in every phase channel at once, with the IAB spline model, for x, y, the envelope's z and the interference phase."""

    Settings = FourPiFitSettings
    # 2: positions on the camera chip, where channel 0 saw them, not in its own pixels
    # 3: a fit region's part on any channel copied onto channel 0 (`fit.region_copies`)
    # 5: two fits of one frame within a pixel are one molecule's (`pipeline.one_per_molecule`)
    version = "5"
    phase = True
    # its *channels* part is its own (`FourPiChannelsSettings`): four regions, not a two-way split
    channel_part = "own"
    # a simulated 4Pi camera's model, with the maps it was drawn with
    calibration_kind = "fourpi"
    calibration_extras = {"model.registration.source": "model"}

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

    def layer_layout(self, settings):
        return self.open_source(settings).layout

    def region_finder(self, settings, camera: Optional[CameraMetadata] = None):
        """The molecules found on the channels' sum, in the main channel's own pixels, judged by a
        region drawn on the frame and copied onto every channel (`fit.ChannelRegionFinder`)."""
        from .fit import ChannelRegionFinder
        finder = self.finder(settings, camera)
        region = getattr(settings, "region", None)
        if region is None or not region.active():
            return finder
        return ChannelRegionFinder(finder, list(region.shapes), self.open_source(settings).layout)

    def engine(self, settings, camera: CameraMetadata, finder, model):
        from ..multichannel import MultiChannelEngine
        source = self.open_source(settings)
        drift = (self.channel_drift(settings, camera, finder, source)
                 if settings.model.channel_drift else None)
        self._drift = drift                 # the run's figure (`run`)
        # a whole file: each block fitted while the next is found (`MultiChannelEngine.overlap`);
        # a live fit shows each block as soon as it is fitted
        return MultiChannelEngine(camera, finder, model, source.layout, settings.fit, drift,
                                  frame_shape=source.base.shape[-2:],
                                  overlap=not settings.source.live)

    def drift_key(self, settings, camera, source) -> str:
        """The cache's key for this run's channel drift (`multichannel.drift_cache_key`): the
        movie, the model, the layout with its maps, and every setting the prefit reads -- the
        frames, the camera, detection, the fit region, the fit's and the drift's own."""
        from dataclasses import asdict
        from ..multichannel import drift_cache_key
        src, model = settings.source, settings.model
        stop = min(src.stop, source.n_frames) if src.stop else source.n_frames
        measured = {"version": self.version, "start": src.start, "stop": stop,
                    "frames": source.n_frames, "camera": camera.to_dict(),
                    "layout": source.layout.to_dict(), "starts": model.starts,
                    "blocks": model.drift_blocks, "block_frames": model.drift_frames,
                    "over_time": model.drift_over_time}
        for part in ("detection", "region", "fit"):
            value = getattr(settings, part, None)
            if value is not None:
                measured[part] = asdict(value)
        # how many threads fit it changes nothing in it
        measured.get("fit", {}).pop("n_threads", None)
        return drift_cache_key(src.path, model.calibration, measured)

    def channel_drift(self, settings, camera, finder, source):
        """The channels' drift, from blocks of frames spread over the
        acquisition fitted with x and y free in every channel
        (`multichannel.measure_channel_drift`); measured once for a movie and
        its settings, and read from beside the movie after that
        (`multichannel.cached_drift`)."""
        from ..multichannel import cached_drift, keep_drift
        key = self.drift_key(settings, camera, source)
        kept = cached_drift(settings.source.path, key)
        self._drift_kept = kept is not None
        if kept is not None:
            return kept
        drift = self._measure_channel_drift(settings, camera, finder, source)
        keep_drift(settings.source.path, key, drift)
        return drift

    def _measure_channel_drift(self, settings, camera, finder, source):
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
        engine = MultiChannelEngine(camera, finder, free, source.layout, settings.fit,
                                    frame_shape=source.base.shape[-2:], overlap=True)
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
                                     lambda f: np.searchsorted(edges, f, side="right") - 1,
                                     over_time=settings.model.drift_over_time,
                                     span=(float(starts[0]), float(min(starts[-1] + length, stop))))

    _drift = None
    _drift_kept = False

    def run(self, ctx: Context, settings) -> Result:
        """The fit; with *channel drift*, its figure beside the run's, channel by channel, as the two-channel
        fitters draw their channel move (`channel_move.draw`)."""
        self._drift, self._drift_kept = None, False
        result = super().run(ctx, settings)
        drift = self._drift
        if drift is not None and self._drift_kept:
            from ..multichannel import drift_cache_path
            result.text += (f"; the channels' drift as measured before, from "
                            f"{drift_cache_path(settings.source.path).name}")
        if drift is not None:
            from ..channel_move import ChannelMove, draw
            for c in range(1, drift.offsets.shape[1]):
                curve = ChannelMove(drift.offsets[:, c], drift.frames if len(drift.frames) > 1 else None)
                result.plots = {**result.plots,
                                f"channel {c + 1} drift": draw(curve, (drift.data or {}).get(c))}
        return result

    def describe(self, ctx: Optional[Context], settings) -> FormState:
        state = super().describe(ctx, settings)
        part, registration = settings.channels, settings.model.registration
        state.shown.update({"model.registration.beads": registration.source == "beads",
                            "model.registration.path": registration.source == "file",
                            **{f"channels.{n}": flag for n, flag in part.shown().items()}})
        state.titles["channels"] = part.title()
        state.titles["model.registration"] = "registration: " + dict(REGISTRATIONS).get(
            registration.source, registration.source)
        state.active.update({"model.drift_blocks": settings.model.channel_drift,
                             "model.drift_frames": settings.model.channel_drift,
                             "model.drift_over_time": settings.model.channel_drift})
        if not settings.model.link_xy:
            state.notice = Notice("x and y fitted per channel: the table's dx_nm_ch*, "
                                  "dy_nm_ch* are each channel's offset from where the "
                                  "registration puts it")
        elif registration.source != "model" and settings.model.calibration:
            try:
                learnt = maps_learnt_in_situ(self._calibration(settings))
            except (OSError, ValueError, KeyError):
                learnt = False
            if learnt:
                label = dict(REGISTRATIONS).get(registration.source, registration.source)
                state.notice = Notice("this model learnt the channels' maps from the movie in situ, "
                                      f"and the registration '{label}' is used instead: set it to "
                                      "'the model's'")
        return state

    def preview(self, ctx: Context, settings, frame: int = 0) -> Result:
        """One frame: the channels' sum with the molecules found in it -- the
        panel a fit region is drawn on, drawn where the main channel lies on
        the frame, as for a whole-chip fit -- and each channel with the
        positions fitted there."""
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
        channels = channel_photons(raw[None], camera, settings, source)
        summed = detection_image(channels, layout)
        finder = self.region_finder(settings, camera)
        candidates, _ = finder(summed, first_frame=index)
        text = (f"frame {index}: {len(candidates)} molecules found on the sum of "
                f"{layout.n_channels} channels")
        fitted = molecule = None
        try:
            model = self.model(settings, camera)
        except Exception as error:
            text += f"; not fitted -- {error}"
        else:
            engine = MultiChannelEngine(camera, finder, model, layout, settings.fit,
                                        frame_shape=source.base.shape[-2:], keep_rois=True)
            engine.push(raw[None], index)
            fitted = engine.flush()
            text += f", {0 if fitted is None else len(fitted)} fitted"
            if engine.rois is not None and len(engine.rois):
                molecule = brightest_molecule(model, engine.rois, camera)
                if molecule is not None:
                    text += "; the brightest molecule's data, model and residual in the figure molecule"
        (px, py), _ = camera.pixelsize_nm_or_default
        roi = np.array(camera.roi_offset, float)
        shapes = list(settings.region.shapes)
        found_on_frame = layout.on_frame(0, np.c_[candidates.x, candidates.y])
        own = None
        if fitted is not None and len(fitted):
            # the table is on the chip: less the ROI's corner, on the frame; then the main channel's
            on_frame = np.c_[np.asarray(fitted["x_nm"]) / px, np.asarray(fitted["y_nm"]) / py] - roi
            own = layout.from_frame(0, on_frame)

        def draw(figure):
            from .fit import draw_region
            panels = figure.subplots(1, layout.n_channels + 1)
            panels[0].imshow(summed[0], cmap="gray", extent=layout.frame_extent(0))
            panels[0].plot(found_on_frame[:, 0], found_on_frame[:, 1], "o", mfc="none",
                           color="C1", ms=8)
            panels[0].set_title("sum")
            draw_region(panels[0], shapes, layout)
            for c in range(layout.n_channels):
                ax = panels[c + 1]
                ax.imshow(channels[0, c], cmap="gray")
                ax.set_title(f"channel {c + 1}")
                if own is not None:
                    there = own if c == 0 else layout.to_channel(c, own)
                    ax.plot(there[:, 0], there[:, 1], "+", color="C3", ms=8)
            for ax in panels:
                ax.set_xticks([])
                ax.set_yticks([])
        plots = {}
        if molecule is not None:
            plots["molecule"] = Plot(lambda figure: draw_molecule(figure, *molecule),
                                     panels=3 * layout.n_channels, size=(2.2 * layout.n_channels, 6.4))
        return Result(text=text, settings=settings, plots=plots,
                      plot=Plot(draw, panels=layout.n_channels + 1,
                                size=(3 * (layout.n_channels + 1), 3.2),
                                edit=ShapeEdit("region.shapes", panel=0, shapes=shapes)))
