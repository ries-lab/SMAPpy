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
LAYOUTS = (("file", "as the file lays them out"), ("typed", "typed corners"),
           ("yale", "4Pi-SMS (Bewersdorf lab)"))
REGISTRATIONS = (("model", "the model's"), ("none", "none"))


@dataclass
class ChannelLayoutSettings:
    layout: str = param("file", label="channels", choices=LAYOUTS,
                        help="where the phase channels are on the frame")
    x_corners: str = param("0, 200, 400, 600", label="x corners", unit="px",
                           help="each channel's left edge on the frame, comma-separated")
    y_corners: str = param("0, 0, 0, 0", label="y corners", unit="px",
                           help="each channel's top edge on the frame")
    width: int = param(200, label="width", unit="px", min=8,
                       help="one channel's width")
    height: int = param(200, label="height", unit="px", min=8, beside=True,
                        help="one channel's height")
    mirrored: str = param("", label="mirrored in x",
                          help="the channels mirrored left-right, by number from 1, "
                               "comma-separated; empty: none")
    registration: str = param("model", label="registration", choices=REGISTRATIONS,
                              help="how the channels map onto the first: the maps kept "
                                   "with the model, or none (the channels are aligned)")


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


@dataclass
class FourPiFitSettings(_Finishing, _Writing, _Reading):
    model: FourPiModelSettings = field(default_factory=FourPiModelSettings)


def _numbers(text: str) -> list:
    return [int(round(float(w))) for w in str(text).replace(";", ",").split(",") if w.strip()]


def layout_of(settings: FourPiFitSettings, source=None, calibration=None):
    """The channel layout the settings describe, with the model's maps."""
    from ..channels import YALE_4PI, from_corners
    part = settings.model.channels
    if part.layout == "file":
        if source is None:
            from .fit import _open
            source = _open(settings.source, watch=False)
        layout = getattr(source, "layout", None)
        if layout is None:
            raise ValueError("this file does not lay out its channels itself (only the "
                             "quadrant .mat files do): type the corners, or choose 4Pi-SMS")
    elif part.layout == "yale":
        layout = YALE_4PI
    else:
        x, y = _numbers(part.x_corners), _numbers(part.y_corners)
        if len(x) != len(y) or not 1 <= len(x) <= 4:
            raise ValueError("give one x and one y corner per channel, one to four of them")
        mirrored = set(_numbers(part.mirrored))
        layout = from_corners(x, y, part.width, part.height,
                              flip_x=[c + 1 in mirrored for c in range(len(x))])
    if part.registration == "model" and calibration is not None:
        maps = calibration.maps()
        if maps is not None:
            if len(maps) != layout.n_channels:
                raise ValueError(f"the model registers {len(maps)} channels, the layout has "
                                 f"{layout.n_channels}")
            layout = layout.with_maps(maps)
    return layout


@register("Localize/Spline 4Pi")
class FourPiFit(_FitPlugin):
    """Fit 4Pi data: each molecule in every phase channel at once, with the IAB spline model, for x, y, the envelope's z and the interference phase."""

    Settings = FourPiFitSettings
    version = "1"
    phase = True

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
                            n_starts=int(settings.model.starts))

    def engine(self, settings, camera: CameraMetadata, finder, model):
        from ..multichannel import MultiChannelEngine
        layout = layout_of(settings, calibration=model.calibration)
        return MultiChannelEngine(camera, finder, model, layout, settings.fit)

    def region_finder(self, settings, camera: Optional[CameraMetadata] = None):
        # the molecules are found on the channels' sum, in channel 0's pixels,
        # not on the frame a region would be drawn on
        return self.finder(settings, camera)

    def describe(self, ctx: Optional[Context], settings) -> FormState:
        state = super().describe(ctx, settings)
        part = settings.model.channels
        typed = part.layout == "typed"
        state.shown.update({"region": False,
                            **{f"model.channels.{n}": typed
                               for n in ("x_corners", "y_corners", "width", "height",
                                         "mirrored")}})
        if not settings.model.link_xy:
            state.notice = Notice("x and y fitted per channel: the table's dx_nm_ch*, "
                                  "dy_nm_ch* are each channel's offset from where the "
                                  "registration puts it")
        return state

    def preview(self, ctx: Context, settings, frame: int = 0) -> Result:
        """One frame: the channels' sum with the molecules found in it, and
        each channel with the positions fitted there."""
        from ..camera import to_photons
        from ..multichannel import detection_image
        from .fit import _open
        src = settings.source
        if not src.path:
            raise ValueError("choose a source file")
        source = _open(src, watch=False)
        index = int(np.clip(frame, 0, max(source.n_frames - 1, 0)))
        camera = self._camera(settings)
        if camera is None:
            raise ValueError(f"could not read the camera from {src.path}")
        camera.require("conversion", "offset")
        try:
            cal = self._calibration(settings)
        except Exception:
            cal = None
        layout = layout_of(settings, source, cal)
        raw = source.frame(index)
        channels = layout.cut(to_photons(raw, camera)[None] / camera.excess_noise)
        summed = detection_image(channels, layout)
        finder = self.finder(settings, camera)
        candidates, _ = finder(summed, first_frame=index)
        text = f"frame {index}: {len(candidates)} molecules found on the sum of " \
               f"{layout.n_channels} channels"
        fitted = None
        if cal is not None:
            model = self.model(settings, camera)
            engine = self.engine(settings, camera, finder, model)
            engine.push(raw[None], index)
            fitted = engine.flush()
            if fitted is not None:
                text += f", {len(fitted)} fitted"
        pixel = (camera.pixelsize_nm_or_default[0] if fitted is not None else 1.0)

        def draw(figure):
            panels = figure.subplots(1, layout.n_channels + 1)
            panels[0].imshow(summed[0], cmap="gray")
            panels[0].plot(candidates.x, candidates.y, "o", mfc="none", color="C1", ms=8)
            panels[0].set_title("sum")
            for c in range(layout.n_channels):
                ax = panels[c + 1]
                ax.imshow(channels[0, c], cmap="gray")
                ax.set_title(f"channel {c + 1}")
                if fitted is not None and len(fitted):
                    x0, y0 = layout.regions[0][:2]
                    xy = np.c_[np.asarray(fitted["x_nm"]) / pixel - x0,
                               np.asarray(fitted["y_nm"]) / pixel - y0]
                    there = xy if c == 0 else layout.to_channel(c, xy)
                    ax.plot(there[:, 0], there[:, 1], "+", color="C3", ms=8)
            for ax in panels:
                ax.set_xticks([])
                ax.set_yticks([])
        return Result(text=text, plot=Plot(draw, panels=layout.n_channels + 1,
                                           size=(3 * (layout.n_channels + 1), 3.2)),
                      settings=settings)
