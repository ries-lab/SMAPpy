"""Process image: background subtraction and local contrast of an image layer.

SMAP's ``Tiff_remove_BG``, which subtracted a wavelet background from a TIFF
associated with a file.  Here it acts on an image layer -- a wide-field
channel, a rendered TIFF, the camera frames a fit kept -- and the result is
a **new image layer** beside it, never the original rewritten: the original
is what was measured, and comparing the two is how the settings are judged
(SMAP's *replace, not append* is gone for the same reason).  The new image
keeps the original's pixel size, corner and placement (`ImageData.placement`),
so it lies exactly over it and over the localizations.

**Operations are a list**, ticked on or off, each with a part shown while it
is ticked, applied in the order of `_STEPS`: the background first, then the
contrast.  A new operation is a function in `smappy.image_processing`, a tick
and a part here, and a line in `_STEPS`.

**Settings in nm, the algorithms in pixels.**  The form speaks the table's
units, as everywhere else; each run converts with the image's pixel size,
and the line under the form says what that is in pixels.  The wavelet's
*level* stays a level, as in SMAP: its scale is 2^level pixels by
construction.

**Which planes**: the one the layer shows, by default -- an average of the
kept frames, a single wide-field picture -- or every plane of a stack, each
on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import Context, FormState, Notice, Plot, Plugin, Result, param, register


def _no_image():
    return [(0, "the first image layer")]


@dataclass
class BackgroundSettings:
    method: str = param("gaussian", label="method",
                        choices=(("gaussian", "Gaussian"),
                                 ("ball", "rolling ball"),
                                 ("wavelet", "wavelet (SMAP)")),
                        help="how the smooth background is estimated")
    sigma_nm: float = param(1000.0, label="width", unit="nm", min=1.0,
                            help="the Gaussian's SD: wider than any structure kept")
    radius_nm: float = param(1000.0, label="ball radius", unit="nm", min=1.0,
                             help="wider than any structure kept")
    level: int = param(3, label="level", min=1, max=10,
                       help="the scale is about 2^level pixels; SMAP's 2-4")
    rounds: int = param(5, label="rounds", min=0, max=20, advanced=True,
                        help="times bright structures are taken out of the "
                             "estimate; 0: a plain high-pass")


@dataclass
class ContrastSettings:
    window_nm: float = param(1000.0, label="window", unit="nm", min=1.0,
                             help="the SD of the Gaussian window of the local "
                                  "mean and SD")
    floor: float = param(0.5, label="floor", min=0.0, advanced=True,
                         help="the local SD is at least this times its median")


@dataclass
class ProcessImageSettings:
    layer: int = param(0, label="image layer", choices=_no_image,
                       help="the image to process; the result is a new layer")
    planes: str = param("shown", label="planes",
                        choices=(("shown", "the one shown"), ("all", "every plane")),
                        help="of a stack: the plane the layer shows, or each")
    background: bool = param(True, label="subtract background",
                             help="a smooth background estimated and subtracted")
    bg: BackgroundSettings = param(default_factory=BackgroundSettings,
                                   label="background")
    contrast: bool = param(False, label="normalise contrast",
                           help="divide by the local SD, after the background")
    lcn: ContrastSettings = param(default_factory=ContrastSettings,
                                  label="local contrast")


def _background(plane: np.ndarray, s: BackgroundSettings, pixel: float):
    from ..image_processing import subtract_background
    return subtract_background(plane, s.method, sigma_px=s.sigma_nm / pixel,
                               radius_px=s.radius_nm / pixel, level=s.level,
                               rounds=s.rounds)[0]


def _contrast(plane: np.ndarray, s: ContrastSettings, pixel: float):
    from ..image_processing import local_contrast
    return local_contrast(plane, s.window_nm / pixel, s.floor)


# The operations in the order they run: (tick, part, words for the name,
# function(plane, part's settings, pixel size in nm)).
_STEPS: List[Tuple[str, str, str, Callable]] = [
    ("background", "bg", "background subtracted", _background),
    ("contrast", "lcn", "contrast normalised", _contrast),
]


def process_plane(plane: np.ndarray, settings: ProcessImageSettings,
                  pixelsize: float) -> np.ndarray:
    """``plane`` through every ticked operation, in order."""
    out = np.asarray(plane, np.float64)
    for tick, part, _, function in _STEPS:
        if getattr(settings, tick):
            out = function(out, getattr(settings, part), float(pixelsize))
    return out


def process_image(image, settings: ProcessImageSettings, frame: int = 0):
    """A new `ImageData`: ``image`` processed, the plane ``frame`` or every
    plane (``settings.planes``), at the same pixel size, corner and
    placement."""
    from ..images import ImageData
    pixel = float(np.sqrt(np.prod(image.pixelsize_xy)))
    frames = None if image.frames is None else np.asarray(image.frames)
    if settings.planes == "all" and image.data.ndim == 3:
        data = np.stack([process_plane(image.plane_at(i), settings, pixel)
                         for i in range(image.n_frames)])
        shown = int(frame)
    else:
        data = process_plane(image.plane_at(int(frame)), settings, pixel)
        shown = 0
        if frames is not None:       # the one plane's frame number, for its label
            frames = frames[int(np.clip(frame, 0, len(frames) - 1)):][:1]
    words = ", ".join(w for tick, _, w, _ in _STEPS if getattr(settings, tick))
    return ImageData(
        data.astype(np.float32), image.pixelsize, x0=image.x0, y0=image.y0,
        name=f"{image.name or 'image'} ({words})", frame=shown,
        metadata={"processed_from": image.name, "operations": words},
        pixelsize_y=image.pixelsize_y, frames=frames,
        placement=None if image.placement is None else np.array(image.placement))


def _image_layers(session) -> List[int]:
    return [i for i, layer in enumerate(session.layers)
            if layer.is_image and not layer.image.is_empty]


@register("Analysis/Process/Process image")
class ProcessImage(Plugin):
    """Subtract the background of an image layer, or normalise its local
    contrast, into a new image layer over the original."""

    description = ("Subtract a smooth background from an image layer -- "
                   "Gaussian, rolling ball or SMAP's wavelet -- and/or "
                   "normalise its local contrast; the result is a new image "
                   "layer at the original's place.")
    Settings = ProcessImageSettings
    changes_locs = False
    version = "1"
    preview_help = "the original, the background and the result, nothing added"

    def describe(self, ctx: Optional[Context], settings: ProcessImageSettings) -> FormState:
        method = settings.bg.method
        state = FormState(shown={
            "bg": settings.background, "lcn": settings.contrast,
            "bg.sigma_nm": method == "gaussian", "bg.radius_nm": method == "ball",
            "bg.level": method == "wavelet", "bg.rounds": method != "ball"})
        session = getattr(ctx, "session", None)
        if session is None:
            return state
        layers = _image_layers(session)
        if layers:
            state.choices["layer"] = [(0, "the first image layer")] + [
                (i + 1, f"{i + 1}: {session.layers[i].name}") for i in layers]
            try:
                image = self._layer(session, settings)[1].image
            except ValueError:           # a layer since removed: say so at Run
                image = session.layers[layers[0]].image
            pixel = float(np.sqrt(np.prod(image.pixelsize_xy)))
            state.active["planes"] = image.n_frames > 1
            scales = []
            if settings.background:
                if method == "wavelet":
                    scales.append(f"level {settings.bg.level}: about "
                                  f"{2 ** settings.bg.level * pixel:.0f} nm")
                else:
                    nm = settings.bg.sigma_nm if method == "gaussian" else settings.bg.radius_nm
                    scales.append(f"{nm:g} nm = {nm / pixel:.1f} pixels")
            if settings.contrast:
                scales.append(f"window {settings.lcn.window_nm / pixel:.1f} pixels")
            if scales:
                state.notice = Notice(f"pixel {pixel:g} nm; " + "; ".join(scales),
                                      kind="info")
        else:
            state.notice = Notice("no image layer: open an image, or add an "
                                  "image layer in the Render tab", kind="info")
        return state

    @staticmethod
    def _layer(session, settings: ProcessImageSettings):
        layers = _image_layers(session) if session is not None else []
        if not layers:
            raise ValueError("no image layer to process: open an image, or "
                             "add an image layer in the Render tab")
        index = layers[0] if int(settings.layer) == 0 else int(settings.layer) - 1
        if index not in layers:
            raise ValueError(f"layer {index + 1} is not an image layer")
        return index, session.layers[index]

    def run(self, ctx: Context, settings: ProcessImageSettings) -> Result:
        return self._work(ctx, settings, apply=True)

    def preview(self, ctx: Context, settings: ProcessImageSettings) -> Result:
        return self._work(ctx, settings, apply=False)

    def _work(self, ctx: Context, settings: ProcessImageSettings, apply: bool) -> Result:
        if not (settings.background or settings.contrast):
            raise ValueError("tick an operation: subtract background or "
                             "normalise contrast")
        index, layer = self._layer(ctx.session, settings)
        image = layer.image
        frame = int(layer.frame)
        if apply:
            ctx.report(f"processing {layer.name}")
            made = process_image(image, settings, frame)
            shown = made.plane
        else:
            made = None
            pixel = float(np.sqrt(np.prod(image.pixelsize_xy)))
            shown = process_plane(image.plane_at(frame), settings, pixel)
        original = np.asarray(image.plane_at(frame), np.float64)
        text = (f"image layer {index + 1} ({layer.name}): "
                + ", ".join(w for tick, _, w, _ in _STEPS if getattr(settings, tick)))
        if made is not None:
            text += f"; new layer {made.name!r}"
        return Result(text=text, settings=settings,
                      plot=Plot(_draw(original, np.asarray(shown, np.float64),
                                      settings.background and not settings.contrast),
                                panels=3, size=(9.0, 3.2)),
                      data={"layer": index, "image": made},
                      image_layers=[made] if made is not None else [],
                      title=f"image layer {index + 1}")


def _draw(original: np.ndarray, processed: np.ndarray, subtracted: bool):
    """The original, what was taken away (the background) or nothing, and
    the result, side by side."""
    def draw(fig):
        axes = fig.subplots(1, 3)
        removed = original - processed if subtracted else None
        lo, hi = np.percentile(original, (1, 99.5))
        panels = [(original, "original", (lo, hi)),
                  (removed, "background", (lo, hi)),
                  (processed, "result", tuple(np.percentile(processed, (1, 99.5))))]
        for ax, (plane, title, (vmin, vmax)) in zip(axes, panels):
            ax.axis("off")
            if plane is None:
                ax.set_title("contrast normalised: no background", fontsize=8)
                continue
            ax.imshow(plane, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=9)
    return draw
