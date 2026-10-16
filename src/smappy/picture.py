"""The picture of the view, as a file: one renderer for every way of saving one.

*File/Export/Image*, the render window's *Save* and *Copy*, and the 3D
window's *Save* used to be three pieces of code that shared none: the window
saved its panels as displayed or a TIFF of the visible layers re-rendered,
the plugin one layer with that layer's settings, and none of them drew a
scale bar.  Here is the one that remains (`render_picture`): what the window
shows -- the visible layers, the panels as they are laid out
(`layer_state.render_panels`, the window's own render), on a field of view at
a pixel size -- with what is drawn over it, a scale bar, the colour bars of
layers coloured by a column and the panels' names.  It needs no window and
no Qt: the text is drawn with Pillow, which matplotlib already brings.

What the window knows and the session does not -- where it looks, how fine,
how the panels are laid out, the 3D view's projection -- comes as a
`ViewFrame`, which each window hands the session (`Session.view_frames`).
Without one (a script, a batch) the picture is of everything at a pixel size
that is set.

The file says how it was made, as SMAP's *TifSaver* did: the pixel size in
the resolution tags, and the files, layers, render settings, filters and
grouping as YAML in the TIFF's description, the PNG's ``Description`` text
chunk or the JPEG's comment (`describe_view`, `read_description`).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import lut as luts
from .layer_state import LayerSnapshot, render_panels
from .render import FieldOfView, axis_unit, finite_range, normalize

# what a picture is made of: (value, label), the plugin's choices
WHAT = (("view", "as the view"), ("composite", "composite only"),
        ("layers", "each layer, colour"), ("layers grey", "each layer, grey"),
        ("intensity", "intensity, 32-bit"))
REGIONS = (("view", "the view"), ("roi", "the ROI"), ("all", "everything"))
FORMATS = {".png": "PNG", ".tif": "TIFF", ".tiff": "TIFF", ".jpg": "JPEG", ".jpeg": "JPEG"}
DEFAULT_PIXEL = 10.0       # nm, when "as displayed" has no window to ask
PANEL_LINE = 2             # px of grey between panels, as SMAP draws it
BAR_FRACTION = 0.2         # the automatic scale bar: a round fifth of the width


@dataclass(frozen=True)
class ViewFrame:
    """What a window shows that the session does not know.

    ``xrange``/``yrange`` are the view's, in render units (the projected
    plane's, for the 3D view); ``pixelsize`` what one screen pixel is;
    ``panels`` and ``stacked`` the 2D window's layout (`RenderView.panels`).
    ``projection`` and ``slab`` make it the 3D view's.
    """

    xrange: Tuple[float, float]
    yrange: Tuple[float, float]
    pixelsize: float
    panels: Tuple = ("overlay",)
    stacked: bool = False
    projection: Any = None
    slab: Any = None


class FrameSource:
    """A window's `ViewFrame` as the session holds it (`Session.view_frames`).

    The window writes ``frame`` on its own thread whenever the view moves,
    and a plugin reads it from its worker: plain data, so no widget is ever
    read, or its last reference dropped, off the GUI thread
    (`gui.collector`).  Calling it gives the frame, None before the first.
    """

    def __init__(self, frame: Optional[ViewFrame] = None):
        self.frame = frame

    def __call__(self) -> Optional[ViewFrame]:
        return self.frame


@dataclass
class Picture:
    """One file's worth: the pixels, what they cover, and what was drawn."""

    image: np.ndarray            # uint8 (h, w, 3) or (h, w); float32 (h, w)
    fov: FieldOfView             # of one panel
    suffix: str = ""             # added to the file's name: "_layer2"
    unit: str = "nm"             # of the pixel size
    scale_bar: float = 0.0       # the bar's length in ``unit``; 0: none
    scale_bar_px: int = 0


# ------------------------------------------------------------------ the bars
def nice_step(span: float) -> float:
    """A round 1 / 2 / 5 x 10^k at or above ``span``: the scale bar's length."""
    decade = 10 ** np.floor(np.log10(span))
    for m in (1, 2, 5, 10):
        if m * decade >= span:
            return float(m * decade)
    return float(decade)


def nice_below(span: float) -> float:
    """A round 1 / 2 / 5 x 10^k at or below ``span``.

    `nice_step` rounds up, which is what a bar of "at least this much" wants;
    a bar that must not eat a third of the picture wants the other direction.
    """
    if not span or not np.isfinite(span):
        return 1.0
    decade = 10.0 ** np.floor(np.log10(abs(span)))
    return float(max((m * decade for m in (1, 2, 5) if m * decade <= abs(span)),
                     default=decade))


def bar_label(value: float, unit: str) -> str:
    """"500 nm", "2 µm", "2000 photons"."""
    if unit == "nm" and value >= 1000:
        return f"{value / 1000:g} µm"
    return f"{value:g} {unit}"


def bar_number(value: float) -> str:
    """Four digits, but a frame number of 20000 as itself, not 2e+04."""
    return f"{value:.0f}" if 1e4 <= abs(value) < 1e9 else f"{value:.4g}"


def auto_bar(width: float) -> float:
    """The scale bar the window draws on a view ``width`` wide."""
    return nice_step(BAR_FRACTION * width) if width > 0 else 0.0


def colour_scales(layers: Sequence[LayerSnapshot]) -> List[tuple]:
    """``(layer names, column, (lo, hi), LUT table)`` per distinct colour
    scale of the visible layers coloured by a column, in their order.

    A bar per scale, not per layer: layers that agree on the column, the
    colours and the range share one, labelled with all of them.  The window's
    bars (`gui.render_view.ColourBars`) and the exported picture's are these.
    """
    found: dict = {}
    for layer in layers:
        if layer.is_image or not layer.visible or layer.settings is None:
            continue
        field = layer.settings.color_field
        if field is None or layer.locs is None or field not in layer.locs:
            continue
        lo, hi = layer.settings.color_range or finite_range(layer.locs[field])
        try:
            table = luts.get(layer.display.lut, layer.display.invert)
        except (KeyError, ValueError):
            continue
        if layer.display.white_background:
            table = luts.on_white(table)
        # the colours themselves, not the LUT's name: an inverted LUT and
        # its plain one share a name and not a scale
        key = (field, float(lo), float(hi), table.tobytes())
        if key in found:
            found[key][0].append(layer.name)
        else:
            found[key] = ([layer.name], field, (float(lo), float(hi)), table)
    return list(found.values())


def colour_bar_labels(names: Sequence[str], field: str, range_: Tuple[float, float]
                      ) -> Tuple[str, str, str]:
    """The title, the top and the bottom label: "layer 1: z_nm", "300 nm",
    "-300 nm"."""
    unit = axis_unit(field)
    unit = "" if unit == field else f" {unit}"
    lo, hi = range_
    return (f"{', '.join(names)}: {field}", f"{bar_number(hi)}{unit}",
            f"{bar_number(lo)}{unit}")


# ------------------------------------------------------------- where, how fine
def frame_fov(frame: Optional[ViewFrame], region: str, pixelsize: Optional[float],
              roi_bounds=None, everything=None) -> FieldOfView:
    """The grid of the picture: the view, the ROI's box or everything, at
    ``pixelsize`` -- None for the view's own.

    ``everything`` is ``(xrange, yrange)`` covering the data
    (`Session.full_view`); for the 3D view it is the slab, projected.
    """
    if pixelsize is None:
        pixelsize = frame.pixelsize if frame is not None else DEFAULT_PIXEL
    if region == "roi" and (frame is None or frame.projection is None):
        if roi_bounds is None:
            raise ValueError("the ROI: draw one first, or export the view")
        x0, y0, x1, y1 = roi_bounds
        return FieldOfView.from_range((x0, x1), (y0, y1), pixelsize)
    if region in ("roi", "all") and frame is not None and frame.projection is not None:
        box = slab_box(frame.projection, frame.slab)
        if box is not None:
            return FieldOfView.from_range(*box, pixelsize)
        region = "view"
    if region == "view" and frame is not None:
        return FieldOfView.from_range(frame.xrange, frame.yrange, pixelsize)
    if everything is None:
        raise ValueError("nothing to frame the picture on")
    return FieldOfView.from_range(*everything, pixelsize)


def slab_box(projection, slab):
    """The slab's corners on the 3D view's plane, as ranges; None without one."""
    if slab is None:
        return None
    c = slab.corners()
    xv, yv, _ = projection.apply(c[:, 0], c[:, 1], c[:, 2])
    return (float(xv.min()), float(xv.max())), (float(yv.min()), float(yv.max()))


# -------------------------------------------------------------------- render
def render_picture(layers: Sequence[LayerSnapshot], fov: FieldOfView,
                   what: str = "view", panels: Sequence = ("overlay",),
                   stacked: bool = False, scale_bar: bool = True,
                   scale_bar_length: Optional[float] = None,
                   colour_bars: bool = True, labels: bool = True,
                   projection=None, slab=None) -> List[Picture]:
    """The pictures of ``layers`` on ``fov``: one, or one per layer.

    ``what`` is one of `WHAT`: ``"view"`` the ``panels`` as the window lays
    them out (side by side, or ``stacked``), ``"composite"`` the visible
    layers added up, ``"layers"`` / ``"layers grey"`` a picture per visible
    layer in its colours or as its intensity in grey, ``"intensity"`` the
    summed intensity as 32-bit floats, for numbers rather than looks, with
    nothing drawn over it.  ``scale_bar_length`` None is the window's: a
    round fifth of the width.  ``projection`` and ``slab`` draw the 3D
    view's projection instead (`view3d.render_layer_3d`).
    """
    if what not in dict(WHAT):
        raise ValueError(f"what: one of {', '.join(dict(WHAT))}")
    shown = [i for i, layer in enumerate(layers) if layer.visible]
    if what == "view":
        specs = list(panels) or ["overlay"]
    elif what == "composite" or what == "intensity":
        specs = ["overlay"]
    else:
        specs = [i for i in shown if not (projection is not None and layers[i].is_image)]
    planes: Dict[int, Any] = {}
    draw = None if projection is None else _draw_3d(projection, slab)
    images, weight = render_panels(layers, fov, specs, planes=planes, draw=draw)
    unit = _unit(layers)
    if what == "intensity":
        return [Picture(np.ascontiguousarray(weight, np.float32), fov, unit=unit)]

    white = any(layers[i].display.white_background for i in shown)
    # a bar in nanometres means nothing on a picture of photons against frame
    bar = scale_bar and unit != "" and _positions(layers)
    length = 0.0
    if bar:
        length = scale_bar_length or auto_bar(fov.x1 - fov.x0)
    scales = colour_scales(layers) if colour_bars else []

    def decorate(rgb, label="", scales_=(), with_bar=True, grey=False):
        return _decorate(rgb, fov, label=label if labels else "",
                         bar=length if with_bar else 0.0, unit=unit,
                         scales=scales_, white=white and not grey)

    if what in ("layers", "layers grey"):
        out = []
        for spec, image in zip(specs, images):
            layer = layers[spec]
            if what == "layers grey":
                pixels = _grey(planes[spec], layer)
                own = []
            else:
                pixels = _bytes(image)
                own = [s for s in scales if layer.name in s[0]]
            pixels, px = decorate(pixels, layer.name, own, grey=what == "layers grey")
            out.append(Picture(pixels, fov, f"_layer{spec + 1}", unit, length if px else 0.0, px))
        return out

    named = labels and len(specs) > 1
    panels_out, drawn = [], 0
    # the window's bars: the scale bar on the first panel, the colour bars in
    # the top right corner of the whole picture
    corner = 0 if stacked else len(specs) - 1
    for n, (spec, image) in enumerate(zip(specs, images)):
        name = ("overlay" if spec == "overlay" else layers[spec].name) if named else ""
        pixels, px = decorate(_bytes(image), name, scales if n == corner else (),
                              with_bar=n == 0)
        drawn = drawn or px
        panels_out.append(pixels)
    return [Picture(_join(panels_out, stacked), fov, "", unit,
                    length if drawn else 0.0, drawn)]


def _draw_3d(projection, slab):
    """One layer through the 3D view's projection, as `render_3d` draws it on
    the CPU; an image layer is not in the 3D view."""
    from .view3d import render_layer_3d

    def draw(layer, fov):
        if layer.is_image:
            return None
        return render_layer_3d(layer.locs, layer.mask, projection, slab, fov,
                               layer.settings, layer.display,
                               n_threads=layer.n_threads, index=layer.index)
    return draw


def _unit(layers) -> str:
    """The x axis's unit, which is the pixel size's."""
    for layer in layers:
        if layer.visible and not layer.is_image and layer.settings is not None:
            try:
                return axis_unit(layer.settings.axes.names(layer.locs)[0])
            except KeyError:
                return ""
    return "nm"


def _positions(layers) -> bool:
    """Whether the picture is of a place, x against y: a scale bar is then
    one length on both axes."""
    return all(layer.settings.axes.is_default for layer in layers
               if layer.visible and not layer.is_image and layer.settings is not None)


def _bytes(rgb: np.ndarray) -> np.ndarray:
    """[0, 1] to 0..255, rounded, as `render.save_image` writes it."""
    return np.round(np.clip(np.asarray(rgb, np.float32), 0, 1) * 255).astype(np.uint8)


def _grey(rendered, layer: LayerSnapshot) -> np.ndarray:
    """A layer's intensity as grey bytes: its brightness scaled as its
    display scales it (contrast, or a set maximum, and gamma), without its
    colours -- for a channel to take apart elsewhere."""
    display = layer.display
    norm, _ = normalize(rendered.weight, display.imax, display.contrast)
    if display.gamma != 1.0:
        norm = norm ** display.gamma
    return np.round(norm * 255).astype(np.uint8)


def _join(panels: List[np.ndarray], stacked: bool) -> np.ndarray:
    if len(panels) == 1:
        return panels[0]
    h, w = panels[0].shape[:2]
    shape = (PANEL_LINE, w) if stacked else (h, PANEL_LINE)
    line = np.full(shape + panels[0].shape[2:], 128, np.uint8)
    joined = []
    for n, panel in enumerate(panels):
        joined += ([line] if n else []) + [panel]
    return np.concatenate(joined, axis=0 if stacked else 1)


# ---------------------------------------------------------- drawn over it
def _font(size: int):
    """DejaVu Sans, which matplotlib ships and which has a µ -- Pillow's own
    font writes "2 µm" with a box for the µ."""
    from PIL import ImageFont
    try:
        import matplotlib
        path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
        return ImageFont.truetype(path, size)
    except Exception:                    # pragma: no cover - a bare install
        pass
    try:
        return ImageFont.load_default(size=size)            # Pillow >= 10.1
    except TypeError:                    # pragma: no cover
        return ImageFont.load_default()


def text_size(height: int, width: int) -> int:
    """The labels' height in pixels: about what the window's are on a picture
    of a window's size, growing with a larger one so a 4000-pixel export does
    not carry specks for labels."""
    return max(11, int(round(min(height, width) / 45)))


def _decorate(pixels: np.ndarray, fov: FieldOfView, label: str, bar: float, unit: str,
              scales, white: bool) -> Tuple[np.ndarray, int]:
    """Draw the label, the scale bar and the colour bars onto 8-bit pixels.
    Returns them and the bar's length in pixels (0: none drawn)."""
    n_bar = int(round(bar / fov.pixelsize)) if bar > 0 else 0
    h, w = pixels.shape[:2]
    if n_bar >= w:
        n_bar = 0                        # a bar wider than the picture says nothing
    if not (label or n_bar or len(scales)):
        return pixels, 0
    from PIL import Image, ImageDraw
    grey = pixels.ndim == 2
    image = Image.fromarray(pixels, "L" if grey else "RGB")
    draw = ImageDraw.Draw(image)
    size = text_size(h, w)
    font = _font(size)
    ink = 255 if grey else ((0, 0, 0) if white else (255, 255, 255))
    margin = max(4, size // 2)
    if label:
        draw.text((margin, margin // 2), label, fill=ink, font=font)
    if n_bar:
        thick = max(2, size // 4)
        x1, y1 = w - 2 * margin, h - 2 * margin
        x0 = x1 - n_bar
        draw.rectangle([x0, y1 - thick + 1, x1 - 1, y1], fill=ink)
        draw.text(((x0 + x1) / 2, y1 - thick - 2), bar_label(bar, unit),
                  fill=ink, font=font, anchor="md")
    if len(scales) and not grey:
        image = _colour_bars(image, scales, size, font, white)
    return np.asarray(image), n_bar


def stamp(pixels: np.ndarray, text: str, white: bool = False) -> np.ndarray:
    """``text`` written in the bottom left corner, the size the labels are:
    a movie's time or depth on each of its frames."""
    if not text:
        return pixels
    from PIL import Image, ImageDraw
    grey = pixels.ndim == 2
    image = Image.fromarray(np.ascontiguousarray(pixels), "L" if grey else "RGB")
    h, w = pixels.shape[:2]
    size = text_size(h, w)
    ink = 255 if grey else ((0, 0, 0) if white else (255, 255, 255))
    margin = max(4, size // 2)
    ImageDraw.Draw(image).text((margin, h - margin), text, fill=ink,
                               font=_font(size), anchor="ls")
    return np.asarray(image)


def _colour_bars(image, scales, size: int, font, white: bool):
    """The window's colour bars, stacked in the top right corner: on a
    darkened (or, on white paper, lightened) box, the LUT upwards, the
    column's range at its ends, whose it is above it."""
    from PIL import Image, ImageDraw
    w = image.width
    box_w, box_h, gap = 9 * size, 9 * size, size // 3
    bar_w = max(4, size)
    base = np.asarray(image).astype(np.float32)
    ground = 255.0 if white else 0.0
    ink = (0, 0, 0) if white else (230, 230, 230)
    boxes = []
    for n in range(len(scales)):
        x0, y0 = w - box_w - size // 2, size // 2 + n * (box_h + gap)
        boxes.append((x0, y0))
        if y0 + box_h > base.shape[0] or x0 < 0:
            boxes.pop()
            break
        # 140 of 255, as the window's box
        sub = base[y0:y0 + box_h, x0:x0 + box_w]
        sub[:] = sub * (1 - 0.55) + ground * 0.55
    image = Image.fromarray(np.round(base).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    for (x0, y0), (names, field, range_, table) in zip(boxes, scales):
        title, top, bottom = colour_bar_labels(names, field, range_)
        line = int(size * 1.25)
        bx, by0, by1 = x0 + size // 2, y0 + line + size // 3, y0 + box_h - size // 3
        n = max(by1 - by0, 1)
        index = np.linspace(len(table) - 1, 0, n).round().astype(int)   # the top at the top
        column = np.round(np.clip(table[index], 0, 1) * 255).astype(np.uint8)
        strip = np.repeat(column[:, None, :], bar_w, axis=1)
        image.paste(Image.fromarray(np.ascontiguousarray(strip), "RGB"), (bx, by0))
        draw.text((x0 + size // 3, y0 + size // 6), _fit(draw, title, font, box_w - size),
                  fill=ink, font=font)
        tx = bx + bar_w + size // 3
        draw.text((tx, by0), _fit(draw, top, font, x0 + box_w - tx), fill=ink, font=font)
        draw.text((tx, by1), _fit(draw, bottom, font, x0 + box_w - tx), fill=ink,
                  font=font, anchor="ls")
    return image


def _fit(draw, text: str, font, width: float) -> str:
    """The text, cut with an ellipsis to ``width`` pixels."""
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


# ---------------------------------------------------------------- the file
def file_paths(path, pictures: Sequence[Picture]) -> List[Path]:
    """Where each picture goes: the path, with a picture's suffix before the
    extension; a float picture as a TIFF whatever the extension, since PNG
    and JPEG have no float pixels."""
    path = Path(path)
    out = []
    for picture in pictures:
        target = path.with_name(path.stem + picture.suffix + path.suffix)
        if picture.image.dtype.kind == "f" and target.suffix.lower() not in (".tif", ".tiff"):
            target = target.with_suffix(".tif")
        out.append(target)
    return out


def write_picture(picture: Picture, path, description: str = "") -> Path:
    """Write one picture: the format by the extension (`FORMATS`), the pixel
    size in the resolution tags when it is a length, ``description`` as the
    TIFF's description, the PNG's ``Description`` text or the JPEG's
    comment."""
    path = Path(path)
    kind = FORMATS.get(path.suffix.lower())
    if kind is None:
        raise ValueError(f"{path.name}: the extension says the format -- "
                         f"{', '.join(sorted(FORMATS))}")
    if picture.image.dtype.kind == "f" and kind != "TIFF":
        raise ValueError("a float picture is written as a TIFF")
    nm = {"nm": 1.0, "µm": 1000.0, "um": 1000.0}.get(picture.unit)
    pixel_nm = picture.fov.pixelsize * nm if nm else None
    image = np.ascontiguousarray(picture.image)
    if kind == "TIFF":
        import tifffile
        extra = {}
        if pixel_nm:
            # SMAP's: pixels per centimetre, which ImageJ reads as µm
            per_cm = 1e7 / pixel_nm
            extra = dict(resolution=(per_cm, per_cm), resolutionunit="CENTIMETER")
        tifffile.imwrite(str(path), image, photometric="rgb" if image.ndim == 3
                         else "minisblack", description=description or None,
                         metadata=None, **extra)
        return path
    from PIL import Image
    pil = Image.fromarray(image, "RGB" if image.ndim == 3 else "L")
    if kind == "PNG":
        from PIL.PngImagePlugin import PngInfo
        info = PngInfo()
        if description:
            info.add_text("Description", description)
        extra = {"pnginfo": info}
        # pHYs holds pixels per metre in 32 bits: from 0.25 nm up
        if pixel_nm and 1e9 / pixel_nm < 2 ** 32 - 1:
            extra["dpi"] = (2.54e7 / pixel_nm,) * 2
        pil.save(path, **extra)
    else:
        # JPEG's density is 16 bits per inch: no room for nanometres
        pil.save(path, quality=95, comment=description.encode() if description else b"")
    return path


def read_description(path) -> Dict[str, Any]:
    """What `describe_view` wrote into a picture, read back: {} if nothing."""
    import yaml
    path = Path(path)
    kind = FORMATS.get(path.suffix.lower())
    text = None
    if kind == "TIFF":
        import tifffile
        with tifffile.TiffFile(str(path)) as tif:
            text = tif.pages[0].description
            if tif.is_imagej:            # a stack: ImageJ's own, ours in *Info*
                text = (tif.imagej_metadata or {}).get("Info")
    else:
        from PIL import Image
        with Image.open(path) as image:
            text = image.info.get("Description") if kind == "PNG" else image.info.get("comment")
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    if not text:
        return {}
    try:
        found = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    return found if isinstance(found, dict) else {}


def pixel_size_of(path) -> Optional[float]:
    """The pixel size a picture's resolution tags give, in nm; None without."""
    path = Path(path)
    if FORMATS.get(path.suffix.lower()) == "TIFF":
        import tifffile
        with tifffile.TiffFile(str(path)) as tif:
            tags = tif.pages[0].tags
            if "XResolution" not in tags:
                return None
            num, den = tags["XResolution"].value
            unit = int(tags["ResolutionUnit"].value) if "ResolutionUnit" in tags else 2
            per = {2: 2.54e7, 3: 1e7}.get(unit)
            return per * den / num if per else None
    from PIL import Image
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
    return 2.54e7 / float(dpi[0]) if dpi else None


# ----------------------------------------------------------- the description
def describe_view(session, layers: Sequence[LayerSnapshot], settings: Any,
                  picture: Picture, frame: Optional[ViewFrame] = None) -> Dict[str, Any]:
    """How the picture was made, as SMAP's *TifSaver* wrote it: the pixel
    size and field, the export's settings, the files, the merging, and per
    visible layer its files, filter, grouping and render and display
    settings.  Plain data, for `description_text`."""
    from . import __version__
    fov = picture.fov
    out: Dict[str, Any] = {
        "software": f"SMAPpy {__version__}",
        "pixelsize": fov.pixelsize, "unit": picture.unit,
        "x0": fov.x0, "y0": fov.y0, "nx": fov.nx, "ny": fov.ny,
        "scale_bar": picture.scale_bar,
        "export": asdict(settings) if is_dataclass(settings) else dict(settings or {}),
    }
    if frame is not None and frame.projection is not None:
        out["projection"] = frame.projection.to_dict()
        out["slab"] = frame.slab.to_dict() if frame.slab is not None else None
    if session is not None:
        out["files"] = [str(getattr(f, "path", "") or getattr(f, "name", ""))
                        for f in getattr(session, "files", [])]
        group = getattr(session, "group_settings", None)
        if group is not None and is_dataclass(group):
            out["grouping"] = asdict(group)
    described = []
    session_layers = list(getattr(session, "layers", []) or [])
    for i, snap in enumerate(layers):
        if not snap.visible:
            continue
        entry: Dict[str, Any] = {"number": i + 1, "name": snap.name}
        if snap.is_image:
            entry["image"] = str(getattr(snap.image_data, "name", "") or "")
            entry["frame"] = snap.frame
        else:
            entry["render"] = asdict(snap.settings)
            entry["n_shown"] = int(np.count_nonzero(snap.mask)) if snap.mask is not None else None
        entry["display"] = asdict(snap.display)
        if i < len(session_layers) and not session_layers[i].is_image:
            layer = session_layers[i]
            entry["grouped"] = bool(layer.grouped)
            entry["files"] = layer.files
            entry["filter"] = {k: list(v) for k, v in layer.ranges.items()}
        described.append(entry)
    out["layers"] = described
    return _plain(out)


def description_text(described: Dict[str, Any]) -> str:
    import yaml
    return yaml.safe_dump(described, sort_keys=False, allow_unicode=True,
                          default_flow_style=None, width=100)


def _plain(value):
    """Numbers, strings, lists and dicts: what YAML writes safely."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else str(value)
    if value is None or isinstance(value, (int, str)):
        return value
    return str(value)
