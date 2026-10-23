"""The ROIs as pictures: a page of them as a PDF, or one per plane of a TIFF.

SMAP's ROI manager could show the sites one by one and save each; a figure
of a hundred NPCs, or a gallery to look through before trusting a
segmentation, needs them all on a page, at one scale.  Here every ROI of
the ROI manager is drawn by the renderer the view uses (`render_locs`,
through each layer's `LayerSnapshot`), on a square of one size and pixel
around its centre -- so the pictures compare, which a picture fitted to
each ROI's own extent would not.

**The layers.**  The visible localization layers, as they show the data:
their filters, merged blinks, colours and contrast.  An ROI belongs to a
file, and its picture holds that file's rows only.  The contrast is the
layer's: by its quantile, each picture is scaled by its own brightest
pixels, as the view is; set the layer to an absolute scale to compare
brightness between ROIs.

**Turned.**  With *turn to* a named line, each picture is turned so that
the ROI's line of that name points to the right: the localizations are
turned about the centre before they are rendered, not the image after, so
no pixel is resampled.  An ROI without that line is drawn as it lies.  An
image layer cannot be turned that way and is left out of turned pictures.

**On top.**  The ROI's outline and its named lines, and as a title its
number and the chosen values of what was measured in it -- read as every
summary reads results (`SiteAnalysisPlugin.rows`), so only the ROIs whose
results still describe them have numbers.

**Written.**  A PDF (matplotlib's ``PdfPages``): a grid of *columns* x
*rows* per page, the outline and lines as vector paths over each picture,
the scale bar's length at the foot of each page.  A TIFF: an RGB stack, one
plane per ROI with the outline and lines drawn in, the pixel size in its
resolution tags and the ROI numbers in its description -- what Fiji opens
as a stack to step through.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import Context, FormState, Notice, Result, param, register
from .roi import SiteAnalysisPlugin, auto_only

PAGES = {"A4": (8.27, 11.69), "letter": (8.5, 11.0)}
OUTLINE_COLOUR = (255, 255, 0)          # as the ROI manager draws an ROI
LINE_COLOURS = ((0, 200, 255), (255, 80, 200), (120, 255, 120), (255, 160, 0))
NO_LINE = ""                            # *turn to*: as the ROI lies


@dataclass
class RoiPicture:
    """One ROI's picture and what is drawn over it, in its own pixels."""
    number: int
    image: np.ndarray                    # (h, w, 3) uint8
    pixelsize: float
    outline: Optional[np.ndarray] = None   # (n, 2) pixel x, y
    lines: Dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    title: str = ""


# ------------------------------------------------------------- the geometry

def line_angle(line) -> Optional[float]:
    """The direction of a named line ``[[x0, y0], [x1, y1]]``, in radians."""
    if line is None:
        return None
    (x0, y0), (x1, y1) = np.asarray(line, float)
    if x0 == x1 and y0 == y1:
        return None
    return float(np.arctan2(y1 - y0, x1 - x0))


def to_picture(xy, centre, angle: float, half: float, pixelsize: float) -> np.ndarray:
    """Points in nm to the picture's pixels: turned by ``-angle`` about
    ``centre`` (so a line at ``angle`` lies along the picture's x), then
    counted from the picture's top left corner, y down as the view draws."""
    xy = np.asarray(xy, float).reshape(-1, 2) - np.asarray(centre, float)
    c, s = np.cos(-angle), np.sin(-angle)
    u = c * xy[:, 0] - s * xy[:, 1]
    v = s * xy[:, 0] + c * xy[:, 1]
    return np.column_stack(((u + half) / pixelsize, (v + half) / pixelsize))


def roi_layers(layers, centre, half: float, angle: float = 0.0,
               filenumber: Optional[int] = None, pixelsize: float = 1.0):
    """The layers cut to what can reach the square of ``2 * half`` round
    ``centre`` -- and to file ``filenumber`` -- with their positions turned
    by ``-angle`` about it: snapshots to render the ROI's picture from.

    The cut is the speed: a gallery of a thousand ROIs renders a thousand
    small tables rather than the whole one a thousand times.  A turned
    table has no spatial index (it is of the unturned positions), which a
    table this small does not need.
    """
    from ..locs import Localizations
    out = []
    cx, cy = float(centre[0]), float(centre[1])
    for snap in layers:
        if not snap.visible:
            out.append(snap)
            continue
        if snap.locs is None:
            # an image layer is resampled, not turned: left out of a turned picture
            out.append(snap if not angle else dataclasses.replace(snap, visible=False))
            continue
        locs = snap.locs
        x_name, y_name = snap.settings.axes.names(locs)
        reach = half * np.sqrt(2.0) + 3.0 * snap.max_sigma(pixelsize)
        if snap.index is not None:
            rows = snap.index.query(cx - reach, cx + reach, cy - reach, cy + reach, 0.0)
            rows = rows[rows < len(snap.mask)]
            rows = rows[np.asarray(snap.mask)[rows]]
        else:
            rows = snap.indices
        x = np.asarray(locs[x_name])[rows]
        y = np.asarray(locs[y_name])[rows]
        near = (np.abs(x - cx) <= reach) & (np.abs(y - cy) <= reach)
        if filenumber is not None and "filenumber" in locs:
            near &= np.asarray(locs["filenumber"])[rows] == filenumber
        rows = rows[near]
        columns = {k: np.asarray(v)[rows] for k, v in locs.columns.items()}
        if angle:
            c, s = np.cos(-angle), np.sin(-angle)
            dx, dy = x[near] - cx, y[near] - cy
            columns[x_name] = (cx + c * dx - s * dy).astype(columns[x_name].dtype)
            columns[y_name] = (cy + s * dx + c * dy).astype(columns[y_name].dtype)
        cut = Localizations(columns, dict(locs.metadata))
        n = len(cut)
        out.append(dataclasses.replace(snap, locs=cut, mask=np.ones(n, bool),
                                       known_indices=np.arange(n), index=None, memo={}))
    return tuple(out)


def render_roi(layers, centre, size_nm: float, pixelsize: float, angle: float = 0.0,
               filenumber: Optional[int] = None) -> np.ndarray:
    """The picture of a square of ``size_nm`` round ``centre``, as the
    visible layers show it: (h, w, 3) uint8."""
    from ..layer_state import render_panels
    from ..picture import _bytes
    from ..render import FieldOfView
    half = size_nm / 2
    fov = FieldOfView.from_range((centre[0] - half, centre[0] + half),
                                 (centre[1] - half, centre[1] + half), pixelsize)
    cut = roi_layers(layers, centre, half, angle, filenumber, pixelsize)
    images, _ = render_panels(cut, fov, ["overlay"])
    return _bytes(images[0])


# ---------------------------------------------------------------- the files

def page_grid(n: int, columns: int, rows: int) -> List[List[int]]:
    """The pictures' positions, a list per page."""
    per = max(1, columns * rows)
    return [list(range(a, min(a + per, n))) for a in range(0, n, per)]


def write_pdf(path, pictures: Sequence[RoiPicture], columns: int = 4, rows: int = 6,
              page: str = "A4", scale_bar_nm: float = 0.0, heading: str = "") -> Path:
    """A grid of the pictures per page, the overlays drawn as vector paths
    on top, the scale bar on each picture and its length at the page's foot."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = PAGES.get(page, PAGES["A4"])
    pages = page_grid(len(pictures), columns, rows)
    with PdfPages(path) as pdf:
        for p, members in enumerate(pages):
            fig = Figure(figsize=(width, height))
            # the grid fills the page bar a margin, a line at the top and the foot
            fig.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.04,
                                wspace=0.08, hspace=0.22)
            axes = fig.subplots(rows, columns, squeeze=False).ravel()
            for ax in axes:
                ax.set_axis_off()
            for ax, k in zip(axes, members):
                draw_picture(ax, pictures[k], scale_bar_nm)
            foot = f"page {p + 1} of {len(pages)}"
            if scale_bar_nm:
                foot = f"scale bar {scale_bar_nm:g} nm   " + foot
            fig.text(0.96, 0.015, foot, ha="right", fontsize=7, color="0.3")
            if heading:
                fig.text(0.04, 0.975, heading, ha="left", fontsize=8, color="0.2")
            pdf.savefig(fig)
        info = pdf.infodict()
        info["Title"] = heading or path.stem
        info["Creator"] = "smappy"
    return path


def draw_picture(ax, pic: RoiPicture, scale_bar_nm: float = 0.0) -> None:
    """One picture into a matplotlib axis, its overlays and its title."""
    h, w = pic.image.shape[:2]
    ax.imshow(pic.image, extent=(0, w, h, 0), interpolation="nearest")
    if pic.outline is not None and len(pic.outline):
        ax.plot(pic.outline[:, 0], pic.outline[:, 1], color=np.divide(OUTLINE_COLOUR, 255),
                linewidth=0.6)
    for k, (name, xy) in enumerate(pic.lines.items()):
        ax.plot(xy[:, 0], xy[:, 1], color=np.divide(LINE_COLOURS[k % len(LINE_COLOURS)], 255),
                linewidth=0.8)
    if scale_bar_nm:
        n = scale_bar_nm / pic.pixelsize
        ax.plot([w * 0.95 - n, w * 0.95], [h * 0.93, h * 0.93], color="white", linewidth=1.5,
                solid_capstyle="butt")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(pic.title, fontsize=6, pad=2)


def burn_in(pic: RoiPicture, scale_bar_nm: float = 0.0) -> np.ndarray:
    """The picture with its outline, lines and scale bar drawn into the
    pixels (Pillow): a TIFF has no layer to draw them on."""
    from PIL import Image, ImageDraw
    image = Image.fromarray(np.ascontiguousarray(pic.image))
    draw = ImageDraw.Draw(image)
    if pic.outline is not None and len(pic.outline) > 1:
        draw.line([tuple(p) for p in pic.outline], fill=OUTLINE_COLOUR, width=1)
    for k, xy in enumerate(pic.lines.values()):
        draw.line([tuple(p) for p in xy], fill=LINE_COLOURS[k % len(LINE_COLOURS)], width=1)
    if scale_bar_nm:
        h, w = pic.image.shape[:2]
        n = scale_bar_nm / pic.pixelsize
        y = int(round(h * 0.93))
        draw.rectangle([w * 0.95 - n, y - 1, w * 0.95, y + 1], fill=(255, 255, 255))
    return np.asarray(image)


def write_tiff(path, pictures: Sequence[RoiPicture], scale_bar_nm: float = 0.0) -> Path:
    """An RGB stack, one plane per ROI, written as ImageJ writes one: the
    pixel size in µm, and each plane labelled with its ROI's title, which
    Fiji shows above the plane."""
    import tifffile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.stack([burn_in(p, scale_bar_nm) for p in pictures])
    px_um = pictures[0].pixelsize / 1000.0
    tifffile.imwrite(path, stack, imagej=True, photometric="rgb",
                     resolution=(1 / px_um, 1 / px_um),
                     metadata={"unit": "um", "axes": "ZYXS",
                               "Labels": [p.title for p in pictures]})
    return path


# ---------------------------------------------------------------- the plugin

def _line_choices(ctx) -> List[Tuple[str, str]]:
    project = getattr(ctx, "rois", None) if ctx is not None else None
    names = list(getattr(project, "line_names", []) or [])
    return [(NO_LINE, "none: as it lies")] + [(n, n) for n in names]


@dataclass
class RoiImagesSettings:
    path: str = param("", label="file", kind="save_file",
                      file_filter="PDF (*.pdf);;TIFF stack (*.tif)",
                      help=".pdf: a page of pictures; .tif: a stack, one ROI per plane")
    size_nm: Optional[float] = param(None, label="size", unit="nm", min=1.0,
                                     help="the square each picture shows round its "
                                          "ROI's centre; auto: the ROI manager's image")
    pixelsize_nm: float = param(2.0, label="pixel", unit="nm", min=0.01,
                                help="of every picture")
    rois: str = param("used", label="ROIs", choices=(("used", "used (ticked)"),
                                                      ("all", "all")),
                      help="which ROIs, in the ROI manager's order")
    turn_to: str = param(NO_LINE, label="turn to", choices=lambda: [(NO_LINE, "none")],
                         help="turn each picture so the ROI's line of this name "
                              "points right")
    outline: bool = param(True, label="outline", help="draw each ROI's outline")
    lines: bool = param(True, label="lines", help="draw each ROI's named lines")
    scale_bar_nm: Optional[float] = param(None, label="scale bar", unit="nm", min=0.0,
                                          help="its length; auto: a round fifth of "
                                               "the picture; 0: none")
    columns: int = param(4, label="columns", min=1, max=20,
                         help="pictures across a PDF page")
    rows: int = param(6, label="rows", min=1, max=30,
                      help="pictures down a PDF page")
    page: str = param("A4", label="page", choices=(("A4", "A4"), ("letter", "Letter")),
                      help="the PDF's page size")
    evaluation: str = param("", label="results from", choices=auto_only,
                            help="the measurement whose values go in the titles; "
                                 "auto: everything measured on the ROIs")
    title_columns: list = param(default_factory=list, label="title", kind="columns",
                                help="values in each title after the ROI's number; "
                                     "none ticked: the number alone")


@register("File/Export/ROI images")
class RoiImages(SiteAnalysisPlugin):
    """Every ROI as a picture at one scale, its outline and measured values
    on it: a PDF gallery or a TIFF stack."""

    Settings = RoiImagesSettings
    name = "Save ROI images"
    changes_locs = False
    logged = True          # where the pictures went is not visible in the table
    version = "1"
    auto_label = "all on the ROIs"

    def evaluation(self, ctx: Context, settings):
        from .roi import choose_evaluation
        if settings.evaluation:
            return choose_evaluation(ctx, "", settings.evaluation)
        return None

    def preflight(self, ctx: Context, settings):
        # titles are optional, so which evaluation is asked only when one is
        # used for them
        if not settings.title_columns:
            return None
        return super().preflight(ctx, settings)

    def describe(self, ctx: Optional[Context], settings: RoiImagesSettings) -> FormState:
        state = super().describe(ctx, settings)
        pdf = Path(settings.path).suffix.lower() != ".tif" and \
            Path(settings.path).suffix.lower() != ".tiff"
        for name in ("columns", "rows", "page"):
            state.active[name] = pdf
        if ctx is None or ctx.rois is None:
            return state
        state.choices["turn_to"] = _line_choices(ctx)
        project = ctx.rois
        state.hints["size_nm"] = f"{project.preview_width:g}"
        try:
            state.choices["title_columns"] = [
                c for c in _columns(self.rows(ctx, settings)[0]) if c not in ("roi_id",
                                                                                "file_id")]
        except ValueError:
            pass
        n = len(_chosen(project, settings))
        size = settings.size_nm or project.preview_width
        px = int(round(size / settings.pixelsize_nm))
        if n:
            pages = len(page_grid(n, settings.columns, settings.rows))
            where = f", {pages} page{'s' * (pages > 1)}" if pdf else ""
            state.notice = Notice(f"{n} ROIs of {px} x {px} pixels{where}", kind="info")
        return state

    def run(self, ctx: Context, settings: RoiImagesSettings) -> Result:
        from ..layer_state import snapshot_layers
        from ..picture import auto_bar
        if not settings.path:
            raise ValueError("choose where to save")
        project, session = ctx.rois, ctx.session
        if project is None or session is None:
            raise ValueError("no ROIs: segment or draw them in the ROI manager first")
        if not session.axes().is_default:
            raise ValueError("the picture is drawn on other axes than the positions: "
                             "set the Render tab's axes back to x and y")
        chosen = _chosen(project, settings)
        if not chosen:
            raise ValueError("no ROI to draw" + (": none is ticked to use"
                                                 if settings.rois == "used" else ""))
        layers = snapshot_layers(session.layers)
        if not any(l.visible and not l.is_image for l in layers):
            raise ValueError("no localization layer is visible")
        size = float(settings.size_nm or project.preview_width)
        bar = (auto_bar(size) if settings.scale_bar_nm is None
               else float(settings.scale_bar_nm))
        values = {}
        if settings.title_columns:
            rows, _ = self.rows(ctx, settings)
            values = {str(r.get("roi_id")): r for r in rows}
        pictures = []
        for k, roi in enumerate(chosen):
            if ctx.stopped:
                break
            if k % 10 == 0:
                ctx.report(f"ROI images: {k + 1} of {len(chosen)}")
            pictures.append(roi_picture(project, layers, roi, size, settings,
                                        values.get(str(roi.number))))
        path = Path(settings.path)
        if path.suffix.lower() in (".tif", ".tiff"):
            written = write_tiff(path, pictures, bar)
        else:
            if path.suffix.lower() != ".pdf":
                path = path.with_suffix(".pdf")
            written = write_pdf(path, pictures, settings.columns, settings.rows,
                                settings.page, bar, heading=session.display_name)
        if not pictures:
            raise ValueError("stopped before the first ROI")
        return Result(text=f"{len(pictures)} ROI images to {written}",
                      written=str(written), settings=settings,
                      stopped=len(pictures) < len(chosen),
                      data={"pictures": pictures, "path": str(written)})


def _chosen(project, settings: RoiImagesSettings):
    """The ROIs to draw, in the ROI manager's order."""
    rois = [project.rois[i] for i in project.ordered_ids() if i in project.rois]
    return [r for r in rois if r.use or settings.rois == "all"]


def _columns(rows) -> List[str]:
    return list(dict.fromkeys(k for row in rows for k in row))


def _title(number: int, row: Optional[dict], columns: Sequence[str]) -> str:
    parts = [f"ROI {number}"]
    for c in columns:
        v = None if row is None else row.get(c)
        if v is None:
            continue
        if isinstance(v, (float, np.floating)):
            parts.append(f"{c} {v:.3g}")
        elif isinstance(v, (int, np.integer, str)):
            parts.append(f"{c} {v}")
    return ", ".join(parts)


def roi_picture(project, layers, roi, size_nm: float, settings: RoiImagesSettings,
                row: Optional[dict] = None) -> RoiPicture:
    """One ROI's picture with its overlays in its own pixels."""
    centre = (np.asarray(roi.center, float) if roi.spatial
              else project.extent(roi)[0])
    angle = 0.0
    if settings.turn_to:
        angle = line_angle(roi.lines.get(settings.turn_to)) or 0.0
    source = project.sources.get(roi.file_id)
    filenumber = source.number if source is not None and len(project.sources) > 1 else None
    px = float(settings.pixelsize_nm)
    image = render_roi(layers, centre, size_nm, px, angle, filenumber)
    half = size_nm / 2
    outline = None
    if settings.outline:
        xs, ys = project.outline(roi)
        outline = to_picture(np.column_stack((xs, ys)), centre, angle, half, px)
    lines = ({name: to_picture(xy, centre, angle, half, px)
              for name, xy in roi.lines.items()} if settings.lines else {})
    return RoiPicture(number=roi.number, image=image, pixelsize=px, outline=outline,
                      lines=lines, title=_title(roi.number, row, settings.title_columns))
