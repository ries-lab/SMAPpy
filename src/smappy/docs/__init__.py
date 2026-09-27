"""Each plugin's page: what it does, how, and every setting it has.

A page is written by hand -- the explanation, the theory, the references --
and completed from the plugin itself, so the parts that can go out of date by
themselves never do:

* the title, where the plugin sits in the menus, its version and the one-line
  summary come from the class (`Plugin.description`);
* the parameter table comes from `Plugin.specs()` -- label, default, unit,
  bounds, choices and the ``help`` that is also the tooltip -- with a page's
  own ``### name`` notes under *Parameters* merged into the rows;
* a ```` ```figure ```` block is Python that draws with the plugin's own
  functions, run when the page is shown, so a figure shows what the code does
  today rather than what it did when somebody took a screenshot.

What cannot be generated is the prose, and for that the page records which
`Plugin.version` it describes.  A version is bumped when a change moves the
numbers, which is exactly when an explanation of the algorithm may have
become wrong, and ``tests/test_plugin_docs.py`` fails until the page has been
re-read and its version brought along.  A refactor that changes nothing a
user could see is not caught, and should not be.

Where a page lives: ``<root>/docs/<plugin path>.md`` for any plugin root --
the shipped pages are ``src/smappy/plugins/docs/`` -- or, for a plugin
dropped in as a single file, ``<file>.md`` beside ``<file>.py``.  A plugin
with no page still gets one, generated: the summary and the parameters.

The page format::

    ---
    version: "1"                          # the Plugin.version described
    covers: [smappy.rcc.estimate_drift_rcc]   # the code the prose explains
    ---
    ## What it does
    ...  $\\sigma = S/\\sqrt{N}$  ...

    ```figure What the caption says.
    fig.set_size_inches(6, 2.5)           # `fig` is a matplotlib Figure
    ax = fig.subplots()
    ...
    ```

    ## Parameters
    ### max_drift_nm
    A longer note than the tooltip, shown in the row.

A ```` ```figure-setup ```` block is run once for all the page's figures --
the simulation they draw from -- and only when one of them is not cached.

``covers`` names the functions whose behaviour the prose describes.  They must
exist (a test checks), and their modules' source is part of a figure's cache
key, so a figure is redrawn when the code under it changes.

`render` gives HTML plus the images it refers to, as bytes, and knows nothing
about Qt: the Help window hands the images to its document, and
``python -m smappy.docs`` writes them next to the HTML.
"""
from __future__ import annotations

import dataclasses
import hashlib
import html
import importlib
import io
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .markup import Converter

DOCS_DIR = "docs"
# a figure's source, hashed with the code it draws; bump to redraw them all
_FIGURE_CACHE_VERSION = "1"


# ---------------------------------------------------------------------- pages
@dataclass
class Page:
    """A page as written: its front matter and its Markdown body."""
    path: Optional[Path]
    body: str = ""
    version: Optional[str] = None
    covers: List[str] = field(default_factory=list)

    @property
    def written(self) -> bool:
        return self.path is not None


def page_file(plugin_path: str, origin: Optional[Path] = None) -> Optional[Path]:
    """The Markdown file documenting ``plugin_path``, or None.

    Looked for under every plugin root, the user's first so a user can
    rewrite a shipped page as they can shadow a shipped plugin, then beside the
    plugin's own file.
    """
    from .. import plugins
    for _, root in reversed(plugins.roots()):
        candidate = Path(root) / DOCS_DIR / f"{plugin_path}.md"
        if candidate.is_file():
            return candidate
    if origin is not None and Path(origin).suffix == ".py":
        beside = Path(origin).with_suffix(".md")
        if beside.is_file():
            return beside
    return None


def read_page(file: Optional[Path]) -> Page:
    """Split front matter from body.  A missing file is an empty page."""
    if file is None:
        return Page(None)
    text = Path(file).read_text(encoding="utf-8")
    meta: Dict[str, Any] = {}
    match = re.match(r"---\s*\n(.*?)\n---\s*\n", text, flags=re.S)
    if match:
        import yaml
        meta = yaml.safe_load(match.group(1)) or {}
        text = text[match.end():]
    version = meta.get("version")
    covers = meta.get("covers") or []
    if isinstance(covers, str):
        covers = [covers]
    return Page(Path(file), text, None if version is None else str(version),
                [str(c) for c in covers])


def page_for(plugin_cls, origin: Optional[Path] = None) -> Page:
    if origin is None:
        origin = _origin(plugin_cls)
    return read_page(page_file(plugin_cls.path, origin))


def _origin(plugin_cls) -> Optional[Path]:
    import sys
    file = getattr(sys.modules.get(plugin_cls.__module__), "__file__", None)
    return Path(file) if file else None


def resolve(dotted: str):
    """``"smappy.rcc.estimate_drift_rcc"`` -> the object, or raise."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for name in parts[cut:]:
            obj = getattr(obj, name)
        return obj
    raise ImportError(dotted)


# ------------------------------------------------------------ generated parts
def _format_default(spec) -> str:
    value = spec.default
    info = spec.info
    if value is None:
        return "auto" if spec.optional else "&ndash;"
    if isinstance(value, bool):
        return "on" if value else "off"
    choices = info.choices
    if callable(choices):
        try:
            choices = choices()
        except Exception:
            choices = None
    if choices:
        for choice in choices:
            if isinstance(choice, (tuple, list)) and choice[0] == value:
                return html.escape(str(choice[1]))
    if isinstance(value, str):
        return html.escape(value) if value else "&ndash;"
    if isinstance(value, float):
        text = f"{value:g}"
    elif isinstance(value, (list, tuple)):
        text = ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in value)
    elif dataclasses.is_dataclass(value):
        return "&ndash;"
    else:
        text = str(value)
    return html.escape(text + (f" {info.unit}" if info.unit else ""))


def _bounds(info) -> str:
    unit = f" {info.unit}" if info.unit else ""
    if info.min == 0 and info.max is None:
        return ""                      # "not negative" goes without saying
    if info.min is not None and info.max is not None:
        return f"{info.min:g} to {info.max:g}{unit}"
    if info.min is not None:
        return f"at least {info.min:g}{unit}"
    if info.max is not None:
        return f"at most {info.max:g}{unit}"
    return ""


def _choices(info) -> str:
    if not info.choices or callable(info.choices):
        return ""
    labels = [str(c[1]) if isinstance(c, (tuple, list)) else str(c)
              for c in info.choices]
    return "; ".join(labels)


def parameter_notes(body: str) -> Tuple[str, Dict[str, str]]:
    """Take the ``### name`` notes out of a page's *Parameters* section.

    Returns the body without them, and the notes by dotted field name.  The
    rest of the section -- an introduction before the first note -- stays
    where it was, and the table goes after it.
    """
    match = re.search(r"^##\s+Parameters\s*$", body, flags=re.M)
    if not match:
        return body, {}
    start = match.end()
    after = re.search(r"^##\s+\S", body[start:], flags=re.M)
    end = start + after.start() if after else len(body)
    section = body[start:end]
    notes: Dict[str, str] = {}
    pieces = re.split(r"^###\s+(\S+)\s*$", section, flags=re.M)
    intro = pieces[0]
    for name, text in zip(pieces[1::2], pieces[2::2]):
        notes[name.strip("`")] = text.strip()
    return body[:start] + intro.rstrip() + "\n\n<!--parameters-->\n\n" + body[end:], notes


_SETUP = re.compile(r"^```figure-setup[^\n]*\n(.*?)^```[^\n]*\n?", flags=re.S | re.M)


def figure_setup(body: str) -> Tuple[str, str]:
    """Take the ```` ```figure-setup ```` blocks out: ``(body, their code)``."""
    return _SETUP.sub("", body), "\n".join(_SETUP.findall(body))


def leaves(specs, prefix: str = ""):
    """``(dotted name, spec, part path)`` for every field, parts expanded."""
    for name, spec in specs.items():
        if spec.info.hidden:
            continue
        dotted = prefix + name
        if spec.children is not None:
            yield from leaves(spec.children, dotted + ".")
        else:
            yield dotted, spec


def _sentence(text: str) -> str:
    """A tooltip as a sentence: capitalised, with a full stop.

    Only a plain word is capitalised; a tooltip that opens with a name or an
    expression (``x_nm=xnm, ...``) means it as written.
    """
    text = text.strip()
    first = text.split(" ", 1)[0].rstrip(":,;")
    if first.isalpha() and first.islower():
        text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def parameter_table(specs, notes: Dict[str, str], converter: Converter) -> str:
    """Every setting as a row: what it is called, its default, what it does."""
    if not specs:
        return "<p><i>This plugin has no settings.</i></p>"
    rows = ['<table class="params" width="100%" cellspacing="0" cellpadding="5">',
            '<tr><th align="left">setting</th><th align="left">default</th>'
            '<th align="left">what it does</th></tr>']

    def emit(specs, prefix: str, depth: int) -> None:
        for name, spec in specs.items():
            info = spec.info
            if info.hidden:
                continue
            dotted = prefix + name
            if spec.children is not None:
                doc = (spec.type.__doc__ or "").strip().split("\n\n")[0]
                doc = " ".join(doc.split()) if not doc.startswith(
                    spec.type.__name__ + "(") else ""
                note = notes.get(dotted, "")
                text = converter.inline(doc)
                if note:
                    text += "".join(converter._blocks(note.splitlines()))
                rows.append(f'<tr class="part"><td colspan="3"><b>'
                            f'{html.escape(info.label or name)}</b>'
                            f'{" &ndash; " + text if text else ""}</td></tr>')
                emit(spec.children, dotted + ".", depth + 1)
                continue
            label = html.escape(info.label or name)
            more = ' <span class="more">(more)</span>' if info.advanced else ""
            parts = []
            if info.help:
                parts.append(converter.inline(_sentence(info.help)))
            note = notes.get(dotted)
            if note:
                parts.append("".join(converter._blocks(note.splitlines())))
            extra = [x for x in (_choices(info) and "Choices: " + _choices(info),
                                 _bounds(info)) if x]
            if extra:
                parts.append(f'<span class="range">{html.escape("; ".join(extra))}'
                             f'</span>')
            indent = "&nbsp;&nbsp;&nbsp;" * depth
            rows.append(f"<tr><td>{indent}<b>{label}</b>{more}<br>{indent}"
                        f'<code class="name">{html.escape(dotted)}</code></td>'
                        f"<td>{_format_default(spec)}</td>"
                        f"<td>{' '.join(parts) or '&ndash;'}</td></tr>")

    emit(specs, "", 0)
    rows.append("</table>")
    return "\n".join(rows)


def header(plugin_cls) -> str:
    """The generated top of every page: name, place, what it is."""
    path = plugin_cls.path or ""
    crumbs = " &rsaquo; ".join(html.escape(p) for p in path.split("/")[:-1])
    facts = [f"version {html.escape(str(plugin_cls.version))}"]
    if getattr(plugin_cls, "scope", "locs") == "site":
        facts.append("runs once per ROI")
    try:
        if plugin_cls.has_preview():
            facts.append("has a preview")
    except Exception:
        pass
    if getattr(plugin_cls, "live", False):
        facts.append("can run live")
    out = [f"<h1>{html.escape(plugin_cls.name or path)}</h1>",
           f'<p class="crumbs">{crumbs} &middot; {" &middot; ".join(facts)}</p>']
    summary = plugin_cls.description or (plugin_cls.__doc__ or "").strip().split("\n")[0]
    if summary:
        out.append(f'<p class="summary">{html.escape(summary)}</p>')
    return "\n".join(out)


# ---------------------------------------------------------------- the images
@dataclass
class Rendered:
    """A page as HTML, and the images it names, as PNG bytes by name."""
    html: str
    images: Dict[str, bytes] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


_MATH_CACHE: Dict[Tuple[str, bool, str, float], Tuple[bytes, float, float]] = {}
_FIGURE_CACHE: Dict[str, bytes] = {}
# rendered at this many device pixels per logical one, and shown at 1/SCALE,
# so the images stay sharp on a high-density screen
SCALE = 2.0


def math_image(tex: str, display: bool, color: str = "black",
               size_pt: float = 10.0) -> Tuple[bytes, float, float]:
    """``(png, width, height)`` of one formula, the size in logical pixels.

    matplotlib's mathtext, not LaTeX: it is already installed, it needs no TeX,
    and it covers the maths a page needs -- fractions, roots, sums, Greek,
    sub- and superscripts.  What it lacks (``align``, matrices) a page does
    without.
    """
    key = (tex, display, color, size_pt)
    if key in _MATH_CACHE:
        return _MATH_CACHE[key]
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser, math_to_image

    size = size_pt * (1.25 if display else 1.0)
    prop = FontProperties(size=size, math_fontfamily="dejavusans")
    text = f"${tex}$"
    # logical px at 96 dpi: the size a sentence around it is drawn at
    parsed = MathTextParser("path").parse(text, dpi=96, prop=prop)
    buffer = io.BytesIO()
    math_to_image(text, buffer, prop=prop, dpi=96 * SCALE, format="png", color=color)
    result = (buffer.getvalue(), float(parsed.width), float(parsed.height + parsed.depth))
    _MATH_CACHE[key] = result
    return result


def _figure_key(code: str, covers: List[str]) -> str:
    digest = hashlib.sha1((_FIGURE_CACHE_VERSION + code).encode())
    for dotted in covers:
        try:
            obj = resolve(dotted)
            module = obj if hasattr(obj, "__file__") else \
                importlib.import_module(obj.__module__)
            digest.update(Path(module.__file__).read_bytes())
        except Exception:
            digest.update(dotted.encode())
    return digest.hexdigest()


def draw_figure(code: str, covers: List[str] = (), setup: str = "",
                shared: Optional[Dict[str, Any]] = None) -> bytes:
    """Run a figure block and return it as a PNG.

    The block gets ``fig`` (a `matplotlib.figure.Figure`, 6 x 3 inches unless
    it says otherwise) and ``np``, and draws with the plugin's own functions.
    No pyplot: the figure is drawn off screen, whatever backend the program
    runs, and nothing is left open afterwards.

    ``setup`` is the page's ```` ```figure-setup ```` code -- a simulation
    several figures draw from -- run at most once per page into ``shared``,
    and only if some figure is not already cached.
    """
    key = _figure_key(setup + "\n" + code, list(covers))
    if key in _FIGURE_CACHE:
        return _FIGURE_CACHE[key]
    cached = _disk_cache() / f"{key}.png" if _disk_cache() else None
    if cached is not None and cached.is_file():
        _FIGURE_CACHE[key] = cached.read_bytes()
        return _FIGURE_CACHE[key]

    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    shared = {} if shared is None else shared
    if not shared:
        shared.update({"np": np, "__name__": "figure"})
        if setup:
            exec(compile(setup, "<figure-setup>", "exec"), shared)
    fig = Figure(figsize=(6, 3))
    FigureCanvasAgg(fig)
    exec(compile(code, "<figure>", "exec"), {**shared, "fig": fig})
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=96 * SCALE, bbox_inches="tight",
                facecolor="white")
    png = buffer.getvalue()
    _FIGURE_CACHE[key] = png
    if cached is not None:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(png)
        except OSError:
            pass
    return png


def _disk_cache() -> Optional[Path]:
    try:
        from .. import config
        return config.config_dir() / "cache" / "docs"
    except Exception:
        return None


def _png_size(png: bytes) -> Tuple[int, int]:
    """Width and height from a PNG's header, without decoding it."""
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


# ------------------------------------------------------------------ rendering
def render(plugin_cls, color: str = "black", size_pt: float = 10.0,
           figures: bool = True, link=None, origin: Optional[Path] = None) -> Rendered:
    """The whole page for ``plugin_cls``: generated and written parts together.

    ``color`` is the text colour the maths is drawn in, so a dark palette gets
    light formulas.  ``link`` rewrites a link target (the exporter turns
    ``plugin:`` links into files).  A figure that fails is shown as its error
    rather than failing the page; `Rendered.errors` collects them.
    """
    page = page_for(plugin_cls, origin)
    out = Rendered("")
    counter = [0]

    def name(kind: str) -> str:
        counter[0] += 1
        return f"{kind}{counter[0]}.png"

    def math(tex: str, display: bool) -> str:
        try:
            png, width, height = math_image(tex, display, color, size_pt)
        except Exception as error:
            out.errors.append(f"maths {tex!r}: {error}")
            return f"<code>{html.escape(tex)}</code>"
        file = name("math")
        out.images[file] = png
        align = "" if display else ' style="vertical-align: middle"'
        img = (f'<img src="{file}" width="{width:.0f}" height="{height:.0f}"'
               f"{align}>")
        return f'<p class="math" align="center">{img}</p>' if display else img

    def figure(code: str, caption: str) -> str:
        cap = f'<p class="caption">{converter.inline(caption)}</p>' if caption else ""
        if not figures:
            return cap
        try:
            png = draw_figure(code, page.covers, setup, shared)
        except Exception:
            error = traceback.format_exc(limit=-2)
            out.errors.append(f"figure {caption!r}: {error}")
            return (f'<p class="error">This figure could not be drawn:</p>'
                    f"<pre>{html.escape(error)}</pre>{cap}")
        file = name("figure")
        out.images[file] = png
        width, height = _png_size(png)
        return (f'<p class="figure" align="center"><img src="{file}" '
                f'width="{width / SCALE:.0f}" height="{height / SCALE:.0f}"></p>{cap}')

    converter = Converter(math=math, figure=figure, link=link)
    body, setup = figure_setup(page.body)
    shared: Dict[str, Any] = {}
    body, notes = parameter_notes(body)
    try:
        specs = plugin_cls.specs()
    except Exception as error:
        specs = {}
        out.errors.append(f"settings: {error}")
    table = parameter_table(specs, notes, converter)
    if "<!--parameters-->" in body:
        body = body.replace("<!--parameters-->", table)
    else:
        body = body.rstrip() + "\n\n## Parameters\n\n" + table + "\n"
    if not page.written:
        body = ('<p class="missing">No written page for this plugin yet: what '
                'follows is generated from the plugin itself.</p>\n\n' + body)
    text = header(plugin_cls) + "\n" + converter.convert(body)
    if page.written and page.version is not None and \
            page.version != str(plugin_cls.version):
        text = text.replace(
            "</h1>", "</h1>\n" + f'<p class="stale">This page was written for version '
            f"{html.escape(page.version)}; the plugin is at version "
            f"{html.escape(str(plugin_cls.version))} and its numbers may have "
            f"moved since.</p>", 1)
    out.html = text
    return out


STYLE = """
body { line-height: 135%; }
h1 { margin-bottom: 0; }
h2 { margin-top: 18px; }
p.crumbs { color: #777; margin-top: 2px; }
p.summary { font-size: large; }
p.caption { color: #555; font-size: small; margin-left: 24px; margin-right: 24px; }
p.missing, p.stale { color: #9a6700; }
p.error { color: #b00020; }
table.params th, table.grid th { background: #eceff3; }
table.params td, table.grid td { border-bottom: 1px solid #ddd; vertical-align: top; }
tr.part td { background: #f4f6f8; padding-top: 8px; }
span.more, span.range, code.name { color: #777; }
table.note { background: #f1f5fb; }
code { font-family: monospace; }
"""
