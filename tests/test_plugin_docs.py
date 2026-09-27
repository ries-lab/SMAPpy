"""The plugins' pages: that each one is there, describes the plugin as it is,
explains every setting, and renders -- maths, figures and all.

The figures are the plugins' own code run on simulated data, so rendering a
page is also a check that the page's claims still run."""
import re
import textwrap

import pytest

from smappy import docs, plugins
from smappy.docs import markup

# Plugins still waiting for a written page.  It may only shrink: a new plugin
# gets a page (docs/<path>.md beside the plugins) or a line here, and a page
# that is written takes its plugin off.  Until then the Help window shows the
# summary and the settings table, generated.
UNDOCUMENTED = {
    "Analysis/Drift/COMET",
    "Analysis/Dual-Color/AssignColors",
    "Analysis/Measure/Ground Truth",
    "Analysis/Measure/Line Profile",
    "Analysis/Measure/Localization Precision",
    "Analysis/Process/History",
    "Analysis/Process/Math Parser",
    "Analysis/Process/Remove Localizations",
    "Analysis/Register/Calibrate transform",
    "Chain/Layers",
    "File/Export/Image",
    "File/Load/Auto",
    "File/Load/MINFLUX",
    "File/Load/SMAP",
    "File/Load/csv",
    "File/Load/smappy HDF5",
    "File/Save/smappy HDF5",
    "File/Simulate/Blinking Structure",
    "Localize/Gaussian 2D 2C",
    "Localize/Spline 3D",
    "Localize/Spline 3D 2C",
    "ROIManager/Analyze/Histograms",
    "ROIManager/Evaluate/Statistics",
    "ROIManager/Segment/Density Peaks",
}


def shipped():
    return {path: ref for path, ref in plugins.refs().items()
            if ref.root == "builtin" and ref.kind == "plugin"}


def documented():
    return sorted(path for path, ref in shipped().items()
                  if docs.page_file(path, ref.origin) is not None)


# ------------------------------------------------------------------ coverage
def test_every_shipped_plugin_has_a_page_or_is_listed_as_waiting_for_one():
    missing = {path for path, ref in shipped().items()
               if docs.page_file(path, ref.origin) is None}
    assert missing - UNDOCUMENTED == set(), "write a page, or list it above"
    assert UNDOCUMENTED - missing == set(), "it has a page now: take it off the list"


def test_every_page_belongs_to_a_plugin():
    """A page left behind by a renamed plugin would never be shown."""
    root = plugins.BUILTIN_ROOT / docs.DOCS_DIR
    pages = {str(p.relative_to(root).with_suffix("")).replace("\\", "/")
             for p in root.rglob("*.md")}
    assert pages - set(shipped()) == set()


@pytest.mark.parametrize("path", documented())
def test_a_page_describes_the_version_its_plugin_is_at(path):
    """A version is bumped when the numbers move -- which is when the page's
    account of the algorithm may have gone wrong.  Re-read it, then bring its
    ``version`` along."""
    cls = plugins.get(path)
    page = docs.page_for(cls)
    assert page.version == str(cls.version), (
        f"{page.path} describes version {page.version}; {path} is at "
        f"{cls.version}.  Check the page against the change, then update it")


@pytest.mark.parametrize("path", documented())
def test_what_a_page_says_it_covers_exists(path):
    page = docs.page_for(plugins.get(path))
    assert page.covers, "name the functions the page explains"
    for dotted in page.covers:
        docs.resolve(dotted)


@pytest.mark.parametrize("path", documented())
def test_every_setting_of_a_documented_plugin_is_explained(path):
    """By its tooltip (`help`), by a note on the page, or both."""
    cls = plugins.get(path)
    _, notes = docs.parameter_notes(docs.page_for(cls).body)
    names = {name for name, _ in docs.leaves(cls.specs())}
    unexplained = [name for name, spec in docs.leaves(cls.specs())
                   if not spec.info.help and name not in notes]
    assert unexplained == []
    stray = set(notes) - names - {n.rpartition(".")[0] for n in names}
    assert stray == set(), "a note for a setting the plugin does not have"


@pytest.mark.parametrize("path", documented())
def test_a_written_page_renders_with_its_maths_and_figures(path):
    rendered = docs.render(plugins.get(path))
    assert rendered.errors == []
    figures = [name for name in rendered.images if name.startswith("figure")]
    assert figures, "a written page shows what the plugin does"
    for name, data in rendered.images.items():
        assert data.startswith(b"<svg" if name.endswith(".svg") else b"\x89PNG\r\n\x1a\n")


def test_a_plugin_without_a_page_still_gets_its_settings():
    rendered = docs.render(plugins.get("File/Load/csv"))
    assert rendered.errors == []
    assert "No written page" in rendered.html
    assert "<code class=\"name\">mapping</code>" in rendered.html


# ---------------------------------------------------------------- the pages
PLUGIN = '''
    from dataclasses import dataclass
    from smappy.plugins import Plugin, Result, param

    @dataclass
    class ThingSettings:
        radius_nm: float = param(50.0, label="radius", unit="nm", min=0.1,
                                 help="how far to look")
        mode: str = param("a", choices=(("a", "the first"), ("b", "the second")))

    class Thing(Plugin):
        Settings = ThingSettings
        version = "2"
        description = "Measures a thing."

        def run(self, ctx, settings):
            return Result()
'''


def thing(tmp_path, page: str):
    """A plugin dropped in as one file, with its page beside it."""
    folder = tmp_path / "Mine" / "Tools"
    folder.mkdir(parents=True)
    (folder / "thing.py").write_text(textwrap.dedent(PLUGIN))
    (folder / "thing.md").write_text(textwrap.dedent(page))
    from smappy.plugins.discovery import load_module
    cls = load_module(folder / "thing.py").Thing
    cls.path, cls.name = "Mine/Tools/Thing", "Thing"
    return cls, folder / "thing.py"


def test_a_page_beside_a_dropped_in_plugin_is_found_and_completed(tmp_path):
    cls, origin = thing(tmp_path, """\
        ---
        version: "2"
        ---
        ## What it does
        Finds things within $r$ of each other.

        ## Parameters
        ### mode
        The second is **faster**.
        """)
    rendered = docs.render(cls, origin=origin)
    assert rendered.errors == []
    text = rendered.html
    assert "<h1>Thing</h1>" in text and "Measures a thing." in text
    assert "Mine &rsaquo; Tools" in text and "version 2" in text
    # the generated table: label, default with its unit, the tooltip, and the
    # page's note in the same row; the choices by their labels
    assert "<b>radius</b>" in text and "50 nm" in text and "How far to look." in text
    row = text[text.index('<code class="name">mode</code>'):]
    row = row[:row.index("</tr>")]
    assert "the first" in row and "<b>faster</b>" in row
    assert "stale" not in text
    assert any(name.startswith("math") for name in rendered.images)


def test_a_page_written_for_another_version_says_so(tmp_path):
    cls, origin = thing(tmp_path, """\
        ---
        version: "1"
        ---
        ## What it does
        Old news.
        """)
    text = docs.render(cls, origin=origin).html
    assert "written for version 1" in text and "version 2" in text


def test_a_figure_that_fails_shows_its_error_rather_than_losing_the_page(tmp_path):
    cls, origin = thing(tmp_path, """\
        ---
        version: "2"
        ---
        ```figure Never drawn.
        raise RuntimeError("no data")
        ```
        """)
    rendered = docs.render(cls, origin=origin)
    assert "could not be drawn" in rendered.html and "no data" in rendered.html
    assert rendered.errors and "Never drawn" in rendered.errors[0]


def test_figures_share_a_setup_and_are_drawn_once():
    code = "fig.subplots().plot(data)"
    setup = "data = [1, 3, 2]"
    first = docs.draw_figure(code, setup=setup, shared={})
    assert first[:4] == b"\x89PNG"
    # cached: the same code is not run again, so a namespace without the
    # setup's variables is never looked at
    assert docs.draw_figure(code, setup=setup, shared={"np": None}) == first


# ---------------------------------------------------------------- the markup
def test_markdown_becomes_html_with_the_maths_handed_over():
    seen = []

    def math(tex, display):
        seen.append((tex, display))
        return "[M]"

    out = markup.to_html(textwrap.dedent("""\
        # Title

        Some *text* with $x_i^2$ and `a*b*c` -- and **bold**, [a link](plugin:A/B C).

        $$\\sum_k d_k = 0$$

        - one
        - two
          continued
            - nested

        | a | b |
        | --- | --- |
        | 1 | $y$ |

        > a note

        ```python
        x = 2 * 3 * 4
        ```
        """), math=math, link=lambda url: url.replace(" ", "%20"))
    assert "<h1>Title</h1>" in out
    assert "<i>text</i>" in out and "<b>bold</b>" in out and "&ndash;" in out
    assert "<code>a*b*c</code>" in out                     # nothing inside code
    assert '<a href="plugin:A/B%20C">a link</a>' in out
    assert seen == [("x_i^2", False), ("\\sum_k d_k = 0", True), ("y", False)]
    assert "<li>two continued<ul><li>nested</li></ul></li>" in out
    assert "<td>1</td><td>[M]</td>" in out
    assert 'class="note"' in out
    assert "<pre>x = 2 * 3 * 4</pre>" in out


def test_a_link_with_parentheses_in_its_url_is_written_encoded():
    out = markup.inline("[doi](https://doi.org/10.1016/S0006-3495%2802%2975618-X)")
    assert 'href="https://doi.org/10.1016/S0006-3495%2802%2975618-X"' in out


def test_a_dollar_can_be_written():
    assert markup.inline("costs \\$5") == "costs $5"


def test_a_formula_is_typeset_as_svg_the_size_of_the_text_around_it():
    image = docs.math_image("\\frac{S}{\\sqrt{N}}", False, renderer="ziamath")
    assert image.renderer == "ziamath" and image.ext == "svg"
    assert image.data.startswith(b"<svg") and b"<symbol" not in image.data
    assert 5 < image.width < 80 and 15 < image.height < 60
    assert 0 < image.depth < image.height / 2       # the fraction reaches below
    shown = docs.math_image("\\frac{S}{\\sqrt{N}}", True, renderer="ziamath")
    assert shown.height > image.height              # displayed: a full-size fraction


def test_the_fallback_is_mathtext_for_a_formula_ziamath_refuses(monkeypatch):
    def refuse(*args):
        raise ValueError("no")
    monkeypatch.setattr(docs, "_ziamath", refuse)
    monkeypatch.setattr(docs, "_MATH_CACHE", {})
    monkeypatch.setattr(docs, "_disk_cache", lambda: None)
    image = docs.math_image("x^2 + 1", False)
    assert image.renderer == "mathtext" and image.data[:4] == b"\x89PNG"


def test_the_fallback_can_be_asked_for(monkeypatch):
    monkeypatch.setenv("SMAPPY_DOCS_MATH", "mathtext")
    assert docs.math_renderer() == "mathtext"
    assert docs.math_image("x_i", False).renderer == "mathtext"


@pytest.mark.parametrize("renderer", ["ziamath", "mathtext"])
def test_a_centred_formula_has_its_baseline_on_the_line(renderer):
    """Qt can only centre an inline image; padded, the centre lands where Qt
    puts it (``middle`` above the baseline) with the baseline on the text's."""
    image = docs.math_image("\\sigma_{\\max}^2", False, renderer=renderer)
    for middle in (1.0, 6.0):
        padded = image.centred(middle)
        above, below = padded.height - padded.depth, padded.depth
        assert abs((above - below) / 2 - middle) < 0.6
        if renderer == "ziamath":                   # the SVG says the same
            height = float(re.search(rb'<svg[^>]*height="([^"]+)"', padded.data).group(1))
            assert abs(height - padded.height) < 0.01


def page_formulas(path):
    found = []
    record = lambda tex, display: found.append((tex, display)) or ""
    inner = markup.Converter(math=record)
    body = docs.figure_setup(docs.page_for(plugins.get(path)).body)[0]
    markup.Converter(math=record, figure=lambda code, caption: inner.inline(caption)
                     ).convert(body)
    return found


@pytest.mark.parametrize("path", documented())
def test_every_formula_on_a_page_is_set_by_both_renderers(path):
    """So falling back never costs a page its maths."""
    formulas = page_formulas(path)
    assert formulas
    for tex, display in formulas:
        assert docs._ziamath(tex, display, "black", 10.0).width > 0, tex
        assert docs._mathtext(tex, display, "black", 10.0).width > 0, tex


# ------------------------------------------------------------------- the GUI
@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_a_plugins_title_bar_has_a_question_mark_that_opens_its_page(app):
    from smappy.gui import help_window
    from smappy.gui.plugin_tab import PluginTab
    from smappy.session import Session
    from smappy.workspace import Instance, Tab

    tab = PluginTab(Tab(name="A", instances=[Instance(plugin="Analysis/Drift/RCC")]),
                    Session())
    section = tab.sections[0]
    assert section.help_button is not None
    section.help_button.click()
    window = help_window._WINDOW
    assert window is not None and window.current == "Analysis/Drift/RCC"
    text = window.browser.toPlainText()
    assert "RCC" in text and "How it works" in text and "time windows" in text
    # its images reached the document, so the maths is drawn and not missing
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QTextDocument
    names = [n for n in docs.render(plugins.get("Analysis/Drift/RCC")).images
             if n.startswith("math")]
    assert names and names[0].endswith(".svg")
    image = window.browser.document().resource(QTextDocument.ImageResource,
                                                QUrl(names[0]))
    assert not image.isNull() and image.width() > 10
    # a link to another plugin's page is followed in the same window
    window._on_link(QUrl("plugin:Analysis/Drift/COMET"))
    assert window.current == "Analysis/Drift/COMET"
    assert "No written page" in window.browser.toPlainText()


def test_a_plugin_in_its_own_window_has_the_question_mark_too(app):
    from smappy.gui import help_window
    from smappy.gui.plugin_panel import PluginWindow
    from smappy.session import Session

    window = PluginWindow(plugins.get("Analysis/Measure/Localization Statistics"),
                          Session())
    window.help_button.click()
    assert help_window._WINDOW.current == "Analysis/Measure/Localization Statistics"
