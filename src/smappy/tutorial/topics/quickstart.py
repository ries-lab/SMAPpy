"""SMAPpy in three minutes: raw camera frames to a super-resolution picture.

The first of the series and the one on the landing page, so it shows the
whole path once and explains nothing twice: the Localize tab fits a
simulated acquisition (`simulate.camera_frames`, the tour's ring and lines as
a camera would have seen them), the picture builds up while it fits, and the
result is saved beside the acquisition.  Where to go from there -- filters,
layers, grouping -- is the tour's job (`layout`), and it says so.

The acquisition is written by the storyboard, not shipped: 2000 frames of
100 x 100 pixels take seven seconds to simulate and two to fit.  It carries no
camera metadata on purpose, which is the one realistic complication worth
showing -- a file SMAPpy cannot recognise the camera of -- and it is typed in.
"""
from __future__ import annotations

from .common import field, open_section, place_beside, show_tab, tab_button, type_into

TITLE = "SMAPpy in three minutes"
DESCRIPTION = ("From raw camera frames to a super-resolution picture: fitting, "
               "checking the fit, and the result.")

# the camera the simulation used: typed in the tutorial, since the file does
# not say, and read back by the fit -- so they must agree
CONVERSION, OFFSET, PIXELSIZE_UM = 0.5, 100.0, 0.1

_SMLM = """
<svg viewBox="0 0 560 170" role="img" aria-label="a few molecules per frame, each fitted, all frames together make the picture">
  <g font-size="13" fill="currentColor">
    <rect x="10" y="20" width="80" height="80" rx="6" fill="#000"/>
    <circle cx="35" cy="45" r="7" fill="#fff" opacity=".85"/><circle cx="68" cy="78" r="7" fill="#fff" opacity=".6"/>
    <rect x="100" y="20" width="80" height="80" rx="6" fill="#000"/>
    <circle cx="150" cy="40" r="7" fill="#fff" opacity=".75"/><circle cx="120" cy="80" r="7" fill="#fff" opacity=".9"/>
    <text x="95" y="122" text-anchor="middle">a few molecules per frame</text>
    <path d="M195 60 h45" class="arrow" marker-end="url(#h2)"/>
    <text x="218" y="48" text-anchor="middle" opacity=".7">fit</text>
    <rect x="255" y="20" width="80" height="80" rx="6" class="panel"/>
    <circle cx="280" cy="45" r="2.5" class="dot"/><circle cx="313" cy="78" r="2.5" class="dot"/>
    <circle cx="305" cy="40" r="2.5" class="dot"/><circle cx="275" cy="80" r="2.5" class="dot"/>
    <text x="295" y="122" text-anchor="middle">positions, nm precise</text>
    <path d="M350 60 h45" class="arrow" marker-end="url(#h2)"/>
    <text x="373" y="48" text-anchor="middle" opacity=".7">all frames</text>
    <rect x="410" y="20" width="80" height="80" rx="6" fill="#000"/>
    <circle cx="450" cy="60" r="28" fill="none" stroke="#ffb040" stroke-width="3"/>
    <text x="450" y="122" text-anchor="middle">the picture</text>
    <text x="10" y="155" opacity=".85">The spots are blurred by diffraction; their centres are not.</text>
  </g>
  <defs><marker id="h2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>
</svg>
"""


def _acquisition(d):
    """Simulate the acquisition and write it where the file field will show it."""
    import tifffile
    from ...simulate import camera_frames
    frames, _ = camera_frames(2000, seed=0, conversion=CONVERSION, offset=OFFSET,
                              pixelsize_nm=PIXELSIZE_UM * 1000)
    path = d.data_dir() / "demo_acquisition.tif"
    tifffile.imwrite(path, frames)
    return path


def make(d) -> None:
    session, render = d.session, d.render
    acquisition = _acquisition(d)

    d.chapter("The idea")
    d.card(TITLE,
           "<p>From raw camera frames to a super-resolution picture: fit the "
           "frames, check the fit, look at the result.</p>"
           "<p>The acquisition here is simulated, so nothing depends on a "
           "microscope.</p>",
           say="In three minutes: from raw camera frames to a super-resolution "
               "picture.")
    d.card("How the picture is made",
           "<p>In each frame only a few molecules shine, far enough apart to "
           "be seen one by one.</p>"
           "<p>Fitting each spot finds its centre to a few nanometres. "
           "Thousands of frames of positions make the picture.</p>",
           figure=_SMLM,
           say="A reminder: only a few molecules shine in each frame. Fitting "
               "each spot finds its centre, and many frames make the picture.")

    d.chapter("The raw data")
    localize = show_tab(d, "Localize")
    d.shot("Raw camera frames are fitted in the Localize tab.",
           spot=[tab_button(d, "Localize")], point=tab_button(d, "Localize"),
           click=True)
    fitters = [s.button for s in localize.sections]
    d.shot("Each section is a fitter. Gaussian 2D is for ordinary data; the "
           "spline fitters use a bead calibration, for 3D.",
           spot=[d.union(*fitters)], zoom=d.around(localize.sections[0], 700))
    sec, panel = open_section(d, localize, "Gaussian 2D")
    type_into(panel, "source.path", str(acquisition))
    # small blocks, so the picture is caught half built (at a thousand frames
    # a second, the default 200 is most of the fit before the first draw)
    type_into(panel, "source.chunk", 50)
    d.settle()
    d.shot("Open Gaussian 2D and choose the acquisition. Any file of a "
           "Micro-Manager or NDTiff dataset will do.",
           spot=[field(panel, "source.path")], point=field(panel, "source.path"),
           zoom=d.around(field(panel, "source.path"), 760))

    d.chapter("The camera")
    for name, value in (("conversion", CONVERSION), ("offset", OFFSET),
                        ("pixelsize_um", PIXELSIZE_UM)):
        type_into(panel, f"camera.{name}", value)
    d.settle()
    camera = [field(panel, f"camera.{n}") for n in ("conversion", "offset", "pixelsize_um")]
    d.shot("SMAPpy normally reads the camera from the file. This simulated file "
           "says nothing about it, so its numbers are typed in here.",
           spot=[d.union(*camera)], zoom=d.around(camera[1], 760))

    d.chapter("Check, then fit")
    d.shot("Preview fits one frame and shows what it found, before anything "
           "is saved.",
           spot=[panel.preview_button], point=panel.preview_button, click=True,
           zoom=d.around(panel.preview_button, 760))
    if panel.preview_frame is not None:
        panel.preview_frame.setValue(10)
    panel.preview_button.click()
    d.settle()
    preview = panel._window
    place_beside(d, preview)
    d.shot("Every spot should have its circle. If noise is circled, or a spot "
           "is missed, change the cutoff in detection.",
           spot=[d.window_rect(preview)])
    preview.hide()
    detection = section_of(panel, "detection")
    d.shot("The cutoff is the one setting that usually needs a look; the rest "
           "can stay as they are.",
           spot=[detection], zoom=d.around(detection, 760))

    d.shot("Then Run.", spot=[panel.run_button], point=panel.run_button,
           click=True, zoom=d.around(panel.run_button, 760))
    panel.run_button.click()
    _wait_for_first_block(d)
    d.shot("The picture builds up while the fit runs, so a bad acquisition "
           "shows early.",
           spot=[d.rect(render.view.graphics, pad=-2)], wait=False)
    d.settle()
    assert len(session.locs) > 1000, "the fit found almost nothing"
    d.shot("When it is done, the localizations are saved beside the "
           "acquisition, and the picture is complete.",
           spot=[field(panel, "output.path")], point=field(panel, "output.path"),
           zoom=d.around(field(panel, "output.path"), 760))

    d.chapter("The picture")
    render.view.frame_on((6400, 8200), (3600, 5400))
    d.settle()
    d.shot("Zoomed in, the ring is sharp far below the size of a camera pixel.",
           spot=[d.rect(render.view.graphics, pad=-2)], point=d.at_data(7480, 4700))
    render.view.reset()
    show_tab(d, "Render")
    d.shot("The Render tab sets how the picture looks: filters, colours, "
           "and grouping of blinks.",
           spot=[tab_button(d, "Render")], point=tab_button(d, "Render"), click=True)

    d.chapter("Next")
    d.card("That is all it takes",
           "<ul><li><b>Localize</b> fits the frames: choose the file, check the "
           "camera, preview, run.</li>"
           "<li><b>Render</b> shows the result.</li>"
           "<li><b>Analysis</b> measures it.</li></ul>"
           "<p>Next: the tour of SMAPpy, with filters, layers and grouping.</p>",
           say="That is all it takes. Next, the tour of SMAPpy: filters, layers "
               "and grouping.")


def section_of(panel, name: str):
    """A part of a plugin's form (``detection``), as the section that holds it."""
    from ...gui.widgets import CollapsibleSection
    widget = field(panel, name)
    while widget is not None and not isinstance(widget, CollapsibleSection):
        widget = widget.parentWidget()
    return widget if widget is not None else field(panel, name)


def _wait_for_first_block(d, timeout: float = 30.0) -> None:
    """Until the fit has streamed something in: a shot of it half done."""
    import time
    end = time.monotonic() + timeout
    while not len(d.session.locs) and time.monotonic() < end:
        d.pump(0.01)
