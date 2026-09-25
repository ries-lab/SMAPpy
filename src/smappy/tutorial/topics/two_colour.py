"""Two colours in 2D: a split camera, the two-colour fit, and the colours.

Ratiometric two-colour imaging, as SMAPpy does it: a dichroic splits each
molecule's light between two halves of the chip, and which dye it is shows in
how the photons split.  So the parts are the transformation (where the second
half sees what the first sees), a fit that fits both spots of a pair at once,
and the colour assignment from the photon ratio.

The transformation is measured from the movie itself ("calibrate from this
movie") because that is the route that needs nothing else -- no beads, no
earlier file -- and the one most 2D users will take; a bead calibration or a
saved ``_2ct.h5`` is said rather than shown.  3D, where the beads are needed
for the PSF anyway, is the next tutorial.

Simulated with `simulate.dual_camera_frames`: the ring one dye, the lines the
other, behind a known transformation, so the storyboard checks that the
transformation came back and that the colours are the dyes.
"""
from __future__ import annotations

import numpy as np

from .common import (field, open_section, place_beside, section_of, show_tab,
                     type_into)

TITLE = "Two colours in 2D: split camera, fit and colours"
DESCRIPTION = ("Ratiometric two-colour imaging: the transformation between the "
               "halves, the two-colour fit, and assigning the colours.")

CONVERSION, OFFSET, PIXELSIZE_UM = 0.5, 100.0, 0.1

_SPLIT = """
<svg viewBox="0 0 560 170" role="img" aria-label="a dichroic splits each molecule's light between two halves of the camera; the split differs by dye">
  <g font-size="13" fill="currentColor">
    <circle cx="40" cy="60" r="8" fill="#ff6a3d"/><text x="40" y="95" text-anchor="middle">dye A</text>
    <circle cx="40" cy="125" r="8" fill="#3dd17a"/><text x="40" y="160" text-anchor="middle">dye B</text>
    <path d="M60 92 h70" class="arrow" marker-end="url(#h8)"/>
    <path d="M150 60 l40 60" stroke="currentColor" stroke-width="3"/>
    <text x="170" y="145" text-anchor="middle" opacity=".75">dichroic</text>
    <path d="M200 70 h70" class="arrow" marker-end="url(#h8)"/>
    <path d="M200 110 h70" class="arrow" marker-end="url(#h8)"/>
    <rect x="285" y="15" width="120" height="70" rx="6" fill="#000"/>
    <rect x="285" y="90" width="120" height="70" rx="6" fill="#000"/>
    <circle cx="320" cy="45" r="9" fill="#fff" opacity=".95"/>
    <circle cx="370" cy="55" r="9" fill="#fff" opacity=".35"/>
    <circle cx="322" cy="120" r="9" fill="#fff" opacity=".35"/>
    <circle cx="372" cy="130" r="9" fill="#fff" opacity=".95"/>
    <text x="420" y="50">main half</text>
    <text x="420" y="125">secondary half</text>
    <text x="285" y="10" font-size="12" opacity=".75">A bright above, B below</text>
  </g>
  <defs><marker id="h8" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>
</svg>
"""


def _data(d):
    import tifffile
    from ...simulate import dual_camera_frames
    path = d.data_dir() / "demo_two_colour.tif"
    frames, truth = dual_camera_frames(2000, seed=0, conversion=CONVERSION,
                                       offset=OFFSET, pixelsize_nm=PIXELSIZE_UM * 1000)
    tifffile.imwrite(path, frames)
    return path, truth


def _agreement(locs, truth) -> float:
    """The fraction of assigned localizations whose colour is their dye, the
    better of the two ways round (which dye is colour 1 is the ratio's order)."""
    from scipy.spatial import cKDTree
    got, want = [], []
    for f in np.unique(locs["frame"]):
        m, t = locs["frame"] == f, truth["frame"] == f
        if not t.any():
            continue
        dist, i = cKDTree(np.column_stack([truth["x_nm"][t], truth["y_nm"][t]])).query(
            np.column_stack([locs["x_nm"][m], locs["y_nm"][m]]))
        near = (dist < 60) & (locs["channel"][m] > 0)
        got += list(locs["channel"][m][near])
        want += list(truth["dye"][t][i[near]])
    got, want = np.asarray(got), np.asarray(want)
    same = float(np.mean(got == want))
    return max(same, 1 - same)


def _colour_layer(d, tab, channel: int, lut: str) -> None:
    filt = tab.filter
    filt.field.setCurrentText("channel")
    d.settle()
    filt.lo.setText(str(channel))
    filt.hi.setText(str(channel))
    filt.lo.editingFinished.emit()
    filt.hi.editingFinished.emit()
    tab.lut.setCurrentText(lut)
    d.settle()


def _check_transformation(path) -> float:
    """Largest distance, in pixels, between the measured and the simulated
    map over the chip's corners and centre."""
    from ...calibrate.transform import load_transform
    from ...simulate import dual_transformation
    probe = np.array([[5.0, 105.0], [95.0, 105.0], [5.0, 195.0], [95.0, 195.0],
                      [50.0, 150.0]])
    true = (np.c_[probe, np.ones(len(probe))] @ dual_transformation().T)[:, :2]
    return float(np.abs(load_transform(path).transform(probe) - true).max())


def make(d) -> None:
    session, render = d.session, d.render
    movie, truth = _data(d)

    d.chapter("Two dyes, one camera")
    d.card(TITLE,
           "<p>Two dyes imaged at once, on one camera: a dichroic splits each "
           "molecule's light between the two halves of the chip.</p>"
           "<p>Both dyes appear in both halves, in different proportions. The "
           "proportion says which dye a molecule is.</p>",
           figure=_SPLIT,
           say="Two dyes on one camera: each molecule's light is split between "
               "two halves of the chip, and how it splits tells the dye.")
    d.card("Three steps",
           "<p><b>Transformation:</b> where the second half sees what the first "
           "sees, a little shifted, turned and magnified.</p>"
           "<p><b>Fit:</b> both spots of a molecule at once, one position, "
           "photons for each half.</p>"
           "<p><b>Colour:</b> from how the photons split.</p>",
           say="Three steps: map one half onto the other, fit both spots of a "
               "molecule together, and assign the colour.")

    d.chapter("The fitter")
    localize = show_tab(d, "Localize")
    _, panel = open_section(d, localize, "Gaussian 2D 2C")
    type_into(panel, "source.path", str(movie))
    for name, value in (("conversion", CONVERSION), ("offset", OFFSET),
                        ("pixelsize_um", PIXELSIZE_UM)):
        type_into(panel, f"camera.{name}", value)
    d.settle()
    d.shot("Gaussian 2D 2C fits a split camera. The acquisition and the camera "
           "go in as for any fit.",
           spot=[field(panel, "source.path")], point=field(panel, "source.path"),
           zoom=d.around(field(panel, "source.path"), 760))

    d.chapter("The transformation")
    calibrate = field(panel, "transform.calibrate")
    type_into(panel, "transform.calibrate", True)
    d.settle()
    d.shot("Calibrate from this movie measures the transformation from the "
           "movie's own molecules, which appear in both halves. No beads needed.",
           spot=[calibrate], point=calibrate, click=True,
           zoom=d.around(calibrate, 760))
    path = field(panel, "transform.path")
    d.shot("It is saved beside the result, to use again here. A dual-colour bead "
           "calibration works too.",
           spot=[path], point=path, zoom=d.around(path, 760))

    d.chapter("The fit")
    model = section_of(panel, "model")
    d.shot("Each molecule is fitted in both halves at once: one position, "
           "photons for each half. The width is free per half, as the colours "
           "differ.",
           spot=[model], zoom=d.around(model, 760))
    after_the_fit(d, panel)
    d.shot("Run.", spot=[panel.run_button], point=panel.run_button, click=True,
           zoom=d.around(panel.run_button, 760))
    panel.run_button.click()
    d.settle()
    # the words say the transformation is measured from the movie
    from pathlib import Path
    output = Path(field(panel, "output.path").value())
    saved = output.with_name(output.stem + "_2ct.h5")
    assert saved.exists(), f"no transformation at {saved}"
    error = _check_transformation(saved)
    assert error < 0.2, f"the transformation is {error:.2f} px off"
    d.shot("The output box says how well the halves were matched, and how many "
           "localizations got a colour.",
           spot=[panel.output], zoom=d.around(panel.output, 760))

    colours_and_layers(d, panel, truth)

    d.chapter("Next")
    d.card("What to remember",
           "<ul><li>A split camera sees each molecule twice; the photon split "
           "is its colour.</li>"
           "<li><b>Transformation:</b> calibrate from the movie, or use a bead "
           "calibration.</li>"
           "<li><b>Gaussian 2D 2C</b> fits both halves together.</li>"
           "<li><b>Assign colours</b> runs at the end of the fit and gives each "
           "localization a channel; run it again to change the rule.</li>"
           "<li>A layer per channel shows the colours.</li></ul>",
           say="That is two colours in 2D. Next: two colours in 3D, with a "
               "dual-colour bead calibration.")


def after_the_fit(d, fitter) -> None:
    """The fit's last step, on by default: Assign colours over the result."""
    automatic = field(fitter, "finish.assign_colors")
    d.shot("After the fit, assign colours is on: the fit ends by running the "
           "Assign colours plugin, so the table comes out with its colours.",
           spot=[section_of(fitter, "finish")], point=automatic,
           zoom=d.around(automatic, 760))


def colours_and_layers(d, fitter, truth) -> None:
    """The last steps of a two-colour fit, the same in 2D and 3D: Assign
    colours, that the fitter can do it by itself, and a layer per channel."""
    session, render = d.session, d.render
    d.chapter("Assign colours")
    locs = session.locs
    assert "channel" in locs, f"no colours; the table has {sorted(locs.keys())}"
    # the words say the colours are the dyes
    agreement = _agreement(locs, truth)
    assert agreement > 0.95, f"only {agreement:.0%} of the colours are the dye"
    analysis = show_tab(d, "Analysis")
    _, colours = open_section(d, analysis, "Assign colours")
    d.shot("The plugin the fit ran is here, in the Analysis tab. Run it to see "
           "how it decided, or to decide again.",
           spot=[colours.form], point=colours.run_button, click=True,
           zoom=d.around(colours.form, 760))
    colours.run_button.click()
    d.settle()
    agreement = _agreement(session.locs, truth)          # it rewrote the table
    assert agreement > 0.95, f"after Assign colours, {agreement:.0%} are the dye"
    colours.plot()
    d.settle()
    window = colours._window
    place_beside(d, window, 0.75)
    d.shot("How the photons split: one peak per dye. Each localization gets the "
           "channel of its peak; the few between the peaks get none.",
           spot=[d.window_rect(window)])
    window.hide()
    mode = field(colours, "mode")
    d.shot("Probabilistic is stricter: it gives a channel only to what is "
           "clearly one dye. Run it with all localizations shown, or it sees "
           "only one dye.",
           spot=[mode], point=mode, zoom=d.around(mode, 760))

    d.chapter("Two layers")
    tab = show_tab(d, "Render")
    _colour_layer(d, tab, 1, "red")
    d.settle()
    d.shot("To draw the colours, filter layer 1 on channel: from 1 to 1, in red.",
           spot=[tab.filter.field, d.union(tab.filter.lo, tab.filter.hi), tab.lut],
           point=tab.filter.field, zoom=d.around(tab.filter, 760))
    tab.strip.add.menu().actions()[0].trigger()          # + -> localizations
    d.settle()
    _colour_layer(d, tab, 2, "green")
    render.view.reset()
    d.settle()
    d.shot("Add a layer for channel 2, in green. Here the ring is one dye and "
           "the lines the other.",
           spot=[d.rect(render.view.graphics, pad=-2), tab.strip],
           point=tab.strip.add)
