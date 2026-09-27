"""Simulating data: a structure, its labelling and blinking, and camera frames.

The simulator is where every other tutorial's data comes from; this one is
about using it on purpose.  A built-in structure (nuclear pores) with a
realistic labelling efficiency, the blinking in the terms one knows a dye by,
then the same simulation as camera frames -- written as a recipe the fitter
opens like an acquisition, camera included -- and the fit scored against the
truth by Ground Truth.

The storyboard checks what the words claim: that the fitter took its camera
from the simulation, and that the isolated spots come back with errors the
size of their reported precision.
"""
from __future__ import annotations

import numpy as np

from .common import (expand, field, figure_panels, open_section, place_beside,
                     section_of, show_tab, type_into)

TITLE = "Simulating data: structures, blinking and camera frames"
DESCRIPTION = ("Simulating a labelled structure as localizations or camera "
               "frames, fitting the frames, and scoring the fit against the truth.")

_YAML = """<pre style="font-size:80%;line-height:1.25">elements:
  - points: [[0, 0, 0], [12, 5, 0]]
  - line: [[0, 0, 0], [1000, 0, 0]]
    density: 0.05
  - circle: 50
    density: 0.2
copies:
  density: 2
  rotation: random</pre>"""


def _run(d, panel):
    panel.run_button.click()
    d.settle()


def make(d) -> None:
    session, control, render = d.session, d.control, d.render

    d.chapter("Why simulate")
    d.card(TITLE,
           "<p>Data whose truth is known: to learn the program, to choose "
           "settings, and to test a fit.</p>"
           "<p>One model gives localizations, or the camera frames to fit.</p>",
           say="Simulated data has a known truth: to try the program, to choose "
               "settings, and to check a fit.")

    d.chapter("The structure")
    file_tab = show_tab(d, "File")
    _, sim = open_section(d, file_tab, "Blinking Structure")
    top = d.union(field(sim, "output"), field(sim, "seed"))
    d.shot("Simulate is in the File tab. What to make, how many frames, the "
           "background, drift and the random seed come first.",
           spot=[top], zoom=d.around(top, 760))
    preset = field(sim, "structure.preset")
    type_into(sim, "structure.preset", "npc")
    d.settle()
    d.shot("The structure: a built-in, here nuclear pores, or a file of your "
           "own.",
           spot=[preset], point=preset, click=True, zoom=d.around(preset, 760))
    d.card("A structure file",
           "<p>Label positions, or lines, circles, areas and images filled at "
           "a density; and how many copies, placed and turned at random.</p>"
           + _YAML,
           say="A structure file lists label positions, or shapes filled at a "
               "density, and how many copies to place.")

    d.chapter("Labelling and blinking")
    efficiency = field(sim, "labelling.efficiency")
    expand(efficiency)
    section_of(sim, "labelling").set_expanded(True)
    type_into(sim, "labelling.efficiency", 0.6)
    d.settle()
    d.shot("Not every site carries a dye: a labelling efficiency near 60 % is "
           "typical for the pores. Each label gets one fluorophore.",
           spot=[efficiency], point=efficiency, click=True,
           zoom=d.around(efficiency, 760))
    blinking = section_of(sim, "blinking")
    d.shot("Blinking in the numbers a dye is known by: how long it is on, how "
           "often it comes back, and its photons per blink.",
           spot=[blinking], zoom=d.around(blinking, 760))
    d.shot("The off time follows from the number of blinks, and switching "
           "happens anywhere within a frame, as in a real movie.",
           spot=[field(sim, "blinking.blinks"), field(sim, "blinking.on_time")],
           zoom=d.around(blinking, 760))
    _run(d, sim)
    assert "copy" in session.locs, sorted(session.locs.keys())
    copy = np.bincount(session.locs["copy"])
    pore = session.locs[session.locs["copy"] == int(np.argmax(copy))]
    cx, cy = float(np.median(pore["x_nm"])), float(np.median(pore["y_nm"]))
    render.view.frame_on((cx - 450, cx + 450), (cy - 280, cy + 280))
    d.settle()
    d.shot("Run, and the pores appear: eight corners each, some missing, as "
           "the labelling left them.",
           spot=[d.rect(render.view.graphics, pad=-2)], point=d.at_data(cx, cy))
    close = field(sim, "localizations.close")
    expand(close)
    section_of(sim, "localizations").set_expanded(True)
    d.settle()
    d.shot("Two molecules on at once within a PSF cannot be told apart: they "
           "are removed, or averaged into one, as a fitter would see them.",
           spot=[close], point=close, zoom=d.around(close, 760))

    d.chapter("Camera frames")
    path = d.data_dir() / "pores.sim.yaml"
    type_into(sim, "output", "camera")
    type_into(sim, "n_frames", 2000)
    camera = section_of(sim, "camera")
    camera.set_expanded(True)
    type_into(sim, "camera.path", str(path))
    d.settle()
    d.shot("Camera frames instead: the same molecules on a camera, with shot "
           "and read noise. They are saved as a recipe, not as a movie.",
           spot=[field(sim, "output"), camera], zoom=d.around(camera, 760))
    _run(d, sim)
    assert path.exists(), sim.output.toPlainText()
    d.shot("The recipe is a few lines: the settings and the seed. The frames "
           "are made as a fitter reads them.",
           spot=[sim.output], zoom=d.around(sim.output, 760))

    d.chapter("Fitting them")
    localize = show_tab(d, "Localize")
    _, fit = open_section(d, localize, "Gaussian 2D")
    type_into(fit, "source.path", str(path))
    d.settle()
    d.shot("Give the fitter the recipe as its file.",
           spot=[field(fit, "source.path")], point=field(fit, "source.path"),
           zoom=d.around(field(fit, "source.path"), 760))
    button = next(b for b in control.findChildren(type(fit.run_button))
                  if b.text().startswith("Camera parameters"))
    button.click()
    d.settle()
    window = control._camera_window
    place_beside(d, window, 0.8)
    d.shot("The camera comes from the simulation: nothing to type in.",
           spot=[d.window_rect(window)])
    window.hide()
    d.shot("Run.", spot=[fit.run_button], point=fit.run_button, click=True,
           zoom=d.around(fit.run_button, 760))
    _run(d, fit)
    assert len(session.locs) > 1000, "the fit of the simulation found almost nothing"
    assert str(session.locs.metadata.get("source")) == str(path)

    d.chapter("Against the truth")
    analysis = show_tab(d, "Analysis")
    _, truth = open_section(d, analysis, "Ground Truth")
    isolated = field(truth, "isolated")
    type_into(truth, "min_photons", 300)
    d.shot("Ground Truth works out where every spot really was and pairs them "
           "with the localizations, frame by frame.",
           spot=[truth.form], zoom=d.around(truth.form, 760))
    _run(d, truth)
    counted = truth.result.data["comparison"]
    d.shot("Found, false and missed, and the Jaccard index that sums them up. "
           "Crowded spots and dim slivers cost detections.",
           spot=[truth.output], zoom=d.around(truth.output, 760))
    type_into(truth, "isolated", True)
    d.settle()
    _run(d, truth)
    c = truth.result.data["comparison"]
    assert c["jaccard"] > counted["jaccard"], (c["jaccard"], counted["jaccard"])
    # the words say the reported precision is honest on isolated spots
    assert abs(c["axes"]["x"]["pull"] - 1) < 0.15, c["axes"]["x"]
    truth.plot()
    d.settle()
    window = truth._window
    place_beside(d, window, 0.8)
    d.shot("With only the isolated spots, nearly all are found.",
           spot=[isolated, d.window_rect(window)], point=isolated)
    panels = figure_panels(d, window)
    assert len(panels) >= 3, f"{len(panels)} ground truth panels"
    d.shot("How many were found, by their photons: the dim ones are the "
           "hardest.", spot=[panels[0]], zoom=d.around(panels[0], 900))
    d.shot("The real error against the precision the table reports. On each "
           "other: the precision can be trusted.",
           spot=[panels[1]], zoom=d.around(panels[1], 900))
    d.shot("Each error over its own precision: a unit Gaussian when the fit is "
           "honest.", spot=[panels[2]], zoom=d.around(panels[2], 900))
    window.hide()

    d.chapter("Next")
    d.card("What to remember",
           "<ul><li><b>Structure:</b> a built-in or a file of positions and "
           "shapes, copied many times.</li>"
           "<li><b>Labelling and blinking:</b> efficiency, on time, blinks and "
           "photons, in the terms a dye is known by.</li>"
           "<li><b>Camera frames:</b> a recipe the fitter opens, camera "
           "included.</li>"
           "<li><b>Ground Truth:</b> what was found, and whether the precision "
           "is honest.</li></ul>",
           say="That is simulation: a known truth, to learn on and to test "
               "against.")
