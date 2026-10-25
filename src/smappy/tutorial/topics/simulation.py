"""Simulating data: a scene from a preset, its components, and camera frames.

The simulator is where every other tutorial's data comes from; this one is
about using it on purpose.  A preset (nuclear pores) and its key parameters,
the components a scene is made of, the blinking in the terms one knows a dye by,
then the same simulation as camera frames -- written as a recipe the fitter
opens like an acquisition, camera included -- and the fit scored against the
truth by Compare to ground truth.

The storyboard checks what the words claim: that the fitter took its camera
from the simulation, and that the isolated spots come back with errors the
size of their reported precision.
"""
from __future__ import annotations

import numpy as np

from .common import (expand, field, figure_panels, open_section, pin, place_beside,
                     section_of, show_tab, type_into)

TITLE = "Simulating data: scenes, blinking and camera frames"
DESCRIPTION = ("Simulating a scene from a preset as localizations or camera "
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

    d.chapter("A scene from a preset")
    file_tab = show_tab(d, "File")
    _, sim = open_section(d, file_tab, "Simulate")
    preset = field(sim, "preset")
    d.shot("Simulate is in the File tab. A preset is a whole scene to start "
           "from; your own are saved beside the shipped ones.",
           spot=[preset], zoom=d.around(preset, 760))
    type_into(sim, "preset", "Nuclear pores (Nup96)")
    d.settle()
    keys = sim.form.keys_section
    assert keys.isVisible() and sim.form.key_fields, "no key parameters shown"
    d.shot("Nuclear pores: the preset fills every setting and puts the few "
           "that matter most at the top.",
           spot=[keys], zoom=d.around(keys, 760))
    efficiency = sim.form.key_fields["components.0.labelling.efficiency"]
    efficiency.set(0.55)
    efficiency.changed.emit()
    d.settle()
    assert sim.form.value().components[0].labelling.efficiency == 0.55
    d.shot("Not every copy of the protein carries a dye: change the labelling "
           "efficiency here, and the preset says it was modified.",
           spot=[efficiency, preset], point=efficiency, click=True,
           zoom=d.around(efficiency, 760))

    d.chapter("Components")
    components = section_of(sim, "components")
    components.set_expanded(True)
    d.settle()
    d.shot("The scene is made of components: here the pores. A background, "
           "beads or a second colour are added with +, or saved for later.",
           spot=[components], zoom=d.around(components, 760))
    structure = section_of(sim, "components.0.structure")
    d.shot("Each component has its structure: a built-in, a file of your own, "
           "labels spread evenly, or beads.",
           spot=[structure], zoom=d.around(structure, 760))
    d.card("A structure file",
           "<p>Label positions, or lines, circles, areas and images filled at "
           "a density; and how many copies, placed and turned at random.</p>"
           + _YAML,
           say="A structure file lists label positions, or shapes filled at a "
               "density, and how many copies to place.")
    blinking = section_of(sim, "components.0.blinking")
    blinking.set_expanded(True)
    d.settle()
    d.shot("Blinking in the numbers a dye is known by: how long it is on, how "
           "often it comes back, and its photons per blink.",
           spot=[blinking], zoom=d.around(blinking, 760))
    d.shot("The off time follows from the number of blinks, and switching "
           "happens anywhere within a frame, as in a real movie.",
           spot=[field(sim, "components.0.blinking.blinks"),
                 field(sim, "components.0.blinking.on_time")],
           zoom=d.around(blinking, 760))
    components.set_expanded(False)
    d.settle()
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
    fit_this = sim.notice_button
    assert fit_this.isVisible() and fit_this.text() == "Fit this"
    d.shot("Fit this writes the recipe from the settings as they are and gives "
           "it to a fitter as its file.",
           spot=[fit_this], point=fit_this, click=True,
           zoom=d.around(fit_this, 760))
    fit_this.click()
    d.settle()
    localize = show_tab(d, "Localize")
    _, fit = open_section(d, localize, "Gaussian 2D")
    assert fit.form.values()["source.path"] == str(path)
    d.shot("The Localize tab opens on the fitter, with the recipe as its file.",
           spot=[field(fit, "source.path")], point=field(fit, "source.path"),
           zoom=d.around(field(fit, "source.path"), 760))
    section_of(fit, "camera").set_expanded(True)
    microscope = fit.form.fields["camera"]
    d.settle()
    d.shot("The camera comes from the simulation: nothing to type in.",
           spot=[microscope.inner])
    section_of(fit, "camera").set_expanded(False)
    d.shot("Run.", spot=[fit.run_button], point=fit.run_button, click=True,
           zoom=d.around(fit.run_button, 760))
    _run(d, fit)
    assert len(session.locs) > 1000, "the fit of the simulation found almost nothing"
    assert str(session.locs.metadata.get("source")) == str(path)

    d.chapter("Against the truth")
    analysis = show_tab(d, "Analysis")
    pin(d, analysis, "Analysis/Measure/Ground Truth", "Compare to ground truth")
    _, truth = open_section(d, analysis, "Compare to ground truth")
    isolated = field(truth, "isolated")
    type_into(truth, "min_photons", 300)
    d.shot("Compare to ground truth works out where every spot really was and pairs them "
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
           "<ul><li><b>Presets:</b> a whole scene, its key parameters at "
           "the top; your own saved beside them.</li>"
           "<li><b>Components:</b> structures, a background, beads, a second "
           "colour, each with its own labelling and blinking.</li>"
           "<li><b>Labelling and blinking:</b> efficiency, on time, blinks and "
           "photons, in the terms a dye is known by.</li>"
           "<li><b>Camera frames:</b> a recipe the fitter opens, camera "
           "included.</li>"
           "<li><b>Compare to ground truth:</b> what was found, and whether the precision "
           "is honest.</li></ul>",
           say="That is simulation: a known truth, to learn on and to test "
               "against.")
