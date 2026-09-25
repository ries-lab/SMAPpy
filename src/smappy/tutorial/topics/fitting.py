"""Fitting, 2D and 3D: what each part of a fitter does, and how to check it.

The quickstart fits once with the defaults; this one opens the fitter up.
The camera and where its numbers come from, detection and what a wrong
cutoff looks like, the PSF model, and -- after the fit -- the two columns
that say whether a localization is a good one.  Then 3D: an astigmatic
acquisition fitted with Spline 3D, against a bead calibration.

Everything is simulated here, the calibration included: `simulate.bead_stacks`
draws beads through focus with the same astigmatic PSF `camera_frames` gives
the acquisition, and `calibrate.core` builds the calibration from them, as
the Bead calibration window would.  Making the calibration is not shown --
that window is a tutorial of its own -- but it is real: the 3D fit gets the
simulated z back (`tests/test_camera_frames.py`).
"""
from __future__ import annotations

from .common import field, open_section, place_beside, section_of, show_tab, type_into

TITLE = "Fitting: 2D and 3D"
DESCRIPTION = ("The camera, finding the spots, the PSF model and checking the "
               "fit; then 3D with a bead calibration.")

CONVERSION, OFFSET, PIXELSIZE_UM = 0.5, 100.0, 0.1

_PIPELINE = """
<svg viewBox="0 0 560 150" role="img" aria-label="camera counts to photons, candidates, a fit per spot, a table">
  <g font-size="13" fill="currentColor">
    <rect x="5" y="30" width="110" height="56" rx="8" class="panel"/>
    <text x="60" y="54" text-anchor="middle" font-weight="600">camera</text>
    <text x="60" y="73" text-anchor="middle" opacity=".75">counts to photons</text>
    <path d="M118 58 h24" class="arrow" marker-end="url(#h3)"/>
    <rect x="145" y="30" width="120" height="56" rx="8" class="panel"/>
    <text x="205" y="54" text-anchor="middle" font-weight="600">detection</text>
    <text x="205" y="73" text-anchor="middle" opacity=".75">where are spots?</text>
    <path d="M268 58 h24" class="arrow" marker-end="url(#h3)"/>
    <rect x="295" y="30" width="120" height="56" rx="8" class="panel"/>
    <text x="355" y="54" text-anchor="middle" font-weight="600">model</text>
    <text x="355" y="73" text-anchor="middle" opacity=".75">fit each spot</text>
    <path d="M418 58 h24" class="arrow" marker-end="url(#h3)"/>
    <rect x="445" y="30" width="110" height="56" rx="8" class="panel"/>
    <text x="500" y="54" text-anchor="middle" font-weight="600">output</text>
    <text x="500" y="73" text-anchor="middle" opacity=".75">the table</text>
    <text x="5" y="125" opacity=".85">Each is a section of the fitter's form, in this order.</text>
  </g>
  <defs><marker id="h3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>
</svg>
"""

_ASTIGMATISM = """
<svg viewBox="0 0 560 150" role="img" aria-label="a spot wide in y below focus, round at focus, wide in x above">
  <g font-size="13" fill="currentColor">
    <rect x="20" y="20" width="100" height="90" rx="8" fill="#000"/>
    <ellipse cx="70" cy="65" rx="11" ry="27" fill="#fff" opacity=".85"/>
    <text x="70" y="132" text-anchor="middle">below focus</text>
    <rect x="230" y="20" width="100" height="90" rx="8" fill="#000"/>
    <circle cx="280" cy="65" r="15" fill="#fff" opacity=".9"/>
    <text x="280" y="132" text-anchor="middle">in focus</text>
    <rect x="440" y="20" width="100" height="90" rx="8" fill="#000"/>
    <ellipse cx="490" cy="65" rx="27" ry="11" fill="#fff" opacity=".85"/>
    <text x="490" y="132" text-anchor="middle">above focus</text>
    <path d="M135 65 h80" class="arrow" marker-end="url(#h4)"/>
    <path d="M345 65 h80" class="arrow" marker-end="url(#h4)"/>
    <text x="175" y="55" text-anchor="middle" opacity=".7">z</text>
    <text x="385" y="55" text-anchor="middle" opacity=".7">z</text>
  </g>
  <defs><marker id="h4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>
</svg>
"""


def _data(d):
    """The two acquisitions and the calibration, written where the form shows them."""
    import numpy as np
    import tifffile
    from ...calibrate.core import CalibrationSettings, build_calibration, collect_beads
    from ...calibrate.input import BeadStack
    from ...simulate import ASTIGMATISM, bead_stacks, camera_frames
    folder = d.data_dir()
    camera = dict(conversion=CONVERSION, offset=OFFSET, pixelsize_nm=PIXELSIZE_UM * 1000)
    flat = folder / "demo_acquisition.tif"
    frames, _ = camera_frames(2000, seed=0, **camera)
    tifffile.imwrite(flat, frames)
    astigmatic = folder / "demo_3d_acquisition.tif"
    frames, _ = camera_frames(2000, seed=1, astigmatism=ASTIGMATISM, **camera)
    tifffile.imwrite(astigmatic, frames)
    stacks, z = bead_stacks(3, seed=0, astigmatism=ASTIGMATISM, **camera)
    beads = collect_beads([BeadStack(s.astype(np.float32), z, source=f"beads {i + 1}")
                           for i, s in enumerate(stacks)], CalibrationSettings())
    calibration = folder / "demo_beads_3dcal.h5"
    calibration.unlink(missing_ok=True)
    build_calibration(beads).save(calibration)
    return flat, astigmatic, calibration


def _camera(panel) -> None:
    for name, value in (("conversion", CONVERSION), ("offset", OFFSET),
                        ("pixelsize_um", PIXELSIZE_UM)):
        type_into(panel, f"camera.{name}", value)


def _preview(d, panel, frame: int = 10):
    if panel.preview_frame is not None:
        panel.preview_frame.setValue(frame)
    panel.preview_button.click()
    d.settle()
    window = panel._window
    place_beside(d, window)
    return window


def make(d) -> None:
    session, control, render = d.session, d.control, d.render
    flat, astigmatic, calibration = _data(d)

    d.chapter("The pipeline")
    d.card(TITLE,
           "<p>What each part of a fitter does, what to check, and how to fit "
           "in 3D with a bead calibration.</p>"
           "<p>The acquisitions and the calibration are simulated.</p>",
           say="This tutorial goes through fitting step by step: first in 2D, "
               "then in 3D.")
    d.card("What a fitter does",
           "<p>It converts the camera's counts to photons, finds candidate "
           "spots, fits a model of the PSF to each, and writes a table of "
           "localizations.</p>",
           figure=_PIPELINE,
           say="A fitter converts the camera's counts to photons, finds the "
               "spots, fits each one, and writes the table.")

    d.chapter("The camera")
    localize = show_tab(d, "Localize")
    sec, panel = open_section(d, localize, "Gaussian 2D")
    type_into(panel, "source.path", str(flat))
    d.settle()
    d.shot("The acquisition goes into source. A Micro-Manager or NDTiff file "
           "also tells SMAPpy which camera took it.",
           spot=[field(panel, "source.path")], point=field(panel, "source.path"),
           zoom=d.around(field(panel, "source.path"), 760))
    camera = section_of(panel, "camera")
    d.shot("For a recognised camera these stay on auto: the numbers come from "
           "the file and from SMAPpy's camera database.",
           spot=[camera], zoom=d.around(camera, 760))
    _camera(panel)
    d.settle()
    button = next(b for b in control.findChildren(type(panel.run_button))
                  if b.text().startswith("Camera parameters"))
    d.shot("This file has no camera information, so the values are typed in. "
           "Camera parameters shows what the fit will use.",
           spot=[d.union(*[field(panel, f"camera.{n}")
                           for n in ("conversion", "offset", "pixelsize_um")]), button],
           point=button, click=True, zoom=d.around(button, 760))
    button.click()
    d.settle()
    window = control._camera_window
    place_beside(d, window, 0.8)
    d.shot("Every number, and where it came from. A wrong conversion still fits "
           "well but gives wrong photon counts, so check it once per camera.",
           spot=[d.window_rect(window)])
    window.hide()

    d.chapter("Finding the spots")
    detection = section_of(panel, "detection")
    d.shot("Detection smooths each frame and keeps the peaks above a cutoff. "
           "The dynamic cutoff is in units of the noise, so it follows the "
           "background.",
           spot=[detection], zoom=d.around(detection, 760))
    default = field(panel, "detection.cutoff").value()
    type_into(panel, "detection.cutoff", 0.8)
    preview = _preview(d, panel)
    d.shot("Too low a cutoff, and noise is taken for molecules.",
           spot=[d.window_rect(preview)])
    preview.hide()
    type_into(panel, "detection.cutoff", default)
    preview = _preview(d, panel)
    d.shot("At the default, the molecules are found and the noise is left "
           "alone. Preview is the place to decide this, before a long fit.",
           spot=[d.window_rect(preview)])
    preview.hide()

    d.chapter("The model")
    model = section_of(panel, "model")
    d.shot("Gaussian 2D fits the position, photons, background and the width "
           "of each spot. Elliptical fits a width in x and one in y.",
           spot=[model], zoom=d.around(model, 760))
    roi = field(panel, "fit.roi_size") if "roi_size" in field(panel, "fit").fields else None
    if roi is not None:
        d.shot("The ROI size is the square fitted around each spot: large enough "
               "for the spot, small enough not to catch a neighbour.",
               spot=[roi], zoom=d.around(roi, 760))

    d.chapter("Fit and check")
    d.shot("Run.", spot=[panel.run_button], point=panel.run_button, click=True,
           zoom=d.around(panel.run_button, 760))
    panel.run_button.click()
    d.settle()
    assert len(session.locs) > 1000, "the 2D fit found almost nothing"
    render_tab = show_tab(d, "Render")
    filt = render_tab.filter
    ll = filt.quick.get("logl_rel")
    if ll is not None:
        ll.click()
        d.settle()
        d.shot("LL is how well the model fits. A spot that is not a single "
               "molecule fits badly, and the default filter drops the worst.",
               spot=[ll, d.rect(filt.plot)], point=ll, click=True,
               zoom=d.around(filt, 700))
    psf = filt.quick.get("sigma_nm")
    if psf is not None:
        psf.click()
        d.settle()
        d.shot("PSF is the fitted width. Two molecules fitted as one come out "
               "too wide; for a 2D fit, the default filter drops them too.",
               spot=[psf, d.rect(filt.plot)], point=psf, click=True,
               zoom=d.around(filt, 700))

    d.chapter("3D")
    d.card("Astigmatism",
           "<p>A cylindrical lens in the detection path makes each spot "
           "elliptical: wide in one direction above the focus, in the other "
           "below it.</p>"
           "<p>How the shape changes with z is measured once, on beads, and "
           "the fit reads z from it.</p>",
           figure=_ASTIGMATISM,
           say="In 3D, a cylindrical lens makes the spots elliptical, and their "
               "shape tells how high each molecule sits.")
    localize = show_tab(d, "Localize")
    beads = next(b for b in control.findChildren(type(panel.run_button))
                 if b.text().startswith("Bead calibration"))
    d.shot("The calibration comes from beads stepped through focus. Bead "
           "calibration makes it; that has a tutorial of its own.",
           spot=[beads], point=beads, zoom=d.around(beads, 760))
    sec3, panel3 = open_section(d, localize, "Spline 3D")
    type_into(panel3, "source.path", str(astigmatic))
    type_into(panel3, "model.calibration", str(calibration))
    _camera(panel3)
    d.settle()
    d.shot("Spline 3D takes the acquisition and the calibration file. The "
           "camera and the detection are as in 2D.",
           spot=[field(panel3, "model.calibration")],
           point=field(panel3, "model.calibration"),
           zoom=d.around(field(panel3, "model.calibration"), 760))
    d.shot("Run.", spot=[panel3.run_button], point=panel3.run_button, click=True,
           zoom=d.around(panel3.run_button, 760))
    panel3.run_button.click()
    d.settle()
    assert "z_nm" in session.locs, "the 3D fit wrote no z"
    render_tab = show_tab(d, "Render")
    render_tab.color.setCurrentIndex(1)
    render_tab.color_field.setCurrentText("z_nm")
    d.settle()
    d.shot("Colour by z, and the ring rises and falls as it was simulated.",
           spot=[d.rect(render.view.graphics, pad=-2),
                 d.union(render_tab.color, render_tab.color_field)])

    d.chapter("Next")
    d.card("What to remember",
           "<ul><li><b>Camera:</b> check its numbers once.</li>"
           "<li><b>Detection:</b> set the cutoff with Preview.</li>"
           "<li><b>Model:</b> Gaussian for 2D; Spline 3D with a bead "
           "calibration for 3D.</li>"
           "<li><b>After the fit:</b> the default filters drop the bad "
           "localizations.</li></ul>",
           say="That is fitting. Next: rendering the result, in 2D and in 3D.")
