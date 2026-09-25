"""3D fitting: a bead calibration, and Spline 3D fitted with it.

Why first, briefly: z is read from the shape of the spot, so the fit needs to
know that shape at every z -- and a real microscope's PSF is not the Gaussian
a formula would give it.  So the PSF is measured, on beads stepped through
focus, and fitted as a spline: a model as realistic as the microscope it came
from.  Then the Bead calibration window, as a user meets it, and the fit.

Everything is simulated: `simulate.bead_stacks` draws the beads with the same
astigmatic PSF `camera_frames` gives the acquisition, written as plain TIFFs
-- which carry no z step, so the storyboard types it, as a user of such files
must.  The save dialog is the one thing not driven: the calibration is saved
where the dialog would have put it, and "Use in the Spline 3D fitter" takes
it from there.  The fit is checked against the simulated z, as
`tests/test_camera_frames.py` checks the same path without a GUI.
"""
from __future__ import annotations

import numpy as np

from .common import field, menu, menu_item, open_section, show_tab, type_into

TITLE = "3D fitting: bead calibration and spline fitting"
DESCRIPTION = ("Why a measured PSF, making a bead calibration, checking it, and "
               "fitting astigmatic data in 3D with Spline 3D.")

CONVERSION, OFFSET, PIXELSIZE_UM, DZ_NM = 0.5, 100.0, 0.1, 20.0

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
    <path d="M135 65 h80" class="arrow" marker-end="url(#h6)"/>
    <path d="M345 65 h80" class="arrow" marker-end="url(#h6)"/>
    <text x="175" y="55" text-anchor="middle" opacity=".7">z</text>
    <text x="385" y="55" text-anchor="middle" opacity=".7">z</text>
  </g>
  <defs><marker id="h6" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>
</svg>
"""

_BEADS = """
<svg viewBox="0 0 560 150" role="img" aria-label="a bead imaged at many objective positions makes a stack; the stack becomes a PSF model">
  <g font-size="13" fill="currentColor">
    <g opacity=".9">
      <rect x="20" y="50" width="60" height="60" rx="4" fill="#000"/>
      <rect x="32" y="38" width="60" height="60" rx="4" fill="#000" stroke="#555"/>
      <rect x="44" y="26" width="60" height="60" rx="4" fill="#000" stroke="#555"/>
      <ellipse cx="74" cy="56" rx="7" ry="14" fill="#fff" opacity=".85"/>
    </g>
    <text x="65" y="135" text-anchor="middle">a bead, stepped through z</text>
    <path d="M125 70 h60" class="arrow" marker-end="url(#h7)"/>
    <rect x="200" y="30" width="150" height="70" rx="8" class="panel"/>
    <text x="275" y="60" text-anchor="middle" font-weight="600">many beads</text>
    <text x="275" y="80" text-anchor="middle" opacity=".75">aligned, averaged</text>
    <path d="M360 70 h60" class="arrow" marker-end="url(#h7)"/>
    <rect x="430" y="30" width="120" height="70" rx="8" class="panel"/>
    <text x="490" y="60" text-anchor="middle" font-weight="600">spline</text>
    <text x="490" y="80" text-anchor="middle" opacity=".75">the PSF at any z</text>
  </g>
  <defs><marker id="h7" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>
</svg>
"""


def _data(d):
    """Bead stacks and an astigmatic acquisition of the same PSF, as TIFFs."""
    import tifffile
    from ...simulate import ASTIGMATISM, bead_stacks, camera_frames
    folder = d.data_dir() / "beads"
    folder.mkdir(parents=True, exist_ok=True)
    camera = dict(conversion=CONVERSION, offset=OFFSET, pixelsize_nm=PIXELSIZE_UM * 1000)
    stacks, _ = bead_stacks(3, seed=0, dz_nm=DZ_NM, astigmatism=ASTIGMATISM, **camera)
    paths = []
    for i, stack in enumerate(stacks):
        path = folder / f"beads_{i + 1}.tif"
        tifffile.imwrite(path, stack)
        paths.append(path)
    acquisition = d.data_dir() / "demo_3d_acquisition.tif"
    frames, truth = camera_frames(2000, seed=1, astigmatism=ASTIGMATISM, **camera)
    tifffile.imwrite(acquisition, frames)
    return paths, acquisition, truth


def _wait(d, window, timeout: float = 300.0) -> None:
    """The calibration runs on a thread of the window's own, which `settle`
    does not know about."""
    import time
    end = time.monotonic() + timeout
    d.pump(0.1)
    while window.busy and time.monotonic() < end:
        d.pump(0.1)
    assert not window.busy, "the calibration did not finish"
    d.settle()


def _z_slope(locs, truth) -> float:
    """Fitted z against simulated z, molecule by molecule: 1 is right."""
    from scipy.spatial import cKDTree
    fitted, true = [], []
    for f in np.unique(locs["frame"]):
        m, t = locs["frame"] == f, truth["frame"] == f
        if not t.any():
            continue
        dist, i = cKDTree(np.column_stack([truth["x_nm"][t], truth["y_nm"][t]])).query(
            np.column_stack([locs["x_nm"][m], locs["y_nm"][m]]))
        near = dist < 60
        fitted += list(locs["z_nm"][m][near])
        true += list(truth["z_nm"][t][i[near]])
    return float(np.polyfit(true, fitted, 1)[0])


def make(d) -> None:
    session, control, render = d.session, d.control, d.render
    stacks, acquisition, truth = _data(d)

    d.chapter("Why a measured PSF")
    d.card(TITLE,
           "<p>In 3D, z is read from the shape of each spot, so the fit has to "
           "know that shape at every z.</p>"
           "<p>A real PSF is not the ideal one a formula gives: aberrations bend "
           "it. So SMAPpy measures it, on beads, and fits with that realistic "
           "model. The fit is more precise, and z comes out right.</p>",
           say="In 3D, z is read from the shape of each spot. SMAPpy uses a "
               "realistic PSF model, measured on beads, not an ideal formula.")
    d.card("Astigmatism",
           "<p>A cylindrical lens in the detection path makes each spot "
           "elliptical: wide in one direction above the focus, in the other "
           "below it.</p>",
           figure=_ASTIGMATISM,
           say="A cylindrical lens makes the spots elliptical, and their shape "
               "tells how high each molecule sits.")
    d.card("Bead stacks",
           "<p>Fluorescent beads on a coverslip, imaged while the objective "
           "steps through focus: each bead records the PSF at every z.</p>"
           "<p>Many beads are aligned and averaged, and a spline makes the "
           "average smooth, so the model can be read at any z.</p>",
           figure=_BEADS,
           say="Beads imaged through focus record the PSF at every z. Averaged "
               "and smoothed, they become the model.")

    d.chapter("The calibration window")
    tools = menu(d, "Tools")
    item = menu_item(d, tools, "Bead calibration")
    d.shot("Tools, Bead calibration opens its window. The Localize tab has the "
           "same button.",
           spot=[item], point=item, click=True, zoom=d.around(tools, 700))
    tools.close()
    control.open_calibration()
    window = control.calibration_window
    window.resize(1540, 860)
    d.place(window, 30, 20)
    window.add_paths([str(p) for p in stacks])
    d.settle()
    d.shot("Add the bead stacks: files, or a whole folder. All of them go into "
           "one calibration, so more beads make a smoother model.",
           spot=[window.file_list], point=window.file_list,
           zoom=d.around(window.file_list, 760))
    form = window.form
    dz = form.fields["dz_nm"]
    dz.set(DZ_NM)
    dz.changed.emit()
    d.settle()
    d.shot("The z step comes from the file's metadata. A plain TIFF has none, "
           "so type it in.",
           spot=[dz], point=dz, zoom=d.around(dz, 700))
    roi = form.fields["roi_size"]
    d.shot("The ROI size is the square cut around each bead: large enough for "
           "the widest spot, far out of focus.",
           spot=[roi], zoom=d.around(roi, 700))

    d.chapter("Calibrate")
    d.shot("Detect and calibrate finds the beads, aligns them in x, y and z, "
           "and builds the model.",
           spot=[window.run_button], point=window.run_button, click=True,
           zoom=d.around(window.run_button, 700))
    window.run()
    _wait(d, window)
    assert window.result is not None, "the calibration produced nothing"
    d.shot("Overview: the beads it found, green if used, and the averaged PSF "
           "seen from the side, narrowing through focus.",
           spot=[window.tabs], zoom=None)
    d.shot("Each bead is a row, with how well it matches the others. Outliers "
           "are left out by themselves; space excludes one by hand, then "
           "Recalculate.",
           spot=[window.bead_table, window.rebuild_button],
           point=window.bead_table)

    d.chapter("Check it")
    d.shot("Calculate fit quality fits the beads back with the spline fitter, "
           "as the data will be fitted.",
           spot=[window.quality_button], point=window.quality_button, click=True,
           zoom=d.around(window.quality_button, 700))
    window.fit_quality()
    _wait(d, window)
    d.shot("The fitted z of each bead against where it was: on the line over "
           "the range the calibration can be trusted.",
           spot=[window.tabs])

    d.chapter("Save and use")
    d.shot("Save the calibration next to the beads; the name says what it is.",
           spot=[window.save_button], point=window.save_button, click=True,
           zoom=d.around(window.save_button, 700))
    from ...calibrate.gui import calibration_save_defaults
    from pathlib import Path
    folder, name = calibration_save_defaults(window.paths, dual=False)
    path = Path(folder) / (name + ".h5")
    window.result.save(path, overwrite=True)
    window.saved_path, window.saved_dual = str(path), False
    window.use_button.setEnabled(True)
    d.settle()
    d.shot("Use in the Spline 3D fitter puts it straight into the fitter.",
           spot=[window.use_button], point=window.use_button, click=True,
           zoom=d.around(window.use_button, 700))
    window.use_button.click()
    d.settle()
    window.hide()
    d.settle()

    d.chapter("Fit in 3D")
    localize = show_tab(d, "Localize")
    _, panel = open_section(d, localize, "Spline 3D")
    calibration = field(panel, "model.calibration")
    assert Path(calibration.value()).name == path.name, calibration.value()
    type_into(panel, "source.path", str(acquisition))
    for name, value in (("conversion", CONVERSION), ("offset", OFFSET),
                        ("pixelsize_um", PIXELSIZE_UM)):
        type_into(panel, f"camera.{name}", value)
    d.settle()
    d.shot("Spline 3D now has the calibration. Give it the acquisition; the "
           "camera and the detection are as in 2D.",
           spot=[calibration, field(panel, "source.path")], point=calibration,
           zoom=d.around(calibration, 760))
    d.shot("Run.", spot=[panel.run_button], point=panel.run_button, click=True,
           zoom=d.around(panel.run_button, 760))
    panel.run_button.click()
    d.settle()
    locs = session.locs
    assert "z_nm" in locs, "the 3D fit wrote no z"
    # the words say z comes out right: fitted against simulated, molecule by
    # molecule, the slope is one (a mirrored calibration gave minus one)
    slope = _z_slope(locs, truth)
    assert abs(slope - 1) < 0.02, f"fitted z against true z has slope {slope:.3f}"

    render_tab = show_tab(d, "Render")
    render_tab.color.setCurrentIndex(1)
    render_tab.color_field.setCurrentText("z_nm")
    d.settle()
    d.shot("Colour by z, and the ring rises and falls as it was simulated.",
           spot=[d.rect(render.view.graphics, pad=-2),
                 d.union(render_tab.color, render_tab.color_field)])
    filt = render_tab.filter
    z = filt.quick.get("z_nm")
    if z is not None:
        z.click()
        d.settle()
        d.shot("Far from focus the spots are faint and wide, and z is less "
               "sure. The z filter keeps the range the calibration covers.",
               spot=[z, d.rect(filt.plot)], point=z, click=True,
               zoom=d.around(filt, 700))

    d.chapter("Next")
    d.card("What to remember",
           "<ul><li>SMAPpy fits 3D with a <b>measured PSF</b>, from beads.</li>"
           "<li><b>Bead calibration:</b> add the stacks, set the z step, "
           "calibrate, check the fit quality, save.</li>"
           "<li><b>Spline 3D:</b> the acquisition and the calibration; the rest "
           "as in 2D.</li>"
           "<li>Trust z over the calibrated range, and filter to it.</li></ul>",
           say="That is 3D fitting: measure the PSF on beads, check it, and fit "
               "with it. Next: rendering the result.")
