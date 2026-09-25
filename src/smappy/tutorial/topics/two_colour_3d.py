"""Two colours in 3D: a dual-colour bead calibration, Spline 3D 2C, the colours.

The two tutorials before it hold the ideas -- the photon split that is the
colour (``two_colour``) and a PSF measured on beads (``fitting_3d``) -- so
this one is what is new when they meet: the calibration window in dual-colour
mode, where the beads give both the transformation between the halves and a
PSF for each, and the fitter that links x, y and z across the halves while
leaving the photons free.  It ends as the 2D one does
(`two_colour.colours_and_layers`).

Simulated with `simulate.dual_bead_stacks` and `dual_camera_frames` behind
one `dual_transformation` and one astigmatism, so the storyboard checks that
the calibration finds the transformation, that z comes back, and that the
colours are the dyes -- the whole chain, beads to colours, which no other
test runs end to end.  The save dialog is not driven, as in ``fitting_3d``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import field, menu, menu_item, open_section, section_of, show_tab, type_into
from .fitting_3d import _wait, _z_slope
from .two_colour import after_the_fit, colours_and_layers

TITLE = "Two colours in 3D: dual-colour calibration and Spline 3D 2C"
DESCRIPTION = ("A dual-colour bead calibration for the transformation and a PSF "
               "per half, fitting with Spline 3D 2C, and the colours.")

CONVERSION, OFFSET, PIXELSIZE_UM, DZ_NM = 0.5, 100.0, 0.1, 20.0


def _data(d):
    import tifffile
    from ...simulate import ASTIGMATISM, dual_bead_stacks, dual_camera_frames
    folder = d.data_dir() / "dual beads"
    folder.mkdir(parents=True, exist_ok=True)
    camera = dict(conversion=CONVERSION, offset=OFFSET, pixelsize_nm=PIXELSIZE_UM * 1000)
    stacks, _ = dual_bead_stacks(3, seed=0, dz_nm=DZ_NM, **camera)
    paths = []
    for i, stack in enumerate(stacks):
        path = folder / f"dual_beads_{i + 1}.tif"
        tifffile.imwrite(path, stack)
        paths.append(path)
    movie = d.data_dir() / "demo_two_colour_3d.tif"
    frames, truth = dual_camera_frames(2000, seed=1, astigmatism=ASTIGMATISM, **camera)
    tifffile.imwrite(movie, frames)
    return paths, movie, truth


def _transformation_error(calibration) -> float:
    """Largest distance, in pixels, between the calibration's map and the
    simulated one over the chip's corners and centre."""
    from ...simulate import dual_transformation
    probe = np.array([[5.0, 105.0], [95.0, 105.0], [5.0, 195.0], [95.0, 195.0],
                      [50.0, 150.0]])
    ones = np.c_[probe, np.ones(len(probe))]
    found = ones @ np.asarray(calibration.transformation).T
    found = found[:, :2] / found[:, 2:]
    true = (ones @ dual_transformation().T)[:, :2]
    return float(np.abs(found - true).max())


def make(d) -> None:
    session, control = d.session, d.control
    stacks, movie, truth = _data(d)

    d.chapter("What is new in 3D")
    d.card(TITLE,
           "<p>Two dyes on a split camera, as in 2D, and z from a measured PSF, "
           "as in 3D fitting.</p>"
           "<p>What is new: the beads give both the map between the halves and "
           "a PSF for each half, and the fit shares x, y and z between them.</p>",
           say="Two colours in 3D combines the two: a split camera, and a PSF "
               "measured on beads, now one for each half.")

    d.chapter("Dual-colour calibration")
    tools = menu(d, "Tools")
    item = menu_item(d, tools, "Dual-colour calibration")
    d.shot("Tools, Dual-colour calibration opens the bead calibration in its "
           "two-colour mode.",
           spot=[item], point=item, click=True, zoom=d.around(tools, 700))
    tools.close()
    control.open_calibration(dual=True)
    window = control.calibration_window
    assert window.is_dual, "the calibration window is not in dual-colour mode"
    window.resize(1540, 860)
    d.place(window, 30, 20)
    window.add_paths([str(p) for p in stacks])
    form = window.form
    for name, value in (("layout", "up-down"), ("main_channel", "upper"),
                        ("dz_nm", DZ_NM)):
        f = form.fields[name]
        f.set(value)
        f.changed.emit()
    d.settle()
    d.shot("Bead stacks from the split camera: every bead appears in both "
           "halves, and the pairs are what the map is measured from.",
           spot=[window.file_list, window.mode_box], point=window.mode_box,
           zoom=d.around(window.file_list, 760))
    layout = form.fields["layout"]
    main = form.fields["main_channel"]
    d.shot("Say how the chip is split: here into an upper and a lower half, "
           "with the main channel above. Mirrored is for a splitter that flips "
           "one half.",
           spot=[layout, main], point=layout, zoom=d.around(layout, 700))

    d.chapter("Calibrate")
    d.shot("Detect and calibrate pairs the beads across the halves, fits the "
           "map, and builds a PSF for each half.",
           spot=[window.run_button], point=window.run_button, click=True,
           zoom=d.around(window.run_button, 700))
    window.run()
    _wait(d, window)
    result = window.result
    assert result is not None, "the dual calibration produced nothing"
    # the words say the beads give the map between the halves
    error = _transformation_error(result.calibration)
    assert error < 0.2, f"the calibration's map is {error:.2f} px off"
    d.shot("Overview: each bead and its partner in the other half, and where "
           "the pairs lie on the chip.",
           spot=[window.tabs])
    window.tabs.setCurrentIndex(1)
    d.settle()
    d.shot("Transformation: how far each pair lands from where the map puts it. "
           "A tight cloud around zero is a good map.",
           spot=[window.tabs])
    d.shot("As in one colour, the table lists the beads; a pair left out of the "
           "PSF can still help the map.",
           spot=[window.bead_table], point=window.bead_table)

    d.chapter("Save and use")
    from ...calibrate.gui import calibration_save_defaults
    folder, name = calibration_save_defaults(window.paths, dual=True)
    path = Path(folder) / (name + ".h5")
    result.save(path, overwrite=True)
    window.saved_path, window.saved_dual = str(path), True
    window.use_button.setEnabled(True)
    d.settle()
    d.shot("Save it, then Use in the fitter puts it into Spline 3D 2C.",
           spot=[window.save_button, window.use_button], point=window.use_button,
           click=True, zoom=d.around(window.use_button, 700))
    window.use_button.click()
    d.settle()
    window.hide()
    d.settle()

    d.chapter("Fit in 3D, two colours")
    localize = show_tab(d, "Localize")
    _, panel = open_section(d, localize, "Spline 3D 2C")
    calibration = field(panel, "model.calibration")
    assert Path(calibration.value()).name == path.name, calibration.value()
    type_into(panel, "source.path", str(movie))
    for name, value in (("conversion", CONVERSION), ("offset", OFFSET),
                        ("pixelsize_um", PIXELSIZE_UM)):
        type_into(panel, f"camera.{name}", value)
    d.settle()
    d.shot("Spline 3D 2C has the calibration. It knows the split from it, so "
           "only the acquisition and the camera are left.",
           spot=[calibration, field(panel, "source.path")], point=calibration,
           zoom=d.around(calibration, 760))
    model = section_of(panel, "model")
    d.shot("A molecule is at one place in both halves, so x, y and z are shared. "
           "Its photons are not: how they split is its colour.",
           spot=[model], zoom=d.around(model, 760))
    after_the_fit(d, panel)
    d.shot("Run.", spot=[panel.run_button], point=panel.run_button, click=True,
           zoom=d.around(panel.run_button, 760))
    panel.run_button.click()
    d.settle()
    locs = session.locs
    assert "z_nm" in locs and "photons_ch1" in locs, sorted(locs.keys())
    slope = _z_slope(locs, truth)
    assert abs(slope - 1) < 0.02, f"fitted z against true z has slope {slope:.3f}"
    d.shot("Each localization has a z, its photons in each half, and a colour.",
           spot=[panel.output], zoom=d.around(panel.output, 760))

    colours_and_layers(d, panel, truth)

    d.chapter("Next")
    d.card("What to remember",
           "<ul><li><b>Dual-colour calibration:</b> the beads give the map "
           "between the halves and a PSF for each; say how the chip is split.</li>"
           "<li><b>Spline 3D 2C:</b> x, y and z shared, the photons free.</li>"
           "<li><b>Assign colours</b> runs at the end of the fit; a layer per "
           "channel shows the colours.</li></ul>",
           say="That is two colours in 3D: calibrate on beads, fit with Spline 3D "
               "2C, and assign the colours.")
