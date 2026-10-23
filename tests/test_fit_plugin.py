"""The fitters as plugins: parts, nesting, a headless run, streaming."""
import numpy as np
import pytest
import tifffile

from smappy import plugins
from smappy.plugins import Context
from smappy.plugins.fit import (CameraSettings, DetectionSettings, GaussianFit,
                                GaussianFitSettings, OutputSettings, SourceSettings)

SHAPE = (48, 48)


@pytest.fixture
def stack(tmp_path):
    """20 frames, a few bright emitters each, plain TIFF (no metadata)."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    frames = []
    for _ in range(20):
        clean = np.zeros(SHAPE)
        for _ in range(3):
            cx, cy = rng.uniform(8, SHAPE[1] - 8), rng.uniform(8, SHAPE[0] - 8)
            clean += 3000.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 2.0)
        frames.append((rng.poisson(clean) + 100).astype(np.uint16))
    path = tmp_path / "run.tif"
    tifffile.imwrite(path, np.stack(frames))
    return path


def test_parts_show_up_as_children():
    specs = GaussianFit.specs()
    assert set(specs) == {"source", "camera", "detection", "region", "model", "fit",
                          "output"}
    assert specs["fit"].children["roisize"].info.label == "fit box"
    assert specs["source"].children["path"].info.kind == "open_file"
    assert "Localize/Spline 3D" in plugins.available("Localize/")


def test_every_fitter_shows_the_shared_parts_in_one_order_and_its_own_in_place():
    """The forms are declared in halves around the model (fit.py, "the
    forms"); a part only one fitter has sits where it always did."""
    from smappy.plugins.fit import DualGaussianFit, DualSplineFit, SplineFit
    shared = ["source", "camera", "detection", "region", "model", "fit", "output"]
    assert list(GaussianFit.specs()) == shared
    assert list(SplineFit.specs()) == shared
    assert list(DualSplineFit.specs()) == shared[:5] + ["transform"] + shared[5:] + ["finish"]
    assert list(DualGaussianFit.specs()) == (shared[:5] + ["transform"]
                                             + shared[5:] + ["finish"])
    # the model is each fitter's own, and the two-channel ones share links
    assert [list(cls.specs()["model"].children) for cls in
            (DualSplineFit, DualGaussianFit)] == [
        ["calibration", "link_xy", "link_z", "link_photons", "link_background",
         "photon_ratio"],
        ["sigma", "link_xy", "link_photons", "link_background", "link_sigma",
         "photon_ratio"]]


def test_fit_runs_without_a_gui_and_streams(stack, tmp_path):
    settings = GaussianFitSettings(
        source=SourceSettings(path=str(stack), chunk=8),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=40.0),
        output=OutputSettings(path=str(tmp_path / "out.hdf5")))
    events, messages = [], []
    ctx = Context(progress=messages.append, stream=lambda e, p: events.append((e, p)))
    result = GaussianFit().run(ctx, settings)
    assert messages and "localizations" in messages[-1]
    assert 50 <= len(result.locs) <= 60          # 3 per frame, some overlap
    assert (tmp_path / "out.hdf5").exists()
    assert events[0][0] == "start" and events[0][1]["extent"] is not None
    blocks = [p for e, p in events if e == "block"]
    assert sum(len(b) for b in blocks) == len(result.locs)
    assert result.locs["x_nm"].max() < 4800


def test_every_camera_comes_through_the_one_resolution(stack):
    """The fit and its parameter view resolve a camera
    through `CameraSettings.resolution`, so neither can disagree with the fit."""
    from smappy.io.tiff import open_stack
    from smappy.plugins.fit import camera_resolution
    camera = CameraSettings(conversion=2.0, offset=100.0, pixelsize_um=0.1)
    settings = GaussianFitSettings(source=SourceSettings(path=str(stack)),
                                   camera=camera)
    source = open_stack(stack)
    fitted = camera.resolve(source)
    assert fitted.conversion == 2.0 and fitted.offset == 100.0
    assert camera_resolution(settings).camera_metadata() == fitted
    assert GaussianFit()._camera(settings) == fitted


def test_the_default_output_goes_next_to_the_image_folder(tmp_path):
    """Named by the acquisition, beside the folder its images are in."""
    from smappy.plugins.fit import default_output_path
    singles = tmp_path / "Nup96_EM100_2_Pos0"
    singles.mkdir()
    for i in range(3):
        tifffile.imwrite(singles / f"img_{i:09d}_Default_000.tif",
                         np.zeros((4, 4), np.uint16))
    expected = tmp_path / "Nup96_EM100_2_Pos0_locs.hdf5"
    assert default_output_path(singles) == expected
    assert default_output_path(singles / "img_000000001_Default_000.tif") == expected

    series = tmp_path / "run_1"
    series.mkdir()
    tifffile.imwrite(series / "run_1_MMStack_Pos0.ome.tif", np.zeros((2, 4, 4), np.uint16))
    assert default_output_path(series / "run_1_MMStack_Pos0.ome.tif") == \
        tmp_path / "run_1_locs.hdf5"

    lone = tmp_path / "stack.tif"
    tifffile.imwrite(lone, np.zeros((2, 4, 4), np.uint16))
    assert default_output_path(lone) == tmp_path / "stack_locs.hdf5"


def test_choosing_a_source_fills_the_output_but_keeps_a_typed_one(stack, tmp_path):
    plugin = GaussianFit()
    settings = GaussianFitSettings(source=SourceSettings(path=str(stack)))
    updates = plugin.react("source.path", settings)
    assert updates["output.path"].endswith("_locs.hdf5")

    # the default follows a new source while it is still the default ...
    settings.output.path = updates["output.path"]
    other = tmp_path / "other.tif"
    tifffile.imwrite(other, np.zeros((2, 8, 8), np.uint16))
    settings.source.path = str(other)
    assert plugin.react("source.path", settings)["output.path"] == \
        str(tmp_path / "other_locs.hdf5")

    # ... and a path the user typed is left alone
    settings.output.path = str(tmp_path / "mine.hdf5")
    assert "output.path" not in (plugin.react("source.path", settings) or {})


def test_progress_counts_the_frames_and_says_how_fast(stack, tmp_path):
    """A fit runs for minutes, and a count with no rate says nothing about
    whether it is worth waiting for."""
    import re

    settings = GaussianFitSettings(
        source=SourceSettings(path=str(stack), chunk=5),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=40.0),
        output=OutputSettings(path=""))
    messages = []
    result = GaussianFit().run(Context(progress=messages.append), settings)

    assert len(messages) == 4                          # 20 frames in blocks of 5
    counted = [int(re.search(r"frame ([\d,]+) of", m).group(1).replace(",", ""))
               for m in messages]
    assert counted == [5, 10, 15, 20]                  # it counts up
    assert all("of 20 " in m and "frames/s" in m for m in messages)
    assert "left" in messages[0] and "left" not in messages[-1]   # not at the end
    assert "frames/s" in result.text


def test_a_fit_keeps_its_warnings_in_its_text_its_history_and_its_file(stack, tmp_path,
                                                                       monkeypatch):
    """An EM-gain mismatch was reported as a progress line, which the frame
    count overwrote: nothing on screen or on disk said the fit had warned."""
    import warnings

    from smappy.io.hdf5 import load_localizations
    from smappy.session import Session

    model = GaussianFit.model

    def warning_model(self, settings, camera):
        warnings.warn("EM gain mismatch: the calibration was taken with EM gain")
        return model(self, settings, camera)

    monkeypatch.setattr(GaussianFit, "model", warning_model)
    out = tmp_path / "warned.hdf5"
    settings = GaussianFitSettings(
        source=SourceSettings(path=str(stack), chunk=8),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=40.0),
        output=OutputSettings(path=str(out)))
    events = []
    ctx = Context(stream=lambda e, p: events.append((e, p)))
    with pytest.warns(UserWarning):
        result = GaussianFit().run(ctx, settings)
    assert [p for e, p in events if e == "warning"] == ctx.warnings
    assert "warning: EM gain mismatch" in result.text
    assert any("EM gain mismatch" in w for w in result.locs.metadata["warnings"])
    assert any("EM gain mismatch" in w
               for w in load_localizations(out).metadata["warnings"])

    session = Session()
    session.apply(GaussianFit(), result)
    assert "warning: EM gain mismatch" in session.history[-1]["text"]


def test_a_fit_without_warnings_writes_none(stack, tmp_path):
    out = tmp_path / "clean.hdf5"
    settings = GaussianFitSettings(
        source=SourceSettings(path=str(stack), chunk=8),
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=40.0),
        output=OutputSettings(path=str(out)))
    result = GaussianFit().run(Context(), settings)
    assert "warning:" not in result.text
    from smappy.io.hdf5 import load_localizations
    assert "warnings" not in load_localizations(out).metadata


def test_a_stack_with_no_pixel_size_is_fitted_on_an_assumed_one(stack, tmp_path):
    """It used to be refused.  The nm columns use the default size, and the
    table and the run both say so -- the table with the size, so the pixels
    divide back out exactly."""
    from smappy.io.hdf5 import load_localizations
    from smappy.metadata import DEFAULT_PIXELSIZE_NM
    settings = GaussianFitSettings(
        source=SourceSettings(path=str(stack)),
        camera=CameraSettings(conversion=1.0, offset=100.0),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=40.0),
        output=OutputSettings(path=str(tmp_path / "out.hdf5")))
    ctx = Context()
    result = GaussianFit().run(ctx, settings)
    assert any("assumed" in w for w in ctx.warnings)
    locs = result.locs
    assert len(locs) and locs.metadata["pixelsize_nm"] == DEFAULT_PIXELSIZE_NM
    # near where the peak finder put it, in pixels of that size
    assert np.median(np.abs(locs["x_nm"] / DEFAULT_PIXELSIZE_NM - locs["peak_x_pix"])) < 0.5
    saved = load_localizations(tmp_path / "out.hdf5")
    assert saved.metadata.get("pixelsize_assumed") is True
    assert saved.metadata.get("pixelsize_nm") == DEFAULT_PIXELSIZE_NM
    assert "x_pix" not in saved and "x_nm" in saved


def test_switching_the_cutoff_mode_puts_that_modes_default_in_the_value():
    """1.7 noise widths kept as 1.7 photons would find every pixel: the
    number follows the mode it is read in, in a two-channel fitter too."""
    from dataclasses import replace

    from smappy.plugins.fit import DualGaussianFit, DualGaussianFitSettings

    settings = GaussianFitSettings(detection=DetectionSettings(cutoff_mode="absolute"))
    assert GaussianFit().react("detection.cutoff_mode", settings) == \
        {"detection.cutoff": 20.0}
    settings = replace(settings, detection=DetectionSettings(cutoff_mode="dynamic",
                                                             cutoff=35.0))
    assert GaussianFit().react("detection.cutoff_mode", settings) == \
        {"detection.cutoff": 1.7}
    dual = DualGaussianFitSettings(detection=DetectionSettings(cutoff_mode="absolute"))
    assert DualGaussianFit().react("detection.cutoff_mode", dual) == \
        {"detection.cutoff": 20.0}


# ------------------------------------------------------------ an unknown camera

def test_a_camera_nobody_entered_is_asked_about_at_run_until_its_numbers_are_typed(stack):
    from smappy.plugins import PreflightQuestion

    bare = GaussianFitSettings(source=SourceSettings(path=str(stack)))
    question = GaussianFit().preflight(Context(), bare)
    assert isinstance(question, PreflightQuestion) and not question.run_label
    assert "not in the camera database" in question.text
    [choice] = question.choices
    assert choice.window == "camera editor" and choice.settings is bare

    typed = GaussianFitSettings(source=SourceSettings(path=str(stack)),
                                camera=CameraSettings(conversion=1.0, offset=100.0,
                                                      pixelsize_um=0.1))
    assert GaussianFit().preflight(Context(), typed) is None


def test_the_camera_question_opens_the_editor_and_does_not_run(stack, monkeypatch):
    """Cancel is the default; the editor is the other answer, and neither
    starts the fit."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from smappy.gui.plugin_panel import PluginPanel
    from smappy.locs import Localizations
    from smappy.session import Session

    QApplication.instance() or QApplication([])
    session = Session(Localizations())
    opened = []
    session.window_openers["camera editor"] = opened.append
    panel = PluginPanel(GaussianFit, session)
    panel.form.set_values({"source.path": str(stack)})

    question = GaussianFit().preflight(Context(), panel.form.value())
    box, answers = panel.choice_box("t", question)
    assert box.defaultButton() is box.escapeButton()          # Cancel

    monkeypatch.setattr(PluginPanel, "ask_choice",
                        lambda self, title, q: q.choices[0])
    assert panel.start_run() is False
    assert len(opened) == 1 and opened[0].source.path == str(stack)
    assert panel._thread is None


def test_the_camera_part_is_folded_and_titled_by_what_it_resolves_to(stack):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from smappy.gui.plugin_panel import PluginPanel
    from smappy.locs import Localizations
    from smappy.session import Session

    QApplication.instance() or QApplication([])
    panel = PluginPanel(GaussianFit, Session(Localizations()))
    section = panel.form.sections["camera"]
    assert not section.button.isChecked()
    panel.form.set_values({"source.path": str(stack), "camera.conversion": 0.5,
                           "camera.pixelsize_um": 0.108})
    panel._describe()
    # no microscope recognises a simulated stack: the camera's numbers stand in the line
    assert section.title.startswith("no microscope recognised \u00b7 0.5 e-/ADU \u00b7 108")
    assert GaussianFit.specs()["output"].children["show_tags"].info.advanced
