"""The DECODE training window (`smappy.gui.train_window`): preparing a job from a movie, the list, the curves.

Torch-free, as the window is: the prefit is SMAPpy's own Spline 3D, the job a folder of files.  The
training itself, here, is tests/decode_tests/test_train_window_run.py.
"""
import json
import sys

import numpy as np
import pytest
import tifffile

from smappy.io.calibration import SplineCalibration, save_spline_calibration


def _calibration(path):
    """A round spline, 21 x 21 pixels and 41 slices of 50 nm, the PSF widening away from focus."""
    z = np.arange(41) - 20.0
    yy, xx = np.mgrid[:21, :21] - 10.0
    stack = np.stack([np.exp(-0.5 * (xx ** 2 + yy ** 2) / (1.3 + 0.002 * k ** 2) ** 2) for k in z])
    stack /= stack.sum(axis=(1, 2)).max()
    from smappy.calibrate.core import spline_coefficients
    cal = SplineCalibration(spline_coefficients(stack.astype(np.float32)), dz=50.0, z0=20.0, x0=10.0, em_on=False)
    save_spline_calibration(path, cal)
    return path


@pytest.fixture
def movie(tmp_path):
    """300 frames of 48 x 48 with spots drawn with the calibration, an sCMOS-like camera."""
    from smappy.io.calibration import load_spline_calibration
    from smappy.simulate.camera import _camera, _draw_spline
    cal_path = _calibration(tmp_path / "beads.h5")
    cal = load_spline_calibration(cal_path)
    rng = np.random.default_rng(2)
    n_frames = 300
    frame = np.repeat(np.arange(n_frames), rng.poisson(4, n_frames))
    xy = rng.uniform(6, 42, (len(frame), 2))
    image = np.full((n_frames, 48, 48), 20.0)
    _draw_spline(image, frame, xy, rng.uniform(-400, 400, len(frame)), rng.normal(3000, 500, len(frame)), cal)
    frames = np.stack([_camera(image[k], np.random.default_rng(k), 0.5, 100.0, 1.5) for k in range(n_frames)])
    path = tmp_path / "data" / "cells.tif"
    path.parent.mkdir()
    tifffile.imwrite(path, frames, photometric="minisblack")
    return path, cal_path


def _prepare_settings(movie, cal_path):
    from smappy.gui.train_window import PrepareSettings
    from smappy.plugins.fit import CameraSettings
    return PrepareSettings(movie=str(movie), calibration=str(cal_path), density="LD", blocks=3, block_frames=50,
                           camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1, em_on=False))


def test_a_job_prepared_from_a_movie_lands_beside_it_and_is_listed(movie):
    from smappy.gui.train_window import default_folder, estimate_settings, run_prefit, with_settings, write_job
    from smappy.training import job as jobfile
    from smappy.training import jobs
    path, cal_path = movie
    prepared = run_prefit(_prepare_settings(path, cal_path))
    est = prepared.estimates
    assert est.n_localizations > 100 and 2000 < est.intensity_mu_sig[0] < 4500
    edited = estimate_settings(est)
    edited.photons, edited.z_min_nm, edited.z_max_nm = 3333.0, -500.0, 500.0     # as typed in the form
    folder = write_job(prepared, with_settings(est, edited), default_folder(path, cal_path))
    assert folder.parent == path.parent / "cells_decode" and folder.name.startswith("beads_")
    job, params = jobfile.read(folder)
    assert params["Simulation"]["intensity_mu_sig"][0] == 3333.0
    assert params["Simulation"]["emitter_extent"][2] == [-500, 500]
    assert job.data["path"] == str(path.resolve()) and job.camera["e_per_adu"] == 0.5
    assert jobfile.check(folder) == []
    assert any(e.folder == str(folder.resolve()) for e in jobs.entries())
    assert default_folder(path, cal_path) != folder                    # a second job does not overwrite it


def test_the_progress_curves_say_where_a_training_is_and_how_long_it_has_left(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from smappy.gui.train_window import draw_progress
    from smappy.training import progress
    progress.append(tmp_path, "start", 0, epochs=100)
    for e in range(5):
        progress.append(tmp_path, "epoch", e, loss_train=None if e == 0 else 500.0 - e, loss_test=480.0 - e,
                        seconds=12.0, metrics={"jaccard": 0.1 * e, "rmse_lat_nm": 40.0, "rmse_ax_nm": 60.0})
    text = draw_progress(Figure(), progress.read(tmp_path))
    assert text == "epoch 4 of at most 100, 12.0 s an epoch, 1 min so far, at most 19 min left"
    progress.append(tmp_path, "finished", 4, converged=True)
    assert draw_progress(Figure(), progress.read(tmp_path)).endswith("1 min in all")   # no time left
    assert draw_progress(Figure(), []) == "no epoch yet"


def test_the_window_lists_the_jobs_and_says_why_it_cannot_train_without_torch(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui import train_window
    from smappy.training import jobs, progress
    folder = tmp_path / "job"
    folder.mkdir()
    (folder / "job.json").write_text(json.dumps({}))
    progress.append(folder, "start", 0, epochs=10)
    jobs.add(folder)
    w = train_window.TrainWindow()
    try:
        rows = {w.job_list.item(i, 1).text(): i for i in range(w.job_list.rowCount())}
        row = rows["job"]
        assert w.job_list.item(row, 0).text() == train_window.SYMBOLS["failed"]      # running, and no lock: died
        assert w.job_list.item(row, 2).text().startswith("died")
        w.job_list.setCurrentCell(row, 0)
        monkeypatch.setattr(train_window, "torch_here", lambda: False)
        w.start()
        assert "no torch" in w.train_note.text()
        w.stop()
        assert (folder / "STOP").exists()
    finally:
        w.close()
        w.timer.stop()


def test_starting_here_runs_this_pythons_decode_cli_detached_with_its_log(tmp_path, monkeypatch):
    from smappy.gui import train_window
    seen = {}

    class Fake:
        def __init__(self, command, **kw):
            seen.update(command=command, **kw)
    monkeypatch.setattr(train_window.subprocess, "Popen", Fake)
    train_window.start_here(tmp_path, train_window.TrainSettings(epochs=7, device="mps"))
    assert seen["command"][:4] == [sys.executable, "-m", "decode_smlm.cli", "train"]
    command = seen["command"]
    assert command[5:9] == ["--epochs", "7", "--device", "mps"] and seen["cwd"] == str(tmp_path)
    assert (tmp_path / "train.log").exists()
    # the Train tab's rule goes along; without the tick every epoch is trained
    rule = json.loads(command[command.index("--stop-rule") + 1])
    assert rule == {"window": 100, "min_epochs": 150, "loss_tol": 0.01, "jaccard_tol": 0.005, "eff_tol": 0.005}
    train_window.start_here(tmp_path, train_window.TrainSettings(epochs=7, until_converged=False))
    assert seen["command"][-1] == "--all-epochs" and "--stop-rule" not in seen["command"]


def test_a_sparse_movie_is_prefitted_further_until_molecules_were_fully_on_often_enough(movie):
    from decode_smlm.prefit import FULLY_ON_ENOUGH
    from smappy.gui.train_window import run_prefit
    path, cal_path = movie
    settings = _prepare_settings(path, cal_path)
    settings.blocks, settings.block_frames = 1, 10
    messages = []
    prepared = run_prefit(settings, report=messages.append)
    assert any("too few" in m["message"] for m in messages)
    assert prepared.estimates.n_fully_on >= FULLY_ON_ENOUGH or prepared.estimates.n_localizations > 600


def test_the_crlb_reference_is_the_gaussian_bound_of_a_bright_spot_and_doubles_in_variance_with_em_gain(tmp_path):
    from smappy.io.calibration import load_spline_calibration
    from smappy.training.reference import crlb_reference
    cal = load_spline_calibration(_calibration(tmp_path / "beads.h5"))
    params = {"Simulation": {"intensity_mu_sig": [4000.0, 0.0], "bg_uniform": [0.01, 0.02],
                             "emitter_extent": [None, None, [0.0, 0.0]]},
              "Camera": {"px_size": [100.0, 100.0], "em_gain": None}}
    # a Gaussian of sigma 1.3 px at focus, without background: sigma_a / sqrt(N) an axis, sigma_a^2
    # = sigma^2 + a^2 / 12 for pixels of size a (Thompson et al. 2002; Mortensen et al. 2010 add a
    # factor near 1 for an MLE without background)
    from smappy.simulate.camera import _draw_spline
    spot = np.zeros((1, 13, 13))
    _draw_spline(spot, np.zeros(1, int), np.full((1, 2), 6.0), np.zeros(1), np.full(1, 4000.0), cal)
    n = spot.sum()                 # this calibration is normalised on its widest slice: 38 % at focus
    expected = np.sqrt(2) * np.sqrt(1.3 ** 2 + 1 / 12) * 100.0 / np.sqrt(n)
    found = crlb_reference(params, cal, spots=200)
    assert found["lateral_nm"] == pytest.approx(expected, rel=0.1)
    params["Camera"]["em_gain"] = 300.0
    assert crlb_reference(params, cal, spots=200)["lateral_nm"] == pytest.approx(np.sqrt(2) * found["lateral_nm"], rel=1e-3)


def test_the_rmse_panel_draws_the_curves_alone(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from smappy.gui.train_window import draw_progress
    from smappy.training import progress
    progress.append(tmp_path, "start", 0, epochs=10)
    progress.append(tmp_path, "epoch", 0, loss_train=1.0, loss_test=1.0, seconds=1.0,
                    metrics={"jaccard": 0.5, "rmse_lat_nm": 40.0, "rmse_ax_nm": 60.0})
    figure = Figure()
    draw_progress(figure, progress.read(tmp_path))
    rmse = next(a for a in figure.axes if a.get_title().startswith("RMSE"))
    assert [line.get_label() for line in rmse.get_lines()] == ["lateral", "axial"]


def test_without_a_prefit_the_window_writes_the_job_from_the_estimates_typed_in(movie):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui import train_window
    from smappy.training import job as jobfile
    path, cal_path = movie
    w = train_window.TrainWindow()
    try:
        w.prepare_form.set(_prepare_settings(path, cal_path))
        w.estimate_form.set_values({"photons": 4321.0, "background_min": 15.0, "background_max": 90.0,
                                    "z_min_nm": -600.0, "z_max_nm": 600.0})
        assert w.estimate_form.isEnabled() and w.write_button.isEnabled()
        w.write()
        folder = w.selected_folder()
        assert folder is not None and "job written" in w.status_bar.currentMessage()
        job, params = jobfile.read(folder)
        sim = params["Simulation"]
        assert sim["intensity_mu_sig"][0] == 4321.0 and sim["bg_uniform"] == [15.0, 90.0]
        assert sim["emitter_extent"][2] == [-600, 600] and job.camera["e_per_adu"] == 0.5
        assert "microscope" in job.calibration and "em_on" in job.calibration    # for the fitter's check
        assert jobfile.check(folder) == []
    finally:
        w.close()
        w.timer.stop()


def test_the_window_takes_the_fitters_run_on_and_opens_the_training_hosts(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui import train_window
    from smappy.plugins.fit_decode import DecodeFitSettings
    opened = []
    w = train_window.TrainWindow(open_hosts=lambda: opened.append(True))
    try:
        settings = DecodeFitSettings()
        settings.source.path = str(tmp_path / "cells.tif")
        settings.model.run_on = "here"
        w.take_fitter(settings)
        assert w.prepare_form.value().movie == settings.source.path
        assert w.train_form.value().run_on == "here"
        w.hosts_button.click()
        assert opened == [True]
    finally:
        w.close()
        w.timer.stop()


def test_the_table_groups_the_states_running_first_with_a_symbol_each_and_marks_jobs_from_elsewhere():
    from smappy.gui.train_window import ORDER, found_row, job_row, state_text
    from smappy.training import jobs
    from smappy.training.remote import Found
    done = jobs.Status("finished", epoch=311, epochs=1000, converged=True,
                       last={"metrics": {"jaccard": 0.371, "rmse_lat_nm": 47.2}})
    assert done.group == "done" and state_text(done) == "done, converged at epoch 311"
    from smappy.gui.train_window import state_word
    assert state_word(done) == "done, converged"                          # the Follow line: no epoch
    row = job_row(jobs.Entry("/x/job"), done)
    assert row["result"] == "test set: Jaccard 0.371, lateral 47 nm" and row["run_on"] == "this computer"
    assert state_text(jobs.Status("finished", epoch=999)) == "done, all 1000 epochs"
    assert jobs.Status("stopped", epoch=40).group == "paused"
    assert [jobs.Status(s).group for s in ("interrupted", "error", "new", "running")] == \
        ["failed", "failed", "not started", "running"]
    assert state_text(jobs.Status("running", epoch=12, epochs=1000)) == "running, epoch 12 of 1000"
    other = found_row(Found("gpuvm", "~/smappy_training/theirs", "running", epoch=5))
    assert other["foreign"] and other["name"].endswith("theirs") and other["run_on"] == "gpuvm"
    assert ORDER[0] == "running" and ORDER[-1] == "done"


def test_a_job_can_start_from_a_model_of_the_same_network_and_refuses_another(movie, tmp_path):
    from smappy.gui.train_window import BASE_MODEL, default_folder, run_prefit, write_job
    from smappy.training import job as jobfile
    from smappy.training import model_file
    path, cal_path = movie
    prepared = run_prefit(_prepare_settings(path, cal_path))
    good, other = tmp_path / "earlier.safetensors", tmp_path / "other.safetensors"
    model_file.write(good, {"w": np.zeros(2, np.float32)},
                     {"method": "decode-v0.10.2", "architecture": "SigmaMUNet", "channels_in": 3})
    model_file.write(other, {"w": np.zeros(2, np.float32)},
                     {"method": "decode-v0.10.2", "architecture": "SigmaMUNet", "channels_in": 5})
    prepared.settings.start_from = str(good)
    folder = write_job(prepared, prepared.estimates, default_folder(path, cal_path))
    job, _ = jobfile.read(folder)
    assert job.base_model == BASE_MODEL and (folder / BASE_MODEL).read_bytes() == good.read_bytes()
    assert jobfile.check(folder) == []
    prepared.settings.start_from = str(other)
    with pytest.raises(ValueError, match="another network"):
        write_job(prepared, prepared.estimates, default_folder(path, cal_path))


def test_the_connection_guide_opens_above_preferences_and_outlives_it():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout
    app = QApplication.instance() or QApplication([])
    from smappy.gui import preferences
    dialog = QDialog()
    page = preferences.HostsPage()
    QVBoxLayout(dialog).addWidget(page)
    page.guide_button.click()
    guide = preferences._GUIDE
    # modal itself, so it can be scrolled while Preferences is open; not the dialog's child, so the
    # dialog going does not take it along (that crashed SMAPpy on macOS)
    assert guide.isModal() or guide.windowModality() == Qt.ApplicationModal
    assert guide.parent() is None and guide.current == "Panels/Training hosts"
    dialog.deleteLater()
    app.processEvents()
    assert guide.isVisible()
    guide.close()
    app.processEvents()


def test_a_prefit_whose_result_starts_the_simulated_frames_does_not_abort(movie, monkeypatch):
    """*Prefit*'s result starts the next job, the simulated frames, from the worker's own signal:
    the window replaced its QThread while that one was still winding down, and Qt aborted the
    program ("QThread: Destroyed while thread is still running") after every prefit with torch."""
    pytest.importorskip("PySide6")
    import time
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from smappy.gui import train_window
    path, cal_path = movie
    monkeypatch.setattr(train_window, "torch_here", lambda: True)
    monkeypatch.setattr(train_window, "simulated_frames", lambda params, cal, n_frames=2: np.zeros((n_frames, 40, 40)))
    w = train_window.TrainWindow()
    try:
        w.prepare_form.set(_prepare_settings(path, cal_path))
        w.prefit()
        deadline = time.monotonic() + 60
        while w.simulated is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert w.prepared is not None and w.simulated is not None
        w.write()                                          # what was asked for when it crashed
        assert "job written" in w.status_bar.currentMessage()
    finally:
        if w._thread is not None:
            w._thread.quit()
            w._thread.wait()
        w.close()
        w.timer.stop()


def test_the_curves_leave_the_first_epochs_out_of_their_range_and_show_the_efficiencies():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from smappy.gui.train_window import draw_progress
    lines = [{"kind": "start", "epoch": 0, "epochs": 1000}]
    for e in range(200):
        far = e < 3                                   # the first epochs: a loss of thousands, an RMSE of 400 nm
        lines.append({"kind": "epoch", "epoch": e, "loss_train": 5000.0 if far else 300.0 - 0.1 * e,
                      "loss_test": 4000.0 if far else 290.0 - 0.1 * e, "seconds": 9.0,
                      "metrics": {"jaccard": 0.3, "rmse_lat_nm": 400.0 if far else 45.0, "rmse_ax_nm": 70.0,
                                  "eff_lat": 0.2, "eff_ax": 0.25, "crlb_width_lat": 1.9, "crlb_width_ax": 2.1,
                                  "pixel_hist_rows": [12] * 10 + [8] * 10, "pixel_hist_cols": [10] * 20}})
    figure = Figure()
    draw_progress(figure, lines)
    loss, jaccard, pixels, rmse, crlb, seconds = figure.axes
    assert loss.get_ylim()[1] < 400 and rmse.get_ylim()[1] < 100      # the first epochs beyond the edge
    assert rmse.get_ylim()[0] > 30                                     # no CRLB line stretches it down
    assert crlb.get_title() == "error / CRLB, width (test set)" and crlb.get_ylim()[0] < 1.0 < crlb.get_ylim()[1]
    assert pixels.get_title() == "pixelation, epoch 199" and seconds.get_title() == "seconds an epoch"
    # the stop rule's window: shaded once, and a line at its median, on the curves it judges
    assert len(jaccard.patches) == 1 and any(line.get_linestyle() == "--" for line in jaccard.lines)


def test_the_follow_line_says_what_the_stop_rule_waits_for():
    from smappy.gui.train_window import rule_line
    lines = [{"kind": "start", "epoch": 0, "epochs": 1000, "stop_rule": {"window": 100, "min_epochs": 150, "smooth": 11,
              "fall_from": 10, "loss_tol": 0.01, "jaccard_tol": 0.005, "eff_tol": 0.005, "rmse_tol": 0.01}}]
    for e in range(200):
        lines.append({"kind": "epoch", "epoch": e, "loss_test": 300.0 - 0.5 * e,
                      "metrics": {"jaccard": 0.3, "eff_lat": 0.2, "eff_ax": 0.2, "rmse_lat_nm": 45.0, "rmse_ax_nm": 70.0}})
    said = rule_line(lines)
    assert said.startswith("waiting for: loss (") and "of its fall" in said and "settled: Jaccard" in said
    assert rule_line([dict(lines[0], stop_rule=None)] + lines[1:]) == \
        "trains every epoch (stop when it no longer improves is off)"
    done = lines + [{"kind": "finished", "epoch": 199, "converged": True}]
    assert rule_line(done).startswith("converged at epoch 199")
