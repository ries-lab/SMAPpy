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
    assert text.startswith("epoch 4 of 100") and "12.0 s an epoch" in text and "0 h 19 min left" in text
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
        texts = [w.job_list.item(i).text() for i in range(w.job_list.count())]
        assert any(t.startswith("job: interrupted") for t in texts)        # running, and no lock
        w.job_list.setCurrentRow(texts.index(next(t for t in texts if t.startswith("job:"))))
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
    assert seen["command"][-4:] == ["--epochs", "7", "--device", "mps"] and seen["cwd"] == str(tmp_path)
    assert (tmp_path / "train.log").exists()


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
