"""Training from a job folder (``decode_smlm.training``, ``smappy-decode``): milestone (f).

A tiny job -- 16 x 16 frames, 8 batches of 4, a narrow network, on the CPU without data-loader
workers -- trains in about a second, so stop, resume, a warm start and the lock are tested on real
trainings.
"""
import json
import os
import socket

import numpy as np
import pytest

from decode_smlm import prefit
from decode_smlm.training import CKPT, LOCK, MODEL, STOP, TEST_SET, JobError, train
from smappy.io.calibration import SplineCalibration
from smappy.training import job as jobs
from smappy.training import model_file, progress


def tiny_job(folder, epochs=3, seed=0, base_model=None):
    g = np.exp(-0.5 * ((np.arange(13) - 6.0) / 1.5) ** 2)
    coeff = np.zeros((64, 11, 13, 13), np.float32)
    coeff[0] = (np.outer(g, g) / np.outer(g, g).sum())[None]
    cal = SplineCalibration(coeff=coeff, dz=100.0, z0=5.0, x0=7.0, em_on=False)
    est = prefit.Estimates(on_time_frames=2.0, lifetime_frames=1.0, intensity_mu_sig=(3000.0, 600.0),
                           intensity_from="fully-on frames", n_fully_on=500, background=(20.0, 60.0),
                           z_nm=(-400.0, 400.0), emitters_per_frame=2.0, n_blinks=100, n_localizations=300)
    params = prefit.method_params(est, prefit.camera(0.5, 100.0, 100.0), "LD", frame_px=16)
    ext = [[-0.5, 15.5], [-0.5, 15.5], None]
    params["Simulation"].update(img_size=[16, 16], psf_extent=ext, emitter_av=3)
    params["TestSet"] = {"test_size": 16, "img_size": [16, 16], "frame_extent": ext}
    params["HyperParameter"] = {"epochs": epochs, "batch_size": 4, "pseudo_ds_size": 32, "max_number_targets": 16,
                                "arch_param": {"initial_features": 8, "inter_features": 8}}
    params["Hardware"] = {"num_worker_train": 0, "torch_threads": 1}
    job = jobs.Job.new(method="decode-v0.10.2", ranges={"density": "LD"}, camera={}, calibration={}, seed=seed,
                       device="cpu", base_model=base_model)
    return jobs.write(folder, job, params, cal)


def _epochs(folder):
    return {line["epoch"]: line for line in progress.read(folder) if line["kind"] == "epoch"}


def test_a_tiny_job_trains_and_every_progress_line_is_valid(tmp_path):
    import time
    folder = tiny_job(tmp_path / "job")
    t = time.perf_counter()
    assert train(folder, quiet=True) == "finished"
    assert time.perf_counter() - t < 20
    lines = progress.read(folder)
    assert [line["kind"] for line in lines] == ["start", "epoch", "epoch", "epoch", "finished"]
    for line in (folder / progress.NAME).read_text().splitlines():
        json.loads(line)                                          # every line valid JSON, no NaN
    e = _epochs(folder)
    assert e[0]["loss_train"] is None and e[2]["loss_train"] > 0 and e[2]["loss_test"] > 0
    assert {"jaccard", "rmse_lat_nm", "rmse_ax_nm", "eff_lat", "eff_ax"} <= set(e[2]["metrics"])
    for name in (CKPT, MODEL, TEST_SET):
        assert (folder / name).exists()
    assert not (folder / LOCK).exists()
    meta = model_file.read_metadata(folder / MODEL)
    assert meta["method"] == "decode-v0.10.2" and meta["epoch"] == 2 and meta["architecture"] == "SigmaMUNet"


def test_stopped_and_resumed_the_next_epoch_is_bitwise_the_uninterrupted_runs(tmp_path):
    through = tiny_job(tmp_path / "through")
    train(through, quiet=True)
    parts = tiny_job(tmp_path / "parts")
    train(parts, epochs=2, quiet=True)                         # stops after epoch 1, its checkpoint written
    assert progress.state(progress.read(parts)) == "finished"
    train(parts, epochs=3, quiet=True)                         # resumes at epoch 2
    a, b = _epochs(through), _epochs(parts)
    assert b[2]["loss_train"] == a[2]["loss_train"] and b[2]["loss_test"] == a[2]["loss_test"]
    assert b[1]["loss_test"] == a[1]["loss_test"]
    assert [line["resumed"] for line in progress.read(parts) if line["kind"] == "start"] == [False, True]


def test_a_stop_file_stops_at_the_checkpoint_and_is_removed(tmp_path):
    folder = tiny_job(tmp_path / "job")
    (folder / STOP).touch()
    assert train(folder, quiet=True) == "stopped"
    lines = progress.read(folder)
    assert lines[-1]["kind"] == "stopped" and lines[-1]["epoch"] == 0 and not (folder / STOP).exists()
    assert train(folder, quiet=True) == "finished"
    assert sorted(_epochs(folder)) == [0, 1, 2]


def test_a_warm_start_starts_from_the_model_files_weights(tmp_path):
    first = tiny_job(tmp_path / "first")
    train(first, quiet=True)
    second = tmp_path / "second"
    second.mkdir()
    (second / "base.safetensors").write_bytes((first / MODEL).read_bytes())
    tiny_job(second, epochs=1, seed=5, base_model="base.safetensors")
    train(second, quiet=True)                                  # epoch 0 is tested, not trained
    base, _ = model_file.read(second / "base.safetensors")
    after, _ = model_file.read(second / MODEL)
    assert all(np.array_equal(base[k], after[k]) for k in base)
    assert _epochs(second)[0]["loss_test"] < _epochs(first)[0]["loss_test"]   # trained weights test better


def test_a_folder_being_trained_is_refused(tmp_path):
    folder = tiny_job(tmp_path / "job")
    (folder / LOCK).write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))
    with pytest.raises(JobError, match="being trained already"):
        train(folder, quiet=True)
    (folder / LOCK).write_text(json.dumps({"host": "elsewhere", "pid": 1}))
    with pytest.raises(JobError, match="locked by a training on elsewhere"):
        train(folder, quiet=True)
    (folder / LOCK).write_text(json.dumps({"host": socket.gethostname(), "pid": 2 ** 22 + 17}))   # gone
    assert train(folder, quiet=True) == "finished"


def test_a_job_of_another_smappy_version_is_refused_unless_allowed(tmp_path):
    folder = tiny_job(tmp_path / "job", epochs=1)
    raw = json.loads((folder / jobs.JOB).read_text())
    raw["smappy_version"] = "0.0.1"
    (folder / jobs.JOB).write_text(json.dumps(raw))
    with pytest.raises(JobError, match="prepared by SMAPpy 0.0.1"):
        train(folder, quiet=True)
    assert progress.read(folder) == []           # refused before it touched the folder: no line written
    assert train(folder, quiet=True, allow_other_version=True) == "finished"


def test_status_reads_the_folder_alone(tmp_path, capsys):
    from decode_smlm.cli import main
    folder = tiny_job(tmp_path / "job", epochs=2)
    train(folder, quiet=True)
    assert main(["status", str(folder)]) == 0
    out = capsys.readouterr().out
    assert "finished" in out and "epoch 1:" in out and "jaccard" in out
