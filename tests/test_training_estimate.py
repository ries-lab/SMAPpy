"""How long a job will train, said before Start (`smappy.training.estimate`): torch-free."""
import json

import pytest
import yaml

from smappy.training import estimate, jobs, progress


def _decode_job(folder, samples=32768, size=40):
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(json.dumps({"method": "decode-v0.10.2"}))
    (folder / "method.yaml").write_text(yaml.safe_dump({"HyperParameter": {"pseudo_ds_size": samples},
                                                        "Simulation": {"img_size": [size, size]}}))
    return folder


def _netfit_job(folder, per_epoch=32768, max_samples=6e6, frame_px=40):
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(json.dumps({"method": "netfit-v1"}))
    (folder / "settings.json").write_text(json.dumps({
        "simulation": {"frame_px": frame_px, "window": 7},
        "parts": {"schedule": {"name": "adamw_step", "settings": {"samples_per_epoch": per_epoch,
                                                                    "max_samples": max_samples}}}}))
    return folder


def test_a_decode_job_on_the_cpu_is_warned_about_with_what_a_gpu_would_take(tmp_path):
    folder = _decode_job(tmp_path / "job")
    e = estimate.estimate(folder, device="cpu", epochs=1000, until_converged=True, entries=[])
    assert e.device == "cpu" and e.epochs == estimate.DECODE_CONVERGED_EPOCHS and not e.measured
    assert e.seconds == pytest.approx(estimate.RATES["decode"]["cpu"] * 400)
    assert "days" in e.text() and "CPU" in e.warning and "a GPU takes about" in e.warning
    fast = estimate.estimate(folder, device="cuda", epochs=1000, until_converged=True, entries=[])
    assert fast.warning == "" and fast.text().startswith("about 81 min for 400 epochs")


def test_without_the_early_stop_it_trains_every_epoch_and_a_long_run_on_a_gpu_is_warned_about(tmp_path):
    folder = _decode_job(tmp_path / "job", size=80)                   # four times the pixels of 40 x 40
    e = estimate.estimate(folder, device="cuda", epochs=1000, until_converged=False, entries=[])
    assert e.epochs == 1000
    assert e.seconds == pytest.approx(12.2 * 4 * 1000)                # 13.6 h
    assert "a long training" in e.warning


def test_an_optimised_model_trains_to_its_samples(tmp_path):
    folder = _netfit_job(tmp_path / "job", per_epoch=32768, max_samples=6e6)
    e = estimate.estimate(folder, device="cuda", epochs=1000, entries=[])
    assert e.epochs == 184 and e.warning == ""                        # ceil(6 M / 32 768)
    assert e.seconds == pytest.approx(7.4 * 184)


def test_earlier_jobs_epochs_on_that_host_replace_the_benchmark_rate(tmp_path):
    old = _decode_job(tmp_path / "old", samples=16384)                # half a reference epoch
    progress.append(old, "start", 0, device="cuda:0")
    for k, seconds in enumerate((20.0, 21.0, 22.0)):
        progress.append(old, "epoch", k + 1, seconds=seconds)
    listed = [jobs.Entry(str(old), host="box", remote="~/old")]
    new = _decode_job(tmp_path / "new")
    e = estimate.estimate(new, run_on="box", epochs=1000, entries=listed)
    assert e.measured and e.device == "cuda" and e.where == "box"
    assert e.seconds == pytest.approx(42.0 * 400)                    # 21 s a half epoch
    here = estimate.estimate(new, run_on="here", device="cuda", entries=listed)
    assert not here.measured                                          # another machine's jobs say nothing here


def test_a_host_whose_jobs_trained_on_the_cpu_is_taken_to_train_on_it_again(tmp_path):
    old = _decode_job(tmp_path / "old")
    progress.append(old, "start", 0, device="cpu")
    listed = [jobs.Entry(str(old), host="box", remote="~/old")]
    assert estimate.host_device("box", entries=listed) == "cpu"
    assert estimate.host_device("other", entries=listed) == "cuda"
    e = estimate.estimate(_decode_job(tmp_path / "new"), run_on="box", entries=listed)
    assert "Training hosts" in e.warning


def test_a_folder_that_is_no_job_has_no_estimate(tmp_path):
    assert estimate.estimate(tmp_path, entries=[]) is None
    assert estimate.duration(30) == "1 min" and estimate.duration(5 * 3600) == "5.0 h"
    assert estimate.duration(3 * 86400) == "3.0 days"


def test_the_window_asks_before_starting_on_the_cpu_and_starts_nothing_when_cancelled(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox
    QApplication.instance() or QApplication([])
    from smappy.gui import train_window
    folder = _decode_job(tmp_path / "job")
    jobs.add(folder)
    w = train_window.TrainWindow()
    try:
        from PySide6.QtCore import Qt
        rows = {str(w.job_list.item(i, 0).data(Qt.UserRole)): i for i in range(w.job_list.rowCount())}
        w.job_list.setCurrentCell(rows[str(folder.resolve())], 0)
        w.train_form.set_values({"device": "cpu"})
        w.show_estimate()
        assert "CPU" in w.estimate_note.text()
        monkeypatch.setattr(train_window, "torch_here", lambda: True)
        asked, started = [], []
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.Cancel)
        monkeypatch.setattr(train_window, "start_here", lambda *a, **k: started.append(a))
        w.start()
        assert asked and "Start anyway?" in asked[0] and not started
        assert w.train_note.text() == "not started"
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
        w.start()
        assert started
    finally:
        w.close()
        w.timer.stop()
        jobs.remove(folder)
