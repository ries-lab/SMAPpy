"""An optimised model's job (*DECODE, optimised*; docs/decode/gui-plan.md, step 2): written by the training
window from the prefit, trained by ``smappy-decode train`` as a DECODE job is -- a lock, a STOP file,
a resume -- and scored every epoch with the standard report's numbers.

The trainings are tiny (16 x 16 frames, 8 features, 64 samples an epoch, on the CPU): seconds.
"""
import json

import numpy as np
import pytest

from smappy.training import job as jobfile
from smappy.training import model_file, progress
from test_train_window import _prepare_settings, movie  # noqa: F401  (the fixture)

torch = pytest.importorskip("torch")


def _tiny(folder):
    """A job folder's settings made tiny, as a test trains them."""
    path = folder / "settings.json"
    s = json.loads(path.read_text())
    s["simulation"].update(frame_px=16, density_per_40px=15.0)
    s["parts"]["backbone"]["settings"]["features"] = 8
    s["parts"]["schedule"]["settings"].update(samples_per_epoch=64, batch=16, test_samples=16, epochs=3,
                                              precision="fp32", compile=False)
    path.write_text(json.dumps(s, indent=1))


def test_the_window_writes_an_optimised_job_from_the_prefits_ranges_with_its_paths_inside(movie):  # noqa: F811
    from smappy.gui.train_window import NETFIT, default_folder, run_prefit, write_job
    path, cal_path = movie
    prepared = run_prefit(_prepare_settings(path, cal_path, model="optimised"))
    est = prepared.estimates
    folder = write_job(prepared, est, default_folder(path, cal_path))
    job, params = jobfile.read(folder)
    assert job.method == NETFIT and not (folder / "method.yaml").exists()
    sim = params["simulation"]
    assert sim["photons_mean"] == pytest.approx(est.intensity_mu_sig[0])
    assert sim["background"] == pytest.approx(list(est.background)) and sim["z_nm"] == pytest.approx(list(est.z_nm))
    assert sim["density_per_40px"] == 15.0
    # step 4's time model (GMN7C): the ConvGRU over 7 frames, on-times drawn over 1 - 3, streamed
    assert params["parts"]["time_model"] == {"name": "convgru", "settings": {"frames": 7, "norm": True}}
    assert (sim["window"], sim["lifetime_frames"], sim["lifetime_max_frames"], sim["movie_windows"]) == (7, 1.0, 3.0, 8)
    # the winners' schedule and targets
    assert params["parts"]["schedule"]["settings"]["decay_samples"] == 327680 and sim["target_photons_min"] == 100.0
    assert params["camera"]["conversion"] == 0.5 and params["camera"]["em_gain"] == 0.0
    assert params["calibration"] == "calibration.h5" and params["parts"]["schedule"]["settings"]["snapshots"] == []
    assert jobfile.check(folder) == []


def test_an_optimised_job_trains_stops_at_a_stop_file_and_resumes_through_the_decode_command(movie, tmp_path):  # noqa: F811
    from decode_smlm.cli import main
    from smappy.gui.train_window import run_prefit, write_job
    path, cal_path = movie
    prepared = run_prefit(_prepare_settings(path, cal_path, model="optimised"))
    folder = write_job(prepared, prepared.estimates, tmp_path / "job")
    _tiny(folder)
    (folder / "STOP").touch()                  # asked before it starts: it stops after its first epoch
    assert main(["train", str(folder), "--device", "cpu", "--all-epochs"]) == 0
    lines = progress.read(folder)
    assert [line["kind"] for line in lines] == ["start", "epoch", "stopped"]
    assert not (folder / "STOP").exists() and not (folder / "lock.json").exists()
    assert main(["train", str(folder), "--device", "cpu", "--all-epochs"]) == 0
    kinds = [line["kind"] for line in progress.read(folder)]
    assert kinds[-1] == "finished" and kinds.count("epoch") == 3            # resumed: epochs 1 and 2 added
    meta = model_file.read_metadata(folder / "model.safetensors")
    assert meta["method"] == "netfit-v1" and meta["trained_on"]["background"] == pytest.approx(
        list(prepared.estimates.background))
    # the settings the job holds stay relative: the folder moves
    assert json.loads((folder / "settings.json").read_text())["calibration"] == "calibration.h5"
    # scored every epoch (an untrained network finds too little for more than the Jaccard)
    assert all("jaccard" in line["metrics"] for line in progress.read(folder) if line["kind"] == "epoch")


def test_a_folder_being_trained_is_refused(movie, tmp_path):  # noqa: F811
    import os
    import socket

    from smappy.gui.train_window import run_prefit, write_job
    from smappy.netfit.job import JobError, train_folder
    path, cal_path = movie
    prepared = run_prefit(_prepare_settings(path, cal_path, model="optimised"))
    folder = write_job(prepared, prepared.estimates, tmp_path / "job")
    (folder / "lock.json").write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))
    with pytest.raises(JobError, match="being trained already"):
        train_folder(folder, device="cpu")


def test_the_report_counts_outliers_beyond_five_bounds_and_the_rest_s_rmse_over_their_bound():
    from smappy.netfit.metrics import outliers_and_inliers
    rng = np.random.default_rng(0)
    n = 10000
    bound = np.c_[np.full(n, 10.0), np.full(n, 10.0), np.full(n, 30.0)]
    d = rng.normal(0, 1, (n, 3)) * bound * 1.2                         # 1.2 times the bound
    d[:500, 0] = 100.0                                                 # 5 % ten bounds off in x
    r = outliers_and_inliers(d[:, 0], d[:, 1], d[:, 2], bound)
    assert r["outliers"] == pytest.approx(0.05, abs=0.002)
    assert r["inlier_rmse_crlb_lat"] == pytest.approx(1.2, abs=0.03)
    assert r["inlier_rmse_crlb_ax"] == pytest.approx(1.2, abs=0.03)
