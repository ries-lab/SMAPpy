"""A training's job folder (`smappy.training.job`): written, read, moved, checked."""
import json
import shutil

import numpy as np
import pytest

from smappy.io.calibration import SplineCalibration
from smappy.training import job as jobs


def _calibration(seed=0):
    rng = np.random.default_rng(seed)
    coeff = rng.normal(0, 1, (64, 11, 9, 9)).astype(np.float32)
    return SplineCalibration(coeff=coeff, dz=50.0, z0=5.0, x0=5.0, em_on=False,
                             parameters={"fminmax": [1, 11], "dz": 50.0})


def _params():
    return {"Camera": {"baseline": 100.0, "e_per_adu": 0.5, "em_gain": None, "px_size": [108.0, 108.0]},
            "InOut": {"calibration_file": "calibration.h5", "experiment_out": "."},
            "Simulation": {"bg_uniform": [10.0, 60.0], "emitter_av": 25, "lifetime_avg": 1.4,
                           "emitter_extent": [[-0.5, 39.5], [-0.5, 39.5], [-500, 500]],
                           "intensity_mu_sig": [2500.0, 500.0]}}


def _job():
    return jobs.Job.new(method="decode-v0.10.2", camera={"conversion": 0.5, "offset": 100.0},
                        ranges={"photons": [2500.0, 500.0], "background": [10.0, 60.0], "z_nm": [-500, 500],
                                "density": "HD", "emitters_per_frame_measured": 7.25},
                        calibration={"source": "/data/beads_3dcal.mat"},
                        data={"path": "/data/cells.tif", "frames": 500, "roi": [0, 0, 128, 128]}, seed=3)


def _files(folder):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def test_a_folder_written_read_and_written_again_is_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    jobs.write(a, _job(), _params(), _calibration())
    job, params = jobs.read(a)
    jobs.write(b, job, params, a / jobs.CALIBRATION)
    assert _files(a) == _files(b)
    jobs.write(a, *jobs.read(a), a / jobs.CALIBRATION)       # over itself
    assert _files(a) == _files(b)


def test_the_job_records_the_calibrations_coefficients(tmp_path):
    cal = _calibration()
    jobs.write(tmp_path, _job(), _params(), cal)
    job, _ = jobs.read(tmp_path)
    assert job.calibration["file"] == "calibration.h5" and job.calibration["source"] == "/data/beads_3dcal.mat"
    assert job.calibration["sha256"] == jobs.coefficients_sha256(cal)
    assert job.schema == 1 and job.smappy_version and job.created


def test_a_moved_folder_is_still_complete(tmp_path):
    jobs.write(tmp_path / "here", _job(), _params(), _calibration())
    moved = shutil.move(str(tmp_path / "here"), str(tmp_path / "elsewhere" / "job"))
    assert jobs.check(moved) == []
    _, params = jobs.read(moved)
    assert (tmp_path / "elsewhere" / "job" / params["InOut"]["calibration_file"]).exists()


def test_check_says_what_is_wrong(tmp_path):
    jobs.write(tmp_path, _job(), _params(), _calibration())
    from smappy.io.calibration import save_spline_calibration
    save_spline_calibration(tmp_path / jobs.CALIBRATION, _calibration(seed=1), overwrite=True)
    assert jobs.check(tmp_path) == ["calibration.h5 is not the calibration the job was prepared with"]
    (tmp_path / jobs.METHOD).unlink()
    assert jobs.check(tmp_path) == ["method.yaml is missing"]


def test_an_absolute_path_in_the_methods_settings_is_refused(tmp_path):
    params = _params()
    params["InOut"]["calibration_file"] = str(tmp_path / "x.h5")
    with pytest.raises(ValueError, match="relative"):
        jobs.write(tmp_path, _job(), params, _calibration())


def test_a_job_of_another_schema_is_not_read(tmp_path):
    jobs.write(tmp_path, _job(), _params(), _calibration())
    raw = json.loads((tmp_path / jobs.JOB).read_text())
    raw["schema"] = 2
    (tmp_path / jobs.JOB).write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="schema 2"):
        jobs.read(tmp_path)


def test_the_job_folder_loads_neither_torch_nor_a_method():
    import subprocess
    import sys
    code = "import sys, smappy.training.job, smappy.training.prefit; print('torch' in sys.modules, 'decode_smlm' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]
