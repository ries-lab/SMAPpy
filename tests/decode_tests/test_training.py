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


def tiny_job(folder, epochs=3, seed=0, base_model=None, batch_size=4, **hyper):
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
    params["HyperParameter"] = {"epochs": epochs, "batch_size": batch_size, "batch_base": 4, "pseudo_ds_size": 32,
                                "max_number_targets": 16,
                                "arch_param": {"initial_features": 8, "inter_features": 8}, **hyper}
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


# ----------------------------------------------------------------------------- the batch size, (f2)
def test_the_batch_chosen_is_the_smallest_within_ten_percent_of_the_best_time_a_frame():
    from decode_smlm.batch import choose, scaled_lr
    h100 = {64: 253e-6, 128: 179e-6, 256: 167e-6, 512: 159e-6}       # gpuvm's H100, 2026-11-11
    assert choose(h100) == 256
    assert choose({64: 1.0, 128: 0.98}) == 64                        # nothing to gain: the base
    assert scaled_lr(2e-4, 256) == pytest.approx(4e-4)               # the square root of 4
    assert scaled_lr(2e-4, 256, rule="linear") == pytest.approx(8e-4)
    assert scaled_lr(2e-4, 256, rule="none") == 2e-4
    with pytest.raises(ValueError):
        choose({})


def test_auto_on_a_gpu_measures_and_scales_the_rate_and_a_number_or_the_cpu_keeps_its_own(monkeypatch):
    from decode_smlm import batch as batches
    from decode_smlm.utils.types import RecursiveNamespace

    def param(size, device):
        return RecursiveNamespace(HyperParameter={"batch_size": size, "batch_base": 64, "lr_scaling": "sqrt",
                                                  "channels_in": 3, "opt_param": {"lr": 2e-4}},
                                  Hardware={"device": device}, Simulation={"img_size": [40, 40]})
    asked = []
    monkeypatch.setattr(batches, "measure", lambda model, ch, frame, device, base: asked.append((ch, frame, base))
                        or {64: 253e-6, 128: 179e-6, 256: 167e-6, 512: 159e-6})
    p = param("auto", "cuda:0")
    done = batches.resolve(p, lambda: None)
    assert asked == [(3, (40, 40), 64)]
    assert p.HyperParameter.batch_size == 256 and p.HyperParameter.opt_param.lr == pytest.approx(4e-4)
    assert done["chosen"] == "measured" and done["us_per_frame"]["256"] == 167.0
    p = param(128, "cuda:0")
    assert batches.resolve(p, lambda: None)["chosen"] == "given" and p.HyperParameter.opt_param.lr == 2e-4
    p = param("auto", "cpu")
    assert batches.resolve(p, lambda: None)["chosen"] == "cpu" and p.HyperParameter.batch_size == 64
    assert len(asked) == 1


def test_the_measurement_touches_no_random_state_and_runs_on_the_cpu():
    import torch

    from decode_smlm.batch import measure
    net = torch.nn.Conv2d(3, 4, 3, padding=1)
    weights = net.weight.detach().clone()
    torch.manual_seed(5)
    before = torch.rand(1)
    torch.manual_seed(5)
    times = measure(net, 3, (16, 16), "cpu", base=4, doublings=2, steps=2, warmup=1)
    assert torch.rand(1) == before and torch.equal(net.weight, weights)
    assert sorted(times) == [4, 8, 16] and all(t > 0 for t in times.values())


def test_an_auto_batch_is_chosen_once_and_a_resume_keeps_it(tmp_path, monkeypatch):
    from decode_smlm import batch as batches
    folder = tiny_job(tmp_path / "job", batch_size="auto")

    def chosen(param, factory):              # as a GPU would have measured: 8 frames, the rate times sqrt 2
        param.HyperParameter.batch_size = 8
        param.HyperParameter.opt_param.lr *= 2 ** 0.5
        return {"batch_size": 8, "chosen": "measured", "lr": param.HyperParameter.opt_param.lr}
    monkeypatch.setattr(batches, "resolve", chosen)
    train(folder, epochs=2, quiet=True)
    lr = progress.read(folder)[0]["batch"]["lr"]

    def never(param, factory):
        raise AssertionError("measured again on resume")
    monkeypatch.setattr(batches, "resolve", never)
    train(folder, epochs=3, quiet=True)
    starts = [line for line in progress.read(folder) if line["kind"] == "start"]
    assert [s["batch"]["batch_size"] for s in starts] == [8, 8] and starts[1]["batch"]["lr"] == lr
    meta = model_file.read_metadata(folder / MODEL)
    assert meta["batch_size"] == 8 and meta["learning_rate"] == pytest.approx(lr)


def test_the_warm_up_ramps_the_rate_and_a_resume_inside_it_is_bitwise_the_run_through(tmp_path):
    hyper = {"lr_warmup_epochs": 2, "lr_warmup_from": 1e-5, "opt_param": {"lr": 4e-4}}
    through = tiny_job(tmp_path / "through", epochs=4, **hyper)
    train(through, quiet=True)
    lrs = [e["lr"] for _, e in sorted(_epochs(through).items())]
    assert lrs[1] == pytest.approx(1e-5) and lrs[2] == pytest.approx(2.05e-4) and lrs[3] == pytest.approx(4e-4)
    parts = tiny_job(tmp_path / "parts", epochs=4, **hyper)
    train(parts, epochs=2, quiet=True)                       # stopped inside the ramp
    train(parts, epochs=4, quiet=True)
    a, b = _epochs(through), _epochs(parts)
    assert [b[i]["loss_train"] for i in (2, 3)] == [a[i]["loss_train"] for i in (2, 3)]
    assert [b[i]["lr"] for i in (1, 2, 3)] == [a[i]["lr"] for i in (1, 2, 3)]


def test_a_job_with_edge_padding_trains_and_its_model_file_says_so(tmp_path):
    folder = tiny_job(tmp_path / "job", epochs=2, edge_pad={"p": 0.5, "max_px": 3, "value": -1.0})
    assert train(folder, quiet=True) == "finished"
    assert model_file.read_metadata(folder / MODEL)["edge_pad"] == {"p": 0.5, "max_px": 3, "value": -1.0}
    assert set(_epochs(folder)) == {0, 1}
