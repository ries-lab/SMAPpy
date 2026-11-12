"""The Train window's list of jobs and its training hosts (`smappy.training.jobs`): torch-free."""
import json
import os
import socket

import numpy as np

from smappy.training import jobs, progress


def test_a_job_listed_twice_is_listed_once_with_where_it_runs_now(tmp_path):
    folder = tmp_path / "movie_decode" / "beads_2026-11-11"
    folder.mkdir(parents=True)
    jobs.add(folder)
    jobs.add(folder, host="gpuvm", remote="~/smappy_training/beads_2026-11-11")
    listed = [e for e in jobs.entries() if e.folder == str(folder.resolve())]
    assert len(listed) == 1 and listed[0].host == "gpuvm" and listed[0].where == "gpuvm"
    jobs.remove(folder)
    assert all(e.folder != str(folder.resolve()) for e in jobs.entries())
    assert folder.is_dir()                                  # taken off the list, never off the disk


def test_a_jobs_state_tells_a_dead_training_from_a_running_one(tmp_path):
    folder = tmp_path / "job"
    folder.mkdir()
    entry = jobs.Entry(str(folder))
    assert jobs.status(entry).state == "new"
    progress.append(folder, "start", 0, epochs=1000)
    progress.append(folder, "epoch", 4, loss_test=500.0)
    assert jobs.status(entry).state == "interrupted"        # running, and no lock: it died
    (folder / jobs.LOCK).write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))
    s = jobs.status(entry)
    assert s.state == "running" and s.epoch == 4 and s.epochs == 1000 and s.last["loss_test"] == 500.0
    (folder / jobs.LOCK).write_text(json.dumps({"host": socket.gethostname(), "pid": 2 ** 22 + 12345}))
    assert jobs.status(entry).state == "interrupted"        # the lock of a process that is gone
    progress.append(folder, "finished", 4)
    assert jobs.status(entry).state == "finished"
    assert jobs.status(jobs.Entry(str(tmp_path / "gone"))).state == "missing"


def test_a_job_on_a_host_is_running_while_the_lock_copied_back_says_so(tmp_path):
    folder = tmp_path / "job"
    folder.mkdir()
    progress.append(folder, "start", 0)
    entry = jobs.Entry(str(folder), host="gpuvm", remote="~/jobs/job")
    assert jobs.status(entry).state == "interrupted"        # `remote.follow` found its process gone
    (folder / jobs.LOCK).write_text(json.dumps({"host": "elsewhere", "pid": 2 ** 22 + 12345}))
    assert jobs.status(entry).state == "running"            # a process there, not here: not asked about here


def test_training_hosts_are_kept_in_the_configuration():
    jobs.set_hosts([jobs.Host("H100", "gpuvm", folder="~/jobs"), jobs.Host("4090", "decodebox")])
    assert [h.name for h in jobs.hosts()] == ["H100", "4090"]
    assert jobs.host("H100").ssh == "gpuvm" and jobs.host("H100").command == "smappy-decode"
    assert jobs.host("nowhere") is None


def test_a_training_that_refused_to_start_is_an_error_with_its_reason(tmp_path):
    folder = tmp_path / "job"
    folder.mkdir()
    (folder / jobs.LOG).write_text("some warning\nsmappy-decode: job was prepared by SMAPpy 0.6.1, this is 0.6.5\n")
    s = jobs.status(jobs.Entry(str(folder)))
    assert s.state == "error" and s.message.startswith("smappy-decode: job was prepared")
    (folder / jobs.LOG).write_text("UserWarning: something harmless\n")
    assert jobs.status(jobs.Entry(str(folder))).state == "new"


def _curve(n, plateau_from, seed=0, tau=30.0):
    rng = np.random.default_rng(seed)
    lines = []
    for e in range(n):
        x = min(e, plateau_from)
        lines.append({"kind": "epoch", "epoch": e, "loss_test": 300.0 + 100.0 * np.exp(-x / tau) + rng.normal(0, 0.2),
                      "metrics": {"jaccard": 0.4 - 0.3 * np.exp(-x / tau)}})
    return lines


def test_a_training_converges_once_neither_loss_nor_jaccard_improved_over_the_window():
    from smappy.training.progress import converged
    still = _curve(260, plateau_from=400, tau=150.0)   # still improving at 260
    assert converged(still) is None
    flat = _curve(260, plateau_from=120)               # flat from 120: 140 epochs without gain
    why = converged(flat)
    assert why is not None and "last 100 epochs" in why
    assert converged(flat[:140]) is None               # never before 150 epochs
    assert converged(flat, {"jaccard_tol": -1.0}) is None    # a rule the job changes
    restarted = flat + [{"kind": "restart", "epoch": 259}] + _curve(50, plateau_from=10)
    assert converged(restarted) is None                # a restart's run begins again


def test_a_training_whose_rmse_still_falls_goes_on_and_one_lucky_epoch_sets_no_bar():
    from smappy.training.progress import converged
    flat = _curve(260, plateau_from=120)
    assert converged(flat) is not None
    falling = [dict(e, metrics={**e["metrics"], "rmse_lat_nm": 60.0 - 0.05 * e["epoch"]}) for e in flat]
    assert converged(falling) is None                      # 5 nm better over the window: not done
    lucky = [dict(e) for e in flat]
    lucky[100] = dict(lucky[100], loss_test=lucky[100]["loss_test"] - 50.0)   # one epoch far below the rest
    assert converged(lucky) is not None                    # the running median does not take it as the bar
    raw = converged(lucky, {"smooth": 1})
    assert raw is not None and "loss" in raw               # without it the bar is that one epoch: easier still


def test_the_loss_is_judged_against_how_far_it_has_fallen_whatever_its_size():
    from smappy.training.progress import converged, stop_status, status_text
    # a sparse training's loss: 24 at the end, still falling by 0.5 a hundred epochs -- 2 % of its
    # value, but under 1 % of the 60 it has fallen since epoch 10
    lines = []
    for e in range(400):
        lines.append({"kind": "epoch", "epoch": e, "loss_test": 24.0 + 60.0 * np.exp(-e / 40.0) + 0.5 * (1 - e / 400),
                      "metrics": {"jaccard": 0.56, "eff_lat": 0.4, "eff_ax": 0.5, "rmse_lat_nm": 42.0, "rmse_ax_nm": 46.0}})
    assert converged(lines) is not None
    loss = stop_status(lines)[0]
    assert loss["kind"] == "fall" and 0 < loss["gain"] < 0.01 and loss["settled"]
    still = [dict(e, loss_test=e["loss_test"] - 0.02 * e["epoch"]) for e in lines]     # 8 more in the window
    assert converged(still) is None and status_text(stop_status(still)).startswith("waiting for: loss")
