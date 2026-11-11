"""The Train window's list of jobs and its training hosts (`smappy.training.jobs`): torch-free."""
import json
import os
import socket

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
