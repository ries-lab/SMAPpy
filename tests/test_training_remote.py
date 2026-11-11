"""A job trained on a host over ssh (`smappy.training.remote`), with ssh played by a shell here.

The runner stands in for ``subprocess.run``: it runs the command ``ssh`` would run there in ``bash``
here, with ``~`` a folder of the test's, so the tar stream, the detached start, the lock's process and
the copies back are all real.  The training is a stand-in script (no torch): it writes a lock, epochs
and a model as `decode_smlm.training` does, and stops at the STOP file.
"""
import json
import os
import subprocess
import sys
import time

import pytest

from smappy.training import jobs, progress, remote

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in host is a POSIX shell")

TRAINER = r'''
import json, os, socket, sys, time
from pathlib import Path
if "--help" in sys.argv:
    sys.exit(0)
folder = Path(sys.argv[2]); epochs = int(sys.argv[4])
(folder / "lock.json").write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))
def line(**kw):
    with open(folder / "progress.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")
line(kind="start", time="", epoch=0, epochs=epochs, resumed=False)
for e in range(epochs):
    time.sleep(0.1)
    line(kind="epoch", time="", epoch=e, loss_train=1.0, loss_test=1.0, seconds=0.1, metrics={})
    (folder / "model.safetensors").write_bytes(b"model %d" % e)
    if (folder / "STOP").exists():
        line(kind="stopped", time="", epoch=e); break
else:
    line(kind="finished", time="", epoch=epochs - 1)
(folder / "lock.json").unlink()
'''


@pytest.fixture
def there(tmp_path, monkeypatch):
    """A host whose ``~`` is ``tmp_path/home``, and the runner that plays ssh to it."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "trainer.py").write_text(TRAINER)
    calls = []
    real = subprocess.run                  # the tests below put `runner` in its place

    def runner(argv, input=None, capture_output=True, timeout=None):
        assert argv[0] == "ssh" and "BatchMode=yes" in argv
        calls.append(argv[-1])
        return real(["bash", "-c", argv[-1]], input=input, capture_output=True, timeout=timeout,
                              env={**os.environ, "HOME": str(home)})
    jobs.set_hosts([jobs.Host("box", "box.example", folder="~/smappy_training",
                              command=f"{sys.executable} ~/trainer.py")])
    return home, runner, calls


def _job(tmp_path):
    folder = tmp_path / "data" / "cells_decode" / "beads_2026-11-12"
    folder.mkdir(parents=True)
    (folder / "job.json").write_text("{}")
    (folder / "method.yaml").write_text("x: 1\n")
    (folder / "progress.jsonl").write_text("")            # a training's own file: not sent
    return folder


def _wait(entry, runner, until, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        remote.follow(entry, runner=runner)
        if until(jobs.status(entry)):
            return jobs.status(entry)
        time.sleep(0.2)
    raise AssertionError(jobs.status(entry))


def test_a_job_is_sent_trained_followed_stopped_and_its_model_fetched(tmp_path, there):
    home, runner, _ = there
    folder = _job(tmp_path)
    entry = remote.send(folder, jobs.host("box"), runner=runner)
    assert entry.remote == "~/smappy_training/beads_2026-11-12" and jobs.entries()[-1].host == "box"
    sent = home / "smappy_training" / "beads_2026-11-12"
    assert sorted(p.name for p in sent.iterdir()) == ["job.json", "method.yaml"]
    remote.start(entry, epochs=200, runner=runner)
    s = _wait(entry, runner, lambda s: s.state == "running" and (s.epoch or 0) >= 2)
    assert s.epochs == 200 and (folder / jobs.LOCK).exists()
    remote.stop(entry, runner=runner)
    s = _wait(entry, runner, lambda s: s.state == "stopped")
    assert not (folder / jobs.LOCK).exists()
    path = remote.fetch_model(entry, runner=runner)
    assert path == folder / "model.safetensors" and path.read_bytes() == f"model {s.epoch}".encode()


def test_a_training_that_died_there_is_interrupted_here(tmp_path, there):
    home, runner, _ = there
    folder = _job(tmp_path)
    entry = remote.send(folder, jobs.host("box"), runner=runner)
    sent = home / "smappy_training" / "beads_2026-11-12"
    progress.append(sent, "start", 0, epochs=10)
    (sent / jobs.LOCK).write_text(json.dumps({"host": "box", "pid": 2 ** 22 + 12345}))
    assert remote.follow(entry, runner=runner) is False
    assert jobs.status(entry).state == "interrupted"


def test_the_start_line_detaches_the_training_and_quotes_the_folder():
    h = jobs.Host("box", "box", command="smappy-decode")
    line = remote.train_command(h, "~/smappy_training/a job", 50, "cuda:1")
    assert line.startswith("cd ~/'a job'") is False and "~/'smappy_training/a job'" in line
    assert "nohup smappy-decode train ~/'smappy_training/a job' --epochs 50 --device cuda:1" in line
    assert "< /dev/null &" in line and "rm -f STOP" in line


def test_what_ssh_says_when_it_fails_reaches_the_window():
    def refuse(argv, **kw):
        return subprocess.CompletedProcess(argv, 255, b"", b"warning\nPermission denied (publickey).\n")
    with pytest.raises(remote.RemoteError, match="box: Permission denied"):
        remote.ssh(jobs.Host("box", "box"), "true", runner=refuse)


def test_the_window_starts_a_job_on_a_host_and_its_follower_brings_back_the_progress_and_model(tmp_path, there,
                                                                                               monkeypatch):
    from smappy.gui import train_window
    _, runner, _ = there
    monkeypatch.setattr(remote.subprocess, "run", runner)
    folder = _job(tmp_path)
    jobs.add(folder)                                       # prepared here, as the window lists it
    said = train_window.start_on_host(folder, train_window.TrainSettings(where="box", epochs=5))
    assert said.startswith("started on box")
    entry = next(e for e in jobs.entries() if e.folder == str(folder.resolve()))
    assert entry.host == "box"
    follower = train_window.HostFollower()
    deadline = time.monotonic() + 30
    while jobs.status(entry).state != "finished" or not (folder / remote.MODEL).exists():
        assert time.monotonic() < deadline, (jobs.status(entry), follower.problems)
        follower.follow_all()
        time.sleep(0.2)
    assert (folder / remote.MODEL).read_bytes() == b"model 4" and not follower.problems


def test_the_preferences_page_keeps_a_training_host_and_tests_it(there, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.preferences import HostsPage
    _, runner, _ = there
    monkeypatch.setattr(remote.subprocess, "run", runner)
    page = HostsPage()
    page.add()
    page.edits["name"].setText("H100")
    page.edits["ssh"].setText("gpuvm")
    page.edits["command"].setText(f"{sys.executable} ~/trainer.py")
    page.apply()
    assert jobs.host("H100").ssh == "gpuvm" and jobs.host("H100").folder == "~/smappy_training"
    page.test()
    assert page.status.text().startswith("H100 answers")
    page.remove()
    assert jobs.host("H100") is None and jobs.host("box") is not None
