"""The Train window's training here: a tiny job started as the window starts it, followed, stopped, resumed.

The window (`smappy.gui.train_window`) runs ``python -m decode_smlm.cli train`` detached and reads its
files; this is that, with the training tests' tiny job (16 x 16 frames, on the CPU), on a real process.
"""
import time

import pytest

from smappy.gui import train_window
from smappy.training import jobs, progress

from .test_training import tiny_job

pytestmark = pytest.mark.slow


def _wait(folder, until, seconds=120):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        lines = progress.read(folder)
        if until(lines):
            return lines
        time.sleep(0.2)
    raise AssertionError(f"waited {seconds} s: {[line['kind'] for line in progress.read(folder)]}"
                         f"\n{(folder / 'train.log').read_text()[-2000:]}")


def test_a_job_started_here_is_followed_stopped_and_resumed(tmp_path):
    folder = tiny_job(tmp_path / "job")
    jobs.add(folder)
    run = train_window.TrainSettings(epochs=400, device="cpu")
    process = train_window.start_here(folder, run)
    _wait(folder, lambda lines: sum(line["kind"] == "epoch" for line in lines) >= 2)
    assert jobs.status(jobs.Entry(str(folder))).state == "running"
    train_window.stop(folder)
    lines = _wait(folder, lambda lines: progress.state(lines) == "stopped")
    assert process.wait(60) == 0
    stopped_at = max(line["epoch"] for line in lines if line["kind"] == "epoch")
    assert jobs.status(jobs.Entry(str(folder))).state == "stopped"
    train_window.start_here(folder, train_window.TrainSettings(epochs=stopped_at + 3, device="cpu")).wait(120)
    lines = progress.read(folder)
    starts = [line for line in lines if line["kind"] == "start"]
    assert [s["resumed"] for s in starts] == [False, True] and progress.state(lines) == "finished"
    assert max(line["epoch"] for line in lines if line["kind"] == "epoch") == stopped_at + 2
