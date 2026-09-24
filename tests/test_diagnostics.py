"""The log a user sends when something crashed or came out wrong."""
import io
import logging
import threading
import zipfile
from dataclasses import dataclass

import numpy as np
import pytest

from smappy import diagnostics
from smappy.locs import Localizations
from smappy.session import Session


@pytest.fixture
def log(tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    path = diagnostics.start("test")
    yield path
    diagnostics.stop()


def text(path) -> str:
    for handler in diagnostics.logger.handlers:
        handler.flush()
    return path.read_text()


def test_the_log_opens_with_the_environment_a_bug_report_asks_for(log):
    content = text(log)
    assert "test started" in content
    assert "python:" in content and "numpy" in content and "platform:" in content


def test_a_plugin_run_is_logged_with_its_settings_and_a_failure_with_its_traceback(log):
    @dataclass
    class Settings:
        radius_nm: float = 50.0

    class Thing:
        path = "Analysis/Measure/Thing"
        version = "3"

    with diagnostics.running(Thing(), "run", Settings(), n=1234):
        pass
    with pytest.raises(ZeroDivisionError):
        with diagnostics.running(Thing(), "preview", Settings(radius_nm=7.0)):
            1 / 0
    content = text(log)
    assert "run Analysis/Measure/Thing (v3) on 1234 localizations: radius_nm=50.0" in content
    assert "run Analysis/Measure/Thing done in" in content
    assert "preview Analysis/Measure/Thing failed" in content
    assert "ZeroDivisionError" in content


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_prints_and_uncaught_exceptions_on_any_thread_reach_the_log(log):
    # the stream `start` puts in place of stdout; built here because pytest
    # swaps sys.stdout for its own capture around every test
    console = io.StringIO()
    tee = diagnostics._Tee(console, logging.INFO, "out")
    print("3D render failed: the table changed", file=tee)
    print("no newline yet", file=tee, end="")
    tee.flush()
    assert console.getvalue() == ("3D render failed: the table changed\n"
                                  "no newline yet")     # still printed as before

    def boom():
        raise RuntimeError("worker died")

    worker = threading.Thread(target=boom, name="worker")
    worker.start()
    worker.join()
    content = text(log)
    assert "(out) 3D render failed: the table changed" in content
    assert "(out) no newline yet" in content
    assert "uncaught exception in thread worker" in content
    assert "RuntimeError: worker died" in content


def test_nothing_is_written_and_nothing_printed_before_the_log_is_started(tmp_path, monkeypatch,
                                                                         capsys):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        with diagnostics.running(object(), "run"):
            raise ValueError("a script's own failure")
    assert not (tmp_path / "logs").exists()
    assert capsys.readouterr().err == ""          # no "last resort" traceback


def test_the_bug_report_has_the_logs_and_the_session_but_not_the_localizations(log, tmp_path):
    rng = np.random.default_rng(0)
    session = Session(Localizations({"x_nm": rng.uniform(0, 1e4, 500),
                                     "y_nm": rng.uniform(0, 1e4, 500),
                                     "photons": rng.gamma(3, 900.0, 500)}, {}))
    session.history.append({"plugin": "Analysis/Drift/RCC", "text": "drift corrected"})
    written = diagnostics.bug_report(tmp_path / "report", session, note="the drift looks wrong")
    assert written.suffix == ".zip"
    with zipfile.ZipFile(written) as z:
        names = set(z.namelist())
        report = z.read("report.txt").decode()
    assert {"report.txt", "smappy.log", "crash.log"} <= names
    assert "the drift looks wrong" in report
    assert "localizations: 500" in report and "photons (float64)" in report
    assert "Analysis/Drift/RCC" in report
    assert len(report) < 5000                      # a description, not the table
