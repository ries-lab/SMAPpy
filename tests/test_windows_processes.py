"""What SMAPpy starts and pipes, as Windows needs it.

Found when the suite first ran on Windows: a Stop that stopped nothing, a
path with an ``ü`` lost between the batch runner and its window, a start
script cmd.exe could not read, an install command cut at ``Program Files``.
Each is checked here as the text, the environment or the flags SMAPpy
hands over, so that they hold on any platform; what Windows itself does
with them is checked where the test runs there.
"""
import io
import os
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")


class _Process:
    def __init__(self):
        self.called = []

    def terminate(self):
        self.called.append("terminate")

    def kill(self):
        self.called.append("kill")


def test_stop_kills_the_runner_on_windows_where_terminate_only_asks_a_window_to_close():
    from smappy.gui.batch_window import stop_process
    windows, posix = _Process(), _Process()
    stop_process(windows, windows=True)
    stop_process(posix, windows=False)
    assert windows.called == ["kill"]
    assert posix.called == ["terminate"]


def test_the_runner_started_as_the_batch_window_starts_it_prints_its_paths_in_utf8():
    from smappy.gui.batch_window import UTF8_ENVIRONMENT
    path = "C:/Daten/Jürgen/Δ 5 µm.tif"
    code = ("from smappy import batch; "
            f"batch.emitter()('START', '1/1', {path!r})")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          env={**os.environ, **UTF8_ENVIRONMENT}, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.decode("utf-8").rstrip("\r\n").split("\t") == \
        ["SMAPPY_START", "1/1", path]


def test_the_batch_runner_writes_utf8_even_on_a_stream_opened_in_the_ansi_code_page():
    from smappy.cli.batch import utf8_streams
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    utf8_streams(stream)
    print("SMAPPY_START\t1/1\tC:/Δ/ü.tif", file=stream, flush=True)
    assert raw.getvalue().decode("utf-8").startswith("SMAPPY_START\t1/1\tC:/Δ/ü.tif")


def test_the_headless_server_reads_and_answers_in_utf8_with_one_newline_a_line():
    from smappy.assistant.server import utf8_protocol
    asked = '{"path": "C:/Jürgen/Δ.tif"}\n'.encode("utf-8")
    reader = io.TextIOWrapper(io.BytesIO(asked), encoding="cp1252")
    out, log = io.BytesIO(), io.BytesIO()
    writer = io.TextIOWrapper(out, encoding="cp1252", newline="\r\n")
    logger = io.TextIOWrapper(log, encoding="cp1252")
    utf8_protocol(reader, writer, logger)
    assert reader.readline() == asked.decode("utf-8")
    writer.write('{"text": "≥ 5 µm"}\n')
    writer.flush()
    assert out.getvalue() == '{"text": "≥ 5 µm"}\n'.encode("utf-8")
    logger.write("Δ")
    logger.flush()
    assert log.getvalue() == "Δ".encode("utf-8")


def test_the_install_command_shown_for_pytorch_quotes_a_python_under_program_files():
    from smappy.calibrate.psf_window import shell_line
    parts = [r"C:\Program Files\Python313\python.exe", "-m", "pip", "install", "torch>=2.4"]
    assert shell_line(parts, windows=True) == \
        r'"C:\Program Files\Python313\python.exe" -m pip install "torch>=2.4"'
    assert shell_line(["/opt/my python/bin/python", "-m", "pip", "install", "torch>=2.4"],
                      windows=False) == "'/opt/my python/bin/python' -m pip install 'torch>=2.4'"
