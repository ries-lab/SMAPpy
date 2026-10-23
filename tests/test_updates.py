"""Checking for a newer SMAPpy: the comparison, the command for each kind of
installation, the once-a-day schedule, and the GUI's request and button."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from smappy import config, updates


@pytest.fixture
def own_config(tmp_path, monkeypatch):
    """A config directory of this test's own: the session's is shared."""
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    config.load(reload=True)
    yield tmp_path
    monkeypatch.undo()
    config.load(reload=True)


class Dist:
    """What `how_installed` reads of a distribution: its metadata files."""

    def __init__(self, **files):
        self.files = files

    def read_text(self, name):
        return self.files.get(name)


def test_versions_compare_as_releases_not_as_text():
    assert updates.is_newer("0.10.0", "0.9.3")
    assert updates.is_newer("0.5.0", "0.5.0rc1")
    assert updates.is_newer("0.5.1", "0.5.0")
    assert not updates.is_newer("0.5", "0.5.0")
    assert not updates.is_newer("0.4.1", "0.4.1+g1234")
    assert not updates.is_newer("0.4.0", "0.4.1")


def test_the_newest_release_is_read_from_pypis_answer_and_nonsense_is_none():
    assert updates.newest_from(json.dumps({"info": {"version": "0.5.0"}})) == "0.5.0"
    assert updates.newest_from(b"<html>proxy login</html>") is None
    assert updates.newest_from(json.dumps({"info": {}})) is None


def test_how_it_was_installed_is_read_from_the_package_metadata(tmp_path):
    editable = Dist(**{
        "direct_url.json": json.dumps({"url": "file:///src/SMAPpy", "dir_info": {"editable": True}})})
    assert updates.how_installed(tmp_path, editable) == "checkout"
    (tmp_path / "uv-receipt.toml").write_text("")
    assert updates.how_installed(tmp_path, Dist(INSTALLER="uv\n")) == "uv tool"
    uvx = tmp_path / "cache" / "archive-v0" / "abc"
    assert updates.how_installed(uvx, Dist(INSTALLER="uv\n")) == "uvx"
    assert updates.how_installed(tmp_path / "venv", Dist(INSTALLER="uv")) == "uv"
    assert updates.how_installed(tmp_path / "venv", Dist(INSTALLER="pip")) == "pip"
    assert updates.how_installed(tmp_path / "venv", Dist()) == "pip"


def test_the_command_runs_pip_through_this_python_in_each_shell():
    posix = updates.update_command("pip", windows=False, python="/opt/my env/bin/python", dist=Dist())
    assert posix == "'/opt/my env/bin/python' -m pip install --upgrade smappy-smlm"
    # PowerShell runs a quoted program only after `&`
    spaced = updates.update_command("pip", windows=True, python=r"C:\Users\J R\python.exe", dist=Dist())
    assert spaced == r'& "C:\Users\J R\python.exe" -m pip install --upgrade smappy-smlm'
    plain = updates.update_command("pip", windows=True, python=r"C:\py\python.exe", dist=Dist())
    assert plain == r"C:\py\python.exe -m pip install --upgrade smappy-smlm"
    assert updates.update_command("uv tool", dist=Dist()) == "uv tool upgrade smappy-smlm"


def test_a_checkout_is_pulled_where_it_is_and_installed_again():
    dist = Dist(**{"direct_url.json": json.dumps(
        {"url": "file:///home/me/SMAPpy", "dir_info": {"editable": True}})})
    command = updates.update_command(windows=False, python="/v/bin/python", dist=dist)
    assert command == "cd /home/me/SMAPpy && git pull && /v/bin/python -m pip install -e ."
    # Windows PowerShell 5.1 has no &&
    assert "; git pull; " in updates.update_command(windows=True, python="py.exe", dist=dist)


def test_a_check_is_due_once_a_day_and_not_at_all_when_switched_off(own_config):
    assert updates.automatic() and updates.due(now=1e9)
    updates.remember("99.0", now=1e9)
    assert not updates.due(now=1e9 + 3600)
    assert updates.due(now=1e9 + updates.INTERVAL_S + 1)
    # a last check in the future (a clock set back) would wait as long: check now
    assert updates.due(now=1e9 - 3600)
    updates.set_automatic(False)
    assert not updates.due(now=1e9 + 10 * updates.INTERVAL_S)


def test_what_an_earlier_check_found_is_offered_only_while_this_copy_is_older(own_config):
    assert updates.known_newer() is None
    updates.remember("99.0")
    assert updates.known_newer() == "99.0"
    updates.remember("0.0.1")
    assert updates.known_newer() is None



def test_what_changed_is_the_comparison_of_the_two_release_tags():
    assert (updates.changes_url("0.4.1+g12ab", "0.5.0")
            == "https://github.com/ries-lab/SMAPpy/compare/v0.4.1...v0.5.0")

# ------------------------------------------------------------------ the GUI

def _app():
    pytest.importorskip("PySide6")
    from smappy.gui import make_app
    return make_app()


def _serve(body: bytes, status: int = 200):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _wait_for(signal, start, timeout_ms=10000):
    from PySide6.QtCore import QEventLoop, QTimer
    got = []
    loop = QEventLoop()

    def take(*args):
        got.append(args)
        loop.quit()

    signal.connect(take)
    QTimer.singleShot(timeout_ms, loop.quit)
    start()
    loop.exec()
    return got


def test_the_checker_reads_the_newest_version_and_remembers_it(own_config):
    _app()
    from smappy.gui.updates import UpdateChecker
    server = _serve(json.dumps({"info": {"version": "99.1"}}).encode())
    try:
        checker = UpdateChecker(url=f"http://127.0.0.1:{server.server_port}/json")
        got = _wait_for(checker.answered, checker.check)
    finally:
        server.shutdown()
    assert got == [("99.1", "")]
    assert config.get("updates_newest") == "99.1"
    assert not updates.due()


def test_a_failed_check_answers_none_and_why(own_config):
    _app()
    from smappy.gui.updates import UpdateChecker, message
    server = _serve(b"maintenance", status=500)
    try:
        checker = UpdateChecker(url=f"http://127.0.0.1:{server.server_port}/json")
        got = _wait_for(checker.answered, checker.check)
    finally:
        server.shutdown()
    assert len(got) == 1 and got[0][0] is None and got[0][1], got
    assert "Could not ask PyPI" in message(None, got[0][1])


def test_a_newer_release_puts_a_button_in_the_status_bar(own_config):
    _app()
    from PySide6.QtWidgets import QMainWindow
    from smappy.gui.updates import Updates, message
    window = QMainWindow()
    found = Updates(window)
    found._answered("0.0.1", "")
    assert found.button is None                  # older: nothing to say
    found._answered("99.2", "")
    assert found.button.text() == "SMAPpy 99.2 is available"
    assert "99.2 is available" in message("99.2") and "pip install" in message("99.2")
    assert "is the newest version" in message("0.0.1")
    window.deleteLater()


def test_startup_offers_what_an_earlier_check_found_without_asking_again(own_config):
    _app()
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMainWindow
    from smappy.gui.updates import Updates
    updates.remember("99.3")
    window = QMainWindow()
    found = Updates(window, url="http://127.0.0.1:9/never")
    found.start()                                # checked just now: not due
    assert found.button.text() == "SMAPpy 99.3 is available"
    assert not found.findChildren(QTimer)        # and no check started
    window.deleteLater()
