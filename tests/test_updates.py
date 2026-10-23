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
    from smappy.gui.updates import START_DELAY_MS
    assert not [t for t in found.findChildren(QTimer) if t.interval() == START_DELAY_MS]
    window.deleteLater()


# ------------------------------------------------------- update and restart

def test_update_and_restart_is_offered_where_pip_or_uv_can_do_it():
    from smappy import updater
    py = "/env/bin/python"
    assert updater.commands("pip", py) == [[py, "-m", "pip", "install", "--upgrade", "smappy-smlm"]]
    assert updater.commands("uv tool", py, uv="/bin/uv") == [["/bin/uv", "tool", "upgrade", "smappy-smlm"]]
    assert updater.commands("uv", py, uv="/bin/uv")[0][:3] == ["/bin/uv", "pip", "install"]
    assert updater.commands("uv tool", py, uv="") is None          # no uv on the PATH
    assert updater.commands("checkout", py) is None                # git pull is the user's
    assert updater.commands("uvx", py) is None


def _python(code):
    import sys
    return [sys.executable, "-c", code]


def _until(condition, seconds=30):
    import time
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.1)
    return False


def test_the_helper_installs_after_smappy_has_gone_and_starts_it_again(tmp_path):
    import subprocess

    from smappy import updater
    gone = tmp_path / "gone"
    # stands for SMAPpy: still running when the helper starts; nobody waits
    # for it, so it stays a zombie on Linux, which must count as gone
    smappy = subprocess.Popen(_python(f"import time; time.sleep(1); open({str(gone)!r}, 'w')"))
    marker = tmp_path / "restarted"
    folder = tmp_path / "plan"
    folder.mkdir()
    plan = updater.write_plan(
        folder, [_python(f"import os; print('installing', os.path.exists({str(gone)!r}))")],
        _python(f"open({str(marker)!r}, 'w').write('yes')"),
        tmp_path / "result.json", tmp_path / "update.log", "0.4.1")
    data = json.loads(plan.read_text())
    data["wait_for"] = [smappy.pid]
    plan.write_text(json.dumps(data))
    updater.start_helper(plan)
    assert _until(marker.exists), (tmp_path / "update.log").read_text()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["ok"] and result["from_version"] == "0.4.1"
    assert "installing True" in (tmp_path / "update.log").read_text()   # only after it had gone
    assert (folder / "smappy_updater.py").exists()                     # run from a copy
    smappy.wait()


def test_a_failed_install_is_tried_twice_and_reported(tmp_path):
    from smappy import updater
    count = tmp_path / "tries"
    plan = {"wait_for": [], "commands": [_python(
        f"import sys; open({str(count)!r}, 'a').write('x'); sys.exit(3)")],
        "relaunch": [], "result": str(tmp_path / "r.json"), "log": str(tmp_path / "u.log"),
        "from_version": "0.4.1"}
    result = updater.run_plan(plan, retry_pause_s=0)
    assert not result["ok"] and result["code"] == 3
    assert count.read_text() == "xx"
    assert "the update failed" in (tmp_path / "u.log").read_text()


def test_the_next_start_says_the_update_worked_and_reads_the_result_once(own_config, monkeypatch):
    _app()
    from PySide6.QtWidgets import QMainWindow
    from smappy.gui.updates import Updates
    updates.result_file().write_text(json.dumps({"ok": True, "from_version": "0.0.1"}))
    window = QMainWindow()
    Updates(window).report_last_update()
    assert window.statusBar().currentMessage().startswith("SMAPpy updated from 0.0.1 to")
    assert updates.last_update() is None
    window.deleteLater()


def test_a_failed_update_is_shown_with_pips_output(own_config, monkeypatch):
    _app()
    from PySide6.QtWidgets import QMainWindow, QMessageBox
    from smappy.gui.updates import Updates
    updates.log_file().write_text("ERROR: Could not install packages: Access is denied\n")
    updates.result_file().write_text(json.dumps({"ok": False, "from_version": "0.4.1",
                                                 "log": str(updates.log_file())}))
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: shown.append(
        (box.text(), box.detailedText())))
    window = QMainWindow()
    Updates(window).report_last_update()
    assert shown and shown[0][0].startswith("The update failed")
    assert "Access is denied" in shown[0][1]
    window.deleteLater()


def test_update_and_restart_starts_the_helper_and_quits_only_if_the_window_agrees(
        own_config, monkeypatch):
    _app()
    from PySide6.QtWidgets import QApplication, QMainWindow
    from smappy import updater
    from smappy.gui import updates as gui_updates
    started, quits = [], []
    monkeypatch.setattr(gui_updates, "install_commands", lambda: [["pip", "install"]])
    monkeypatch.setattr(updater, "start_helper", lambda plan: started.append(json.loads(plan.read_text())))
    monkeypatch.setattr(QApplication, "quit", lambda: quits.append(1))
    window = QMainWindow()
    window.may_close = lambda: False                 # Cancel at "unsaved changes"
    found = gui_updates.Updates(window)
    found.update_and_restart()
    assert not started and not quits
    window.may_close = lambda: True
    found.update_and_restart()
    assert quits and window.leaving
    plan = started[0]
    assert plan["commands"] == [["pip", "install"]] and plan["relaunch"][-2:] == ["-m", "smappy.cli.gui"]
    assert plan["result"] == str(updates.result_file())
    window.deleteLater()
