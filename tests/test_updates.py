"""Checking for a newer SMAPpy: the comparison, the command for each kind of
installation, the once-a-day schedule, and the GUI's request and button."""
import json
import shlex
import sys
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
    pixi = _pixi_env(tmp_path / "ws" / ".pixi" / "envs" / "default", tmp_path / "ws" / "pixi.toml")
    assert updates.how_installed(pixi, Dist(INSTALLER="uv-pixi")) == "pixi"
    assert updates.how_installed(pixi, editable) == "checkout"      # pixi installs it again, below


def _pixi_env(prefix, manifest, name="default"):
    """An environment as pixi leaves it: its workspace and name in conda-meta/pixi."""
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "conda-meta" / "pixi").write_text(json.dumps(
        {"manifest_path": str(manifest), "environment_name": name, "pixi_version": "0.81.0"}))
    return prefix


def test_a_pixi_environment_names_its_workspace_wherever_it_is(tmp_path):
    # detached: in pixi's cache, nowhere near the workspace it belongs to
    detached = _pixi_env(tmp_path / "cache" / "analysis-1a2b" / "envs" / "smlm",
                         tmp_path / "analysis" / "pixi.toml", "smlm")
    assert updates.pixi_environment(detached) == (tmp_path / "analysis" / "pixi.toml", "smlm")
    # an older pixi's, without the file: the environment's place says it
    workspace = tmp_path / "project"
    (workspace / ".pixi" / "envs" / "gpu").mkdir(parents=True)
    (workspace / "pyproject.toml").write_text("[tool.pixi.workspace]\n")
    assert updates.pixi_environment(workspace / ".pixi" / "envs" / "gpu") == (
        workspace / "pyproject.toml", "gpu")
    assert updates.pixi_environment(tmp_path / "venv") is None


def test_the_command_runs_pip_through_this_python_in_each_shell(monkeypatch):
    posix = updates.update_command("pip", windows=False, python="/opt/my env/bin/python",
                                   dist=Dist(), pip=True)
    assert posix == "'/opt/my env/bin/python' -m pip install --upgrade smappy-smlm"
    # PowerShell runs a quoted program only after `&`
    spaced = updates.update_command("pip", windows=True, python=r"C:\Users\J R\python.exe",
                                    dist=Dist(), pip=True)
    assert spaced == r'& "C:\Users\J R\python.exe" -m pip install --upgrade smappy-smlm'
    plain = updates.update_command("pip", windows=True, python=r"C:\py\python.exe",
                                   dist=Dist(), pip=True)
    assert plain == r"C:\py\python.exe -m pip install --upgrade smappy-smlm"
    # a uv tool without a receipt to read: the plain upgrade is all there is to say
    assert updates.update_command("uv tool", dist=Dist(), prefix="/nowhere") == (
        "uv tool upgrade smappy-smlm")


def test_the_command_for_a_uv_tool_is_its_install_quoted_for_each_shell(tmp_path):
    _receipt(tmp_path, '{ name = "smappy-smlm", extras = ["uipsf"], specifier = "==0.7.1" }')
    posix = updates.update_command("uv tool", windows=False, dist=Dist(), prefix=str(tmp_path),
                                   extras=[], torch="2.9.0+cu126", version="0.7.5")
    assert posix == ("uv tool install --refresh-package smappy-smlm 'smappy-smlm[uipsf]>=0.7.5' "
                     "--torch-backend cu126")
    windows = updates.update_command("uv tool", windows=True, dist=Dist(), prefix=str(tmp_path),
                                     extras=[], torch=None, version="0.7.5")
    # PowerShell's > is a redirection: the requirement is quoted there too
    assert windows == 'uv tool install --refresh-package smappy-smlm "smappy-smlm[uipsf]>=0.7.5"'


def _editable(url="file:///home/me/SMAPpy", installer=None):
    files = {"direct_url.json": json.dumps({"url": url, "dir_info": {"editable": True}})}
    if installer:
        files["INSTALLER"] = installer + "\n"
    return Dist(**files)


def test_a_checkout_is_pulled_where_it_is_and_installed_again(tmp_path):
    dist = _editable()
    command = updates.update_command(windows=False, python="/v/bin/python", dist=dist,
                                     prefix=tmp_path, pip=True)
    if sys.platform != "win32":             # there the file URL is a Windows path, \home\me
        assert command == "cd /home/me/SMAPpy && git pull && /v/bin/python -m pip install -e ."
    assert command.endswith(" && git pull && /v/bin/python -m pip install -e .")
    # Windows PowerShell 5.1 has no &&
    assert "; git pull; " in updates.update_command(windows=True, python="py.exe", dist=dist,
                                                    prefix=tmp_path, pip=True)


def test_a_checkout_is_installed_again_by_the_tool_that_installed_it(tmp_path):
    # neither a pixi environment nor a uv one has pip, which the command named
    # whatever had installed the checkout
    manifest = tmp_path / "my analysis" / "pixi.toml"
    pixi = _pixi_env(tmp_path / "env", manifest, "smlm")
    command = updates.update_command(windows=False, python="/v/bin/python",
                                     dist=_editable(installer="uv-pixi"), prefix=pixi, pip=False)
    assert command.endswith(f" && git pull && pixi reinstall --manifest-path "
                            f"{shlex.quote(str(manifest))} -e smlm smappy-smlm")
    command = updates.update_command(windows=False, python="/v/bin/python",
                                     dist=_editable(installer="uv"), prefix=tmp_path, pip=False)
    assert command.endswith(" && git pull && uv pip install --python /v/bin/python -e .")
    # installed by something else, and no pip to fall back on: no command at all
    assert updates.update_command(windows=False, dist=_editable(installer="hatch"),
                                  prefix=tmp_path, pip=False) == ""


def test_a_pixi_environment_upgrades_in_its_workspace_and_installs_the_environment(tmp_path):
    from smappy import updater
    manifest = tmp_path / "analysis" / "pixi.toml"
    pixi = _pixi_env(tmp_path / "env", manifest, "smlm")
    # upgrade, not update: it moves the manifest's range (">=0.6.2, <0.7" as
    # `pixi add` writes it) past a new minor version; and it installs only the
    # default environment
    assert updates.update_command(windows=False, dist=Dist(INSTALLER="uv-pixi"), prefix=pixi,
                                  pip=False) == (
        f"pixi upgrade --manifest-path {shlex.quote(str(manifest))} smappy-smlm"
        f" && pixi install --manifest-path {shlex.quote(str(manifest))} -e smlm")
    workspace = updates.pixi_environment(pixi)
    assert updater.commands("pixi", "/env/bin/python", pixi="/bin/pixi", workspace=workspace) == [
        ["/bin/pixi", "upgrade", "--manifest-path", str(manifest), "smappy-smlm"],
        ["/bin/pixi", "install", "--manifest-path", str(manifest), "-e", "smlm"]]
    assert updater.commands("pixi", "/env/bin/python", pixi="", workspace=workspace) is None


def test_pixi_is_found_where_smappy_was_started_from_or_where_its_installer_put_it(
        tmp_path, monkeypatch):
    from smappy import updater
    monkeypatch.setenv("PATH", "")                  # a launcher's: no ~/.pixi/bin
    monkeypatch.delenv("PIXI_EXE", raising=False)
    monkeypatch.setenv("PIXI_HOME", str(tmp_path / "nowhere"))
    assert updater.find_pixi() is None
    installed = tmp_path / "home" / "bin" / ("pixi.exe" if sys.platform == "win32" else "pixi")
    installed.parent.mkdir(parents=True)
    installed.write_text("")
    monkeypatch.setenv("PIXI_HOME", str(tmp_path / "home"))
    assert updater.find_pixi() == str(installed)
    started = tmp_path / "pixi-that-ran-us"
    started.write_text("")
    monkeypatch.setenv("PIXI_EXE", str(started))
    assert updater.find_pixi() == str(started)


def test_without_pip_or_a_known_installer_there_is_no_command_to_give(tmp_path):
    from smappy import updater
    assert updates.update_command(dist=Dist(), prefix=tmp_path, pip=False) == ""
    assert updater.commands("pip", "/env/bin/python", pip=False) is None


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


def test_a_newer_release_puts_a_button_in_the_status_bar(own_config, monkeypatch):
    _app()
    from PySide6.QtWidgets import QMainWindow
    from smappy.gui.updates import NO_COMMAND, Updates, message
    window = QMainWindow()
    found = Updates(window)
    found._answered("0.0.1", "")
    assert found.button is None                  # older: nothing to say
    found._answered("99.2", "")
    assert found.button.text() == "SMAPpy 99.2 is available"
    # the command is this installation's: pin one
    monkeypatch.setattr(updates, "update_command", lambda: "pip install --upgrade smappy-smlm")
    assert "99.2 is available" in message("99.2") and "pip install" in message("99.2")
    assert "is the newest version" in message("0.0.1")
    monkeypatch.setattr(updates, "update_command", lambda: "")
    assert NO_COMMAND in message("99.2") and "<pre>" not in message("99.2")
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

def test_update_and_restart_is_offered_where_pip_or_uv_can_do_it(tmp_path):
    from smappy import updater
    py = "/env/bin/python"
    assert updater.commands("pip", py) == [[py, "-m", "pip", "install", "--upgrade", "smappy-smlm"]]
    assert updater.commands("uv tool", py, uv="/bin/uv", prefix=_receipt(
        tmp_path, '{ name = "smappy-smlm" }'))[0][:3] == ["/bin/uv", "tool", "install"]
    assert updater.commands("uv tool", py, uv="/bin/uv", prefix=tmp_path / "none") is None
    assert updater.commands("uv", py, uv="/bin/uv")[0][:3] == ["/bin/uv", "pip", "install"]
    assert updater.commands("uv tool", py, uv="") is None          # no uv on the PATH
    assert updater.commands("checkout", py) is None                # git pull is the user's
    assert updater.commands("uvx", py) is None


def _receipt(folder, *requirements, extra=""):
    """A uv tool's environment as uv 0.12.4 writes its receipt."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "uv-receipt.toml").write_text(
        "[tool]\nrequirements = [\n" + "".join(f"    {r},\n" for r in requirements) + "]\n"
        'entrypoints = [\n    { name = "smappy-gui", install-path = "/bin/smappy-gui", '
        'from = "smappy-smlm" },\n]\n' + extra)
    return folder


def test_a_uv_tool_is_updated_by_one_install_that_keeps_everything_it_has(tmp_path):
    """Issue 189: `uv tool upgrade` kept the pin of `smappy-smlm[uipsf]==0.7.1`
    and fetched PyPI's torch over the CUDA build uv had not recorded.  The
    install names the new version, repeats the receipt -- which it replaces --
    and adds what the receipt cannot hold."""
    from smappy import updater
    prefix = _receipt(
        tmp_path, '{ name = "smappy-smlm", extras = ["uipsf"], specifier = "==0.7.1" }',
        '{ name = "rich", specifier = ">=13" }',
        extra='python = "3.13"\n\n[tool.options]\nprerelease = "allow"\n'
              'index = [{ name = "lab", url = "https://pypi.lab.org/simple", explicit = false, '
              'default = false }]\nextra-index-url = ["https://more.org/simple"]\n')
    assert updater.tool_install(prefix, "0.7.5", extras=["decode"], torch="2.9.0+cu126") == [
        "tool", "install", "--refresh-package", "smappy-smlm", "smappy-smlm[decode,uipsf]>=0.7.5",
        "--with", "rich>=13", "--python", "3.13", "--prerelease", "allow",
        "--index", "lab=https://pypi.lab.org/simple", "--extra-index-url", "https://more.org/simple",
        "--torch-backend", "cu126"]
    # torch added afterwards by Install PyTorch, which the receipt never heard of
    plain = _receipt(tmp_path / "plain", '{ name = "smappy-smlm" }')
    assert updater.tool_install(plain, None, extras=["uipsf"], torch="2.9.0") == [
        "tool", "install", "--refresh-package", "smappy-smlm", "smappy-smlm[uipsf]@latest"]
    # SMAPpy from a folder or git: nothing an index could update
    local = _receipt(tmp_path / "local", '{ name = "smappy-smlm", directory = "/src/SMAPpy" }')
    assert updater.tool_install(local, "0.7.5", extras=[], torch=None) is None
    assert updater.tool_install(tmp_path / "no receipt", "0.7.5", extras=[], torch=None) is None


def test_the_torch_backend_is_read_from_the_installed_build():
    from smappy import updater
    assert updater.torch_backend("2.9.0+cu126") == "cu126"
    assert updater.torch_backend("2.9.0+cpu") == "cpu"
    assert updater.torch_backend("2.9.0+rocm6.3") == "rocm6.3"
    assert updater.torch_backend("2.9.0") is None              # PyPI's own
    assert updater.torch_backend("2.9.0+g1234") is None        # a local build of anything else
    assert updater.torch_backend(None) is None


def test_the_extras_kept_are_those_whose_packages_are_all_here():
    from smappy import updater
    requires = ["numpy>=1.24", "matplotlib>=3.7", 'matplotlib>=3.7.3; extra == "drift"',
                'torch>=2.4; extra == "uipsf"', 'torch>=2.4; extra == "decode"',
                'tqdm>=4.60; extra == "decode"', "pillow>=10.0 ; extra == 'image'"]
    here = {"numpy", "torch", "pillow"}.__contains__
    assert updater.extras_present(requires, here) == ["image", "uipsf"]
    assert updater.extras_present(requires, lambda name: True) == ["decode", "image", "uipsf"]


def test_a_uv_older_than_the_command_needs_is_said_before_the_update():
    from smappy import updater
    assert "0.9.19" in updater.uv_too_old("uv", (0, 8, 4))
    assert updater.uv_too_old("uv", (0, 9, 19)) is None
    assert updater.uv_too_old("uv", (0, 12, 4)) is None
    assert updater.uv_too_old(None) is None                   # cannot tell: the install says


def test_an_update_that_loses_torch_says_so_at_the_next_start(tmp_path):
    from smappy import updater
    plan = {"wait_for": [], "commands": [_python("pass")], "relaunch": [],
            "checks": [_python("import torch_that_is_gone")],
            "result": str(tmp_path / "r.json"), "log": str(tmp_path / "u.log"), "from_version": "0.7.1"}
    result = updater.run_plan(plan, retry_pause_s=0)
    assert result["ok"] and result["lost"] == ["import torch_that_is_gone"]
    assert "after the update this fails" in (tmp_path / "u.log").read_text()
    said = updates.lost_message(result["lost"], "0.7.1", "0.7.5")
    assert said.startswith("SMAPpy was updated from 0.7.1 to 0.7.5, but torch_that_is_gone")
    assert "Install PyTorch" in said
    assert updater.torch_check("py", torch="2.9.0") == [["py", "-I", "-c", "import torch"]]
    assert updater.torch_check("py", torch=None) == []


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


def test_the_helper_runs_on_the_python_the_environment_was_made_from(monkeypatch, tmp_path):
    from smappy import updater
    base = tmp_path / "python.exe"
    base.write_text("")
    # in a venv (a uv tool's): its own python.exe is locked on Windows while
    # the helper runs, and uv rebuilds the environment around it
    monkeypatch.setattr(sys, "_base_executable", str(base), raising=False)
    assert updater.helper_python() == str(base)
    monkeypatch.setattr(sys, "_base_executable", str(tmp_path / "gone.exe"), raising=False)
    assert updater.helper_python() == sys.executable


def test_the_failure_window_quotes_uvs_error_and_its_cause(own_config):
    updates.log_file().write_text(
        "SMAPpy update\n\n$ uv tool upgrade smappy-smlm\n"
        "error: Failed to upgrade smappy-smlm\n"
        "  Caused by: failed to remove directory `C:\\uv\\tools\\smappy-smlm`\n"
        "  Caused by: The process cannot access the file (os error 32)\n"
        "exit code 2; trying again in 5 s\n\nthe update failed\n")
    error = updates.log_error()
    assert error.startswith("error: Failed to upgrade") and error.endswith("(os error 32)")
    assert "trying again" not in error
    updates.log_file().write_text("nothing went wrong here\n")
    assert updates.log_error() == ""


@pytest.mark.parametrize("output, advice", [
    # a user's uv, October 2026
    ("error: unexpected argument '--refresh-package' found\n"
     "  tip: a similar argument exists: '--prerelease-package'\n", "uv self update"),
    ("error: failed to remove file `C:\\uv\\tools\\smappy-smlm\\Scripts\\python.exe`\n"
     "  Caused by: The process cannot access the file (os error 32)\n", "Close every SMAPpy window"),
    ("ERROR: Could not install packages due to an OSError: [WinError 5] Access is denied\n",
     "Close every SMAPpy window"),
    ("error: Failed to fetch: `https://pypi.org/simple/smappy-smlm/`\n"
     "  Caused by: dns error: No such host is known\n", "HTTPS_PROXY"),
    ("error: something nobody has seen\n", ""),
])
def test_a_known_failure_says_what_to_do_about_it(own_config, output, advice):
    updates.log_file().write_text(
        f"SMAPpy update\n\n$ C:\\Users\\x\\.local\\bin\\uv.EXE tool upgrade smappy-smlm\n{output}"
        "\nthe update failed\n")
    found = updates.failure_advice()
    assert advice in found if advice else found == ""


def test_an_update_that_ran_but_changed_nothing_says_why_it_might_have(own_config):
    assert "ten minutes" in updates.failure_advice(ok=True)


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
        (box.text(), box.detailedText(), box.informativeText())))
    window = QMainWindow()
    Updates(window).report_last_update()
    assert shown and shown[0][0].startswith("The update failed")
    assert "Access is denied" in shown[0][1]
    assert shown[0][2].startswith("A file of SMAPpy was still in use")     # what to do first
    assert "ERROR: Could not install packages" in shown[0][2]            # then why
    window.deleteLater()


def test_an_update_that_lost_torch_says_so_and_how_to_put_it_back(own_config, monkeypatch):
    _app()
    from PySide6.QtWidgets import QMainWindow, QMessageBox
    from smappy.gui.updates import Updates
    updates.log_file().write_text("after the update this fails: -I -c import torch\n")
    updates.result_file().write_text(json.dumps({"ok": True, "from_version": "0.0.1",
                                                 "lost": ["import torch"],
                                                 "log": str(updates.log_file())}))
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: shown.append(box.text()))
    window = QMainWindow()
    Updates(window).report_last_update()
    assert shown and "torch no longer imports" in shown[0] and "Install PyTorch" in shown[0]
    window.deleteLater()


def test_a_uv_too_old_is_said_before_smappy_closes(own_config, monkeypatch):
    _app()
    from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
    from smappy import updater
    from smappy.gui import updates as gui_updates
    monkeypatch.setattr(gui_updates, "install_commands", lambda: [["/old/uv", "tool", "install"]])
    monkeypatch.setattr(updates, "how_installed", lambda *a, **k: "uv tool")
    monkeypatch.setattr(updater, "uv_version", lambda uv: (0, 8, 0))
    started, quits, shown = [], [], []
    monkeypatch.setattr(updater, "start_helper", started.append)
    monkeypatch.setattr(QApplication, "quit", lambda: quits.append(1))
    monkeypatch.setattr(QMessageBox, "exec", lambda box: shown.append(box.text()))
    window = QMainWindow()
    window.may_close = lambda: True
    gui_updates.Updates(window).update_and_restart()
    assert not started and not quits
    assert "0.9.19" in shown[0] and "uv self update" in shown[0]
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
    assert plan["checks"] == updater.torch_check(sys.executable)
    window.deleteLater()
