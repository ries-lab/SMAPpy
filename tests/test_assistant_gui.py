"""The assistant's tools from the running GUI: the socket, the queue, the pane
and ``smappy-mcp --attach``."""
import functools
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from smappy.assistant import attach  # noqa: E402

SIMULATION = {"n_frames": 5000}


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class Viewer:
    """The 2D view as `RenderViewer` is, without a window."""

    def __init__(self):
        self.box = ((0.0, 4000.0), (0.0, 3000.0))

    def frame(self, xrange, yrange):
        self.box = (tuple(xrange), tuple(yrange))

    def reset(self):
        self.box = ((0.0, 20000.0), (0.0, 15000.0))

    def current(self):
        (x0, x1), _ = self.box
        return (*self.box, (x1 - x0) / 800)


def wait_for(condition, seconds=10.0):
    from PySide6.QtWidgets import QApplication
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def server(app, tmp_path):
    from smappy.gui.assistant_server import AssistantServer
    from smappy.session import Session
    made = AssistantServer(Session(), Viewer(), registry=tmp_path / "gui")
    made.assistant.knowledge = tmp_path / "kb"
    yield made
    made.stop()


def simulate(server):
    """Data to work on, made straight on the session: not through a panel,
    whose run would end after this returns."""
    a = server.assistant
    (mode, a.mode), (runner, a.runner) = (a.mode, "act"), (a.runner, None)
    a.call("run_plugin", {"path": "File/Simulate/Simulate", "settings": SIMULATION,
                          "apply": True})
    a.mode, a.runner = mode, runner


class Client:
    """A client on the GUI's socket, in this thread: the server's queue runs
    while it waits."""

    def __init__(self, address):
        from PySide6.QtNetwork import QLocalSocket
        self.socket = QLocalSocket()
        self.socket.connectToServer(address)
        assert wait_for(lambda: self.socket.state() == QLocalSocket.ConnectedState)
        self.buffer = b""
        self.n = 0

    def send(self, method, params=None):
        self.n += 1
        message = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}
        self.socket.write((json.dumps(message) + "\n").encode())
        self.socket.flush()
        return self.n

    def answer(self, seconds=30.0):
        def complete():
            self.buffer += bytes(self.socket.readAll().data())
            return b"\n" in self.buffer
        assert wait_for(complete, seconds), "no answer"
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def call(self, name, arguments=None):
        self.send("tools/call", {"name": name, "arguments": arguments or {}})
        return self.answer()["result"]


# ------------------------------------------------------------------ the socket

def test_a_client_on_the_socket_gets_the_tools_and_each_call_runs_on_the_gui_thread(
        server, tmp_path, monkeypatch):
    assert server.start() and server.listening
    entries = attach.listening(tmp_path / "gui")
    assert [e["server"] for e in entries] == [server.address]
    threads = []
    original = type(server.assistant).session_state

    @functools.wraps(original)          # keeps the mark that makes it a tool
    def recorded(self):
        threads.append(threading.current_thread())
        return original(self)
    monkeypatch.setattr(type(server.assistant), "session_state", recorded)
    client = Client(server.address)
    client.send("initialize", {"protocolVersion": "2025-06-18",
                               "clientInfo": {"name": "test", "version": "1"}})
    hello = client.answer()["result"]
    assert "running SMAPpy GUI" in hello["instructions"]
    from smappy.assistant.tools import source_folder
    assert str(source_folder()) in hello["instructions"]      # where its code is
    assert server.client_names() == ["test 1"]
    client.send("tools/list")
    names = {t["name"] for t in client.answer()["result"]["tools"]}
    # no `open_plugin` here: the bare server has no windows to open
    assert {"show", "session_state", "run_plugin"} <= names
    state = json.loads(client.call("session_state")["content"][0]["text"])
    assert state["mode"] == "suggest"
    assert threads == [threading.main_thread()]
    assert server.calls[-1][1:] == ("session_state", True)
    server.stop()
    assert attach.listening(tmp_path / "gui") == []
    assert wait_for(lambda: client.socket.state() != type(client.socket).ConnectedState)


def test_lines_are_answered_in_order_and_one_that_breaks_does_not_stop_the_queue(
        server, monkeypatch):
    server.start()
    client = Client(server.address)

    @functools.wraps(type(server.assistant).list_proposals)
    def broken(self):
        raise RuntimeError("a bug")
    monkeypatch.setattr(type(server.assistant), "list_proposals", broken)
    # three at once, in one write: the queue takes them one per tick
    for name in ("list_proposals", "session_state", "nonexistent"):
        client.send("tools/call", {"name": name, "arguments": {}})
    first, second, third = (client.answer() for _ in range(3))
    assert [a["id"] for a in (first, second, third)] == [1, 2, 3]
    assert first["result"]["isError"] and "a bug" in first["result"]["content"][0]["text"]
    assert not second["result"]["isError"]
    assert third["result"]["isError"]
    client.socket.write(b"not json\n")
    assert client.answer()["error"]["code"] == -32700
    assert not client.call("session_state")["isError"]


def test_the_name_given_to_attach_picks_a_window_by_its_session_or_process():
    entries = [{"session": "cells A", "pid": 11}, {"session": "beads", "pid": 12},
               {"session": "cells B", "pid": 13}]
    assert attach.choose(entries) == entries
    assert attach.choose(entries, "beads") == [entries[1]]
    assert attach.choose(entries, "13") == [entries[2]]
    assert attach.choose(entries, "Cells") == [entries[0], entries[2]]
    assert attach.choose(entries, "nuclei") == []


# ------------------------------------------------------------------ the modes

def test_with_the_gui_attached_view_changes_are_made_in_suggest_mode(server):
    simulate(server)
    a = server.assistant
    assert a.mode == "suggest"
    shown = a.call("set_filter", {"column": "photons", "lo": 500})
    assert shown["filter"]["photons"][0] == 500 and not a.proposals
    a.call("set_grouped", {"on": True})
    assert server.session.layers[0].grouped and not a.proposals
    view = a.call("show", {"x_nm": 1000, "y_nm": 2000, "width_nm": 400})["view"]
    assert view["x_nm"] == [800, 1200] and view["y_nm"] == [1850, 2150]
    assert a.call("show", {"everything": True})["view"]["x_nm"] == [0, 20000]
    with pytest.raises(Exception, match="one of"):
        a.call("show", {"x_nm": 1, "everything": True})
    # a change of the data is still a proposal
    a.call("run_plugin", {"path": "Analysis/Drift/RCC", "apply": True})
    assert [p.tool for p in a.proposals] == ["run_plugin"]


def test_headless_there_is_no_show_and_a_filter_stays_a_proposal(tmp_path):
    from smappy.assistant.tools import Assistant
    a = Assistant(mode="suggest", output=tmp_path, knowledge_folder=tmp_path / "kb")
    assert "show" in Assistant.specs() and "show" not in a.tools()
    with pytest.raises(Exception, match="no tool 'show'"):
        a.call("show", {"everything": True})


def test_show_frames_an_roi_of_the_roi_manager_by_its_number(server):
    simulate(server)
    project = server.session.rois
    roi = project.add_roi(next(iter(project.sources)), [5000.0, 6000.0])
    shown = server.assistant.call("show", {"roi": roi.number})["view"]
    half = project.size_nm / 2 + 0.1 * project.size_nm     # the margin
    assert shown["x_nm"] == pytest.approx([5000 - half, 5000 + half], abs=1)
    with pytest.raises(Exception, match="no ROI 99"):
        server.assistant.call("show", {"roi": 99})


# ------------------------------------------------------------------ the pane

@pytest.fixture
def controls(server, app, monkeypatch):
    from smappy.assistant import launch
    from smappy.gui import assistant_pane
    monkeypatch.setattr(assistant_pane, "preferences", lambda: dict(assistant_pane.DEFAULTS))
    # never the developer's own Terminal: an assistant open there would be
    # raised by the test, and a start skipped for it
    monkeypatch.setattr(launch, "raise_terminal", lambda: False)
    made = assistant_pane.AssistantControls(server.session, server)
    yield made
    for window in made._windows.values():
        window.close()


def test_a_proposed_run_opens_filled_in_and_running_it_there_carries_it_out(controls):
    from smappy.gui.assistant_pane import ProposalList
    server = controls.server
    simulate(server)
    listed = ProposalList(controls)
    a = server.assistant
    ident = a.call("run_plugin", {"path": "Analysis/Drift/RCC", "apply": True,
                                  "settings": {"n_timepoints": 7},
                                  "reason": "the structure smears along x"})["proposed"]["id"]
    server.changed.emit()
    assert not listed.empty.isVisibleTo(listed)
    assert listed.text.text() == "Run RCC drift correction: n_timepoints = 7"
    assert listed.why.text() == "the structure smears along x"
    assert not listed.page.isVisibleTo(listed)          # one: no pager
    window = controls.open_proposal(ident)
    assert window.panel.form.value().n_timepoints == 7
    assert a.proposal(ident).status == "opened" and not server.session.can_undo
    # the user changes a setting and runs it from the panel
    settings = window.panel.form.value()
    settings.n_timepoints = 6
    window.panel.form.set(settings)
    window.panel.run()
    assert wait_for(lambda: a.proposal(ident).status == "done", 60)
    done = a.call("list_proposals")[0]
    assert done["ran_with"]["n_timepoints"] == 6 and done["result"].startswith("r")
    assert a.call("figure", {"result": done["result"]})
    assert server.session.can_undo
    assert listed.empty.isVisibleTo(listed)


def test_open_plugin_opens_its_window_filled_in_and_proposes_nothing(controls):
    """"Open Tracking analysis" in a chat: the window, not a proposal to run."""
    a = controls.server.assistant
    assert a.mode == "suggest" and "open_plugin" in a.tools()
    opened = a.call("open_plugin", {"path": "Analysis/Drift/RCC",
                                    "settings": {"n_timepoints": 7}})
    assert opened["opened"] and not a.proposals
    window = controls._windows["Analysis/Drift/RCC"]
    assert window.isVisible() and window.panel.form.value().n_timepoints == 7
    assert not controls.server.session.can_undo          # nothing ran
    with pytest.raises(Exception, match="no plugin"):
        a.call("open_plugin", {"path": "Analysis/Nothing"})


def test_several_proposals_are_paged_the_newest_shown_and_the_next_after_one_goes(
        controls):
    from smappy.gui.assistant_pane import ProposalList
    simulate(controls.server)
    listed = ProposalList(controls)
    a = controls.server.assistant
    ids = [a.call("run_plugin", {"path": "Analysis/Drift/RCC", "apply": True,
                                 "settings": {"n_timepoints": n}})["proposed"]["id"]
           for n in (4, 5, 6)]
    controls.changed.emit()
    assert listed.current == ids[-1] and listed.page.text() == "3 / 3"
    assert listed.page.isVisibleTo(listed) and not listed.next.isEnabled()
    listed.previous.click()
    assert listed.current == ids[1] and "n_timepoints = 5" in listed.text.text()
    listed.dismiss.click()              # dealt with: the next takes its place
    assert listed.current == ids[2] and listed.page.text() == "2 / 2"
    listed.previous.click()
    listed.go.click()                   # Open: the first one's window, filled in
    window = controls._windows["Analysis/Drift/RCC"]
    assert window.panel.form.value().n_timepoints == 4
    assert a.proposal(ids[0]).status == "opened" and "(opened)" in listed.text.text()


def test_a_preset_named_loosely_fills_the_window_and_a_wrong_one_is_refused(controls):
    """"Simulate NPCs" opened Simulate with preset "Nuclear pores", which is
    no preset's name: the form showed it as missing and simulated the demo's
    ring and cross."""
    a = controls.server.assistant
    a.call("open_plugin", {"path": "File/Simulate/Simulate",
                           "settings": {"preset": "nuclear pores"}})
    form = controls._windows["File/Simulate/Simulate"].panel.form.value()
    assert form.preset == "Nuclear pores (Nup96)"
    assert form.components[0].structure.preset == "npc"      # the scene, filled
    with pytest.raises(Exception, match="Nuclear pores \\(Nup96\\)"):
        a.call("open_plugin", {"path": "File/Simulate/Simulate",
                               "settings": {"preset": "mitochondria"}})


def test_a_proposal_is_done_from_the_list_dismissed_and_the_switch_followed(controls):
    from smappy.gui.assistant_pane import ProposalList
    server = controls.server
    simulate(server)
    a = server.assistant
    a.mode = "act"
    controls.show_runs(False)           # applied at once, to have one to undo
    a.call("run_plugin", {"path": "Analysis/Drift/RCC", "apply": True,
                          "settings": {"n_timepoints": 5}})
    listed = ProposalList(controls)
    listed.suggest.setChecked(True)
    assert a.mode == "suggest"
    undo = a.call("undo")["proposed"]["id"]
    other = a.call("run_plugin", {"path": "Analysis/Drift/RCC", "apply": True})["proposed"]["id"]
    controls.do_proposal(undo)
    assert not server.session.can_undo and a.proposal(undo).status == "done"
    controls.dismiss(other)
    assert a.proposal(other).status == "dismissed" and controls.live() == []
    assert a.mode == "suggest"
    listed.act.setChecked(True)
    assert a.mode == "act"


def test_the_toolbar_button_says_who_is_connected_the_mode_and_what_waits(controls):
    from PySide6.QtWidgets import QWidget
    from smappy.gui.assistant_pane import AssistantButton
    picture = QWidget()
    picture.resize(900, 700)
    picture.show()
    button = AssistantButton(lambda: controls, host=picture)
    button.attach(controls)
    overlay = button.overlay
    assert button.text() == "Assistant \u25be" and not overlay.isVisible()
    controls.server.start()
    client = Client(controls.server.address)
    client.send("initialize", {"clientInfo": {"name": "claude-code", "version": "2"}})
    client.answer()
    assert button.text() == "\u25cf claude-code \u00b7 suggest \u25be"
    simulate(controls.server)
    proposed = client.call("run_plugin", {"path": "Analysis/Drift/RCC", "apply": True})
    ident = json.loads(proposed["content"][0]["text"])["proposed"]["id"]
    # a proposal opens the list, inside the picture's window, top right
    assert overlay.isVisible() and overlay.parent() is picture
    assert overlay.geometry().right() < picture.width() and overlay.y() < 20
    assert button.text().endswith("\u00b7 1 \u25be") and button.styleSheet() == ""
    overlay.collapse()                  # to look at the whole picture first
    assert "bold" in button.styleSheet()        # it still says something waits
    button.toggle()
    assert overlay.isVisible() and button.styleSheet() == ""
    controls.dismiss(ident)             # the last one dealt with: it folds away
    assert not overlay.isVisible() and button.text() == "\u25cf claude-code \u00b7 suggest \u25be"
    picture.close()


def test_start_assistant_brings_a_connected_one_forward_rather_than_a_second(
        controls, monkeypatch):
    from smappy.assistant import launch
    from smappy.gui import assistant_pane
    started, raised = [], []
    monkeypatch.setattr(assistant_pane, "preferences",
                        lambda: {**assistant_pane.DEFAULTS, "client": "claude"})
    monkeypatch.setattr(launch, "start", lambda *a, **k: started.append(a) or "started")
    open_terminal = []                  # no terminal of SMAPpy's is open yet
    monkeypatch.setattr(launch, "raise_terminal",
                        lambda: raised.append(1) or bool(open_terminal))
    controls.start()
    assert len(started) == 1 and controls.server.listening
    client = Client(controls.server.address)
    client.send("initialize", {"clientInfo": {"name": "claude-code"}})
    client.answer()
    controls.start()
    assert len(started) == 1 and len(raised) == 2
    assert "connected already" in controls.message


def test_after_a_restart_start_brings_the_open_terminal_forward_to_reconnect(
        controls, monkeypatch):
    from smappy.assistant import launch
    from smappy.gui import assistant_pane
    started = []
    monkeypatch.setattr(assistant_pane, "preferences",
                        lambda: {**assistant_pane.DEFAULTS, "client": "claude"})
    monkeypatch.setattr(launch, "start", lambda *a, **k: started.append(a) or "started")
    monkeypatch.setattr(launch, "raise_terminal", lambda: True)
    controls.start()                    # nobody connected, but its terminal is open
    assert not started and controls.server.listening
    assert "reconnects in a moment" in controls.message
    controls._check_reconnected()       # and if it does not
    assert "/mcp" in controls.message
    client = Client(controls.server.address)    # it comes late, after /mcp say
    client.send("initialize", {"clientInfo": {"name": "claude-code"}})
    client.answer()
    assert controls.message == ""       # nothing left saying it did not come
    controls.server.stop()


def test_the_first_start_asks_one_question_and_starts_what_was_chosen(controls, monkeypatch):
    from smappy.assistant import launch
    from smappy.gui import assistant_pane
    prefs = dict(assistant_pane.DEFAULTS)
    monkeypatch.setattr(assistant_pane, "preferences", lambda: dict(prefs))
    monkeypatch.setattr(assistant_pane, "save_preferences", prefs.update)
    started = []
    monkeypatch.setattr(launch, "start", lambda key, *a, **k: started.append(key) or "ok")
    controls.start()
    chooser = controls.chooser
    assert chooser.isVisible() and chooser.chosen() == "claude" and not started
    chooser.buttons["codex"].setChecked(True)
    chooser.start()
    assert started == ["codex"] and prefs["client"] == "codex"
    assert not chooser.isVisible()
    controls.server.stop()


def test_the_settings_window_keeps_the_client_and_shows_its_own_command(controls, monkeypatch):
    from smappy.gui import assistant_pane
    kept = {}
    monkeypatch.setattr(assistant_pane, "save_preferences", kept.update)
    window = assistant_pane.AssistantSettings(controls.session, controls)
    assert not window.command.isVisibleTo(window)
    window.client.setCurrentIndex(window.client.findData("command"))
    assert kept["client"] == "command" and window.command.isVisibleTo(window)
    assert not window.pre_allow.isVisibleTo(window) and not window.other.isVisibleTo(window)
    window.client.setCurrentIndex(window.client.findData("claude"))
    assert window.pre_allow.isVisibleTo(window) and not window.command.isVisibleTo(window)
    # one started by hand: what to give it, and only then
    window.client.setCurrentIndex(window.client.findData("manual"))
    assert window.other.isVisibleTo(window) and not window.pre_allow.isVisibleTo(window)
    assert not window.folder.isVisibleTo(window)
    # the mode it starts in: both choices in sight
    window.start_act.setChecked(True)
    assert kept["mode"] == "act"
    window.start_suggest.setChecked(True)
    assert kept["mode"] == "suggest"
    window.close()


def test_one_started_by_hand_is_listened_for_and_nothing_is_started(controls, monkeypatch):
    from smappy.assistant import launch
    from smappy.gui import assistant_pane
    monkeypatch.setattr(assistant_pane, "preferences",
                        lambda: {**assistant_pane.DEFAULTS, "client": "manual"})
    started = []
    monkeypatch.setattr(launch, "start", lambda *a, **k: started.append(a))
    controls.start()
    assert controls.server.listening and not started
    assert "Start your assistant" in controls.message
    controls.server.stop()


def test_the_output_folder_follows_the_first_file_until_the_user_sets_one(server, tmp_path):
    simulate(server)
    a = server.assistant
    # a simulation is no file on disk: not "." beside its label, but home
    assert a.output == Path.home()
    a.mode = "act"
    server.set_output(str(tmp_path / "out"))
    assert a.output == tmp_path / "out" and server.output_pinned
    saved = Path(a.call("save", {"path": "x.h5"})["saved"])
    server.set_output(None)
    server.session.load(saved)
    assert not server.output_pinned and a.output == saved.parent


# ------------------------------------------------------------- starting one

def test_claude_code_starts_with_the_server_on_its_command_line(tmp_path, monkeypatch):
    from smappy.assistant import launch
    monkeypatch.delenv("PYTHONPATH", raising=False)
    config = launch.write_config(tmp_path / "settings", pid=4242)
    entry = json.loads(config.read_text())["mcpServers"]["smappy-gui"]
    assert entry == {"command": sys.executable,
                     "args": ["-m", "smappy.assistant.server", "--attach", "4242"]}
    from smappy.assistant.tools import source_folder
    assert launch.client_command("claude", config) == [
        "claude", "--mcp-config", str(config), "--add-dir", str(source_folder()),
        "--allowedTools", "mcp__smappy-gui"]
    assert "--allowedTools" not in launch.client_command("claude", config, pre_allow=False)
    codex = launch.client_command("codex", config)
    assert codex[0] == "codex" and f'mcp_servers.smappy_gui.command={json.dumps(sys.executable)}' in codex
    own = launch.client_command("command", config, own="agent --mcp {config} -- {server}")
    assert own.startswith("agent --mcp ") and own.endswith("--attach 4242")
    with pytest.raises(ValueError, match="braces"):
        launch.client_command("command", config, own="agent {nonsense}")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="a shell script; Windows runs a .cmd")
def test_the_terminal_offers_to_install_a_missing_client_and_welcomes_the_user(tmp_path):
    from smappy.assistant import launch
    ran = []
    line = launch.start("claude", tmp_path / "work dir", tmp_path / "settings", 4242,
                        bounds=(10, 20, 500, 800), run=ran.append)
    assert "started" in line
    file = tmp_path / "settings" / "start-claude.sh"
    script = file.read_text()
    assert "cd '" + str(tmp_path / "work dir") + "'" in script
    assert "Install it now? [Y/n]" in script and "claude.ai/install.sh | bash" in script
    assert "Ask here what you need:" in script
    assert script.rstrip().split("\n")[-1].startswith("exec claude --mcp-config ")
    if sys.platform == "darwin":
        assert ran[0][0] == "osascript" and "{10, 20, 500, 800}" in ran[0][2]
    # not installed, and the user says no: nothing is installed, and they are
    # told the one line that would.  (Answered, never left to the default yes:
    # a test must not reach a real installer.)
    (tmp_path / "work dir").mkdir()
    empty = tmp_path / "bin"
    empty.mkdir()
    out = subprocess.run(["/bin/sh", str(file)], input="n\n", capture_output=True,
                         text=True, env={"PATH": str(empty), "HOME": str(tmp_path)})
    assert out.returncode == 1
    assert "Claude Code is not installed on this computer yet." in out.stdout
    assert "Not installed." in out.stdout and "claude.ai/install.sh" in out.stdout
    # installed: the welcome, then the client
    fake = empty / "claude"
    fake.write_text("#!/bin/sh\necho CLAUDE \"$@\"\n")
    fake.chmod(0o755)
    out = subprocess.run(["/bin/sh", str(file)], capture_output=True, text=True,
                         env={"PATH": f"{empty}:/bin:/usr/bin", "HOME": str(tmp_path)})
    assert out.returncode == 0, out.stderr
    assert "Ask here what you need:" in out.stdout and "CLAUDE --mcp-config" in out.stdout


def test_the_windows_script_is_read_in_utf8_keeps_its_percent_signs_and_returns_from_npm(
        tmp_path, monkeypatch):
    from smappy.assistant import launch
    monkeypatch.setattr(sys, "platform", "win32")
    folder = tmp_path / "Jürgen 100%"
    ran = []
    launch.start("codex", folder, tmp_path / "settings", 4242, run=ran.append)
    written = (tmp_path / "settings" / "start-codex.cmd").read_bytes()
    # cmd reads a batch file in the OEM code page until told otherwise
    lines = written.decode("utf-8").split("\r\n")
    assert lines[:3] == ["@echo off", "chcp 65001 >nul",
                         'cd /d "' + str(folder)[:-1] + '%%"']
    assert b"\r\r\n" not in written
    # npm is npm.cmd: without call the script ends inside it
    assert "call npm install -g @openai/codex" in lines
    assert lines[-2].startswith("codex -c ")
    assert ran[0][:2] == ["cmd", "/c"]


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="cmd.exe runs it")
def test_cmd_runs_the_start_script_in_a_folder_named_with_an_umlaut_and_a_percent_sign(tmp_path):
    from smappy.assistant import launch
    folder = tmp_path / "Jürgen 100%"
    folder.mkdir()
    # a command of the user's own, `cd`, prints where the script went
    launch.start("command", folder, tmp_path / "settings", 4242, own="cd",
                 run=lambda command: None)
    out = subprocess.run(["cmd", "/c", str(tmp_path / "settings" / "start-command.cmd")],
                         capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert str(folder) in out.stdout.decode("utf-8").splitlines()


def test_for_the_claude_app_the_server_goes_into_the_folders_mcp_json(tmp_path):
    from smappy.assistant import launch
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    ran = []
    if sys.platform.startswith("linux"):
        with pytest.raises(RuntimeError):
            launch.start("claude-app", tmp_path, tmp_path / "s", 4242, run=ran.append)
    else:
        line = launch.start("claude-app", tmp_path, tmp_path / "s", 4242, run=ran.append)
        assert "Code session" in line
    servers = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]
    assert set(servers) == {"other", "smappy-gui"}
    # no process number: the file outlives this SMAPpy
    assert servers["smappy-gui"]["args"][-1] == "--attach"


# --------------------------------------------------------- smappy-mcp --attach

def _attach(*args):
    return subprocess.Popen([sys.executable, "-m", "smappy.assistant.server", "--attach",
                             *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=dict(os.environ))


@pytest.mark.slow
def test_smappy_mcp_attach_forwards_a_client_to_the_running_window(server, tmp_path,
                                                                     monkeypatch):
    # the forwarder reads the registry from the configuration folder
    monkeypatch.setattr(server, "registry", None)
    server.start()
    process = _attach()
    lines = []
    reader = threading.Thread(target=lambda: lines.extend(process.stdout), daemon=True)
    reader.start()
    for n, (method, params) in enumerate((
            ("initialize", {"protocolVersion": "2025-06-18", "clientInfo": {"name": "fwd"}}),
            ("tools/call", {"name": "session_state", "arguments": {}})), 1):
        process.stdin.write((json.dumps({"jsonrpc": "2.0", "id": n, "method": method,
                                         "params": params}) + "\n").encode())
    process.stdin.flush()
    assert wait_for(lambda: len(lines) >= 2, 60), process.stderr.read1(4000)
    hello, state = (json.loads(line) for line in lines[:2])
    assert hello["result"]["serverInfo"]["name"] == "smappy"
    assert json.loads(state["result"]["content"][0]["text"])["mode"] == "suggest"
    assert server.client_names() == ["fwd"]
    process.stdin.close()               # the client went: the forwarder ends
    assert wait_for(lambda: process.poll() is not None, 20)
    assert process.returncode == 0
    assert wait_for(lambda: not server.clients)
    server.stop()                       # while the registry is the configuration's


@pytest.mark.slow
def test_the_forwarder_outlives_smappy_and_attaches_again_when_it_is_back(
        server, monkeypatch):
    """Quitting SMAPpy left the client's forwarder dead -- and Python aborting
    as it went.  Now the client is told SMAPpy is gone, and goes on once a
    SMAPpy listens again."""
    from smappy.gui.assistant_server import AssistantServer
    monkeypatch.setattr(server, "registry", None)
    server.start()
    process = _attach()
    lines = []
    threading.Thread(target=lambda: lines.extend(process.stdout), daemon=True).start()

    def ask(n, method, params):
        process.stdin.write((json.dumps({"jsonrpc": "2.0", "id": n, "method": method,
                                         "params": params}) + "\n").encode())
        process.stdin.flush()
        assert wait_for(lambda: len(lines) >= n, 60), process.stderr.read1(4000)
        return json.loads(lines[n - 1])

    ask(1, "initialize", {"protocolVersion": "2025-06-18", "clientInfo": {"name": "fwd"}})
    call = {"name": "session_state", "arguments": {}}
    assert not ask(2, "tools/call", call)["result"]["isError"]
    server.stop()                       # SMAPpy quits
    gone = ask(3, "tools/call", call)["result"]
    assert gone["isError"] and "not running" in gone["content"][0]["text"]
    assert process.poll() is None       # still there for the client
    again = AssistantServer(server.session, parent=None)
    again.start()                       # SMAPpy is back
    assert wait_for(lambda: again.client_names() == ["fwd"], 20)
    back = ask(4, "tools/call", call)["result"]
    assert not back["isError"] and len(lines) == 4   # the replayed hello not seen
    process.stdin.close()
    assert wait_for(lambda: process.poll() is not None, 20)
    assert process.returncode == 0      # not an abort
    again.stop()


@pytest.mark.slow
def test_attach_with_no_window_listening_says_how_to_allow_it(tmp_path):
    process = _attach("no such session")
    _, err = process.communicate(timeout=60)
    assert process.returncode == 1
    assert b"no SMAPpy window" in err



def test_open_window_opens_the_roi_manager_on_an_roi_and_the_3d_view(app, tmp_path):
    """"Open the ROI manager for me": a window that is not a plugin, which
    open_plugin had to refuse (noted as a missing tool, 2026-10-26)."""
    from smappy.assistant.tools import Assistant
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.session import Session
    session = Session()
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    a = control._assistant().server.assistant
    a.knowledge = tmp_path / "kb"
    simulate(control._assistant().server)
    assert "ROI manager" in a.call("session_state")["windows"]
    project = session.rois
    roi = project.add_roi(next(iter(project.sources)), [5000.0, 6000.0])
    assert a.call("open_window", {"name": "roi manager", "roi": roi.number}) == {
        "opened": "ROI manager", "roi": roi.number}
    manager = control.roi_tab.manager
    assert manager.isVisible() and manager.active == roi.id
    a.call("open_window", {"name": "3D view"})
    assert control.view3d_window is not None and control.view3d_window.isVisible()
    with pytest.raises(Exception, match="no 'mitochondria'.*ROI manager"):
        a.call("open_window", {"name": "mitochondria"})
    with pytest.raises(Exception, match="no ROI 99"):
        a.call("open_window", {"name": "ROI manager", "roi": 99})
    assert not a.proposals
    assert "open_window" not in Assistant(output=tmp_path, knowledge_folder=tmp_path).tools()
    for window in (manager, control.view3d_window, render, control):
        window.close()


# ------------------------------------------------- act mode through the panels

def test_in_act_mode_a_run_goes_through_its_panel_and_is_answered_when_it_ends(controls):
    """Act mode used to run plugins straight on the session: nothing in the
    GUI showed which plugin or settings, no progress, no Stop, the window
    held.  Now the run is the user's own click, watched."""
    server = controls.server
    simulate(server)
    server.assistant.mode = "act"
    server.start()
    client = Client(server.address)
    client.send("initialize", {"clientInfo": {"name": "claude-code"}})
    client.answer()
    # two calls at once: the second waits for the first run's end
    client.send("tools/call", {"name": "run_plugin", "arguments": {
        "path": "Analysis/Drift/RCC", "apply": True, "settings": {"n_timepoints": 6}}})
    client.send("tools/call", {"name": "session_state", "arguments": {}})
    assert wait_for(lambda: "Analysis/Drift/RCC" in controls._windows)
    window = controls._windows["Analysis/Drift/RCC"]
    assert window.isVisible() and window.panel.form.value().n_timepoints == 6
    ran = client.answer(120)
    assert ran["id"] == 2
    answer = json.loads(ran["result"]["content"][0]["text"])
    assert answer["applied"] and answer["changed_table"] and server.session.can_undo
    state = client.answer()
    assert state["id"] == 3 and "RCC" in " ".join(
        json.loads(state["result"]["content"][0]["text"])["undo"])
    server.stop()


def test_a_pinned_plugin_runs_in_its_tab_with_its_section_open(app, tmp_path):
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.session import Session
    session = Session()
    render = RenderWindow(session)
    control = ControlWindow(session, render)
    controls = control._assistant()
    a = controls.server.assistant
    a.knowledge = tmp_path / "kb"
    a.mode = "act"
    simulate(controls.server)
    path = "Analysis/Measure/Localization Statistics"
    pinned = control._pinned(path)
    assert pinned is not None, "pinned in the shipped GUI's Analysis tab"
    index, tab, instance = pinned
    control.tabs.setCurrentIndex(0)
    outcome = []
    later = a.call("run_plugin", {"path": path, "apply": True})
    later.then(lambda value, why: outcome.append((value, why)))
    assert control.tabs.currentIndex() == index               # its tab in front
    assert tab.open_instance() == instance.id                   # its section open
    assert wait_for(lambda: outcome, 120)
    value, why = outcome[0]
    assert why is None and value["applied"] and value["figures"]
    render.close()
    control.close()


def test_with_the_panels_switched_off_a_run_goes_straight_to_the_session(controls):
    server = controls.server
    simulate(server)
    server.assistant.mode = "act"
    controls.show_runs(False)
    answer = server.assistant.call("run_plugin", {"path": "Analysis/Drift/RCC",
                                                  "apply": True})
    assert isinstance(answer, dict) and answer["applied"] and not controls._windows
