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
    a = server.assistant
    mode, a.mode = a.mode, "act"
    a.call("run_plugin", {"path": "File/Simulate/Simulate", "settings": SIMULATION,
                          "apply": True})
    a.mode = mode


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
    assert server.client_names() == ["test 1"]
    client.send("tools/list")
    names = {t["name"] for t in client.answer()["result"]["tools"]}
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

def test_the_pane_opens_a_proposed_run_in_the_plugins_panel_filled_in(server, app):
    from smappy.gui.assistant_pane import AssistantPane
    simulate(server)
    pane = AssistantPane(server.session, server)
    a = server.assistant
    proposed = a.call("run_plugin", {"path": "Analysis/Drift/RCC", "apply": True,
                                     "settings": {"n_timepoints": 7},
                                     "reason": "the structure smears along x"})
    ident = proposed["proposed"]["id"]
    server.changed.emit()
    assert not pane.empty.isVisibleTo(pane)
    texts = [w.text() for w in pane.rows.findChildren(type(pane.state))]
    assert any("Run Analysis/Drift/RCC: n_timepoints = 7" in t for t in texts)
    assert "the structure smears along x" in texts
    window = pane.open_proposal(ident)
    assert window.panel.form.value().n_timepoints == 7
    assert a.proposal(ident).status == "opened"
    assert a.call("list_proposals")[0]["status"] == "opened"
    # nothing ran: the user runs it from the panel
    assert not server.session.can_undo
    pane.dismiss(ident)
    assert a.proposal(ident).status == "dismissed"
    assert pane.empty.isVisibleTo(pane)
    window.close()
    pane.close()


def test_the_pane_does_a_proposed_undo_and_follows_the_switch(server, app):
    from smappy.gui.assistant_pane import AssistantPane
    simulate(server)
    a = server.assistant
    a.mode = "act"
    a.call("run_plugin", {"path": "Analysis/Drift/RCC", "apply": True,
                          "settings": {"n_timepoints": 5}})
    pane = AssistantPane(server.session, server)
    pane.suggest.setChecked(True)
    assert a.mode == "suggest"
    ident = a.call("undo")["proposed"]["id"]
    assert server.session.can_undo
    pane.do_proposal(ident)
    assert not server.session.can_undo and a.proposal(ident).status == "done"
    assert a.mode == "suggest"
    pane.act.setChecked(True)
    assert a.mode == "act"
    pane.close()


def test_the_output_folder_follows_the_first_file_until_the_user_sets_one(server, tmp_path):
    simulate(server)
    a = server.assistant
    a.mode = "act"
    server.set_output(str(tmp_path / "out"))
    assert a.output == tmp_path / "out" and server.output_pinned
    saved = Path(a.call("save", {"path": "x.h5"})["saved"])
    server.set_output(None)
    server.session.load(saved)
    assert not server.output_pinned and a.output == saved.parent


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
def test_attach_with_no_window_listening_says_how_to_allow_it(tmp_path):
    process = _attach("no such session")
    _, err = process.communicate(timeout=60)
    assert process.returncode == 1
    assert b"no SMAPpy window" in err
