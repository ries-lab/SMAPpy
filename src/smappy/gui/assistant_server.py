"""The assistant's tools served from the running GUI, over a local socket.

*Assistant > Allow connections* starts it; ``smappy-mcp --attach`` is what an
MCP client launches to reach it (`assistant.attach`, which also says why a
local socket and not a port).  The tools are the headless server's
(`assistant.tools.Assistant`) over the GUI's own session, so what the
assistant does the user sees in the windows as it happens.

**Every call runs on the GUI thread, one at a time, from a queue.**  The
session, the layers and every listener of `Session.on_change` are the GUI
thread's; a call made from a socket thread would change them under the
render.  So the socket's lines are queued as they arrive and a timer on the
GUI thread takes them one per tick: never inside the socket's own signal,
where a plugin that lets events through would read the next line half-way
through this one.  The timer's ``timeout`` is a connected signal, not a
``QTimer.singleShot`` callable, and the tick catches everything: an
exception left pending inside Qt's event loop crashes the next Python
override it calls (`CLAUDE.md`).  A tool's own failure is the protocol's
answer already (`protocol.Server.tools_call`); what is caught here is the
rest.

**The window is busy while a call runs**: a long applied run -- a fit, COMET
-- holds the GUI as it would hold a script.  A tool call is announced in the
pane one tick before it runs, so the pane says what is running while it
does.  A worker thread for the long ones is a later step
(`docs/ai-integration.md`).

Each connection has its own `protocol.Server` -- its own ``initialize`` --
and all share the one `Assistant`: one session, one switch, one list of
proposals, whoever asked.
"""
from __future__ import annotations

import json
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QRectF, QTimer, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

# what the client is told on connecting, after the tools' own instructions
GUI_NOTE = """
You are connected to the running SMAPpy GUI: the user sees this session in
its windows as you work.  open_plugin opens a plugin's window filled in, for
the user to run; open_window one of SMAPpy's other windows (the ROI manager,
the 3D view, ...); filters, grouping and show() change what is shown; all are
made at once, in either mode.  In suggest mode a change to the data appears
in the GUI's Assistant pane as a proposal, with a button that opens the
plugin's window filled in.  Each call holds the GUI while it runs: try long
runs on a part of the data first.

The user is usually a biologist at the microscope, not a programmer: answer
briefly, in plain words, and say what you did or found -- not how your tools
work or what they cannot do.
"""


class RenderViewer:
    """The 2D view as `Assistant.show` moves it."""

    def __init__(self, view):
        self.view = view                    # a `render_view.RenderView`

    def frame(self, xrange, yrange) -> None:
        (x0, x1), (y0, y1) = xrange, yrange
        self.view.view.setRange(QRectF(x0, y0, x1 - x0, y1 - y0), padding=0)
        self.view.schedule()

    def reset(self) -> None:
        self.view.reset()

    def current(self):
        rect = self.view.view.viewRect()
        return ((rect.left(), rect.right()), (rect.top(), rect.bottom()),
                self.view.pixel_size())


class _Client:
    def __init__(self, socket: QLocalSocket, server):
        self.socket = socket
        self.server = server                # its `protocol.Server`
        self.buffer = b""

    @property
    def name(self) -> str:
        info = self.server.client
        return " ".join(str(info[k]) for k in ("name", "version") if info.get(k)) \
            or "a client"


class AssistantServer(QObject):
    """The local socket, its connections and the queue of their lines."""

    changed = Signal()          # listening, a client, a call, the proposals

    def __init__(self, session, viewer=None, parent=None, registry: Optional[Path] = None):
        super().__init__(parent)
        from ..assistant.tools import Assistant
        self.session = session
        self.registry = registry            # where `attach.announce` writes; None: the config's
        self.assistant = Assistant(session, mode="suggest", viewer=viewer)
        # the output folder follows the first open file until the user sets one
        self.output_pinned = False
        self._follow_output()
        session.on_change(self._on_session)
        self.server: Optional[QLocalServer] = None
        self.clients: Dict[int, _Client] = {}
        self.queue: Deque[Tuple[int, str]] = deque()
        self.running: str = ""              # the tool running now, for the pane
        self.calls: Deque[Tuple[str, str, bool]] = deque(maxlen=20)  # time, tool, ok
        self.error: str = ""                # why listening failed
        self._announced = None              # the next call, shown before it runs
        # the panels opened on a proposal, by the panel: the proposal its next
        # run carries out (`watch`)
        self._watched: Dict[int, str] = {}
        self._listening_to: set = set()     # the panels whose `ended` is connected
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._tick)

    # -------------------------------------------------------------- listening
    @property
    def listening(self) -> bool:
        return self.server is not None and self.server.isListening()

    def start(self) -> bool:
        from ..assistant import attach
        if self.listening:
            return True
        name = attach.server_name()
        QLocalServer.removeServer(name)      # a socket file left by a crash
        server = QLocalServer(self)
        server.setSocketOptions(QLocalServer.UserAccessOption)
        if not server.listen(name):
            self.error = server.errorString()
            server.deleteLater()
            self.changed.emit()
            return False
        server.newConnection.connect(self._on_connection)
        self.server = server
        self.error = ""
        self._announce()
        self.changed.emit()
        return True

    def stop(self) -> None:
        from ..assistant import attach
        attach.withdraw(self.registry)
        for client in list(self.clients.values()):
            client.socket.abort()
        self.clients.clear()
        self.queue.clear()
        if self.server is not None:
            self.server.close()
            self.server.deleteLater()
            self.server = None
        self.changed.emit()

    def _announce(self) -> None:
        from ..assistant import attach
        if self.listening:
            attach.announce(self.server.fullServerName(),
                            getattr(self.session, "display_name", "") or "",
                            self.registry)

    @property
    def address(self) -> str:
        return self.server.fullServerName() if self.listening else ""

    # ------------------------------------------------------------ the session
    def _on_session(self, what: str) -> None:
        try:
            if what in ("locs", "files"):
                self._follow_output()
            if what in ("name", "locs"):
                self._announce()            # the name `--attach NAME` matches
        except Exception:
            traceback.print_exc()

    def _follow_output(self) -> None:
        if self.output_pinned:
            return
        # a file on disk: a simulation's "path" is its label ("simulation:
        # Nuclear pores (Nup96)"), whose parent was ".", wherever SMAPpy started
        files = [Path(f.path).expanduser() for f in self.session.files
                 if getattr(f, "path", None)]
        files = [f for f in files if f.is_absolute() and f.is_file()]
        self.assistant.output = files[0].parent if files else Path.home()

    def set_output(self, folder: Optional[str]) -> None:
        """A folder the user chose, or None to follow the first file again."""
        self.output_pinned = folder is not None
        if folder is not None:
            self.assistant.output = Path(folder).expanduser()
        self._follow_output()
        self.changed.emit()

    def set_mode(self, mode: str) -> None:
        self.assistant.mode = mode
        self.changed.emit()

    # ------------------------------------------------------------ connections
    def _on_connection(self) -> None:
        from ..assistant.protocol import Server
        from ..assistant.tools import INSTRUCTIONS
        while self.server is not None and self.server.hasPendingConnections():
            socket = self.server.nextPendingConnection()
            key = id(socket)
            self.clients[key] = _Client(socket, Server(
                self.assistant, name="smappy", instructions=INSTRUCTIONS + GUI_NOTE))
            socket.readyRead.connect(lambda key=key: self._read(key))
            socket.disconnected.connect(lambda key=key: self._gone(key))
        self.changed.emit()

    def _gone(self, key: int) -> None:
        client = self.clients.pop(key, None)
        if client is not None:
            client.socket.deleteLater()
        self.changed.emit()

    def _read(self, key: int) -> None:
        client = self.clients.get(key)
        if client is None:
            return
        client.buffer += bytes(client.socket.readAll().data())
        *lines, client.buffer = client.buffer.split(b"\n")
        for line in lines:
            if line.strip():
                self.queue.append((key, line.decode("utf-8", errors="replace")))
        self._wake()

    def _wake(self) -> None:
        if self.queue and not self._timer.isActive():
            self._timer.start(0)

    # ---------------------------------------------------------------- the queue
    def _tick(self) -> None:
        """One line of the queue, on the GUI thread.  Never raises."""
        try:
            self._handle_next()
        except Exception:
            traceback.print_exc()
        finally:
            try:
                self._wake()
            except Exception:
                traceback.print_exc()

    def _handle_next(self) -> None:
        if not self.queue:
            return
        key, line = self.queue[0]
        tool = _tool_of(line)
        if tool and self._announced != (key, line):
            # shown first, run on the next tick: the pane repaints in between
            self._announced = (key, line)
            self.running = tool
            self.changed.emit()
            self._timer.start(15)
            return
        self.queue.popleft()
        self._announced = None
        client = self.clients.get(key)
        if client is None:
            self.running = ""
            return
        try:
            answer = client.server.handle_line(line)
        finally:
            self.running = ""
        if tool:
            self.calls.append((time.strftime("%H:%M:%S"), tool, _succeeded(answer)))
        if answer is not None and key in self.clients \
                and client.socket.state() == QLocalSocket.ConnectedState:
            client.socket.write((answer + "\n").encode("utf-8"))
            client.socket.flush()
        self.changed.emit()

    # -------------------------------------------------------------- proposals
    def watch(self, panel, ident: str) -> None:
        """The next run of ``panel`` carries out proposal ``ident``: when the
        user runs it, the proposal is done and its result goes to the
        assistant.  A panel opened on another proposal since answers that one."""
        key = id(panel)
        if key not in self._listening_to:
            self._listening_to.add(key)
            panel.ended.connect(lambda kind, result, panel=panel: self._ran(panel, kind, result))
        self._watched[key] = ident

    def _ran(self, panel, kind: str, result) -> None:
        try:
            if kind != "done" or panel.job != "run":
                return                      # a preview, a failure, a stop
            ident = self._watched.pop(id(panel), None)
            if ident is None:
                return
            proposal = self.assistant.proposal(ident)
            if proposal.status in ("opened", "proposed"):
                self.assistant.ran(ident, result, getattr(panel.plugin, "path", ""))
            self.changed.emit()
        except Exception:
            traceback.print_exc()

    # ------------------------------------------------------------ for the pane
    def client_names(self) -> List[str]:
        return [c.name for c in self.clients.values()]


def _succeeded(answer: Optional[str]) -> bool:
    try:
        result = json.loads(answer or "").get("result") or {}
    except (ValueError, AttributeError):
        return False
    return isinstance(result, dict) and not result.get("isError", True)


def _tool_of(line: str) -> str:
    """The tool a line calls, or '' for any other message."""
    try:
        message = json.loads(line)
    except ValueError:
        return ""
    if isinstance(message, dict) and message.get("method") == "tools/call":
        params = message.get("params") or {}
        return str(params.get("name") or "") if isinstance(params, dict) else ""
    return ""
