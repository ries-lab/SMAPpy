"""Reaching the running GUI: where it listens, and ``smappy-mcp --attach``.

An MCP client launches a command and talks to it over stdio; the GUI is
already running and listens on a local socket (`gui/assistant_server.py`,
switched on by *Assistant > Allow connections*).  ``smappy-mcp --attach``
joins the two: it finds the GUI and copies the client's lines to the socket
and the GUI's answers back -- the tools, the session and the mode are the
GUI's.  It outlives the GUI: SMAPpy quit and started again finds the same
client, whose conversation knows what was done, attached again by itself
(`_Forwarder`).

**Where a GUI listens** is a small JSON file per GUI in
``<config dir>/assistant/gui/``, written while connections are allowed and
removed when they are not: its process, its socket's full name and its
session's name, so that ``--attach NAME`` picks one of several windows.
A file left by a GUI that crashed names a socket nobody listens on; the
forwarder skips it and removes it.

**A local socket, not a TCP port**: Qt's `QLocalServer` is a Unix socket on
macOS and Linux and a named pipe on Windows, readable only by the user who
started the GUI.  A port on 127.0.0.1 would be open to every program on the
computer -- a web page can send a POST to one -- and would have needed a
secret to keep them out.

The forwarder uses Qt's `QLocalSocket` too, on every platform: Python's own
sockets reach a Unix socket but not a Windows pipe in both directions at
once, and one code path is the one that gets tested.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CONNECT_MS = 3000


def folder() -> Path:
    from .. import config
    return config.config_dir() / "assistant" / "gui"


def server_name() -> str:
    """This process's socket name; Qt makes it a path in the temporary
    folder (Unix) or a pipe (Windows), and the full name is what is kept."""
    return f"smappy-assistant-{os.getpid()}"


def announce(full_name: str, session_name: str = "", where: Optional[Path] = None) -> Path:
    """Say that this process listens at ``full_name``."""
    target = (where or folder()) / f"{os.getpid()}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    entry = {"pid": os.getpid(), "server": full_name, "session": session_name,
             "since": time.strftime("%Y-%m-%d %H:%M:%S")}
    target.write_text(json.dumps(entry, indent=1), encoding="utf-8")
    return target


def withdraw(where: Optional[Path] = None) -> None:
    (where or folder()).joinpath(f"{os.getpid()}.json").unlink(missing_ok=True)


def listening(where: Optional[Path] = None) -> List[Dict[str, Any]]:
    """The GUIs that said they listen, newest first: each entry with its
    ``file``.  Not checked: a connection is the only test that tells."""
    out = []
    for file in (where or folder()).glob("*.json"):
        try:
            entry = json.loads(file.read_text(encoding="utf-8"))
            entry["file"] = file
            entry["mtime"] = file.stat().st_mtime
        except (OSError, ValueError):
            continue
        if isinstance(entry, dict) and entry.get("server"):
            out.append(entry)
    return sorted(out, key=lambda e: e["mtime"], reverse=True)


def _describe(entry: Dict[str, Any]) -> str:
    name = entry.get("session") or "unnamed session"
    return f"{name!r} (process {entry.get('pid')})"


def choose(entries: List[Dict[str, Any]], name: str = "") -> List[Dict[str, Any]]:
    """The entries ``name`` may mean, best first: a session's name (exactly,
    then as a part, ignoring case) or a process number; all of them when no
    name is given."""
    if not name:
        return entries
    exact = [e for e in entries if (e.get("session") or "") == name
             or str(e.get("pid")) == name]
    if exact:
        return exact
    low = name.lower()
    return [e for e in entries if low in (e.get("session") or "").lower()]


def connect(entries: List[Dict[str, Any]], log=None):
    """A connected `QLocalSocket` to the first entry that answers, and that
    entry; (None, None) if none does.  An entry nobody answers is a GUI that
    has gone, and its file is removed."""
    from PySide6.QtNetwork import QLocalSocket
    log = log or (lambda text: print(text, file=sys.stderr))
    for entry in entries:
        socket = QLocalSocket()
        socket.connectToServer(entry["server"])
        if socket.waitForConnected(CONNECT_MS):
            return socket, entry
        log(f"smappy-mcp: {_describe(entry)} does not answer; forgotten")
        Path(entry["file"]).unlink(missing_ok=True)
    return None, None


NOT_RUNNING = ("SMAPpy is not running, or does not allow connections.  Ask the "
               "user to start SMAPpy and choose Assistant > Start assistant (or "
               "Allow connections); this reconnects by itself, so then call again.")
RETRY_MS = 1000


def forward(name: str = "", stdin=None, stdout=None, where: Optional[Path] = None) -> int:
    """Copy lines between stdio and the GUI's socket until the client ends.

    When the GUI goes -- quit, or connections switched off -- the client is
    not dropped with it: it is the user's conversation, which has learnt
    what the session is about.  So this stays, answers the client's calls
    with why they cannot be made, and looks for a GUI every `RETRY_MS`;
    when one listens again it connects and replays the client's
    ``initialize``, whose answer the client already had, so the client goes
    on as if nothing happened.
    """
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance() or QCoreApplication([sys.argv[0]])
    entries = listening(where)
    candidates = choose(entries, name)
    if not candidates and name.isdigit() and entries:
        # a process number from a file written for a SMAPpy that has since
        # quit: the newest that listens is the one meant now
        candidates = entries
    if not candidates:
        if entries and name:
            print(f"smappy-mcp: no SMAPpy window called {name!r} allows connections; "
                  f"these do: {', '.join(_describe(e) for e in entries)}", file=sys.stderr)
        else:
            print("smappy-mcp: no SMAPpy window allows connections.  In SMAPpy, "
                  "choose Assistant > Allow connections, then reconnect.",
                  file=sys.stderr)
        return 1
    socket, entry = connect(candidates)
    if socket is None:
        print("smappy-mcp: no SMAPpy window answered.  In SMAPpy, choose "
              "Assistant > Allow connections, then reconnect.", file=sys.stderr)
        return 1
    others = [e for e in candidates if e is not entry]
    print(f"smappy-mcp: attached to {_describe(entry)}"
          + (f"; also listening: {', '.join(_describe(e) for e in others)} "
             "(--attach NAME chooses)" if others else ""), file=sys.stderr)
    forwarder = _Forwarder(app, stdin or sys.stdin.buffer, stdout or sys.stdout.buffer,
                           where, entry)
    forwarder.attach(socket)
    return forwarder.run()


class _Forwarder:
    """The state between the client and whichever GUI it is attached to."""

    def __init__(self, app, stdin, stdout, where, entry):
        from PySide6.QtCore import QObject, QTimer, Signal, Slot

        self.app, self.stdin, self.stdout, self.where = app, stdin, stdout, where
        self.session = entry.get("session") or ""
        self.socket = None
        self.buffer = b""
        self.hello: Optional[Dict[str, Any]] = None    # the client's initialize
        self.waiting: Dict[Any, str] = {}               # id -> method, sent to the GUI
        self.swallow: set = set()                       # ids of replayed initializes
        self.replays = 0

        forwarder = self

        class Pipe(QObject):
            """Lines read on the stdin thread, handled on this one: a
            QLocalSocket is used only from the thread that made it."""
            line = Signal(bytes)
            ended = Signal()

            @Slot(bytes)
            def take(self, data: bytes) -> None:
                forwarder._from_client(data)

            @Slot()
            def finish(self) -> None:
                if forwarder.socket is not None:
                    forwarder.socket.disconnected.disconnect()
                    forwarder.socket.waitForBytesWritten(CONNECT_MS)
                forwarder.app.exit(0)

        self.pipe = Pipe()
        self.pipe.line.connect(self.pipe.take)       # queued: emitted on the reader thread
        self.pipe.ended.connect(self.pipe.finish)
        self.retry = QTimer()
        self.retry.setInterval(RETRY_MS)
        self.retry.timeout.connect(self._look_again)

    def run(self) -> int:
        def read() -> None:
            for line in self.stdin:
                self.pipe.line.emit(bytes(line))
            self.pipe.ended.emit()

        threading.Thread(target=read, daemon=True, name="stdin").start()
        return self.app.exec()

    # ------------------------------------------------------------ the client
    def _from_client(self, data: bytes) -> None:
        try:
            message = json.loads(data)
        except ValueError:
            message = None
        if isinstance(message, dict):
            if message.get("method") == "initialize" and self.hello is None:
                self.hello = message
            if "id" in message and "method" in message:
                self.waiting[message["id"]] = message["method"]
        if self.socket is not None:
            self.socket.write(data)
            self.socket.flush()
        elif isinstance(message, dict) and "id" in message and "method" in message:
            self.waiting.pop(message["id"], None)
            self._unanswerable(message["id"], message["method"])

    def _to_client(self, message: Dict[str, Any]) -> None:
        self.stdout.write((json.dumps(message) + "\n").encode("utf-8"))
        self.stdout.flush()

    def _unanswerable(self, ident, method: str) -> None:
        """What the client is told while no GUI is there: a tool's error,
        which the model reads, for a tool call; a protocol error otherwise."""
        if method == "tools/call":
            self._to_client({"jsonrpc": "2.0", "id": ident, "result": {
                "content": [{"type": "text", "text": NOT_RUNNING}], "isError": True}})
        elif method == "ping":
            self._to_client({"jsonrpc": "2.0", "id": ident, "result": {}})
        else:
            self._to_client({"jsonrpc": "2.0", "id": ident,
                             "error": {"code": -32000, "message": NOT_RUNNING}})

    # ---------------------------------------------------------------- the GUI
    def attach(self, socket) -> None:
        self.socket = socket
        self.buffer = b""
        socket.readyRead.connect(self._from_gui)
        socket.disconnected.connect(self._gone)

    def _from_gui(self) -> None:
        try:
            self.buffer += bytes(self.socket.readAll().data())
        except (RuntimeError, AttributeError):    # being deleted
            return
        *lines, self.buffer = self.buffer.split(b"\n")
        for line in lines:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except ValueError:
                message = None
            if isinstance(message, dict) and "id" in message:
                if message["id"] in self.swallow:
                    self.swallow.discard(message["id"])
                    continue                  # a replayed initialize's answer
                self.waiting.pop(message["id"], None)
            self.stdout.write(line + b"\n")
        self.stdout.flush()

    def _gone(self) -> None:
        self._from_gui()
        socket, self.socket = self.socket, None
        if socket is not None:
            socket.deleteLater()
        for ident, method in list(self.waiting.items()):
            self._unanswerable(ident, method)    # asked, and now never answered
        self.waiting.clear()
        print("smappy-mcp: the SMAPpy window went; waiting for it to allow "
              "connections again", file=sys.stderr)
        self.retry.start()

    def _look_again(self) -> None:
        entries = listening(self.where)
        if not entries:
            return
        # the same session first, if it is there again
        entries.sort(key=lambda e: (e.get("session") or "") != self.session)
        socket, entry = connect(entries, log=lambda text: None)
        if socket is None:
            return
        self.retry.stop()
        self.session = entry.get("session") or ""
        self.attach(socket)
        if self.hello is not None:
            self.replays += 1
            replay = dict(self.hello, id=f"smappy-reattach-{self.replays}")
            self.swallow.add(replay["id"])
            socket.write((json.dumps(replay) + "\n").encode("utf-8"))
            socket.write(b'{"jsonrpc": "2.0", "method": "notifications/initialized"}\n')
            socket.flush()
        print(f"smappy-mcp: attached again, to {_describe(entry)}", file=sys.stderr)
