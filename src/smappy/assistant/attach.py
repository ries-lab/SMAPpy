"""Reaching the running GUI: where it listens, and ``smappy-mcp --attach``.

An MCP client launches a command and talks to it over stdio; the GUI is
already running and listens on a local socket (`gui/assistant_server.py`,
switched on by *Assistant > Allow connections*).  ``smappy-mcp --attach``
joins the two: it finds the GUI and copies the client's lines to the socket
and the GUI's answers back, nothing more -- the tools, the session and the
mode are the GUI's.

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


def forward(name: str = "", stdin=None, stdout=None, where: Optional[Path] = None) -> int:
    """Copy lines between stdio and the GUI's socket until either ends."""
    from PySide6.QtCore import QCoreApplication, QObject, Signal, Slot

    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    app = QCoreApplication.instance() or QCoreApplication([sys.argv[0]])
    entries = listening(where)
    candidates = choose(entries, name)
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

    class Pipe(QObject):
        """Lines read on the stdin thread, written to the socket on this one:
        a QLocalSocket is used only from the thread that made it."""
        line = Signal(bytes)
        ended = Signal()

        @Slot(bytes)
        def write(self, data: bytes) -> None:
            socket.write(data)
            socket.flush()

        @Slot()
        def finish(self) -> None:
            # the client went: an orderly end, not the window's (`gone`)
            socket.disconnected.disconnect(gone)
            socket.waitForBytesWritten(CONNECT_MS)
            app.exit(0)

    pipe = Pipe()
    pipe.line.connect(pipe.write)          # queued: emitted on the reader thread
    pipe.ended.connect(pipe.finish)
    status = {"code": 0}

    def answer() -> None:
        try:
            data = bytes(socket.readAll().data())
        except RuntimeError:            # the socket is already being deleted
            return
        stdout.write(data)
        stdout.flush()

    def gone() -> None:
        answer()
        print("smappy-mcp: the SMAPpy window closed the connection", file=sys.stderr)
        status["code"] = 1
        app.exit(1)

    socket.readyRead.connect(answer)
    socket.disconnected.connect(gone)

    def read() -> None:
        for line in stdin:
            pipe.line.emit(bytes(line))
        pipe.ended.emit()

    threading.Thread(target=read, daemon=True, name="stdin").start()
    code = app.exec()
    return code or status["code"]
