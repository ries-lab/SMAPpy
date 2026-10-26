"""The assistant in the GUI: a button, the proposals, the settings, the start.

The assistant runs in the user's own client (`docs/ai-integration.md`); what
is here is SMAPpy's side of it, shaped by what is looked at how often:

* **Every moment**: whether an assistant is connected, whether it may act
  or only suggest, and what it proposed.  That is one button in the render
  window's toolbar (`AssistantButton`, where one's eyes are), and under it
  the list of proposals (`ProposalList`), dropped down inside the window
  over the picture's corner (`ProposalOverlay`).  It opens by itself when
  something arrives, stays until it is collapsed -- the button still says
  what waits -- and collapses by itself when the last proposal is dealt
  with.  Inside the window rather than a window of its own: it moves with
  the picture, never takes the focus from the terminal one types in, and
  cannot get lost behind it.  The mode is in sight there: it decides what
  the assistant may do to the data.
* **Once**: which client to start and where, the folder it may write into,
  the mode to start in, whether to listen from the start.  That is the
  settings window (`AssistantSettings`), behind *settings...*, kept in the
  configuration (`preferences`).

A proposal to run a plugin or a chain is never run from here: *Open* shows
the plugin's own window filled in (`PluginPanel.take_settings`), the user
runs it with its Run, and that run carries the proposal out
(`AssistantServer.watch`): it is done, and the assistant can read its result.
The rest -- open a file, undo, save -- is one click, *Do it*.
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Callable, Dict, Optional

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                               QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton,
                               QRadioButton, QScrollArea, QToolButton,
                               QVBoxLayout, QWidget)

from .widgets import note_style

OPENS = ("run_plugin", "run_chain")      # proposals shown in a panel, not done here
NEW_STYLE = "font-weight: bold; color: #c06000;"

DEFAULTS = {"client": "", "command": "", "folder": "", "output": "",
            "mode": "suggest", "allow_at_start": False, "pre_allow": True}


def preferences() -> Dict:
    from .. import config
    saved = config.get("assistant") or {}
    return {**DEFAULTS, **(saved if isinstance(saved, dict) else {})}


def save_preferences(values: Dict) -> None:
    from .. import config
    config.set("assistant", {**preferences(), **values})


def describe(proposal) -> str:
    """A proposal in a line, as the user reads it."""
    a = proposal.arguments
    if proposal.tool == "run_plugin":
        values = a.get("settings") or {}
        shown = ", ".join(f"{k} = {v}" for k, v in values.items())
        layer = f" on layer {a['layer']}" if a.get("layer") else ""
        return f"Run {a.get('path')}{layer}" + (f": {shown}" if shown else "")
    if proposal.tool == "run_chain":
        return "Run a chain of plugins"
    if proposal.tool == "load":
        return ("Add " if a.get("append") else "Open ") + str(a.get("path"))
    if proposal.tool == "undo":
        n = a.get("steps", 1)
        return f"Undo {n} step{'s' if n != 1 else ''}"
    if proposal.tool == "save":
        return f"Save the localizations as {a.get('path')}"
    if proposal.tool == "save_chain":
        return f"Keep the chain as {a.get('name')!r}"
    if proposal.tool == "set_filter":
        return (f"Remove the filter on {a.get('column')}" if a.get("remove") else
                f"Filter {a.get('column')} to {a.get('lo')} .. {a.get('hi')}")
    if proposal.tool == "set_grouped":
        return f"Show layer {a.get('layer', 0)} {'grouped' if a.get('on') else 'ungrouped'}"
    return f"{proposal.tool}({a})"


class AssistantControls(QObject):
    """What the button, the list and the settings act through: the server,
    the session's windows, and starting a client.  One per control window."""

    changed = Signal()          # the server's, and a proposal opened or seen

    def __init__(self, session, server=None, parent=None,
                 open_plugin: Optional[Callable] = None,
                 load: Optional[Callable] = None,
                 beside: Optional[QWidget] = None):
        super().__init__(parent)
        prefs = preferences()
        if server is None:
            from .assistant_server import AssistantServer
            server = AssistantServer(session, parent=self)
        server.assistant.mode = prefs["mode"]
        if prefs["output"]:
            server.set_output(prefs["output"])
        self.session = session
        self.server = server
        # the main window's own plugin windows and loader when there is one,
        # so a plugin opened here is the one the Plugins menu opens
        self.open_plugin = open_plugin or self._own_window
        self.load = load
        self.beside = beside                # the render window, for placing
        # the assistant's open_plugin: a plugin's window, filled in
        server.assistant.panels = self.show_plugin
        self._windows: Dict[str, QWidget] = {}
        self.settings_window: Optional[QWidget] = None
        # shows the list of proposals: the toolbar button's overlay
        # (`AssistantButton.attach`); nothing without one
        self.reveal: Callable[[], None] = lambda: None
        self.message = ""                   # the last thing said to the user
        # after Start found the old terminal: did it come back?
        self._reconnect_check = QTimer(self)
        self._reconnect_check.setSingleShot(True)
        self._reconnect_check.setInterval(6000)
        self._reconnect_check.timeout.connect(self._check_reconnected)
        server.changed.connect(self._server_changed)
        self._about_connecting = False      # `message` says a client is awaited

    # ---------------------------------------------------------------- state
    def live(self):
        return [p for p in self.server.assistant.proposals
                if p.status in ("proposed", "opened")]

    def client_label(self) -> str:
        names = self.server.client_names()
        if names:
            return names[0].split(" ")[0]
        from ..assistant.launch import CLIENTS
        chosen = CLIENTS.get(preferences()["client"])
        return "" if chosen is None else chosen.label.split(" ")[0]

    def _server_changed(self) -> None:
        try:
            if self._about_connecting and self.server.clients:
                # it came: what was said while waiting for it is not true now
                self._about_connecting = False
                self.message = ""
        except Exception:
            traceback.print_exc()
        self.changed.emit()

    def say(self, text: str) -> None:
        self.message = text
        self.changed.emit()

    # -------------------------------------------------------------- actions
    def set_mode(self, mode: str) -> None:
        self.server.set_mode(mode)

    def allow(self, on: bool) -> bool:
        if not on:
            self.server.stop()
            return True
        ok = self.server.start()
        if not ok:
            self.say(f"could not listen: {self.server.error}")
        return ok

    def start(self) -> None:
        """*Start assistant*: listen, start the chosen client and place it
        beside the picture.  The first time, one question."""
        from .. import config
        from ..assistant import launch
        prefs = preferences()
        if prefs["client"] not in launch.CLIENTS:
            self.ask_which()                # the first time: one question
            return
        if self.server.listening and self.server.clients:
            # one assistant at a time: two share one list and one switch, and
            # in act mode would change the table without knowing of each other
            raised = launch.raise_terminal()
            self.show_proposals()
            names = ", ".join(self.server.client_names())
            self.say(f"{names} is connected already" + ("" if raised else
                     ": switch to its window.  To start another, stop this one first."))
            return
        if not self.allow(True):
            return
        folder = Path(prefs["folder"]).expanduser() if prefs["folder"] \
            else self.server.assistant.output
        client = launch.CLIENTS[prefs["client"]]
        if prefs["client"] == "manual":
            # SMAPpy listens; the user starts theirs with what the settings copy
            self.show_proposals()
            self.say("Listening.  Start your assistant with the command (or the "
                     "JSON) the settings copy.")
            return
        if prefs["client"] == "claude-app" and not self._may_write_project(folder):
            return
        if client.terminal and launch.raise_terminal():
            # the terminal of a SMAPpy quit and started again: its conversation
            # goes on, and its connection comes back by itself
            # (`attach._Forwarder`) once this listens
            self.show_proposals()
            self.say("Your assistant's terminal is still open: it reconnects in a moment.")
            self._about_connecting = True
            self._reconnect_check.start()
            return
        terminal = self._column()
        try:
            line = launch.start(prefs["client"], folder,
                                config.config_dir() / "assistant", os.getpid(),
                                own=prefs["command"], pre_allow=prefs["pre_allow"],
                                bounds=terminal)
        except Exception as error:
            self.say(f"could not start {client.label}: {error}")
            self.show_settings()
            return
        self.show_proposals()
        self.say(line)

    def _check_reconnected(self) -> None:
        try:
            if not self.server.clients:
                self._about_connecting = True     # cleared if it comes late
                self.say("The open terminal did not reconnect (a Claude started "
                         "before this version?).  Type /mcp in it and reconnect "
                         "smappy-gui, or close it and press Start assistant again.")
        except Exception:
            traceback.print_exc()

    def _may_write_project(self, folder: Path) -> bool:
        from ..assistant.launch import NAME
        target = folder / ".mcp.json"
        current = {}
        try:
            current = json.loads(target.read_text(encoding="utf-8")).get("mcpServers", {})
        except (OSError, ValueError, AttributeError):
            pass
        if NAME in current:
            return True
        answer = QMessageBox.question(
            self.settings_window or self.beside,
            "Assistant",
            f"The Claude app finds the connection to SMAPpy in a file in the "
            f"folder its session works in.  Add '{NAME}' to\n{target}?")
        return answer == QMessageBox.Yes

    def _column(self):
        """Where the terminal goes: a column beside the picture -- right of
        it if the screen has room, else over its right edge -- as
        AppleScript takes it, (left, top, right, bottom)."""
        window = self.beside
        screen = (window.screen() if window is not None else
                  QApplication.primaryScreen()).availableGeometry()
        width = 560
        if window is not None:
            left = window.frameGeometry().right() + 8
            if screen.right() - left < 420:
                left = screen.right() - width
            else:
                width = min(width, screen.right() - left)
        else:
            left = screen.right() - width
        return (left, screen.top() + 30, left + width, screen.bottom())

    def show_proposals(self) -> None:
        """The list of proposals, dropped down under the toolbar's button."""
        self.reveal()

    def ask_which(self) -> QWidget:
        """The first *Start assistant*: which assistant, and Start."""
        self.chooser = FirstStart(self, self.beside)
        self.chooser.show()
        self.chooser.raise_()
        return self.chooser

    def show_settings(self) -> QWidget:
        if self.settings_window is None:
            self.settings_window = AssistantSettings(self.session, self)
        self.settings_window.show()
        self.settings_window.raise_()
        return self.settings_window

    def show_plugin(self, path: str, settings=None) -> Optional[QWidget]:
        """A plugin's window, its settings filled in; nothing run."""
        window = self.open_plugin(path)
        if window is not None and settings is not None:
            window.panel.take_settings(settings)
        return window

    def open_proposal(self, ident: str) -> Optional[QWidget]:
        """The proposal's plugin in its window, filled in; nothing run.  Its
        next run carries the proposal out."""
        from ..assistant.tools import ToolError
        assistant = self.server.assistant
        proposal = assistant.proposal(ident)
        try:
            if proposal.tool == "run_chain":
                from ..chain import chain_class
                cls = chain_class(assistant._chain(proposal.arguments["chain"]))
                window = self._own_window(cls.path or "chain", cls)
                settings = None
            else:
                path = proposal.arguments["path"]
                settings = assistant.settings_for(path, proposal.arguments.get("settings"))
                window = self.open_plugin(path)
        except ToolError as error:
            self.say(f"could not open it: {error}")
            return None
        if window is None:
            return None
        if settings is not None:
            window.panel.take_settings(settings)
        self.server.watch(window.panel, ident)
        proposal.status = "opened"
        self.changed.emit()
        return window

    def do_proposal(self, ident: str) -> None:
        proposal = self.server.assistant.proposal(ident)
        try:
            if proposal.tool == "load" and self.load is not None:
                # the window's own loader: in the background, the view reset
                self.load([proposal.arguments["path"]], bool(proposal.arguments.get("append")))
                proposal.status = "done"
            else:
                self.server.assistant.carry_out(ident)
        except Exception as error:
            self.say(f"{proposal.tool} failed: {error}")
        self.changed.emit()

    def dismiss(self, ident: str) -> None:
        self.server.assistant.proposal(ident).status = "dismissed"
        self.changed.emit()

    def _own_window(self, path: str, cls=None):
        """A plugin window kept here, when there is no main window's."""
        from .. import plugins
        from .plugin_panel import PluginWindow
        window = self._windows.get(path)
        if window is not None and cls is not None:
            window.close()                  # a chain is a new class each time
            window.deleteLater()
            window = None
        if window is None:
            window = PluginWindow(cls or plugins.get(path), self.session)
            self._windows[path] = window
        window.show()
        return window


class ProposalList(QWidget):
    """The everyday part: what is connected and running, the proposals, the
    switch.  In the toolbar's popup and in the proposals' window."""

    def __init__(self, controls: AssistantControls, parent=None):
        super().__init__(parent)
        self.controls = controls
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        self.state = QLabel("")
        self.state.setWordWrap(True)
        self.state.setStyleSheet(note_style())
        column.addWidget(self.state)

        self.rows = QWidget()
        self.rows_layout = QVBoxLayout(self.rows)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(2)
        self.empty = QLabel("No proposals.  In suggest mode what the assistant would "
                            "change waits here.")
        self.empty.setWordWrap(True)
        self.empty.setStyleSheet(note_style())
        self.rows_layout.addWidget(self.empty)
        self.rows_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.rows)
        column.addWidget(scroll, 1)

        footer = QHBoxLayout()
        self.suggest = QRadioButton("suggest")
        self.suggest.setToolTip("read, try on copies and propose; a change waits here. "
                                "Filters, grouping and the view it changes at once")
        self.act = QRadioButton("act")
        self.act.setToolTip("change the session itself: run plugins, open and save "
                            "files under the output folder")
        group = QButtonGroup(self)
        for button in (self.suggest, self.act):
            group.addButton(button)
            footer.addWidget(button)
        self.suggest.toggled.connect(
            lambda on: self.controls.set_mode("suggest" if on else "act"))
        footer.addStretch(1)
        self.start_button = QPushButton("Start assistant")
        self.start_button.setToolTip("start your assistant in a window of its own, "
                                     "connected to this session (Ctrl+Shift+A)")
        self.start_button.clicked.connect(self.controls.start)
        self.stop_button = QToolButton(text="stop")
        self.stop_button.setToolTip("stop listening: the assistant is disconnected")
        self.stop_button.clicked.connect(lambda: self.controls.allow(False))
        settings = QToolButton(text="settings...")
        settings.setToolTip("which assistant to start and where; the folder it may "
                            "write into; how to connect another client")
        settings.clicked.connect(self.controls.show_settings)
        for button in (self.start_button, self.stop_button, settings):
            footer.addWidget(button)
        column.addLayout(footer)
        self._shown: Optional[tuple] = None
        controls.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        try:
            self._refresh()
        except Exception:
            traceback.print_exc()

    def _refresh(self) -> None:
        server = self.controls.server
        mode = server.assistant.mode
        for button, name in ((self.suggest, "suggest"), (self.act, "act")):
            button.blockSignals(True)
            button.setChecked(mode == name)
            button.blockSignals(False)
        self.start_button.setVisible(not server.clients)
        self.stop_button.setVisible(server.listening)
        if server.running:
            state = f"running {server.running}..."
        elif server.listening:
            names = server.client_names()
            state = (f"connected: {', '.join(names)}" if names
                     else "listening, no assistant connected yet")
        else:
            state = "off: no assistant can connect"
        if self.controls.message:
            state += "\n" + self.controls.message
        self.state.setText(state)
        live = self.controls.live()
        key = tuple((p.id, p.status) for p in live)
        if key == self._shown:
            return
        self._shown = key
        while self.rows_layout.count() > 2:  # the note and the stretch stay
            item = self.rows_layout.takeAt(1)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.empty.setVisible(not live)
        for n, proposal in enumerate(reversed(live)):     # the newest first
            self.rows_layout.insertWidget(1 + n, self._row(proposal))

    def _row(self, proposal) -> QWidget:
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        text = QLabel(describe(proposal) + ("  (opened)" if proposal.status == "opened" else ""))
        text.setWordWrap(True)
        if proposal.reason:
            text.setToolTip(proposal.reason)
        line.addWidget(text, 1)
        if proposal.tool in OPENS:
            go = QPushButton("Open")
            go.setToolTip("the plugin's window with these settings: check them, then "
                          "run it there; that run carries the proposal out")
            go.clicked.connect(lambda _=False, p=proposal.id: self.controls.open_proposal(p))
        else:
            go = QPushButton("Do it")
            go.setToolTip("do what is proposed, now")
            go.clicked.connect(lambda _=False, p=proposal.id: self.controls.do_proposal(p))
        dismiss = QToolButton(text="×")
        dismiss.setToolTip("dismiss: off the list; the assistant can see you did")
        dismiss.clicked.connect(lambda _=False, p=proposal.id: self.controls.dismiss(p))
        line.addWidget(go)
        line.addWidget(dismiss)
        return row


class FirstStart(QWidget):
    """The first *Start assistant*: one question, which assistant.  The
    settings hold the rest, and nobody needs them to begin."""

    CHOICES = (("claude", "Claude", "Anthropic's Claude Code"),
               ("codex", "Codex", "OpenAI's Codex"),
               ("claude-app", "the Claude app", "you start its session in the app"))

    def __init__(self, controls: AssistantControls, parent=None):
        super().__init__(parent)
        self.controls = controls
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Start assistant")
        column = QVBoxLayout(self)
        column.addWidget(QLabel("Which assistant should SMAPpy start?"))
        self.group = QButtonGroup(self)
        self.buttons: Dict[str, QRadioButton] = {}
        for n, (key, label, help_) in enumerate(self.CHOICES):
            button = QRadioButton(label)
            button.setToolTip(help_)
            button.setChecked(n == 0)
            self.group.addButton(button)
            self.buttons[key] = button
            column.addWidget(button)
        note = QLabel("It opens in a window of its own and works with your own "
                      "account.  If it is not installed yet, that window offers to "
                      "install it.")
        note.setWordWrap(True)
        note.setStyleSheet(note_style())
        column.addWidget(note)
        row = QHBoxLayout()
        other = QToolButton(text="another assistant...")
        other.setToolTip("the settings: a command of your own, or how to connect "
                         "an assistant you start yourself")
        other.clicked.connect(self._other)
        row.addWidget(other)
        row.addStretch(1)
        start = QPushButton("Start")
        start.setDefault(True)
        start.clicked.connect(self.start)
        row.addWidget(start)
        column.addLayout(row)

    def chosen(self) -> str:
        return next(k for k, b in self.buttons.items() if b.isChecked())

    def start(self) -> None:
        save_preferences({"client": self.chosen()})
        self.close()
        self.controls.start()

    def _other(self) -> None:
        self.close()
        self.controls.show_settings()


class ProposalOverlay(QFrame):
    """The list dropped down inside the render window, over the picture's
    top-right corner, under the button: *collapse* folds it back into it."""

    WIDTH = 440

    def __init__(self, controls: AssistantControls, host: QWidget):
        super().__init__(host)
        self.host = host
        self.setFrameShape(QFrame.StyledPanel)
        self.setAutoFillBackground(True)        # over the picture, not see-through
        column = QVBoxLayout(self)
        column.setContentsMargins(8, 4, 8, 8)
        top = QHBoxLayout()
        title = QLabel("Assistant")
        title.setStyleSheet("font-weight: bold")
        top.addWidget(title)
        top.addStretch(1)
        self.collapse_button = QToolButton(text="\u2014", autoRaise=True)
        self.collapse_button.setToolTip("collapse: look at the whole picture first; "
                                        "the button keeps showing what waits")
        self.collapse_button.clicked.connect(self.collapse)
        top.addWidget(self.collapse_button)
        column.addLayout(top)
        self.list = ProposalList(controls, self)
        column.addWidget(self.list)
        host.installEventFilter(self)
        self.hide()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt's name)
        try:
            if watched is self.host and event.type() == QEvent.Resize and self.isVisible():
                self.place()
        except Exception:
            traceback.print_exc()
        return False

    def place(self) -> None:
        width = min(self.WIDTH, max(200, self.host.width() - 16))
        height = min(max(self.sizeHint().height(), 160), max(120, self.host.height() - 16))
        self.setGeometry(self.host.width() - width - 8, 8, width, height)

    def open(self) -> None:
        self.place()
        self.show()
        self.raise_()

    def collapse(self) -> None:
        self.hide()


class AssistantButton(QToolButton):
    """The toolbar's button: the connection, the client, the mode and how
    many proposals wait -- highlighted while they wait out of sight -- and a
    click opens or collapses the list under it (`ProposalOverlay`, inside
    ``host``, the picture).  Made before anything of the assistant is
    imported: ``controls`` is called on the first click."""

    def __init__(self, controls: Callable[[], AssistantControls],
                 host: Optional[QWidget] = None, parent=None):
        super().__init__(parent)
        self._controls = controls
        self._made: Optional[AssistantControls] = None
        self.host = host
        self.overlay: Optional[ProposalOverlay] = None
        self._known = 0                 # proposals there were at the last look
        self._had_live = False
        self._said = ""
        self.setText("Assistant \u25be")
        self.setToolTip("an AI assistant working with this session: who is connected, "
                        "what it may do, what it proposed")
        self.clicked.connect(self.toggle)

    def attach(self, controls: AssistantControls) -> None:
        if self._made is controls:
            return
        self._made = controls
        host = self.host if self.host is not None else self.window()
        self.overlay = ProposalOverlay(controls, host)
        self.overlay.installEventFilter(self)       # shown or hidden: the highlight
        controls.reveal = self.overlay.open
        self._known = len(controls.server.assistant.proposals)
        controls.changed.connect(self._react)
        self.refresh()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt's name)
        if watched is self.overlay and event.type() in (QEvent.Show, QEvent.Hide):
            self.refresh()
        return False

    def toggle(self) -> None:
        try:
            self.attach(self._controls())
            if self.overlay.isVisible():
                self.overlay.collapse()
            else:
                self.overlay.open()
        except Exception:
            traceback.print_exc()

    def _react(self) -> None:
        """Open for what arrives; collapse once nothing is left to do."""
        try:
            controls = self._made
            proposals = controls.server.assistant.proposals
            live = bool(controls.live())
            if len(proposals) > self._known and live:
                self.overlay.open()             # a new proposal
            elif self._had_live and not live:
                self.overlay.collapse()         # the last one dealt with
            if controls.message and controls.message != self._said:
                self.overlay.open()             # something said to the user
            self._known, self._had_live, self._said = len(proposals), live, controls.message
        except Exception:
            traceback.print_exc()
        self.refresh()

    def refresh(self) -> None:
        try:
            controls = self._made
            if controls is None:
                return
            server = controls.server
            count = len(controls.live())
            waiting = count and not (self.overlay is not None and self.overlay.isVisible())
            if not server.listening and not count:
                self.setText("Assistant \u25be")
                self.setStyleSheet("")
                return
            dot = "\u25cf" if server.clients else "\u25cb"
            name = controls.client_label() or "Assistant"
            text = f"{dot} {name} \u00b7 {server.assistant.mode}"
            if count:
                text += f" \u00b7 {count}"
            self.setText(text + " \u25be")
            self.setStyleSheet(f"QToolButton {{ {NEW_STYLE} }}" if waiting else "")
        except Exception:
            traceback.print_exc()


class AssistantSettings(QWidget):
    """What is set once: the client, the folders, the mode to start in;
    listening; how to connect a client SMAPpy does not start."""

    def __init__(self, session, controls: Optional[AssistantControls] = None, parent=None):
        super().__init__(parent)
        from ..assistant import launch
        self.controls = controls or AssistantControls(session, parent=self)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Assistant settings")
        prefs = preferences()
        column = QVBoxLayout(self)
        top = QHBoxLayout()
        self.allow = QCheckBox("allow connections")
        self.allow.setToolTip("listen for an assistant's client on this computer "
                              "(only your user can connect)")
        self.allow.toggled.connect(self.controls.allow)
        help_ = QToolButton(text="?", autoRaise=True)
        help_.setToolTip("what the assistant can do here, and how to connect one")
        help_.clicked.connect(self.show_help)
        top.addWidget(self.allow)
        top.addStretch(1)
        top.addWidget(help_)
        column.addLayout(top)
        self.state = QLabel("")
        self.state.setWordWrap(True)
        self.state.setStyleSheet(note_style())
        column.addWidget(self.state)

        grid = QGridLayout()
        grid.addWidget(QLabel("assistant"), 0, 0)
        self.client = QComboBox()
        self.client.addItem("choose...", "")
        for key, client in launch.CLIENTS.items():
            self.client.addItem(client.label, key)
        self.client.setCurrentIndex(max(0, self.client.findData(prefs["client"])))
        self.client.setToolTip("what Start assistant starts")
        self.client.currentIndexChanged.connect(self._client_changed)
        grid.addWidget(self.client, 0, 1, 1, 2)
        self.command_label = QLabel("command")
        self.command = QLineEdit(prefs["command"])
        self.command.setPlaceholderText("e.g. myagent --mcp {config}")
        self.command.setToolTip("run in a terminal; {server} is the command that "
                                "reaches this window, {config} a JSON file naming it")
        self.command.editingFinished.connect(
            lambda: save_preferences({"command": self.command.text()}))
        grid.addWidget(self.command_label, 1, 0)
        grid.addWidget(self.command, 1, 1, 1, 2)

        self.folder_label = QLabel("works in")
        grid.addWidget(self.folder_label, 2, 0)
        self.folder = QLabel("")
        self.folder.setWordWrap(True)
        self.folder.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(self.folder, 2, 1)
        folder_buttons = QHBoxLayout()
        folder_button = self.folder_button = QToolButton(text="choose...")
        folder_button.setToolTip("the folder the assistant starts in")
        folder_button.clicked.connect(self._choose_folder)
        self.same = QToolButton(text="same")
        self.same.setToolTip("start where it may write")
        self.same.clicked.connect(lambda: (save_preferences({"folder": ""}), self.refresh()))
        folder_buttons.addWidget(folder_button)
        folder_buttons.addWidget(self.same)
        grid.addLayout(folder_buttons, 2, 2)

        grid.addWidget(QLabel("writes under"), 3, 0)
        self.output = QLabel("")
        self.output.setWordWrap(True)
        self.output.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(self.output, 3, 1)
        out_buttons = QHBoxLayout()
        choose = QToolButton(text="choose...")
        choose.setToolTip("the only folder the assistant may write files into")
        choose.clicked.connect(self._choose_output)
        self.follow = QToolButton(text="follow file")
        self.follow.setToolTip("the folder of the first open file, as it changes")
        self.follow.clicked.connect(self._follow_file)
        out_buttons.addWidget(choose)
        out_buttons.addWidget(self.follow)
        grid.addLayout(out_buttons, 3, 2)
        grid.setColumnStretch(1, 1)
        column.addLayout(grid)

        # the mode each start, both choices in sight: a tick for one of two
        # left the other to be guessed
        modes = QHBoxLayout()
        modes.addWidget(QLabel("starts in"))
        self.start_suggest = QRadioButton("suggest")
        self.start_suggest.setToolTip("it reads and tries; what it would change "
                                      "waits for you")
        self.start_act = QRadioButton("act")
        self.start_act.setToolTip("it changes the session itself")
        start_mode = QButtonGroup(self)
        for button in (self.start_suggest, self.start_act):
            start_mode.addButton(button)
            modes.addWidget(button)
        modes.addStretch(1)
        (self.start_suggest if prefs["mode"] == "suggest" else self.start_act).setChecked(True)
        self.start_suggest.toggled.connect(
            lambda on: save_preferences({"mode": "suggest" if on else "act"}))
        column.addLayout(modes)
        self.at_start = QCheckBox("allow connections when SMAPpy starts")
        self.at_start.setChecked(prefs["allow_at_start"])
        self.at_start.toggled.connect(lambda on: save_preferences({"allow_at_start": on}))
        self.pre_allow = QCheckBox("Claude Code uses SMAPpy's tools without asking")
        self.pre_allow.setChecked(prefs["pre_allow"])
        self.pre_allow.setToolTip("what the assistant may change is decided by suggest "
                                  "and act here; its shell and files still ask")
        self.pre_allow.toggled.connect(lambda on: save_preferences({"pre_allow": on}))
        for box in (self.at_start, self.pre_allow):
            column.addWidget(box)

        # for "one I start myself": what to give that client
        self.other = QWidget()
        other = QHBoxLayout(self.other)
        other.setContentsMargins(0, 0, 0, 0)
        other.addWidget(QLabel("connect it with"))
        copy_command = QToolButton(text="copy command")
        copy_command.setToolTip("the command an MCP client runs to reach this window")
        copy_command.clicked.connect(self._copy_command)
        copy_json = QToolButton(text="copy JSON")
        copy_json.setToolTip("the mcpServers entry most clients' settings take")
        copy_json.clicked.connect(self._copy_json)
        other.addWidget(copy_command)
        other.addWidget(copy_json)
        other.addStretch(1)
        grid.addWidget(self.other, 1, 1, 1, 2)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        start = QPushButton("Start assistant")
        start.setToolTip("listen, start the assistant chosen above and place it "
                         "beside the picture")
        start.clicked.connect(self._start)
        buttons.addWidget(start)
        column.addLayout(buttons)
        column.addStretch(1)

        self.controls.changed.connect(self.refresh)
        self._client_changed()
        self.resize(520, 380)

    def refresh(self) -> None:
        try:
            server = self.controls.server
            self.allow.blockSignals(True)
            self.allow.setChecked(server.listening)
            self.allow.blockSignals(False)
            names = server.client_names()
            if server.listening:
                state = "listening -- " + (f"connected: {', '.join(names)}"
                                           if names else "nobody connected yet")
            elif server.error:
                state = f"could not listen: {server.error}"
            else:
                state = "off: no assistant can connect"
            if self.controls.message:
                state += "\n" + self.controls.message
            self.state.setText(state)
            self.output.setText(str(server.assistant.output))
            self.follow.setEnabled(server.output_pinned)
            folder = preferences()["folder"]
            self.folder.setText(folder or "where it writes")
            self.same.setEnabled(bool(folder))
        except Exception:
            traceback.print_exc()

    def _client_changed(self, *_) -> None:
        key = self.client.currentData() or ""
        if key != preferences()["client"]:
            save_preferences({"client": key})
        # each choice shows what it needs, and nothing else
        own = key == "command"
        self.command_label.setVisible(own)
        self.command.setVisible(own)
        self.other.setVisible(key == "manual")
        for widget in (self.folder_label, self.folder, self.folder_button, self.same):
            widget.setVisible(key != "manual")      # it starts wherever its user does
        self.pre_allow.setVisible(key == "claude")
        self.refresh()

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "The folder the assistant starts in", preferences()["folder"]
            or str(self.controls.server.assistant.output))
        if folder:
            save_preferences({"folder": folder})
            self.refresh()

    def _choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "The folder the assistant may write into",
            str(self.controls.server.assistant.output))
        if folder:
            save_preferences({"output": folder})
            self.controls.server.set_output(folder)

    def _follow_file(self) -> None:
        save_preferences({"output": ""})
        self.controls.server.set_output(None)

    def _copy_command(self) -> None:
        import subprocess
        from ..assistant.launch import server_command
        QApplication.clipboard().setText(subprocess.list2cmdline(server_command()))

    def _copy_json(self) -> None:
        from ..assistant.launch import mcp_config
        QApplication.clipboard().setText(json.dumps(mcp_config(), indent=1))

    def _start(self) -> None:
        save_preferences({"command": self.command.text()})
        self.controls.start()

    def show_help(self) -> None:
        from .help_window import show_help
        show_help("Panels/Assistant", self)
