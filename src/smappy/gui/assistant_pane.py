"""The Assistant pane: the connection, the switch and the proposals.

The assistant itself runs in the user's own client (`docs/ai-integration.md`);
this window is SMAPpy's side of it.  It says whether connections are allowed
and who is connected, how a client connects, what the assistant may do --
*suggest* or *act* -- and where it may write, and it lists what was proposed
in *suggest* mode.  A proposal to run a plugin or a chain is never run from
here: *Open* shows the plugin's own panel filled in (`PluginPanel.
take_settings`), so the user reads the settings where they always do and runs
it with the panel's own Run.  The rest -- open a file, undo, save -- is one
click, *Do it*.
"""
from __future__ import annotations

import traceback
from typing import Callable, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QFileDialog,
                               QFrame, QHBoxLayout, QLabel, QPushButton,
                               QRadioButton, QScrollArea, QToolButton,
                               QVBoxLayout, QWidget)

from .widgets import note_style

# the command a client runs to reach this window; shown with a Copy button
ATTACH = "smappy-mcp --attach"
CLAUDE_CODE = f"claude mcp add smappy-gui -- {ATTACH}"
OPENS = ("run_plugin", "run_chain")      # proposals shown in a panel, not done here


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


class AssistantPane(QWidget):
    """The pane over an `AssistantServer`; made with a server of its own
    when given none (a test, the panel page's check)."""

    def __init__(self, session, server=None, parent=None,
                 open_plugin: Optional[Callable] = None,
                 load: Optional[Callable] = None):
        super().__init__(parent)
        if server is None:
            from .assistant_server import AssistantServer
            server = AssistantServer(session, parent=self)
        self.session = session
        self.server = server
        # how a proposal is opened: the main window's own plugin windows and
        # loader when there is one, so a plugin opened here is the one the
        # Plugins menu opens, with what was typed into it before
        self.open_plugin = open_plugin or self._own_window
        self.load = load
        self._windows: Dict[str, QWidget] = {}
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Assistant")

        column = QVBoxLayout(self)
        top = QHBoxLayout()
        self.allow = QCheckBox("allow connections")
        self.allow.setToolTip("listen for an AI assistant's client on this computer "
                              "(only your user can connect)")
        self.allow.toggled.connect(self._toggle)
        help_ = QToolButton(text="?", autoRaise=True)
        help_.setToolTip("what the assistant can do here, and how to connect one")
        help_.clicked.connect(self.show_help)
        top.addWidget(self.allow)
        top.addStretch(1)
        top.addWidget(help_)
        self.state = QLabel("")
        self.state.setWordWrap(True)
        self.state.setTextInteractionFlags(Qt.TextSelectableByMouse)
        column.addLayout(top)
        column.addWidget(self.state)

        connect = QHBoxLayout()
        self.command = QLabel(f"<code>{CLAUDE_CODE}</code>")
        self.command.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.command.setWordWrap(True)
        self.command.setToolTip("how Claude Code connects; another client runs "
                                f"'{ATTACH}' as its MCP server's command")
        copy = QToolButton(text="copy")
        copy.setToolTip("copy the command")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(CLAUDE_CODE))
        connect.addWidget(self.command, 1)
        connect.addWidget(copy)
        column.addLayout(connect)

        modes = QHBoxLayout()
        modes.addWidget(QLabel("the assistant may"))
        self.suggest = QRadioButton("suggest")
        self.suggest.setToolTip("read, try on copies and propose; a change waits "
                                "for you below.  Filters, grouping and the view "
                                "it changes at once")
        self.act = QRadioButton("act")
        self.act.setToolTip("change the session itself: run plugins, open and "
                            "save files under the output folder")
        group = QButtonGroup(self)
        for button in (self.suggest, self.act):
            group.addButton(button)
            modes.addWidget(button)
        modes.addStretch(1)
        self.suggest.toggled.connect(
            lambda on: self.server.set_mode("suggest" if on else "act"))
        column.addLayout(modes)

        output = QHBoxLayout()
        output.addWidget(QLabel("writes under"))
        self.output = QLabel("")
        self.output.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.output.setWordWrap(True)
        choose = QToolButton(text="choose...")
        choose.setToolTip("the only folder the assistant may write files into")
        choose.clicked.connect(self._choose_output)
        self.follow = QToolButton(text="follow file")
        self.follow.setToolTip("the folder of the first open file, as it changes")
        self.follow.clicked.connect(lambda: self.server.set_output(None))
        output.addWidget(self.output, 1)
        output.addWidget(choose)
        output.addWidget(self.follow)
        column.addLayout(output)

        self.activity = QLabel("")
        self.activity.setStyleSheet(note_style())
        self.activity.setWordWrap(True)
        column.addWidget(self.activity)

        heading = QLabel("proposals")
        heading.setStyleSheet("font-weight: bold")
        column.addWidget(heading)
        self.rows = QWidget()
        self.rows_layout = QVBoxLayout(self.rows)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.rows)
        column.addWidget(scroll, 1)
        self.empty = QLabel("none: in suggest mode what the assistant would change "
                            "appears here, to open or dismiss")
        self.empty.setWordWrap(True)
        self.empty.setStyleSheet(note_style())
        self.rows_layout.insertWidget(0, self.empty)
        self._shown_proposals: tuple = ()

        server.changed.connect(self.refresh)
        self.resize(420, 480)
        self.refresh()

    # ------------------------------------------------------------- the state
    def refresh(self) -> None:
        try:
            self._refresh()
        except Exception:
            traceback.print_exc()

    def _refresh(self) -> None:
        server, assistant = self.server, self.server.assistant
        self.allow.blockSignals(True)
        self.allow.setChecked(server.listening)
        self.allow.blockSignals(False)
        if server.listening:
            names = server.client_names()
            who = (f"{len(names)} connected: {', '.join(names)}" if names
                   else "no client connected yet")
            self.state.setText(f"listening -- {who}")
        elif server.error:
            self.state.setText(f"could not listen: {server.error}")
        else:
            self.state.setText("off: no assistant can connect")
        for button, mode in ((self.suggest, "suggest"), (self.act, "act")):
            button.blockSignals(True)
            button.setChecked(assistant.mode == mode)
            button.blockSignals(False)
        self.output.setText(str(assistant.output))
        self.follow.setEnabled(server.output_pinned)
        if server.running:
            self.activity.setText(f"running {server.running}...")
        elif server.calls:
            when, tool, ok = server.calls[-1]
            self.activity.setText(f"last call: {tool} at {when}"
                                  + ("" if ok else " (refused or failed)"))
        else:
            self.activity.setText("")
        self._fill_proposals()

    def _fill_proposals(self) -> None:
        live = [p for p in self.server.assistant.proposals
                if p.status in ("proposed", "opened")]
        key = tuple((p.id, p.status) for p in live)
        if key == self._shown_proposals:
            return                          # a call that proposed nothing new
        self._shown_proposals = key
        while self.rows_layout.count() > 2:  # the note and the stretch stay
            item = self.rows_layout.takeAt(1)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.empty.setVisible(not live)
        for n, proposal in enumerate(live):
            self.rows_layout.insertWidget(1 + n, self._row(proposal))

    def _row(self, proposal) -> QWidget:
        row = QFrame()
        row.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(row)
        layout.setContentsMargins(6, 4, 6, 4)
        text = QLabel(describe(proposal))
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(text)
        if proposal.reason:
            why = QLabel(proposal.reason)
            why.setWordWrap(True)
            why.setStyleSheet(note_style())
            layout.addWidget(why)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        if proposal.status == "opened":
            buttons.addWidget(QLabel("opened"))
        if proposal.tool in OPENS:
            go = QPushButton("Open")
            go.setToolTip("the plugin's panel with these settings: check them, "
                          "then run it there")
            go.clicked.connect(lambda _=False, p=proposal.id: self.open_proposal(p))
        else:
            go = QPushButton("Do it")
            go.setToolTip("do what is proposed, now")
            go.clicked.connect(lambda _=False, p=proposal.id: self.do_proposal(p))
        dismiss = QPushButton("Dismiss")
        dismiss.setToolTip("take it off the list; the assistant can see you did")
        dismiss.clicked.connect(lambda _=False, p=proposal.id: self.dismiss(p))
        buttons.addWidget(go)
        buttons.addWidget(dismiss)
        layout.addLayout(buttons)
        return row

    # ------------------------------------------------------------ the actions
    def show_help(self) -> None:
        from .help_window import show_help
        show_help("Panels/Assistant", self)

    def _toggle(self, on: bool) -> None:
        if on:
            self.server.start()
        else:
            self.server.stop()

    def _choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "The folder the assistant may write into",
            str(self.server.assistant.output))
        if folder:
            self.server.set_output(folder)

    def open_proposal(self, ident: str) -> Optional[QWidget]:
        """The proposal's plugin in its window, filled in; nothing run."""
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
            self.activity.setText(f"could not open it: {error}")
            return None
        if window is None:
            return None
        if settings is not None:
            window.panel.take_settings(settings)
        proposal.status = "opened"
        self.server.changed.emit()
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
            self.activity.setText(f"{proposal.tool} failed: {error}")
        self.server.changed.emit()

    def dismiss(self, ident: str) -> None:
        self.server.assistant.proposal(ident).status = "dismissed"
        self.server.changed.emit()

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
            window = PluginWindow(cls or plugins.get(path), self.session, self)
            self._windows[path] = window
        window.show()
        return window
