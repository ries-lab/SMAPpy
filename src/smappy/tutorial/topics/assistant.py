"""An AI assistant: started beside the picture, run on its own, writing plugins.

The assistant is the user's own client (Claude Code, Codex, the Claude app;
`docs/ai-integration.md`), so a tutorial cannot run it.  What it would do
is done for it, through the same doors:

* *Start assistant* is clicked for real -- the menu, the first start's one
  question, the window listening, the terminal's script written -- with
  `launch.open_terminal` answered, so no terminal opens on the machine that
  builds the tutorial.
* A client is connected to the window's real socket (`_Client`) and calls
  the tools an assistant calls: `show`, then `run_plugin` in suggest mode,
  so the proposals in the overlay are real ones, *Open* fills the real
  window and its Run carries the proposal out.
* The terminal and the chat are drawn on cards, and say they are drawings.
  The terminal's welcome is `launch.welcome`'s, as the real one prints it.

The plugin part builds on `own_plugins` rather than repeating it.
"""
from __future__ import annotations

import html
import json
import time
from contextlib import contextmanager

from .common import menu, menu_item, open_section, show_tab, type_into

TITLE = "An AI assistant: asking SMAPpy in plain words"
DESCRIPTION = ("Starting an assistant beside the picture, its proposals, suggest "
               "and act; running it on its own; and writing plugins with it.")

CLIENT = "claude-code"          # what Claude Code calls itself on connecting


def _terminal(lines, label: str, prompt_from: int = 0) -> str:
    """A drawn terminal: ``lines`` in a monospace box, those before
    ``prompt_from`` dimmed (what was there before, or the program's own)."""
    height = 40 + 20 * len(lines)
    rows = "".join(
        f'<text x="28" y="{36 + 20 * i}"{"" if i >= prompt_from else " opacity=\".7\""}>'
        f"{html.escape(line)}</text>"
        for i, line in enumerate(lines))
    return (f'<svg viewBox="0 0 600 {height}" role="img" aria-label="{html.escape(label)}">'
            f'<rect x="10" y="8" width="580" height="{height - 16}" rx="8" fill="none" '
            f'stroke="currentColor" opacity=".35"/>'
            f'<g font-size="13" fill="currentColor" font-family="ui-monospace, monospace">'
            f"{rows}</g></svg>")


class _Client:
    """What an assistant's client is to the window: a line of JSON-RPC each
    way over its local socket.  The window answers on its own event loop,
    which `Director.pump` runs while this waits."""

    def __init__(self, d, address: str):
        from PySide6.QtNetwork import QLocalSocket
        self.d = d
        self.socket = QLocalSocket()
        self.socket.connectToServer(address)
        self._wait(lambda: self.socket.state() == QLocalSocket.ConnectedState)
        self.buffer = b""
        self.n = 0

    def _wait(self, condition, seconds: float = 60.0) -> None:
        end = time.monotonic() + seconds
        while not condition():
            if time.monotonic() > end:
                raise TimeoutError("the window did not answer the assistant's client")
            self.d.pump(0.02)

    def send(self, method: str, params=None) -> dict:
        self.n += 1
        message = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}
        self.socket.write((json.dumps(message) + "\n").encode())
        self.socket.flush()

        def complete():
            self.buffer += bytes(self.socket.readAll().data())
            return b"\n" in self.buffer
        self._wait(complete)
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def call(self, tool: str, arguments=None) -> dict:
        result = self.send("tools/call", {"name": tool, "arguments": arguments or {}})["result"]
        assert not result.get("isError"), result
        return json.loads(result["content"][0]["text"])

    def close(self) -> None:
        self.socket.disconnectFromServer()


@contextmanager
def _no_terminal():
    """*Start assistant* without its terminal: the machine that builds the
    tutorial is not the viewer's, and a Terminal opening on it helps nobody."""
    from ...assistant import launch
    kept = launch.open_terminal, launch.raise_terminal
    try:
        # the script is written as for anyone; only the terminal is not opened
        launch.open_terminal = lambda *a, **k: None
        launch.raise_terminal = lambda *a, **k: False
        yield
    finally:
        launch.open_terminal, launch.raise_terminal = kept


def _menu_action(d, title: str, text: str):
    menu_ = next(a.menu() for a in d.control.menuBar().actions()
                 if a.text().replace("&", "") == title)
    return next(a for a in menu_.actions() if a.text().replace("&", "") == text)


def _place(d, window) -> None:
    """A window at the left of the picture, clear of the proposals' corner,
    as tall as its contents."""
    x, y, _, _ = d.window_rect(d.render)
    d.place(window, x + 20, y + 50, 430, window.sizeHint().height())
    d.settle()


def make(d) -> None:
    from PySide6.QtWidgets import QPushButton, QToolButton
    from ...assistant import launch
    control = d.control
    button = control.assistant_button

    # nuclear pores, simulated as in the simulation tutorial
    file_tab = show_tab(d, "File")
    _, simulate = open_section(d, file_tab, "Simulate")
    type_into(simulate, "preset", "Nuclear pores (Nup96)")
    d.settle()
    simulate.run_button.click()
    d.settle()
    show_tab(d, "Render")

    d.chapter("An assistant")
    d.card(TITLE,
           "<p>An AI assistant such as Claude or Codex can work with SMAPpy. You "
           "ask in plain words; it reads the help pages, tries plugins on a copy "
           "of your data and looks at the results.</p>"
           "<p>It runs in its own program, with your own account. SMAPpy has no "
           "AI of its own.</p>",
           say="An AI assistant can work with SMAPpy: you ask in plain words, and "
               "it uses SMAPpy's plugins for you.")

    # where the assistant may write: the demo's folder, as a user's own data
    # folder would be, rather than wherever the tutorial was built
    from ...gui.assistant_pane import save_preferences
    save_preferences({"output": str(d.data_dir())})

    d.chapter("Starting it")
    assistant_menu = menu(d, "Assistant")
    item = menu_item(d, assistant_menu, "Start assistant")
    d.shot("Assistant, Start assistant, or Ctrl+Shift+A, starts it beside the "
           "picture, already connected.",
           spot=[item], point=item, click=True, zoom=d.around(assistant_menu, 700))
    assistant_menu.close()
    _menu_action(d, "Assistant", "Start assistant").trigger()
    d.settle()
    chooser = control.assistant_controls.chooser
    _place(d, chooser)
    d.shot("The first time it asks one question: which assistant. That is all "
           "there is to set up.",
           spot=[d.window_rect(chooser)], point=chooser.buttons["claude"])
    start = next(b for b in chooser.findChildren(QPushButton) if b.text() == "Start")
    with _no_terminal():
        start.click()
        d.settle()
    server = control.assistant_controls.server
    assert server.listening, "Start assistant did not listen"
    d.card("Its own window",
           "<p>The assistant opens in a terminal beside the picture. If it is "
           "not installed yet, the terminal offers to install it: press "
           "Enter.</p>"
           "<p>The first time, it asks you to sign in with your account. Then "
           "you type your questions there.</p>",
           figure=_terminal(
               ["Claude Code is not installed on this computer yet.",
                "Install it now? [Y/n]", ""] + launch.welcome(launch.CLIENTS["claude"]),
               "a drawing of the assistant's terminal: the offer to install it, "
               "then SMAPpy's welcome", prompt_from=3),
           say="It opens in a window of its own. If it is not installed yet, "
               "that window offers to install it.")
    client = _Client(d, server.address)
    client.send("initialize", {"clientInfo": {"name": CLIENT}})
    button.overlay.collapse()
    d.settle()
    assert button.text().startswith("● " + CLIENT), button.text()
    d.shot("The button in the picture's toolbar says that an assistant is "
           "connected, and that it may only suggest.",
           spot=[button], point=button, zoom=d.around(button, 760))

    d.chapter("Asking")
    d.card("Ask in plain words",
           "<p>You type a question in the assistant's window. It looks up which "
           "plugins measure that, reads their help pages and tries them on a "
           "copy of the data.</p>"
           "<p>It answers in its window, and shows you in SMAPpy what it "
           "found.</p>",
           figure=_terminal(
               ["> How many nuclear pores are there? Is there drift?", "",
                "  Find clusters, tried on a copy, finds one cluster per",
                "  pore. RCC finds little drift. Both wait in SMAPpy."],
               "a drawing of a question to the assistant and its answer"),
           say="You ask in its window. It tries plugins on a copy of the data, "
               "and answers there.")
    # what an assistant does for that question: look, try, then propose
    client.call("show", {"x_nm": 5000, "y_nm": 5000, "width_nm": 3000})
    d.settle()
    d.shot("It can move the view to show you what it means. That changes no "
           "data, so it does it at once.",
           spot=[d.rect(d.render.view.graphics, pad=-2)])
    client.call("run_plugin", {"path": "Analysis/Regions/Find clusters",
                               "settings": {"dbscan.radius_nm": 60}, "apply": True,
                               "reason": "one cluster per pore: their number is the "
                                         "number of pores"})
    client.call("run_plugin", {"path": "Analysis/Drift/RCC", "apply": True,
                               "reason": "the drift is small, but correcting it "
                                         "costs little"})
    d.settle()
    overlay, listed = button.overlay, button.overlay.list
    assert overlay.isVisible() and listed.page.text() == "2 / 2", listed.page.text()
    d.shot("What it would change waits here, under the button, with its reason. "
           "You decide.",
           spot=[overlay], zoom=d.around(overlay, 900))
    d.shot("There are two: the arrows step from one to the other.",
           spot=[d.union(listed.previous, listed.page, listed.next)],
           point=listed.previous, click=True, zoom=d.around(overlay, 900))
    listed.previous.click()
    d.settle()
    overlay.collapse_button.click()
    d.settle()
    d.shot("The dash folds the list away, to see the whole picture. The button "
           "stays highlighted while something waits.",
           spot=[button], point=button, click=True, zoom=d.around(button, 760))
    button.click()
    d.settle()
    assert "Find clusters" in listed.text.text(), listed.text.text()
    d.shot("Open shows the plugin with the settings it proposes.",
           spot=[listed.card], point=listed.go, click=True, zoom=d.around(overlay, 900))
    listed.go.click()
    d.settle()
    window = control._plugin_windows["Analysis/Regions/Find clusters"]
    _place(d, window)
    assert window.panel.form.value().dbscan.radius_nm == 60
    d.shot("Nothing has run yet. Check the settings, change them if you like, "
           "and press Run.",
           spot=[d.window_rect(window)], point=window.panel.run_button, click=True)
    window.panel.run_button.click()
    d.settle()
    found = server.assistant.proposals[0]
    assert found.status == "done", found.status
    figure = getattr(window.panel, "_window", None)
    if figure is not None:
        figure.hide()
    window.hide()
    d.settle()
    d.shot("Your Run carries it out, and the assistant reads the result. The "
           "next proposal takes its place.",
           spot=[overlay], zoom=d.around(overlay, 900))
    d.shot("The cross dismisses a proposal. The assistant sees what became of "
           "each one.",
           spot=[listed.dismiss], point=listed.dismiss, click=True,
           zoom=d.around(overlay, 900))
    listed.dismiss.click()
    d.settle()
    assert not overlay.isVisible(), "the list did not fold away when empty"

    d.chapter("Suggest or act")
    button.click()
    d.settle()
    d.shot("Under the list: suggest, where every change waits for you, and act.",
           spot=[d.union(listed.suggest, listed.act)], point=listed.suggest,
           zoom=d.around(overlay, 900))
    listed.act.click()
    d.settle()
    assert server.assistant.mode == "act"
    d.shot("In act mode the assistant changes the data itself. Each change can "
           "be undone with Edit, Undo. Only you can switch.",
           spot=[listed.act, button], point=listed.act, click=True,
           zoom=d.around(overlay, 900))
    listed.suggest.click()
    d.settle()
    next(b for b in listed.findChildren(QToolButton) if b.text() == "settings...").click()
    d.settle()
    settings = control.assistant_controls.settings_window
    _place(d, settings)
    d.shot("Settings: which assistant starts, the only folder it may write "
           "into, and the mode it starts in.",
           spot=[d.window_rect(settings)], point=settings.output)

    d.chapter("On its own")
    d.card("Without the window",
           "<p>The assistant can also use SMAPpy on its own, with no window "
           "open: it opens files, tries methods on simulated data whose truth "
           "it knows, and compares settings.</p>"
           "<p>Connect it once with one line. Then ask it, in a folder of your "
           "data.</p>",
           figure=_terminal(
               ["$ claude mcp add smappy -- smappy-mcp --output ~/analysis",
                "$ claude",
                "> Which radius counts the pores best? Check it on",
                "  simulated pores, then make it a chain."],
               "a drawing of a terminal: SMAPpy's tools added to Claude Code "
               "once, then a question", prompt_from=2),
           say="On its own, the assistant can try methods on simulated data, "
               "with no window open.")
    d.card("Ending in a chain",
           "<p>What it settles on, it writes as a chain of plugins. You can open "
           "the chain in SMAPpy, and run it over all your files with "
           "smappy-batch.</p>"
           "<p>The tutorial on chains and batch runs shows how.</p>",
           say="What it finds, it writes as a chain, which runs over many files.")
    settings.client.setCurrentIndex(settings.client.findData("manual"))
    d.settle()
    assert settings.other.isVisible()
    d.shot("An assistant you start yourself: SMAPpy only listens, and copy "
           "command gives the line that connects it, smappy-mcp --attach.",
           spot=[settings.client, settings.other], point=settings.other)
    settings.hide()
    client.close()
    server.stop()
    overlay.collapse()
    d.settle()

    d.chapter("Writing plugins")
    plugins_menu = menu(d, "Plugins")
    new = menu_item(d, plugins_menu, "New plugin")
    folder = menu_item(d, plugins_menu, "Show my plugin folder")
    d.shot("The assistant can also write a plugin for you. Plugins, New plugin "
           "starts one; Show my plugin folder opens where yours live.",
           spot=[new, folder], point=new, zoom=d.around(plugins_menu, 700))
    plugins_menu.close()
    d.card("Describe it",
           "<p>Your plugin folder holds a file for the assistant, AGENTS.md, "
           "that says how a SMAPpy plugin is written.</p>"
           "<p>Start the assistant in that folder and describe what to "
           "measure. Its plugin appears under Plugins, User, with no "
           "restart.</p>"
           "<p>The tutorial on your own plugins shows it step by step.</p>",
           figure=_terminal(
               ['$ cd "your plugin folder"', "$ claude",
                "> Count the localizations within a radius of each one,",
                "  as a histogram, and the median per ROI."],
               "a drawing of a terminal in the plugin folder: the assistant "
               "started there and told what to measure", prompt_from=2),
           say="Start the assistant in your plugin folder and describe what to "
               "measure: it writes the plugin.")

    d.chapter("Next")
    d.card("What to remember",
           "<ul><li><b>Assistant, Start assistant</b>: your assistant, beside "
           "the picture.</li>"
           "<li>What it would change waits under its button: <b>Open</b>, then "
           "your Run, or the cross.</li>"
           "<li><b>suggest</b> or <b>act</b>: only you switch.</li>"
           "<li>On its own, with <b>smappy-mcp</b>; and in your plugin folder, "
           "for plugins of your own.</li>"
           "<li>The help page <b>Assistant</b> has the details.</li></ul>",
           say="That is the assistant: ask in plain words, and decide what it "
               "may change.")
