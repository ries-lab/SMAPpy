"""Starting the user's own assistant, already connected to the running GUI.

*Assistant > Start assistant* runs the client the user chose -- Claude Code
in a terminal, the Claude app, Codex, or a command of their own -- so that
nobody types ``claude mcp add`` to begin.  The AI stays the user's: SMAPpy
opens their program the way they would, in a terminal or as the app, and
passes it no key and no login (`docs/ai-integration.md`).

**The connection is on the client's command line**, not in its settings:
``claude --mcp-config FILE`` reads the server from a file SMAPpy writes, and
Codex takes the same as ``-c`` overrides.  The server command is this
Python's own ``-m smappy.assistant.server --attach PID``, which needs no
``smappy-mcp`` on the PATH and reaches this window, not another one.  The
Claude app cannot be started on a folder from outside, so for it the server
goes into the folder's ``.mcp.json`` -- after the user agreed -- where a Code
session started there finds it.

**The terminal is placed** beside SMAPpy's windows where the platform lets a
program do that: Terminal on a Mac through AppleScript.  On Windows and Linux
it opens where the system puts it.

A script is written and the terminal runs it, rather than a command typed
into AppleScript or ``cmd``: three layers of quoting (Python, the terminal's
own language, the shell) around a JSON path were where this went wrong.  The
script says how to install the client when it is not found, instead of the
shell's bare "command not found".
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

NAME = "smappy-gui"           # the server's name in the client's settings


@dataclass(frozen=True)
class Client:
    key: str
    label: str
    program: str = ""         # what must be installed; "" for none
    name: str = ""            # what it is called in a sentence
    installer: str = ""       # a shell line that installs it
    installer_windows: str = ""
    terminal: bool = True     # runs in a terminal SMAPpy opens


CLIENTS: Dict[str, Client] = {c.key: c for c in (
    Client("claude", "Claude Code in a terminal", "claude", "Claude Code",
           "curl -fsSL https://claude.ai/install.sh | bash",
           'powershell -NoProfile -ExecutionPolicy Bypass -Command '
           '"irm https://claude.ai/install.ps1 | iex"'),
    Client("claude-app", "Claude app (Code tab)", terminal=False),
    Client("codex", "Codex CLI in a terminal", "codex", "Codex",
           # Homebrew where there is one, else npm; neither: said
           "if command -v brew >/dev/null 2>&1; then brew install codex; "
           "elif command -v npm >/dev/null 2>&1; then npm install -g @openai/codex; "
           "else echo 'Codex needs Homebrew (https://brew.sh) or Node.js "
           "(https://nodejs.org) to install.'; false; fi",
           "npm install -g @openai/codex"),
    Client("command", "a command of your own"),
    # any other MCP client, started by the user with the command or the JSON
    # the settings copy; SMAPpy only listens
    Client("manual", "one I start myself", terminal=False),
)}


def server_command(pid: Optional[int] = None) -> List[str]:
    """What a client runs to reach this window; ``pid`` pins it to this one
    (a file kept for later sessions leaves it out: the process changes)."""
    command = [sys.executable, "-m", "smappy.assistant.server", "--attach"]
    return command + ([str(pid)] if pid is not None else [])


def _server_entry(pid: Optional[int]) -> Dict:
    command = server_command(pid)
    entry: Dict = {"command": command[0], "args": command[1:]}
    # a checkout run with PYTHONPATH (a branch being tried) reaches its own
    # code in the server too: the client starts it without this environment
    if os.environ.get("PYTHONPATH"):
        entry["env"] = {"PYTHONPATH": os.environ["PYTHONPATH"]}
    return entry


def mcp_config(pid: Optional[int] = None) -> Dict:
    """The ``mcpServers`` JSON Claude Code (and most clients) read."""
    return {"mcpServers": {NAME: _server_entry(pid)}}


def _quote(part: str) -> str:
    return subprocess.list2cmdline([part]) if sys.platform.startswith("win") \
        else shlex.quote(part)


def client_command(key: str, config_file: Path, own: str = "",
                   pre_allow: bool = True):
    """The client's command line, its server in ``config_file``: a list of
    arguments, or for a command of the user's own the line as they wrote it
    with ``{server}`` (the server's command) and ``{config}`` (the JSON
    file) filled in."""
    if key == "claude":
        from .tools import source_folder
        # SMAPpy's own code beside the data it works in: read and grepped
        # without a question each time (edits there still ask)
        command = ["claude", "--mcp-config", str(config_file),
                   "--add-dir", str(source_folder())]
        if pre_allow:
            # SMAPpy's own switch decides what the assistant may change, so
            # asking again before every tool call would only be clicks
            command += ["--allowedTools", f"mcp__{NAME}"]
        return command
    if key == "codex":
        entry = json.loads(config_file.read_text(encoding="utf-8"))["mcpServers"][NAME]
        name = NAME.replace("-", "_")
        command = ["codex", "-c", f"mcp_servers.{name}.command={json.dumps(entry['command'])}",
                   "-c", f"mcp_servers.{name}.args={json.dumps(entry['args'])}"]
        for k, v in entry.get("env", {}).items():
            command += ["-c", f"mcp_servers.{name}.env.{k}={json.dumps(v)}"]
        return command
    if key == "command":
        if not own.strip():
            raise ValueError("no command: set it in the assistant's settings")
        entry = json.loads(config_file.read_text(encoding="utf-8"))["mcpServers"][NAME]
        server = " ".join(_quote(p) for p in [entry["command"], *entry["args"]])
        try:
            return own.format(server=server, config=_quote(str(config_file)))
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(f"the command's braces: only {{server}} and {{config}} "
                             f"are filled in ({error})") from None
    raise ValueError(f"no terminal command for {key!r}")


def write_config(folder: Path, pid: Optional[int]) -> Path:
    """The server's JSON, beside SMAPpy's settings."""
    target = folder / "mcp-gui.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(mcp_config(pid), indent=1), encoding="utf-8")
    return target


def welcome(client: Client) -> List[str]:
    """What the terminal says before the assistant starts: whoever has never
    seen one should know where to type and where its proposals go."""
    name = client.name or "The assistant"
    return [f"{name}, connected to the session open in SMAPpy.",
            "Ask here what you need: it explains how, suggests the plugins and their",
            "settings, or does it for you.  What it proposes to change waits in",
            "SMAPpy, under the Assistant button over the picture, to open or dismiss.",
            "The first time it asks you to sign in and whether to trust this folder."]


def script(command, folder: Path, client: Client) -> str:
    """A shell script that starts ``command`` in ``folder`` -- after offering
    to install the program it needs, so nobody copies a command anywhere."""
    lines = ["#!/bin/sh", f"cd {shlex.quote(str(folder))} || exit 1",
             # where Claude Code's installer puts it, which it leaves the user
             # to add to the PATH themselves: found without that step too
             'PATH="$PATH:$HOME/.local/bin"']
    if client.program:
        found = f"command -v {client.program} >/dev/null 2>&1"
        lines += [f"if ! {found}; then",
                  f"  echo {shlex.quote(client.name + ' is not installed on this computer yet.')}",
                  "  printf 'Install it now? [Y/n] '",
                  "  read answer",
                  '  case "$answer" in [nN]*)',
                  "    echo 'Not installed.  To install it later, run:'",
                  f"    echo {shlex.quote('    ' + client.installer)}",
                  "    exit 1;;",
                  "  esac",
                  f"  {client.installer} || {{ echo 'The installation failed: the lines "
                  "above say why.'; exit 1; }",
                  f"  hash -r 2>/dev/null; if ! {found}; then",
                  f"    echo {shlex.quote(client.name + ' was installed but is not found; open a new terminal and choose Start assistant in SMAPpy again.')}",
                  "    exit 1",
                  "  fi",
                  "fi"]
    lines += ["echo"] + [f"echo {shlex.quote(line)}" for line in welcome(client)] + ["echo"]
    lines.append(command if isinstance(command, str)
                 else "exec " + " ".join(shlex.quote(p) for p in command))
    return "\n".join(lines) + "\n"


def _percent(text: str) -> str:
    """``text`` as a batch file reads it literally: there ``%`` starts a
    variable, and a folder called ``100%`` lost its sign."""
    return text.replace("%", "%%")


def windows_script(command, folder: Path, client: Client) -> str:
    """A batch file that starts ``command`` in ``folder``.

    cmd.exe reads a batch file in the console's OEM code page, not in UTF-8,
    so ``cd /d "C:\\Users\\Jürgen\\..."`` went to a folder that did not
    exist; ``chcp 65001`` has it read the lines after it in UTF-8, as the
    file is written.  A command of the user's own is left as they wrote it,
    ``%VAR%`` and all.
    """
    lines = ["@echo off", "chcp 65001 >nul", f'cd /d "{_percent(str(folder))}"',
             r'set "PATH=%PATH%;%USERPROFILE%\.local\bin"']
    if client.program:
        lines += [f"where {client.program} >nul 2>nul && goto start",
                  f"echo {client.name} is not installed on this computer yet.",
                  'set /p answer="Install it now? [Y/n] "',
                  'if /i "%answer:~0,1%"=="n" (',
                  "  echo Not installed.  To install it later, run:",
                  f"  echo     {client.installer_windows}",
                  "  exit /b 1)",
                  # call: npm is npm.cmd, and a batch file run without it
                  # never returns, so the client was never started
                  f"call {client.installer_windows}",
                  f"where {client.program} >nul 2>nul || (",
                  f"  echo {client.name} was installed but is not found: open a new "
                  "window and choose Start assistant in SMAPpy again.",
                  "  exit /b 1)",
                  ":start"]
    lines += ["echo."] + [f"echo {line}" for line in welcome(client)] + ["echo."]
    lines.append(command if isinstance(command, str)
                 else _percent(subprocess.list2cmdline(list(command))))
    return "\r\n".join(lines) + "\r\n"


def _applescript_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def open_terminal(script_file: Path, title: str = "SMAPpy assistant",
                  bounds: Optional[Tuple[int, int, int, int]] = None,
                  run=subprocess.Popen) -> List[str]:
    """Run ``script_file`` in a new terminal window, at ``bounds`` (left,
    top, right, bottom in screen points) where that can be set: on a Mac
    Terminal's bounds, on Linux the place (not the size, which xterm and
    gnome-terminal take in characters) where the terminal takes
    ``--geometry``.  On Windows ``start`` opens the console where Windows
    puts it.  Returns the command used, for the log and the tests."""
    if sys.platform == "darwin":
        lines = ['tell application "Terminal"', "activate",
                 f"set t to do script {_applescript_string(shlex.quote(str(script_file)))}"]
        if bounds is not None:
            lines.append("set bounds of front window to {%d, %d, %d, %d}" % tuple(bounds))
        lines += [f"set custom title of t to {_applescript_string(title)}", "end tell"]
        command = ["osascript", "-e", "\n".join(lines)]
    elif sys.platform.startswith("win"):
        command = ["cmd", "/c", "start", title, "cmd", "/k", str(script_file)]
    else:
        command = _linux_terminal(script_file, title, bounds)
    run(command)
    return command


def raise_terminal(run=subprocess.run) -> bool:
    """The terminal SMAPpy started, brought to the front: on a Mac, Terminal's
    window whose name shows the config file on its command line (the client
    sets its own title over the one given).  False where that cannot be done
    or no such window is found."""
    if sys.platform != "darwin":
        return False
    script = ('tell application "Terminal"\n'
              'set w to (first window whose name contains "mcp-gui.json")\n'
              'set index of w to 1\nactivate\nend tell')
    try:
        done = run(["osascript", "-e", script], capture_output=True, timeout=5)
    except Exception:
        return False
    return getattr(done, "returncode", 1) == 0


#: finds the terminal SMAPpy started: the client's command line, which the
#: window's name shows, names the config file (the client sets its own title)
_TERMINAL_WINDOW = '(first window whose name contains "mcp-gui.json")'


def terminal_bounds(run=subprocess.run) -> Optional[Tuple[int, int, int, int]]:
    """Where the terminal SMAPpy started is, (left, top, right, bottom), for
    *File > Workspace > Save*: on a Mac, and only while it is open.  Asked
    only if Terminal is running, which telling it would start."""
    if sys.platform != "darwin":
        return None
    script = ('if application "Terminal" is running then\n'
              f'tell application "Terminal" to get bounds of {_TERMINAL_WINDOW}\n'
              'end if')
    try:
        done = run(["osascript", "-e", script], capture_output=True, timeout=5, text=True)
        values = [int(v) for v in str(done.stdout).strip().split(",")]
    except Exception:
        return None
    if getattr(done, "returncode", 1) != 0 or len(values) != 4:
        return None
    return tuple(values)


def set_terminal_bounds(bounds: Tuple[int, int, int, int], run=subprocess.run) -> bool:
    """The open terminal SMAPpy started moved to ``bounds``: a workspace
    opened with the terminal still up.  False where it cannot be."""
    if sys.platform != "darwin":
        return False
    script = ('if application "Terminal" is running then\n'
              f'tell application "Terminal" to set bounds of {_TERMINAL_WINDOW} '
              'to {%d, %d, %d, %d}\nend if' % tuple(bounds))
    try:
        done = run(["osascript", "-e", script], capture_output=True, timeout=5)
    except Exception:
        return False
    return getattr(done, "returncode", 1) == 0


#: the terminals whose ``--geometry`` takes a place alone, ``+x+y``
_PLACED = {"gnome-terminal": "--geometry=+{0}+{1}", "xfce4-terminal": "--geometry=+{0}+{1}",
           "xterm": "-geometry +{0}+{1}"}


def _linux_terminal(script_file: Path, title: str, bounds=None) -> List[str]:
    target = ["/bin/sh", str(script_file)]
    for name, args in (("x-terminal-emulator", ["-T", title, "-e"]),
                       ("gnome-terminal", ["--title", title, "--"]),
                       ("konsole", ["-p", f"tabtitle={title}", "-e"]),
                       ("xfce4-terminal", ["-T", title, "-x"]),
                       ("xterm", ["-T", title, "-e"])):
        if shutil.which(name):
            place = (_PLACED[name].format(bounds[0], bounds[1]).split()
                     if bounds is not None and name in _PLACED else [])
            return [name, *place, *args, *target]
    raise RuntimeError("no terminal program found (x-terminal-emulator, gnome-terminal, "
                       "konsole, xfce4-terminal, xterm)")


def add_to_project(folder: Path) -> Path:
    """The server in ``folder``'s ``.mcp.json``, what is there kept: for the
    Claude app, whose Code sessions read it.  Without a process number, so it
    keeps working when SMAPpy is started again."""
    target = folder / ".mcp.json"
    data: Dict = {}
    if target.exists():
        data = json.loads(target.read_text(encoding="utf-8") or "{}")
        if not isinstance(data, dict):
            raise ValueError(f"{target} is not a JSON object; left as it is")
    data.setdefault("mcpServers", {})[NAME] = _server_entry(None)
    target.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return target


def open_app(run=subprocess.Popen) -> List[str]:
    """The Claude app, brought up; the session is started in it by hand."""
    if sys.platform == "darwin":
        command = ["open", "-a", "Claude"]
    elif sys.platform.startswith("win"):
        command = ["cmd", "/c", "start", "", "claude:"]
    else:
        raise RuntimeError("the Claude app is not available on Linux")
    run(command)
    return command


def start(key: str, folder: Path, settings_folder: Path, pid: int, own: str = "",
          pre_allow: bool = True, bounds=None, run=subprocess.Popen) -> str:
    """Start the client ``key`` connected to process ``pid``, working in
    ``folder``.  Returns a line for the user: what was started, or what they
    do next."""
    client = CLIENTS[key]
    folder = Path(folder).expanduser()
    if key == "manual":
        raise ValueError("an assistant started by hand is not started here")
    if key == "claude-app":
        written = add_to_project(folder)
        open_app(run)
        return (f"Claude app opened.  Start a Code session in {folder} and approve "
                f"the '{NAME}' server it finds ({written.name}).")
    config = write_config(settings_folder, pid)
    command = client_command(key, config, own, pre_allow)
    if sys.platform.startswith("win"):
        file = settings_folder / f"start-{key}.cmd"
        # newline="": the lines end in \r\n already, which text mode on
        # Windows wrote as \r\r\n
        file.write_text(windows_script(command, folder, client), encoding="utf-8",
                        newline="")
    else:
        file = settings_folder / f"start-{key}.sh"
        file.write_text(script(command, folder, client), encoding="utf-8")
        file.chmod(0o700)
    open_terminal(file, f"SMAPpy assistant: {client.label}", bounds, run)
    return f"{client.label} started in {folder}."
