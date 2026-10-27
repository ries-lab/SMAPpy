"""``smappy-gui --desktop-icon``: an icon on the desktop that starts SMAPpy, with SMAPpy's own picture.

The install guide used to make the Windows shortcut with one line of PowerShell, which set the target
and nothing else, so the desktop showed the launcher's generic program icon (October 2026).  The icon
lives inside the installed package, whose folder differs between a uv tool, a venv and conda, so the
installed SMAPpy is what knows where it is -- hence a command of its own rather than a longer line.

The icon starts this Python with ``-m smappy.cli.gui``, which is the ``smappy-gui`` command itself,
without depending on where a launcher was put.  It keeps the console window: its output is how a
problem at start-up is seen and reported.

* Windows: ``SMAPpy.lnk`` on the desktop with ``smappy.ico``, made through the shell's own COM object
  in PowerShell, so nothing is needed beyond Windows; the desktop is the one Windows names (OneDrive
  moves it).
* Linux: ``smappy.desktop`` in ``~/.local/share/applications``, which puts it in the applications
  menu, and a copy on the desktop when there is one.
* macOS: ``SMAPpy.command`` on the desktop, as the guide made before; it keeps Terminal's icon, since
  setting another needs AppKit.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

ICONS = Path(__file__).resolve().parent / "data" / "icons"


def _quote_ps(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def windows_script(python: str, icon: Path, folder: Path | None = None) -> str:
    """The PowerShell that makes the shortcut in ``folder`` (the desktop) and prints its path."""
    where = _quote_ps(folder) if folder else "([Environment]::GetFolderPath('Desktop'))"
    return "; ".join([
        # the path printed back in UTF-8: in the ANSI code page a desktop in a
        # user folder named with an umlaut came back garbled
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8",
        f"$p = Join-Path {where} 'SMAPpy.lnk'",
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($p)",
        f"$s.TargetPath = {_quote_ps(python)}",
        "$s.Arguments = '-m smappy.cli.gui'",
        f"$s.IconLocation = {_quote_ps(str(icon) + ',0')}",
        "$s.WorkingDirectory = $HOME",
        "$s.Description = 'SMAPpy: single-molecule localization microscopy'",
        "$s.Save()",
        "Write-Output $p"])


def desktop_entry(python: str, icon: Path) -> str:
    """A freedesktop.org launcher; ``%F`` passes files dropped on it to the GUI."""
    return "\n".join([
        "[Desktop Entry]", "Type=Application", "Name=SMAPpy",
        "Comment=Single-molecule localization microscopy",
        f'Exec="{python}" -m smappy.cli.gui %F', f"Icon={icon}",
        "Terminal=false", "Categories=Science;Education;", ""])


def make(home: Path | None = None) -> Path:
    """Make the icon for this machine and return where it is; ``home`` stands in for the user's
    folder (on Windows, for the desktop itself), so a test writes nowhere real."""
    python = sys.executable
    system = platform.system()
    if system == "Windows":
        done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                               windows_script(python, ICONS / "smappy.ico", home)],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        if done.returncode != 0:
            raise RuntimeError(done.stderr.strip() or "PowerShell could not make the shortcut")
        return Path(done.stdout.strip().splitlines()[-1])
    home = home or Path.home()
    if system == "Darwin":
        path = home / "Desktop" / "SMAPpy.command"
        path.write_text(f'#!/bin/sh\nexec "{python}" -m smappy.cli.gui "$@"\n', encoding="utf-8")
        path.chmod(0o755)
        return path
    entry = desktop_entry(python, ICONS / "smappy_256.png")
    apps = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share")) / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    path = apps / "smappy.desktop"
    path.write_text(entry, encoding="utf-8")
    path.chmod(0o755)
    desktop = home / "Desktop"
    if desktop.is_dir():
        (desktop / "smappy.desktop").write_text(entry, encoding="utf-8")
        (desktop / "smappy.desktop").chmod(0o755)
        return desktop / "smappy.desktop"
    return path
