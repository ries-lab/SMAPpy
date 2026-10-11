"""smappy's icon, on every window of the application.

The files are drawn by ``scripts/make_icon.py``: an SVG, and PNGs of it at the
sizes a window manager asks for.  The PNGs are what the icon is built from --
Qt draws an SVG icon only when its SVG image plugin is installed, and a 16 px
rendering picked from a list is crisper than one scaled on the fly.
"""
from __future__ import annotations

from pathlib import Path

ICONS = Path(__file__).resolve().parents[1] / "data" / "icons"


def app_icon():
    from PySide6.QtGui import QIcon
    icon = QIcon()
    for path in sorted(ICONS.glob("smappy_*.png")):
        icon.addFile(str(path))
    return icon


def install_icon(app) -> None:
    """Give ``app`` the icon: its windows and, on macOS, the dock tile."""
    app.setWindowIcon(app_icon())
