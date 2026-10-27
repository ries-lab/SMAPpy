"""The look of every window: Qt's Fusion style, light or dark.

Fusion rather than the platform's own style, because it is the same on a
Mac, on Windows and on Linux -- a screenshot in a tutorial or a page looks
like what the reader has -- and because its boxes show the nesting of a
plugin's parts and their *more* plainly.  It was chosen over the macOS style
and over variants that marked the levels with colour bands (2026-10).

Light, dark, or as the operating system is set (*system*, the default),
in Preferences > General.  Both palettes are set here in full, not left to
the platform: Qt's own dark palette differs by platform (on a Mac its *mid*,
which the fields' borders and disabled text use, is darker than the window),
`QStyleHints.setColorScheme` exists only from Qt 6.8 and is ignored where
the platform has no theme -- the tests' offscreen one -- so a palette of our
own is the one way the choice looks the same everywhere.
"""
from __future__ import annotations

from typing import Dict, Optional

from .. import config

THEMES = ("system", "light", "dark")
KEY = "theme"

# Qt's Fusion palette as it is where no platform theme changes it, which is
# what "light" looked like before there was a choice
LIGHT = {"Window": "#efefef", "WindowText": "#000000", "Base": "#ffffff",
         "AlternateBase": "#f7f7f7", "ToolTipBase": "#ffffdc", "ToolTipText": "#000000",
         "PlaceholderText": "#8a8a8a", "Text": "#000000", "Button": "#efefef",
         "ButtonText": "#000000", "BrightText": "#ffffff", "Light": "#ffffff",
         "Midlight": "#cacaca", "Mid": "#b8b8b8", "Dark": "#9f9f9f", "Shadow": "#767676",
         "Highlight": "#308cc6", "HighlightedText": "#ffffff", "Link": "#0000ff",
         "LinkVisited": "#ff00ff"}
LIGHT_DISABLED = {"WindowText": "#bebebe", "Text": "#bebebe", "ButtonText": "#bebebe",
                  "Base": "#efefef", "Highlight": "#919191"}

# The same roles a step at a time in the other direction: the fields (*base*)
# darker than the window, *mid* -- borders, disabled fields -- lighter than
# it so that a field's edge still shows, and a link light enough to read
DARK = {"Window": "#2b2d30", "WindowText": "#e4e4e4", "Base": "#1e1f22",
        "AlternateBase": "#303236", "ToolTipBase": "#3a3d41", "ToolTipText": "#e4e4e4",
        "PlaceholderText": "#7d8085", "Text": "#e4e4e4", "Button": "#36393d",
        "ButtonText": "#e4e4e4", "BrightText": "#ffffff", "Light": "#4a4d52",
        "Midlight": "#3e4145", "Mid": "#62656a", "Dark": "#1a1b1d", "Shadow": "#000000",
        "Highlight": "#3d7bd9", "HighlightedText": "#ffffff", "Link": "#6aa6ff",
        "LinkVisited": "#c38cff"}
DARK_DISABLED = {"WindowText": "#6d7075", "Text": "#6d7075", "ButtonText": "#6d7075",
                 "Base": "#2b2d30", "Highlight": "#4a4d52"}

# what the colours baked into widgets' own style sheets were made from, so a
# switch can find them again (`_rebake`)
_baked: Optional[str] = None
_applying = False


def theme() -> str:
    """The choice saved in the config; anything unknown is *system*."""
    name = config.get(KEY, "system")
    return name if name in THEMES else "system"


def set_theme(name: str) -> None:
    """Save the choice and show it at once, in every open window."""
    if name not in THEMES:
        raise ValueError(f"theme is one of {', '.join(THEMES)}, not {name!r}")
    config.set(KEY, name)
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is not None:
        apply_theme(app, name)


def palette_of(name: str):
    """The QPalette of "light" or "dark"."""
    from PySide6.QtGui import QColor, QPalette
    colours, disabled = (DARK, DARK_DISABLED) if name == "dark" else (LIGHT, LIGHT_DISABLED)
    palette = QPalette()
    for role, colour in colours.items():
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(colour))
    for role, colour in disabled.items():
        palette.setColor(QPalette.ColorGroup.Disabled, getattr(QPalette.ColorRole, role),
                         QColor(colour))
    return palette


def system_is_dark(app) -> bool:
    """Whether the operating system is set to dark; False where Qt cannot
    tell (before Qt 6.5, or a platform with no theme)."""
    from PySide6.QtCore import Qt
    hints = app.styleHints()
    scheme = getattr(hints, "colorScheme", None)
    return scheme is not None and scheme() == Qt.ColorScheme.Dark


def resolved(app, name: str) -> str:
    """"light" or "dark": *system* read from the operating system."""
    if name == "system":
        return "dark" if system_is_dark(app) else "light"
    return name


def install_theme(app) -> None:
    """Fusion, in the saved theme, and following the operating system while
    that is the choice.  Called once per application (`gui.make_app`)."""
    from PySide6.QtWidgets import QStyleFactory
    if app.property("smappy_theme_installed"):
        return
    app.setProperty("smappy_theme_installed", True)
    app.setStyle(QStyleFactory.create("Fusion"))
    apply_theme(app, theme())
    changed = getattr(app.styleHints(), "colorSchemeChanged", None)
    if changed is not None:
        changed.connect(lambda *_: _follow_system(app))


def _follow_system(app) -> None:
    # A signal, not a QTimer.singleShot: an exception here is printed by
    # PySide and goes no further, where one in a single shot is left pending
    if theme() == "system":
        apply_theme(app, "system")


def apply_theme(app, name: str) -> None:
    """Put the palette of ``name`` on the application and on what is open."""
    global _baked, _applying
    if _applying:
        return
    _applying = True
    try:
        _tell_the_platform(app, name)
    finally:
        _applying = False
    from .widgets import note_colour
    before = _baked or note_colour()
    app.setPalette(palette_of(resolved(app, name)))
    # The style sheet's palette(...) references are read when a widget is
    # polished, not when the palette changes: without this the fields kept
    # their white background on a dark window, with white text in them.
    sheet = app.styleSheet()
    app.setStyleSheet("")
    app.setStyleSheet(sheet)
    _baked = note_colour()
    if _baked != before:
        _rebake(app, {before: _baked})


def _tell_the_platform(app, name: str) -> None:
    """The choice to Qt as well (6.8 and later), for what the palette does
    not reach: a Mac's title bars and its file dialogs.  Telling it emits
    `colorSchemeChanged`, which `_applying` keeps from coming back here."""
    from PySide6.QtCore import Qt
    hints = app.styleHints()
    if not hasattr(hints, "setColorScheme"):
        return
    if name == "system":
        hints.unsetColorScheme()
    else:
        hints.setColorScheme(Qt.ColorScheme.Dark if name == "dark"
                             else Qt.ColorScheme.Light)


def _rebake(app, colours: Dict[str, str]) -> None:
    """Rewrite colours that widgets copied into their own style sheets.

    `note_style` writes the secondary grey as a number, made from the palette
    of the moment, into each note's style sheet -- a hundred-odd places.  A
    switch finds that number and puts the new one in its place, rather than
    every one of them listening for the change.
    """
    for widget in app.allWidgets():
        sheet = widget.styleSheet()
        if sheet and any(old in sheet for old in colours):
            for old, new in colours.items():
                sheet = sheet.replace(old, new)
            widget.setStyleSheet(sheet)
