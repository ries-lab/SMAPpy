"""Every window on its screen: no larger than it, and not off its edge.

Windows size themselves from what they hold -- a fitter's preview from its
figure, a dialog from a long list, a plot from its size hint in inches -- and
on a laptop screen that came out taller than the screen, with the window's
bottom edge, and its resize corner, out of reach.  Rather than each window
measuring the screen, one application-wide filter (`install_screen_fit`, from
`gui.make_app`) shrinks any window that would be larger than the free part of
its screen, and on showing moves it back onto the screen.

Shrinking only: a window is never grown, and one dragged partly off the
screen by hand stays there until it is shown again.
"""
from __future__ import annotations

import traceback

_installed = None


def fit_to_screen(window, move: bool = True) -> None:
    """Shrink ``window`` to its screen's free area and, with ``move``, put
    it back on the screen where it sticks out."""
    from PySide6.QtWidgets import QApplication
    screen = window.screen() or QApplication.primaryScreen()
    if screen is None:
        return
    room = screen.availableGeometry()
    frame = window.frameGeometry()
    # the title bar and borders; before the window is first shown there are
    # none yet, and a title bar's height is kept free anyway
    dw = max(frame.width() - window.width(), 0)
    dh = max(frame.height() - window.height(), 0) or 28
    width = min(window.width(), room.width() - dw)
    height = min(window.height(), room.height() - dh)
    if (width, height) != (window.width(), window.height()):
        window.resize(max(width, 1), max(height, 1))
    if not move:
        return
    frame = window.frameGeometry()
    x = min(max(frame.x(), room.left()), max(room.left(), room.right() + 1 - frame.width()))
    y = min(max(frame.y(), room.top()), max(room.top(), room.bottom() + 1 - frame.height()))
    if (x, y) != (frame.x(), frame.y()):
        window.move(x, y)


def install_screen_fit(app) -> None:
    """The filter on ``app``; safe to call again."""
    global _installed
    if _installed is None:
        _installed = _make_filter(app)
        app.installEventFilter(_installed)


def _make_filter(app):
    """The filter, its class made here so that importing this costs no Qt."""
    from PySide6.QtCore import QEvent, QObject, Qt, QTimer
    from PySide6.QtWidgets import QWidget

    # windows proper: a popup or a tooltip places itself (`widgets.Popup`)
    kinds = (Qt.Window, Qt.Dialog, Qt.Tool, Qt.Sheet)

    class ScreenFit(QObject):
        def _later(self, window, move: bool) -> None:
            # after the event: a window resized from inside its own Show or
            # Resize is resized again by the platform once the event is through
            def fit():
                try:            # a singleShot callable must never raise
                    if window.isVisible():
                        fit_to_screen(window, move)
                except RuntimeError:
                    pass        # deleted meanwhile
                except Exception:
                    traceback.print_exc()
            QTimer.singleShot(0, fit)

        def eventFilter(self, obj, event) -> bool:
            # every event of the application passes here: the type first,
            # nothing else for the rest; and never raise (`gui.identity`)
            try:
                kind = event.type()
                if kind in (QEvent.Type.Show, QEvent.Type.Resize) \
                        and isinstance(obj, QWidget) and obj.isWindow() \
                        and obj.windowType() in kinds:
                    if kind == QEvent.Type.Show:
                        self._later(obj, True)
                    elif obj.isVisible():
                        self._later(obj, False)
            except Exception:
                traceback.print_exc()
            return False

    return ScreenFit(app)
