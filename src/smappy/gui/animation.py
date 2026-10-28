"""What Space starts and stops: the animation used last.

Two things move the pictures by themselves: the Render tab's player
(`gui.scan_player`) and the 3D window's ⟳ (`gui.view3d.Motion`).  Space
in the render window and in the 3D window plays or pauses whichever of
the two was used last -- opening the player, playing it, moving its slider
or scrubbing it is using it; starting the ⟳ is using that.  Until the
player has been used, Space in the 3D window rocks the view as it always
did, and in the render window it rocks a 3D window that is open.

A window shortcut and not a key press of the picture: a button or a tick
that has the focus takes Space for itself, and the picture need not have
the focus.  It is installed on the render and the 3D windows only, so the
ROI window's and the bead window's own Space (which tick the selected
entry) stay theirs.

One record per session, so two sessions in one process (*Open in a new
window* within a test, say) keep their own.
"""
from __future__ import annotations

import weakref
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut

PLAYER = "player"
MOTION = "motion"


class Animations:
    """The session's player and ⟳, and which was used last."""

    def __init__(self) -> None:
        self.player = None          # the Render tab's `ScanPlayer`
        self.motion = None          # the 3D window's `Motion`
        self.last: Optional[str] = None

    def used(self, which: str) -> None:
        self.last = which

    def set_player(self, player) -> None:
        """The player Space and the picture's Alt+wheel move; a Render tab
        built again (a workspace loaded) hands its new one."""
        self.player = player
        player.used.connect(lambda: self.used(PLAYER))

    def set_motion(self, motion) -> None:
        self.motion = motion
        motion.running.connect(lambda on: on and self.used(MOTION))

    def live_player(self):
        return self.player if _alive(self.player) else None

    def _motion(self, here=None):
        """The ⟳ of the window Space was pressed in, else of a 3D window
        that is showing: one rocked behind a closed window is not seen."""
        if here is not None:
            return here
        motion = self.motion if _alive(self.motion) else None
        if motion is not None and motion.view.isVisible():
            return motion
        return None

    def space(self, motion=None) -> None:
        """Play or pause the animation used last; ``motion`` is the ⟳ of
        the 3D window Space was pressed in."""
        player = self.live_player() if self.last == PLAYER else None
        if player is not None:
            player.toggle_play()
            return
        motion = self._motion(motion)
        if motion is not None:
            motion.toggle()
        elif self.live_player() is not None and self.last is not None:
            self.live_player().toggle_play()


_records: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def animations(session) -> Animations:
    record = _records.get(session)
    if record is None:
        record = _records[session] = Animations()
    return record


def install_space(window, session, motion=None) -> None:
    """Space in ``window`` plays or pauses the animation used last (once
    per window, however often it is asked)."""
    if getattr(window, "_animation_space", None) is not None:
        return
    shortcut = QShortcut(QKeySequence(Qt.Key_Space), window)
    shortcut.setContext(Qt.WindowShortcut)
    record = animations(session)
    shortcut.activated.connect(lambda: _space(record, motion))
    window._animation_space = shortcut


def _space(record: Animations, motion) -> None:
    # a slot Qt calls: what it raises is printed, never left to Qt
    try:
        record.space(motion)
    except Exception:
        import traceback
        traceback.print_exc()


def _alive(obj) -> bool:
    if obj is None:
        return False
    try:
        from shiboken6 import isValid
        return isValid(obj)
    except ImportError:          # pragma: no cover - shiboken ships with PySide
        return True
