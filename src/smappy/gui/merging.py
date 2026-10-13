"""Merging blinks in the background, so that a file opens at once.

A layer shows merged blinks by default, and every layer shares one grouped
table (`layer_state.Tables`).  Headless, it is linked when it is first read.
In the GUI that read is on the GUI thread, and on a large table it is
seconds -- 5 M localizations at 500 a frame take 4.5-6 s, 57 M about 24 s --
so the window does not do it there.  A table of `IN_PLACE` rows or more is
*deferred* (`Tables.defer_from`): a layer set to blinks shows the
localizations until they are merged, and this module merges them in a
worker, with the stage and a clock in the status bar, then tells the session
("regrouped") and the layers switch.  It starts whenever the session has a
layer set to blinks and no current grouped table: right after a file opens,
when *merge blinks* is ticked, when the table or the merge settings change.
A smaller table is linked in place on first read, as headless: a few
hundred milliseconds are not worth a picture that changes twice.

The worker links a table nobody writes to (`Tables.link_job`: the table
itself, or a snapshot of a live one) with ``attach_columns=False``, and the
GUI thread keeps the answer (`Tables.take_grouped`), which writes `group_id`
and `n_in_group` back: a column added to the table while a render thread
reads it is a race the worker has no business starting.  If the table or the
merge settings changed meanwhile the answer is of nothing current, and the
merge starts again for what is there now.  The answer is kept whether or not
a layer still wants it: unticking *merge blinks* while it runs costs nothing
later.
"""
from __future__ import annotations

import traceback
import weakref
from typing import Optional

from PySide6.QtCore import QCoreApplication, QElapsedTimer, QObject, QThread, QTimer, Signal

# Below this many localizations a merge is a fraction of a second (2 M take
# 1.4 s), and is done in place, on first read, rather than in a worker.
IN_PLACE = 300_000

# The merges running, held here rather than by a widget: a window closed
# would take a running thread down with it, which aborts the program.
# Waited for when the application quits, for the same reason.
_RUNNING: set = set()

# one merger per session, whichever window made it first
_MERGERS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _wait_for_merges() -> None:
    for task in list(_RUNNING):
        task.wait()


def merger_for(session, parent: QObject) -> "Merger":
    """The session's merger: the one there is, or a new one owned by
    ``parent`` (whose window's status bar it reports in)."""
    from shiboken6 import isValid
    found = _MERGERS.get(session)
    if found is not None and isValid(found):
        return found
    made = _MERGERS[session] = Merger(session, parent)
    return made


class MergeTask(QThread):
    """`group.group` on a worker: the grouped table and the group index."""

    progress = Signal(str)
    done = Signal(object, object)                 # grouped, group index
    failed = Signal(str, str)                     # the message; the traceback

    def __init__(self, locs, settings, parent=None):
        super().__init__(parent)
        self.locs, self.settings = locs, settings

    def run(self) -> None:
        from ..group import group
        try:
            grouped, index = group(self.locs, self.settings, attach_columns=False,
                                   progress=lambda text, _f: self.progress.emit(text))
            self.done.emit(grouped, index)
        except Exception as e:                    # said in the status bar
            self.failed.emit(str(e) or type(e).__name__, traceback.format_exc())


class Merger(QObject):
    """Merges one session's blinks in a worker whenever a layer wants them
    and the session's table is large and not merged (`ensure`)."""

    def __init__(self, session, parent):
        super().__init__(parent)
        self.session = session
        self._task: Optional[MergeTask] = None
        self._tables = None
        self._key = None
        self._failed_key = None           # not tried again for the same rows
        self._stage = ""
        self._elapsed = QElapsedTimer()
        self._ticker = QTimer(self)
        self._ticker.timeout.connect(self._tick)
        app = QCoreApplication.instance()
        if app is not None and not getattr(app, "_smappy_waits_for_merges", False):
            app.aboutToQuit.connect(_wait_for_merges)
            app._smappy_waits_for_merges = True
        # from now on a large table of this session is not linked by reading it
        session.defer_merge_from = IN_PLACE
        session.tables.defer_from = IN_PLACE
        session.on_change(self._on_session)
        self.ensure()

    @property
    def busy(self) -> bool:
        return self._task is not None

    def _on_session(self, what: str) -> None:
        from shiboken6 import isValid
        if not isValid(self):                     # its window is gone
            self.session.off_change(self._on_session)
            return
        if what in ("locs", "layer", "layers", "regrouped"):
            try:
                self.ensure()
            except Exception:                     # a listener must not raise
                traceback.print_exc()

    def wanted(self) -> bool:
        """Whether merging now is due: a layer is set to blinks, and the
        table is one whose merge is deferred to a worker."""
        tables = self.session.tables
        if tables.growing or tables.is_current or not tables.deferred():
            return False
        if "frame" not in tables.locs and "tid" not in tables.locs:
            return False
        return any(l.state.use_grouped for l in self.session.layers if not l.is_image)

    def ensure(self) -> None:
        """Start a merge if one is due and none is running; a running one
        that turns out to be of rows no longer there starts again when it
        is back (`_done`)."""
        if self._task is not None or not self.wanted():
            return
        job = self.session.tables.link_job()
        if job is None or (job[2], id(self.session.tables)) == self._failed_key:
            return
        self._start(job)

    def _start(self, job) -> None:
        tables = self.session.tables
        locs, settings, self._key = job
        self._tables = tables
        tables.merging = True
        task = self._task = MergeTask(locs, settings)     # no parent: `_RUNNING`
        _RUNNING.add(task)
        task.progress.connect(self._progress)
        task.done.connect(self._done)
        task.failed.connect(self._failed)
        task.finished.connect(lambda: _RUNNING.discard(task))
        task.finished.connect(task.deleteLater)
        self._elapsed.restart()
        self._stage = "merging blinks"
        self._ticker.start(500)
        self._tick()
        task.start()

    def _progress(self, text: str) -> None:
        self._stage = f"merging blinks: {text}"
        self._tick()

    def _tick(self) -> None:
        try:
            seconds = self._elapsed.elapsed() // 1000
            self._say(f"{self._stage}... ({seconds // 60}:{seconds % 60:02d})")
        except Exception:                         # a timer slot must never raise
            traceback.print_exc()

    def _say(self, text: str, timeout: int = 0) -> None:
        parent = self.parent()
        window = parent.window() if parent is not None else None
        bar = getattr(window, "statusBar", None)
        if bar is not None:
            bar().showMessage(text, timeout)

    def _stop(self) -> None:
        self._ticker.stop()
        self._task = None
        if self._tables is not None:
            self._tables.merging = False
        self._tables = None

    def _done(self, grouped, index) -> None:
        try:
            tables = self._tables
            self._stop()
            kept = tables is self.session.tables and \
                tables.take_grouped(grouped, index, self._key)
            if kept:
                seconds = self._elapsed.elapsed() / 1000
                self._say(f"{len(grouped):,} blinks from {len(tables.locs):,} "
                          f"localizations ({seconds:.1f} s)", 8000)
                self.session.changed("regrouped")     # the layers switch
            self.ensure()                 # the table or the settings moved: again
        except Exception:                 # a slot must not raise into Qt
            traceback.print_exc()

    def _failed(self, message: str, details: str) -> None:
        try:
            print(details)
            self._failed_key = (self._key, id(self._tables))
            self._stop()
            self._say(f"could not merge blinks: {message}", 15000)
        except Exception:
            traceback.print_exc()
