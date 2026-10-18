"""Merging blinks in the background, so that a file opens at once.

A layer shows merged blinks by default, and every layer shares one grouped
table (`layer_state.Tables`).  Headless, it is linked when it is first read.
In the GUI that read is on the GUI thread, and on a large table it is
seconds -- 5 M localizations at 500 a frame take 4.5-6 s, 57 M about 24 s --
so the window does not do it there.  A table of `IN_PLACE` rows or more is
*deferred* (`Tables.defer_from`): a layer set to blinks draws nothing until
they are merged -- not the localizations, which would be taken for blinks --
and the picture says why (`RenderView.merge_hint`); this module merges them
in a worker, with the stage and a clock in the status bar (the walk is one
call into C++ with nothing to count, so no percentage), then tells the
session ("regrouped") and the layers draw the blinks.  It starts whenever the session has a
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
a layer still wants it: unticking *merge blinks* while it runs shows the
localizations at once, the merge finishes anyway, and ticking it again costs
nothing.

**Analysis waits.**  A run that reads the session's localizations
(`plugins.reads_locs`: everything but a fit, a loader, a simulation, or a
chain starting with one) is not started on a table whose blinks are being
merged: the panel says so and starts it when the merge is done (`after`),
and Stop gives up waiting.  A chain then shares the session's merged table
(`Tables.sharing`) instead of merging on its worker.  A follow-ROI
measurement and the ROI manager's live evaluation wait the same way.  Just
before a run starts, a small table a layer shows as blinks is linked on the
GUI thread (`prepare`), so no worker links it: linking writes `group_id`
and `n_in_group` onto the session's table.
"""
from __future__ import annotations

import traceback
import weakref
from typing import Callable, List, Optional, Tuple

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
    and the session's table is large and not merged (`ensure`), and holds
    back what must not run on unmerged data until it is done (`after`).

    ``message(text, timeout)`` is for a status bar, ``status(text)`` for
    the picture's overlay and for a panel whose run waits: the stage and a
    clock while a merge runs, "" when none does.
    """

    message = Signal(str, int)
    status = Signal(str)

    def __init__(self, session, parent):
        super().__init__(parent)
        self.session = session
        self._task: Optional[MergeTask] = None
        self._tables = None
        self._key = None
        self._rows = 0
        self._failed_key = None           # not tried again for the same rows
        self._stage = ""
        self._text = ""
        # (token, then, failed): what waits for the merge (`after`)
        self._waiting: List[Tuple[int, Callable[[], None],
                                  Optional[Callable[[str], None]]]] = []
        self._tokens = 0
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

    @property
    def text(self) -> str:
        """What the merge is doing now, with its clock; "" when idle."""
        return self._text

    def pending(self) -> bool:
        """Whether a merge runs or is about to: what a run waits for."""
        return self.busy or self.wanted()

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
        if tables.merge_failed:
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
        if job is None:
            return
        self._start(job)

    def prepare(self) -> None:
        """Before a run reads the tables, on this thread: a small table a
        layer shows as blinks is linked here, so that no worker links it
        and writes its columns onto the session's table."""
        tables = self.session.tables
        if any(l.state.use_grouped for l in self.session.layers if not l.is_image):
            tables.available()

    def after(self, then: Callable[[], None],
              failed: Optional[Callable[[str], None]] = None) -> Optional[int]:
        """``then()`` once no merge is pending -- now, if none is, and None
        is returned; otherwise when it is done, and a token for `cancel`.
        ``failed(message)`` instead if the merge fails."""
        self.ensure()
        if not self.pending():
            then()
            return None
        self._tokens += 1
        self._waiting.append((self._tokens, then, failed))
        return self._tokens

    def cancel(self, token: Optional[int]) -> None:
        """Stop waiting: what `after` was given is not called."""
        self._waiting = [w for w in self._waiting if w[0] != token]

    def _start(self, job) -> None:
        tables = self.session.tables
        locs, settings, self._key = job
        self._rows = len(locs)          # one file of several, or the table
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
        # the walk is one call into C++ with nothing to count, so the stages
        # (sorting, linking, combining n/12) and a clock, not a percentage
        self._stage = f"merging blinks: {text}"
        self._tick()

    def _tick(self) -> None:
        try:
            seconds = self._elapsed.elapsed() // 1000
            self._text = f"{self._stage}... ({seconds // 60}:{seconds % 60:02d})"
            self.message.emit(self._text, 0)
            self.status.emit(self._text)
        except Exception:                         # a timer slot must never raise
            traceback.print_exc()

    def _stop(self) -> None:
        self._ticker.stop()
        self._task = None
        self._text = ""
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
                self.message.emit(f"{len(grouped):,} blinks from {self._rows:,} "
                                  f"localizations ({seconds:.1f} s)", 8000)
            self.ensure()                 # the table or the settings moved: again
            if not self.busy:
                self.status.emit("")
            if kept:
                self.session.changed("regrouped")     # the layers switch
            if not self.pending():
                waiting, self._waiting = self._waiting, []
                for _token, then, _failed in waiting:
                    _call(then)
        except Exception:                 # a slot must not raise into Qt
            traceback.print_exc()

    def _failed(self, message: str, details: str) -> None:
        try:
            print(details)
            tables = self._tables
            self._stop()
            if tables is not None:
                # the layers show the localizations rather than nothing
                tables.merge_failed = True
            self.message.emit(f"could not merge blinks: {message}", 15000)
            self.status.emit("")
            self.session.changed("regrouped")
            waiting, self._waiting = self._waiting, []
            for _token, _then, failed in waiting:
                if failed is not None:
                    _call(failed, message)
        except Exception:
            traceback.print_exc()


def _call(fn, *args) -> None:
    """A callback that raises is printed, not passed into Qt's event loop."""
    try:
        fn(*args)
    except Exception:
        traceback.print_exc()
