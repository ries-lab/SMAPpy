import gc
import threading
import time
import weakref

import pytest


@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def collector(app):
    from smappy.gui import collector
    was_enabled = gc.isenabled()
    old, collector._timer = collector._timer, None      # a fresh install, quickly
    timer = collector.collect_on_gui_thread(app, interval_ms=20)
    yield collector
    timer.stop()
    collector._timer = old
    if was_enabled:
        gc.enable()


class Cycle:
    """Freed only by the cyclic collector; says on which thread it went."""

    def __init__(self, freed_on):
        self.me = self
        weakref.finalize(self, lambda: freed_on.append(threading.current_thread()))


def test_a_cycle_is_freed_on_the_gui_thread_however_much_a_worker_allocates(app, collector):
    assert not gc.isenabled()
    freed_on = []
    Cycle(freed_on)

    def allocate():                     # enough to trip every generation's threshold
        for _ in range(200_000):
            [[]]
    worker = threading.Thread(target=allocate)
    worker.start()
    worker.join()
    assert freed_on == []               # the worker's allocations collected nothing

    end = time.monotonic() + 2.0
    while not freed_on and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    assert freed_on == [threading.main_thread()]


def test_installing_twice_keeps_the_first_timer(app, collector):
    assert collector.collect_on_gui_thread(app) is collector._timer


SCREEN_SCRIPT = r"""
import gc, sys
from PySide6.QtCore import QEvent
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QWidget
from smappy.gui.collector import keep_screens
app = QApplication([])
gc.disable()
keep_screens(app)
# a window Qt deletes invalidates the screen's wrapper it handed out
gone = QWidget()
gone.show()
app.processEvents()
gone.windowHandle().screen()
gone.deleteLater()
app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
for _ in range(2):
    window = QWidget()
    window.show()
    app.processEvents()
    window.windowHandle().screen()      # what matplotlib's canvas asks on show
    window.cycle = window               # only the collector frees it
del window
if sys.argv[1] == "keep":
    keep_screens(app)                   # as the collector does before collecting
gc.collect()                            # hides both windows: QCursor::pos
print(QCursor.pos(), repr(app.primaryScreen().name()), "alive")
"""


def run_screen_script(mode):
    import os
    import subprocess
    import sys
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    return subprocess.run([sys.executable, "-c", SCREEN_SCRIPT, mode], env=env,
                          capture_output=True, text=True, timeout=120)


def test_collecting_two_windows_that_asked_for_their_screen_keeps_the_screen():
    """PySide deleted the C++ QScreen when the collector freed two shown
    widgets whose window had handed back its screen, and the next window to
    hide segfaulted in `QCursor::pos` -- one xdist worker a full run on
    PySide6 6.9 and 6.10, and the app collects on a timer too.  The hold is
    renewed before the collection, because a window deleted by Qt takes the
    wrapper that was held with it.  In a process of its own, because the
    failure is a segfault."""
    pytest.importorskip("PySide6")
    result = run_screen_script("keep")
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("alive")
