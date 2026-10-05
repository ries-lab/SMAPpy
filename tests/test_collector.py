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
