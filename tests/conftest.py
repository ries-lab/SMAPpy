"""Keep the tests off the developer's screen and out of their real settings.

`smappy.config` lives in the platform's config directory, and a GUI test
builds a `ControlWindow`, which opens the GUI file named there on the way up
and writes the window position on the way down.  Without this the suite would
take its plugin roots and tabs from whatever the developer has configured.
"""
import importlib.util
import os

import pytest

# Qt and matplotlib draw into nothing unless the developer asks otherwise.  A
# suite that opens windows takes the keyboard away from whatever is in front of
# it, several times, and that is reason enough; it also makes the live-view
# test deterministic, since a real window manager is free to resize the figure
# under it and move the axes the test is checking.  Set either variable
# yourself to watch the windows.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
# The app is Fusion on every platform (`gui/theme.py`), but a test that makes
# its own QApplication gets the platform's style, and Windows 11's widgets are
# far wider: the Spline 3D panel measured 482 px there against 279 under
# Fusion, and 29 tests of widths, wrapping and the tutorials failed on Windows
# for a look the app never shows (October 2026).
os.environ.setdefault("QT_STYLE_OVERRIDE", "Fusion")
# and Qt's offscreen platform on Windows finds no fonts unless told where they
# are: every text drew 2.3 times as wide as in a window, so every panel failed
# its width
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

# Under pytest-xdist (`-n auto`) every worker is a process of its own, and
# OpenBLAS would start a thread per core in each: four workers on four cores
# took 9:36 for what one does in 11:22, and capped at one thread each 5:57.
# Set before numpy is imported, and inherited by the tutorials' subprocesses.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "1")


# uipsf's tests need torch, which is an extra (`pip install -e .[uipsf]`); without it they
# are not collected, so a plain checkout runs the rest of the suite as before.
if importlib.util.find_spec("torch") is None:
    collect_ignore = ["uipsf_tests"]


#: a long test's limit: the region study takes four minutes and more alone
LONG_TIMEOUT = 1200


def pytest_addoption(parser):
    """`timeout` in pyproject.toml belongs to pytest-timeout; without that
    plugin it is declared here, so that it is not an unknown-option warning.
    The faulthandler's dump at `faulthandler_timeout` is what is left then."""
    parser.addoption("--gpu", action="store_true",
                     help="also run uipsf's GPU tier (needs CUDA or Apple's MPS)")
    parser.addoption("--slow", action="store_true",
                     help="also run the tests marked slow: the tutorials, the "
                          "pages' figures and the large numerical checks")
    parser.addoption("--long", action="store_true",
                     help="also run the tests marked long, uipsf's learning studies "
                          "of 20 s to minutes each (implies --slow)")
    try:
        import pytest_timeout  # noqa: F401
    except ImportError:
        parser.addini("timeout", "per-test limit in seconds (needs pytest-timeout)")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: seconds or more each; skipped unless --slow is given")
    config.addinivalue_line(
        "markers", "long: 20 s to minutes each; skipped unless --long is given (uipsf)")
    config.addinivalue_line(
        "markers", "gpu: needs CUDA or Apple's MPS; skipped unless --gpu is given (uipsf)")


def pytest_collection_modifyitems(config, items):
    """Two tiers.  The tests marked slow are about two thirds of the suite's
    time in a few dozen tests -- the tutorials alone half of it -- and check
    what a change to the GUI, a page or an algorithm breaks rather than what
    every edit might.  They are skipped, not deselected, so the summary says
    how many were left out; `--slow` runs everything, as before a PR.

    `--long` is a third, above `--slow` and including it: uipsf's learning
    studies -- a benchmark against the CRLB, a pupil learned in the window,
    one region size against another -- of 20 s to over four minutes each.
    In `--slow` they had taken it from 3.5 to 10.5 minutes on four cores, so
    they are run by hand when uipsf's learning changes, not before every PR.
    Their limit is `LONG_TIMEOUT`, not the 300 s every other test has.

    uipsf's GPU tier is another, `--gpu`: run by hand on a Mac and a CUDA
    machine, since no CI runner has the GPU."""
    if not config.getoption("--gpu"):
        gpu = pytest.mark.skip(reason="gpu tier: run with --gpu")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(gpu)
    long_ = config.getoption("--long")
    if long_:
        try:
            import pytest_timeout  # noqa: F401
        except ImportError:
            pass
        else:
            for item in items:
                if "long" in item.keywords and item.get_closest_marker("timeout") is None:
                    item.add_marker(pytest.mark.timeout(LONG_TIMEOUT))
    else:
        skip = pytest.mark.skip(reason="long: run with --long")
        for item in items:
            if "long" in item.keywords:
                item.add_marker(skip)
    if long_ or config.getoption("--slow"):
        # longest first: collected last (`test_tutorial`), the tutorials were
        # a tail of 20-50 s tests that four workers sat through after
        # everything else had finished
        items.sort(key=lambda item: ("long" not in item.keywords,
                                     "slow" not in item.keywords))
        return
    skip = pytest.mark.skip(reason="slow: run with --slow")
    for item in items:
        if "slow" in item.keywords and "long" not in item.keywords:
            item.add_marker(skip)


_exitstatus = 0


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    global _exitstatus
    _exitstatus = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """A process that loaded Qt leaves without Python's teardown.

    After the run, Python frees the windows the GUI tests left in whatever
    order it reaches them, and PySide 6.8 with pyqtgraph does not survive
    every order: a main window dropped mid-cascade through a pyqtgraph scene,
    a slot's weak reference firing into PySide's connection table, or
    `Py_FinalizeEx` clearing that table under connections to objects already
    gone.  Each was a segfault after the results were in -- the run green,
    and on a Mac a "Python quit unexpectedly" for about one worker a full
    `--slow` run.  Deleting the windows first, with `deleteLater`, is worse:
    pyqtgraph's layouts crash in Qt's order too, in every worker.  So this is
    pyqtgraph's own remedy (`pg.exit`): everything is reported by now -- an
    xdist worker sent its "workerfinished", flushed, in `pytest_sessionfinish`
    -- and the process ends with the run's status and nothing freed.
    """
    import sys
    if "PySide6.QtCore" not in sys.modules:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exitstatus)


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_make_scheduler(config, log):
    """Work stealing, unless `--dist` asked for something else.

    xdist's default hands each worker a contiguous run of the list, and the
    tutorials -- twelve tests of 15-50 s -- landed on one worker: 372 s for it
    while the other three idled after ~100 s, and `--slow -n 4` took 6:50.
    Stealing evens it out: 3:53.
    """
    if config.getoption("dist") != "load":
        return None
    from xdist.scheduler import WorkStealingScheduling
    return WorkStealingScheduling(config, log)


@pytest.fixture(autouse=True, scope="session")
def collect_on_the_main_thread():
    """Garbage is collected here, after each test, and never by chance.

    A GUI test that drops a widget in a reference cycle leaves it for the
    cyclic collector, which runs in whichever thread happens to allocate next
    -- often a fit's reader thread in a later test.  A QWidget destroyed there
    closes its window, and `QWindow::close` waits for the GUI thread to flush
    window-system events: the GUI thread is the test, blocked waiting for the
    fit, and the worker hangs.  That was `test_live` failing in about half of
    the parallel `--slow` runs, and the 300 s timeouts in the 3D view and
    render-axes tests.  Collecting only on the main thread is pyqtgraph's
    `GarbageCollector` remedy for the same thing.
    """
    import gc
    gc.disable()
    yield
    gc.enable()


_tests_since_collect = 0


@pytest.fixture(autouse=True)
def _collect_after_each_test(collect_on_the_main_thread):
    """It is `gc.disable` that keeps collection off the other threads; how
    often this runs is only memory.  After every test it doubled the suite's
    time, so every 100th.

    The app's entry points hold its screens before they collect
    (`collector.keep_screens`); a test makes its own QApplication, so the
    screens are held here, or collecting two matplotlib windows deletes the
    `QScreen` and the next window to hide segfaults."""
    global _tests_since_collect
    yield
    _tests_since_collect += 1
    if _tests_since_collect >= 100:
        import gc
        import sys
        if "PySide6.QtWidgets" in sys.modules:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                from smappy.gui.collector import keep_screens
                keep_screens(app)
        gc.collect()
        _tests_since_collect = 0


@pytest.fixture(autouse=True, scope="session")
def isolated_config(tmp_path_factory):
    import os
    directory = tmp_path_factory.mktemp("config")
    os.environ["SMAPPY_CONFIG_DIR"] = str(directory)
    from smappy import config, plugins
    config.load(reload=True)
    plugins.discover(force=True)
    yield directory
