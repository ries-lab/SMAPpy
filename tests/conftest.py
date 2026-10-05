"""Keep the tests off the developer's screen and out of their real settings.

`smappy.config` lives in the platform's config directory, and a GUI test
builds a `ControlWindow`, which opens the GUI file named there on the way up
and writes the window position on the way down.  Without this the suite would
take its plugin roots and tabs from whatever the developer has configured.
"""
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

# Under pytest-xdist (`-n auto`) every worker is a process of its own, and
# OpenBLAS would start a thread per core in each: four workers on four cores
# took 9:36 for what one does in 11:22, and capped at one thread each 5:57.
# Set before numpy is imported, and inherited by the tutorials' subprocesses.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_name, "1")


def pytest_addoption(parser):
    """`timeout` in pyproject.toml belongs to pytest-timeout; without that
    plugin it is declared here, so that it is not an unknown-option warning.
    The faulthandler's dump at `faulthandler_timeout` is what is left then."""
    parser.addoption("--slow", action="store_true",
                     help="also run the tests marked slow: the tutorials, the "
                          "pages' figures and the large numerical checks")
    try:
        import pytest_timeout  # noqa: F401
    except ImportError:
        parser.addini("timeout", "per-test limit in seconds (needs pytest-timeout)")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: seconds or more each; skipped unless --slow is given")


def pytest_collection_modifyitems(config, items):
    """Two tiers.  The tests marked slow are about two thirds of the suite's
    time in a few dozen tests -- the tutorials alone half of it -- and check
    what a change to the GUI, a page or an algorithm breaks rather than what
    every edit might.  They are skipped, not deselected, so the summary says
    how many were left out; `--slow` runs everything, as before a PR."""
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="slow: run with --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


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
    time, so every 100th."""
    global _tests_since_collect
    yield
    _tests_since_collect += 1
    if _tests_since_collect >= 100:
        import gc
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
