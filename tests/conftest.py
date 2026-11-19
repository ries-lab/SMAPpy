"""Keep the tests off the developer's screen and out of their real settings.

`smappy.config` lives in the platform's config directory, and a GUI test
builds a `ControlWindow`, which opens the GUI file named there on the way up
and writes the window position on the way down.  Without this the suite would
take its plugin roots and tabs from whatever the developer has configured.
"""
import contextlib
import importlib.util
import os
import sys
import time

import pytest

# The tests test the tree they are in.  SMAPpy is installed editable from one checkout, so in any
# other -- a session's worktree (CLAUDE.md, "Work in a worktree of your own") -- `import smappy`
# loaded that checkout's code and the tests passed or failed on what was not being changed (found
# 2026-11-19 in .claude/worktrees/roi-angle).  So this tree's `src` goes first, for this process and,
# through PYTHONPATH, for the tests' subprocesses; and its compiled extensions must be its own and
# built since `csrc/` last changed, or the run stops and says how to build them.
_TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_TREE, "src")
if sys.path[:1] != [_SRC]:
    sys.path.insert(0, _SRC)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [_SRC] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != _SRC])


def _stale_extensions():
    """The extension modules of `setup.py` missing from this tree's `src/smappy`, or older than their
    own sources or any header in `csrc/` (setup.py's ``depends``): what `python setup.py build_ext
    --inplace` would build."""
    import importlib.machinery
    import re
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    csrc = os.path.join(_TREE, "csrc")
    headers = max((os.path.getmtime(os.path.join(csrc, f)) for f in os.listdir(csrc) if f.endswith(".hpp")),
                  default=0.0)
    with open(os.path.join(_TREE, "setup.py"), encoding="utf-8") as f:
        found = re.findall(r'Pybind11Extension\(\s*"smappy\.(\w+)",\s*\[([^\]]*)\]', f.read())
    out = []
    for name, sources in found:
        newest = max([headers] + [os.path.getmtime(os.path.join(_TREE, s)) for s in re.findall(r'"([^"]+)"', sources)
                                  if os.path.exists(os.path.join(_TREE, s))])
        path = os.path.join(_SRC, "smappy", name + suffix)
        if not os.path.exists(path) or os.path.getmtime(path) < newest:
            out.append(name)
    return out


if os.path.isdir(os.path.join(_TREE, "csrc")) and _stale_extensions():
    pytest.exit(f"{_TREE}: the compiled extensions {', '.join(_stale_extensions())} are missing or older "
                f"than csrc/; build this tree's first:\n    cd {_TREE} && python setup.py build_ext --inplace",
                returncode=4)

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


# uipsf's and decode_smlm's tests need torch, which is an extra (`pip install -e .[uipsf]`,
# `.[decode]`); without it they are not collected, so a plain checkout runs the rest of the
# suite as before.
if importlib.util.find_spec("torch") is None:
    collect_ignore = ["uipsf_tests", "decode_tests"]


#: a long test's limit: the region study takes four minutes and more alone
LONG_TIMEOUT = 1200


#: The packages kept in this repository beside smappy, each with its own tests
#: folder (the first path).  Their fast tests always run; their slow tests run
#: under `--slow` only when one of the package's paths changed
#: (`package_changes`), and always under `--release`, `--long`, `--package
#: NAME` or with the package's tests named on the command line.  The paths are
#: the package, its tests and the parts of smappy its tests lean on most --
#: not all of them: uipsf's tests import a dozen smappy modules, and a change
#: to `smappy.simulate` reaches them through the fast tier and `--release`,
#: not here.  A new package is an entry here and a folder of its tests.
PACKAGES = {
    "uipsf": ("tests/uipsf_tests", "src/uipsf", "benchmarks/uipsf", "scripts/uipsf",
              "src/smappy/calibrate", "src/smappy/psf.py", "src/smappy/io/calibration.py",
              "src/smappy/plugins/fit.py", "csrc"),
    "decode": ("tests/decode_tests", "src/decode_smlm", "src/smappy/training",
               "src/smappy/plugins/fit_decode.py", "src/smappy/decode_remote.py",
               "src/smappy/window_engine.py", "src/smappy/io/calibration.py"),
}

#: a change here is a change to every package: how the tests run, what is installed
SHARED_PATHS = ("tests/conftest.py", "pyproject.toml", "setup.py")


def _git(root, *args):
    import subprocess
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                          check=True, timeout=30).stdout


def package_changes(root, since):
    """Which packages' paths differ from `since`: a set of names, and a line
    saying what they were compared with.  The comparison is with the merge
    base of HEAD and `since`, so a branch is judged by its own commits and not
    by what landed upstream meanwhile, plus what is staged, unstaged and
    untracked.  Every package counts as changed when git cannot answer -- no
    git, no such ref, an sdist -- since a skip nobody asked for is the failure
    to avoid."""
    try:
        base = _git(root, "merge-base", "HEAD", since).strip()
        files = _git(root, "diff", "--name-only", base).splitlines()
        files += _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    except Exception as exc:  # noqa: BLE001 -- any failure means "assume changed"
        return set(PACKAGES), f"git could not compare with {since} ({type(exc).__name__}): all run"

    def touches(prefixes):
        return any(f == p or f.startswith(p + "/") for f in files for p in prefixes)

    if touches(SHARED_PATHS):
        return set(PACKAGES), f"shared files changed since {since} ({base[:8]})"
    return ({name for name, paths in PACKAGES.items() if touches(paths)},
            f"compared with {since} ({base[:8]})")


def _packages_to_run(config):
    """The packages whose slow tier runs, and why; computed once per process
    (each xdist worker asks git itself, a few milliseconds)."""
    cached = getattr(config, "_smappy_packages", None)
    if cached is not None:
        return cached
    asked = set(config.getoption("--package") or ())
    unknown = asked - set(PACKAGES)
    if unknown:
        raise pytest.UsageError(f"--package {', '.join(sorted(unknown))}: "
                                f"the packages are {', '.join(PACKAGES)}")
    root = str(config.rootpath)
    # a package's tests named on the command line are wanted whatever changed
    for arg in config.args:
        path = os.path.relpath(os.path.abspath(arg.split("::")[0]), root).replace(os.sep, "/")
        for name, paths in PACKAGES.items():
            if path == paths[0] or path.startswith(paths[0] + "/"):
                asked.add(name)
    if config.getoption("--release") or config.getoption("--long"):
        result = set(PACKAGES), "--release" if config.getoption("--release") else "--long"
    else:
        changed, why = package_changes(root, config.getoption("--since"))
        result = changed | asked, why
    config._smappy_packages = result
    return result


def _package_of(item, root):
    path = os.path.relpath(str(item.path), root).replace(os.sep, "/")
    for name, paths in PACKAGES.items():
        if path.startswith(paths[0] + "/"):
            return name
    return None


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
    parser.addoption("--release", action="store_true",
                     help="before a release: --slow with every package's slow tier, "
                          "changed or not")
    parser.addoption("--package", action="append", metavar="NAME",
                     help=f"run this package's slow tier under --slow even if it did not "
                          f"change (repeatable): {', '.join(PACKAGES)}")
    parser.addoption("--since", default="origin/main", metavar="REF",
                     help="what a package's change is measured against under --slow "
                          "(default origin/main, through the merge base)")
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
    if long_ or config.getoption("--slow") or config.getoption("--release"):
        # a package's slow tier only where the package changed (`PACKAGES`)
        run, why = _packages_to_run(config)
        root = str(config.rootpath)
        for item in items:
            name = _package_of(item, root)
            if name is not None and name not in run and "slow" in item.keywords:
                item.add_marker(pytest.mark.skip(
                    reason=f"{name} slow: unchanged ({why}); run with --package {name} or --release"))
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


def pytest_terminal_summary(terminalreporter, config):
    """Say which packages' slow tiers were left out: a skip count alone hid
    that the run had not checked them."""
    if not (config.getoption("--slow") or config.getoption("--long")
            or config.getoption("--release")):
        return
    run, why = _packages_to_run(config)
    for name in PACKAGES:
        state = "run" if name in run else f"SKIPPED (unchanged) -- --package {name} to run it"
        terminalreporter.write_line(f"{name} slow tier: {state}; {why}")


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


# A modal dialog open this long in a test is one nobody will answer: it is closed and the test
# fails, naming it.  Without this a test that reached an unpatched question waited in Qt's
# event loop for good, where pytest-timeout's signal never runs, and the run hung (2026-11).
MODAL_SECONDS = 10.0


def _dialog_words(widget) -> str:
    """What a dialog says: its class, title and text, for the failure."""
    bits = [type(widget).__name__, repr(widget.windowTitle())]
    text = getattr(widget, "text", None)
    if callable(text):
        try:
            bits.append(repr(str(text())[:200]))
        except Exception:
            pass
    return " ".join(bits)


@contextlib.contextmanager
def modal_guard(seconds: float = MODAL_SECONDS):
    """While inside, a modal dialog left open ``seconds`` is rejected and said in the list this
    yields.  A timer on the GUI thread, which runs inside the dialog's own event loop: the static
    `QMessageBox.question` and `QFileDialog.getOpenFileName` included, which a patch of
    `QDialog.exec` would not see.  Nothing without a QApplication."""
    left: list = []
    app = None
    if "PySide6.QtWidgets" in sys.modules:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
    if app is None:
        yield left
        return
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    seen: dict = {}

    def check() -> None:
        # never raises: a timer's exception left pending in Qt segfaults the next call (CLAUDE.md)
        try:
            widget = QApplication.activeModalWidget()
            if widget is None:
                return
            now = time.monotonic()
            if now - seen.setdefault(id(widget), now) < seconds:
                return
            seen.pop(id(widget), None)
            left.append(_dialog_words(widget))
            if hasattr(widget, "reject"):
                widget.reject()
            else:
                widget.close()
        except Exception as error:                     # noqa: BLE001
            print(f"modal_guard: {type(error).__name__}: {error}", file=sys.stderr)

    timer = QTimer()
    timer.setInterval(250)
    timer.timeout.connect(check)
    timer.start()
    try:
        yield left
    finally:
        timer.stop()


@pytest.fixture(autouse=True, scope="session")
def _pyplot_show_returns():
    """`pyplot.show` returns at once in the tests.  `smappy.gui.figures` makes QtAgg pyplot's
    backend as it is imported, and on it `show` runs Qt's event loop until its window is closed:
    a test after any GUI test on the same worker that ends in ``plt.show()`` -- DECODE's ported
    ones do -- waited for good (2026-11)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        yield
        return
    shown = plt.show
    plt.show = lambda *args, **kwargs: None
    yield
    plt.show = shown


@pytest.fixture(autouse=True)
def _no_dialog_left_open():
    """Fail a test that left a modal dialog waiting (`modal_guard`) instead of hanging the run."""
    with modal_guard() as left:
        yield
    if left:
        pytest.fail(f"a modal dialog waited {MODAL_SECONDS:g} s for an answer and was closed: "
                    + "; ".join(left) + " -- patch the question the test reaches", pytrace=False)


@pytest.fixture(autouse=True, scope="session")
def isolated_config(tmp_path_factory):
    import os
    directory = tmp_path_factory.mktemp("config")
    os.environ["SMAPPY_CONFIG_DIR"] = str(directory)
    from smappy import config, plugins
    config.load(reload=True)
    plugins.discover(force=True)
    yield directory
