"""The GUI's background warm-up (`smappy.gui.warmup`)."""
import importlib.util

from smappy.gui import warmup


def test_every_module_the_warm_up_names_exists():
    # the warm-up swallows a failed import, so a renamed module would
    # silently stop being warmed: this is where that shows
    missing = [m for m in warmup.MODULES if importlib.util.find_spec(m) is None]
    assert not missing


def test_the_warm_up_imports_on_a_thread_of_its_own():
    import sys
    thread = warmup.start(("smappy.calibrate.transform",))
    assert thread.daemon and thread.name == "smappy-warmup"
    thread.join(60)
    assert "smappy.calibrate.transform" in sys.modules
