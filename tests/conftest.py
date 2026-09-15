"""Keep the tests off the developer's screen and out of their real settings.

`smappy.config` and the workspace live in the platform's config directory, and
a GUI test builds a `ControlWindow`, which reads the workspace on the way up
and writes it on the way down.  Without this the suite would take its plugin
roots from whatever the developer has configured and overwrite their tabs.
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


@pytest.fixture(autouse=True, scope="session")
def isolated_config(tmp_path_factory):
    import os
    directory = tmp_path_factory.mktemp("config")
    os.environ["SMAPPY_CONFIG_DIR"] = str(directory)
    from smappy import config, plugins
    config.load(reload=True)
    plugins.discover(force=True)
    yield directory
