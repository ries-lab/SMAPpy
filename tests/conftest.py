"""Keep the tests out of the developer's real settings.

`smappy.config` and the workspace live in the platform's config directory, and
a GUI test builds a `ControlWindow`, which reads the workspace on the way up
and writes it on the way down.  Without this the suite would take its plugin
roots from whatever the developer has configured and overwrite their tabs.
"""
import pytest


@pytest.fixture(autouse=True, scope="session")
def isolated_config(tmp_path_factory):
    import os
    directory = tmp_path_factory.mktemp("config")
    os.environ["SMAPPY_CONFIG_DIR"] = str(directory)
    from smappy import config, plugins
    config.load(reload=True)
    plugins.discover(force=True)
    yield directory
