"""Every shipped plugin says before it runs whether it changes the table.

The GUI asks before a table change while a fit is still appending, and it can
ask only if the plugin says so up front.  A plugin that does not say cannot
be defined (`Plugin.__init_subclass__`); this checks that the shipped ones say
it truly.
"""
import numpy as np
import pytest

from smappy import plugins
from smappy.locs import Localizations
from smappy.plugins import Context


def shipped():
    return sorted(plugins.refs())


@pytest.mark.parametrize("path", shipped())
def test_every_shipped_plugin_says_whether_it_changes_the_localizations(path):
    cls = plugins.get(path)
    settings = cls.Settings() if cls.Settings else None
    assert cls().changes_locs_for(settings) in (True, False), (
        f"{path} does not declare changes_locs")


def table(n=300):
    rng = np.random.default_rng(3)
    return Localizations({
        "x_nm": rng.uniform(0, 3000, n), "y_nm": rng.uniform(0, 3000, n),
        "frame": np.sort(rng.integers(0, 100, n)),
        "photons": rng.uniform(100, 3000, n),
        "xy_err_nm": rng.uniform(5, 20, n), "sigma_nm": rng.uniform(80, 150, n),
    }, {"units": "nm"})


@pytest.mark.parametrize("path, overrides", [
    ("Analysis/Process/Math Parser", {"expression": "photons > 1000"}),
    ("Analysis/Process/Remove Localizations", {"region": "selection",
                                               "which": "outside"}),
    ("Analysis/Measure/Localization Statistics", {}),
    ("Analysis/Process/Layers", {"remove": False}),
    ("Analysis/Process/Layers", {"remove": True}),
])
def test_what_a_plugin_declares_is_what_its_run_does(path, overrides, monkeypatch,
                                                     tmp_path):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    import dataclasses
    cls = plugins.get(path)
    plugin = cls()
    settings = dataclasses.replace(cls.Settings(), **overrides)
    locs = table()
    # half selected, so Remove Localizations has something to remove
    selection = plugins.Selection(np.asarray(locs["photons"]) > 1000)
    result = plugin.run(Context(locs=locs, selection=selection), settings)
    changed = result.locs is not None or bool(result.files)
    assert changed == plugin.changes_locs_for(settings)

