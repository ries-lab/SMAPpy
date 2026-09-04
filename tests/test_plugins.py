"""The plugin contract: declared parameters, registry, and a run without a GUI."""
import numpy as np

from smappy import plugins
from smappy.drift import DriftSettings
from smappy.plugins import Result, Selection
from smappy.session import Session
from test_drift import SETTINGS, simulate


def test_registry_has_comet_and_specs_cover_every_field():
    cls = plugins.get("Analysis/Drift/COMET")
    assert cls.name == "COMET"
    specs = cls.specs()
    assert set(specs) == {f.name for f in DriftSettings.__dataclass_fields__.values()}
    assert specs["segmentation_var"].info.advanced is False
    assert specs["optimizer_ftol"].info.advanced is True
    assert specs["initial_sigma_nm"].optional and specs["initial_sigma_nm"].type is float
    assert "Drift" in plugins.tree()["Analysis"]


def test_plugin_runs_as_a_function_and_through_a_session():
    locs, truth, _ = simulate()
    plugin = plugins.get("Analysis/Drift/COMET")()
    result = plugin(locs, Selection.all(len(locs)), SETTINGS)
    assert isinstance(result, Result) and result.locs is not None
    assert "drift over 50 frames" in result.text

    session = Session(locs)
    session.layers[0].filter.set("loc_precision_nm", None, 30.0)
    before = session.locs
    session.run(plugin, SETTINGS)
    assert session.locs is not before and session.can_undo
    assert session.history[-1]["what"] == "Analysis/Drift/COMET"
    corrected_span = np.ptp(session.locs["x_nm"]) - np.ptp(before["x_nm"])
    assert corrected_span < 0                      # drift removed: tighter
    session.undo()
    assert session.locs is before and not session.can_undo


def test_rcc_is_registered_and_runs():
    from smappy.rcc import RCCSettings
    locs, truth, _ = simulate(n_frames=200, per_frame=60)
    plugin = plugins.get("Analysis/Drift/RCC")()
    result = plugin(locs, Selection.all(len(locs)),
                    RCCSettings(n_timepoints=5, pixelsize_nm=20, group=False, use_z=False))
    assert result.locs is not None and "drift over 200 frames" in result.text
