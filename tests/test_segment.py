"""Segment regions: one segmenter, a method each, presets that fill it."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.plugins.roi import DensityPeakSettings
from smappy.plugins.region_column import ColumnROIsSettings
from smappy.plugins.segment import (PATH, SegmentRegions, SegmentSettings, made_by,
                                    preset_values, presets)
from smappy.session import Session


def blobs(centres=((0, 0), (3000, 0), (0, 3000)), per=40, seed=0):
    rng = np.random.default_rng(seed)
    xy = np.vstack([rng.normal(c, 30, (per, 2)) for c in centres])
    ids = np.repeat(np.arange(1, len(centres) + 1), per)
    n = len(xy)
    return Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1],
                          "frame": np.arange(n), "photons": np.full(n, 300.0),
                          "xy_err_nm": np.full(n, 9.0), "cluster_id": ids})


def session_of(locs):
    session = Session(locs)
    session.show_grouped(0, False)
    return session


PEAKS = SegmentSettings(method="peaks",
                        peaks=DensityPeakSettings(separation_nm=1000.0, min_count=20))


def test_the_npc_preset_fills_the_ring_template_and_is_its_defaults():
    assert "NPC" in presets()
    values = preset_values("NPC")
    assert values["method"] == "ring" and values["ring.radius_nm"] == 55.0
    filled = SegmentRegions().react("preset", SegmentSettings(preset="NPC"))
    assert filled == values
    assert SegmentRegions().react("method", SegmentSettings()) is None
    # the shipped values are the ring template's defaults, as the plugin had
    defaults = SegmentSettings().ring
    for key, value in values.items():
        if key.startswith("ring."):
            assert getattr(defaults, key[5:]) == value, key


def test_a_users_preset_wins_over_a_shipped_one_and_a_broken_one_costs_itself(
        tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    mine = tmp_path / "segment_presets"
    mine.mkdir()
    (mine / "NPC.yaml").write_text("method: ring\nring:\n  radius_nm: 60\n")
    (mine / "CME.yaml").write_text("method: peaks\npeaks:\n  sigma_nm: 40\n"
                                   "  no_such_setting: 1\n")
    (mine / "broken.yaml").write_text("method: [unclosed\n")
    found = presets()
    assert set(found) >= {"NPC", "CME"} and "broken" not in found
    assert preset_values("NPC")["ring.radius_nm"] == 60
    assert preset_values("CME") == {"method": "peaks", "peaks.sigma_nm": 40}
    with pytest.raises(ValueError, match="no segmentation preset"):
        preset_values("nothing")


def test_the_form_shows_only_the_chosen_methods_part():
    for method in ("ring", "peaks", "ids"):
        state = SegmentRegions().describe(None, SegmentSettings(method=method))
        assert state.shown == {"ring": method == "ring", "peaks": method == "peaks",
                               "ids": method == "ids"}
        assert ("NPC", "NPC") in state.choices["preset"]


def test_replace_removes_the_same_methods_rois_and_leaves_another_methods():
    session = session_of(blobs())
    project = session.rois
    file_id = next(iter(project.sources))
    ring = project.add_roi(file_id, (9000.0, 9000.0),
                           origin={"method": PATH, "segment": "ring"})
    first = SegmentRegions().run(session.context(), PEAKS).rois
    assert len(first) == 3 and all(made_by(r, "peaks") for r in first)
    again = SegmentRegions().run(session.context(), PEAKS).rois
    assert len(again) == 3
    assert ring.id in project.rois                         # another method's
    assert not {r.id for r in first} & set(project.rois)   # its own: replaced


def test_rois_of_an_id_column_have_a_preview_and_then_are_made():
    session = session_of(blobs())
    settings = SegmentSettings(method="ids", ids=ColumnROIsSettings(min_size=10))
    preview = SegmentRegions().preview(session.context(), settings)
    assert session.rois.rois == {}
    assert len(preview.roi_preview["sites"]) == 3
    run = SegmentRegions().run(session.context(), settings)
    assert len(run.rois) == 3 and all(made_by(r, "ids") for r in run.rois)
    assert sorted(map(tuple, (s["center"] for s in preview.roi_preview["sites"]))) == \
        sorted(tuple(r.center) for r in run.rois)


def test_an_unknown_method_is_refused_with_the_ones_there_are():
    session = session_of(blobs())
    with pytest.raises(ValueError, match="ring, peaks, ids"):
        SegmentRegions().run(session.context(), SegmentSettings(method="cme"))
