"""Segment regions: one segmenter, a method each, presets that fill it, and
keep on what was measured on each candidate."""
import numpy as np
import pytest

from smappy import segmentation
from smappy.locs import Localizations
from smappy.plugins.region_column import ColumnROIsSettings
from smappy.plugins.segment import (PATH, BeadSegmentSettings, CircleSettings,
                                    PositionsSettings, SegmentRegions, SegmentSettings,
                                    made_by, preset_values, presets)
from smappy.session import Session


def blobs(centres=((0, 0), (3000, 0), (0, 3000)), per=40, seed=0):
    rng = np.random.default_rng(seed)
    xy = np.vstack([rng.normal(c, 30, (per, 2)) for c in centres])
    ids = np.repeat(np.arange(1, len(centres) + 1), per)
    n = len(xy)
    return Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1],
                          "frame": np.arange(n), "photons": np.full(n, 300.0),
                          "xy_err_nm": np.full(n, 9.0), "cluster_id": ids})


def table(xy, precision=8.0, **columns):
    n = len(xy)
    return Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1], "frame": np.arange(n),
                          "xy_err_nm": np.full(n, precision), **columns})


def discs(radii, spacing=600.0, precision=8.0, background=400, seed=0, aspect=1.0):
    """Filled discs (ellipses with ``aspect`` < 1) on a grid, with uniform
    background: the positions, and each disc's centre and radius."""
    rng = np.random.default_rng(seed)
    radii = np.asarray(radii, float)
    across = 7
    centres = np.array([(spacing * (i % across), spacing * (i // across))
                        for i in range(len(radii))], float)
    pts = []
    for (cx, cy), R in zip(centres, radii):
        n = int(400 * (R / 100) ** 2) + 60
        r = R * np.sqrt(rng.uniform(0, 1, n))
        a = rng.uniform(0, 2 * np.pi, n)
        pts.append(np.column_stack((cx + r * np.cos(a), cy + aspect * r * np.sin(a))))
    xy = np.concatenate(pts) + rng.normal(0, precision, (sum(map(len, pts)), 2))
    lo, hi = centres.min(axis=0) - 300, centres.max(axis=0) + 300
    xy = np.concatenate([xy, rng.uniform(lo, hi, (background, 2))])
    return xy, centres, radii


def session_of(locs):
    session = Session(locs)
    session.show_grouped(0, False)
    return session


DISCS = SegmentSettings(method="circle", keep="n_locs >= 20",
                        circle=CircleSettings(shape="disc", min_radius_nm=30.0,
                                              max_radius_nm=100.0))
NPC_KEEP = "n_ring >= 4 and 40 <= radius_nm <= 70 and far_p >= 0.05"


# ------------------------------------------------------------------ presets

def test_the_npc_preset_fills_the_circle_template_with_its_ring_and_keep():
    assert {"NPC", "CME", "Circles"} <= set(presets())
    values = preset_values("NPC")
    assert values["method"] == "circle" and values["circle.shape"] == "ring"
    assert values["circle.min_radius_nm"] == values["circle.max_radius_nm"] == 55.0
    assert values["keep"] == NPC_KEEP
    filled = SegmentRegions().react("preset", SegmentSettings(preset="NPC"))
    assert filled == values
    assert SegmentRegions().react("method", SegmentSettings()) is None
    # the shipped chain segments with the preset's values
    import yaml
    from smappy.plugins import segment
    chain = yaml.safe_load((segment.SHIPPED.parent.parent / "plugins"
                            / "npc_analysis.chain.yaml").read_text())
    step = next(s for s in chain["steps"] if s.get("plugin") == PATH)["values"]
    assert {k: v for k, v in values.items() if k in step} == \
        {k: v for k, v in step.items() if k in values}
    assert {"keep", "circle.shape", "circle.min_radius_nm"} <= set(step)
    assert preset_values("CME")["circle.shape"] == "disc"
    assert (preset_values("CME")["circle.min_radius_nm"],
            preset_values("CME")["circle.max_radius_nm"]) == (30.0, 150.0)
    assert (preset_values("Circles")["circle.min_radius_nm"],
            preset_values("Circles")["circle.max_radius_nm"]) == (100.0, 500.0)


def test_a_users_preset_wins_over_a_shipped_one_and_a_broken_one_costs_itself(
        tmp_path, monkeypatch):
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path))
    mine = tmp_path / "segment_presets"
    mine.mkdir()
    (mine / "NPC.yaml").write_text("method: circle\ncircle:\n  min_radius_nm: 60\n")
    (mine / "Pits.yaml").write_text("method: circle\ncircle:\n  shape: disc\n"
                                    "  no_such_setting: 1\n")
    (mine / "broken.yaml").write_text("method: [unclosed\n")
    found = presets()
    assert set(found) >= {"NPC", "Pits"} and "broken" not in found
    assert preset_values("NPC")["circle.min_radius_nm"] == 60
    assert preset_values("Pits") == {"method": "circle", "circle.shape": "disc"}
    with pytest.raises(ValueError, match="no segmentation preset"):
        preset_values("nothing")


def test_the_form_shows_only_the_chosen_methods_part_and_what_keep_can_name():
    for method in ("circle", "beads", "ids", "file"):
        state = SegmentRegions().describe(None, SegmentSettings(method=method))
        for key in ("circle", "beads", "ids", "file"):
            assert state.shown[key] == (key == method)
        measured = method in ("circle", "beads")
        assert state.shown["keep"] == state.shown["keep_rejected"] == measured
        assert ("NPC", "NPC") in state.choices["preset"]
    ring = SegmentRegions().describe(None, SegmentSettings(
        circle=CircleSettings(shape="ring")))
    assert "n_ring" in ring.notice.text and "far_p" in ring.notice.text
    disc = SegmentRegions().describe(None, SegmentSettings())
    assert "centre_ratio" in disc.notice.text and "n_ring" not in disc.notice.text
    assert disc.active["circle.ring_width_nm"] is False
    wrong = SegmentRegions().describe(None, SegmentSettings(keep="n_ring >= 4"))
    assert "n_ring" in wrong.notice.text and wrong.notice.kind == "warning"


# ------------------------------------------------------------------ the circle template

def test_discs_of_30_to_150_nm_return_their_radii_within_15_percent():
    xy, centres, radii = discs([30, 45, 60, 80, 100, 120, 150] * 3)
    sites = segmentation.find_circles(table(xy), "disc", 30.0, 150.0)
    segmentation.judge(sites, "n_locs >= 20", segmentation.VALUES["disc"])
    used = [s for s in sites if s["use"]]
    from scipy.spatial import cKDTree
    d, i = cKDTree(centres).query([s["center"] for s in used])
    assert len(used) == len(centres) and set(i) == set(range(len(centres)))
    assert d.max() < 10
    error = np.array([s["values"]["radius_nm"] for s in used]) / radii[i] - 1
    assert np.abs(error).max() < 0.15
    # a uniform disc has as dense a centre as an edge
    ratio = np.array([s["values"]["centre_ratio"] for s in used])
    assert 0.8 < np.median(ratio) < 1.3


def test_a_ring_has_a_low_centre_ratio_as_a_disc():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 2 * np.pi, 300)
    xy = np.column_stack((80 * np.cos(a), 80 * np.sin(a))) + rng.normal(0, 6, (300, 2))
    site = segmentation.measure_disc(xy[:, 0], xy[:, 1], (5, 5), 80.0,
                                     np.full(300, 6.0))
    assert site["values"]["centre_ratio"] < 0.2


@pytest.mark.parametrize("aspect", [1.0, 0.6, 0.35])
def test_ellipses_return_their_roundness(aspect):
    xy, centres, _ = discs([100] * 7, aspect=aspect, background=0, seed=2)
    sites = segmentation.find_circles(table(xy), "disc", 40.0, 150.0)
    segmentation.judge(sites, "n_locs >= 100", segmentation.VALUES["disc"])
    roundness = np.array([s["values"]["roundness"] for s in sites if s["use"]])
    assert len(roundness) >= 7
    # a uniform ellipse's SDs are in the ratio of its axes; the 8 nm of
    # precision rounds it a little
    assert np.median(roundness) == pytest.approx(aspect, abs=0.06)


def test_touching_pairs_are_flagged_and_kept_apart():
    rng = np.random.default_rng(3)

    def ring_at(c, n=80):
        a = rng.uniform(0, 2 * np.pi, n)
        return np.column_stack((c[0] + 55 * np.cos(a), c[1] + 55 * np.sin(a)))

    # a pair 95 nm apart (their circles overlap by 15 nm) and one alone
    xy = np.concatenate([ring_at((0, 0)), ring_at((95, 0)), ring_at((1000, 0))])
    xy += rng.normal(0, 4, xy.shape)
    sites = segmentation.find_circles(table(xy, 4.0), "ring", 55.0, 55.0)
    # without the far test, which a neighbour's localizations in the window
    # fail: they lie far from this ring
    keep = "n_ring >= 4 and 40 <= radius_nm <= 70"
    segmentation.judge(sites, keep, segmentation.VALUES["ring"])
    used = sorted((s for s in sites if s["use"]), key=lambda s: s["center"][0])
    assert len(used) == 3
    # the weaker of the pair is fitted on what the stronger left of it
    assert np.allclose([s["center"][0] for s in used], [0, 95, 1000], atol=5)
    assert [s["values"]["touching"] for s in used] == [1, 1, 0]
    # a keep that leaves touching structures out
    segmentation.judge(sites, keep + " and touching == 0", segmentation.VALUES["ring"])
    assert sum(s["use"] for s in sites) == 1
    # the far test fails both: each sees its neighbour's localizations far
    # from its ring, as the ring template's did
    segmentation.judge(sites, NPC_KEEP, segmentation.VALUES["ring"])
    assert sum(s["use"] for s in sites) == 1


def test_the_overlap_rule_is_half_the_smaller_radius():
    assert segmentation.too_close(82.0, 55.0, 55.0)
    assert not segmentation.too_close(83.0, 55.0, 55.0)
    assert segmentation.too_close(100.0, 30.0, 100.0)       # 130 - 15 = 115
    assert not segmentation.too_close(116.0, 30.0, 100.0)


def test_the_radii_of_a_range_step_by_the_ring_width_or_15_percent():
    assert segmentation.radii(55.0, 55.0) == [55.0]
    rr = segmentation.radii(100.0, 500.0, 20.0)
    assert rr[0] == 100.0 and rr[-1] == 500.0
    steps = np.diff(rr)[:-1]
    assert np.allclose(steps, np.maximum(20.0, 0.15 * np.asarray(rr[:-2])))
    with pytest.raises(ValueError):
        segmentation.radii(60.0, 50.0)


def test_both_templates_sum_to_zero():
    for shape in ("ring", "disc"):
        kernel, weight = segmentation.template(shape, 50.0, 15.0, 10.0)
        assert abs(kernel.sum()) < 1e-9 * weight and weight > 0


# ------------------------------------------------------------------ keep

def test_keep_names_what_it_failed_and_its_bounds():
    text = "n_ring >= 4 and 40 <= radius_nm <= 70 and `far_p` >= 0.05"
    assert segmentation.clauses(text) == ["n_ring >= 4", "40 <= radius_nm <= 70",
                                          "`far_p` >= 0.05"]
    assert segmentation.bounds(text) == {"n_ring": [4.0], "radius_nm": [40.0, 70.0],
                                         "far_p": [0.05]}
    sites = [{"center": [0, 0], "values": {"n_ring": 9, "radius_nm": 30.0, "far_p": 0.01,
                                            "touching": 0}},
             {"center": [500, 0], "values": {"n_ring": 9, "radius_nm": 55.0,
                                              "far_p": 0.5, "touching": 0}}]
    segmentation.judge(sites, text, segmentation.VALUES["ring"])
    assert sites[0]["failed"] == ["40 <= radius_nm <= 70", "`far_p` >= 0.05"]
    assert sites[1]["use"] and sites[1]["failed"] == []
    segmentation.judge(sites, "", segmentation.VALUES["ring"])
    assert all(s["use"] for s in sites)


def test_a_name_the_method_does_not_measure_is_an_error_naming_those_it_does():
    with pytest.raises(ValueError, match="centre_ratio.*does not measure.*n_ring"):
        segmentation.keep_test("centre_ratio > 0.5", segmentation.VALUES["ring"])
    session = session_of(blobs())
    settings = SegmentSettings(method="circle", circle=DISCS.circle,
                               keep="psf_nm <= 150")
    with pytest.raises(ValueError, match="no sigma_nm"):
        SegmentRegions().run(session.context(), settings)


def test_rejected_candidates_are_unused_rois_saying_what_they_failed():
    session = session_of(blobs())
    settings = SegmentSettings(method="circle", circle=DISCS.circle, keep_rejected=True,
                               keep="n_locs >= 20 and radius_nm < 40")
    found = SegmentRegions().run(session.context(), settings).rois
    assert len(found) >= 3
    blob_rois = [r for r in found if r.origin["values"]["n_locs"] >= 20]
    assert len(blob_rois) == 3 and not any(r.use for r in blob_rois)
    assert all(r.origin["failed"] == ["radius_nm < 40"] for r in blob_rois)


def test_the_values_are_columns_of_the_results_table_that_filter_rois_reads():
    from smappy.plugins.roi_order import FilterROIs, FilterROIsSettings
    session = session_of(blobs())
    project = session.rois
    SegmentRegions().run(session.context(), DISCS)
    rows = project.results()
    assert len(rows) == 3
    assert {"n_locs", "radius_nm", "roundness", "centre_ratio", "touching"} <= set(rows[0])
    assert "radius_nm" in project.sort_columns()
    # an ROI moved keeps what was measured on its candidate
    roi = next(iter(project.rois.values()))
    project.move_roi(roi.id, (roi.center[0] + 200, roi.center[1]))
    assert len(project.results()) == 3
    FilterROIs().run(session.context(), FilterROIsSettings(condition="n_locs >= 1000"))
    assert not any(r.use for r in project.rois.values())


# ------------------------------------------------------------------ the plugin

def test_replace_removes_the_same_methods_rois_and_leaves_another_methods():
    session = session_of(blobs())
    project = session.rois
    file_id = next(iter(project.sources))
    beads = project.add_roi(file_id, (9000.0, 9000.0),
                            origin={"method": PATH, "segment": "beads"})
    first = SegmentRegions().run(session.context(), DISCS).rois
    assert len(first) == 3 and all(made_by(r, "circle") for r in first)
    again = SegmentRegions().run(session.context(), DISCS).rois
    assert len(again) == 3
    assert beads.id in project.rois                        # another method's
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
    # added as previewed: ROIs of their id alone, as the run made them
    tracks = SegmentSettings(method="ids", ids=ColumnROIsSettings(
        min_size=10, membership="id"))
    session.show_preview(SegmentRegions(), SegmentRegions().preview(session.context(),
                                                                    tracks))
    added = session.add_preview()
    assert len(added) == 3 and all(not r.spatial and r.ids for r in added)
    assert len(session.rois.rois) == 3                    # the earlier ones replaced


def test_an_unknown_method_is_refused_with_the_ones_there_are():
    session = session_of(blobs())
    with pytest.raises(ValueError, match="circle, beads, ids, file"):
        SegmentRegions().run(session.context(), SegmentSettings(method="ring"))


# ------------------------------------------------------------------ beads

def bead_session(n_beads=5, seed=5):
    from smappy.io.formats import FileInfo
    from smappy.simulate import simulate, with_beads
    locs = with_beads(simulate(n_frames=3000, seed=seed, drift=True),
                      n_beads=n_beads, seed=seed)
    session = Session()
    session.add_file(locs, FileInfo(name="b", path="b", format="test", n=len(locs)))
    session.show_grouped(0, False)
    return session, locs


def test_simulated_beads_become_rois_with_the_fraction_of_frames_they_are_seen_in():
    session, locs = bead_session()
    project = session.rois
    settings = SegmentSettings(method="beads", keep="frames_fraction >= 0.9")
    found = SegmentRegions().run(session.context(), settings).rois
    assert len(found) == 5 and all(made_by(r, "beads") for r in found)
    bead = np.asarray(locs["bead_true"])
    truth = np.array([[np.median(locs["x_nm"][bead == b]), np.median(locs["y_nm"][bead == b])]
                      for b in range(5)])
    from scipy.spatial import cKDTree
    d, i = cKDTree(truth).query([r.center for r in found])
    assert set(i) == set(range(5)) and d.max() < 1.0
    for roi in found:
        values = roi.origin["values"]
        assert values["frames_fraction"] == pytest.approx(0.98, abs=0.02)   # 2 % missed
        assert values["photons"] > 5000 and 0 < values["spread_nm"] < 200
    assert {"frames_fraction", "photons", "spread_nm"} <= set(project.results()[0])


def test_fiducials_on_bead_rois_gives_the_drift_it_finds_itself():
    from smappy.fiducials import FiducialSettings
    from smappy.plugins.drift_fiducials import FiducialDrift
    session, _ = bead_session()
    session.rois.set_geometry(800, "square")
    SegmentRegions().run(session.context(), SegmentSettings(method="beads"))
    ctx = session.context(grouping="ungrouped")
    own = FiducialDrift().run(ctx, FiducialSettings()).data["drift"].drift
    ctx = session.context(grouping="ungrouped")
    from_rois = FiducialDrift().run(ctx, FiducialSettings(source="ROI manager"))
    assert np.abs(from_rois.data["drift"].drift - own).max() < 0.5


def test_beads_refuse_merged_blinks():
    session, _ = bead_session()
    session.show_grouped(0, True)
    with pytest.raises(ValueError, match="single localizations"):
        SegmentRegions().run(session.context(),
                             SegmentSettings(method="beads", beads=BeadSegmentSettings()))


# ------------------------------------------------------------------ positions

def test_roi_centres_exported_with_export_table_come_back_as_the_same_rois(tmp_path):
    from smappy.plugins.region_export import ExportTable, ExportTableSettings
    session = session_of(blobs())
    project = session.rois
    made = SegmentRegions().run(session.context(), DISCS).rois
    path = tmp_path / "sites.csv"
    ExportTable().run(session.context(), ExportTableSettings(path=str(path)))
    settings = SegmentSettings(method="file", file=PositionsSettings(path=str(path)))
    preview = SegmentRegions().preview(session.context(), settings)
    assert len(preview.roi_preview["sites"]) == 3
    back = SegmentRegions().run(session.context(), settings).rois
    assert all(made_by(r, "file") for r in back)
    assert sorted(tuple(np.round(r.center, 6)) for r in back) == \
        sorted(tuple(np.round(r.center, 6)) for r in made)
    assert {r.file_id for r in back} == {r.file_id for r in made}
    assert len(project.rois) == 6                         # the circle's stay


def test_positions_are_read_by_the_csv_readers_conventions(tmp_path):
    path = tmp_path / "pos.tsv"
    path.write_text("X [um]\ty [um]\tfilenumber\n1.5\t2\t3\n\t1\t3\n0.25\t0.5\t\n")
    read = segmentation.read_positions(path)
    assert np.allclose(read["x_nm"], [1500, 250]) and np.allclose(read["y_nm"], [2000, 500])
    assert read["filenumber"] == [3, ""]
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="no column for x_nm, y_nm"):
        segmentation.read_positions(tmp_path / "bad.csv")
