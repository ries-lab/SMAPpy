"""Nuclear pores: segmenting, counting corners, the labelling efficiency."""
import numpy as np
import pytest
from scipy.stats import binom

from smappy import npc as model
from smappy import plugins
from smappy.locs import Localizations
from smappy.plugins.npc import (FITS, LabelingEfficiencySettings, NPCCornersSettings,
                                NPCSegmentSettings, count_corners, fit_circle,
                                labeling_efficiency, ring_kernel, ring_quality,
                                segment_npcs)
from smappy.simulate import simulate
from smappy.simulate.settings import (BlinkingSettings, LabellingSettings,
                                      SimulationSettings, StructureSettings)

R = 53.7


def pores(efficiency, seed=1, n_frames=3000, blinks=3.0):
    """Nup96 pores imaged until every fluorophore has bleached."""
    return simulate(SimulationSettings(
        n_frames=n_frames, seed=seed, structure=StructureSettings(preset="npc"),
        labelling=LabellingSettings(efficiency=efficiency),
        blinking=BlinkingSettings(blinks=blinks, activation="all")))


def corners_at(which, rng, n=15, rotation=0.0, center=(0.0, 0.0), sd=6.0):
    """Localizations at some of the eight corners of a pore."""
    xy = [np.column_stack((center[0] + R * np.cos(rotation + k * np.pi / 4)
                           + rng.normal(0, sd, n),
                           center[1] + R * np.sin(rotation + k * np.pi / 4)
                           + rng.normal(0, sd, n))) for k in which]
    return np.concatenate(xy)


def disc(rng, n=300, radius=60.0, center=(0.0, 0.0)):
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack((center[0] + r * np.cos(a), center[1] + r * np.sin(a)))


# ------------------------------------------------------------------ geometry

def test_the_circle_fit_finds_centre_and_radius_through_background():
    rng = np.random.default_rng(0)
    angle = rng.uniform(0, 2 * np.pi, 300)
    x = 1000 + 55 * np.cos(angle) + rng.normal(0, 5, 300)
    y = -400 + 55 * np.sin(angle) + rng.normal(0, 5, 300)
    # a fifth of the points scattered over the window
    x = np.concatenate([x, 1000 + rng.uniform(-100, 100, 60)])
    y = np.concatenate([y, -400 + rng.uniform(-100, 100, 60)])
    x0, y0, r = fit_circle(x, y, (1030, -380))
    assert abs(x0 - 1000) < 2 and abs(y0 + 400) < 2 and abs(r - 55) < 2
    x0, y0, r = fit_circle(x, y, (1030, -380), radius=50)
    assert abs(x0 - 1000) < 2 and abs(y0 + 400) < 2 and r == 50


def test_the_ring_kernel_sums_to_zero_so_flat_background_gives_nothing():
    kernel = ring_kernel(55, 15, 10)
    assert abs(kernel.sum()) < 1e-12
    assert kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] < 0


# --------------------------------------------------------------- segmenting

def test_every_simulated_pore_is_found_centred_and_nothing_else():
    locs = pores(0.6, seed=3)
    truth = np.column_stack((locs.metadata["copies"]["x_nm"],
                             locs.metadata["copies"]["y_nm"]))
    sites = segment_npcs(locs, NPCSegmentSettings())
    used = np.array([s["center"] for s in sites if s["use"]])
    from scipy.spatial import cKDTree
    distance, index = cKDTree(truth).query(used)
    assert len(set(index[distance < 10])) >= 0.97 * len(truth)
    assert np.sum(distance >= 20) <= 0.02 * len(truth)
    assert np.median(distance) < 3
    radius = np.array([s["radius_nm"] for s in sites if s["use"]])
    assert abs(np.median(radius) - R) < 3


def test_a_pore_passes_and_a_blob_fails_on_its_radius():
    rng = np.random.default_rng(0)
    settings = NPCSegmentSettings()
    full = corners_at(range(8), rng, sd=4)
    sigma = np.full(len(full), 4.0)
    quality = ring_quality(full[:, 0], full[:, 1], (5, -5), settings, sigma)
    assert quality["use"] and abs(quality["radius_nm"] - R) < 3
    blob = corners_at([0], rng, sd=4)
    quality = ring_quality(blob[:, 0], blob[:, 1], (0, 0), settings, sigma[:len(blob)])
    assert "radius" in quality["failed"]


def test_a_sparse_pore_is_kept():
    """Three neighbouring corners fit a ring of the right size, and nothing
    lies far from it: the pores with few corners are what the efficiency is
    measured from."""
    rng = np.random.default_rng(1)
    arc = corners_at([0, 1, 2], rng, n=4, sd=4)
    quality = ring_quality(arc[:, 0], arc[:, 1], (0, 0), NPCSegmentSettings(),
                           np.full(len(arc), 4.0))
    assert quality["use"], quality["failed"]


@pytest.mark.parametrize("offset", [0.0, 55.0])
def test_a_filled_disc_fails_on_its_localizations_far_from_the_ring(offset):
    """Centred on it, or beside it, as the ring filter often puts the site."""
    rng = np.random.default_rng(2)
    xy = disc(rng)
    quality = ring_quality(xy[:, 0], xy[:, 1], (offset, 0), NPCSegmentSettings(),
                           np.full(len(xy), 5.0))
    assert not quality["use"]
    assert "far" in quality["failed"] or "radius" in quality["failed"]


def test_the_far_test_is_off_without_a_precision():
    rng = np.random.default_rng(2)
    xy = corners_at(range(8), rng, sd=4)
    quality = ring_quality(xy[:, 0], xy[:, 1], (0, 0), NPCSegmentSettings(), None)
    assert quality["use"] and quality["far_fraction"] == 0


def test_rejected_candidates_become_unused_rois_only_when_asked():
    from smappy.roi_manager import ROIProject
    from smappy.plugins.npc import NPCSegment
    rng = np.random.default_rng(3)
    good = corners_at(range(8), rng, center=(1000, 1000), sd=4)
    filled = disc(rng, center=(3000, 1000))
    xy = np.concatenate([good, filled])
    locs = Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1],
                          "frame": np.arange(len(xy)),
                          "xy_err_nm": np.full(len(xy), 5.0)})
    for keep, expected in ((False, 1), (True, 2)):
        project = ROIProject()
        file_id = project.add_source(locs).id
        found = project.find(file_id, plugin=NPCSegment(),
                             settings=NPCSegmentSettings(keep_rejected=keep))
        assert len(found) >= expected and found[0].use
        assert np.hypot(*(np.asarray(found[0].center) - 1000)) < 3
        assert found[0].origin["method"] == "ROIManager/Segment/NPC"
        assert abs(found[0].origin["radius_nm"] - R) < 3
        assert all(not roi.use for roi in found[1:])
    assert all(roi.origin["failed"] for roi in found[1:])


# ------------------------------------------------------------------ corners

@pytest.mark.parametrize("which", [[0], [0, 4], [1, 2, 5], [0, 1, 2, 3, 5, 7], range(8)])
def test_the_corners_with_a_localization_are_counted_at_any_rotation(which):
    rng = np.random.default_rng(len(list(which)))
    xy = corners_at(which, rng, n=4, rotation=0.37, center=(500, 200), sd=4)
    values = count_corners(xy[:, 0], xy[:, 1], (510, 190), NPCCornersSettings(),
                           precision=np.full(len(xy), 5.0))
    assert values["n_corners"] == len(list(which))
    assert values["n_localizations"] == len(xy)
    if len(list(which)) >= 3:       # one or two corners do not fix a centre
        assert abs(values["x_nm"] - 500) < 8 and abs(values["y_nm"] - 200) < 8


def test_an_imprecise_localization_counts_for_the_pore_but_opens_no_corner():
    rng = np.random.default_rng(4)
    xy = corners_at([0, 2], rng, n=5, sd=3)
    stray = np.array([[R * np.cos(np.pi / 2), R * np.sin(np.pi / 2)]])
    xy = np.concatenate([xy, stray])
    precision = np.r_[np.full(len(xy) - 1, 5.0), 25.0]
    values = count_corners(xy[:, 0], xy[:, 1], (0, 0), NPCCornersSettings(), precision)
    assert values["n_corners"] == 2 and values["n_corners_smap"] == 2
    assert values["n_localizations"] == len(xy)      # 25 nm is better than 30


def test_a_localization_near_the_centre_opens_no_corner():
    rng = np.random.default_rng(5)
    xy = np.concatenate([corners_at([0, 2, 4, 6], rng, n=5, sd=3),
                         [[35 * np.cos(np.pi / 4), 35 * np.sin(np.pi / 4)]]])
    values = count_corners(xy[:, 0], xy[:, 1], (0, 0), NPCCornersSettings(),
                           np.full(len(xy), 3.0))
    assert values["n_corners"] == 4


# --------------------------------------------------------- the joint model

def test_without_spill_or_extra_localizations_the_corners_are_binomial():
    le, p, a = 0.4, 0.3, 0.25
    pmf = model.joint_pmf(le, p, a, 0.0, 0.0, 256)
    assert abs(pmf.sum() - 1) < 1e-9
    seen = a / (a + p * (1 - a))             # a copy with a blink that passes
    expected = binom.pmf(np.arange(9), 8, 1 - (1 - le * seen) ** 4)
    assert np.abs(pmf.sum(axis=1) - expected).max() < 1e-9


def test_the_joint_distribution_matches_a_monte_carlo_of_its_assumptions():
    le, p, a, b, eps = 0.4, 0.3, 0.25, 0.15, 0.08
    pmf = model.joint_pmf(le, p, a, b, eps, 256)
    rng = np.random.default_rng(1)
    n = 60000
    blinks = np.where(rng.random((n, 8, 4)) < le, rng.geometric(p, (n, 8, 4)), 0)
    good = rng.binomial(blinks, a)
    extra = rng.binomial(blinks - good, b / (1 - a))
    per_corner = good.sum(axis=2)
    right = rng.binomial(per_corner, eps)
    left = rng.binomial(per_corner - right, eps / (1 - eps))
    stay = per_corner - right - left
    k = ((stay > 0) | (np.roll(right, 1, axis=1) > 0)
         | (np.roll(left, -1, axis=1) > 0)).sum(axis=1)
    total = (good + extra).sum(axis=(1, 2))
    assert np.abs(np.bincount(k, minlength=9) / n - pmf.sum(axis=1)).max() < 0.006
    mean_n = (pmf * np.arange(256)).sum(axis=1) / np.maximum(pmf.sum(axis=1), 1e-300)
    for j in range(4, 9):
        assert abs(total[k == j].mean() - mean_n[j]) < 0.4


def test_the_fit_recovers_efficiency_and_blinks_from_pores_above_the_cut():
    le, p, a, b, eps = 0.55, 0.3, 0.3, 0.1, 0.03
    pmf = model.joint_pmf(le, p, a, b, eps, 512)
    rng = np.random.default_rng(2)
    draw = rng.choice(pmf.size, 1500, p=pmf.ravel() / pmf.sum())
    k, n = np.unravel_index(draw, pmf.shape)
    fitted = model.fit(k, n, a, b, eps, k_min=5)
    assert fitted["pores"] == int(np.sum(k >= 5))
    assert abs(fitted["efficiency"] - le) < 3 * fitted["error"]
    assert abs(fitted["blinks"] - 1 / p) < 3 * fitted["blinks_error"]


def test_worse_precision_spills_more():
    rng = np.random.default_rng(3)
    good = model.localization_classes(rng.gamma(4, 1.5, 5000), 53.7, (40, 70), 20, 30, 100)
    poor = model.localization_classes(rng.gamma(4, 4.0, 5000), 53.7, (40, 70), 20, 30, 100)
    assert poor[2] > 2 * good[2] and good[0] > poor[0]
    assert all(0 <= v <= 1 for v in good + poor) and good[0] + good[1] <= 1


# ----------------------------------------------------- SMAP's method

@pytest.mark.parametrize("efficiency", [0.3, 0.5, 0.65])
def test_both_fits_recover_the_efficiency_of_binomial_corners(efficiency):
    rng = np.random.default_rng(int(100 * efficiency))
    p_corner = 1 - (1 - efficiency) ** 4
    n = rng.binomial(8, p_corner, 400)
    found = {fit: labeling_efficiency(n, LabelingEfficiencySettings(fit=fit, fit_min=3))
             for fit in FITS}
    for result in found.values():
        assert abs(result["efficiency"] - efficiency) < 3 * result["error"] + 0.01
    # the likelihood and SMAP's least squares agree to well within the error
    ml, lsq = (found[f] for f in ("likelihood", "sqrt least squares (SMAP)"))
    assert abs(ml["efficiency"] - lsq["efficiency"]) < ml["error"]


def test_the_fit_range_makes_missing_sparse_pores_harmless():
    """Pores with few corners are the ones a segmenter misses; conditioning
    on the fit range is what keeps their absence from biasing the answer."""
    rng = np.random.default_rng(7)
    n = rng.binomial(8, 1 - (1 - 0.25) ** 4, 2000)
    n = n[(n >= 5) | (rng.uniform(size=len(n)) < 0.1)]       # most sparse ones lost
    fitted = labeling_efficiency(n, LabelingEfficiencySettings(fit_min=5))
    assert abs(fitted["efficiency"] - 0.25) < 3 * fitted["error"] + 0.01
    naive = labeling_efficiency(n, LabelingEfficiencySettings(fit_min=0))
    assert naive["efficiency"] > fitted["efficiency"] + 0.02


def test_too_few_pores_in_the_fit_range_is_refused():
    with pytest.raises(ValueError, match="at least 5"):
        labeling_efficiency([8, 8, 7, 1, 2], LabelingEfficiencySettings())


# ------------------------------------------------------------- the plugins

def test_segment_evaluate_and_analyse_through_the_session():
    """The whole chain on grouped localizations, against the simulation."""
    from smappy.session import Session
    from smappy.workspace import Instance
    session = Session(pores(0.5, seed=2))
    session.show_grouped(0, True)
    found = plugins.get("ROIManager/Segment/NPC")()(ctx=session.context())
    assert "pores of" in found.text and found.plots["checks"].panels == 3
    project = session.rois
    assert len(project.rois) == len(found.data["centers"]) > 140

    project.evaluate([Instance(plugin="ROIManager/Evaluate/NPC Corners")])
    rows = project.results()
    assert len(rows) == len(project.rois)
    assert {"n_corners", "n_localizations", "n_corners_smap"} <= set(rows[0])

    analyze = plugins.get("ROIManager/Analyze/NPC Labeling Efficiency")
    result = analyze()(ctx=session.context())
    data = result.data
    assert data["simulated_efficiency"] == 0.5
    assert abs(data["efficiency"] - 0.5) < 0.04
    assert abs(data["efficiency"] - data["true_efficiency"]) < 0.03
    assert abs(data["blinks"] - 3.0) < 0.4
    assert "labelling efficiency" in result.text
    assert result.plot is not None and result.plot.panels == 2

    smap = analyze()(ctx=session.context(),
                     settings=LabelingEfficiencySettings(method="smap"))
    assert "SMAP" in smap.text and 0.3 < smap.data["efficiency"] < 0.7


def test_the_analysis_asks_for_the_corner_counter_when_it_is_missing():
    from smappy.plugins import Context
    with pytest.raises(ValueError, match="NPC Corners"):
        plugins.get("ROIManager/Analyze/NPC Labeling Efficiency")().run(
            Context(site_table=[{"roi_id": "a", "n_localizations": 3}]),
            LabelingEfficiencySettings())


def test_the_npc_chain_gives_what_the_three_steps_give_and_leaves_the_pipeline_alone():
    """The shipped NPC Analysis is a chain: segment, evaluate every pore,
    analyse.  Its ROIs and counts reach the session's project; the evaluation
    window's pipeline stays what the user set up."""
    from smappy.session import Session
    from smappy.workspace import Instance
    session = Session(pores(0.5, seed=2))
    project = session.rois
    project.pipeline = [Instance(plugin="ROIManager/Evaluate/Statistics")]
    file_id = next(iter(project.sources))
    drawn = project.add_roi(file_id, (100.0, 100.0))       # by hand, and empty
    chain = plugins.get("ROIManager/Workflow/NPC Analysis")
    result = session.run(chain(), chain.Settings())
    assert session.layers[0].grouped                       # the chain's first step
    steps = result.data["results"]
    found = steps["find_the_pores"].data["rois"]
    assert len(found) > 140
    # every included ROI is evaluated; the empty one fails on its own
    counted = steps["npc_corners"].data
    assert counted["sites"] == len(found) + 1 and counted["failed"] == 1
    assert "failed on 1: ValueError: no localizations" in steps["npc_corners"].text
    fitted = steps["labelling_efficiency"].data
    assert abs(fitted["efficiency"] - 0.5) < 0.04
    assert {"find the pores", "find the pores: checks",
            "labelling efficiency"} <= set(result.plots)
    assert drawn.id in project.rois and len(project.rois) == len(found) + 1
    assert [i.plugin for i in project.pipeline] == ["ROIManager/Evaluate/Statistics"]
    assert project.runs[-1]["chain"] == "ROIManager/Workflow/NPC Analysis"

    # the analysis run from the GUI afterwards finds the chain's counts
    analysis = plugins.get("ROIManager/Analyze/NPC Labeling Efficiency")
    again = analysis()(ctx=session.context())
    assert abs(again.data["efficiency"] - fitted["efficiency"]) < 1e-6
    # and so does the analysis by hand, of the same rows
    rows = project.results()
    assert len(rows) == len(found)
    by_hand = analysis().analyse(project, rows, LabelingEfficiencySettings(),
                                 NPCCornersSettings())
    assert abs(by_hand.data["efficiency"] - fitted["efficiency"]) < 1e-6

    rerun = session.run(chain(), chain.Settings())
    assert len(project.rois) == len(found) + 1 and drawn.id in project.rois
    assert abs(rerun.data["results"]["labelling_efficiency"].data["efficiency"]
               - fitted["efficiency"]) < 1e-6


def test_a_renamed_corner_count_stands_beside_the_first_and_is_chosen_by_name():
    """Two settings of NPC Corners, under two names, to compare: the analysis
    takes the one not renamed, or the one named, with that one's cutoffs."""
    from smappy.session import Session
    from smappy.workspace import Instance
    session = Session(pores(0.5, seed=2))
    session.show_grouped(0, True)
    project = session.rois
    plugins.get("ROIManager/Segment/NPC")()(ctx=session.context())
    project.evaluate([Instance(plugin="ROIManager/Evaluate/NPC Corners")])
    project.evaluate([Instance(plugin="ROIManager/Evaluate/NPC Corners", label="strict",
                               values={"precision_nm": 12.0})])
    rows = project.results()
    assert "NPC Corners.n_corners" in rows[0] and "strict.n_corners" in rows[0]
    assert all(r["strict.n_corners"] <= r["NPC Corners.n_corners"] for r in rows)
    analysis = plugins.get("ROIManager/Analyze/NPC Labeling Efficiency")
    plain = analysis()(ctx=session.context())
    strict = analysis()(ctx=session.context(),
                        settings=LabelingEfficiencySettings(evaluation="strict"))
    assert "counted by 'strict'" in strict.text and "counted by" not in plain.text
    # both answer the same question, each with its own cutoffs
    assert abs(plain.data["efficiency"] - 0.5) < 0.05
    assert abs(strict.data["efficiency"] - 0.5) < 0.05
    assert plain.data["counts"] != strict.data["counts"]
    with pytest.raises(ValueError, match="no NPC Corners evaluation is called 'loose'"):
        analysis()(ctx=session.context(),
                   settings=LabelingEfficiencySettings(evaluation="loose"))
    # running the first again under its own name replaces it, not adds a third
    project.evaluate([Instance(plugin="ROIManager/Evaluate/NPC Corners",
                               values={"precision_nm": 25.0})])
    assert project.labels("ROIManager/Evaluate/NPC Corners") == ["NPC Corners", "strict"]
    _, count = __import__("smappy.plugins.npc", fromlist=["x"]).corner_evaluation(
        project, project.results())
    assert count.precision_nm == 25.0


def test_the_segmenter_replaces_only_its_own_pores_when_asked():
    from smappy.session import Session
    session = Session(pores(0.5, seed=3))
    session.show_grouped(0, True)
    project = session.rois
    file_id = next(iter(project.sources))
    drawn = project.add_roi(file_id, (100.0, 100.0))
    segment = plugins.get("ROIManager/Segment/NPC")
    first = len(segment()(ctx=session.context()).data["rois"])
    assert len(segment()(ctx=session.context()).data["rois"]) == 0   # all suppressed
    again = segment()(ctx=session.context(), settings=NPCSegmentSettings(replace=True))
    assert len(again.data["rois"]) == first
    assert len(project.rois) == first + 1 and drawn.id in project.rois


def test_the_analysis_says_when_the_layer_is_not_grouped():
    from smappy.session import Session
    from smappy.workspace import Instance
    session = Session(pores(0.5, seed=4))
    session.show_grouped(0, False)
    project = session.rois
    plugins.get("ROIManager/Segment/NPC")()(ctx=session.context())
    project.evaluate([Instance(plugin="ROIManager/Evaluate/NPC Corners")])
    result = plugins.get("ROIManager/Analyze/NPC Labeling Efficiency")()(
        ctx=session.context())
    assert "not grouped" in result.text
