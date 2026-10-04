"""Nuclear pores: segmenting, counting corners, the labelling efficiency."""
import numpy as np
import pytest

from smappy import plugins
from smappy.locs import Localizations
from smappy.plugins.npc import (FITS, LabelingEfficiencySettings, NPCCornersSettings,
                                NPCSegmentSettings, count_corners, fit_circle,
                                labeling_efficiency, ring_kernel, ring_quality,
                                segment_npcs, true_corners)
from smappy.simulate import simulate
from smappy.simulate.settings import (LabellingSettings, SimulationSettings,
                                      StructureSettings)

R = 53.7


def pores(efficiency, seed=1, n_frames=3000):
    return simulate(SimulationSettings(
        n_frames=n_frames, seed=seed, structure=StructureSettings(preset="npc"),
        labelling=LabellingSettings(efficiency=efficiency)))


def corners_at(which, rng, n=15, rotation=0.0, center=(0.0, 0.0), sd=6.0):
    """Localizations at some of the eight corners of a pore."""
    xy = [np.column_stack((center[0] + R * np.cos(rotation + k * np.pi / 4)
                           + rng.normal(0, sd, n),
                           center[1] + R * np.sin(rotation + k * np.pi / 4)
                           + rng.normal(0, sd, n))) for k in which]
    return np.concatenate(xy)


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
    assert len(used) == len(truth)
    assert len(set(index)) == len(truth)
    assert np.median(distance) < 3 and distance.max() < 10
    radius = np.array([s["radius_nm"] for s in sites if s["use"]])
    assert abs(np.median(radius) - R) < 3


def test_a_blob_fails_on_its_radius_and_an_arc_only_on_its_spread():
    """Why the spread check exists, and why it is off by default: a single
    corner collapses the free radius, but three adjacent corners fit a
    perfectly good radius -- whether they are a sparse pore or an arc of the
    neighbour's ring, only the spread tells them apart from a full ring."""
    rng = np.random.default_rng(0)
    settings = NPCSegmentSettings(min_spread_nm=25.0)
    blob = corners_at([0], rng)
    assert {"radius", "spread"} <= set(ring_quality(blob[:, 0], blob[:, 1],
                                                    (0, 0), settings)["failed"])
    arc = corners_at([0, 1, 2], rng)
    assert ring_quality(arc[:, 0], arc[:, 1], (0, 0), settings)["failed"] == ["spread"]
    full = corners_at(range(8), rng)
    quality = ring_quality(full[:, 0], full[:, 1], (5, -5), settings)
    assert quality["use"] and abs(quality["radius_nm"] - R) < 3
    assert quality["fraction_inside"] < 0.05 and quality["fraction_outside"] < 0.05


def test_a_filled_disc_is_rejected_for_what_is_inside_the_ring():
    rng = np.random.default_rng(2)
    r = 70 * np.sqrt(rng.uniform(0, 1, 400))
    a = rng.uniform(0, 2 * np.pi, 400)
    quality = ring_quality(r * np.cos(a), r * np.sin(a), (0, 0), NPCSegmentSettings())
    assert "inside" in quality["failed"]


def test_rejected_candidates_become_unused_rois_only_when_asked():
    from smappy.roi_manager import ROIProject
    from smappy.plugins.npc import NPCSegment
    rng = np.random.default_rng(3)
    good = corners_at(range(8), rng, center=(1000, 1000))
    filled = 70 * np.sqrt(rng.uniform(0, 1, 300))
    angle = rng.uniform(0, 2 * np.pi, 300)
    disc = np.column_stack((3000 + filled * np.cos(angle), 1000 + filled * np.sin(angle)))
    xy = np.concatenate([good, disc])
    locs = Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1],
                          "frame": np.arange(len(xy))})
    for keep, expected in ((False, [True]), (True, [True, False])):
        project = ROIProject()
        file_id = project.add_source(locs).id
        found = project.find(file_id, plugin=NPCSegment(),
                             settings=NPCSegmentSettings(keep_rejected=keep))
        assert [roi.use for roi in found] == expected
        assert np.hypot(*(np.asarray(found[0].center) - 1000)) < 3
        assert found[0].origin["method"] == "ROIManager/Segment/NPC"
        assert abs(found[0].origin["radius_nm"] - R) < 3
    assert "inside" in found[1].origin["failed"]


# ------------------------------------------------------------------ corners

@pytest.mark.parametrize("which", [[0], [0, 4], [1, 2, 5], [0, 1, 2, 3, 5, 7], range(8)])
def test_the_corners_with_a_localization_are_counted_at_any_rotation(which):
    rng = np.random.default_rng(len(list(which)))
    xy = corners_at(which, rng, n=4, rotation=0.37, center=(500, 200), sd=4)
    values = count_corners(xy[:, 0], xy[:, 1], (510, 190), NPCCornersSettings(),
                           precision=np.full(len(xy), 5.0))
    assert values["n_corners"] == len(list(which))
    if len(list(which)) >= 3:       # one or two corners do not fix a centre
        assert abs(values["x_nm"] - 500) < 4 and abs(values["y_nm"] - 200) < 4


def test_an_imprecise_localization_does_not_open_a_corner():
    rng = np.random.default_rng(4)
    xy = corners_at([0, 2], rng, n=5, sd=3)
    stray = np.array([[R * np.cos(np.pi / 2), R * np.sin(np.pi / 2)]])
    xy = np.concatenate([xy, stray])
    precision = np.r_[np.full(len(xy) - 1, 5.0), 30.0]
    values = count_corners(xy[:, 0], xy[:, 1], (0, 0), NPCCornersSettings(), precision)
    assert values["n_corners"] == 2


# ----------------------------------------------------- labelling efficiency

@pytest.mark.parametrize("efficiency", [0.3, 0.5, 0.65])
def test_both_fits_recover_the_efficiency_of_binomial_corners(efficiency):
    rng = np.random.default_rng(int(100 * efficiency))
    p_corner = 1 - (1 - efficiency) ** 4
    n = rng.binomial(8, p_corner, 400)
    found = {fit: labeling_efficiency(n, LabelingEfficiencySettings(fit=fit))
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


@pytest.mark.parametrize("efficiency", [0.3, 0.5])
def test_the_chain_recovers_the_labelling_of_a_simulation(efficiency):
    """Segment, count, fit -- against the corners that were truly labelled.

    The comparison is with the labelled corners of the same simulation, which
    is what the method measures; below about 0.2 a stray localization between
    two corners opens a corner often enough to bias it upwards."""
    locs = pores(efficiency, seed=2)
    from scipy.spatial import cKDTree
    xy = np.column_stack((locs["x_nm"], locs["y_nm"]))
    tree = cKDTree(xy)
    precision = np.asarray(locs["xy_err_nm"])
    n = []
    for site in segment_npcs(locs, NPCSegmentSettings()):
        if site["use"]:
            near = tree.query_ball_point(site["center"], 150)
            n.append(count_corners(xy[near, 0], xy[near, 1], site["center"],
                                   NPCCornersSettings(), precision[near])["n_corners"])
    measured = labeling_efficiency(n, LabelingEfficiencySettings())
    labelled = labeling_efficiency(list(true_corners(locs).values()),
                                   LabelingEfficiencySettings(bootstrap=0))
    assert abs(labelled["efficiency"] - efficiency) < 0.03
    assert abs(measured["efficiency"] - labelled["efficiency"]) < 0.025


# ------------------------------------------------------------- the plugins

def test_segment_evaluate_and_analyse_through_the_session():
    from smappy.session import Session
    from smappy.workspace import Instance
    session = Session(pores(0.5, seed=2))
    session.show_grouped(0, False)
    found = plugins.get("ROIManager/Segment/NPC")()(ctx=session.context())
    assert "pores of" in found.text and found.plots["checks"].panels == 4
    project = session.rois
    assert len(project.rois) == len(found.data["centers"]) > 100

    project.evaluate([Instance(plugin="ROIManager/Evaluate/NPC Corners")])
    rows = project.results()
    assert len(rows) == len(project.rois)
    assert {"n_corners", "radius_nm", "rotation_deg"} <= set(rows[0])

    result = plugins.get("ROIManager/Analyze/NPC Labeling Efficiency")()(
        ctx=session.context())
    data = result.data
    assert data["simulated_efficiency"] == 0.5
    assert abs(data["efficiency"] - data["true_efficiency"]) < 0.025
    assert abs(data["efficiency"] - 0.5) < 3 * data["error"] + 0.02
    assert "labelling efficiency" in result.text and result.plot is not None


def test_the_analysis_asks_for_the_corner_counter_when_it_is_missing():
    from smappy.plugins import Context
    with pytest.raises(ValueError, match="NPC Corners"):
        plugins.get("ROIManager/Analyze/NPC Labeling Efficiency")().run(
            Context(site_table=[{"roi_id": "a", "n_localizations": 3}]),
            LabelingEfficiencySettings())
