"""Segment regions' *like examples*: Picasso's pick similar, a few ROIs
marked with a keyword standing for every structure like them."""
import numpy as np
import pytest

from smappy import segmentation
from smappy.locs import Localizations
from smappy.plugins.segment import (PATH, SegmentRegions, SegmentSettings,
                                    SimilarSettings, made_by)
from smappy.session import Session


def scene(seed=1, size=20000.0):
    """150 discs of 40 nm with 40 localizations each, beside two kinds of
    clutter -- 40 large blobs of 160, 60 small ones of 12 -- and uniform
    background: the positions and the three sets of centres."""
    rng = np.random.default_rng(seed)

    def blobs(centres, r, n):
        out = []
        for c in centres:
            k = rng.poisson(n)
            a = rng.uniform(0, 2 * np.pi, k)
            rr = r * np.sqrt(rng.uniform(0, 1, k))
            out.append(np.column_stack((c[0] + rr * np.cos(a), c[1] + rr * np.sin(a)))
                       + rng.normal(0, 8, (k, 2)))
        return np.vstack(out)

    good, big, small = (rng.uniform(500, size - 500, (n, 2)) for n in (150, 40, 60))
    xy = np.vstack([blobs(good, 40, 40), blobs(big, 90, 160), blobs(small, 15, 12),
                    rng.uniform(0, size, (4000, 2))])
    return xy, good, big, small


def table(xy):
    n = len(xy)
    return Localizations({"x_nm": xy[:, 0], "y_nm": xy[:, 1], "frame": np.arange(n),
                          "xy_err_nm": np.full(n, 8.0)})


def nearest(points, others):
    """Each point's distance to the nearest of ``others``."""
    if not len(others):
        return np.full(len(points), np.inf)
    d = np.hypot(*(np.asarray(points)[:, None] - np.asarray(others)[None]).transpose(2, 0, 1))
    return d.min(axis=1)


# ------------------------------------------------------------------ the arrays

def test_five_examples_find_most_discs_like_them_and_none_of_the_clutter():
    xy, good, big, small = scene()
    locs = table(xy)
    rng = np.random.default_rng(2)
    # drawn by hand: a little off each structure's centre
    examples = segmentation.measure_at(locs, good[:5] + rng.normal(0, 15, (5, 2)), 75)
    window = segmentation.similarity_window(examples, segmentation.SIMILAR_COMPARED)
    sites = segmentation.similar_candidates(locs, 75, window,
                                            existing=[e["center"] for e in examples])
    segmentation.judge(sites, segmentation.window_condition(window),
                       segmentation.VALUES["similar"])
    kept = [s["center"] for s in sites if s["use"]]
    assert np.mean(nearest(good[5:], kept) < 20) >= 0.75
    assert not np.any(nearest(big, kept) < 40)
    assert not np.any(nearest(small, kept) < 20)
    # the examples' own structures are not found again
    assert not np.any(nearest(good[:5], kept) < 20)


def test_the_mean_shift_settles_a_circle_on_its_structures_centre_of_mass():
    rng = np.random.default_rng(0)
    xy = rng.normal((1000.0, 2000.0), 20, (300, 2))
    settled = segmentation.mean_shift(xy, [(1040.0, 1960.0)], 80.0)[0]
    assert np.allclose(settled, xy.mean(axis=0), atol=1.0)


def test_a_small_structure_anywhere_between_the_lattice_points_is_found():
    rng = np.random.default_rng(3)
    corners = rng.uniform(0, 3000, (40, 2))
    for k, centre in enumerate(rng.uniform(500, 2500, (12, 2))):
        xy = np.vstack([corners, rng.normal(centre, 5, (30, 2))])
        sites = segmentation.similar_candidates(table(xy), 50.0)
        found = [s["center"] for s in sites if s["values"]["n_locs"] >= 25]
        assert nearest([centre], found)[0] < 5, k


def test_the_window_widens_a_few_examples_spread_and_needs_two_of_them():
    examples = [{"values": {"n_locs": n, "rmsd_nm": r}}
                for n, r in ((40, 30.0), (41, 31.0), (40, 32.0))]
    window = segmentation.similarity_window(examples, ("n_locs", "rmsd_nm"), 2.0)
    # the count's spread is at least Poisson's, sqrt(40.33)
    assert window["n_locs"] == pytest.approx((40.33 - 2 * 6.351, 40.33 + 2 * 6.351),
                                             abs=0.01)
    # the sample standard deviation of 30, 31, 32 is 1
    assert window["rmsd_nm"] == pytest.approx((29.0, 33.0))
    few = [{"values": {"n_locs": 4, "rmsd_nm": 10.0}}, {"values": {"n_locs": 5,
                                                                   "rmsd_nm": 11.0}}]
    assert segmentation.similarity_window(few, ("n_locs",), 3.0)["n_locs"][0] == 3.0
    with pytest.raises(ValueError, match="at least 2"):
        segmentation.similarity_window(examples[:1], ("n_locs",))
    assert segmentation.window_condition({"n_locs": (12.0, 30.25)}) \
        == "12 <= n_locs <= 30.25"


# ------------------------------------------------------------------ the plugin

def session_with_examples(n_examples=5, seed=1, shape="circle"):
    """The scene in a session, with an ROI of the manager's shape put down by
    hand on each of the first discs."""
    xy, good, big, small = scene(seed)
    session = Session(table(xy))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(150.0, shape)
    project.set_annotation_lists({"kind": ["example", "junk"]})
    file_id = next(iter(project.sources))
    examples = [project.add_roi(file_id, list(centre)) for centre in good[:n_examples]]
    return session, good, examples


SIMILAR = SegmentSettings(method="similar", keep="")


def test_the_used_rois_are_the_examples_and_make_rois_like_them():
    session, good, examples = session_with_examples()
    result = SegmentRegions().run(session.context(), SIMILAR)
    made = [r for r in result.rois if r.use]
    assert all(made_by(r, "similar") for r in result.rois)
    assert np.mean(nearest(good[5:], [r.center for r in made]) < 20) >= 0.75
    assert "learned from 5 examples" in result.text and "n_locs <=" in result.text
    assert set(result.data["window"]) == {"n_locs", "rmsd_nm"}
    # the examples stay as they were, and are not found twice
    assert all(r.id in session.rois.rois for r in examples)
    assert not np.any(nearest([r.center for r in examples], [r.center for r in made]) < 20)
    assert {"rmsd_nm", "n_locs"} <= set(made[0].origin["values"])


def test_an_example_is_everything_its_roi_holds_whatever_its_shape():
    session, good, examples = session_with_examples(n_examples=2)
    project = session.rois
    # a free ROI: a square polygon of 150 nm round the disc
    cx, cy = good[2]
    polygon = [[cx - 75, cy - 75], [cx + 75, cy - 75], [cx + 75, cy + 75], [cx - 75, cy + 75]]
    free = project.add_roi(examples[0].file_id, [cx, cy], polygon=polygon)
    state = project.state(free.file_id)
    rows = project.indices(free)
    measured = segmentation.measure_rows(state.locs, rows)
    inside = np.column_stack((state.locs["x_nm"][rows], state.locs["y_nm"][rows]))
    assert measured["values"]["n_locs"] == len(rows) > 20
    assert measured["center"] == pytest.approx(inside.mean(axis=0).tolist())
    assert measured["values"]["rmsd_nm"] == pytest.approx(
        np.sqrt(np.mean(np.sum((inside - inside.mean(axis=0)) ** 2, axis=1))))


def test_an_unused_roi_is_no_example_unless_all_are_asked_for():
    session, good, examples = session_with_examples(n_examples=3)
    examples[0].use = False
    examples[1].use = False
    with pytest.raises(ValueError, match="1 used ROIs as examples"):
        SegmentRegions().run(session.context(), SIMILAR)
    every = SegmentSettings(method="similar", keep="",
                            similar=SimilarSettings(examples="all"))
    assert "learned from 3 examples" in SegmentRegions().run(session.context(), every).text


def test_a_second_run_learns_from_the_same_examples_and_replaces_the_first():
    session, good, examples = session_with_examples()
    project = session.rois
    first = SegmentRegions().run(session.context(), SIMILAR)
    again = SegmentRegions().run(session.context(), SIMILAR)
    assert "learned from 5 examples" in again.text
    assert again.data["window"] == first.data["window"]
    assert not {r.id for r in first.rois} & set(project.rois)
    assert all(r.id in project.rois for r in examples)


def test_with_a_keyword_a_found_roi_given_it_becomes_an_example_replace_leaves():
    session, good, examples = session_with_examples()
    project = session.rois
    for roi in examples:
        project.annotate(roi.id, "kind", "example")
    keyword = SegmentSettings(method="similar", keep="", similar=SimilarSettings(
        examples="keyword", keyword_list="kind", keyword="example"))
    first = SegmentRegions().run(session.context(), keyword).rois
    promoted = next(r for r in first if r.use)
    project.annotate(promoted.id, "kind", "example")
    again = SegmentRegions().run(session.context(), keyword)
    assert promoted.id in project.rois
    assert not ({r.id for r in first} - {promoted.id}) & set(project.rois)
    # a found ROI was not an example in the first place: still 5
    assert "learned from 5 examples" in again.text


def test_square_rois_make_square_candidates():
    session, good, examples = session_with_examples(shape="square")
    result = SegmentRegions().run(session.context(), SIMILAR)
    made = [r for r in result.rois if r.use]
    assert np.mean(nearest(good[5:], [r.center for r in made]) < 20) >= 0.7
    assert "squares of 150 nm" in SegmentRegions().describe(
        session.context(), SIMILAR).notice.text


def test_the_preview_proposes_what_a_run_adds_and_adds_nothing():
    session, good, examples = session_with_examples()
    before = set(session.rois.rois)
    preview = SegmentRegions().preview(session.context(), SIMILAR)
    assert set(session.rois.rois) == before
    assert preview.roi_preview["sites"]
    assert preview.text.startswith("preview: ")


def test_the_users_keep_joins_the_learned_window():
    session, good, examples = session_with_examples()
    settings = SegmentSettings(method="similar", keep="roundness >= 0.99")
    result = SegmentRegions().run(session.context(), settings)
    assert not any(r.use for r in result.rois)


@pytest.mark.parametrize("similar, message", [
    (SimilarSettings(examples="keyword"), "choose the keyword list"),
    (SimilarSettings(examples="keyword", keyword_list="nothing", keyword="example"),
     "no keyword list"),
    (SimilarSettings(examples="keyword", keyword_list="kind", keyword="junk"),
     "0 ROIs with the keyword 'junk' as examples"),
    (SimilarSettings(compare="far_p"), "compare names far_p"),
])
def test_what_is_missing_for_like_examples_is_said(similar, message):
    session, good, examples = session_with_examples(n_examples=3)
    settings = SegmentSettings(method="similar", keep="", similar=similar)
    with pytest.raises(ValueError, match=message):
        SegmentRegions().run(session.context(), settings)


def test_the_form_counts_the_examples_and_greys_the_keyword_unless_asked():
    session, good, examples = session_with_examples()
    state = SegmentRegions().describe(session.context(), SIMILAR)
    assert state.shown["similar"] and not state.shown["circle"]
    assert ("kind", "kind") in state.choices["similar.keyword_list"]
    assert not state.active["similar.keyword_list"]
    assert state.notice.text.startswith("5 examples; candidates are circles of 150 nm")
