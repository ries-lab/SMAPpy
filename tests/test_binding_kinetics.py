"""Binding kinetics and qPAINT: events and dark times checked on frames
written by hand, the rates and the site counts on simulated DNA-PAINT
(`simulate.simulate_binding_sites`) with known bright and dark times.

A binding shorter than about half a frame lights no frame and is not seen,
which makes the dark times look longer: by about 12 % at a mean bright
time of 5 frames, less at 20.  The rates are checked where that is small; the site
counts are calibrated, and the loss is the same for every site.
"""
import numpy as np
import pytest

from smappy import binding_kinetics as bk
from smappy.chain import Block, ChainSpec, Step, chain_class
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.binding_kinetics import (PATH, BindingKinetics, Qpaint,
                                             QpaintSettings, acquisition)
from smappy.session import Session
from smappy.simulate import simulate_binding_sites, spot_centres

N_FRAMES = 20000


# ------------------------------------------------------------- by hand

def test_localizations_are_linked_into_events_across_the_link_gap():
    frame = np.array([0, 1, 2, 5, 20, 21, 30])
    events = bk.binding_events(frame, link_gap=3)
    assert events.first.tolist() == [0, 20, 30]
    assert events.last.tolist() == [5, 21, 30]
    assert events.n.tolist() == [4, 2, 1]
    assert events.bright.tolist() == [6, 2, 1]
    gaps = bk.dark_times(events)
    assert np.isnan(gaps[0]) and gaps[1:].tolist() == [14, 8]
    # a gap of four frames is a dark time at a link gap of three
    assert len(bk.binding_events(np.array([0, 5]), link_gap=3)) == 2
    assert len(bk.binding_events(np.array([0, 4]), link_gap=3)) == 1


def test_each_file_keeps_its_events_and_its_first_dark_time_apart():
    frame = np.array([0, 10, 100, 0, 50])
    file = np.array([0, 0, 0, 1, 1])
    events = bk.binding_events(frame, file=file, link_gap=0)
    assert events.file.tolist() == [0, 0, 0, 1, 1]
    gaps = bk.dark_times(events)
    assert np.isnan(gaps[0]) and np.isnan(gaps[3])
    assert gaps[[1, 2, 4]].tolist() == [9, 89, 49]


def test_the_likelihood_counts_the_ends_and_takes_off_the_link_gap():
    total, periods = bk.dark_ml([14.0, 8.0, np.nan], link_gap=3, ends=(10, 20))
    assert total == pytest.approx(14 + 8 - 2 * 3.5 + 30)
    assert periods == 3
    total, periods = bk.dark_ml([14.0, 8.0], link_gap=3)
    assert (total, periods) == (pytest.approx(15.0), 2)


def test_values_of_a_region_by_hand():
    frame = np.array([10, 11, 12, 50, 90, 91])
    values, arrays, events = bk.region_kinetics(frame, link_gap=0, spans={0: (0, 99)},
                                                frame_time_ms=100.0)
    assert values["n_events"] == 3 and values["n_localizations"] == 6
    assert values["first_frame"] == 10 and values["last_frame"] == 91
    assert values["longest_dark_frames"] == 39 and values["longest_bright_frames"] == 3
    assert values["mean_bright_frames"] == pytest.approx(2.0)
    # 10 dark frames before, 8 after, 37 and 39 between less half a frame each
    total = 10 + 8 + 36.5 + 38.5
    assert values["dark_total_frames"] == pytest.approx(total)
    assert values["dark_periods"] == 3
    assert values["tau_dark_frames"] == pytest.approx(total / 3)
    assert values["qpaint_index"] == pytest.approx(2 / total)
    assert values["tau_dark_s"] == pytest.approx(values["tau_dark_frames"] / 10)
    assert np.isnan(values["n_sites"])
    assert arrays["events"].sum() == 3 and arrays["cumulative"][-1] == pytest.approx(1)
    assert arrays["bright_frames"].tolist() == [3, 1, 2]


def test_an_empty_region_is_an_answer():
    values, arrays, events = bk.region_kinetics(np.zeros(0, np.int64), spans={0: (0, 9)})
    assert values["n_events"] == 0 and np.isnan(values["qpaint_index"])


def test_picassos_fit_is_the_mean_for_two_values_and_tau_for_many():
    assert bk.fit_cumulative([3.0, 5.0]) == pytest.approx(4.0)
    rng = np.random.default_rng(1)
    assert bk.fit_cumulative(rng.exponential(50.0, 2000)) == pytest.approx(50, rel=0.1)


# ------------------------------------------------------------- simulated

def spots(locs):
    frame, spot = np.asarray(locs["frame"]), np.asarray(locs["spot_true"])
    return [frame[spot == k + 1] for k in range(len(locs.metadata["sites_true"]))]


def test_the_bright_and_dark_times_simulated_are_found():
    locs = simulate_binding_sites([1] * 30, N_FRAMES, tau_bright=20.0, tau_dark=800.0,
                                  seed=2)
    found = [bk.region_kinetics(f, spans={0: (0, N_FRAMES - 1)})[0] for f in spots(locs)]
    index, _ = bk.pooled_index([v["dark_total_frames"] for v in found],
                               [v["dark_periods"] for v in found])
    assert 1 / index == pytest.approx(800.0, rel=0.1)
    bright = np.mean([v["tau_bright_frames"] for v in found])
    assert bright == pytest.approx(20.0, rel=0.1)


def test_the_censored_likelihood_is_unbiased_where_the_mean_between_events_is_not():
    # few events per site: leaving out the ends makes tau short by n / (n + 1)
    locs = simulate_binding_sites([1] * 60, N_FRAMES, tau_bright=20.0, tau_dark=3000.0,
                                  seed=4)
    ml, between = [], []
    for f in spots(locs):
        known = bk.region_kinetics(f, spans={0: (0, N_FRAMES - 1)})[0]
        alone = bk.region_kinetics(f)[0]
        ml.append((known["dark_total_frames"], known["dark_periods"]))
        between.append((alone["dark_total_frames"], alone["dark_periods"]))
    tau_ml = 1 / bk.pooled_index(*zip(*ml))[0]
    tau_between = 1 / bk.pooled_index(*zip(*between))[0]
    assert tau_ml == pytest.approx(3000.0, rel=0.1)
    assert tau_between < 0.92 * tau_ml


def test_sites_come_back_from_an_influx_calibrated_on_single_sites():
    sites = [1] * 20 + [2] * 10 + [4] * 10
    locs = simulate_binding_sites(sites, N_FRAMES, tau_bright=5.0, tau_dark=1000.0,
                                  seed=3)
    found = [bk.region_kinetics(f, spans={0: (0, N_FRAMES - 1)})[0] for f in spots(locs)]
    influx, _ = bk.pooled_index([v["dark_total_frames"] for v in found[:20]],
                                [v["dark_periods"] for v in found[:20]])
    counted = bk.sites_from([v["qpaint_index"] for v in found], influx)
    sites = np.array(sites)
    for n in (1, 2, 4):
        assert counted[sites == n].mean() == pytest.approx(n, rel=0.15)
    # one site is told from four
    assert counted[sites == 1].max() < counted[sites == 4].min()


def test_a_sticky_event_shows_in_the_curation_values():
    locs = simulate_binding_sites([1, 1], N_FRAMES, tau_bright=5.0, tau_dark=1000.0,
                                  sticky=[(1, 8000, 8600)], seed=5)
    good, sticky = (bk.region_kinetics(f, spans={0: (0, N_FRAMES - 1)})[0]
                    for f in spots(locs))
    assert sticky["longest_bright_frames"] >= 600 > 10 * good["longest_bright_frames"]
    assert sticky["peak_window_fraction"] > 0.8 > good["peak_window_fraction"]
    assert 0.2 < good["mean_frame_rel"] < 0.8
    assert good["std_frame_rel"] == pytest.approx(0.29, abs=0.08)


def test_a_site_lost_half_way_fails_the_uniform_test():
    locs = simulate_binding_sites([4], N_FRAMES, tau_bright=5.0, tau_dark=1000.0, seed=6)
    frame = spots(locs)[0]
    whole = bk.region_kinetics(frame, spans={0: (0, N_FRAMES - 1)})[0]
    half = bk.region_kinetics(frame[frame < N_FRAMES // 2], spans={0: (0, N_FRAMES - 1)})[0]
    assert whole["uniform_p"] > 0.01 > half["uniform_p"]
    assert half["mean_frame_rel"] < 0.3


# ------------------------------------------------------------- the plugin

def test_on_the_view_the_selection_is_one_site():
    locs = simulate_binding_sites([2], N_FRAMES, seed=7)
    result = BindingKinetics()(locs)
    assert result.values["n_events"] > 10
    assert result.values["tau_dark_s"] == pytest.approx(
        result.values["tau_dark_frames"] * 0.1)
    assert "events" in result.text and result.plot is not None
    again = BindingKinetics().restore(BindingKinetics().keep(result))
    assert again is not None and again.plot is not None


def test_the_acquisition_comes_from_the_setting_the_file_or_the_session():
    locs = Localizations({"frame": np.array([5, 9])}, {"n_frames": 100})
    ctx = Context(locs=locs)
    assert acquisition(ctx, locs, [0], None) == {0: (0, 99)}
    assert acquisition(ctx, locs, [0], 50) == {0: (0, 49)}
    bare = Localizations({"frame": np.array([5, 9])}, {})
    assert acquisition(Context(locs=bare), bare, [0], None) is None
    session = Session(Localizations({"frame": np.array([3, 400]),
                                     "x_nm": np.zeros(2), "y_nm": np.zeros(2)}, {}))
    assert acquisition(Context(locs=bare, session=session), bare, [0], None) == {0: (3, 400)}


def kinetics_session(sites, **simulated):
    """A session over simulated spots, an ROI on each, the calibration spots
    (one site) annotated, and Binding kinetics run in every ROI."""
    locs = simulate_binding_sites(sites, N_FRAMES, **simulated)
    session = Session(locs)
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(200.0, "circle")
    file_id = next(iter(project.sources))
    project.set_annotation_lists({"kind": ["calibration", "sample"]})
    for n, centre in zip(sites, spot_centres(locs)):
        roi = project.add_roi(file_id, list(centre))
        roi_id = roi if isinstance(roi, str) else getattr(roi, "id", roi)
        project.annotate(roi_id, "kind", "calibration" if n == 1 else "sample")
    cls = chain_class(ChainSpec(steps=[Block(steps=[Step(plugin=PATH)])]))
    session.run(cls(), cls.Settings())
    return session


def test_qpaint_calibrates_on_the_keyword_and_counts_the_rest():
    sites = [1] * 12 + [3] * 6
    session = kinetics_session(sites, tau_bright=5.0, tau_dark=1000.0, seed=8)
    result = session.run(Qpaint(), QpaintSettings(keyword_list="kind",
                                                  keyword="calibration"))
    assert len(result.data["calibration"]) == 12
    counted = np.array(list(result.data["sites"].values()))
    assert np.mean(counted[:12]) == pytest.approx(1.0, rel=0.15)
    assert np.mean(counted[12:]) == pytest.approx(3.0, rel=0.15)
    assert "influx" in result.text and result.plot is not None
    # an influx typed in is used as it is
    fixed = session.run(Qpaint(), QpaintSettings(influx=0.002))
    assert fixed.data["influx"] == 0.002
    assert Qpaint().describe(session.context(), QpaintSettings()).choices[
        "keyword_list"][1] == ("kind", "kind")


def test_the_roi_rows_keep_the_trace_to_draw_again():
    session = kinetics_session([1, 2], seed=9)
    project = session.rois
    roi_id = project.ordered_ids()[0]
    entry = project.evaluations(roi_id)["Binding kinetics"][0]
    assert entry["kept"]["first"] and entry["kept"]["span"] == [0, N_FRAMES - 1]
    assert BindingKinetics().restore(entry["kept"]).plot is not None
    assert len(entry["arrays"]["frame"]) == len(entry["arrays"]["cumulative"])
