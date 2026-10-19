import numpy as np
import pytest

from smappy.drift import Drift
from smappy.fiducials import (FiducialSettings, estimate_drift_fiducials,
                              find_beads, fit_drift)
from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.drift_fiducials import FiducialDrift
from smappy.simulate import simulate, simulate_beads, with_beads

N_FRAMES = 6000


def smooth_drift(n_frames=N_FRAMES):
    """A drift a spline can follow: slow creep and one wobble, zero at frame 0."""
    t = np.linspace(0.0, 1.0, n_frames)
    drift = np.column_stack([120 * t + 25 * np.sin(5 * t),
                             -80 * t ** 2 + 15 * np.sin(3 * t),
                             40 * t - 20 * np.sin(4 * t)])
    return drift - drift[0]


def beads_with_trouble(drift, **values):
    """Six beads: one interrupted for longer than the memory, one with a
    short gap, one that lands on the glass late."""
    return simulate_beads(len(drift), drift, n_beads=6, precision_nm=3.0,
                          z_precision_nm=8.0,
                          gaps=[(0, 1500, 2300), (1, 4000, 4030)],
                          late=[(2, 3000)], seed=4, **values)


def without_z(locs):
    return Localizations({k: v for k, v in locs.columns.items()
                          if not k.startswith("z_")}, dict(locs.metadata))


def test_beads_with_gaps_and_a_late_one_give_the_drift_within_their_precision():
    drift = smooth_drift()
    locs = without_z(beads_with_trouble(drift))
    found, beads, fit = estimate_drift_fiducials(locs)
    # bead 0's gap is longer than the memory: what comes before it is too
    # short to be a bead, what comes after is one; the short gap is bridged
    # and the late bead is in
    assert len(beads) == 6
    assert sorted(beads.first)[-2:] == [2301, 3000]
    error = found.drift[:, :2] - drift[:, :2]
    # six beads at 3 nm, a coefficient every 100 frames: well under one
    # bead's precision everywhere
    assert np.sqrt(np.mean(error ** 2, axis=0)).max() < 1.0
    assert np.abs(error).max() < 3.0
    assert not np.any(found.drift[:, 2])
    assert np.allclose(fit.rms_nm, 3.0, rtol=0.1)


def test_in_3d_the_axial_drift_comes_back_too():
    drift = smooth_drift()
    found, _, fit = estimate_drift_fiducials(beads_with_trouble(drift))
    error = found.drift - drift
    rms = np.sqrt(np.mean(error ** 2, axis=0))
    assert rms[:2].max() < 1.0 and rms[2] < 2.5
    assert np.allclose(fit.rms_nm, [3.0, 3.0, 8.0], rtol=0.1)


def test_a_late_bead_is_placed_by_its_fitted_offset_not_its_mean():
    """Two beads, one from frame 3000: subtracting each bead's mean (SMAP)
    would shift the second half of the curve by the mean drift of its half."""
    drift = smooth_drift()
    locs = without_z(simulate_beads(N_FRAMES, drift, n_beads=2, late=[(1, 3000)],
                                    seed=2))
    found, beads, _ = estimate_drift_fiducials(locs)
    assert len(beads) == 2
    assert np.abs(found.drift[:, :2] - drift[:, :2]).max() < 3.0


def test_the_curve_applied_removes_the_drift_from_beads_and_structure():
    locs = simulate(n_frames=N_FRAMES, seed=3, drift=True)
    truth = np.asarray(locs.metadata["drift_truth"])
    locs = with_beads(locs, n_beads=6, gaps=[(0, 1000, 1200)], late=[(1, 2500)],
                      seed=3)
    out = FiducialDrift().run(Context(locs=locs), FiducialSettings())
    drift = out.data["drift"]
    assert isinstance(drift, Drift) and drift.method == "fiducials"
    corrected = out.locs
    assert len(corrected) == len(locs)          # the beads stay in the table
    frame = np.asarray(locs["frame"])
    # every localization moved back by the true drift, less the reference
    for axis, name in enumerate(("x_nm", "y_nm", "z_nm")):
        expected = np.asarray(locs[name], float) - (truth[frame, axis] - truth[0, axis])
        residual = np.asarray(corrected[name], float) - expected
        assert np.sqrt(np.mean(residual ** 2)) < 4.0
    # and a bead, corrected, sits still
    bead = np.asarray(locs["bead_true"])
    for b in range(6):
        mine = bead == b
        assert np.std(np.asarray(corrected["x_nm"])[mine]) < 5.0
        assert np.std(np.asarray(locs["x_nm"])[mine]) > 20.0


def test_the_layers_filter_decides_what_may_be_a_bead():
    drift = smooth_drift()
    locs = without_z(beads_with_trouble(drift))
    x = np.asarray(locs["x_nm"])
    left = x < np.median(x)
    _, beads, fit = estimate_drift_fiducials(locs, select=left)
    assert np.all(left[fit.rows])
    assert 0 < len(beads) < 6


def test_dim_or_short_tracks_are_not_beads():
    frame = np.repeat(np.arange(1000), 1)
    x = np.zeros(1000)
    photons = np.full(1000, 20000.0)
    # a bright spot seen in 20% of the frames is too short at 25%
    seen = np.arange(1000) < 200
    beads = find_beads(x[seen], x[seen], frame[seen], photons[seen], n_frames=1000)
    assert len(beads) == 0
    beads = find_beads(x, x, frame, photons / 10)
    assert len(beads) == 0
    beads = find_beads(x, x, frame, photons)
    assert len(beads) == 1 and beads.count[0] == 1000


def test_no_beads_says_what_to_change():
    locs = simulate(n_frames=2000, seed=1)
    with pytest.raises(ValueError, match="min photons"):
        estimate_drift_fiducials(locs)


def test_the_drift_is_zero_at_the_first_frame():
    drift = smooth_drift() + 50.0
    fit = fit_drift(*_rows(simulate_beads(N_FRAMES, drift, n_beads=3, seed=1)),
                    N_FRAMES)
    assert np.allclose(fit.drift_nm[0], 0.0)


def _rows(locs):
    return (np.asarray(locs["frame"]),
            np.column_stack([locs["x_nm"], locs["y_nm"]]),
            np.asarray(locs["bead_true"]))


def test_a_stretch_no_bead_covers_is_bridged():
    drift = smooth_drift()
    locs = simulate_beads(N_FRAMES, drift, n_beads=3, seed=5,
                          gaps=[(b, 2000, 2600) for b in range(3)])
    fit = fit_drift(*_rows(locs), N_FRAMES)
    assert np.all(np.isfinite(fit.drift_nm))
    # straight across the gap: within the drift's own bend over 600 frames
    assert np.abs(fit.drift_nm[2000:2601, :2] - drift[2000:2601, :2]).max() < 6.0


def test_the_kept_drift_comes_back_with_the_fiducial_settings():
    locs = without_z(beads_with_trouble(smooth_drift()))
    plugin = FiducialDrift()
    settings = FiducialSettings(spline_knot_frames=150)
    result = plugin.run(Context(locs=locs), settings)
    restored = plugin.restore(plugin.keep(result))
    assert isinstance(restored.settings, FiducialSettings)
    assert restored.settings.spline_knot_frames == 150
    assert np.allclose(restored.data["drift"].drift, result.data["drift"].drift)


def test_more_than_one_file_is_refused():
    locs = without_z(beads_with_trouble(smooth_drift()))
    files = np.arange(len(locs)) % 2
    locs = Localizations({**locs.columns, "filenumber": files}, dict(locs.metadata))
    with pytest.raises(ValueError, match="2 files"):
        FiducialDrift().run(Context(locs=locs), FiducialSettings())


def test_referenced_to_the_last_frame_the_drift_is_zero_there():
    """Step 5a's *reference*: the same curve, moved to be zero at the end."""
    locs = without_z(beads_with_trouble(smooth_drift()))
    first, _, _ = estimate_drift_fiducials(locs, FiducialSettings())
    last, _, _ = estimate_drift_fiducials(locs, FiducialSettings(reference="last frame"))
    assert np.allclose(last.drift[-1], 0.0)
    assert np.allclose(last.drift, first.drift - first.drift[-1])


# ------------------------------------------------------- which beads (v3)

def _with_a_wandering_bead(seed=3, wander_nm=40.0):
    """Beads in a blinking structure, one of which (bead_true 3) creeps away
    by ``wander_nm`` in x over the acquisition."""
    from smappy.simulate import simulate, with_beads
    locs = simulate(n_frames=4000, seed=seed, drift=True)
    truth = np.asarray(locs.metadata["drift_truth"])
    locs = with_beads(locs, n_beads=5, seed=seed)
    bead = np.asarray(locs["bead_true"])
    frame = np.asarray(locs["frame"])
    x = np.asarray(locs["x_nm"], float).copy()
    x[bead == 3] += wander_nm * frame[bead == 3] / 4000
    locs = Localizations({**locs.columns, "x_nm": x}, dict(locs.metadata))
    return locs, truth - truth[0]


def test_a_bead_that_creeps_away_is_left_out_and_the_drift_is_the_others():
    from smappy.beads import RejectionSettings
    from smappy.fiducials import FiducialSettings, estimate_drift_fiducials
    locs, truth = _with_a_wandering_bead()
    drift, beads, fit = estimate_drift_fiducials(locs)
    chosen = fit.chosen
    out = [chosen.names[b] for b in range(len(chosen)) if not chosen.used[b]]
    assert len(out) == 1 and "wander" in chosen.reasons[chosen.names.index(out[0])]
    # the bead left out is the one that crept: its rows are bead_true 3
    true = np.asarray(locs["bead_true"])[beads.bead == chosen.names.index(out[0])]
    assert set(np.unique(true)) == {3}
    every, _, _ = estimate_drift_fiducials(
        locs, FiducialSettings(rejection=RejectionSettings(k=1e9)))
    error = lambda d: np.sqrt(np.mean((d.drift[:, 0] - truth[:, 0]) ** 2))
    assert error(drift) < 0.5 * error(every)


def test_a_bead_chosen_by_hand_stays_that_bead_however_the_beads_are_numbered():
    from smappy.fiducials import FiducialSettings, find_and_choose
    locs, _ = _with_a_wandering_bead(wander_nm=0.0)
    found = find_and_choose(locs)
    name = found.chosen.names[2]
    x, y = found.where_nm[2]
    # named a little off where it is (as after a drift correction), and
    # found among fewer beads, numbered otherwise
    nearby = f"x{(x + 120) / 1000:.1f}y{y / 1000:.1f}"
    first = np.asarray(locs["bead_true"]) != np.asarray(locs["bead_true"])[found.beads.bead == 0][0]
    again = find_and_choose(locs, FiducialSettings(beads=f"{nearby} out"), select=first)
    b = int(np.argmin(np.hypot(*(again.where_nm - [x, y]).T)))
    assert again.chosen.names[b] == name and not again.chosen.used[b]
    assert again.chosen.state(b) == "out, by hand"
    assert len(again.beads) == len(found.beads) - 1


def test_the_fiducials_plugin_offers_the_bead_selector_and_takes_its_choice():
    from smappy.plugins import Context
    from smappy.plugins.drift_fiducials import FiducialDrift
    from smappy.fiducials import FiducialSettings
    locs, _ = _with_a_wandering_bead()
    plugin = FiducialDrift()
    ctx = Context(locs=locs)
    notice = plugin.describe(ctx, FiducialSettings()).notice
    assert notice.window == "bead selector" and notice.action == "Select beads..."
    view = plugin.bead_view(ctx, FiducialSettings())
    assert view.time_unit == "frame" and len(view.columns["frames"]) == len(view.beads)
    out = int(np.flatnonzero(~view.beads.used)[0])
    settings = plugin.with_overrides(FiducialSettings(), view, {view.beads.names[out]: True})
    assert settings.beads == f"{view.beads.names[out]} in"
    result = plugin.run(ctx, settings)
    assert "left out" not in result.text
    assert "1 left out" in plugin.run(ctx, FiducialSettings()).text


def test_beads_can_come_from_the_roi_managers_rois_one_bead_each():
    from smappy.io.formats import FileInfo
    from smappy.fiducials import FiducialSettings
    from smappy.session import Session
    locs = simulate(n_frames=3000, seed=5, drift=True)
    truth = np.asarray(locs.metadata["drift_truth"])
    truth = truth - truth[0]
    locs = with_beads(locs, n_beads=5, seed=5)
    session = Session()
    session.add_file(locs, FileInfo(name="b", path="b", format="test", n=len(locs)))
    project = session.rois
    file_id = next(iter(project.sources))
    bead = np.asarray(locs["bead_true"])
    for b in range(4):                                  # four of the five
        x = float(np.median(locs["x_nm"][bead == b]))
        y = float(np.median(locs["y_nm"][bead == b]))
        project.add_roi(file_id, [x, y], polygon=[[x - 300, y - 300], [x + 300, y - 300],
                                                  [x + 300, y + 300], [x - 300, y + 300]])
    project.add_roi(file_id, [-5000, -5000],            # round nothing
                    polygon=[[-5300, -5300], [-4700, -5300], [-4700, -4700], [-5300, -4700]])
    settings = FiducialSettings(source="ROI manager")
    plugin = FiducialDrift()
    ctx = session.context(grouping="ungrouped")
    state = plugin.describe(ctx, settings)
    assert "5 used ROIs" in state.notice.text and state.active["memory"] is False
    view = plugin.bead_view(ctx, settings)
    assert view.beads.names == ["ROI 1", "ROI 2", "ROI 3", "ROI 4"]
    warnings = []
    ctx = session.context(grouping="ungrouped")
    ctx.warn = warnings.append
    result = plugin.run(ctx, settings)
    assert warnings and "ROI 5" in warnings[0]
    drift = result.data["drift"]
    error = np.sqrt(np.mean((drift.drift[:, :2] - truth[:, :2]) ** 2, axis=0))
    assert np.all(error < 4.0)
    # a choice by hand is the ROI's
    again = plugin.with_overrides(settings, view, {"ROI 2": False})
    assert again.beads == "ROI 2 out"
    assert plugin.bead_view(ctx, again).beads.state(1) == "out, by hand"


def test_each_bead_is_named_in_the_table_and_kept_as_a_row():
    """`bead_id` on each bead's localizations, and what the bead selector
    shows of each as a region table of them, current as a block's."""
    from smappy.region_results import Evaluation, is_current, rows
    from smappy.session import Session
    drift = smooth_drift()
    locs = without_z(beads_with_trouble(drift))
    session = Session(locs)
    session.show_grouped(0, False)
    session.run(FiducialDrift(), FiducialSettings())
    ids = np.asarray(session.locs["bead_id"])
    truth = np.asarray(session.locs["bead_true"])
    for b in np.unique(truth):
        # one id per simulated bead; its localizations too dim to be linked
        # into its track have none (0)
        named = ids[(truth == b) & (ids > 0)]
        assert len(np.unique(named)) == 1 and len(named) > 0.5 * (truth == b).sum()
    ev = Evaluation("bead_id", "Fiducials drift correction")
    ctx = session.context()
    assert is_current(ctx, ev)
    found = rows(ctx, ev)
    assert len(found) == len(np.unique(truth))
    assert all(r["state"] in ("used", "not used") or r["state"].startswith("out")
               for r in found)
    assert {"noise_nm", "wander_nm", "jump_nm", "used", "name", "offset_x_nm"} <= set(found[0])
