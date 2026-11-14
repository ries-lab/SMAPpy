"""*Fiducials* with structures: DNA origami and their binding sites as
drift markers (Picasso's *Undrift from picked*), from ROIs and from an id
column, against a drift that was simulated."""
from functools import lru_cache

import numpy as np
import pytest

from smappy.beads import fit_offsets_and_spline
from smappy.fiducials import (BEAD_KNOT, STRUCTURE_KNOT, FiducialSettings,
                              estimate_drift_fiducials, marker_weights)
from smappy.locs import Localizations
from smappy.plugins.drift_fiducials import FiducialDrift

N_FRAMES = 20000


def smooth_drift(n_frames=N_FRAMES):
    """Slow creep and a wobble, x, y and z, zero at frame 0."""
    t = np.linspace(0.0, 1.0, n_frames)
    drift = np.column_stack([120 * t + 25 * np.sin(5 * t),
                             -80 * t ** 2 + 15 * np.sin(3 * t),
                             40 * t - 20 * np.sin(4 * t)])
    return drift - drift[0]


@lru_cache(maxsize=None)
def _origami(field_um: float, seed: int, with_z: bool):
    from smappy.plugins import settings_from
    from smappy.plugins.simulate import SimulateSettings, preset_values
    from smappy.simulate import ground_truth, localizations
    values = preset_values("DNA-PAINT origami (20 nm grid)")
    values.update(field_um=field_um, seed=seed)
    locs = localizations(ground_truth(settings_from(SimulateSettings, values)))
    drift = smooth_drift()
    frame = np.asarray(locs["frame"])
    columns = dict(locs.columns)
    for axis, name in enumerate(("x_nm", "y_nm", "z_nm")):
        columns[name] = np.asarray(locs[name], float) + drift[frame, axis]
    # the simulation's origami, numbered from 1 as an id column is
    columns["origami"] = np.asarray(locs["copy"]).astype(np.int64) + 1
    if not with_z:
        columns = {k: v for k, v in columns.items() if not k.startswith("z_")}
        drift = drift[:, :2]
    return Localizations(columns, dict(locs.metadata)), drift


def origami(field_um=10.0, seed=0, with_z=False):
    """The *Simulate* preset "DNA-PAINT origami (20 nm grid)" -- twelve sites
    on a 20 nm grid, about 85 origami in 10 um, each site binding some 13
    times in 20,000 frames -- with `smooth_drift` added."""
    return _origami(field_um, seed, with_z)


def error_nm(drift, truth):
    """The curve against the truth up to a constant, rms per axis, and the
    constant: only how the markers moved is measured, and where the curve is
    zero (its first frame) rests on the few localizations there."""
    e = drift[:, :truth.shape[1]] - truth
    constant = e.mean(axis=0)
    return np.sqrt(np.mean((e - constant) ** 2, axis=0)), constant


STRUCTURES = dict(markers="structures", spline_knot_frames=STRUCTURE_KNOT)


def test_whole_origami_from_an_id_column_give_the_drift_to_about_a_nanometre():
    locs, truth = origami(with_z=True)
    settings = FiducialSettings(source="id column", id_column="origami", **STRUCTURES)
    drift, beads, fit = estimate_drift_fiducials(locs, settings)
    assert 70 <= len(beads) <= 100
    rms, constant = error_nm(drift.drift, truth)
    # some 600 localizations an origami, its sites up to 30 nm from its
    # centre: a nanometre and a half in x and y; the origami are flat, so z
    # has only the precision, weighted by z_err_nm (0.2 nm)
    assert np.all(rms[:2] < 2.5) and rms[2] < 1.0
    assert np.all(np.abs(constant[:2]) < 6.0)
    # the beads' way with the same markers: one point a frame, above 5000
    # photons -- the bright tail of the binding events, a tenth of them
    beads_way, _, used = estimate_drift_fiducials(
        locs, FiducialSettings(source="id column", id_column="origami",
                               spline_knot_frames=STRUCTURE_KNOT))
    assert len(used.rows) < 0.2 * len(fit.rows)
    assert error_nm(beads_way.drift, truth)[0][:2].mean() > 2 * rms[:2].mean()


def test_finding_beads_in_dna_paint_finds_none():
    locs, _ = origami()
    with pytest.raises(ValueError, match="no beads"):
        estimate_drift_fiducials(locs)


def test_the_second_round_on_binding_sites_from_local_maxima_reaches_a_fraction_of_a_nanometre():
    """Picasso's two rounds: whole origami, then -- on the table they
    corrected -- each binding site of Find clusters' local maxima."""
    from smappy.plugins.clusters import local_maxima
    locs, truth = origami()
    first, _, _ = estimate_drift_fiducials(
        locs, FiducialSettings(source="id column", id_column="origami", **STRUCTURES))
    corrected = first.apply(locs)
    sites = local_maxima(np.column_stack([corrected["x_nm"], corrected["y_nm"]]), 10.0, 15)
    corrected = Localizations({**corrected.columns, "cluster_id": sites},
                              dict(corrected.metadata))
    second, beads, fit = estimate_drift_fiducials(
        corrected, FiducialSettings(source="id column", **STRUCTURES))
    assert len(beads) > 400 and fit.chosen.names[0].startswith("cluster_id ")
    rms, constant = error_nm(first.drift + second.drift, truth)
    assert np.all(rms[:2] < 0.4) and np.all(np.abs(constant[:2]) < 2.0)
    assert error_nm(first.drift, truth)[0].mean() > 3 * rms.mean()
    # a site is markers' noise about the curve: the precision, a few nm
    assert np.all(fit.rms_nm < 8.0)


def test_origami_picked_as_rois_are_markers_and_the_bead_selector_names_them():
    from smappy.io.formats import FileInfo
    from smappy.session import Session
    locs, truth = origami(field_um=7.0, seed=3)
    session = Session()
    session.add_file(locs, FileInfo(name="o", path="o", format="test", n=len(locs)))
    project = session.rois
    file_id = next(iter(project.sources))
    copy = np.asarray(locs["origami"])
    x, y = np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"])
    for c in np.unique(copy):
        # a square round each origami and its drift, as a pick would be
        cx, cy = float(np.median(x[copy == c])), float(np.median(y[copy == c]))
        project.add_roi(file_id, [cx, cy], polygon=[[cx - 150, cy - 150], [cx + 150, cy - 150],
                                                    [cx + 150, cy + 150], [cx - 150, cy + 150]])
    plugin = FiducialDrift()
    settings = FiducialSettings(source="ROI manager", **STRUCTURES)
    ctx = session.context(grouping="ungrouped")
    state = plugin.describe(ctx, settings)
    n = len(np.unique(copy))
    assert f"a structure in each of the {n} used ROIs" in state.notice.text
    assert state.active["min_photons"] is False and state.active["min_fraction"] is False
    result = plugin.run(ctx, settings)
    rms, _ = error_nm(result.data["drift"].drift, truth)
    assert np.all(rms[:2] < 3.0)
    assert f"{n} structure(s)" in result.text
    # every localization of an origami is named after its marker
    assert (np.asarray(result.locs["bead_id"]) > 0).mean() > 0.9
    view = plugin.bead_view(session.context(grouping="ungrouped"), settings)
    assert view.beads.names[0] == "ROI 1"
    again = plugin.with_overrides(settings, view, {"ROI 2": False})
    assert again.beads == "ROI 2 out"
    view = plugin.bead_view(session.context(grouping="ungrouped"), again)
    assert view.beads.state(1) == "out, by hand"


def test_structures_are_named_never_found_and_the_form_says_what_applies():
    plugin = FiducialDrift()
    locs, _ = origami(field_um=4.0)
    with pytest.raises(ValueError, match="structures are not found"):
        estimate_drift_fiducials(locs, FiducialSettings(markers="structures"))
    # choosing structures sets a knot spacing for sparse markers and a source
    # they can come from; back to beads, the beads' spacing
    reply = plugin.react("markers", FiducialSettings(markers="structures"))
    assert reply == {"spline_knot_frames": STRUCTURE_KNOT, "source": "ROI manager"}
    reply = plugin.react("markers", FiducialSettings(spline_knot_frames=STRUCTURE_KNOT))
    assert reply == {"spline_knot_frames": BEAD_KNOT}
    assert plugin.react("markers", FiducialSettings(markers="structures",
                                                    spline_knot_frames=300,
                                                    source="id column")) is None
    state = plugin.describe(None, FiducialSettings(markers="structures"))
    assert state.notice.kind == "warning" and state.shown["id_column"] is False
    from smappy.plugins import Context
    state = plugin.describe(Context(locs=locs), FiducialSettings(source="id column",
                                                                 id_column="origami"))
    assert state.shown["id_column"] and "origami" in dict(state.choices["id_column"])
    assert state.active["min_photons"] and not state.active["memory"]
    assert "a bead per id of origami" in state.notice.text
    with pytest.raises(ValueError, match="no 'cluster_id'.*origami"):
        estimate_drift_fiducials(locs, FiducialSettings(source="id column", **STRUCTURES))


def test_the_id_column_takes_only_what_the_selection_keeps():
    locs, _ = origami(field_um=6.0)
    x = np.asarray(locs["x_nm"])
    left = x < np.median(x)
    settings = FiducialSettings(source="id column", id_column="origami", **STRUCTURES)
    _, beads, fit = estimate_drift_fiducials(locs, settings, select=left)
    assert np.all(left[fit.rows])
    assert (beads.bead[~left] < 0).all()


def _markers(extent_nm, seed=1, n_frames=2000, n=4000, n_markers=40):
    """Points of still markers, each ``extent_nm`` wide, at a precision of
    3-12 nm each."""
    rng = np.random.default_rng(seed)
    frame = np.sort(rng.integers(0, n_frames, n))
    marker = rng.integers(0, n_markers, n)
    sigma = rng.uniform(3.0, 12.0, n)
    place = rng.normal(0, 1000, (n_markers, 2))
    positions = (place[marker] + rng.normal(0, 1, (n, 2)) * extent_nm
                 + rng.normal(0, 1, (n, 2)) * sigma[:, None])
    return frame, positions, np.column_stack([sigma, sigma]), marker, sigma


def test_weights_are_the_precision_for_a_site_and_nearly_equal_for_a_whole_origami():
    frame, positions, errors, marker, sigma = _markers(0.0)
    w = marker_weights(frame, positions, errors, marker, 2000, 200)
    assert w.shape == (len(frame), 2) and np.array_equal(w[:, 0], w[:, 1])
    # 1 / sigma^2, up to the little extent a fit's residuals leave
    scaled = w[:, 0] * sigma ** 2
    assert np.allclose(scaled, np.median(scaled), rtol=0.1)
    assert w[:, 0].max() / w[:, 0].min() > 10
    frame, positions, errors, marker, sigma = _markers(30.0)
    w = marker_weights(frame, positions, errors, marker, 2000, 200)
    assert w[:, 0].max() / w[:, 0].min() < 1.3
    # a broken precision cannot carry the curve: a third of the median at least
    frame, positions, errors, marker, sigma = _markers(0.0)
    errors[0] = 1e-3
    w = marker_weights(frame, positions, errors, marker, 2000, 200)
    assert w[0, 0] == pytest.approx(w[:, 0].max())
    assert w[0, 0] == pytest.approx(1.0 / (np.median(errors[:, 0]) / 3) ** 2, rel=0.05)


def test_a_weight_per_axis_is_the_fit_of_each_axis_with_its_own():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 100, 600))
    index = rng.integers(0, 5, 600)
    p = rng.normal(0, 3, (600, 3)) + np.column_stack([t * 0.3, -t * 0.2, np.sin(t / 10)])
    w = np.column_stack([np.full(600, 1.0), np.full(600, 1.0), rng.uniform(0.1, 1, 600)])
    together, curve = fit_offsets_and_spline(t, p, index, 5, 0.0, 100.0, 13, weights=w)
    xy, xy_curve = fit_offsets_and_spline(t, p[:, :2], index, 5, 0.0, 100.0, 13)
    z, z_curve = fit_offsets_and_spline(t, p[:, 2], index, 5, 0.0, 100.0, 13,
                                        weights=w[:, 2])
    assert np.allclose(together[:, :2], xy) and np.allclose(together[:, 2:], z)
    assert np.allclose(curve(t)[:, :2], xy_curve(t)) and np.allclose(curve(t)[:, 2:],
                                                                       z_curve(t))


def test_the_panel_sets_up_structures_and_the_bead_selector_chooses_among_them():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from smappy.gui.bead_window import BeadWindow
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.session import Session
    app = QApplication.instance() or QApplication([])
    locs, _ = origami(field_um=4.0)
    session = Session(locs)
    opened = []

    def opener(settings=None, plugin=None, reply=None):
        opened.append(BeadWindow(session, plugin, settings, reply))
    opener.replies = True
    session.window_openers["bead selector"] = opener
    panel = PluginPanel(FiducialDrift, session)
    panel.form.set_values({"markers": "structures"})
    panel._react("markers")
    settings = panel.form.value()
    assert settings.spline_knot_frames == STRUCTURE_KNOT and settings.source == "ROI manager"
    panel.form.set_values({"source": "id column", "id_column": "origami"})
    panel._describe()
    assert "a structure per id of origami" in panel.notice.text()
    panel.notice_button.click()
    window = opened[0]
    assert window.table.rowCount() == len(np.unique(np.asarray(locs["origami"])))
    window.table.selectRow(1)
    window.exclude.click()
    assert panel.form.value().beads == "origami 2 out"
    window.close()
    app.processEvents()
