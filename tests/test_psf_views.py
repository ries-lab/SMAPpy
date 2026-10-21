"""The PSF calibration's views of the model against the data, and its progress bar: Qt-free parts."""
import numpy as np
from matplotlib.figure import Figure

from smappy.calibrate.core import build_calibration
from smappy.calibrate.model_views import average_views, draw_model_vs_data, planes
from smappy.calibrate.psf_methods import LearningProgress, duration


def test_the_bead_averages_beads_aligned_onto_the_model_average_to_its_own_average():
    from test_bead_calibration import synthetic_collection
    result = build_calibration(synthetic_collection())
    views = average_views(result)
    assert views.data[0].shape == views.model[0].shape == result.raw_psf.shape
    assert len(views.z_nm) == result.raw_psf.shape[0]
    used = [b for b, state in views.states.items() if state == "used"]
    assert len(used) == int(result.accepted.sum())
    mean = np.mean([views.beads[b][0][0] for b in used], axis=0)
    peak = result.raw_psf.max()
    assert np.abs(mean - result.raw_psf).max() < 0.02 * peak             # the same alignment and scale
    assert np.abs(views.data[0] - views.model[0]).max() < 0.2 * peak       # the model is the smoothed data
    data, model = views.of(used[0])
    assert model is views.model                                           # one model for every bead


def test_the_planes_shown_are_100_nm_apart_or_150_when_100_would_be_too_many():
    z = np.arange(600., -600.1, -50.)                                     # 25 planes, 50 nm
    picked = planes(z)
    assert len(picked) <= 9 and np.all(np.diff(z[picked]) == -150.)        # 1200 nm: 13 at 100 nm
    assert 0. in z[picked]                                                 # the focus is one of them
    assert np.all(np.diff(np.arange(300., -300.1, -50.)[planes(np.arange(300., -300.1, -50.))]) == -100.)


def test_model_vs_data_is_a_column_per_plane_and_rows_of_data_model_and_residual():
    from test_bead_calibration import synthetic_collection
    views = average_views(build_calibration(synthetic_collection()))
    figure = Figure(constrained_layout=True)
    draw_model_vs_data(figure, views, None, "XY")
    picked = planes(views.z_nm)
    assert len(figure.axes) == 3 * len(picked)
    grid = np.array(figure.axes).reshape(3, len(picked))
    k = picked[len(picked) // 2]
    data, model, residual = (ax.images[0] for ax in grid[:, len(picked) // 2])
    assert np.allclose(data.get_array(), views.data[0][k]) and np.allclose(model.get_array(), views.model[0][k])
    # data and model on one contrast, the data's peak; the residual symmetric about 0
    assert data.get_clim() == model.get_clim() == (0, np.nanmax(views.data[0][k]))
    low, high = residual.get_clim()
    assert low == -high and high == np.nanmax(np.abs(views.data[0][k] - views.model[0][k]))
    # under each residual its largest |value| in % of the plane's data peak
    percent = 100 * high / np.nanmax(views.data[0][k])
    assert grid[2, len(picked) // 2].get_xlabel() == f"±{percent:.2g} %"
    assert [ax.get_ylabel() for ax in grid[:, 0]] == ["data", "model", "residual"]
    figure = Figure()
    text = draw_model_vs_data(figure, views, 1, "XZ")
    # a side section is one picture: data | model | residual side by side, 1 px apart
    assert len(figure.axes) == 1 and "bead 1" in text
    assert figure.axes[0].images[0].get_array().shape[1] == 3 * views.data[0].shape[2] + 2
    assert "average" in draw_model_vs_data(Figure(), views, 1, "XY", average=True)


def test_model_vs_data_puts_the_second_channel_beside_the_first():
    from dataclasses import replace

    from test_bead_calibration import synthetic_collection
    views = average_views(build_calibration(synthetic_collection()))
    dual = replace(views, data=views.data * 2, model=views.model * 2, titles=["main", "secondary"])
    figure = Figure(constrained_layout=True)
    draw_model_vs_data(figure, dual, None, "XY")
    picked = planes(views.z_nm)
    panels = figure.subfigs
    assert [p.get_suptitle() for p in panels] == ["main", "secondary"]
    assert all(len(p.axes) == 3 * len(picked) for p in panels)            # three rows each, not six
    figure = Figure()
    draw_model_vs_data(figure, dual, None, "XZ")
    width = views.data[0].shape[2]
    assert figure.axes[0].images[0].get_array().shape[1] == 2 * (3 * width + 2) + 4


def test_the_bar_follows_a_fit_whose_gain_falls_geometrically_and_never_goes_back():
    now = [0.0]
    tracker = LearningProgress(clock=lambda: now[0])
    seen = []
    for stage in (1, 2):
        objective = 1e6
        for it in range(80):
            now[0] += 1.0
            gain = 1e4 * 0.7 ** it
            objective -= gain
            fraction, left, text = tracker.update({"stage": stage, "iteration": it, "objective": objective})
            seen.append(fraction)
            if gain < LearningProgress.TOL:
                break
    assert all(b >= a for a, b in zip(seen, seen[1:]))                      # never back
    assert abs(seen[len(seen) // 2 - 1] - 0.45) < 0.03 and abs(seen[-1] - 0.9) < 0.03
    assert left is not None and 0 < left < 30                               # 90 % in about 80 s
    assert tracker.update({"message": "fitting the beads back", "fraction": 0.95})[0] == 0.95
    assert tracker.update({"restart": True})[0] is None


def test_a_duration_reads_at_a_glance():
    assert [duration(s) for s in (12, 300, 7200)] == ["12 s", "5 min", "2.0 h"]
