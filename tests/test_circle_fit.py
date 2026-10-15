"""Circle fit and Centre ROIs: a ring's centre and radius, per ROI, and ROIs
moved onto their structures."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.plugins import Context
from smappy.plugins.circle_fit import CircleFit, CircleFitSettings, circle_fit


def ring(center, radius, n, spread=8.0, rng=None, background=0, half=150):
    rng = rng or np.random.default_rng(0)
    phi = rng.uniform(0, 2 * np.pi, n)
    r = radius + rng.normal(0, spread, n)
    x = center[0] + r * np.cos(phi)
    y = center[1] + r * np.sin(phi)
    if background:
        x = np.concatenate([x, rng.uniform(center[0] - half, center[0] + half, background)])
        y = np.concatenate([y, rng.uniform(center[1] - half, center[1] + half, background)])
    return x, y


def table(xs, ys, precision=8.0):
    x, y = np.concatenate(xs), np.concatenate(ys)
    n = len(x)
    return Localizations({"x_nm": x, "y_nm": y, "frame": np.arange(n),
                          "photons": np.full(n, 1000.0),
                          "xy_err_nm": np.full(n, precision)})


def test_a_simulated_ring_returns_its_radius_and_centre():
    x, y = ring((1234.0, 567.0), 50.0, 300)
    fit = circle_fit(x, y, np.full(len(x), 8.0))
    assert fit["n_locs"] == 300
    assert abs(fit["x_nm"] - 1234) < 3 * fit["x_err_nm"] + 0.5
    assert abs(fit["y_nm"] - 567) < 3 * fit["y_err_nm"] + 0.5
    assert abs(fit["radius_nm"] - 50) < 3 * fit["radius_err_nm"] + 0.5
    assert fit["rms_nm"] == pytest.approx(8.0, rel=0.15)
    # the spread is the precision: a thin ring, reduced chi-square near 1
    assert fit["chi2_red"] == pytest.approx(1.0, abs=0.25)


def test_the_radius_error_is_the_scatter_over_repeats():
    rng = np.random.default_rng(1)
    radii, errors = [], []
    for _ in range(200):
        x, y = ring((0.0, 0.0), 60.0, 80, spread=10.0, rng=rng)
        fit = circle_fit(x, y)
        radii.append(fit["radius_nm"])
        errors.append(fit["radius_err_nm"])
    assert np.mean(radii) == pytest.approx(60.0, abs=0.5)
    assert np.std(radii) == pytest.approx(np.mean(errors), rel=0.2)


def test_background_pulls_the_robust_circle_less_than_least_squares():
    x, y = ring((0.0, 0.0), 55.0, 200, spread=6.0, background=60, half=120,
                rng=np.random.default_rng(2))
    fit = circle_fit(x, y)
    plain = circle_fit(x, y, scale=1e6)             # ordinary least squares
    assert np.hypot(fit["x_nm"], fit["y_nm"]) < 3
    assert abs(fit["radius_nm"] - 55) < 0.6 * abs(plain["radius_nm"] - 55)
    assert fit["radius_nm"] == pytest.approx(55.0, abs=5)


def test_a_fixed_radius_fits_the_centre_of_an_arc():
    rng = np.random.default_rng(3)
    phi = rng.uniform(0, np.pi / 2, 40)                # a quarter of the ring
    x, y = 100 + 50 * np.cos(phi), -20 + 50 * np.sin(phi)
    fit = circle_fit(x, y, radius=50.0)
    assert fit["radius_nm"] == 50.0 and fit["radius_err_nm"] == 0.0
    assert np.hypot(fit["x_nm"] - 100, fit["y_nm"] + 20) < 2


def test_too_few_localizations_give_nan_not_an_error():
    fit = circle_fit([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    assert fit["n_locs"] == 3 and np.isnan(fit["radius_nm"])


def rings_project(centres, radii, offsets, size=300):
    """A session with rings at ``centres``, an ROI each at centre + offset."""
    from smappy.session import Session
    rng = np.random.default_rng(4)
    xs, ys = zip(*(ring(c, r, 150, spread=6.0, rng=rng) for c, r in zip(centres, radii)))
    session = Session(table(xs, ys, precision=6.0))
    session.show_grouped(0, False)
    project = session.rois
    project.set_geometry(size)
    file_id = next(iter(project.sources))
    ids = [project.add_roi(file_id, list(np.add(c, o))).id
           for c, o in zip(centres, offsets)]
    return session, project, ids


def test_circle_fit_measures_each_roi_with_its_own_ring():
    centres, radii = [(0, 0), (2000, 0), (0, 2000)], [40.0, 55.0, 70.0]
    session, project, ids = rings_project(centres, radii, [(0, 0)] * 3)
    for roi_id, c, r in zip(ids, centres, radii):
        result = CircleFit().run(project.region_context(project.rois[roi_id]),
                                 CircleFitSettings())
        v = result.values
        assert v["n_locs"] == 150
        assert v["radius_nm"] == pytest.approx(r, abs=3)
        assert np.hypot(v["x_nm"] - c[0], v["y_nm"] - c[1]) < 3


def test_centre_rois_moves_an_off_centre_roi_onto_its_ring_in_one_undo_step():
    from smappy.plugins.roi_centre import CentreROIs, CentreROIsSettings
    centres = [(0, 0), (2000, 0)]
    session, project, ids = rings_project(centres, [50.0, 50.0], [(60, -40), (-30, 50)])
    ctx = Context(locs=session.locs, session=session)
    result = CentreROIs().run(ctx, CentreROIsSettings(centre="circle", iterations=2))
    for roi_id, c in zip(ids, centres):
        assert np.hypot(*(np.asarray(project.rois[roi_id].center) - c)) < 3
    assert result.values["moved"] == 2
    project.undo()
    assert project.rois[ids[0]].center == [60.0, -40.0]


def test_centre_rois_on_the_median_and_leaves_an_empty_roi_where_it_is():
    from smappy.plugins.roi_centre import CentreROIs, CentreROIsSettings
    session, project, ids = rings_project([(0, 0)], [30.0], [(40, 0)])
    empty = project.add_roi(next(iter(project.sources)), [5000.0, 5000.0])
    excluded = project.add_roi(next(iter(project.sources)), [30.0, 0.0])
    excluded.use = False
    ctx = Context(locs=session.locs, session=session)
    result = CentreROIs().run(ctx, CentreROIsSettings(centre="median", iterations=3))
    assert np.hypot(*project.rois[ids[0]].center) < 5
    assert project.rois[empty.id].center == [5000.0, 5000.0]
    assert project.rois[excluded.id].center == [30.0, 0.0]        # not used: not moved
    assert "1 left" in result.text
