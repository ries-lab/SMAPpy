"""COMET with a transformation over time besides the shift (wave 5, step 5d):
a sample that expands, or optics whose magnification drifts."""
import numpy as np
import pytest

from smappy import _drift, plugins
from smappy.drift import (TRANSFORMATIONS, Drift, DriftSettings, _TransformationFit,
                          estimate_drift, load_drift, save_drift_corrected)
from smappy.locs import Localizations
from smappy.plugins import Context

FIELD = 20000.0
TOLERANCE = 150e-6    # on a parameter, ppm: 1.5 nm at 10 um from the centre
N_FRAMES = 200


def simulate(scale=5e-3, rotation=0.0, seed=1, per_frame=150, n_points=3000,
             noise=8.0):
    """A random structure over a 20 um field, drifting and changing size: the
    sample's scale rises and falls by ``scale`` (a sine over the movie),
    and it turns by ``rotation`` radians, linearly, about the field's
    centre.  Returns the table, the true positions and the sample's scale
    and angle per frame."""
    rng = np.random.default_rng(seed)
    structure = rng.random((n_points, 3)) * [FIELD, FIELD, 600] - [0, 0, 300]
    t = np.arange(N_FRAMES)
    base = (3 * np.sin(t / 60 + .37) + t / 50) * 10.0
    drift = np.column_stack([base, 0.7 * base, -0.5 * base])
    s = scale * np.sin(np.pi * t / N_FRAMES)
    r = rotation * t / N_FRAMES
    matrix = np.repeat(np.eye(3)[None], N_FRAMES, 0)
    matrix[:, 0, 0] += s
    matrix[:, 1, 1] += s
    matrix[:, 1, 0] += r
    matrix[:, 0, 1] -= r
    frame = np.repeat(t, per_frame)
    rng.shuffle(frame)
    true = structure[rng.integers(0, n_points, frame.size)]
    centre = np.array([FIELD / 2, FIELD / 2, 0.0])
    seen = (centre + np.einsum("fij,fj->fi", matrix[frame], true - centre)
            + drift[frame] + rng.normal(0, noise, true.shape))
    locs = Localizations({"frame": frame.astype(np.int64),
                          **{name: seen[:, i].astype(np.float32)
                             for i, name in enumerate(("x_nm", "y_nm", "z_nm"))}},
                         {"units": "nm"})
    return locs, true, s, r


def settings(transformation, **values):
    values = {"group": False, "spline_knot_frames": 40, "max_drift_nm": 150,
              "target_sigma_nm": 10, **values}
    return DriftSettings(transformation=transformation, **values)


def residual_nm(locs, true, edge=False):
    """How far the corrected positions are from the truth (rms, x and y)
    once one global affine map -- which no overlap can measure -- is taken
    out; with ``edge``, only those more than 8 um from the centre, where a
    change of scale moves them furthest."""
    xyz = np.column_stack([np.asarray(locs[k], float)
                           for k in ("x_nm", "y_nm", "z_nm")])
    design = np.column_stack([true, np.ones(len(true))])
    coefficients, *_ = np.linalg.lstsq(design, xyz, rcond=None)
    error = (xyz - design @ coefficients)[:, :2]
    if edge:
        error = error[np.hypot(*(true[:, :2] - FIELD / 2).T) > 8000]
    return np.sqrt((error ** 2).mean())


def test_a_changing_magnification_is_recovered_and_corrected():
    locs, true, scale, _ = simulate()
    drift = estimate_drift(locs, settings("xy scale", reference="first frame"))

    found = drift.parameters()["scale"]
    # the sample's scale relative to the first frame, to within 150 ppm of
    # the 5000 it changes by: 1.5 nm at the edge of the field, against
    # localizations precise to 8
    assert np.abs(found - scale).max() < TOLERANCE
    assert drift.matrix.shape == (N_FRAMES, 3, 3)
    # corrected down to the localization noise (8 nm) ...
    assert residual_nm(drift.apply(locs), true) < 8.5
    assert residual_nm(drift.apply(locs), true, edge=True) < 8.5
    # ... where the shift alone leaves the field's edges smeared by the
    # tens of nm the magnification moved them
    shift_only = estimate_drift(locs, settings("shift"))
    assert residual_nm(shift_only.apply(locs), true, edge=True) > 11.5


def test_a_rotation_is_recovered_with_the_scale():
    locs, true, scale, rotation = simulate(rotation=1e-3)
    drift = estimate_drift(locs, settings("xy scale + rotation",
                                          reference="first frame"))
    parameters = drift.parameters()
    assert np.abs(parameters["rotation"] - rotation).max() < TOLERANCE
    assert np.abs(parameters["scale"] - scale).max() < TOLERANCE
    assert residual_nm(drift.apply(locs), true) < 8.5


def test_the_affine_models_find_the_same_scale():
    locs, true, scale, _ = simulate()
    drift = estimate_drift(locs, settings("xy affine", reference="first frame"))
    parameters = drift.parameters()
    for name in ("a_xx", "a_yy"):
        assert np.abs(parameters[name] - scale).max() < TOLERANCE
    for name in ("a_xy", "a_yx"):
        assert np.abs(parameters[name]).max() < TOLERANCE
    assert residual_nm(drift.apply(locs), true) < 8.5


def test_no_transformation_is_found_where_there_is_none():
    locs, _, _, _ = simulate(scale=0.0)
    drift = estimate_drift(locs, settings("xy scale + rotation"))
    for values in drift.parameters().values():
        assert np.abs(values).max() < TOLERANCE / 2


def test_the_reference_frame_is_left_where_it_was():
    locs, _, _, _ = simulate()
    for reference, frame in (("first frame", 0), ("last frame", -1)):
        drift = estimate_drift(locs, settings("xy scale", reference=reference))
        np.testing.assert_allclose(drift.matrix[frame], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(drift.drift[frame], 0.0, atol=1e-9)
        assert drift.reference == reference


def test_the_kernels_gradient_is_the_costs_derivative():
    rng = np.random.default_rng(0)
    n, n_frames = 2000, 40
    coords = np.column_stack([rng.uniform(0, 5000, n), rng.uniform(0, 5000, n),
                              rng.uniform(-300, 300, n)]).astype(np.float32)
    frames = rng.integers(0, n_frames, n).astype(np.int32)
    from smappy._comet.core.pair_indices import pair_indices_kdtree
    for name in TRANSFORMATIONS:
        if name == "shift":
            continue
        fit = _TransformationFit(settings(name, spline_knot_frames=10,
                                          spline_penalty=0.1,
                                          transformation_penalty=0.5),
                                 coords, n_frames, False, None)
        idx_i, idx_j, _ = pair_indices_kdtree(fit.coords, 300)
        flat = rng.normal(0, 3, fit.n_variables)
        mu = rng.normal(0, 20, (n_frames, 3))

        def cost(f, m):
            return fit.cost(f, np.ascontiguousarray(m), frames, idx_i, idx_j,
                            100.0, 1.0, 0.1)

        _, d_mu, d_flat = cost(flat, mu)
        eps = 1e-4
        for k in range(4):
            step = np.zeros_like(flat)
            step[k] = eps
            numeric = (cost(flat + step, mu)[0] - cost(flat - step, mu)[0]) / (2 * eps)
            assert numeric == pytest.approx(d_flat[k], rel=1e-5, abs=1e-12)
        step = np.zeros_like(mu)
        step[5, 1] = eps
        numeric = (cost(flat, mu + step)[0] - cost(flat, mu - step)[0]) / (2 * eps)
        assert numeric == pytest.approx(d_mu[5, 1], rel=1e-5, abs=1e-12)


def test_with_identity_matrices_the_kernel_is_the_plain_one():
    rng = np.random.default_rng(2)
    n, n_frames = 1500, 30
    coords = rng.uniform(0, 3000, (n, 3)).astype(np.float32)
    frames = rng.integers(0, n_frames, n).astype(np.int32)
    idx_i = rng.integers(0, n, 20000).astype(np.int32)
    idx_j = rng.integers(0, n, 20000).astype(np.int32)
    mu = rng.normal(0, 30, (n_frames, 3))
    params = np.column_stack([np.tile(np.eye(3).ravel(), (n_frames, 1)), mu,
                              np.ones(n_frames), np.ones(n_frames)])
    plain, gradient = _drift.cost_and_gradient(coords, frames, idx_i, idx_j, mu,
                                               80.0, 1.0)
    total, full = _drift.transformed_cost_and_gradient(
        coords, frames, idx_i, idx_j, params, 80.0, 1.0)
    assert total == pytest.approx(plain, rel=1e-12)
    np.testing.assert_allclose(full[:, 9:12], gradient, rtol=1e-9, atol=1e-12)


def _drift_with_matrix(n_frames=100, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)
    matrix = np.repeat(np.eye(3)[None], n_frames, 0)
    matrix[:, 0, 0] += 1e-3 * t / n_frames
    matrix[:, 1, 1] += 1e-3 * t / n_frames
    matrix[:, 0, 1] += 5e-4 * t / n_frames
    return Drift(rng.normal(0, 20, (n_frames, 3)), matrix=matrix,
                 centre_nm=[5000.0, 4000.0, 0.0], transformation="xy affine")


def _table(n_frames=100, n=500, seed=5):
    rng = np.random.default_rng(seed)
    return Localizations({"frame": rng.integers(0, n_frames, n),
                          "x_nm": rng.uniform(0, 10000, n),
                          "y_nm": rng.uniform(0, 8000, n),
                          "z_nm": rng.uniform(-200, 200, n)}, {"units": "nm"})


def test_apply_moves_each_localization_by_its_frames_transformation():
    drift, locs = _drift_with_matrix(), _table()
    out = drift.apply(locs)
    xyz = np.column_stack([np.asarray(locs[k]) for k in ("x_nm", "y_nm", "z_nm")])
    f = np.asarray(locs["frame"])
    expected = (drift.centre_nm + np.einsum("fij,fj->fi", drift.matrix[f],
                                            xyz - drift.centre_nm)
                - drift.drift[f])
    for i, k in enumerate(("x_nm", "y_nm", "z_nm")):
        np.testing.assert_allclose(out[k], expected[:, i], atol=1e-6)
    assert out.metadata["drift_correction"]["transformation"] == "xy affine"


def test_a_table_in_pixels_moves_as_the_one_in_nm():
    drift, locs = _drift_with_matrix(), _table()
    columns = dict(locs.columns)
    columns["x_pix"] = np.asarray(columns["x_nm"]) / 100.0
    columns["y_pix"] = np.asarray(columns["y_nm"]) / 100.0
    out = drift.apply(Localizations(columns, {"units": "nm"}), pixelsize_nm=100.0)
    np.testing.assert_allclose(np.asarray(out["x_pix"]) * 100.0, out["x_nm"],
                               atol=1e-6)
    np.testing.assert_allclose(np.asarray(out["y_pix"]) * 100.0, out["y_nm"],
                               atol=1e-6)


def test_two_corrections_in_sequence_are_one():
    first, locs = _drift_with_matrix(seed=4), _table()
    second = _drift_with_matrix(seed=6)
    second.centre_nm = np.array([3000.0, 1000.0, 0.0])
    both = first.then(second)
    a, b = both.apply(locs), second.apply(first.apply(locs))
    for k in ("x_nm", "y_nm", "z_nm"):
        np.testing.assert_allclose(a[k], b[k], atol=1e-6)
    # the plain shifts still simply add
    plain = Drift(first.drift).then(Drift(second.drift))
    np.testing.assert_allclose(plain.drift, first.drift + second.drift)


def test_the_transformation_is_kept_in_the_session_and_in_the_driftc_file(tmp_path):
    drift, locs = _drift_with_matrix(), _table()
    back = Drift.from_dict(drift.to_dict())
    path = save_drift_corrected(tmp_path / "a.hdf5", drift.apply(locs), drift)
    loaded = load_drift(path)
    for other in (back, loaded):
        np.testing.assert_allclose(other.matrix, drift.matrix)
        np.testing.assert_allclose(other.centre_nm, drift.centre_nm)
        assert other.transformation == "xy affine"
        np.testing.assert_allclose(other.apply(locs)["x_nm"],
                                   drift.apply(locs)["x_nm"], atol=1e-4)


def test_transform_applies_another_files_transformation(tmp_path):
    drift, locs = _drift_with_matrix(), _table()
    path = save_drift_corrected(tmp_path / "cam1_driftc.hdf5", drift.apply(locs),
                                drift)
    from smappy.plugins.transform import TransformSettings
    result = plugins.get("Analysis/Process/Transform")().run(
        Context(locs=locs), TransformSettings(source="drift", drift_path=str(path)))
    for k in ("x_nm", "y_nm"):
        np.testing.assert_allclose(result.locs[k], drift.apply(locs)[k], atol=1e-4)
    assert "xy affine" in result.text


def test_the_time_window_fit_refuses_a_transformation():
    locs, _, _, _ = simulate(per_frame=20)
    with pytest.raises(ValueError, match="spline"):
        estimate_drift(locs, settings("xy scale", spline=False))


def test_a_3d_affine_transformation_needs_z():
    locs, _, _, _ = simulate(per_frame=20)
    flat = Localizations({k: locs[k] for k in ("frame", "x_nm", "y_nm")},
                         {"units": "nm"})
    with pytest.raises(ValueError, match="needs z"):
        estimate_drift(flat, settings("3D affine"))


def test_the_comet_plugin_shows_the_transformation_under_the_drift():
    locs, _, _, _ = simulate(per_frame=60)
    result = plugins.get("Analysis/Drift/COMET")().run(
        Context(locs=locs), settings("xy scale"))
    plot = result.plot
    assert getattr(plot, "panels", 1) == 2
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    figure = Figure()
    plot.draw_into(figure)
    assert figure.axes[1].get_ylabel() == "transformation (ppm)"
    assert "xy scale" in result.text and "ppm" in result.text


def simulate_z(z_scale, seed=1, per_frame=150, n_points=3000, noise=8.0,
               depth=600.0):
    """As `simulate`, with no rotation, and z scaled by ``z_scale`` over
    the movie, linearly, about z = 0: a focus-dependent axial scale."""
    rng = np.random.default_rng(seed)
    structure = rng.random((n_points, 3)) * [FIELD, FIELD, depth] - [0, 0, depth / 2]
    t = np.arange(N_FRAMES)
    base = (3 * np.sin(t / 60 + .37) + t / 50) * 10.0
    drift = np.column_stack([base, 0.7 * base, -0.5 * base])
    s = 5e-3 * np.sin(np.pi * t / N_FRAMES)
    matrix = np.repeat(np.eye(3)[None], N_FRAMES, 0)
    matrix[:, 0, 0] += s
    matrix[:, 1, 1] += s
    matrix[:, 2, 2] += z_scale * t / N_FRAMES
    frame = np.repeat(t, per_frame)
    rng.shuffle(frame)
    true = structure[rng.integers(0, n_points, frame.size)]
    centre = np.array([FIELD / 2, FIELD / 2, 0.0])
    seen = (centre + np.einsum("fij,fj->fi", matrix[frame], true - centre)
            + drift[frame] + rng.normal(0, noise, true.shape))
    locs = Localizations({"frame": frame.astype(np.int64),
                          **{name: seen[:, i].astype(np.float32)
                             for i, name in enumerate(("x_nm", "y_nm", "z_nm"))}},
                         {"units": "nm"})
    return locs, true


def z_residual_nm(locs, true):
    xyz = np.column_stack([np.asarray(locs[k], float)
                           for k in ("x_nm", "y_nm", "z_nm")])
    design = np.column_stack([true, np.ones(len(true))])
    coefficients, *_ = np.linalg.lstsq(design, xyz, rcond=None)
    return np.sqrt(((xyz - design @ coefficients)[:, 2] ** 2).mean())


def test_a_scale_of_z_is_fitted_apart_from_the_xy_map():
    # 5% over 600 nm of depth: 15 nm at the top and bottom, against 8 nm noise
    locs, true = simulate_z(0.05)
    plain = estimate_drift(locs, settings("xy affine"))
    with_z = estimate_drift(locs, settings("xy affine + z scale",
                                           reference="first frame"))
    assert set(with_z.parameters()) == {"a_xx", "a_xy", "a_yx", "a_yy", "z_scale"}
    assert z_residual_nm(with_z.apply(locs), true) < 7.95
    assert z_residual_nm(plain.apply(locs), true) > 8.3
    # in nm, z is as well determined as x and y: a few thousand ppm of a
    # 300 nm half-depth is a nanometre
    assert np.abs(with_z.parameters()["z_scale"]
                  - 0.05 * np.arange(N_FRAMES) / N_FRAMES).max() < 5e-3
    # nothing turns about x or y in this model
    np.testing.assert_allclose(with_z.matrix[:, :2, 2], 0.0)
    np.testing.assert_allclose(with_z.matrix[:, 2, :2], 0.0)


def test_the_transformation_penalty_holds_the_z_entries_near_none():
    locs, true = simulate_z(0.0)
    free = estimate_drift(locs, settings("3D affine", reference="first frame"))
    held = estimate_drift(locs, settings("3D affine", reference="first frame",
                                         transformation_penalty=1e-3))
    z_entries = ("a_xz", "a_yz", "a_zz")
    wander = max(np.abs(free.parameters()[k]).max() for k in z_entries)
    kept = max(np.abs(held.parameters()[k]).max() for k in z_entries)
    assert wander > 2e-3 and kept < 1e-3
    # while the scale of x and y, which the field pins down, is still found
    scale = 5e-3 * np.sin(np.pi * np.arange(N_FRAMES) / N_FRAMES)
    for k in ("a_xx", "a_yy"):
        assert np.abs(held.parameters()[k] - scale).max() < TOLERANCE
