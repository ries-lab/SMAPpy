"""Rendering any column against any other: the versatile renderer.

The mapping is a division per axis, so the grid stays square and everything
downstream is unchanged; what the tests here pin down is that the division
lands where it should, that a width per axis means what it says (zero bins,
a precision only blurs an axis that is a position), and that the ordinary
picture is untouched by any of it.
"""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.render import (FieldOfView, RenderAxes, RenderSettings, axis_unit,
                           explicit_sigmas, render, render_locs, render_sigmas)


def table(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    frame = np.arange(n, dtype=np.int64)
    return Localizations({
        "x_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "y_nm": rng.uniform(0, 10_000, n).astype(np.float32),
        "z_nm": rng.uniform(-400, 400, n).astype(np.float32),
        "frame": frame,
        # a photon count that is a hundred times the frame: on axes of frame
        # against photons the picture is a straight diagonal, wherever the
        # scales put it
        "photons": (100.0 * frame).astype(np.float32),
        "xy_err_nm": np.full(n, 12.0, np.float32),
    }, {"units": "nm"})


# ------------------------------------------------------------- the kernel
def test_a_width_per_axis_agrees_with_the_reference_implementation():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 100, 400).astype(np.float32)
    y = rng.uniform(0, 100, 400).astype(np.float32)
    fov = FieldOfView.from_range((0, 100), (0, 100), 1.0)
    fast = render(x, y, fov, sigma=4.0, sigma_y=1.0)
    slow = render(x, y, fov, sigma=4.0, sigma_y=1.0, use_extension=False)
    assert np.allclose(fast.weight, slow.weight, atol=1e-5)
    # and it is not the same picture as the isotropic one it came from
    assert not np.allclose(fast.weight, render(x, y, fov, sigma=4.0).weight, atol=1e-3)


def test_a_width_of_zero_bins_along_that_axis():
    fov = FieldOfView.from_range((0, 20), (0, 20), 1.0)
    img = render([10.5], [10.5], fov, sigma=3.0, sigma_y=0.0)
    rows = img.weight.sum(axis=1)
    assert rows[10] == pytest.approx(1.0, abs=1e-5)     # one row holds everything
    assert rows.sum() == pytest.approx(1.0, abs=1e-5)
    assert (img.weight[10] > 0).sum() > 5               # and it is spread along x
    same = render([10.5], [10.5], fov, sigma=3.0, sigma_y=0.0, use_extension=False)
    assert np.allclose(img.weight, same.weight, atol=1e-6)


def test_a_per_localization_width_on_one_axis_and_a_constant_on_the_other():
    """The kernel reads both widths through one stride, so they have to meet."""
    fov = FieldOfView.from_range((0, 20), (0, 20), 1.0)
    img = render([5.5, 15.5], [10.5, 10.5], fov, sigma=np.array([1.0, 4.0]), sigma_y=0.5)
    narrow = (img.weight[:, :10] > 1e-4).sum()
    wide = (img.weight[:, 10:] > 1e-4).sum()
    assert wide > narrow


# --------------------------------------------------------- the sigma rule
def test_the_precision_blurs_a_position_axis_and_nothing_else():
    locs = table()
    fov = FieldOfView.from_range((0, 1000), (0, 1000), 1.0)
    axes = RenderAxes(x="x_nm", y="photons", y_scale=100.0)
    sigma_x, sigma_y = render_sigmas(locs, RenderSettings(axes=axes), fov)
    assert np.allclose(sigma_x, 12.0)     # the lateral precision, in nm
    assert sigma_y == 0.0                 # photons have none: bin them


def test_a_scaled_position_axis_carries_its_precision_into_render_units():
    locs = table()
    fov = FieldOfView.from_range((0, 1000), (0, 1000), 1.0)
    axes = RenderAxes(x="x_nm", y="y_nm", x_scale=4.0, y_scale=4.0)
    sigma_x, sigma_y = render_sigmas(locs, RenderSettings(axes=axes), fov)
    assert np.allclose(sigma_x, 3.0)      # 12 nm at 4 nm per render unit
    assert sigma_y is None                # the same on both: handed over once


def test_a_constant_gaussian_uses_the_explicit_widths():
    locs = table()
    fov = FieldOfView.from_range((0, 1000), (0, 1000), 1.0)
    axes = RenderAxes(x="frame", y="photons", y_scale=100.0)
    settings = RenderSettings(mode="gauss", sigma=5.0, sigma_y=250.0, axes=axes)
    assert render_sigmas(locs, settings, fov) == (5.0, 2.5)   # 250 photons / 100


def test_an_unset_second_width_follows_x_only_where_that_means_something():
    locs = table()
    same = RenderSettings(mode="gauss", sigma=8.0, axes=RenderAxes(x="x_nm", y="z_nm"))
    assert explicit_sigmas(locs, same) == (8.0, 8.0)          # both in nm
    mixed = RenderSettings(mode="gauss", sigma=8.0, axes=RenderAxes(x="x_nm", y="photons"))
    assert explicit_sigmas(locs, mixed) == (8.0, 0.0)         # nm says nothing about photons


def test_a_missing_precision_is_still_refused_on_a_position_axis():
    locs = Localizations({"x_nm": np.zeros(3, np.float32), "y_nm": np.zeros(3, np.float32)})
    fov = FieldOfView.from_range((0, 10), (0, 10), 1.0)
    with pytest.raises(KeyError, match="precision"):
        render_sigmas(locs, RenderSettings(), fov)


def test_the_unit_of_an_axis_is_what_a_scale_bar_should_say():
    assert axis_unit("x_nm") == "nm" and axis_unit("x_pix") == "pixels"
    assert axis_unit("photons") == "photons"


# ------------------------------------------------------------- the mapping
def test_the_axes_divide_each_column_by_its_own_scale():
    locs = table(n=10)
    axes = RenderAxes(x="frame", y="photons", x_scale=2.0, y_scale=100.0)
    x, y = axes.coordinates(locs)
    assert np.allclose(x, np.arange(10) / 2.0)
    assert np.allclose(y, np.arange(10))
    assert np.allclose(axes.coordinates(locs, select=np.array([3, 4]))[0], [1.5, 2.0])


def test_a_column_against_another_lands_where_the_numbers_say():
    """photons = 100 x frame, so at 100 photons per unit it is the diagonal."""
    locs = table(n=200)
    axes = RenderAxes(x="frame", y="photons", y_scale=100.0)
    fov = FieldOfView.from_range((0, 200), (0, 200), 1.0)
    settings = RenderSettings(mode="hist", axes=axes)
    img = render_locs(locs, fov, settings)
    assert img.n_locs == 200
    rows, cols = np.nonzero(img.weight)
    assert np.array_equal(rows, cols)                 # every one on the diagonal
    assert img.weight.sum() == 200


def test_the_third_axis_is_z_unless_it_is_told_otherwise():
    locs = table(n=10)
    assert RenderAxes().depth_name(locs) == "z_nm"
    assert np.allclose(RenderAxes(z="frame", z_scale=2.0).depth(locs), np.arange(10) / 2)
    assert RenderAxes(z="nothing").depth_name(locs) is None


def test_spelling_the_usual_axes_out_is_still_the_usual_picture():
    """So it keeps the index culling and the single scale bar."""
    locs = table(n=10)
    assert RenderAxes(x="x_nm", y="y_nm").normalised(locs).is_default
    assert not RenderAxes(x="x_nm", y="y_nm", y_scale=2.0).normalised(locs).is_default
    assert not RenderAxes(x="x_nm", y="photons").normalised(locs).is_default


def test_the_ordinary_picture_is_exactly_what_it_was():
    locs = table()
    fov = FieldOfView.around(locs["x_nm"], locs["y_nm"], 20.0)
    plain = render_locs(locs, fov, RenderSettings())
    spelled = render_locs(locs, fov, RenderSettings(axes=RenderAxes(x="x_nm", y="y_nm")))
    assert np.array_equal(plain.weight, spelled.weight)


def test_scaling_both_axes_together_is_the_same_picture_on_a_smaller_grid():
    locs = table()
    fov = FieldOfView.around(locs["x_nm"], locs["y_nm"], 20.0)
    plain = render_locs(locs, fov, RenderSettings())
    scaled = FieldOfView(fov.x0 / 4, fov.y0 / 4, fov.pixelsize / 4, fov.nx, fov.ny)
    settings = RenderSettings(axes=RenderAxes(x="x_nm", y="y_nm", x_scale=4.0, y_scale=4.0))
    assert np.allclose(plain.weight, render_locs(locs, scaled, settings).weight, atol=2e-3)


# ------------------------------------------------------------- the session
def session_on(axes: RenderAxes):
    from smappy.session import Session
    session = Session(table())
    session.set_axes(axes)
    return session


def test_the_axes_are_the_picture_s_and_go_on_every_layer():
    from smappy.session import Session
    session = Session(table())
    session.add_layer()
    axes = RenderAxes(x="frame", y="photons", y_scale=100.0)
    session.set_axes(axes)
    assert [l.state.settings.axes for l in session.layers] == [axes, axes]
    assert session.axes() == axes


def test_the_view_is_framed_on_the_columns_that_are_drawn():
    session = session_on(RenderAxes(x="frame", y="photons", y_scale=100.0))
    (x0, x1), (y0, y1) = session.full_view(0.0)
    assert (x0, x1) == (0.0, 2999.0)              # frames
    assert (y0, y1) == (0.0, 2999.0)              # photons / 100, the same numbers


def test_an_roi_on_a_versatile_picture_selects_what_it_encloses():
    """It is drawn on the picture, so it is in the picture's coordinates."""
    from smappy.regions import Region
    session = session_on(RenderAxes(x="frame", y="photons", y_scale=100.0))
    session.set_roi(Region.rect(100, 100, 200, 200))
    mask = session.selection().mask
    frame = np.asarray(session.locs["frame"])
    assert np.array_equal(np.flatnonzero(mask), np.flatnonzero((frame >= 100) & (frame <= 200)))


def test_the_index_is_not_used_to_narrow_a_versatile_view():
    """It answers questions about positions, which is not where we are looking."""
    session = session_on(RenderAxes(x="frame", y="photons", y_scale=100.0))
    state = session.layers[0].state
    assert state.cull_index is None
    fov = FieldOfView.from_range((0, 10), (0, 10), 1.0)      # a corner of the picture
    assert state.select(fov).size == len(session.locs)       # everything, unculled
    session.set_axes(RenderAxes())
    assert state.cull_index is not None and state.select(fov).size < len(session.locs)


# ------------------------------------------------------------------ in 3D
def test_a_rotated_view_folds_a_common_scale_into_the_precision():
    from smappy.view3d import projected_settings
    locs = table()
    settings = RenderSettings(axes=RenderAxes(x="x_nm", y="y_nm", z="z_nm",
                                              x_scale=4.0, y_scale=4.0, z_scale=4.0))
    projected, scale = projected_settings(locs, settings)
    assert projected.axes.is_default and projected.mode == "precision"
    assert scale == 4.0 and projected.sigma_settings.factor == pytest.approx(0.25)


def test_a_rotated_view_of_mixed_axes_falls_back_to_one_constant_width():
    from smappy.view3d import projected_settings
    locs = table()
    settings = RenderSettings(mode="gauss", sigma=4.0, sigma_y=200.0,
                              axes=RenderAxes(x="x_nm", y="photons", y_scale=100.0))
    projected, scale = projected_settings(locs, settings)
    # a projection mixes the axes, so 4 nm across and 2 render units up average
    assert projected.mode == "gauss" and projected.sigma == pytest.approx(3.0)
    assert projected.axes.is_default and scale == 1.0


def test_the_3d_view_draws_the_versatile_axes():
    from smappy.session import Session
    from smappy.view3d import Projection, Slab, render_3d
    session = Session(table())
    session.set_axes(RenderAxes(x="frame", y="photons", z="x_nm",
                                y_scale=100.0, z_scale=10.0))
    fov = FieldOfView.from_range((0, 3000), (0, 3000), 20.0)
    slab = Slab.from_bounds(0, 3000, 0, 3000, 0, 1000)
    rgb, hist = render_3d(session.layers, Projection(azimuth=30, elevation=20), slab, fov)
    assert rgb.shape == (fov.ny, fov.nx, 3) and rgb.max() > 0
    assert hist[:, 1].sum() > 0


def test_the_slab_is_rebuilt_when_the_picture_changes_quantity():
    """The slab is a box in render units, so it belongs to the axes it was
    built on.  Left alone across a change it is a ROI-sized box of nanometres
    over a picture of photons against frame, and the 3D window is empty."""
    from smappy.regions import Region
    from smappy.session import Session
    from smappy.view3d import Projection, render_3d

    session = Session(table(20_000))
    session.set_roi(Region.rect(4000, 4000, 5000, 5000))
    session.slab_from_roi()
    assert tuple(np.round(session.slab.size[:2])) == (1000.0, 1000.0)

    session.set_axes(RenderAxes(x="frame", y="photons", x_scale=10.0, y_scale=1000.0))
    projection = Projection()
    projection.fit(session.slab, 200, 200)
    rgb, _ = render_3d(session.layers, projection, session.slab, projection.fov(200, 200))
    assert (rgb > 0).any(), "the 3D view came up empty on the new axes"

    session.set_axes(RenderAxes())          # and back to the ordinary picture
    assert tuple(np.round(session.slab.size[:2])) != (1000.0, 1000.0)
    projection = Projection()
    projection.fit(session.slab, 200, 200)
    rgb, _ = render_3d(session.layers, projection, session.slab, projection.fov(200, 200))
    assert (rgb > 0).any()


def test_setting_the_axes_to_what_they_already_are_leaves_the_slab_alone():
    """A slab the user set by hand is not thrown away by a no-op."""
    from smappy.session import Session
    from smappy.view3d import Slab

    session = Session(table())
    session.set_slab(Slab.from_bounds(100, 200, 300, 400, -50, 50))
    session.set_axes(RenderAxes())
    assert tuple(session.slab.size) == (100.0, 100.0, 100.0)


def test_a_second_layer_knows_its_extent_on_custom_axes():
    """A layer over the same table reuses the first's spatial index and used
    to skip the bounds cache with it, so asking a two-layer session for its
    extent on any picture but the ordinary one raised -- which is what left
    the 3D window empty on dual-colour data."""
    from smappy.session import Session

    locs = table(5000)
    locs.columns["channel"] = (np.arange(len(locs)) % 2).astype(np.int64)
    session = Session(locs)
    session.add_layer(like=0)
    session.layers[0].set_bound("channel", -0.5, 0.5)
    session.layers[1].set_bound("channel", 0.5, 1.5)
    session.set_axes(RenderAxes(x="frame", y="photons", x_scale=10.0, y_scale=1000.0))
    (x0, x1), (y0, y1) = session.full_view()
    assert x1 > x0 and y1 > y0
