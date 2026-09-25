"""Projection, slab and engine A: the tilted image is the 2D image of the tilted table."""
from dataclasses import replace
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.regions import Region
from smappy.render import DisplaySettings, FieldOfView, RenderSettings, normalize, render_locs
from smappy.view3d import Projection, Slab, project_layer, render_layer_3d


def _table(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({"x_nm": rng.uniform(0, 1000, n).astype(np.float32),
                          "y_nm": rng.uniform(0, 800, n).astype(np.float32),
                          "z_nm": rng.uniform(-300, 300, n).astype(np.float32),
                          "xy_err_nm": rng.uniform(5, 20, n).astype(np.float32)}, {})


def test_top_view_is_the_2d_image():
    locs = _table()
    proj = Projection(pivot=[500, 400, 0], zoom=5.0)
    fov = proj.fov(200, 160)
    settings, display = RenderSettings(), DisplaySettings()
    rgb, planes = render_layer_3d(locs, np.ones(len(locs), bool), proj, None, fov, settings, display)
    shifted = Localizations({"x_nm": locs["x_nm"] - 500, "y_nm": locs["y_nm"] - 400,
                             "xy_err_nm": locs["xy_err_nm"]}, {})
    ref = render_locs(shifted, fov, settings, display)
    assert np.allclose(planes.weight, ref.weight, atol=1e-5)


def test_rotations_are_what_they_say():
    proj = Projection(azimuth=90)                      # x -> y' ... a quarter turn about z
    xv, yv, d = proj.apply([1.0], [0.0], [0.0])
    assert np.allclose([xv[0], yv[0], d[0]], [0, 1, 0], atol=1e-12)
    proj = Projection(elevation=90)                    # front view: depth is y
    xv, yv, d = proj.apply([0.0], [1.0], [0.0])
    assert np.allclose([xv[0], yv[0], d[0]], [0, 0, 1], atol=1e-12)
    proj.preset("side")
    xv, yv, d = proj.apply([1.0], [0.0], [0.0])
    assert np.allclose(d[0], 1, atol=1e-12)


def test_slab_masks_and_comes_from_regions():
    locs = _table()
    slab = Slab.from_bounds(100, 300, 100, 300, -50, 50)
    m = slab.mask(locs["x_nm"], locs["y_nm"], locs["z_nm"])
    direct = ((locs["x_nm"] >= 100) & (locs["x_nm"] <= 300) & (locs["y_nm"] >= 100)
              & (locs["y_nm"] <= 300) & (np.abs(locs["z_nm"]) <= 50))
    assert (m == direct).all()
    line = Slab.from_region(Region.line((0, 0), (1000, 1000), 100), (-100, 100))
    assert abs(line.angle - 45) < 1e-9 and abs(line.size[0] - 1000 * np.sqrt(2)) < 1e-6
    # points along the diagonal are in, points off it by more than 50 nm are out
    assert line.mask([500, 500], [500, 600], [0, 0]).tolist() == [True, False]
    assert Slab.from_dict(line.to_dict()).size[1] == 100


def test_engine_respects_slab_preview_and_attenuation():
    locs = _table()
    slab = Slab.from_bounds(0, 1000, 0, 800, -50, 50)
    proj = Projection(pivot=[500, 400, 0], zoom=5.0, depth_lambda=100.0)
    table, idx = project_layer(locs, np.ones(len(locs), bool), proj, slab, RenderSettings())
    assert np.abs(locs["z_nm"][idx]).max() <= 50 and "_weight" in table
    assert table["_weight"].max() <= 1.0 and table["_weight"].min() < 0.5
    big = _table(3_000_000)
    small, idx = project_layer(big, np.ones(len(big), bool), proj, None, RenderSettings(), preview=True)
    assert len(small) == 2_000_000


def test_fit_frames_the_slab():
    proj = Projection(azimuth=30, elevation=40)
    slab = Slab.from_bounds(0, 2000, 0, 1000, -200, 200)
    proj.fit(slab, 400, 300)
    c = slab.corners()
    xv, yv, _ = proj.apply(c[:, 0], c[:, 1], c[:, 2])
    fov = proj.fov(400, 300)
    assert xv.min() >= fov.x0 and xv.max() <= fov.x1 and yv.min() >= fov.y0 and yv.max() <= fov.y1


def test_opacity_zero_is_the_plain_sum_and_one_hides_the_back():
    rng = np.random.default_rng(1)
    n = 4000
    # two sheets at different depths, exactly on top of each other in x/y
    x = np.tile(rng.uniform(0, 500, n // 2), 2).astype(np.float32)
    y = np.tile(rng.uniform(0, 500, n // 2), 2).astype(np.float32)
    z = np.r_[np.zeros(n // 2), np.full(n // 2, 300.0)].astype(np.float32)
    locs = Localizations({"x_nm": x, "y_nm": y, "z_nm": z,
                          "xy_err_nm": np.full(n, 10, np.float32)}, {})
    sel = np.ones(n, bool)
    settings, display = RenderSettings(), DisplaySettings()
    proj = Projection(pivot=[250, 250, 150], zoom=5.0)
    fov = proj.fov(120, 120)
    plain = render_layer_3d(locs, sel, proj, None, fov, settings, display)[1]
    composed = render_layer_3d(locs, sel, replace(proj, opacity=0.0, slices=8), None, fov,
                               settings, display)[1]
    assert np.allclose(plain.weight, composed.weight)
    front_only = render_layer_3d(locs[z > 100], np.ones(n // 2, bool), proj, None, fov,
                                 settings, display)[1]
    back_only = render_layer_3d(locs[z < 100], np.ones(n // 2, bool), proj, None, fov,
                                settings, display)[1]
    # a fixed scale at half the front sheet's peak: there the front saturates
    imax = float(front_only.weight.max()) * 0.5
    opaque = render_layer_3d(locs, sel, replace(proj, opacity=1.0, slices=8), None, fov,
                             settings, DisplaySettings(imax=imax))[1]
    assert opaque.weight.sum() < plain.weight.sum()
    # where the front sheet saturates, the back sheet must be gone: the image
    # there is the front sheet, not the back one
    covered = front_only.weight >= imax
    assert covered.any()
    assert np.allclose(opaque.weight[covered], front_only.weight[covered], rtol=1e-4)
    assert not np.allclose(opaque.weight[covered], back_only.weight[covered] + front_only.weight[covered])


def test_render_3d_gives_depth_histogram_and_selection_in_slab():
    from smappy.session import Session
    from smappy.view3d import render_3d
    locs = _table()
    s = Session(locs)
    s.set_slab(Slab.from_bounds(0, 1000, 0, 800, -100, 100))
    proj = Projection(pivot=s.slab.center, zoom=5.0)
    rgb, hist = render_3d(s.layers, proj, s.slab, proj.fov(200, 160))
    assert rgb.shape == (160, 200, 3) and hist[:, 1].sum() > 0 and abs(hist[:, 0]).max() <= 100
    s.select_in_slab = True
    assert np.abs(locs["z_nm"][s.selection(0).mask]).max() <= 100


def test_plugins_see_the_slab_only_while_the_3d_window_shows_it():
    """Ticked, the slab cuts every plugin's selection -- but a closed window
    is a restriction nobody can see, so closing it lifts the cut."""
    from smappy.session import Session
    locs = _table()
    s = Session(locs)
    s.set_slab(Slab.from_bounds(0, 1000, 0, 800, -100, 100))
    everything = int(s.selection(0).mask.sum())
    s.select_in_slab = True
    assert s.selects_slab and s.selection(0).mask.sum() < everything
    s.slab_shown = False                 # what the window does when it closes
    assert not s.selects_slab and s.selection(0).mask.sum() == everything
    assert s.scratch().selects_slab is False      # and a chain's copy agrees
    s.slab_shown = True
    assert s.selection(0).mask.sum() < everything


def test_view_rotation_is_continuous_and_angles_round_trip():
    proj = Projection(azimuth=20, elevation=70, roll=10)
    R = proj.matrix
    proj.set_matrix(R)
    assert np.allclose(proj.matrix, R, atol=1e-9)
    # past the side view and on: no pole, the matrix keeps turning the same way
    proj = Projection()
    before = None
    for _ in range(12):                              # 12 x 30 deg = a full turn
        proj.rotate_view(0.0, 30.0)
        if before is not None:
            step = proj.matrix @ before.T             # the relative rotation
            assert np.allclose(step, _rx_matrix(30.0), atol=1e-9)
        before = proj.matrix
    assert np.allclose(proj.matrix, np.eye(3), atol=1e-9)


def _rx_matrix(deg):
    from smappy.view3d import _rx
    return _rx(deg)


def test_rotated_slab_faces_move_one_at_a_time():
    slab = Slab.from_region(Region.line((0, 0), (1000, 1000), 200), (-100, 100))
    lo, hi = slab.axis_range(0)
    far_face = slab.corners()[[0, 1, 2, 3]].mean(axis=0)      # the -x face centre
    slab.set_axis_range(0, lo, hi + 300)                     # push the +x face out
    assert np.allclose(slab.corners()[[0, 1, 2, 3]].mean(axis=0), far_face)
    assert abs(slab.size[0] - (hi - lo + 300)) < 1e-9
    assert abs(slab.angle - 45) < 1e-9


def test_pivot_moves_without_the_image():
    proj = Projection(azimuth=30, elevation=50, pivot=[0, 0, 0], zoom=2.0)
    xv, yv, _ = proj.apply([100.0], [200.0], [50.0])
    fov0 = proj.fov(100, 100)
    proj.move_pivot([300, -100, 20])
    xv2, yv2, _ = proj.apply([100.0], [200.0], [50.0])
    fov1 = proj.fov(100, 100)
    assert np.allclose(xv - fov0.x0, xv2 - fov1.x0) and np.allclose(yv - fov0.y0, yv2 - fov1.y0)


def test_fixed_roll_is_a_turntable():
    proj = Projection(fix_roll=True)
    for _ in range(10):
        proj.rotate_view(7.0, 3.0)
    assert proj.roll == 0.0 and abs(proj.azimuth - 290) < 1e-9 and abs(proj.elevation - 30) < 1e-9
    # the data's z axis stays vertical on screen: no x' component
    zx, zy, _ = proj.apply([0.0], [0.0], [1.0])
    assert abs(zx[0]) < 1e-9
    free = Projection(fix_roll=False)
    free.rotate_view(30.0, 0.0)
    assert free.roll != 0.0 or free.azimuth != 0.0


def test_a_drag_to_the_right_carries_the_picture_right_in_both_modes():
    """The face towards the viewer must follow the mouse, turntable or trackball."""
    for fix_roll in (True, False):
        proj = Projection(azimuth=0.0, elevation=90.0, fix_roll=fix_roll)
        # the point the camera is looking at, one unit in front of the pivot
        front = proj.matrix.T @ np.array([0.0, 0.0, 1.0])
        proj.rotate_view(10.0, 0.0)                  # drag right
        xv, _, _ = proj.apply(*[[v] for v in front])
        assert xv[0] > 0.05, f"fix_roll={fix_roll}: the front face went left"


def test_the_index_picks_the_same_rows_as_a_walk_over_the_table():
    """The spatial index only avoids reading rows that cannot be in the slab."""
    from smappy.spatial import SpatialIndex
    from smappy.view3d import slab_candidates
    locs = _table(20_000)
    index = SpatialIndex(np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"]))
    select = np.asarray(locs["z_nm"]) > -100            # a filter, not everything
    proj = Projection(azimuth=25, elevation=55, zoom=5.0)
    for slab in (Slab.from_bounds(200, 400, 300, 500, -50, 50),      # a crop
                 Slab.from_bounds(0, 1000, 0, 800, -300, 300),       # the whole table
                 Slab([500, 400, 0], [300, 120, 200], angle=35.0)):  # rotated in plane
        walked = slab_candidates(locs, select, slab, None)
        indexed = slab_candidates(locs, select, slab, index)
        # the index hands back a superset of the slab's rows; the slab decides
        assert set(np.asarray(indexed)) <= set(np.asarray(walked))
        a = project_layer(locs, select, proj, slab, RenderSettings())[1]
        b = project_layer(locs, select, proj, slab, RenderSettings(), index=index)[1]
        assert np.array_equal(np.sort(a), np.sort(b)) and a.size


def test_the_depth_histogram_is_the_slabs_depths_with_or_without_an_index():
    """Sampling stands in for every row -- the panel hides the count axis --
    but it has to sample the slab, not the table around it."""
    from smappy.spatial import SpatialIndex
    from smappy.view3d import depth_histogram, depth_sample
    locs = _table(50_000)
    index = SpatialIndex(np.asarray(locs["x_nm"]), np.asarray(locs["y_nm"]))
    select = np.ones(len(locs), bool)
    slab = Slab.from_bounds(400, 600, 300, 500, -100, 100)    # a small part of the table
    proj = Projection(azimuth=20, elevation=50, pivot=slab.center, zoom=5.0)
    walked = depth_sample(locs, select, proj, slab, None)
    indexed = depth_sample(locs, select, proj, slab, index)
    exact = project_layer(locs, select, proj, slab, RenderSettings())[0]["depth"]
    assert walked.size == indexed.size == len(exact)          # small enough to be whole
    assert abs(float(np.mean(indexed)) - float(np.mean(exact))) < 1e-3
    hist = depth_histogram([indexed])
    assert hist[:, 1].sum() == len(exact)
    lo, hi = float(np.min(exact)), float(np.max(exact))
    assert lo - 1 <= hist[0, 0] and hist[-1, 0] <= hi + 1


def test_a_sampled_histogram_keeps_the_shape_of_the_whole_one():
    from smappy.view3d import HIST_POINTS, depth_histogram, depth_sample
    locs = _table(HIST_POINTS)                      # comfortably over the limit below
    select = np.ones(len(locs), bool)
    proj = Projection(azimuth=15, elevation=40, zoom=5.0)
    limit = 20_000        # 64 bins of ~300: the sampling noise is well under the bound
    sampled = depth_sample(locs, select, proj, None, None, limit=limit)
    whole = project_layer(locs, select, proj, None, RenderSettings())[0]["depth"]
    assert sampled.size <= limit < len(whole)
    a = depth_histogram([sampled])[:, 1]
    b = depth_histogram([np.asarray(whole)])[:, 1]
    # same shape, different scale: compare the normalised profiles
    a, b = a / a.sum(), b / b.sum()
    assert np.abs(a - b).sum() < 0.15


def test_thinning_keeps_the_count_the_order_and_the_density():
    from smappy.view3d import thinned
    rng = np.random.default_rng(3)
    rows = np.sort(rng.choice(400_000, 300_000, replace=False))
    assert thinned(rows, None) is rows and thinned(rows, 400_000) is rows   # nothing to do
    cut = thinned(rows, 20_000)
    assert cut.size == 20_000                       # exactly the budget, not about it
    assert np.all(np.diff(cut) > 0)                 # ascending: the gather stays tidy
    assert set(cut.tolist()) <= set(rows.tolist())
    # unbiased in space: the sample's density matches the whole, as a draw would
    locs = _table(300_000, seed=5)
    x = np.asarray(locs["x_nm"])[rows % 300_000]
    bins = np.linspace(x.min(), x.max(), 33)
    whole, _ = np.histogram(x, bins=bins)
    part, _ = np.histogram(np.asarray(locs["x_nm"])[cut % 300_000], bins=bins)
    a, b = part / part.sum(), whole / whole.sum()
    assert np.abs(a - b).sum() < 0.05


def test_the_budget_settles_on_the_frame_rate_it_is_asked_for():
    """A fixed point count cannot serve a base M1 and a 3090, so it is timed."""
    from smappy.view3d import PreviewBudget
    from smappy.render import FieldOfView
    fov = FieldOfView(0, 0, 10.0, 450, 350)
    for rate, overhead in ((60e6, 0.004), (6e6, 0.002), (400e6, 0.010)):
        budget, seconds = PreviewBudget(), None
        for _ in range(30):                          # a machine: fixed cost + per point
            drawn = min(budget.points(fov), 20_000_000)
            seconds = overhead + drawn / rate
            budget.record(drawn, seconds)
        assert 1 / seconds > 20, f"{rate:g}: settled at {1 / seconds:.1f} fps"
        # either it is hitting the target, or it is already drawing all it can see
        assert abs(seconds - budget.target) < 0.35 * budget.target \
            or budget.points(fov) >= budget.per_pixel * fov.nx * fov.ny * 0.99


def test_a_preview_is_bounded_by_the_budget_and_a_final_frame_is_not():
    from smappy.session import Session
    from smappy.view3d import PreviewBudget, project_layer, render_3d
    locs = _table(400_000)
    s = Session(locs)
    slab = Slab.from_bounds(0, 1000, 0, 800, -300, 300)
    s.set_slab(slab)
    proj = Projection(pivot=slab.center, zoom=5.0)
    st = s.layers[0].state
    budget = PreviewBudget(per_pixel=0.5, minimum=1000)      # 200x160 -> 16,000
    limit = budget.points(proj.fov(200, 160))
    assert limit < len(st.locs)
    drawn = project_layer(st.locs, st.filter.mask, proj, slab, st.settings,
                          preview=True, index=st.index, budget=limit)[1]
    assert drawn.size <= limit
    whole = project_layer(st.locs, st.filter.mask, proj, slab, st.settings)[1]
    assert whole.size > limit                                 # the final frame is all of it
    rgb, hist = render_3d(s.layers, proj, slab, proj.fov(200, 160), True, budget=budget)
    assert rgb.shape == (160, 200, 3) and hist[:, 1].sum() > 0
    assert budget.rate is not None                            # and it timed itself


def test_after_a_pan_a_rotation_turns_about_the_middle_of_the_screen():
    """In a large field the slab's centre is off screen once one has panned
    away from it; turning about it swings the data out of the picture."""
    proj = Projection(pivot=[0.0, 0.0, 0.0], zoom=5.0, elevation=30.0)
    proj.offset = np.array([4000.0, -2500.0])            # panned
    target = proj.screen_centre()
    xv, yv, d = proj.apply(*target)
    assert np.allclose([xv, yv, d], [4000.0, -2500.0, 0.0])

    proj.pivot_at_centre()
    assert np.allclose(proj.offset, 0) and np.allclose(proj.pivot, target)
    for _ in range(5):
        proj.rotate_view(17.0, 11.0)
        xv, yv, _ = proj.apply(*target)
        assert np.allclose([xv, yv], proj.offset)       # still in the middle


def test_moving_along_the_sight_brings_the_middle_of_the_screen_nearer():
    proj = Projection(pivot=[0.0, 0.0, 0.0], elevation=40.0, azimuth=25.0,
                      focal=5000.0)
    ahead = np.array(proj.pivot) - proj.view_axis(2) * 500.0     # behind the pivot
    before = proj.apply(*(ahead + proj.view_axis(0) * 100.0))
    proj.move_along_sight(1000.0)
    after = proj.apply(*(ahead + proj.view_axis(0) * 100.0))
    assert after[2] > before[2]                         # nearer the eye
    assert abs(after[0]) > abs(before[0])               # so it looks bigger
    assert proj.apply(*proj.screen_centre())[:2] == pytest.approx((0.0, 0.0), abs=1e-6)
