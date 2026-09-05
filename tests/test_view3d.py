"""Projection, slab and engine A: the tilted image is the 2D image of the tilted table."""
from dataclasses import replace
import numpy as np

from smappy.locs import Localizations
from smappy.regions import Region
from smappy.render import DisplaySettings, FieldOfView, RenderSettings, render_locs
from smappy.view3d import Projection, Slab, project_layer, render_layer_3d


def _table(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({"x_nm": rng.uniform(0, 1000, n).astype(np.float32),
                          "y_nm": rng.uniform(0, 800, n).astype(np.float32),
                          "z_nm": rng.uniform(-300, 300, n).astype(np.float32),
                          "loc_precision_nm": rng.uniform(5, 20, n).astype(np.float32)}, {})


def test_top_view_is_the_2d_image():
    locs = _table()
    proj = Projection(pivot=[500, 400, 0], zoom=5.0)
    fov = proj.fov(200, 160)
    settings, display = RenderSettings(), DisplaySettings()
    rgb, planes = render_layer_3d(locs, np.ones(len(locs), bool), proj, None, fov, settings, display)
    shifted = Localizations({"x_nm": locs["x_nm"] - 500, "y_nm": locs["y_nm"] - 400,
                             "loc_precision_nm": locs["loc_precision_nm"]}, {})
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
                          "loc_precision_nm": np.full(n, 10, np.float32)}, {})
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
    opaque = render_layer_3d(locs, sel, replace(proj, opacity=1.0, slices=8), None, fov,
                             settings, display)[1]
    assert opaque.weight.sum() < plain.weight.sum()
    assert opaque.weight.sum() >= front_only.weight.sum() * 0.99


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
