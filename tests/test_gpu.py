"""The GPU engine gives the CPU's image, in 2D and tilted; needs a GPU to run."""
import numpy as np
import pytest

from smappy.locs import Localizations
from smappy.render import DisplaySettings, FieldOfView, RenderSettings, render
from smappy.view3d import Projection, Slab, render_3d
from smappy.session import Session

gpu = pytest.importorskip("smappy.gpu")
try:
    ENGINE = gpu.GPUEngine()
except Exception as e:                      # no adapter on this machine
    pytest.skip(f"no GPU: {e}", allow_module_level=True)


def _table(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return Localizations({"x_nm": rng.uniform(0, 1000, n).astype(np.float32),
                          "y_nm": rng.uniform(0, 800, n).astype(np.float32),
                          "z_nm": rng.uniform(-300, 300, n).astype(np.float32),
                          "loc_precision_nm": rng.uniform(5, 20, n).astype(np.float32),
                          "photons": rng.uniform(100, 1000, n).astype(np.float32)}, {})


def test_2d_kernel_matches_the_cpu():
    rng = np.random.default_rng(1)
    n = 50000
    x, y = rng.uniform(0, 2000, n).astype(np.float32), rng.uniform(0, 1500, n).astype(np.float32)
    sigma, w = rng.uniform(4, 40, n).astype(np.float32), rng.uniform(0.5, 2, n).astype(np.float32)
    fov = FieldOfView(0, 0, 10.0, 200, 150)
    cpu, gpu_ = render(x, y, fov, sigma=sigma, weights=w), ENGINE.render(x, y, fov, sigma=sigma, weights=w)
    assert np.abs(cpu.weight - gpu_.weight).max() < 1e-3 * cpu.weight.max()
    assert np.array_equal(render(x, y, fov).weight, ENGINE.render(x, y, fov).weight)   # histogram


def test_tilted_slab_matches_the_cpu_engine():
    locs = _table()
    s = Session(locs)
    s.set_slab(Slab.from_bounds(100, 900, 100, 700, -150, 150))
    for settings in (RenderSettings(), RenderSettings(color_field="photons"),
                     RenderSettings(mode="gauss", sigma=15.0)):
        s.layers[0].state.settings = settings
        for extra in ({}, {"depth_lambda": 200.0}, {"color_by_depth": True}, {"opacity": 0.7, "slices": 6}):
            proj = Projection(azimuth=35, elevation=50, pivot=s.slab.center, zoom=5.0, **extra)
            fov = proj.fov(200, 160)
            cpu, _ = render_3d(s.layers, proj, s.slab, fov)
            proj.engine = "gpu"
            gpu_, _ = render_3d(s.layers, proj, s.slab, fov, engine=ENGINE)
            # hue-normalised colour at faint pixels is sensitive to which slice a
            # point at a float32 boundary lands in; elsewhere the two agree closely
            tolerance = 0.05 if (extra.get("opacity") and settings.color_field) else 0.02
            assert np.abs(cpu - gpu_).max() < tolerance, (settings, extra)


def test_points_mode_draws_something():
    locs = _table(2000)
    s = Session(locs)
    proj = Projection(pivot=[500, 400, 0], zoom=5.0, engine="points", point_size=3, point_alpha=0.6)
    rgb, _ = render_3d(s.layers, proj, None, proj.fov(200, 160), engine=ENGINE)
    assert rgb.shape == (160, 200, 3) and 0 < rgb.max() <= 1


def test_spheres_are_shaded_and_occlude_by_depth():
    from smappy import lut as luts
    from smappy.render import DisplaySettings
    one = Localizations({"x_nm": np.array([500.0], np.float32), "y_nm": np.array([500.0], np.float32),
                         "z_nm": np.zeros(1, np.float32)}, {})
    s = Session(one)
    proj = Projection(pivot=[500, 500, 0], zoom=2.0, engine="spheres", point_size=100, ssao_strength=0.0)
    rgb, _ = render_3d(s.layers, proj, None, proj.fov(200, 200), engine=ENGINE)
    g = rgb.max(axis=2)
    iy, ix = np.unravel_index(g.argmax(), g.shape)
    assert iy < 100 and ix < 100                          # lit from the upper left
    assert 7000 < (g > 0.05).sum() < 8200                  # a disc of radius 50 px
    # the nearer sphere (larger depth) hides the farther one, whatever the draw order
    t = luts.get("turbo")
    for zs in ([200.0, -200.0], [-200.0, 200.0]):
        two = Localizations({"x_nm": np.array([500.0, 520.0], np.float32),
                             "y_nm": np.array([500.0, 500.0], np.float32),
                             "z_nm": np.array(zs, np.float32)}, {})
        s2 = Session(two)
        s2.layers[0].state.settings = RenderSettings(color_field="z_nm", color_range=(-200, 200))
        s2.layers[0].state.display = DisplaySettings(lut="turbo")
        p2 = Projection(pivot=[510, 500, 0], zoom=2.0, engine="spheres", point_size=100, ssao_strength=0.0)
        mid = render_3d(s2.layers, p2, None, p2.fov(200, 200), engine=ENGINE)[0][100, 100]
        assert np.abs(mid - t[-1]).sum() < np.abs(mid - t[0]).sum()


def test_ssao_darkens_a_crevice_and_not_the_open():
    from smappy.render import DisplaySettings
    rng = np.random.default_rng(0)
    n = 40000
    x, y = rng.uniform(0, 2000, n), rng.uniform(0, 2000, n)
    blob = rng.normal(size=(3000, 3)) * 60
    blob[:, 2] = np.abs(blob[:, 2]) + 40
    locs = Localizations({"x_nm": np.r_[x, 1000 + blob[:, 0]].astype(np.float32),
                          "y_nm": np.r_[y, 1000 + blob[:, 1]].astype(np.float32),
                          "z_nm": np.r_[np.zeros(n), blob[:, 2]].astype(np.float32)}, {})
    s = Session(locs)
    s.layers[0].state.display = DisplaySettings(lut="gray")

    def render(strength):
        p = Projection(pivot=[1000, 1000, 0], zoom=4.0, engine="spheres", point_size=20,
                       ssao_strength=strength)
        return render_3d(s.layers, p, None, p.fov(500, 500), engine=ENGINE)[0]

    on, off = render(0.9), render(0.0)
    foot, open_ = (slice(280, 300), slice(240, 260)), (slice(50, 70), slice(50, 70))
    assert on[foot].mean() < off[foot].mean() * 0.9
    assert np.allclose(on[open_].mean(), off[open_].mean(), atol=0.02)
