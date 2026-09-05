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
