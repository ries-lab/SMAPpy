"""The Voronoi render mode: each pixel the density of the Voronoi cell it
lies in, the cells cached per table and filter, offered by the Render tab."""
import dataclasses

import numpy as np
import pytest

from smappy import render
from smappy.locs import Localizations
from smappy.render import FieldOfView, RenderSettings, render_locs

VORONOI = RenderSettings(mode="voronoi")


def lattice(spacing, x0, x1, y0, y1, seed=0):
    """A square lattice, its points a thousandth of a nm off so that no four
    are on one circle: every cell is a spacing^2 square."""
    gx = np.arange(x0 + spacing / 2, x1, spacing)
    gy = np.arange(y0 + spacing / 2, y1, spacing)
    x, y = (v.ravel() for v in np.meshgrid(gx, gy))
    jitter = np.random.default_rng(seed).normal(0, 1e-3, (2, x.size))
    return x + jitter[0], y + jitter[1]


def table(x, y, **columns):
    return Localizations({"x_nm": np.asarray(x, float), "y_nm": np.asarray(y, float),
                          **columns}, {"units": "nm"})


@pytest.mark.parametrize("pixel", [4.0, 60.0])
def test_a_uniform_field_is_a_flat_image_at_its_density(pixel):
    # 20 nm apart: 2500 per um^2.  At 4 nm a pixel lies in one cell, at
    # 60 nm it holds nine whole ones
    locs = table(*lattice(20.0, 0, 4000, 0, 4000))
    fov = FieldOfView(1000, 1000, pixel, int(1800 // pixel), int(1800 // pixel))
    image = render_locs(locs, fov, VORONOI).weight
    np.testing.assert_allclose(image, 2500.0, rtol=1e-3)


def test_a_random_field_averages_to_its_density_at_any_zoom():
    rng = np.random.default_rng(1)
    n = 40_000
    locs = table(rng.uniform(0, 20_000, n), rng.uniform(0, 20_000, n))   # 100 per um^2
    for pixel in (10.0, 100.0, 400.0):      # a cell is about 100 nm across
        size = int(10_000 // pixel)
        image = render_locs(locs, FieldOfView(5000, 5000, pixel, size, size), VORONOI).weight
        assert image.mean() == pytest.approx(100.0, rel=0.05)


def test_a_dense_cluster_is_brighter_by_its_density_ratio():
    # 30 nm apart around, 10 nm in a 600 nm square: nine times as dense
    bx, by = lattice(30.0, 0, 3000, 0, 3000)
    out = ~((bx > 1200) & (bx < 1800) & (by > 1200) & (by < 1800))
    cx, cy = lattice(10.0, 1200, 1800, 1200, 1800, seed=1)
    locs = table(np.r_[bx[out], cx], np.r_[by[out], cy])
    fov = FieldOfView(600, 600, 5.0, 360, 360)
    image = render_locs(locs, fov, VORONOI).weight
    inner = image[150:210, 150:210]               # 1350-1650 nm, in the cluster
    outer = image[20:80, 20:80]                   # 700-1000 nm, the background
    assert np.median(inner) / np.median(outer) == pytest.approx(9.0, rel=0.02)


def test_open_cells_at_the_edge_of_the_field_are_left_empty():
    rng = np.random.default_rng(2)
    locs = table(rng.uniform(0, 400, 160), rng.uniform(0, 400, 160))   # 1000 per um^2
    fov = FieldOfView(-200, -200, 5.0, 160, 160)       # the field and around it
    image = render_locs(locs, fov, VORONOI).weight
    beyond = np.ones(image.shape, bool)
    beyond[40:120, 40:120] = False
    # the open cells reach to infinity, and are empty: the corners beyond
    # the field are theirs.  A cell just inside the edge closes beyond it,
    # long and thin, so of a low density
    for corner in (image[:30, :30], image[:30, -30:], image[-30:, :30], image[-30:, -30:]):
        assert corner.max() == 0
    assert image[beyond].mean() < 0.1 * 1000
    assert image[60:100, 60:100].mean() == pytest.approx(1000, rel=0.25)


def test_the_cells_are_computed_once_for_a_table_and_reused_when_the_view_moves(monkeypatch):
    calls = []
    of = render.VoronoiCells.of.__func__

    def counted(cls, x, y):
        calls.append(len(x))
        return of(cls, x, y)
    monkeypatch.setattr(render.VoronoiCells, "of", classmethod(counted))
    x, y = lattice(20.0, 0, 2000, 0, 2000, seed=7)
    locs = table(x, y)
    a = render_locs(locs, FieldOfView(0, 0, 10.0, 100, 100), VORONOI).weight
    b = render_locs(locs, FieldOfView(500, 500, 10.0, 100, 100), VORONOI).weight
    assert len(calls) == 1
    np.testing.assert_array_equal(a[50:, 50:], b[:50, :50])   # the same cells
    render_locs(locs, FieldOfView(0, 0, 10.0, 100, 100), VORONOI, select=x < 1000)
    assert len(calls) == 2                                     # a filter: new cells


def test_a_layer_draws_the_cells_of_everything_its_filter_keeps():
    from smappy.session import Session
    x, y = lattice(20.0, 0, 2000, 0, 2000, seed=3)
    layer = Session(table(x, y, xy_err_nm=np.full(len(x), 10.0))).layers[0]
    layer.state.settings = VORONOI
    snapshot = layer.state.snapshot()
    view = FieldOfView(0, 0, 10.0, 10, 10)                # 100 nm of 2000
    assert snapshot.index is not None
    assert len(snapshot.select(view)) == len(layer.filter)


def test_colour_by_a_field_takes_each_cell_its_localizations_colour():
    x, y = lattice(20.0, 0, 1000, 0, 1000)
    z = np.where(x < 500, -100.0, 100.0)
    locs = table(x, y, z_nm=z)
    settings = dataclasses.replace(VORONOI, color_field="z_nm", color_range=(-100.0, 100.0))
    for pixel in (5.0, 60.0):
        out = render_locs(locs, FieldOfView(100, 100, pixel, int(800 // pixel),
                                            int(800 // pixel)), settings)
        hue = out.color / out.weight[..., None]
        assert not np.allclose(hue[0, 0], hue[0, -1])      # the two halves differ
        np.testing.assert_allclose(hue[0, 0], hue[-1, 0], atol=1e-5)


def test_the_3d_view_draws_a_voronoi_layer_as_points():
    from smappy.view3d import projected_settings
    locs = table([0.0, 10.0], [0.0, 10.0], xy_err_nm=np.full(2, 10.0))
    settings, _ = projected_settings(locs, VORONOI)
    assert settings.mode == "precision"


def test_the_render_tab_offers_the_voronoi_mode():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.render_tab import RenderTab
    from smappy.session import Session
    x, y = lattice(20.0, 0, 1000, 0, 1000)
    session = Session(table(x, y, xy_err_nm=np.full(len(x), 10.0)))
    tab = RenderTab(session)
    assert tab.mode.findData("voronoi") >= 0
    tab.set_mode("voronoi")
    assert session.layers[0].state.settings.mode == "voronoi"
    assert not tab._display_form.isRowVisible(tab.track_column)
    rgb, rendered = session.layers[0].render(FieldOfView(200, 200, 10.0, 60, 60))
    assert rendered.weight.max() > 0 and rgb.shape == (60, 60, 3)
