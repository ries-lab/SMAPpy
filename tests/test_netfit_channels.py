"""netfit on two channels (docs/decode/plex-plan.md §16): a split camera's halves on one grid, the
registration's remainder as offset maps, two-channel batches, and a network with a layout for them.

The dual calibration is synthetic: an up-down splitter with a mirror in the secondary path, a small
rotation and scale so that the remainder changes across the field, as a real one does.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smappy.calibrate.core import spline_coefficients
from smappy.calibrate.dual import DualColorCalibration, split_geometry
from smappy.io.calibration import SplineCalibration
from smappy.netfit import channels as CH
from smappy.netfit import extract, loss, network
from smappy.netfit.render import SplineRenderer
from smappy.netfit.settings import CameraModel, ChannelSettings, SimulationSettings
from smappy.netfit.simulate import draw_batch_2c

SHAPE = (160, 96)          # rows, columns: the main half above row 80, the secondary below
SPLIT = 80


def _volume(nz=21, size=17, astigmatism=0.5):
    z, y, x = np.mgrid[:nz, :size, :size].astype(float)
    t = (z - (nz - 1) / 2) / 6
    sx, sy = 1.3 * np.sqrt(1 + (t - astigmatism) ** 2), 1.3 * np.sqrt(1 + (t + astigmatism) ** 2)
    return np.exp(-.5 * (((x - size / 2) / sx) ** 2 + ((y - size / 2) / sy) ** 2)) / (2 * np.pi * sx * sy)


def _spline(astigmatism):
    v = _volume(astigmatism=astigmatism)
    return SplineCalibration(spline_coefficients(v), 50., 10., x0=8.0, psf=v, parameters={})


def dual():
    """Secondary (lower, mirrored in y) onto main: a mirror about the split, a 1 % scale, a turn of
    0.01 rad and a shift of a few pixels."""
    a, s = 0.01, 1.01
    t = np.array([[s * np.cos(a), -s * np.sin(a), 3.4],
                  [-s * np.sin(a), -s * np.cos(a), 2 * SPLIT - 1 - 5.7],
                  [0., 0., 1.]])
    secondary = _spline(-0.5)
    geometry = split_geometry("up-down mirrored", "upper", SHAPE, split_position=SPLIT)
    return DualColorCalibration(_spline(0.5), SplineCalibration(
        spline_coefficients(np.flip(secondary.psf, 1).copy()), 50., 10., x0=8.0, psf=np.flip(secondary.psf, 1).copy(),
        parameters={}), t, geometry, {})


def test_channel_one_holds_a_molecule_where_the_registration_puts_its_partner():
    cal = dual()
    two = CH.TwoChannels.from_calibration(cal)
    assert all(isinstance(v, int) for v in two.shift)
    r0, r1, c0, c1 = two.box
    rng = np.random.default_rng(0)
    p = np.c_[rng.uniform(c0, c1 - 1, 200), rng.uniform(r0, r1 - 1, 200)]
    q = p + np.asarray(two.shift) + two.remainder(p)          # in channel 1's own pixels
    np.testing.assert_allclose(two.layout.on_frame(1, q), cal.to_secondary(two.on_frame(p)), atol=1e-9)
    # every pixel of the box is a pixel of both channels
    (w1, h1) = two.layout.sizes[1]
    assert r0 + two.shift[1] >= 0 and r1 - 1 + two.shift[1] <= h1 - 1
    assert c0 + two.shift[0] >= 0 and c1 - 1 + two.shift[0] <= w1 - 1
    # the turn makes the remainder change across the field, a few tenths of a pixel to a pixel
    d = two.offsets()
    assert d.shape == (2, r1 - r0, c1 - c0) and 0.5 < np.ptp(d[1]) < 3


def test_the_cut_brings_channel_one_onto_channel_zero_s_grid_by_whole_pixels():
    cal = dual()
    two = CH.TwoChannels.from_calibration(cal)
    rows, cols = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    frame = (1000 * rows + cols).astype(float)[None]           # each pixel says where it is
    cut = two.cut(frame)[0]
    r0, r1, c0, c1 = two.box
    gr, gc = np.mgrid[r0:r1, c0:c1]
    p = np.c_[gc.ravel(), gr.ravel()].astype(float)
    np.testing.assert_array_equal(cut[0].ravel(), 1000 * gr.ravel() + gc.ravel())
    on = two.layout.on_frame(1, p + np.asarray(two.shift)).round().astype(int)
    np.testing.assert_array_equal(cut[1].ravel(), 1000 * on[:, 1] + on[:, 0])


def test_a_whole_grid_spot_is_the_box_spot_where_the_box_reaches():
    cal = _spline(0.5)
    box, whole = SplineRenderer(cal, dtype=torch.float64), SplineRenderer(cal, dtype=torch.float64, whole=True)
    x, y, z = torch.tensor([20.3, 31.7]), torch.tensor([22.6, 18.2]), torch.tensor([-150.0, 220.0])
    a = box.draw(torch.zeros(2, 48, 48, dtype=torch.float64), torch.arange(2), x, y, z, torch.ones(2))
    b = whole.draw(torch.zeros(2, 48, 48, dtype=torch.float64), torch.arange(2), x, y, z, torch.ones(2))
    inside = a > 0
    torch.testing.assert_close(b[inside], a[inside])
    assert float(b.sum()) > float(a.sum())                     # the whole grid holds the light the box cut


@pytest.mark.parametrize("kind", ["biplane", "colour"])
def test_a_two_channel_batch_carries_each_crop_s_offset_maps_and_photons(kind):
    cal = dual()
    two = CH.TwoChannels.from_calibration(cal)
    grid = CH.Grid(two)
    renderers = [SplineRenderer(s, whole=True) for s in CH.splines(cal, kind)]
    sim = SimulationSettings(frame_px=24, photons_range=(1000.0, 5000.0), background=(10.0, 20.0), z_nm=(-300.0, 300.0))
    chs = ChannelSettings(kind=kind, share=(0.2, 0.4), background=[(10.0, 20.0), (50.0, 60.0)])
    g = torch.Generator().manual_seed(1)
    b = draw_batch_2c(renderers, grid, sim, CameraModel(), chs, 6, g)
    assert b.frames.shape == (6, 2, 3, 24, 24) and b.aux.shape == (6, 2, 24, 24)
    # each crop's maps are the box's remainder at its pixels: some crop of the whole map
    d = torch.as_tensor(two.offsets())
    found = [(d[:, r:r + 24, c:c + 24] == b.aux[0]).all() for r in range(d.shape[1] - 23) for c in range(d.shape[2] - 23)]
    assert any(found)
    assert ((b.background[:, 1] >= 50) & (b.background[:, 1] <= 60)).all()
    if kind == "colour":
        on = b.mask
        share = b.photons[..., 1][on] / b.photons.sum(-1)[on]
        assert b.photons.shape[-1] == 2 and ((share > 0.2 - 1e-5) & (share < 0.4 + 1e-5)).all()
    else:
        assert b.photons.dim() == 2


def test_a_one_channel_network_s_layout_is_the_one_its_output_always_had():
    assert network.Layout.of(torch.zeros(1, 10, 2, 2)) == network.Layout()
    lay = network.Layout(reported=True)
    assert (lay.phot(), lay.xyz(), lay.sig(), lay.sig_xyz(), lay.bg, lay.rep, lay.channels) == (1, 2, 5, 6, 9, 10, 13)
    two = network.Layout(photons=2, backgrounds=2, reported=True)
    assert (two.xyz(), two.sig(), two.sig_xyz(), two.bg, two.rep, two.channels) == (3, 6, 8, 11, 13, 16)


def test_a_two_channel_network_learns_a_batch_it_is_shown():
    """A small network on one colour batch, 25 steps: the loss falls by a third and the extraction reads
    the layout -- each detection with both channels' photons."""
    cal = dual()
    two = CH.TwoChannels.from_calibration(cal)
    sim = SimulationSettings(frame_px=16, photons_range=(3000.0, 6000.0), background=(10.0, 20.0),
                             z_nm=(-300.0, 300.0), density_per_40px=20.0)
    chs = ChannelSettings(kind="colour")
    g = torch.Generator().manual_seed(2)
    torch.manual_seed(2)
    b = draw_batch_2c([SplineRenderer(s, whole=True) for s in CH.splines(cal, "colour")], CH.Grid(two), sim,
                      CameraModel(), chs, 4, g)
    model = network.TwoStageUNet(window=3, features=8, channels=2, photons=2, aux=2, init="kaiming")
    sc = network.Scaling.for_simulation(sim, chs)
    x = (b.frames - sc.input_offset) / sc.input_scale
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    first = None
    for _ in range(25):
        value, _ = loss.loss(model(x, aux=b.aux), b, sc, layout=model.layout)
        opt.zero_grad()
        value.backward()
        opt.step()
        first = float(value.detach()) if first is None else first
    assert float(value.detach()) < 2 / 3 * first
    e = extract.extract(model(x, aux=b.aux).detach(), sc, 0.05, "mixture_peak", layout=model.layout)
    assert e["photons_c"].shape == (len(e["x"]), 2) and e["background_c"].shape == (len(e["x"]), 2)
