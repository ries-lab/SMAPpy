"""The goodness of fit of a network's localizations (`smappy.netfit.goodness`): about 0 where the
fit explains the frame, large where a spot is off or a neighbour is missing."""
import numpy as np
from test_netfit_channels import _spline

from smappy.netfit import goodness as G


def _scene(rng, n=12, shape=(64, 64)):
    splines = (_spline(0.5), _spline(-0.5))
    xy0 = rng.uniform(10, 54, (n, 2))
    xy1 = xy0 + [1.7, -2.3]
    z = rng.uniform(-200, 200, n)
    photons = np.full(n, 3000.0)
    bg = np.tile([10.0, 20.0], (n, 1))
    frame = np.zeros(n, int)
    stack = np.zeros((1, 2) + shape)
    for c, xy in enumerate((xy0, xy1)):
        p, corner = G._spots(splines[c], xy[:, 0], xy[:, 1], z, photons)
        G._paint(stack[0, c], p, corner)
        stack[0, c] += bg[0, c]
    return splines, rng.poisson(stack).astype(float), frame, xy0, xy1, z, photons, bg


def test_the_true_molecules_explain_their_frame():
    splines, data, frame, xy0, xy1, z, ph, bg = _scene(np.random.default_rng(1))
    g = G.goodness(splines, data, frame, xy0, xy1, z, ph, bg)
    assert abs(np.median(g)) < 1.0 and np.all(g < 3.5)


def test_a_spot_put_off_or_a_neighbour_missed_shows():
    splines, data, frame, xy0, xy1, z, ph, bg = _scene(np.random.default_rng(2))
    off = xy0.copy(); off[0] += [1.5, 0]
    g = G.goodness(splines, data, frame, off, off + [1.7, -2.3], z, ph, bg)
    assert g[0] > 5 and g[0] > np.median(g[1:]) + 4
    # the network found only every other molecule: those kept see the missed ones' light
    keep = np.arange(len(frame)) % 2 == 0
    g_all = G.goodness(splines, data, frame, xy0, xy1, z, ph, bg)
    g_kept = G.goodness(splines, data, frame[keep], xy0[keep], xy1[keep], z[keep], ph[keep], bg[keep])
    assert np.median(g_kept) > np.median(g_all[keep]) + 2


def test_an_uneven_background_is_explained_by_the_map_and_not_by_a_level():
    """A network that maps its background is judged with its map (plex-plan §16, Fig. 2c): a level a
    frame counts a slope as the spots' misfit."""
    rng = np.random.default_rng(3)
    splines, _, frame, xy0, xy1, z, ph, bg = _scene(rng)
    ramp = np.linspace(0.2, 1.8, 64)[None, None, None, :] * np.array([10.0, 20.0])[None, :, None, None]
    stack = np.zeros((1, 2, 64, 64)) + ramp
    for c, xy in enumerate((xy0, xy1)):
        p, corner = G._spots(splines[c], xy[:, 0], xy[:, 1], z, ph)
        G._paint(stack[0, c], p, corner)
    data = rng.poisson(stack).astype(float)
    g_map = G.goodness(splines, data, frame, xy0, xy1, z, ph, None, background_map=ramp.repeat(64, 2))
    g_level = G.goodness(splines, data, frame, xy0, xy1, z, ph, bg)
    assert abs(np.median(g_map)) < 1.0 and np.median(g_level) > np.median(g_map) + 2
