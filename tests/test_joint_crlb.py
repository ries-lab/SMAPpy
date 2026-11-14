"""The joint Cramer-Rao bound of overlapping spots and of a molecule over several frames
(`smappy.training.joint_crlb`), on an astigmatic Gaussian whose derivatives are taken numerically:
alone it is the single-emitter bound, apart two spots do not touch, together the bound grows as
they approach and a prior caps it, three frames divide the variance by three, the units do not
matter, and a maximum-likelihood fit of two overlapping spots scatters by the bound."""
import numpy as np
import pytest

from smappy.training.joint_crlb import (
    Spots,
    bounds,
    frame_fisher,
    frames_on,
    nearest_neighbour,
    single_bounds,
    variances,
)

H = 11                                  # the patch
SHAPE = (32, 32)


def spot(x, y, z, photons):
    """An astigmatic Gaussian's expected photons on its patch, x along rows and y along columns
    (DECODE's order), z in nm; and the patch's corner."""
    corner = np.array([round(x) - H // 2, round(y) - H // 2])
    r = corner[0] + np.arange(H)[:, None] - x
    c = corner[1] + np.arange(H)[None, :] - y
    sx = 1.3 * np.sqrt(1 + ((z - 300) / 400) ** 2)
    sy = 1.3 * np.sqrt(1 + ((z + 300) / 400) ** 2)
    g = np.exp(-0.5 * (r / sx) ** 2 - 0.5 * (c / sy) ** 2) / (2 * np.pi * sx * sy)
    return photons * g, corner


def derivatives(x, y, z, photons, step=(1e-4, 1e-4, 1e-2)):
    """d model / d (x, y, z, photons) on the patch, central differences, the patch held at x, y's."""
    corner = spot(x, y, z, photons)[1]

    def at(dx=0.0, dy=0.0, dz=0.0, n=photons):
        r = corner[0] + np.arange(H)[:, None] - (x + dx)
        c = corner[1] + np.arange(H)[None, :] - (y + dy)
        zz = z + dz
        sx = 1.3 * np.sqrt(1 + ((zz - 300) / 400) ** 2)
        sy = 1.3 * np.sqrt(1 + ((zz + 300) / 400) ** 2)
        return n * np.exp(-0.5 * (r / sx) ** 2 - 0.5 * (c / sy) ** 2) / (2 * np.pi * sx * sy)
    hx, hy, hz = step
    return np.stack([(at(dx=hx) - at(dx=-hx)) / (2 * hx), (at(dy=hy) - at(dy=-hy)) / (2 * hy),
                     (at(dz=hz) - at(dz=-hz)) / (2 * hz), at() / photons])


def make(emitters, background):
    """`Spots` of ``(frame, ident, x, y, z, photons)`` rows; ``background`` a value per frame."""
    e = np.asarray(emitters, float)
    models, corners, grads = [], [], []
    for _, _, x, y, z, n in e:
        m, c = spot(x, y, z, n)
        models.append(m)
        corners.append(c)
        grads.append(derivatives(x, y, z, n))
    return Spots(frame=e[:, 0].astype(int), ident=e[:, 1].astype(int), corner=np.array(corners),
                 model=np.array(models), grads=np.array(grads), background=np.asarray(background, float),
                 shape=SHAPE)


def direct(x, y, z, photons, background):
    """The single-emitter bound by the textbook: a 5 x 5 Fisher matrix, numpy's inverse."""
    m = spot(x, y, z, photons)[0] + background
    d = np.concatenate([derivatives(x, y, z, photons), np.ones((1, H, H))]).reshape(5, -1)
    return np.sqrt(np.diag(np.linalg.inv((d / m.ravel()) @ d.T))[:4])


def test_a_spot_alone_has_the_single_emitter_bound_of_the_textbook():
    s = make([(0, 0, 15.3, 16.2, 100.0, 2000.0)], [20.0])
    b = bounds(s)
    want = direct(15.3, 16.2, 100.0, 2000.0, 20.0)
    assert b.sd[0] == pytest.approx(want, rel=1e-8)
    assert single_bounds(s)[0] == pytest.approx(want, rel=1e-8)
    assert b.group[0] == 1
    assert 0.03 < want[0] < 0.1                                    # a twentieth of a pixel, about


def test_two_spots_whose_patches_do_not_touch_are_each_alone():
    s = make([(0, 0, 8.0, 8.0, 0.0, 1500.0), (0, 1, 22.4, 23.1, -200.0, 3000.0)], [10.0])
    b = bounds(s)
    assert b.sd == pytest.approx(single_bounds(s), rel=1e-8)
    assert list(b.group) == [1, 1]


def test_a_neighbour_raises_the_bound_more_the_closer_and_the_frame_alone_cannot_separate_a_pair_on_top():
    def lateral(d, prior=None):
        s = make([(0, 0, 16.0, 16.0, 0.0, 2000.0), (0, 1, 16.0 + d, 16.0, 0.0, 2000.0)], [20.0])
        b = bounds(s, prior=prior)
        return b.sd[0, 0], b.condition[0], b.group[0]
    alone = direct(16.0, 16.0, 0.0, 2000.0, 20.0)[0]
    far, _, group = lateral(9.0)                                   # 5.5 widths: tails touch
    assert group == 2 and far == pytest.approx(alone, rel=0.002)
    assert lateral(6.0)[0] > 1.05 * alone                          # 3.7 widths: already 10 %
    curve = [lateral(d)[0] for d in (3.0, 2.0, 1.5, 1.0, 0.6, 0.3)]
    assert np.all(np.diff(curve) > 0)                              # closer, worse
    assert curve[2] > 1.3 * alone
    on_top, condition, _ = lateral(0.02)
    assert on_top > 20 * alone and condition > 1e4                # diverging, finite, and flagged
    capped = lateral(0.02, prior=(1.0, 1.0, 500.0, None))[0]
    assert alone < capped < 1.0                                    # a pixel at most
    assert lateral(9.0, prior=(1.0, 1.0, 500.0, None))[0] == pytest.approx(far, rel=0.01)


def test_three_frames_of_one_molecule_divide_its_variance_by_three():
    rows = [(f, 7, 14.2, 15.6, 150.0, 2500.0) for f in range(3)]
    s = make(rows, [15.0, 15.0, 15.0])
    alone = single_bounds(s)
    with_context = bounds(s, context=1)
    # x, y and z are one parameter over the frames: the centre frame sees all three, an end two
    assert with_context.sd[1, :3] == pytest.approx(alone[1, :3] / np.sqrt(3), rel=1e-8)
    assert with_context.sd[0, :3] == pytest.approx(alone[0, :3] / np.sqrt(2), rel=1e-8)
    # the photons are a frame's own: better only for the place being better known
    assert alone[1, 3] / np.sqrt(3) < with_context.sd[1, 3] < alone[1, 3]
    assert bounds(s).sd == pytest.approx(alone, rel=1e-8)          # no context: each frame alone
    assert list(frames_on(s.frame, s.ident, 1)) == [2, 3, 2]


def test_context_chains_through_a_neighbour_in_another_frame():
    """A molecule beside another in frame 0 only: with context, its frame 1 alone is helped by its
    frame 0, which its neighbour there costs it."""
    s = make([(0, 0, 16.0, 16.0, 0.0, 2000.0), (0, 1, 17.2, 16.0, 0.0, 2000.0),
              (1, 0, 16.0, 16.0, 0.0, 2000.0)], [20.0, 20.0])
    b = bounds(s, context=1)
    alone = single_bounds(s)[2, 0]
    assert alone / np.sqrt(2) < b.sd[2, 0] < alone
    assert b.group[2] == 3


def test_the_units_and_the_range_of_the_numbers_do_not_matter():
    s = make([(0, 0, 16.0, 16.0, 50.0, 1e5), (0, 1, 17.0, 16.5, -100.0, 300.0)], [0.5])
    b = bounds(s)
    um = make([(0, 0, 16.0, 16.0, 50.0, 1e5), (0, 1, 17.0, 16.5, -100.0, 300.0)], [0.5])
    um.grads[:, 2] *= 1000.0                                       # z in µm
    b_um = bounds(um)
    assert b_um.sd[:, 2] * 1000 == pytest.approx(b.sd[:, 2], rel=1e-7)
    assert b_um.sd[:, [0, 1, 3]] == pytest.approx(b.sd[:, [0, 1, 3]], rel=1e-7)
    # the same matrix inverted by numpy's float64 inverse: the scaling changes nothing exact
    fisher = frame_fisher(s.corner, s.model, s.grads, 0.5, SHAPE)[0].toarray()
    assert np.sqrt(variances(fisher)[0]) == pytest.approx(np.sqrt(np.diag(np.linalg.inv(fisher))), rel=1e-6)


def test_em_excess_noise_doubles_the_variance():
    s = make([(0, 0, 16.0, 16.0, 0.0, 2000.0), (0, 1, 17.5, 16.0, 0.0, 1000.0)], [20.0])
    assert bounds(s, excess=2.0).sd == pytest.approx(bounds(s).sd * np.sqrt(2), rel=1e-8)


def test_a_spot_partly_off_the_frame_loses_its_pixels_there():
    s = make([(0, 0, 1.0, 16.0, 0.0, 2000.0)], [20.0])
    assert bounds(s).sd[0, 0] > 1.05 * single_bounds(s)[0, 0]


def test_maximum_likelihood_on_two_overlapping_spots_scatters_by_the_joint_bound_not_the_single():
    """Two spots 1.2 pixels apart, 10 000 photons each: 300 Poisson frames, each fitted by
    Fisher scoring from the truth over both spots and the background.  At 3000 photons the fit is
    not yet efficient at this distance (its scatter 5 - 27 % above the bound)."""
    truth = np.array([16.0, 16.0, 100.0, 1e4, 17.2, 16.3, -100.0, 1e4, 15.0])
    corner = np.array([16 - 8, 16 - 8])
    n_pix = 18
    rr = corner[0] + np.arange(n_pix)[:, None]
    cc = corner[1] + np.arange(n_pix)[None, :]

    def model(p):
        out = np.full((n_pix, n_pix), p[8])
        for x, y, z, n in (p[:4], p[4:8]):
            sx = 1.3 * np.sqrt(1 + ((z - 300) / 400) ** 2)
            sy = 1.3 * np.sqrt(1 + ((z + 300) / 400) ** 2)
            out = out + n * np.exp(-0.5 * ((rr - x) / sx) ** 2 - 0.5 * ((cc - y) / sy) ** 2) / (2 * np.pi * sx * sy)
        return out.ravel()
    step = np.array([1e-4, 1e-4, 1e-2, 1e-2] * 2 + [1e-3])

    def jac(p):
        return np.stack([(model(p + e) - model(p - e)) / (2 * h) for e, h in zip(np.diag(step), step)])
    rng = np.random.default_rng(1)
    found = []
    for _ in range(300):
        k = rng.poisson(model(truth))
        p = truth.copy()
        for _ in range(15):                                        # Levenberg-Marquardt on Fisher scoring
            mu, j = model(p), jac(p)
            fisher = (j / mu) @ j.T
            p = p + np.linalg.solve(fisher + 1e-3 * np.diag(np.diag(fisher)), j @ (k / mu - 1))
            p[[3, 7]] = np.maximum(p[[3, 7]], 10.0)
            p[8] = max(p[8], 0.1)
        found.append(p)
    spread = np.std(found, axis=0)
    # the bound of the same configuration: both spots on the 18 x 18 pixels, one background
    j, mu = jac(truth), model(truth)
    exact = np.sqrt(np.diag(np.linalg.inv((j / mu) @ j.T)))
    s = make([(0, 0, 16.0, 16.0, 100.0, 1e4), (0, 1, 17.2, 16.3, -100.0, 1e4)], [15.0])
    joint = bounds(s).sd
    alone = single_bounds(s)
    # 11 x 11 patches, each spot cut at its own, against the whole 18 x 18 pixels: a few percent
    assert joint[:, :2].ravel() == pytest.approx(exact[[0, 1, 4, 5]], rel=0.06)
    assert spread[[0, 1, 4, 5]] == pytest.approx(joint[:, :2].ravel(), rel=0.12)
    assert np.all(spread[[0, 1, 4, 5]] > 2 * alone[:, :2].ravel())


def test_nearest_neighbour_is_per_frame():
    d = nearest_neighbour([0, 0, 0, 1], [[0, 0], [3, 4], [10, 0], [0, 0.5]])
    assert d[:3] == pytest.approx([5.0, 5.0, np.hypot(7, 4)])
    assert np.isinf(d[3])


def test_smappys_spline_spots_are_drawn_as_the_simulation_draws_them_and_alone_give_the_calibrations_crlb():
    from smappy.calibrate.crlb import spline_crlb
    from smappy.simulate.camera import _draw_spline
    from smappy.training.joint_crlb import spline_spots
    from test_calibration_crlb import astigmatic
    cal = astigmatic()
    rng = np.random.default_rng(0)
    n = 30
    frame, x, y = rng.integers(0, 3, n), rng.uniform(2, 30, n), rng.uniform(2, 30, n)
    z, photons = rng.uniform(-500, 500, n), rng.uniform(1000, 5000, n)
    drawn = np.zeros((3, 32, 32))
    _draw_spline(drawn, frame, np.c_[x, y], z, photons, cal)
    s = spline_spots(cal, frame, x, y, z, photons, np.zeros(3), (32, 32))
    h = s.model.shape[1]
    canvas = np.zeros((3, 32 + 2 * h, 32 + 2 * h))
    for f, (r, c), m in zip(s.frame, s.corner, s.model):
        canvas[f, r + h:r + 2 * h, c + h:c + 2 * h] += m
    assert canvas[:, h:h + 32, h:h + 32] == pytest.approx(drawn, abs=1e-9)
    one = spline_spots(cal, [0], [16.0], [16.0], [300.0], [2000.0], [20.0], (32, 32))
    want = spline_crlb(cal, 300.0, photons=2000, background=20, roisize=h)
    assert single_bounds(one)[0] == pytest.approx([want["x_px"][0], want["y_px"][0], want["z_nm"][0],
                                                   want["photons"][0]], rel=1e-6)
