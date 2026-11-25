"""netfit on a 4Pi microscope's phase channels (`netfit.fourpi`, docs/4pi.md §30): the channels on one
grid, batches whose targets are what Spline 4Pi fits from them, the bound the C++ fitter reports, the
phase as a factor of the mixture, and a training that runs.

The model is the closed-form astigmatic one (`fourpi.model_calibration`) given a layout: four channels
on a strip, two mirrored, each moved against channel 0 by a shift and, one, a stretch -- the remainders
change across the field, as a real registration's do.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smappy import fourpi
from smappy.channels import from_corners
from smappy.netfit import channels, extract, loss, network
from smappy.netfit import fourpi as NF
from smappy.netfit.render_4pi import IABRenderer
from smappy.netfit.settings import (
    CameraModel,
    ChannelSettings,
    Parts,
    SimulationSettings,
)
from smappy.psf import Global4PiPSF


@pytest.fixture(scope="module")
def cal():
    cal = fourpi.model_calibration()
    layout = from_corners([0, 70, 140, 210], [0] * 4, 64, 64, flip_x=[0, 0, 1, 1])
    maps = [np.array([[s, 0, dx], [0, s, dy], [0, 0, 1.0]])
            for dx, dy, s in [(0, 0, 1), (0.6, -0.4, 1.002), (-2.3, 1.2, 0.998), (3.4, 0.7, 1.0)]]
    cal.layout = layout.with_maps(maps).to_dict()
    cal.transforms = np.stack(maps)
    return cal


SIM = SimulationSettings(frame_px=32, z_nm=(-500, 500), background=(10, 50), photons_mean=3000, photons_sd=1000)
CAM = CameraModel(conversion=0.5, offset=100, em_gain=0, read_noise_e=1.5, pixel_nm=(100.0, 100.0))


def test_the_channels_are_put_on_channel_0s_grid_by_whole_pixels_and_a_remainder(cal):
    """A molecule drawn by `fourpi.render_frames` at p in channel 0 is, once the frames are cut onto the
    grid, at p + d_c in channel c: the spot's centroid there, to a few hundredths of a pixel."""
    four = NF.FourChannels.from_calibration(cal)
    assert four.shifts[0] == (0, 0) and four.n_channels == 4
    p = np.array([[30.3, 28.6]])
    frames = fourpi.render_frames(cal, four.layout, [0], p[:, 0], p[:, 1], [0.0], 1e6, [0.3], 1,
                                  background=0.0, seed=0)
    grid = four.cut(frames)[0]                                                # (C, rows, columns)
    r0, _, c0, _ = four.box
    d = four.remainder(p)[0]
    for c in range(4):
        at = p[0] + (d[2 * c - 2:2 * c] if c else 0) - [c0, r0]
        img = grid[c]
        rows, cols = np.mgrid[:img.shape[0], :img.shape[1]]
        near = np.hypot(cols - at[0], rows - at[1]) < 6
        w = img * near
        centroid = np.array([(w * cols).sum(), (w * rows).sum()]) / w.sum()
        assert np.abs(centroid - at).max() < 0.08


@pytest.fixture(scope="module")
def bright(cal):
    """A batch of bright, sparse molecules, and Spline 4Pi's fits of its middle frames at the targets."""
    four = NF.FourChannels.from_calibration(cal)
    sim = SimulationSettings(frame_px=32, z_nm=(-400, 400), background=(20, 20), photons_mean=20000,
                             photons_sd=0, density_per_40px=4.0, lifetime_frames=1.0)
    g = torch.Generator().manual_seed(1)
    r = IABRenderer(cal)
    batch = NF.draw_batch(r, channels.Grid(four), sim, CAM, ChannelSettings(kind="4pi"), 256, g)
    m = batch.mask.numpy()
    f, k = np.nonzero(m)
    xyz = batch.xyz.numpy()[f, k]
    d = batch.shift.numpy()[f, k]
    # alone in the middle frame within 8 px: a neighbour in the fitter's 13-px ROIs is its error, not the draw's
    others = [batch.xyz.numpy()[i][m[i]] for i in f]
    alone = np.array([np.sort(np.hypot(*(o[:, :2] - p[:2]).T))[1] > 8 if len(o) > 1 else True
                      for o, p in zip(others, xyz)])
    ok = (xyz[:, :2].min(1) > 8) & (xyz[:, :2].max(1) < 23) & (batch.photons.numpy()[f, k] > 12000) & alone
    f, xyz, d = f[ok], xyz[ok], d[ok]
    mid = batch.frames.numpy()[:, :, 1]                                       # (b, C, s, s)
    X = np.c_[xyz[:, 0], xyz[:, :1] + d[:, 0::2]]
    Y = np.c_[xyz[:, 1], xyz[:, 1:2] + d[:, 1::2]]
    ox, oy = np.rint(X).astype(int) - 6, np.rint(Y).astype(int) - 6
    rois = np.stack([np.stack([mid[f[i], c, oy[i, c]:oy[i, c] + 13, ox[i, c]:ox[i, c] + 13] for c in range(4)])
                     for i in range(len(f))]).astype(np.float32)
    xr, yr = X - ox, Y - oy
    link = fourpi.link(len(f), 4, dx=xr - xr[:, :1], dy=yr - yr[:, :1])
    model = Global4PiPSF(cal, phase_starts=2)
    v = model.unpack(model.fit(rois, link, iterations=100))
    return batch, ok, v, xr, yr, xyz


def test_a_batchs_targets_are_what_spline_4pi_fits_from_its_frames(bright):
    """The targets' x, y, z and phase -- the phase as the fitter reports it, phase_per_nm z plus the
    cavity's, wrapped -- are what the fitter finds in the drawn frames: unbiased, their errors the size
    of its CRLBs."""
    batch, ok, v, xr, _, xyz = bright
    m = batch.mask.numpy()
    phase = batch.phase.numpy()[m][ok]
    assert len(phase) > 25
    for err in (np.angle(np.exp(1j * (v["phase_rad"] - phase))) / v["phase_err_rad"],
                (v["x_roi"] - xr[:, 0]) / v["x_err_pix"], (v["z_nm"] - xyz[:, 2]) / v["z_err_nm"]):
        assert abs(np.mean(err)) < 0.6 and np.median(np.abs(err)) < 1.0      # a Gaussian's: 0, 0.67


def test_the_bound_is_the_one_the_cpp_fitter_reports(cal):
    """`netfit.fourpi.crlb`, from torch derivatives, against the CRLB Spline 4Pi's C++ returns at the
    truth (noise-free spots, so the fit sits on it): x, z and the phase to 0.5 %."""
    r = IABRenderer(cal, dtype=torch.float64)
    rng = np.random.default_rng(0)
    m = 100
    c0 = (r.size - 1) / 2
    x, y = c0 + rng.uniform(-1, 1, m), c0 + rng.uniform(-1, 1, m)
    sh = rng.uniform(-0.4, 0.4, (m, 6))
    z, ph = rng.uniform(-500, 500, m), rng.uniform(-np.pi, np.pi, m)
    N, bg = rng.uniform(500, 5000, m), rng.uniform(5, 50, (m, 4))
    bound = NF.crlb(cal, x, y, sh, z, ph, N, bg)
    X, Y = np.c_[x, x[:, None] + sh[:, 0::2]], np.c_[y, y[:, None] + sh[:, 1::2]]
    sp, ox, oy = r.spots(*(torch.as_tensor(a) for a in (X, Y, z, ph)), clamp=False)
    rois = (N[:, None, None, None] * sp.numpy() + bg[:, :, None, None]).astype(np.float32)
    xr, yr = X - ox.numpy(), Y - oy.numpy()
    model = Global4PiPSF(cal, phase_starts=2)
    v = model.unpack(model.fit(rois, fourpi.link(m, 4, dx=xr - xr[:, :1], dy=yr - yr[:, :1]), iterations=200))
    for key, k in (("x_err_pix", 0), ("y_err_pix", 1), ("z_err_nm", 2), ("phase_err_rad", 3)):
        assert np.abs(v[key] / bound[:, k] - 1).max() < 0.005


def _network(cal, seed=0):
    torch.manual_seed(seed)
    ch = ChannelSettings(kind="4pi")
    net = network.build(Parts(), SIM.window, ch)
    return net, network.Scaling.for_simulation(SIM, ch), ch


def test_a_4pi_network_reads_four_channels_and_six_maps_and_puts_out_a_phase(cal):
    """Its output has three more channels a slot -- the phase's cos, sin (a unit vector) and sigma --
    and four backgrounds; the extraction reports the phase as an angle with its error."""
    net, sc, ch = _network(cal)
    four = NF.FourChannels.from_calibration(cal)
    b = NF.draw_batch(IABRenderer(cal), channels.Grid(four), SIM, CAM, ch, 4, torch.Generator().manual_seed(0))
    assert b.frames.shape == (4, 4, 3, 32, 32) and b.aux.shape == (4, 6, 32, 32)
    out = net((b.frames - sc.input_offset) / sc.input_scale, 1, b.aux)
    lay = net.layout
    assert lay.phase and lay.backgrounds == 4 and out.shape[1] == lay.channels
    cs = out[:, lay.ph():lay.ph() + 2]
    assert torch.allclose(torch.linalg.vector_norm(cs, dim=1), torch.ones(1), atol=1e-5)
    e = extract.extract(out.detach(), sc, 0.0, "mixture_peak", layout=lay)
    assert len(e["x"]) and torch.all(e["phase_err_rad"] > 0) and torch.all(e["phase_rad"].abs() <= np.pi)


def test_the_mixture_rewards_the_right_phase_and_its_concentration(cal):
    """One molecule, every component pointing at its phase or at the opposite one: the loss is lower
    for the right phase, and lower still when the components are sure of it."""
    net, sc, ch = _network(cal)
    four = NF.FourChannels.from_calibration(cal)
    b = NF.draw_batch(IABRenderer(cal), channels.Grid(four), SIM, CAM, ch, 2, torch.Generator().manual_seed(3))
    with torch.no_grad():
        out = net((b.frames - sc.input_offset) / sc.input_scale, 1, b.aux)
    lay = net.layout
    t = b.phase[:, :1]

    def mixture(angle, sigma):
        o = out.clone()
        o[:, lay.ph()] = torch.cos(angle)[:, :, None]
        o[:, lay.ph() + 1] = torch.sin(angle)[:, :, None]
        o[:, lay.ph() + 2] = sigma
        one = type(b)(b.frames, b.xyz[:, :1], b.photons[:, :1], b.mask[:, :1], b.background, aux=b.aux,
                      phase=b.phase[:, :1])
        return loss.terms(o, one, sc, names=("mixture",), layout=lay)["mixture"].sum()

    right, wrong = mixture(t, 0.5), mixture(t + np.pi, 0.5)
    assert right < wrong
    assert mixture(t, 0.1) < right


@pytest.mark.slow
def test_a_4pi_training_runs_and_scores_the_phase(cal, tmp_path):
    """Two short epochs on the CPU: the progress lines carry the phase's RMSE and its width over the
    bound beside x, y and z, and the model file the grid it was trained on."""
    import json

    from smappy.netfit import train
    from smappy.netfit.settings import TrainingSettings
    path = tmp_path / "model_4pi.h5"
    fourpi.save_fourpi_calibration(path, cal)
    parts = Parts()
    parts.backbone.settings["features"] = 8
    parts.schedule.settings.update(batch=8, samples_per_epoch=16, test_samples=16, threshold=0.0, max_samples=None)
    parts.loss.settings["reported"] = []
    s = TrainingSettings(simulation=SIM, camera=CAM, parts=parts, channels=ChannelSettings(kind="4pi"),
                         calibration=str(path))
    assert train.train(tmp_path / "run", s, "cpu", epochs=2, log=lambda *_: None, early_stop=False) == "finished"
    lines = [json.loads(v) for v in (tmp_path / "run" / "progress.jsonl").read_text().splitlines()]
    scored = [v for v in lines if v.get("kind") == "epoch"]
    assert len(scored) == 2
    assert "rmse_phase_rad" in scored[-1]["metrics"] and "crlb_width_phase" in scored[-1]["metrics"]


def test_z_from_the_phase_takes_the_fringe_nearest_the_envelope_and_counts_the_wrong_ones():
    """With the cavity's phase known, z from the phase is the truth when the envelope's z is within half a
    fringe of it, and a whole fringe off when it is beyond: the wrong-fringe share counts exactly those."""
    k = -2 * np.pi / 260.0
    rng = np.random.default_rng(0)
    z = rng.uniform(-500, 500, 400)
    phase0 = rng.uniform(-np.pi, np.pi, 400)
    phase = np.angle(np.exp(1j * (k * z + phase0)))
    off = np.where(np.arange(400) < 100, 200.0, 40.0)          # a quarter of the envelopes beyond half a fringe
    zp = NF.unwrap_known(phase, z + off, phase0, k)
    assert np.allclose(zp[100:], z[100:], atol=1e-6)
    assert np.allclose(np.abs(zp[:100] - z[:100]), 260.0, atol=1e-6)
    truth = {"frame": np.arange(400), "xyz": np.c_[np.full(400, 5.0), np.full(400, 5.0), z],
             "photons": np.full(400, 1000.0), "phase": phase}
    found = {"frame": np.arange(400), "xyz": np.c_[np.full(400, 5.0), np.full(400, 5.0), z + off], "phase": phase}
    s = NF.phase_scores(truth, found, (100.0, 100.0), k)
    assert s["wrong_fringe_share"] == pytest.approx(0.25) and s["rmse_z_phase_nm"] < 1e-6


def test_the_prefit_reads_the_cavitys_drift_and_each_channels_background_from_a_spline_4pi_table():
    """A table as *4Pi z* leaves it: the cavity drifting 0.004 rad a frame with a walk, backgrounds of
    their own per channel.  The prefit gives back the rate, ranges covering each channel's background,
    and a training's settings with them."""
    rng = np.random.default_rng(0)
    n_frames, per = 20000, 5
    frame = np.repeat(np.arange(n_frames), per)
    cav = fourpi.cavity_drift(n_frames, rate_rad=0.004, wander_rad=0.003, start_rad=0.5, seed=1)
    cols = {"frame": frame, "photons": rng.lognormal(np.log(1500), 0.5, len(frame)),
            "z_env_nm": rng.uniform(-500, 500, len(frame)),
            "cavity_phase_rad": np.angle(np.exp(1j * (cav[frame] + 0.05 * rng.normal(size=len(frame)))))}
    for c, level in enumerate((10, 12, 9, 11)):
        cols[f"background_ch{c}"] = rng.normal(level, 1.0, len(frame))
    pre = NF.prefit(cols)
    assert pre["rate_measured"] == pytest.approx(0.004, rel=0.05)
    for c, level in enumerate((10, 12, 9, 11)):
        lo, hi = pre["background"][c]
        assert lo < level - 1.5 and hi > level + 1.5
    assert pre["z_nm"][1] == pytest.approx(495, abs=10)
    s = NF.settings_from_prefit(pre, CAM, "model.h5")
    assert s.channels.kind == "4pi" and s.channels.cavity_rate_rad == pytest.approx(pre["rate_measured"], rel=1e-6)
    lo, hi = s.simulation.photons_range
    assert lo < 500 and hi > 5000
