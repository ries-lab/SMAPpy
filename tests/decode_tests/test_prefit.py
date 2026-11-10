"""The training's ranges from a fit of the data (``decode_smlm.prefit``): milestone (e).

The estimates are SMAP's rules, so most tests check a rule on numbers whose answer is known; the
one that matters is the first, a simulation with a known on-time, brightness, background and z,
whose values must come back within 10 %.
"""
import subprocess
import sys

import numpy as np
import pytest

from decode_smlm import prefit


def _simulated(on_time, seed=2):
    from smappy.group import GroupSettings, group
    from smappy.simulate import simulate
    from smappy.simulate.localizations import settings_with

    s = settings_with(n_frames=12000, seed=seed, background=60.0, background_std=15.0)
    s.structure.density_um2 = 0.03          # sparse: a blink is one molecule's
    s.structure.z_range_nm = 1200.0
    s.blinking.on_time, s.blinking.photons, s.blinking.photons_std = on_time, 4000.0, 1000.0
    s.localizations.close = "average"
    locs = simulate(s)
    # a box that holds a blink's dim first and last frames (the prefit's grouping)
    blinks, _ = group(locs, GroupSettings(dx=150.0))
    return s, locs, blinks


@pytest.mark.parametrize("on_time", [1.5, 3.0])
def test_a_simulations_lifetime_brightness_background_and_z_come_back(on_time):
    """Within 10 % (SMAP's cut at the 95th percentile shortens the lifetime at 3 frames, 2.70)."""
    s, locs, blinks = _simulated(on_time)
    cols = {k: np.asarray(blinks[k]) for k in ("photons", "n_in_group", "frame", "z_nm")}
    loc_cols = {k: np.asarray(locs[k]) for k in ("photons", "background", "group_id", "frame", "n_in_group", "z_nm")}
    background = loc_cols["background"]
    est = prefit.estimates(cols, loc_cols, area_px=1e4, n_frames=12000, z_limit_nm=2000.0)
    assert est.lifetime_frames == pytest.approx(on_time, rel=0.1)
    # photons a frame while on: the fully-on frames, whatever the switching frames do
    assert est.intensity_from == "fully-on frames" and est.n_fully_on > 1000
    rate, spread = s.blinking.photons / on_time, s.blinking.photons_std / on_time
    assert est.intensity_mu_sig[0] == pytest.approx(rate, rel=0.05)
    # the molecules' own spread, and the shot noise on top
    assert spread <= est.intensity_mu_sig[1] <= 1.3 * spread
    lo, hi = np.quantile(background, [0.01, 0.98])
    assert est.background[0] <= lo and est.background[1] >= hi
    assert est.background[1] - est.background[0] == pytest.approx(1.6 * (hi - lo), rel=0.1)
    z = np.asarray(blinks["z_nm"])
    zlo, zhi = np.quantile(z, [0.01, 0.99])
    assert est.z_nm == pytest.approx((zlo - 0.2 * (zhi - zlo), zhi + 0.2 * (zhi - zlo)), rel=0.05)


def test_the_on_time_is_the_mean_up_to_smaps_cut():
    """1 to K + 1 frames, K the 95th percentile and at least 13: the long tail is left out."""
    n = np.r_[np.ones(500), np.full(300, 2), np.full(150, 5), np.full(45, 12), np.full(5, 40)]
    assert prefit.on_time(n) == pytest.approx(n[n <= 14].mean())
    long = np.r_[np.full(900, 20.0), np.full(100, 30.0)]          # 95th percentile 30: kept
    assert prefit.on_time(long) == pytest.approx(21.0)


def test_lifetime_is_the_on_time_less_one_and_at_least_one():
    assert prefit.lifetime(3.2) == pytest.approx(2.2)
    assert prefit.lifetime(1.4) == 1.0


def test_the_fully_on_frames_are_the_middle_of_blinks_of_three_frames_or_more():
    # blink 1: frames 0-3, blink 2: frames 5-6, blink 3: 9-11 with a poor fit in its middle
    gid = np.array([1, 1, 1, 1, 2, 2, 3, 3, 3])
    frame = np.array([0, 1, 2, 3, 5, 6, 9, 10, 11])
    n = np.array([4, 4, 4, 4, 2, 2, 3, 3, 3])
    logl = np.array([-1, -2, -1.5, -1, -2, -1, -1.5, -30, -1.0])     # 1 MAD below the median: -2.24
    assert list(prefit.fully_on(gid, frame, n)) == [0, 1, 1, 0, 0, 0, 0, 1, 0]
    assert list(prefit.fully_on(gid, frame, n, logl_rel=logl)) == [0, 1, 1, 0, 0, 0, 0, 0, 0]
    z = np.array([0, 0, 950, 0, 0, 0, 0, 0, 0.0])
    assert list(prefit.fully_on(gid, frame, n, z_nm=z, z_limit_nm=1000)) == [0, 1, 0, 0, 0, 0, 0, 1, 0]


def test_with_few_fully_on_frames_the_brightness_is_smaps_rule():
    blinks = {"photons": np.full(50, 2000.0), "n_in_group": np.full(50, 2), "frame": np.arange(50)}
    locs = {"photons": np.full(100, 1000.0), "background": np.full(100, 30.0), "group_id": np.repeat(np.arange(50), 2),
            "frame": np.arange(100), "n_in_group": np.full(100, 2)}
    est = prefit.estimates(blinks, locs, area_px=1600.0, n_frames=100)
    assert est.intensity_from == "blinks" and est.intensity_mu_sig == pytest.approx((2000.0, 400.0))


def test_a_short_on_time_makes_the_molecules_brighter():
    assert prefit.intensity(np.full(10, 1000.0), 0.7)[0] == pytest.approx(1300.0)
    assert prefit.intensity(np.full(10, 1000.0), 3.0)[0] == pytest.approx(500.0)


def test_the_background_never_goes_below_nine_tenths_of_its_lowest_values():
    bg = np.linspace(2.0, 100.0, 10000)          # widened by 30 %, the range would go below 0
    lo, hi = prefit.background_range(bg)
    assert lo == pytest.approx(0.9 * np.quantile(bg, 0.0005, method="hazen"))
    q = np.quantile(bg, [0.01, 0.98], method="hazen")
    assert hi == pytest.approx(q[1] + 0.3 * (q[1] - q[0]))


def test_z_is_clamped_to_the_calibration_and_without_z_is_750():
    assert prefit.z_range(np.linspace(-900, 900, 1001), 800.0) == (-800.0, 800.0)
    assert prefit.z_range(None) == (-750.0, 750.0)


def test_the_calibrations_z_limit_is_smaps_fminmax_or_the_splines_grid():
    from types import SimpleNamespace
    cal = SimpleNamespace(parameters={"fminmax": np.array([1, 200]), "dz": 10.0}, dz=10.0, z0=100.0,
                          coeff=np.zeros((64, 197, 2, 2)))
    assert prefit.z_limit(cal) == pytest.approx(995.0)
    cal.parameters = {}
    assert prefit.z_limit(cal) == pytest.approx(960.0)          # 96 voxels above the focus, 100 below


def test_the_camera_is_decodes_with_smaps_read_noise_and_pixel_order():
    em = prefit.camera(5.0, 398.6, (117.0, 127.0), em_gain=50.0)
    assert em["px_size"] == [127.0, 117.0]                       # DECODE's rows first: SMAPpy's y
    assert em["read_sigma"] == 74.4 and em["spur_noise"] == 0.0015 and em["qe"] == 1.0
    scmos = prefit.camera(0.5, 100.0, 108.0, read_noise_e=1.2)
    assert scmos["em_gain"] is None and scmos["read_sigma"] == 1.2 and scmos["spur_noise"] == 0.0
    assert prefit.camera(0.5, 100.0, 108.0)["read_sigma"] == 1.5


def test_the_density_is_chosen_and_the_measured_one_shown():
    est = _est()
    assert prefit.simulation(est, "UHD")["emitter_av"] == 50
    with pytest.raises(ValueError):
        prefit.simulation(est, "medium")
    # 1600 blinks over 100 frames on 1600 pixels: one a pixel per 100 frames, 16 in a 40 x 40 frame
    assert prefit.measured_density(1600, 1600.0, 100) == pytest.approx(16.0)


def test_the_estimates_load_without_torch():
    code = "import sys, decode_smlm.prefit; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def _est():
    return prefit.Estimates(on_time_frames=2.0, lifetime_frames=1.0, intensity_mu_sig=(3000.0, 600.0),
                            intensity_from="fully-on frames", n_fully_on=500, background=(20.0, 80.0),
                            z_nm=(-600.0, 600.0), emitters_per_frame=7.3, n_blinks=100, n_localizations=300)


def test_a_jobs_method_yaml_loads_as_decodes_parameters():
    from decode_smlm.utils import param_io
    from decode_smlm.utils.types import RecursiveNamespace
    params = prefit.method_params(_est(), prefit.camera(0.5, 100.0, (108.0, 110.0)), "HD")
    full = RecursiveNamespace(**param_io.autofill_dict(params, param_io.load_reference()))
    assert full.Simulation.emitter_av == 25 and full.Simulation.lifetime_avg == 1.0
    assert full.Simulation.emitter_extent[2] == [-600, 600]
    assert full.Camera.px_size == [110.0, 108.0] and full.Camera.em_gain is None
    assert full.InOut.calibration_file == "calibration.h5"
    assert full.HyperParameter.architecture == "SigmaMUNet"          # the rest from reference.yaml


def test_the_simulated_frames_are_in_photons_with_the_drawn_background(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from decode_smlm import figures
    from smappy.io.calibration import SplineCalibration

    # a Gaussian spot as the spline's constant term: a PSF of one photon in all
    g = np.exp(-0.5 * ((np.arange(13) - 6.0) / 1.5) ** 2)
    coeff = np.zeros((64, 11, 13, 13), np.float32)
    coeff[0] = (np.outer(g, g) / np.outer(g, g).sum())[None]
    cal = SplineCalibration(coeff=coeff, dz=100.0, z0=5.0, x0=7.0, em_on=False)
    params = prefit.method_params(_est(), prefit.camera(0.5, 100.0, 108.0), "LD")
    frames = figures.simulate_frames(params, cal, n_frames=6)
    assert frames.shape == (6, 40, 40)
    # most pixels are background, drawn uniformly from 20 to 80 photons a frame
    assert 20 <= np.median(frames) <= 80
    fig, axes = plt.subplots(2, 3)
    figures.draw_frames(axes[0], frames[:1] * 0.9, frames[1:2])
    figures.draw_ranges(axes[1], {"photons": np.full(50, 3000.0), "background": np.linspace(20, 80, 50),
                                  "z_nm": np.linspace(-500, 500, 50)}, params)
    plt.close(fig)


def test_counts_become_photons_as_decode_converts_them():
    from decode_smlm import figures
    cam = prefit.camera(5.0, 400.0, 117.0, em_gain=50.0)
    assert figures.photons(np.array([400.0, 1400.0]), cam) == pytest.approx([-0.0015, 99.9985])
