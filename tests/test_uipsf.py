"""uiPSF as a calibration: its model comes back in smappy's conventions.

Most of this needs no uiPSF: a uiPSF result is an HDF5 file whose layout and
conventions are written here by hand, so what is tested is the conversion
and the geometry -- that z keeps its sign and its zero, that a split camera's
channels go back on the chip, that uiPSF's transformation becomes smappy's.
The last test runs uiPSF itself, when ``SMAPPY_UIPSF_PYTHON`` names a Python
that has it (slow tier: minutes on a CPU).
"""
import json
import os
import sys
import textwrap

import h5py
import numpy as np
import pytest
from scipy.special import erf

from smappy.calibrate.dual import map_points
from smappy.io.calibration import load_spline_calibration
from smappy.simulate import ASTIGMATISM, astigmatic_sigmas
from smappy.uipsf import convert, microscopes, prepare, runner, uipsf_parameters

PIXEL_NM, DZ_NM, SIZE = 100.0, 50.0, 21


def spot(z_nm, x=0.0, y=0.0, size=SIZE, photons=1.0):
    """The simulator's astigmatic spot at emitter height ``z_nm``, pixel-integrated."""
    sx, sy = (w / PIXEL_NM * np.sqrt(2.0) for w in astigmatic_sigmas(z_nm, 130.0, *ASTIGMATISM))
    pix, c = np.arange(size), (size - 1) / 2
    px = 0.5 * (erf((pix + 0.5 - c - x) / sx) - erf((pix - 0.5 - c - x) / sx))
    py = 0.5 * (erf((pix + 0.5 - c - y) / sy) - erf((pix - 0.5 - c - y) / sy))
    return photons * np.outer(py, px)


def params(psftype="zernike_vector", stage_pos=1.0, med=1.335, imm=1.516):
    return {"PSFtype": psftype, "pixel_size": {"x": 0.1, "y": 0.1, "z": DZ_NM / 1000},
            "option": {"imaging": {"emission_wavelength": 0.68, "NA": 1.43,
                                   "RI": {"imm": imm, "med": med, "cov": 1.516}},
                       "insitu": {"stage_pos": stage_pos},
                       "model": {"zernike_nl": []}}}


def write_result(path, res, p, rois=None, locres=None):
    """A file laid out as uiPSF's `writeh5file` lays one out."""
    def put(group, items):
        for key, value in items.items():
            if isinstance(value, dict):
                put(group.create_group(key), value)
            else:
                group[key] = value
    with h5py.File(path, "w") as f:
        f.attrs["params"] = json.dumps(p)
        put(f.create_group("res"), res)
        put(f.create_group("locres"), locres or {"coeff": np.zeros(1)})
        if rois is not None:
            put(f.create_group("rois"), rois)
    return path


def bead_model(nz=33):
    """uiPSF's bead model: planes are the objective's steps, focus in the middle.

    Raised by dz, the objective leaves a bead on the coverslip dz *below* its
    focus, as `simulate.bead_stacks` draws it.
    """
    objective = (np.arange(nz) - (nz - 1) / 2) * DZ_NM
    return np.stack([spot(-z) for z in objective])


def fitted_z(calibration, z_true):
    from smappy.psf import SplinePSF
    rois = np.stack([spot(z, 0.3, -0.2, size=13, photons=5000) + 10 for z in z_true])
    fit = SplinePSF(calibration).fit(rois.astype(np.float32), iterations=100)
    return np.array([calibration.z_index_to_nm(t) for t in fit.theta[:, 4]]), fit.theta


def test_a_bead_model_keeps_its_sign_and_its_zero(tmp_path):
    zernike = np.zeros((2, 45))
    zernike[1, 5] = 0.5                     # Noll 6, astigmatism 0, in radians
    path = write_result(tmp_path / "beads_zernike_vector_single.h5",
                        {"I_model": bead_model(), "zernike_coeff": zernike}, params())
    calibration = convert.single_calibration(path)
    assert calibration.z0 == 16 and calibration.dz == DZ_NM
    assert calibration.psf.sum(axis=(1, 2)).max() == pytest.approx(1.0)
    z_true = np.array([-400.0, -150.0, 0.0, 200.0, 450.0])
    z, theta = fitted_z(calibration, z_true)
    np.testing.assert_allclose(z, z_true, atol=8)
    np.testing.assert_allclose(theta[:, :2] - theta[0, :2], 0, atol=0.02)
    # the aberration, read out in nm of wavefront: 0.5 rad of 680 nm
    row = next(r for r in calibration.parameters["zernike"] if r[0] == 6)
    assert row[1] == "astigmatism 0" and row[3] == pytest.approx(0.5 * 680 / (2 * np.pi))


def test_an_in_situ_model_is_turned_over_and_zeroed_at_the_focal_plane(tmp_path):
    """In situ planes are emitter heights above the coverslip, the other way up.

    uiPSF's planes run ``zoffset + i`` steps above the coverslip, the focus
    sitting at ``stagepos`` um of stage travel, ``stagepos * n_med / n_imm``
    into the sample -- that plane is z = 0.
    """
    nz, stagepos, zoffset, med, imm = 25, 1.137, 13.4, 1.335, 1.516
    focus_um = stagepos * med / imm
    heights_um = (zoffset + np.arange(nz)) * DZ_NM / 1000
    model = np.stack([spot((h - focus_um) * 1000) for h in heights_um])
    path = write_result(tmp_path / "psfmodel0_insitu_zernike_single.h5",
                        {"I_model": model, "stagepos": stagepos, "zoffset": zoffset},
                        params("insitu_zernike", stagepos, med, imm))
    calibration = convert.single_calibration(path)
    z_true = np.array([-300.0, -100.0, 0.0, 150.0, 300.0])
    z, _ = fitted_z(calibration, z_true)
    np.testing.assert_allclose(z, z_true, atol=8)
    assert calibration.parameters["uipsf_stagepos"] == [stagepos]


@pytest.mark.parametrize("layout,main", [("up-down", "upper"), ("up-down mirrored", "lower"),
                                         ("right-left mirrored", "left"),
                                         ("right-left", "right")])
def test_a_split_frame_goes_back_on_the_chip(layout, main):
    shape, origin = (64, 80), (300, 120)
    split = prepare.make_split(shape, {"layout": layout, "main_channel": main,
                                       "split_position": 34 if "up" in layout else 38},
                               origin=origin)
    frame = np.zeros(shape)
    marks = {(5, 7): 1.0, (50, 70): 2.0, (40, 3): 3.0, (20, 60): 4.0}
    for (y, x), value in marks.items():
        frame[y, x] = value
    channels = split.cut(frame[None])[:, 0]
    assert channels.shape[0] == 2 and channels[0].shape == channels[1].shape
    for channel in (0, 1):
        for y, x in zip(*np.nonzero(channels[channel])):
            chip_x, chip_y = split.to_chip(channel, [[y, x]])[0]
            assert frame[int(chip_y) - origin[1], int(chip_x) - origin[0]] == \
                channels[channel][y, x]


def test_uipsfs_channel_transformation_becomes_smappys():
    """uiPSF's ``T``, made from a known chip map, gives that map back.

    uiPSF maps the main channel's (y, x, 1) rows about ``imgcenter`` into
    the secondary channel's array, which a mirrored splitter has flipped.
    """
    split = prepare.make_split((128, 100), {"layout": "up-down mirrored",
                                            "main_channel": "upper"}, origin=(16, 8))
    truth = np.array([[1.003, 0.004, 1.7], [-0.006, 0.998, 130.2], [0.0, 0.0, 1.0]])
    truth = np.linalg.inv(truth)                         # secondary -> main
    rng = np.random.default_rng(3)
    main_yx = rng.uniform(5, 58, (40, 2))
    secondary_chip = map_points(np.linalg.inv(truth), split.to_chip(0, main_yx))
    # back into the secondary channel's array: unshift, unflip, (y, x)
    local = secondary_chip - np.asarray(split.origin, float)
    secondary_yx = np.c_[split.length - 1 - (local[:, 1] - split.starts()[1]), local[:, 0]]
    centre = np.array([split.length / 2, 50.0, 0.0])
    rows = np.c_[main_yx, np.ones(len(main_yx))] - centre
    target = np.c_[secondary_yx - centre[:2], np.ones(len(main_yx))]
    T, *_ = np.linalg.lstsq(rows, target, rcond=None)
    found = convert.channel_transformation(T, centre, split)
    points = np.array([[20.0, 70.0], [90.0, 120.0], [50.0, 100.0]])
    np.testing.assert_allclose(map_points(found, points), map_points(truth, points), atol=1e-6)


def test_the_channel_shift_is_found_on_a_bead_grid_that_fools_nearest_neighbours():
    """uiPSF's own guess, from bead coordinates, took the neighbouring bead of
    a 30 px grid for the partner and kept one pair of nine."""
    from scipy import ndimage
    image = np.zeros((2, 3, 9, 100, 100))
    for gy in (20, 50, 80):
        for gx in (20, 50, 80):
            image[0, :, :, gy, gx] = 1000.0
    image[1] = ndimage.shift(image[0], (0, 0, -1.0, 2.0), order=1)
    image += np.random.default_rng(1).poisson(5.0, image.shape)
    assert prepare.channel_shift(image) == [[0.0, 0.0], [-1.0, 2.0]]


def test_two_channels_share_one_normalisation_and_the_secondary_is_unmirrored(tmp_path):
    split = prepare.make_split((128, 100), {"layout": "up-down mirrored",
                                            "main_channel": "upper"})
    model = bead_model()
    lopsided = model * (1 + 0.5 * np.linspace(0, 1, SIZE))[None, :, None]   # not y-symmetric
    T = np.eye(3)
    T[2, :2] = [0.5, -0.3]
    res = {"channel0": {"I_model": model, "intensity": np.full(9, 3000.0)},
           "channel1": {"I_model": lopsided, "intensity": np.full(9, 1000.0)},
           "T": T, "imgcenter": np.array([32.0, 50.0, 0.0]), "xyshift": np.zeros(2)}
    path = write_result(tmp_path / "psfmodel_zernike_vector_multi.h5", res, params())
    calibration = convert.dual_calibration(path, split)
    main, secondary = calibration.main, calibration.secondary
    window = slice(14, 19)
    light = [m.psf[window].sum(axis=(1, 2)).mean() for m in (main, secondary)]
    assert sum(light) == pytest.approx(1.0, rel=1e-3)
    assert main.parameters["photon_normalization"] == pytest.approx(light[0], rel=1e-3)
    assert light[0] / light[1] == pytest.approx(3000 / (1000 * lopsided[window].sum()
                                                        / model[window].sum()), rel=1e-3)
    # flipped back along y, the mirror axis, to the camera's orientation
    np.testing.assert_allclose(secondary.psf / secondary.psf.max(),
                               (lopsided[:, ::-1, :] / lopsided.max()).astype(np.float32),
                               atol=1e-6)
    path_out = convert.save(tmp_path / "dual.h5", calibration)
    from smappy.calibrate.dual import load_dual_color_calibration
    again = load_dual_color_calibration(path_out)
    np.testing.assert_allclose(again.transformation, calibration.transformation)


def bead_result(tmp_path, dual=False):
    """A complete uiPSF bead result, one or two channels, as the plots read it.

    The second channel sits 2 px right and 1 px down of the first, which is
    what ``T`` says: the transformation's residuals are zero by construction.
    The localization bias is a known 3 nm in x and one plane in z.
    """
    rng = np.random.default_rng(5)
    model = bead_model()
    nz, n = len(model), 6
    zernike = np.zeros((2, 45))
    zernike[0, 0], zernike[1, 5] = 1.0, 0.5
    yy, xx = np.mgrid[:64, :64]
    pupil = ((yy - 31.5) ** 2 + (xx - 31.5) ** 2 < 30 ** 2).astype(np.complex64)
    cor = np.array([[20, 20], [20, 60], [60, 20], [60, 60], [40, 40], [30, 70]])
    pos = np.c_[np.full(n, nz / 2), cor + rng.uniform(-0.3, 0.3, (n, 2))]

    def channel(shift):
        return {"I_model": model, "intensity": np.full((n, nz), 2000.0),
                "zernike_coeff": zernike, "zernike_polynomial": np.ones((45, 64, 64)),
                "pupil": pupil, "pos": pos + [0, *shift], "cor": cor + shift,
                "cor_all": np.r_[cor + shift, [[5, 5]]]}
    stacks = np.stack([model * 2000 + rng.normal(0, 1, model.shape) for _ in range(n)])
    loc = {"x": np.full((n, nz), 0.03), "y": np.zeros((n, nz)),
           "z": np.tile(np.arange(nz) + 1.0, (n, 1))}
    if not dual:
        return write_result(tmp_path / "single.h5", channel(np.zeros(2)), params(),
                            {"psf_data": stacks, "psf_fit": stacks * 0.98,
                             "cor": cor, "image_size": [1, nz, 80, 80]},
                            {"loc": loc})
    T = np.eye(3)
    T[2, :2] = [1.0, 2.0]                   # rows (y, x, 1) - c: + (1, 2)
    return write_result(tmp_path / "dual.h5",
                        {"channel0": channel(np.zeros(2)), "channel1": channel(np.array([1, 2])),
                         "T": T, "imgcenter": np.array([40.0, 40.0, 0.0])},
                        params(), {"psf_data": np.stack([stacks, stacks]),
                                   "psf_fit": np.stack([stacks, stacks]),
                                   "cor": np.stack([cor, cor + [1, 2]]),
                                   "image_size": [2, 1, nz, 80, 80]}, {"loc": loc})


@pytest.mark.parametrize("dual", [False, True])
def test_every_tab_of_a_result_draws_into_a_figure_and_into_a_page(tmp_path, dual):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from smappy.uipsf import plots
    data = convert.read(bead_result(tmp_path, dual))
    tabs = plots.figures(data)
    from smappy.plugins import Plot
    expected = ["data vs model", "localization bias", "pupil", "Zernike", "beads"]
    assert list(tabs) == expected + (["channel transformation"] if dual else [])
    for draw, panels, size in tabs.values():
        plot = Plot(draw=draw, panels=panels, size=size)
        figure = Figure(figsize=size, layout="constrained")
        plot.draw_into(figure)
        figure.savefig(tmp_path / "tab.png")
        page = Figure().subfigures(1, 2)[1]     # the All page hands out subfigures
        plot.draw_into(page)
        assert page.axes


def test_the_bias_and_the_transformation_residuals_come_out_in_nanometres(tmp_path):
    from smappy.uipsf import plots
    data = convert.read(bead_result(tmp_path, dual=True))
    x, y, z = plots.localization_bias(data)
    assert np.allclose(x, 3.0) and np.allclose(y, 0.0) and np.allclose(z, DZ_NM)
    main, residual = plots.transformation_residuals(data)
    np.testing.assert_allclose(residual, 0, atol=1e-6)
    # 0.5 rad of Noll 6 at 680 nm, in both channels
    for label, noll, nm in plots.zernike_phases(data):
        assert nm[list(noll).index(6)] == pytest.approx(0.5 * 680 / (2 * np.pi))


def test_in_situ_data_are_averaged_per_plane_of_the_model(tmp_path):
    """Molecules fall into the model's planes by their fitted height, as uiPSF bins them."""
    from smappy.uipsf import plots
    nz, zoffset = 5, 10.0
    model = np.ones((nz, 7, 7))
    z = np.array([10.2, 10.7, 12.5, 14.9])          # planes 0, 0, 2, 4
    rois = np.stack([np.full((7, 7), v) for v in (1.0, 3.0, 5.0, 7.0)])
    path = write_result(tmp_path / "insitu.h5",
                        {"I_model": model, "pos": np.c_[z, np.zeros((4, 2))],
                         "zoffset": np.array([[zoffset]]), "stagepos": np.array([1.0]),
                         "cor": np.zeros((4, 2)), "cor_all": np.zeros((9, 2)),
                         "pupil": np.ones((8, 8), np.complex64)},
                        params("insitu_zernike"),
                        {"psf_data": rois, "psf_fit": rois, "image_size": [100, 64, 64]},
                        {"loc": {"x": np.zeros((4, 1)), "y": np.zeros((4, 1)),
                                 "z": np.zeros((4, 1))}})
    data = convert.read(path)
    (label, measured, modelled, what), = plots.data_and_model(data)
    np.testing.assert_allclose(measured[:, 0, 0], [2.0, 0.0, 5.0, 0.0, 7.0])
    assert list(plots.figures(data)) == ["data vs model", "pupil", "Zernike", "emitters"]


def test_a_profile_says_what_it_knows_and_refuses_what_it_does_not(tmp_path, monkeypatch):
    shipped = microscopes.load("example")
    assert shipped.dual["layout"] == "up-down mirrored" and shipped.NA == 1.43
    (tmp_path / "example.yaml").write_text("name: ours\nNA: 1.5\n")
    (tmp_path / "typo.yaml").write_text("NA: 1.4\nrefractve_index: {}\n")
    monkeypatch.setenv("SMAPPY_MICROSCOPES", str(tmp_path))
    assert microscopes.load("example").name == "ours"            # the group's wins
    with pytest.raises(ValueError, match="unknown keys"):
        microscopes.load("typo")
    assert ("typo", "typo (unreadable)") in microscopes.choices()


def test_the_profile_becomes_uipsfs_parameters_and_its_own_section_has_the_last_word():
    profile = microscopes.Microscope(NA=1.35, emission_wavelength_nm=600,
                                     uipsf={"option": {"model": {"n_max": 6}},
                                            "batch_size": 400})
    p = uipsf_parameters(profile, data="blinking", pixelsize_xy_um=(0.11, 0.12), dz_nm=40,
                         depth_um=2.5, zernike_order=8)
    assert p["pixel_size"] == {"x": 0.11, "y": 0.12, "z": 0.04}
    assert p["option"]["imaging"]["NA"] == 1.35
    assert p["option"]["imaging"]["emission_wavelength"] == pytest.approx(0.6)
    assert p["option"]["insitu"]["stage_pos"] == 2.5
    assert p["option"]["model"]["n_max"] == 6 and p["batch_size"] == 400
    assert "insitu" not in uipsf_parameters(profile, data="beads",
                                            pixelsize_xy_um=(0.1, 0.1), dz_nm=50)["option"]


def test_the_runner_follows_the_worker_and_passes_its_failure_on(tmp_path, monkeypatch):
    """The protocol, with a stand-in for the worker: stages, bars, result, error."""
    fake = tmp_path / "worker.py"
    fake.write_text(textwrap.dedent(f"""
        import sys
        print("{runner.TAG}\\tstage\\tlearning the PSF", flush=True)
        sys.stderr.write("3/6: learning: 1/150 [00:01s]\\r3/6: learning: 2/150 [00:02s]\\r")
        sys.stderr.flush()
        if "fail" in sys.argv[1]:
            print("{runner.TAG}\\terror\\tValueError: no bead is found", flush=True)
            sys.exit(1)
        print("{runner.TAG}\\tresult\\t/somewhere/psfmodel.h5", flush=True)
    """))
    monkeypatch.setattr(runner, "WORKER", fake)
    said = []
    (tmp_path / "ok").mkdir()
    assert str(runner.run(tmp_path / "ok", python=sys.executable,
                          progress=said.append)) == "/somewhere/psfmodel.h5"
    assert said[0] == "uiPSF: learning the PSF"
    assert any("3/6: learning" in s for s in said)
    (tmp_path / "fail").mkdir()
    with pytest.raises(runner.UiPSFError, match="no bead is found"):
        runner.run(tmp_path / "fail", python=sys.executable)
    assert "3/6: learning: 2/150" in (tmp_path / "fail" / "uipsf.log").read_text()


def test_without_uipsf_it_says_where_to_put_it(monkeypatch):
    monkeypatch.delenv("SMAPPY_UIPSF_PYTHON", raising=False)
    monkeypatch.setattr("smappy.config.get", lambda key, default=None: default)
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(runner.UiPSFError, match="uiPSF python"):
        runner.find_python("")
    with pytest.raises(runner.UiPSFError, match="no Python at"):
        runner.find_python("/no/such/python")


@pytest.mark.slow
@pytest.mark.timeout(1200)          # uiPSF on a CPU: two to five minutes alone
@pytest.mark.skipif(not os.environ.get("SMAPPY_UIPSF_PYTHON"),
                    reason="SMAPPY_UIPSF_PYTHON names no Python with uiPSF")
def test_uipsf_learns_simulated_beads_that_spline_3d_then_fits_in_z(tmp_path, monkeypatch):
    """The whole path with uiPSF itself: beads -> uiPSF -> Spline 3D.

    The beads are the simulator's astigmatic Gaussians, which no pupil
    produces exactly, so the slope is allowed a few per cent where smappy's
    own calibration of the same beads gets 2 (test_camera_frames).
    """
    tifffile = pytest.importorskip("tifffile")
    from scipy.spatial import cKDTree
    from smappy.plugins import Context
    from smappy.plugins.fit import (CameraSettings, OutputSettings, SourceSettings,
                                    SplineFit, SplineFitSettings, SplineModelSettings)
    from smappy.plugins.uipsf import (BeadPart, LearningPart, UiPSFCalibration,
                                      UiPSFSettings)
    from smappy.session import Session
    from smappy.simulate import ISOLATED_NM, LabellingSettings, bead_stacks, camera_frames
    stacks, z = bead_stacks(2, seed=0, dz_nm=50, z_range_nm=(-1000, 1000))
    for i, stack in enumerate(stacks):
        tifffile.imwrite(tmp_path / f"beads{i}.tif", stack)
    camera = CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1)
    (tmp_path / "scope").mkdir()
    (tmp_path / "scope" / "sim.yaml").write_text(
        "NA: 1.45\nemission_wavelength_nm: 670\nroi_size_px: 21\n")
    monkeypatch.setenv("SMAPPY_MICROSCOPES", str(tmp_path / "scope"))
    # a test worker's one BLAS thread (conftest.py) would be uiPSF's too
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    result = UiPSFCalibration().run(Context(), UiPSFSettings(
        path=str(tmp_path / "beads0.tif"), microscope="sim", channels="single",
        beads=BeadPart(z_step_nm=50, max_beads=18), camera=camera,
        learning=LearningPart(iterations=60)))
    calibration = result.data["calibration"]
    assert load_spline_calibration(calibration).psf.shape[1] == 21
    assert [f.name for f in result.figures()] == ["data vs model", "localization bias",
                                                  "pupil", "Zernike", "beads"]
    frames, truth = camera_frames(300, seed=3, astigmatism=ASTIGMATISM,
                                  labelling=LabellingSettings(efficiency=0.1))
    tifffile.imwrite(tmp_path / "frames.tif", frames)
    SplineFit().run(Context(), SplineFitSettings(
        source=SourceSettings(path=str(tmp_path / "frames.tif")), camera=camera,
        model=SplineModelSettings(calibration=calibration),
        output=OutputSettings(path=str(tmp_path / "out.hdf5"))))
    session = Session()
    session.load(tmp_path / "out.hdf5")
    locs = session.locs
    clean = (truth["photons"] > 500) & (truth["neighbour_nm"] > ISOLATED_NM)
    fitted, true = [], []
    for f in np.unique(locs["frame"]):
        m, t = locs["frame"] == f, (truth["frame"] == f) & clean
        if not t.any():
            continue
        d, i = cKDTree(np.c_[truth["x_nm"][t], truth["y_nm"][t]]).query(
            np.c_[locs["x_nm"][m], locs["y_nm"][m]])
        fitted += list(locs["z_nm"][m][d < 60])
        true += list(truth["z_nm"][t][i[d < 60]])
    slope, intercept = np.polyfit(true, fitted, 1)
    assert slope == pytest.approx(1.0, abs=0.06)
    assert abs(intercept) < 15
