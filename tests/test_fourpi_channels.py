"""4Pi across a frame: the channel layout, the multi-channel engine, the
sources that read 4Pi data, and the Spline 4Pi fitter end to end."""

import numpy as np
import pytest

from smappy import fourpi
from smappy.channels import (ChannelLayout, from_corners, maps_from_uipsf_v1,
                             side_by_side)


@pytest.fixture(scope="module")
def cal():
    return fourpi.model_calibration()


def _shifted_layout():
    """Four 64-px channels on a strip, the last two mirrored in x, the third
    also shifted by a fraction of a pixel and stretched a little."""
    layout = from_corners([0, 70, 140, 210], [0] * 4, 64, 64, flip_x=[0, 0, 1, 1])
    m = np.eye(3)
    m[0, 2], m[1, 2], m[0, 0] = 0.6, -0.4, 1.002
    return layout.with_maps([np.eye(3), np.eye(3), m, np.eye(3)])


def test_a_layout_cuts_its_channels_and_undoes_their_mirror():
    frames = np.arange(2 * 10 * 30, dtype=float).reshape(2, 10, 30)
    layout = from_corners([0, 10, 20], [0, 0, 0], 8, 10, flip_x=[0, 0, 1])
    cut = layout.cut(frames)
    assert cut.shape == (2, 3, 10, 8)
    assert np.array_equal(cut[:, 1], frames[:, :, 10:18])
    assert np.array_equal(cut[:, 2], frames[:, :, 20:28][:, :, ::-1])
    # a channel's pixel back on the frame
    assert np.allclose(layout.on_frame(2, [[0, 3]]), [[27, 3]])


def test_a_layout_maps_both_ways_and_survives_its_record():
    layout = _shifted_layout()
    xy = np.array([[10.0, 20.0], [50.0, 5.0]])
    there = layout.to_channel(2, xy)
    assert np.allclose(layout.to_reference(2, there), xy)
    back = ChannelLayout.from_dict(layout.to_dict())
    assert np.allclose(back.to_channel(2, xy), there)
    assert back.flips == layout.flips


def test_uipsf_v1s_affines_put_channel_0_where_v1_did():
    """v1: channel 0's (a, b) lands in channel c at (p - c0) @ T + c0, its
    (a, b) this package's (x, y) for a transposed .mat."""
    T = np.array([[[1.001, 0.004, 0], [-0.005, 0.999, 0], [-0.45, 0.2, 1]]])
    centre = np.array([100.0, 100.0, 0.0])
    p = np.array([[30.0, 160.0]])
    v1 = (p - centre[:2]) @ T[0][:2, :2] + T[0][2, :2] + centre[:2]
    layout = side_by_side(2, 200, 200).with_maps(maps_from_uipsf_v1(T, centre, transpose=True))
    assert np.allclose(layout.to_channel(1, p), v1)
    plain = side_by_side(2, 200, 200).with_maps(maps_from_uipsf_v1(T, centre, transpose=False))
    assert np.allclose(plain.to_channel(1, p[:, ::-1])[:, ::-1], v1)


def test_the_engine_finds_and_fits_molecules_across_mirrored_shifted_channels(cal):
    """Simulated 4Pi frames: found on the channels' sum, fitted in all four,
    each where the layout's map puts it -- the positions come back unbiased
    and as precise as the CRLB says."""
    from scipy.spatial import cKDTree
    from smappy.metadata import CameraMetadata
    from smappy.multichannel import MultiChannelEngine
    from smappy.pipeline import FitSettings
    from smappy.plugins.fit import DetectionSettings
    from smappy.psf import Global4PiPSF
    layout = _shifted_layout()
    rng = np.random.default_rng(1)
    n_frames, per = 300, 2
    frame = np.repeat(np.arange(n_frames), per)
    x, y = rng.uniform(8, 56, (2, len(frame)))
    z = rng.uniform(-400, 400, len(frame))
    frames = fourpi.render_frames(cal, layout, frame, x, y, z, 3000, phase0=0.7,
                                  n_frames=n_frames, background=10)
    camera = CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=0.1)
    engine = MultiChannelEngine(camera, DetectionSettings().finder(), Global4PiPSF(cal),
                                layout, FitSettings(roisize=13))
    engine.push(layout.cut(frames), 0)
    locs = engine.flush()
    assert locs.metadata["fourpi"]["phase_per_nm"] == cal.phase_per_nm
    lx, ly = np.asarray(locs["x_nm"]) / 100, np.asarray(locs["y_nm"]) / 100
    found = []
    for f in range(n_frames):
        here, there = np.flatnonzero(locs["frame"] == f), np.flatnonzero(frame == f)
        if len(here):
            d, i = cKDTree(np.c_[x[there], y[there]]).query(np.c_[lx[here], ly[here]])
            found += [(h, there[j]) for h, j, dd in zip(here, i, d) if dd < 1]
    loc, true = map(np.array, zip(*found))
    assert len(loc) > 0.9 * len(frame)

    def spread(v):
        return 1.4826 * np.median(np.abs(v - np.median(v)))
    dx = lx[loc] - x[true]
    assert abs(np.median(dx)) < 0.01
    assert spread(dx) == pytest.approx(np.median(locs["x_err_nm"][loc]) / 100, rel=0.25)
    dz = np.asarray(locs["z_nm"])[loc] - z[true]
    assert spread(dz) == pytest.approx(np.median(locs["z_err_nm"][loc]), rel=0.25)
    dp = np.angle(np.exp(1j * (np.asarray(locs["phase_rad"])[loc]
                               - (cal.phase_per_nm * z[true] + 0.7))))
    assert spread(dp) == pytest.approx(np.median(locs["phase_err_rad"][loc]), rel=0.25)


def test_a_fit_clamped_at_the_end_of_the_z_grid_gets_the_whole_range_as_error(cal):
    from smappy.psf import FitResult, Global4PiPSF
    model = Global4PiPSF(cal)
    nz = cal.shape[0]
    theta = np.zeros((2, model.n_values), np.float32)
    theta[:, model.slot(4)] = [nz / 2, nz]
    result = FitResult(theta, np.ones_like(theta), np.zeros(2), np.zeros(2, int))
    out = model.unpack(result)
    assert out["z_err_nm"][0] == pytest.approx(cal.dz)
    assert out["z_err_nm"][1] == pytest.approx(nz * cal.dz)


def _write_quadrant_series(folder, stacks, files=2):
    """MATLAB files of qd1..qd4, (frames, x, y) as h5py reads them."""
    import h5py
    per = len(stacks[0]) // files
    paths = []
    for k in range(files):
        path = folder / f"cell_560_{43 + k:05d}_00000.mat"
        with h5py.File(path, "w") as f:
            for c, s in enumerate(stacks):
                f[f"qd{c + 1}"] = np.swapaxes(s[k * per:(k + 1) * per], -1, -2)
        paths.append(path)
    return paths


def test_a_quadrant_series_reads_as_one_acquisition_of_side_by_side_channels(tmp_path):
    from smappy.io.tiff import open_stack
    rng = np.random.default_rng(2)
    stacks = [rng.integers(100, 200, (30, 20, 24)).astype(np.uint16) for _ in range(4)]
    paths = _write_quadrant_series(tmp_path, stacks, files=3)
    source = open_stack(paths[1])
    assert source.n_frames == 30 and source.shape == (20, 96)
    assert source.layout.regions[2] == (48, 0, 24, 20)
    blocks = list(source.frames(chunk=7, start=5, stop=25))
    assert [b[0] for b in blocks] == [5, 12, 19]
    whole = np.concatenate([b[1] for b in blocks])
    assert np.array_equal(source.layout.cut(whole)[:, 3], stacks[3][5:25])


def test_spline_4pi_fits_a_quadrant_series_and_puts_its_phase_into_z(cal, tmp_path):
    """The fitter on simulated quadrant files: the molecules fitted in all
    four channels, then 4Pi z, after the fit, unwraps their phase -- z on the
    right fringe for nearly all of them."""
    from scipy.spatial import cKDTree
    from smappy.plugins import Context
    from smappy.plugins.fit import CameraSettings, OutputSettings, SourceSettings
    from smappy.plugins.fit_4pi import FourPiFit, FourPiFitSettings, FourPiModelSettings
    model_path = tmp_path / "model_4pi.h5"
    fourpi.save_fourpi_calibration(model_path, cal)
    layout = side_by_side(4, 48, 48)
    rng = np.random.default_rng(3)
    n_frames = 400
    frame = np.arange(n_frames)
    x, y = rng.uniform(10, 38, (2, n_frames))
    z = rng.uniform(-400, 400, n_frames)
    drift = fourpi.cavity_drift(n_frames, rate_rad=0.002, start_rad=0.5)
    frames = fourpi.render_frames(cal, layout, frame, x, y, z, 4000, phase0=drift[frame],
                                  n_frames=n_frames, background=8)
    adu = (frames / 0.5 + 100).astype(np.uint16)          # conversion 0.5, offset 100
    stacks = [layout.cut(adu)[:, c] for c in range(4)]
    paths = _write_quadrant_series(tmp_path, stacks, files=2)
    settings = FourPiFitSettings(
        source=SourceSettings(path=str(paths[0])),
        camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1),
        model=FourPiModelSettings(calibration=str(model_path)),
        output=OutputSettings(path=str(tmp_path / "out_locs.hdf5")))
    settings.finish.fourpi.window_frames = 100
    result = FourPiFit().run(Context(), settings)
    locs = result.locs
    assert "4Pi z" in result.text
    assert {"phase_rad", "z_env_nm", "fringe_probability"} <= set(locs.keys())
    d, i = cKDTree(np.c_[x, y]).query(np.c_[locs["x_nm"], locs["y_nm"]] / 100)
    same = (d < 1) & (np.asarray(locs["frame"]) == frame[i])
    dz = np.asarray(locs["z_nm"])[same] - z[i[same]]
    assert same.sum() > 0.9 * n_frames
    assert np.mean(np.abs(dz) < 15) > 0.95


def test_after_the_fit_offers_4pi_z_only_for_a_table_with_a_phase():
    from smappy.plugins.after_fit import AfterFitSettings, applicable, shown
    s = AfterFitSettings()
    assert not applicable(s, False, False).fourpi_z
    assert applicable(s, False, False, phase=True).fourpi_z
    assert not shown(s, False, False)["fourpi_z"]
    assert shown(s, False, False, phase=True)["fourpi"]


def _v1_insitu():
    import os
    from pathlib import Path
    root = Path(os.environ.get("UIPSF_DATA", Path.home() / ".cache/uipsf/zenodo"))
    path = root / "results/4pi_smlm_nup96/psfmodel_iter1_insitu_zernike_4pi.h5"
    if not path.exists():
        pytest.skip("uiPSF's in-situ 4Pi model is not downloaded "
                    "(scripts/uipsf/fetch_zenodo.py results/4pi_smlm_nup96/)")
    return path


def test_v1s_in_situ_model_has_its_phase_rising_with_z():
    """v1 demodulated its in-situ models the other way round from its bead
    models.  On the Nup96 molecules v1 learned from, the phase's z then
    agrees with the envelope's in sign and nearly in scale."""
    import h5py
    from smappy.psf import Global4PiPSF
    path = _v1_insitu()
    cal = fourpi.load_uipsf_v1(path, transpose=False)
    assert cal.phase_per_nm > 0
    with h5py.File(path, "r") as f:
        data, cor = f["rois/psf_data"][...], f["rois/cor"][...]
        T, centre, frames = f["res/T"][...], f["res/imgcenter"][...], f["rois/fileID"][0]
    C, n, size, _ = data.shape
    o = (size - 13) // 2
    rois = np.moveaxis(data[..., o:o + 13, o:o + 13], 0, 1).astype(np.float32)
    ref = np.c_[cor[0], np.ones(n)]
    transforms = np.concatenate([np.eye(3)[None], T])
    dx, dy = np.zeros((n, C)), np.zeros((n, C))
    for c in range(C):
        m = (ref - centre) @ transforms[c] + centre
        dy[:, c], dx[:, c] = m[:, 0] - cor[c][:, 0], m[:, 1] - cor[c][:, 1]
    link = fourpi.link(n, C, dx=dx, dy=dy)
    model = Global4PiPSF(cal)
    fit = model.unpack(model.fit(rois, link, iterations=100), link)
    k = cal.phase_per_nm
    cav = fourpi.cavity_phase(fit["phase_rad"], fit["phase_err_rad"], fit["z_env_nm"],
                              fit["z_env_err_nm"], frames, k, window_frames=1000)
    found = fourpi.unwrap(fit["phase_rad"], fit["phase_err_rad"], fit["z_env_nm"],
                          fit["z_env_err_nm"], cav.at, k)
    sure = found.probability > 0.9
    slope = np.polyfit(fit["z_env_nm"][sure], found.z_nm[sure], 1)[0]
    assert 0.9 < slope < 1.25
    assert np.mean(1 - found.probability) < 0.05


# ---------------------------------------------------------------- registration

def _bead_projection(cal, layout, xy, seed=0):
    """Beads at ``xy`` (channel-0 pixels) scanned through z and the phase
    steps, summed: what `bead_projections` gives for a bead stack."""
    z = np.linspace(-400, 400, 17)
    n = len(xy)
    frames = []
    for k, phase in enumerate((-2.0, 0.0, 2.0)):
        frame = np.repeat(np.arange(len(z)), n)
        frames.append(fourpi.render_frames(
            cal, layout, frame, np.tile(xy[:, 0], len(z)), np.tile(xy[:, 1], len(z)),
            np.repeat(z, n), 20000, phase0=phase, n_frames=len(z), background=5, seed=seed + k))
    return layout.cut(np.concatenate(frames).sum(axis=0, keepdims=True))[0]


def test_beads_register_every_channel_onto_the_first(cal):
    """A projective map with a shift, a stretch and a slight rotation, found
    again from the beads' summed images to a few hundredths of a pixel."""
    from smappy.channels import register_channels
    truth = np.eye(3)
    truth[:2, :2] = [[1.003, 0.004], [-0.004, 0.998]]
    truth[:2, 2] = [1.3, -0.7]
    layout = side_by_side(4, 96, 96).with_maps([np.eye(3), truth, np.eye(3), truth])
    rng = np.random.default_rng(4)
    projections = [_bead_projection(cal, layout, rng.uniform(12, 84, (5, 2)), seed=s)
                   for s in range(3)]
    reg = register_channels(projections)
    assert reg.pairs[1] >= 12 and reg.kind[1] == "projective"
    grid = np.array([[15.0, 15.0], [80.0, 20.0], [48.0, 48.0], [20.0, 80.0], [80.0, 80.0]])
    found = side_by_side(4, 96, 96).with_maps(reg.maps)
    for c in (1, 2, 3):
        assert np.abs(found.to_channel(c, grid) - layout.to_channel(c, grid)).max() < 0.05


@pytest.mark.slow
def test_the_real_beads_register_as_uipsf_v1_learned_its_maps():
    """uiPSF's 4Pi beads, summed, registered as here: within a tenth of a
    pixel of the maps v1 learned with its model, where the beads are (the
    corners, beyond them, are the two fits' extrapolations)."""
    import os
    from pathlib import Path
    from smappy.channels import bead_projections, register_channels
    root = Path(os.environ.get("UIPSF_DATA", Path.home() / ".cache/uipsf/zenodo"))
    beads, model = root / "4pi_40nm_bead", root / "results/4pi_40nm_bead/psfmodel_zernike_4pi.h5"
    if not beads.exists() or not model.exists():
        pytest.skip("uiPSF's 4Pi beads and model are not downloaded")
    reg = register_channels(bead_projections(beads, side_by_side(4, 200, 200)))
    assert min(reg.pairs[1:]) > 50 and max(reg.rms_px) < 0.1
    v1 = side_by_side(4, 200, 200).with_maps(fourpi.load_uipsf_v1(model).maps())
    found = side_by_side(4, 200, 200).with_maps(reg.maps)
    grid = np.array([[60.0, 60.0], [140.0, 60.0], [100.0, 100.0], [60.0, 140.0], [140.0, 140.0]])
    for c in (1, 2, 3):
        assert np.abs(found.to_channel(c, grid) - v1.to_channel(c, grid)).max() < 0.1


def test_a_bead_scan_file_reads_its_z_and_phase_steps_as_frames(tmp_path):
    import h5py
    from smappy.io.tiff import open_stack
    rng = np.random.default_rng(5)
    stacks = [rng.integers(0, 1000, (5, 3, 8, 10)).astype(np.uint16) for _ in range(4)]
    with h5py.File(tmp_path / "bead1.mat", "w") as f:
        for i, a in enumerate(stacks):
            f[f"qd{i + 1}"] = np.swapaxes(a, -1, -2)
    source = open_stack(tmp_path / "bead1.mat")
    assert source.n_frames == 15 and source.shape == (8, 40)
    frames = np.concatenate([b for _, b in source.frames(chunk=4)])
    assert np.array_equal(source.layout.cut(frames)[:, 2], stacks[2].reshape(15, 8, 10))


# ---------------------------------------------------------------- channel drift

@pytest.mark.slow
def test_a_drift_of_one_channel_is_measured_and_followed(cal):
    """Channel 2 moves by 0.8 pixels in x over the acquisition, against a
    registration that does not know it.  Fitted with x and y free per
    channel the drift comes back; followed, the fit is unbiased again."""
    from smappy.metadata import CameraMetadata
    from smappy.multichannel import MultiChannelEngine, measure_channel_drift
    from smappy.pipeline import FitSettings
    from smappy.plugins.fit import DetectionSettings
    from smappy.psf import Global4PiPSF
    layout = side_by_side(4, 64, 64)
    rng = np.random.default_rng(6)
    blocks, per_block = 4, 150
    frames, truth = [], []
    for b in range(blocks):
        m = np.eye(3)
        m[0, 2] = -0.8 * b / (blocks - 1)          # channel 2 drawn further right over time
        moving = layout.with_maps([np.eye(3), np.eye(3), m, np.eye(3)])
        frame = np.repeat(np.arange(per_block), 2)
        x, y = rng.uniform(10, 54, (2, len(frame)))
        z = rng.uniform(-300, 300, len(frame))
        frames.append(fourpi.render_frames(cal, moving, frame, x, y, z, 4000, phase0=0.3,
                                           n_frames=per_block, background=8, seed=b))
        truth.append((frame + b * per_block, x, y, z))
    frames = np.concatenate(frames)
    camera = CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=0.1)
    finder = DetectionSettings().finder()

    def fit(model, drift=None):
        engine = MultiChannelEngine(camera, finder, model, layout, FitSettings(roisize=13), drift)
        engine.push(layout.cut(frames), 0)
        return engine.flush()

    free = fit(Global4PiPSF(cal, shared=(False, False, True, False, True, True)))
    drift = measure_channel_drift(free, 4, 100.0, lambda f: f // per_block)
    assert np.allclose(drift.offsets[:, 2, 0], 0.8 * np.arange(blocks) / (blocks - 1), atol=0.06)
    assert np.allclose(drift.offsets[:, [1, 3], :], 0, atol=0.06)

    followed = fit(Global4PiPSF(cal), drift)
    assert followed.metadata["channel_drift"]["offsets_px"]
    from scipy.spatial import cKDTree
    frame, x, y, _ = map(np.concatenate, zip(*truth))
    d, i = cKDTree(np.c_[x, y, frame * 1000.0]).query(
        np.c_[followed["x_nm"] / 100, followed["y_nm"] / 100, followed["frame"] * 1000.0])
    near = d < 1
    late = frame[i] >= 3 * per_block
    dx = followed["x_nm"][near & late] / 100 - x[i[near & late]]
    assert abs(np.median(dx)) < 0.03


# ---------------------------------------------------------------- grouping

def test_grouping_averages_the_phase_as_an_angle():
    from smappy.group import combine
    from smappy.locs import Localizations
    locs = Localizations({
        "x_nm": np.zeros(4, np.float32), "y_nm": np.zeros(4, np.float32),
        "frame": np.arange(4), "x_err_nm": np.ones(4, np.float32),
        "y_err_nm": np.ones(4, np.float32),
        "phase_rad": np.array([3.1, -3.1, 0.1, 0.3], np.float32),
        "phase_err_rad": np.array([0.1, 0.1, 0.1, 0.05], np.float32)}, {})
    grouped = combine(locs, np.array([1, 1, 2, 2]))
    assert abs(abs(grouped["phase_rad"][0]) - np.pi) < 1e-5     # not 0, the linear mean
    assert grouped["phase_rad"][1] == pytest.approx((0.1 * 100 + 0.3 * 400) / 500, abs=1e-3)
    assert grouped["phase_err_rad"][1] == pytest.approx(1 / np.sqrt(100 + 400), rel=1e-5)


# ---------------------------------------------------------------- the channel layer

def test_a_channel_source_reads_any_acquisition_as_its_channels(tmp_path):
    from smappy.channels import ChannelSource
    from smappy.io.tiff import open_stack
    rng = np.random.default_rng(7)
    stacks = [rng.integers(100, 200, (12, 16, 20)).astype(np.uint16) for _ in range(4)]
    paths = _write_quadrant_series(tmp_path, stacks, files=2)
    base = open_stack(paths[0])
    source = ChannelSource.wrap(base, base.layout)
    assert source.shape == (16, 20) and source.n_frames == 12 and source.n_channels == 4
    blocks = list(source.frames(chunk=5))
    assert blocks[0][1].shape == (5, 4, 16, 20)
    assert np.array_equal(np.concatenate([b for _, b in blocks])[:, 1], stacks[1])
    assert np.array_equal(source.frame(7)[3], stacks[3][7])


def test_a_layout_on_the_chip_is_put_on_a_camera_roi():
    layout = from_corners([100, 300], [50, 50], 64, 64, flip_x=[False, True])
    on = layout.on_camera((90, 40))
    assert on.regions == [(10, 10, 64, 64), (210, 10, 64, 64)]
    assert on.flips == layout.flips
    with pytest.raises(ValueError, match="cuts off"):
        layout.on_camera((120, 0))


def test_a_map_is_described_about_its_channels_centre():
    from smappy.channels import describe_map
    m = np.eye(3)
    angle = 0.004
    m[:2, :2] = 1.002 * np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
    centre = np.array([99.5, 99.5])
    m[:2, 2] = centre + [0.3, -0.2] - m[:2, :2] @ centre       # 0.3, -0.2 px at the centre
    d = describe_map(m, (200, 200))
    assert np.allclose(d["shift_px"], [0.3, -0.2], atol=1e-6)
    assert np.allclose(d["scale"], 1.002, atol=1e-6)
    assert d["rotation_mrad"] == pytest.approx(4.0, abs=1e-3)
    assert d["projective_px"] < 1e-6


def test_the_microscope_database_places_the_channels_on_the_chip():
    from types import SimpleNamespace
    from smappy.microscope_db import Microscope
    from smappy.plugins.fit_4pi import FourPiFitSettings, layout_of
    scope = Microscope.from_dict({
        "name": "4Pi", "camera": "orca",
        "channels": {"p1": {"region": [151, 0, 168, 168]},
                     "s1": {"region": [389, 0, 168, 168]},
                     "p2": {"region": {"orca": [1474, 0, 168, 168]}, "mirror": "x"},
                     "s2": {"region": {"orca": [1713, 0, 168, 168]}, "mirror": "x"}}})
    chip = scope.channel_layout("orca")
    assert chip.regions[2] == (1474, 0, 168, 168) and chip.flips[3] == (True, False)
    assert scope.channel_layout("another").n_channels == 2     # only the two for every camera
    resolution = SimpleNamespace(microscope=SimpleNamespace(microscope=scope, metadata={},
                                                            camera="orca"))
    settings = FourPiFitSettings()
    settings.model.channels.layout = "microscope"
    settings.model.channels.registration = "none"
    camera = SimpleNamespace(roi_offset=(100, 0))
    layout = layout_of(settings, None, None, camera, resolution)
    assert layout.regions[0] == (51, 0, 168, 168)
    assert Microscope.from_dict(scope.to_dict()).channel_layout("orca").regions == chip.regions


def test_a_bead_registration_is_measured_once_and_kept_beside_the_beads(cal, tmp_path,
                                                                       monkeypatch):
    import smappy.channels as channels
    from smappy.plugins import fit_4pi
    truth = np.eye(3)
    truth[:2, 2] = [0.7, -0.4]
    layout = side_by_side(4, 64, 64).with_maps([np.eye(3), truth, truth, np.eye(3)])
    rng = np.random.default_rng(8)
    xy = rng.uniform(10, 54, (6, 2))
    z = np.linspace(-300, 300, 9)
    frame = np.repeat(np.arange(len(z)), len(xy))
    frames = fourpi.render_frames(cal, layout, frame, np.tile(xy[:, 0], len(z)),
                                  np.tile(xy[:, 1], len(z)), np.repeat(z, len(xy)), 20000,
                                  phase0=0.0, n_frames=len(z), background=5)
    beads = tmp_path / "beads"
    beads.mkdir()
    _write_quadrant_series(beads, [layout.cut(frames)[:, c].astype(np.uint16)
                                   for c in range(4)], files=1)
    plain = side_by_side(4, 64, 64)
    fit_4pi._REGISTERED.clear()
    maps = fit_4pi.registration_from_beads(str(beads), plain)
    assert np.allclose(maps[1][:2, 2], [0.7, -0.4], atol=0.05)
    kept = channels.registration_path(beads)
    assert kept.exists()
    assert np.allclose(channels.load_registration(kept).maps[1], maps[1])
    # the second time it is read, not measured
    fit_4pi._REGISTERED.clear()
    monkeypatch.setattr(channels, "register_channels", lambda *a, **k: 1 / 0)
    again = fit_4pi.registration_from_beads(str(beads), plain)
    assert np.allclose(again[1], maps[1])


def test_a_fit_region_is_drawn_in_channel_0s_pixels(cal, tmp_path):
    from smappy.plugins import Context
    from smappy.plugins.fit import CameraSettings, OutputSettings, SourceSettings
    from smappy.plugins.fit_4pi import FourPiFit, FourPiFitSettings, FourPiModelSettings
    model_path = tmp_path / "model_4pi.h5"
    fourpi.save_fourpi_calibration(model_path, cal)
    layout = side_by_side(4, 48, 48)
    rng = np.random.default_rng(9)
    n = 120
    x, y = rng.uniform(10, 38, (2, n))
    frames = fourpi.render_frames(cal, layout, np.arange(n), x, y, rng.uniform(-300, 300, n),
                                  4000, phase0=0.5, n_frames=n, background=8)
    adu = (frames / 0.5 + 100).astype(np.uint16)
    paths = _write_quadrant_series(tmp_path, [layout.cut(adu)[:, c] for c in range(4)], 1)
    settings = FourPiFitSettings(
        source=SourceSettings(path=str(paths[0])),
        camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1),
        model=FourPiModelSettings(calibration=str(model_path)),
        output=OutputSettings(path=str(tmp_path / "out_locs.hdf5")))
    settings.finish.fourpi_z = False
    settings.region.shapes = [{"kind": "include",
                               "points": [[0, 0], [24, 0], [24, 48], [0, 48]]}]
    locs = FourPiFit().run(Context(), settings).locs
    assert len(locs) > 0.3 * n
    assert np.all(np.asarray(locs["x_nm"]) / 100 < 25)


def test_spline_4pi_reads_a_dcimg_chip_by_typed_mirrored_corners(cal, tmp_path):
    """Four channels typed on a DCIMG chip, the last two mirrored: the layer
    takes them apart, and the molecules come back where they were drawn in
    channel 0's pixels."""
    from tests.test_fourpi import _write_new_dcimg
    from smappy.plugins import Context
    from smappy.plugins.fit import CameraSettings, OutputSettings, SourceSettings
    from smappy.plugins.fit_4pi import FourPiFit, FourPiFitSettings, FourPiModelSettings
    model_path = tmp_path / "model_4pi.h5"
    fourpi.save_fourpi_calibration(model_path, cal)
    layout = from_corners([4, 54, 104, 154], [2] * 4, 44, 44, flip_x=[0, 0, 1, 1])
    rng = np.random.default_rng(10)
    n = 80
    x, y = rng.uniform(10, 34, (2, n))
    frames = fourpi.render_frames(cal, layout, np.arange(n), x, y, rng.uniform(-300, 300, n),
                                  4000, phase0=0.2, n_frames=n, background=8,
                                  frame_shape=(48, 200))
    adu = (frames / 0.5 + 100).astype(np.uint16)
    _write_new_dcimg(tmp_path / "movie.dcimg", adu, adu[:, 1023 - (1024 - 24), :4])
    settings = FourPiFitSettings(
        source=SourceSettings(path=str(tmp_path / "movie.dcimg")),
        camera=CameraSettings(conversion=0.5, offset=100.0, pixelsize_um=0.1),
        model=FourPiModelSettings(calibration=str(model_path)),
        output=OutputSettings(path=str(tmp_path / "out_locs.hdf5")))
    part = settings.model.channels
    part.layout, part.registration = "typed", "none"
    part.x_corners, part.y_corners, part.width, part.height = "4, 54, 104, 154", "2,2,2,2", 44, 44
    part.mirrored = "3, 4"
    settings.finish.fourpi_z = False
    locs = FourPiFit().run(Context(), settings).locs
    from scipy.spatial import cKDTree
    d, i = cKDTree(np.c_[x, y]).query(np.c_[locs["x_nm"], locs["y_nm"]] / 100)
    same = (d < 1) & (np.asarray(locs["frame"]) == i)
    assert same.sum() > 0.8 * n
    assert abs(np.median(np.asarray(locs["x_nm"])[same] / 100 - x[i[same]])) < 0.02


# ---------------------------------------------------------------- detection's resampling

def test_the_nearest_pixel_lookup_reads_each_pixel_where_the_map_puts_it_to_half_a_pixel():
    from smappy.channels import channel_grid, detection_lookup, onto_reference
    m = np.eye(3)
    angle, scale = 0.01, 1.003
    m[:2, :2] = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    m[:2, 2] = [1.4, -2.2]
    layout = side_by_side(2, 60, 50).with_maps([np.eye(3), m])
    lookup = detection_lookup(layout, 1, (50, 60), "nearest")
    rows, cols = channel_grid(layout, 1, (50, 60))
    picked_rows, picked_cols = np.divmod(lookup.index.reshape(50, 60), 60)
    inside = lookup.inside.reshape(50, 60)
    assert np.abs(picked_rows - rows)[inside].max() <= 0.5 + 1e-9
    assert np.abs(picked_cols - cols)[inside].max() <= 0.5 + 1e-9
    image = np.arange(3 * 50 * 60, dtype=np.float32).reshape(3, 50, 60)
    out = onto_reference(image, lookup)
    assert np.array_equal(out[2][inside], image[2].ravel()[lookup.index][inside.ravel()])
    assert np.all(out[:, ~inside] == 0)


def test_a_shift_lookup_moves_a_channel_by_whole_pixels():
    from smappy.channels import detection_lookup, onto_reference
    m = np.eye(3)
    m[:2, 2] = [-3.2, 1.9]                 # channel 1's (x, y) is channel 0's minus that
    layout = side_by_side(2, 30, 20).with_maps([np.eye(3), m])
    lookup = detection_lookup(layout, 1, (20, 30), "shift")
    assert lookup.offset == (3, -2)
    image = np.random.default_rng(0).random((1, 20, 30)).astype(np.float32)
    out = onto_reference(image, lookup)
    assert np.array_equal(out[0, 2:, :27], image[0, :18, 3:])


def test_nearest_pixel_detection_finds_what_bilinear_does(cal):
    """The molecules found on the channels' sum, with a registration that rotates and stretches: the same
    with the nearest pixel as with bilinear interpolation, which costs several times more."""
    from smappy.metadata import CameraMetadata
    from smappy.multichannel import MultiChannelEngine
    from smappy.pipeline import FitSettings
    from smappy.plugins.fit import DetectionSettings
    from smappy.psf import Global4PiPSF
    m = np.eye(3)
    angle = 0.01
    m[:2, :2] = 1.003 * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    m[:2, 2] = [0.7, -0.4]
    layout = side_by_side(4, 64, 64).with_maps([np.eye(3), m, m, np.eye(3)])
    rng = np.random.default_rng(13)
    frame = np.repeat(np.arange(200), 2)
    x, y = rng.uniform(10, 54, (2, len(frame)))
    frames = fourpi.render_frames(cal, layout, frame, x, y, rng.uniform(-300, 300, len(frame)), 2500,
                                  phase0=0.3, n_frames=200, background=10)
    found = {}
    for mode in ("bilinear", "nearest"):
        engine = MultiChannelEngine(CameraMetadata(conversion=1.0, offset=0.0, pixelsize_um=0.1),
                                    DetectionSettings().finder(), Global4PiPSF(cal), layout,
                                    FitSettings(roisize=13), resampling=mode)
        engine.push(layout.cut(frames), 0)
        found[mode] = len(engine.flush())
    assert abs(found["nearest"] - found["bilinear"]) <= 0.02 * found["bilinear"]
