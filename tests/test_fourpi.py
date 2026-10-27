"""4Pi: the IAB model's global fit, its calibration file and the 4Pi readers."""

import numpy as np
import pytest

from smappy import fourpi
from smappy.psf import Global4PiPSF


@pytest.fixture(scope="module")
def cal():
    return fourpi.model_calibration()


@pytest.fixture(scope="module")
def fitted(cal):
    rois, truth = fourpi.simulate_rois(cal, 600, photons=4000, background=10, seed=1)
    model = Global4PiPSF(cal)
    return model.unpack(model.fit(rois)), truth


def _wrapped(d):
    return np.angle(np.exp(1j * d))


def test_the_4pi_fit_reaches_the_crlb_in_x_z_and_phase(fitted):
    values, truth = fitted
    for name, err, true in (("x_roi", "x_err_pix", truth["x"]),
                            ("y_roi", "y_err_pix", truth["y"]),
                            ("z_nm", "z_err_nm", truth["z_nm"])):
        d = values[name] - true
        assert abs(np.median(d)) < 0.2 * np.median(values[err])
        assert np.std(d) == pytest.approx(np.median(values[err]), rel=0.15)
    d = _wrapped(values["phase_rad"] - truth["phase"])
    assert np.std(d) == pytest.approx(np.median(values["phase_err_rad"]), rel=0.15)


def test_the_phasor_start_finds_the_phase_all_round_the_circle(fitted):
    values, truth = fitted
    d = np.abs(_wrapped(values["phase_rad"] - truth["phase"]))
    quadrant = ((truth["phase"] + np.pi) // (np.pi / 2)).astype(int)
    for q in range(4):
        assert (quadrant == q).sum() > 50
        assert np.max(d[quadrant == q]) < 0.3      # no fit stuck off the true phase


def test_a_shared_photon_number_is_the_emitters_total(fitted):
    values, _ = fitted
    assert np.median(values["photons"]) == pytest.approx(4000, rel=0.02)
    assert values["z_env_nm"] is values["z_nm"] or np.array_equal(values["z_env_nm"],
                                                                  values["z_nm"])


def test_the_link_carries_a_channels_sub_pixel_offset(cal):
    """Channel 2's ROI was cut 0.4 pixels off where the emitter is: with that
    offset in the link the shared x is channel 0's position."""
    from smappy.fourpi import evaluate
    rng = np.random.default_rng(3)
    n, size = 200, 13
    x = 6 + rng.uniform(-0.3, 0.3, n)
    z = rng.uniform(-300, 300, n)
    phase = rng.uniform(-np.pi, np.pi, n)
    shift = np.array([0.0, 0.0, 0.4, 0.0])
    rois = np.empty((n, 4, size, size), np.float32)
    for i in range(n):
        for c in range(4):
            mu = 4000 * np.clip(evaluate(cal, c, x[i] + shift[c], 6.0,
                                         cal.z_nm_to_index(z[i]), phase[i], size), 0, None)
            rois[i, c] = rng.poisson(mu + 10)
    link = fourpi.link(n, 4, dx=np.broadcast_to(shift, (n, 4)))
    values = Global4PiPSF(cal).unpack(Global4PiPSF(cal).fit(rois, link), link)
    assert abs(np.median(values["x_roi"] - x)) < 0.01
    without = Global4PiPSF(cal).unpack(Global4PiPSF(cal).fit(rois))
    assert abs(np.median(without["x_roi"] - x)) > 0.05


def test_a_4pi_calibration_survives_its_file(cal, tmp_path):
    path = tmp_path / "model_4pi.h5"
    fourpi.save_fourpi_calibration(path, cal)
    back = fourpi.load_fourpi_calibration(path)
    assert np.array_equal(back.coeff, cal.coeff)
    assert (back.dz, back.z0, back.period_nm, back.phase_per_nm) == \
        (cal.dz, cal.z0, cal.period_nm, cal.phase_per_nm)
    assert back.parameters["modulation"] == cal.parameters["modulation"]
    assert fourpi.is_fourpi_calibration(path)
    with pytest.raises(FileExistsError):
        fourpi.save_fourpi_calibration(path, cal)


def test_the_channels_are_normalised_to_one_photon_together(cal):
    assert cal.photon_shares().sum() == pytest.approx(1.0, abs=1e-5)
    I, A, B = cal.volumes[:, 0], cal.volumes[:, 1], cal.volumes[:, 2]
    assert np.min(I - np.hypot(A, B)) == pytest.approx(0.0, abs=1e-7)


def test_the_phase_follows_z_by_the_calibrations_slope(cal):
    """A cavity phase held fixed: the fitted phase then moves with z by
    `phase_per_nm`, which is what unwrapping relies on."""
    z = np.linspace(-100, 100, 9)
    rois, _ = fourpi.simulate_rois(cal, len(z), photons=20000, background=2,
                                       z_nm=z, phase0=0.3, xy_spread=0, seed=4)
    values = Global4PiPSF(cal).unpack(Global4PiPSF(cal).fit(rois))
    slope = np.polyfit(z, np.unwrap(values["phase_rad"]), 1)[0]
    assert slope == pytest.approx(cal.phase_per_nm, rel=0.02)


# ---------------------------------------------------------------- readers

def _write_new_dcimg(path, frames, four, y0=None):
    """A DCIMG file of the new (0x1000000) version, as the camera lays it out:
    each frame followed by a 32-byte footer holding its number, its time
    stamp and the 4 pixels, which the image itself holds wrongly."""
    n, h, w = frames.shape
    header_size, offset_to_data = 112, 1024
    image = h * w * 2
    out = bytearray(header_size + offset_to_data + n * (image + 32))
    out[0:8] = b"DCIMG\0\0\0"
    np.frombuffer(out, "<u4", 1, 8)[0] = 0x1000000
    np.frombuffer(out, "<u4", 1, 0x28)[0] = header_size
    s = header_size
    np.frombuffer(out, "<u4", 1, s + 8 + 52)[0] = n          # n_frames
    np.frombuffer(out, "<u4", 1, s + 64)[0] = 2              # byte depth
    np.frombuffer(out, "<u4", 1, s + 72)[0] = w
    np.frombuffer(out, "<u4", 1, s + 76)[0] = h
    np.frombuffer(out, "<u4", 1, s + 80)[0] = w * 2          # bytes per row
    np.frombuffer(out, "<u4", 1, s + 84)[0] = image
    np.frombuffer(out, "<u8", 1, s + 96)[0] = offset_to_data
    if y0 is not None:
        np.frombuffer(out, "<u2", 4, s + 712)[:] = (0, w, y0, h)
    for k in range(n):
        at = header_size + offset_to_data + k * (image + 32)
        bad = frames[k].copy()
        row = (1023 - (1024 - h // 2 if y0 is None else y0))
        bad[row, :4] = (0, 65535, 0, 65535)
        out[at:at + image] = bad.astype("<u2").tobytes()
        np.frombuffer(out, "<u4", 1, at + image)[0] = k
        np.frombuffer(out, "<u2", 4, at + image + 12)[:] = four[k]
    path.write_bytes(bytes(out))
    return row


@pytest.mark.parametrize("y0", [None, 1000])
def test_a_dcimg_file_reads_with_its_four_pixels_put_back(tmp_path, y0):
    from smappy.io.dcimg import DcimgFile, read_dcimg
    rng = np.random.default_rng(5)
    frames = rng.integers(100, 200, (3, 40, 64)).astype(np.uint16)
    row = _write_new_dcimg(tmp_path / "a.dcimg", frames, frames[:, (1023 - (1024 - 20 if y0 is None else y0)), :4], y0)
    f = DcimgFile(tmp_path / "a.dcimg")
    assert f.shape == (3, 40, 64)
    assert np.array_equal(f.frames(), frames)
    assert np.array_equal(f.frames(1, 2), frames[1:2])
    assert np.array_equal(read_dcimg([tmp_path / "a.dcimg"] * 2),
                          np.concatenate([frames, frames]))
    assert 0 <= row < 40


def test_a_quadrant_mat_reads_as_channel_z_phase_y_x(tmp_path):
    """MATLAB's (y, x, phase, z) arrives through h5py as (z, phase, x, y)."""
    import h5py

    from smappy.io.quadrants import read_quadrant_mat
    rng = np.random.default_rng(6)
    stacks = [rng.integers(0, 1000, (5, 3, 8, 10)).astype(np.uint16) for _ in range(4)]
    with h5py.File(tmp_path / "bead.mat", "w") as f:
        for i, a in enumerate(stacks):
            f[f"qd{i + 1}"] = np.swapaxes(a, -1, -2)      # as MATLAB writes it
    out = read_quadrant_mat(tmp_path / "bead.mat")
    assert out.shape == (4, 5, 3, 8, 10)
    assert np.array_equal(out[2], stacks[2])


# ---------------------------------------------------------------- uiPSF v1

def _v1_model():
    import os
    from pathlib import Path
    root = Path(os.environ.get("UIPSF_DATA", Path.home() / ".cache/uipsf/zenodo"))
    path = root / "results/4pi_40nm_bead/psfmodel_zernike_4pi.h5"
    if not path.exists():
        pytest.skip("uiPSF's 4Pi bead model is not downloaded "
                    "(scripts/uipsf/fetch_zenodo.py results/4pi_40nm_bead/)")
    return path


def test_the_fit_of_v1s_beads_matches_v1s_own_4pi_fitter():
    """The beads uiPSF v1 learned its 4Pi model from, fitted with that model
    by this fitter and by v1's (closed-source) one, whose results the file
    keeps: x, y, z and the phase agree per localization far inside their
    precision.  The ROIs are in v1's frame, so the model is not transposed."""
    import h5py
    path = _v1_model()
    cal = fourpi.load_uipsf_v1(path, transpose=False)
    with h5py.File(path, "r") as f:
        data = f["rois/psf_data"][...]                 # (C, bead, phase, z, y, x)
        P = f["locres/P"][...]                         # v1: x, y, N, bg, z, phase
        cor, T, centre = f["rois/cor"][...], f["res/T"][...], f["res/imgcenter"][...]
    C, nb, nph, nz, size, _ = data.shape
    crop = 15
    o = (size - crop) // 2
    rois = np.moveaxis(data[..., o:o + crop, o:o + crop], 0, 3).reshape(-1, C, crop, crop)
    # v1's sub-pixel offsets of each channel's ROI from where its affine
    # puts the reference channel's (loclib.loc_4pi)
    ref = np.c_[cor[0], np.ones(nb)]
    transforms = np.concatenate([np.eye(3)[None], T])
    dx, dy = np.zeros((nb, C)), np.zeros((nb, C))
    for c in range(C):
        mapped = (ref - centre) @ transforms[c] + centre
        dy[:, c], dx[:, c] = mapped[:, 0] - cor[c][:, 0], mapped[:, 1] - cor[c][:, 1]
    link = fourpi.link(len(rois), C, dx=np.repeat(dx, nph * nz, 0),
                       dy=np.repeat(dy, nph * nz, 0))
    model = Global4PiPSF(cal)
    result = model.fit(rois.astype(np.float32), link, iterations=100)
    theta = result.theta

    def spread(d):
        return np.median(np.abs(d - np.median(d)))

    # v1's ROIs were 20 wide, cut from the same 21: 3 pixels and a half-pixel
    # convention apart
    assert spread(theta[:, model.slot(0)] + o - P[0]) < 0.01
    assert spread(theta[:, model.slot(1)] + o - P[1]) < 0.01
    assert spread(theta[:, model.slot(4)] - P[4]) < 0.05           # z, in slices
    phase = np.angle(np.exp(1j * (theta[:, model.slot(5)] - P[5])))
    assert spread(phase) < 0.01 and np.max(np.abs(phase)) < 0.2
    # v1 normalised its model to two photons
    assert np.median(theta[:, model.slot(2)] / P[2]) == pytest.approx(2, rel=0.03)


def test_v1s_beads_measure_the_fringe_period_and_the_mirror_steps():
    """Each bead was scanned in 50 nm steps at three phases of the deformable
    mirrors, [2, 0, -2] rad as set: the fitted phase gives back both the
    steps and the fringe period, which v1 had been told (260 nm)."""
    import h5py
    path = _v1_model()
    cal = fourpi.load_uipsf_v1(path, transpose=False)
    with h5py.File(path, "r") as f:
        data = f["rois/psf_data"][...]
    C, nb, nph, nz, size, _ = data.shape
    o = (size - 15) // 2
    rois = np.moveaxis(data[..., o:o + 15, o:o + 15], 0, 3).reshape(-1, C, 15, 15)
    values = Global4PiPSF(cal).unpack(Global4PiPSF(cal).fit(rois.astype(np.float32)))
    phase = values["phase_rad"].reshape(nb, nph, nz)
    stage = np.arange(nz) * 50.0
    slopes = [np.polyfit(stage[3:-3], np.unwrap(phase[b, 0, 3:-3]), 1)[0] for b in range(nb)]
    assert 2 * np.pi / abs(np.median(slopes)) == pytest.approx(258.5, abs=3)
    steps = np.angle(np.exp(1j * (phase[:, 1:] - phase[:, :1])))
    assert np.median(steps[:, 0]) == pytest.approx(2.0, abs=0.15)
    assert np.median(steps[:, 1]) == pytest.approx(-2.0, abs=0.15)
