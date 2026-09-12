"""Scientific contracts for bead calibration and acquisition assembly."""
import numpy as np
import pytest
from scipy import ndimage

from smappy.calibrate import (BeadStack, CalibrationSettings, collect_beads,
                              build_calibration, discover_acquisitions, read_bead_stacks)
from smappy.calibrate.core import (BeadCollection, estimate_shift, robust_shape_error,
                                   spline_coefficients)
from smappy.calibrate.input import _assemble
from smappy.calibrate.gui import _stack_contrast, _stack_slice
from smappy.io.calibration import (SplineCalibration, evaluate_spline,
                                   load_spline_calibration, save_spline_calibration)


def psf(nz=41, size=17):
    z, y, x = np.mgrid[:nz, :size, :size].astype(float)
    t = (z-(nz-1)/2)/12
    sx = 1.2*np.sqrt(1+(t-.6)**2)
    sy = 1.2*np.sqrt(1+(t+.6)**2)
    a = np.exp(-.5*(((x-size//2)/sx)**2+((y-size//2)/sy)**2))/(2*np.pi*sx*sy)
    return a


def test_coefficients_preserve_asymmetric_polynomial_and_axis_order():
    z, y, x = np.mgrid[:7, :9, :11].astype(float)
    a = 1 + .02*x**3 + .03*y**2 + .1*z + .007*x*y*z
    c = spline_coefficients(a)
    np.testing.assert_allclose(c[0], a[:-1, :-1, :-1], rtol=1e-6)
    zz, yy, xx = 2.3, 3.2, 4.7
    from smappy.io.calibration import compute_delta3d
    got = np.dot(c[:, 2, 3, 4], compute_delta3d(xx-4, yy-3, zz-2))
    assert got == pytest.approx(1+.02*xx**3+.03*yy**2+.1*zz+.007*xx*yy*zz, rel=1e-6)


def test_subpixel_registration_recovers_known_shift():
    a = psf(61, 23)
    shift = np.array([3.4, -.7, .45])
    b = ndimage.shift(a, shift, order=3)
    got = estimate_shift(a, b, [6, 2, 2], z_window=35)
    np.testing.assert_allclose(got, -shift, atol=.12)


def test_shape_error_ignores_scale_but_detects_psf_distortion():
    reference = psf(41, 17)
    same_shape = 2.7 * reference
    distorted = same_shape.copy()
    distorted[:, :, 9:] = ndimage.shift(distorted[:, :, 9:], (3, 0, 0), order=1)
    same_error, same_correlation = robust_shape_error(same_shape, reference)
    distorted_error, distorted_correlation = robust_shape_error(distorted, reference)
    assert same_error < 1e-10
    assert same_correlation == pytest.approx(1.)
    assert distorted_error > same_error + .02
    assert distorted_correlation < same_correlation


def test_stack_browser_orientations_preserve_zyx_axes():
    volume = np.arange(4*5*6).reshape(4, 5, 6)
    np.testing.assert_array_equal(_stack_slice(volume, 'XY', 2), volume[2, :, :])
    np.testing.assert_array_equal(_stack_slice(volume, 'XZ', 3), volume[:, 3, :])
    np.testing.assert_array_equal(_stack_slice(volume, 'YZ', 4), volume[:, :, 4])
    image = np.array([[-2., 1.], [3., 8.]])
    assert _stack_contrast(volume, image, True) == (0., 8.)
    assert _stack_contrast(volume, image, False)[1] > 8


def test_z_groups_are_sorted_and_repeats_not_concatenated():
    records = []
    for t in (0, 1):
        for i in (2, 0, 3, 1):
            records.append((np.full((9, 9), i), {'time': t, 'channel': 0, 'z': i},
                            {'ZPositionUm': 4-i*.01}))
    stacks = _assemble(records, {}, 'test', None)
    assert len(stacks) == 2
    np.testing.assert_allclose(stacks[0].images[:, 0, 0], [3, 2, 1, 0])
    assert stacks[0].dz == pytest.approx(10)
    with pytest.raises(ValueError, match='duplicate'):
        _assemble(records+records[:1], {}, 'test', None)
    with pytest.raises(ValueError, match='spacing'):
        _assemble([(a, b, {}) for a, b, _ in records], {}, 'test', None)


def test_native_roundtrip_and_fitter_knot_convention(tmp_path):
    a = psf()
    cal = SplineCalibration(spline_coefficients(a), 10., 20., psf=a,
                            em_mirror=False, parameters={'settings': {'smooth_z_nm': 20}})
    path = tmp_path/'cal.h5'
    save_spline_calibration(path, cal)
    loaded = load_spline_calibration(path)
    np.testing.assert_array_equal(loaded.coeff, cal.coeff)
    assert loaded.parameters == cal.parameters
    assert loaded.z_index_to_nm(21) == -10
    # Fitter ROI is two pixels smaller than knots: one-pixel offset each side.
    rendered = evaluate_spline(loaded, 7, 7, 20., 15)
    np.testing.assert_allclose(rendered, a[20, 1:16, 1:16], rtol=1e-6)
    with pytest.raises(FileExistsError):
        save_spline_calibration(path, cal)


def synthetic_collection():
    rng = np.random.default_rng(11)
    images = []
    base = psf(61, 23)
    for i, delta in enumerate([-2., 0., 2., 1.]):
        frame = np.full((61, 49, 49), 100.)
        v = ndimage.shift(base, (delta, .12*i, -.1*i), order=3)
        frame[:, 13:36, 13:36] += 30000*v
        images.append(BeadStack(rng.poisson(frame).astype(np.float32), np.arange(61)*10.,
                                source=f'synthetic{i}', axes={'channel': 0}))
    settings = CalibrationSettings(roi_size=15, padding=4, max_xy_shift_px=2.,
                    min_distance_px=17, max_z_shift_nm=50, alignment_range_nm=300,
                    smooth_z_nm=10, registration_iterations=2)
    return collect_beads(images, settings)


def test_end_to_end_build_exclude_and_fit(tmp_path):
    beads = synthetic_collection()
    assert len(beads.records) == 4
    result = build_calibration(beads, excluded=[3])
    assert result.reasons[3] == 'manual exclusion'
    assert result.accepted.sum() == 3
    # True relative axial shifts of the three calibration beads.
    np.testing.assert_allclose(result.shifts[:3, 0], [2, 0, -2], atol=.5)
    assert result.calibration.psf.sum(axis=(1, 2)).max() == pytest.approx(1.)
    path = tmp_path/'result.h5'
    result.save(path)
    cal = load_spline_calibration(path)
    from smappy.psf import SplinePSF
    roi = evaluate_spline(cal, 5.2, 4.8, cal.z0+3.2, 11)
    fit = SplinePSF(cal).fit(np.array([roi*20000+10], dtype=np.float32), iterations=100)
    np.testing.assert_allclose(fit.theta[0, [0, 1, 4]], [5.2, 4.8, cal.z0+3.2], atol=.12)
    from smappy.calibrate.validation import fit_bead_diagnostics, aligned_midline_profiles
    check = fit_bead_diagnostics(result, bead_ids=[3])
    assert -1 in check['bead_id']  # separately fitted average bead stack
    assert check['fit_valid'].all()
    assert np.median(abs(check['centered_error_nm'])) < 10
    assert np.all(abs(check['expected_z_nm']) < cal.z0*cal.dz)
    profiles = aligned_midline_profiles(result, bead_ids=[0, 1, 2])
    assert profiles['x_profiles'].shape == (3, result.raw_psf.shape[2])
    assert profiles['z_profiles'].shape == (3, result.raw_psf.shape[0])
    np.testing.assert_allclose(profiles['x_profiles'].max(axis=1), 1)
    np.testing.assert_allclose(profiles['z_profiles'].max(axis=1), 1)


def test_full_z_search_then_limit_rejection_does_not_bias_template():
    rng = np.random.default_rng(17)
    base = psf(61, 23)
    stacks = []
    true_offsets = [-2., 0., 2., 12.]
    for i, delta in enumerate(true_offsets):
        frame = np.full((61, 49, 49), 100.)
        frame[:, 13:36, 13:36] += 30000*ndimage.shift(base, (delta, 0, 0), order=3)
        stacks.append(BeadStack(rng.poisson(frame).astype(np.float32), np.arange(61)*10.,
                                source=f'z-shift-{i}', axes={'channel': 0}))
    settings = CalibrationSettings(roi_size=15, padding=4, max_xy_shift_px=2,
                                   min_distance_px=17, max_z_shift_nm=50,
                                   alignment_range_nm=300, registration_iterations=2)
    result = build_calibration(collect_beads(stacks, settings))
    assert result.reasons[3] == 'z shift outlier'
    assert not result.accepted[3]
    # The broad search recovers the outlier rather than clipping it at 5 planes.
    assert abs(result.coarse_shifts[3, 0]) > settings.max_z_shift_nm/10
    # Accepted relative shifts retain the consensus established by the three
    # central beads; the absolute registration origin is intentionally arbitrary.
    centered = result.shifts[:3, 0]-np.median(result.shifts[:3, 0])
    np.testing.assert_allclose(centered, [2, 0, -2], atol=.6)


def test_full_xy_search_uses_radial_limit_only_after_registration():
    base = psf(61, 23)
    offsets = [(0, 0, 0), (1, .1, 0), (-1, -.1, 0), (0, 4.2, 0)]
    volumes = np.stack([ndimage.shift(base, offset, order=3) for offset in offsets]).astype(np.float32)
    records = [{'brightness_accepted': True, 'brightness_reason': 'accepted',
                'brightness_adu': 1., 'background_adu': 0.} for _ in offsets]
    settings = CalibrationSettings(roi_size=15, padding=4, max_xy_shift_px=2,
                                   max_z_shift_nm=50, alignment_range_nm=300)
    beads = BeadCollection(volumes, records, [], [], 10., settings)
    result = build_calibration(beads)
    assert result.reasons[3] == 'xy shift outlier'
    assert np.hypot(*result.coarse_shifts[3, 1:]) > settings.max_xy_shift_px
    assert result.accepted.sum() == 3


def test_discovery_deduplicates_acquisitions(tmp_path):
    a = tmp_path/'a'; a.mkdir()
    for n in ('image.ome.tif', 'image_1.ome.tif'):
        (a/n).touch()
    b = tmp_path/'b'; b.mkdir()
    (b/'NDTiff.index').touch(); (b/'NDTiffStack.tif').touch()
    found = discover_acquisitions([tmp_path, a, a/'image_1.ome.tif', b])
    assert found == [a/'image.ome.tif', b]


def test_ome_tiff_reads_embedded_plane_coordinates(tmp_path):
    import json
    import tifffile
    path = tmp_path/'beads.ome.tif'
    with tifffile.TiffWriter(path, ome=False) as writer:
        for z in (2, 0, 3, 1):
            md = json.dumps({'SliceIndex': z, 'FrameIndex': 0, 'ChannelIndex': 0,
                             'PositionIndex': 0, 'ZPositionUm': 1+z*.02})
            writer.write(np.full((9, 9), z, np.uint16), metadata=None,
                         extratags=[(51123, 's', len(md)+1, md, False)])
    stacks = read_bead_stacks(path)
    assert len(stacks) == 1
    np.testing.assert_allclose(stacks[0].images[:, 0, 0], np.arange(4))
    assert stacks[0].dz == pytest.approx(20)


def test_ndtiff_preserves_interleaved_channels_and_repeats(tmp_path):
    import json
    import struct
    from smappy.io.ndtiff import SUMMARY_OFFSET, SUMMARY_MARKER
    folder = tmp_path/'nd'; folder.mkdir()
    summary = json.dumps({'z-step_um': .01}).encode()
    blob = bytearray(b'\0'*SUMMARY_OFFSET)
    blob += struct.pack('<II', SUMMARY_MARKER, len(summary))+summary
    index = bytearray()
    name = 'NDTiffStack.tif'
    for z in (3, 0, 2, 1):
        for t in (1, 0):
            for c in (0, 1):
                axes = json.dumps({'time': t, 'channel': c, 'z': z}).encode()
                pixels = np.full((9, 9), 100*t+10*c+z, np.uint16).tobytes()
                md = json.dumps({'ZPositionUm': 3+z*.01}).encode()
                offset = len(blob)
                blob += pixels+md
                index += struct.pack('<I', len(axes))+axes
                index += struct.pack('<I', len(name))+name.encode()
                index += struct.pack('<8I', offset, 9, 9, 1, 0, offset+len(pixels), len(md), 0)
    (folder/name).write_bytes(blob)
    (folder/'NDTiff.index').write_bytes(index)
    stacks = read_bead_stacks(folder)
    assert len(stacks) == 4
    for s in stacks:
        np.testing.assert_array_equal(s.images[:, 0, 0],
            np.arange(4)+100*s.axes['time']+10*s.axes['channel'])
    with pytest.raises(ValueError, match='multiple channels'):
        collect_beads(stacks)


def test_invalid_stack_and_settings_fail_explicitly():
    defaults = CalibrationSettings()
    assert defaults.max_z_shift_nm == 250
    assert defaults.min_beads == 1
    assert defaults.brightness_range == 6
    assert defaults.min_distance_px == 25
    assert defaults.alignment_range_nm == 500
    assert defaults.rejection_mad == 1.5
    with pytest.raises(ValueError, match='padding'):
        CalibrationSettings(padding=3).validate()
    with pytest.raises(ValueError, match='odd'):
        CalibrationSettings(roi_size=14).validate()
    with pytest.raises(ValueError, match='brightness_range'):
        CalibrationSettings(brightness_range=.5).validate()
    records = [(np.zeros((9, 9)), {'z': i}, {'ZPositionUm': z})
               for i, z in enumerate([1, 1.01, 1.02, 1.1])]
    with pytest.raises(ValueError, match='irregular'):
        _assemble(records, {}, 'test', None)


def test_one_background_per_bead_stack_preserves_axial_variation():
    images = np.broadcast_to(100+2*np.arange(41)[:, None, None], (41, 49, 49)).astype(np.float32).copy()
    spot = psf(41, 17)*30000
    images[:, 16:33, 16:33] += spot
    settings = CalibrationSettings(roi_size=15, padding=4, max_xy_shift_px=2,
                                   min_distance_px=17, min_beads=1)
    beads = collect_beads(BeadStack(images, np.arange(41)*10.), settings)
    assert len(beads.records) == 1
    rec = beads.records[0]
    assert rec['background_adu'] == 140.
    restored = beads.volumes[0]*rec['brightness_adu']+rec['background_adu']
    np.testing.assert_allclose(restored, images[:, 13:36, 13:36], atol=2e-4)
    # The 80 ADU change between first/last plane borders must remain in the volume.
    difference = (beads.volumes[0, -1, 0, 0]-beads.volumes[0, 0, 0, 0])*rec['brightness_adu']
    assert difference == pytest.approx(80., abs=1e-4)


def test_brightness_range_and_integer_saturation_reject_but_retain_beads():
    images = np.full((41, 61, 61), 100., dtype=float)
    bead = psf(41, 17)
    positions = [(15, 15), (15, 45), (45, 15), (45, 45)]
    for (y, x), amplitude in zip(positions, (1000, 4000, 9000, 4000)):
        images[:, y-8:y+9, x-8:x+9] += amplitude*bead
    images[20, 45, 45] = np.iinfo(np.uint16).max
    settings = CalibrationSettings(roi_size=15, padding=4, max_xy_shift_px=2,
                                   min_distance_px=20, brightness_range=4)
    beads = collect_beads(BeadStack(images.astype(np.uint16), np.arange(41)*10.), settings)
    assert len(beads.records) == 4
    reasons = sorted(r['brightness_reason'] for r in beads.records)
    assert reasons == ['accepted', 'saturated', 'too bright', 'too dim']
    result = build_calibration(beads)
    assert result.accepted.sum() == 1
    assert sorted(result.reasons) == reasons
    no_range = CalibrationSettings(roi_size=15, padding=4, max_xy_shift_px=2,
                                   min_distance_px=20, brightness_range=None)
    unfiltered = collect_beads(BeadStack(images.astype(np.uint16), np.arange(41)*10.), no_range)
    assert [r['brightness_reason'] for r in unfiltered.records].count('accepted') == 3
    assert [r['brightness_reason'] for r in unfiltered.records].count('saturated') == 1


def test_calibration_save_defaults_and_single_extension(tmp_path):
    """A calibration names the day, the dataset, the scope and the mode."""
    import re
    from smappy.calibrate.gui import calibration_save_defaults, calibration_save_path
    dataset = tmp_path/'bead_dataset'
    stack = dataset/'Pos0'
    stack.mkdir(parents=True)
    image = stack/'beads.ome.tif'
    image.touch()
    for given in (image, stack):
        folder, name = calibration_save_defaults([given])
        assert folder == dataset
        assert re.fullmatch(r'\d{6}_bead_dataset_3Dcal', name), name
        assert calibration_save_defaults([given], dual=True)[1].endswith('_2Ccal')
    for name in ('result', 'result.h5', 'result.h5.h5', 'result.H5.h5'):
        assert calibration_save_path(dataset/name) == dataset/'result.h5'


def test_a_name_that_already_says_the_date_or_the_microscope_does_not_repeat_it(tmp_path):
    from smappy.calibrate.gui import calibration_save_defaults
    stack = tmp_path/'230501_Ulf_NPC_M5'/'Pos0'
    stack.mkdir(parents=True)
    (stack/'beads.ome.tif').touch()
    assert calibration_save_defaults([stack])[1] == '230501_Ulf_NPC_M5_3Dcal'
    # a microscope named only in the stack folder is carried up into the name
    other = tmp_path/'Ulf_NPC'/'M7_Pos0'
    other.mkdir(parents=True)
    (other/'beads.ome.tif').touch()
    assert calibration_save_defaults([other])[1].endswith('_Ulf_NPC_M7_3Dcal')


def test_save_dialog_normalizes_before_overwrite_check(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tkinter import filedialog, messagebox
    from smappy.calibrate.gui import CalibrationWindow
    stack = tmp_path/'bead_dataset'/'Pos0'
    stack.mkdir(parents=True)
    image = stack/'beads.ome.tif'
    image.touch()
    from smappy.calibrate.gui import calibration_save_defaults
    destination = stack.parent/(calibration_save_defaults([image])[1]+'.h5')
    destination.touch()
    dialogs, saved, confirmations = [], [], []
    def choose(**kwargs):
        dialogs.append(kwargs)
        return str(destination)+'.h5'
    monkeypatch.setattr(filedialog, 'asksaveasfilename', choose)
    monkeypatch.setattr(messagebox, 'askyesno', lambda *args, **kwargs: confirmations.append(args) or True)
    settings = object()
    result = SimpleNamespace(reasons=[], beads=SimpleNamespace(settings=settings),
                             save=lambda path, overwrite: saved.append((path, overwrite)))
    def unexpected_error(exc):
        raise exc
    window = SimpleNamespace(root=None, result=result, excluded=set(), settings=lambda:settings,
        paths=[image], result_paths=[image], status=SimpleNamespace(set=lambda value:None), error=unexpected_error)
    CalibrationWindow.save(window)
    assert dialogs[0]['initialdir'] == str(stack.parent)
    assert dialogs[0]['initialfile'].endswith('bead_dataset_3Dcal')
    assert saved == [(destination, True)]
    assert confirmations[0][1] == str(destination)
