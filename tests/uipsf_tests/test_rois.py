import numpy as np
import pytest
import torch

from uipsf.data.io import BeadStack
from uipsf.data.rois import Beads, extract_rois, find_beads


def _stack(spots, shape=(120, 140), K=7, seed=0, origin=(10, 20)):
    """Gaussian spots that sharpen towards the middle slice, Poisson noise on a background of 20."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    img = np.full((K, *shape), 20.0)
    for k in range(K):
        sd = 1.5 + 0.4 * abs(k - K // 2)
        for x, y, n in spots:
            img[k] += n / (2 * np.pi * sd ** 2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sd ** 2))
    photons = rng.poisson(img).astype(np.float32)
    return BeadStack(photons, np.arange(K) * 50.0 + 1000.0, read_noise_var=1.0, chip_origin=origin)


def test_isolated_beads_are_found_and_crowded_or_cut_off_ones_are_not():
    spots = [(40, 40, 5000), (100, 60, 3000),          # isolated
             (60, 95, 4000), (68, 97, 4000),           # a pair closer than one ROI: neither kept
             (4, 70, 5000)]                            # its ROI would leave the image
    beads = find_beads([_stack(spots)], roi_size=15)
    found = sorted(zip(beads.x.tolist(), beads.y.tolist(), strict=True))
    assert found == [(40, 40), (100, 60)]


def test_rois_are_centred_on_the_beads_and_carry_z_noise_saturation_and_field_position():
    # By default the first slice is dropped; here every slice is kept to check the z grid.
    st = _stack([(40, 40, 5000)])
    st.saturated = np.zeros(st.photons.shape, bool)
    st.saturated[3, 40, 40] = True
    beads = Beads(np.array([40, 50]), np.array([40, 50]), np.array([0, 1]), np.zeros(2))
    stacks = [st, _stack([(50, 50, 5000)], seed=1, origin=(0, 0))]
    assert extract_rois(stacks, beads, 15).data.shape[2] == 6                 # the first slice dropped
    batch = extract_rois(stacks, beads, 15, slices=slice(None))
    assert batch.data.shape == (2, 1, 7, 15, 15)
    centre = batch.data[0, 0, 3]
    assert torch.argmax(centre) == 7 * 15 + 7                   # the spot's peak on the centre pixel
    assert torch.allclose(batch.z_nm, -(torch.arange(7) * 50.0 - 150.0))   # SMAPpy's sign, about the middle
    assert not batch.mask[0, 0, 3, 7, 7] and batch.mask.sum() == batch.mask.numel() - 1
    assert batch.field_xy[0].tolist() == [50.0, 60.0] and batch.stack_id.tolist() == [0, 1]
    assert batch.s == 1.0
    sub = batch.subset(slice(1, 2))
    assert sub.field_xy.tolist() == [[50.0, 50.0]] and sub.stack_id.tolist() == [1]


def test_thinning_keeps_the_largest_multiple_of_the_step_not_above_the_maximum_and_the_focus():
    import torch

    from uipsf import Batch
    z = Batch.z_from_objective(torch.arange(-1000.0, 1001.0, 20.0))              # 101 slices, 20 nm
    data = torch.arange(101.0)[None, None, :, None, None].expand(2, 1, 101, 3, 3)
    b = Batch(data=data, z_nm=z, mask=torch.ones(2, 1, 101, 3, 3, dtype=torch.bool))
    for max_dz, step in ((10.0, 20.0), (20.0, 20.0), (50.0, 40.0), (59.9, 40.0), (60.0, 60.0), (100.0, 100.0)):
        t = b.thinned(max_dz)
        assert abs(t.z_step_nm - step) < 1e-4, max_dz
        assert 0.0 in t.z_nm.tolist()                                           # the focus is kept
        assert torch.equal(t.data[0, 0, :, 0, 0], -t.z_nm / 20 + 50)            # data and z stay paired
        assert t.mask.shape[2] == t.data.shape[2]
    assert b.thinned(None) is b


def test_bead_finding_is_smappys_convention_in_photons_or_adu():
    # The same beads as SMAPpy's calibration window picks (plan §5.7), whatever the camera's units.
    core = pytest.importorskip("smappy.calibrate.core")
    spots = [(40, 40, 5000), (100, 60, 3000), (60, 95, 4000), (68, 97, 4000), (4, 70, 5000), (120, 100, 900)]
    st = _stack(spots)
    beads = find_beads([st], roi_size=15)
    settings = core.CalibrationSettings(roi_size=15)
    adu = st.photons / 4.7 + 398.6
    yx, _ = core.detect_beads(adu, settings)
    assert sorted(zip(beads.x.tolist(), beads.y.tolist(), strict=True)) == sorted((int(x), int(y)) for y, x in yx)
