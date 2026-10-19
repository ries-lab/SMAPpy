import numpy as np
import pytest
import torch

from uipsf.data.io import BeadStack, objective_positions
from uipsf.noise import Camera


def test_photons_and_read_noise_are_smappys_for_the_same_adu():
    metadata = pytest.importorskip("smappy.metadata")
    adu = np.array([[398.6, 410.0, 1000.0]], np.float32)
    for kw, smappy_kw in [(dict(conversion=4.7, offset=398.6), dict(conversion=4.7, offset=398.6)),
                          (dict(conversion=0.5, offset=100.0, read_noise_e=1.6),
                           dict(conversion=0.5, offset=100.0, read_noise_e=1.6)),
                          (dict(conversion=10.0, offset=100.0, em_gain=150.0),
                           dict(conversion=10.0, offset=100.0, em_on=True, emgain=150.0))]:
        ours = Camera(**kw)
        theirs = metadata.CameraMetadata(**smappy_kw)
        expected = (adu - theirs.offset) * theirs.adu_to_photons
        assert np.allclose(ours.to_photons(torch.tensor(adu)).numpy(), expected, rtol=1e-6)
        assert ours.read_noise_var == pytest.approx(theirs.readout_variance)
        assert ours.excess == theirs.excess_noise


def test_a_stack_is_ordered_by_increasing_objective_position_and_marks_saturation():
    planes = [{"ZPositionUm": z} for z in (3.0, 2.95, 2.9)]
    z = objective_positions(planes, {}, 3)
    assert np.allclose(z, [3000, 2950, 2900])
    assert np.allclose(objective_positions([{}] * 3, {"z-step_um": 0.05}, 3), [0, 50, 100])
    with pytest.raises(ValueError, match="z spacing"):
        objective_positions([{}] * 3, {}, 3)
    adu = np.full((2, 3, 3), 500, np.uint16)
    adu[1, 1, 1] = 65535
    st = BeadStack.from_adu(adu, [0.0, 50.0], Camera(conversion=2.0, offset=400.0))
    assert st.saturated.sum() == 1 and st.saturated[1, 1, 1]
    assert np.allclose(st.photons[0], 200.0) and st.read_noise_var == 1.0


def test_mirroring_flips_the_stack_for_data_from_the_other_em_port():
    photons = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    st = BeadStack(photons, np.array([0.0, 50.0]), saturated=photons > 20)
    x = st.mirrored("x")
    assert np.array_equal(x.photons, photons[:, :, ::-1]) and np.array_equal(x.saturated, photons[:, :, ::-1] > 20)
    assert np.array_equal(st.mirrored("y").photons, photons[:, ::-1])
    assert st.mirrored(None) is st
    with pytest.raises(ValueError):
        st.mirrored("z")
