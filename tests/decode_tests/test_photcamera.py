import pytest
import torch

import decode_smlm.simulation.camera as camera
from decode_smlm.generic import test_utils

from decode_smlm.utils.hardware import default_device

GPU = default_device()  # CUDA, else Apple's GPU, else 'cpu': DECODE tested CUDA only


class TestPhoton2Camera:

    @pytest.fixture(scope='class')
    def cam_fix(self):
        return camera.Photon2Camera(qe=1.0, spur_noise=0.002, em_gain=300., e_per_adu=45.,
                                    baseline=100, read_sigma=74.4, photon_units=False)

    def test_shape(self, cam_fix):
        x = torch.rand((32, 3, 64, 64))
        assert x.size() == cam_fix.forward(x).size()

    def test_edge_frames(self, cam_fix):

        cam_fix.forward(torch.zeros((32, 3, 64, 64)))

    def test_photon_units(self, cam_fix):
        cam_fix.photon_units = True

        x = torch.rand((32, 3, 64, 64)) * 2000
        out = cam_fix.forward(x)

        tol = 0.01
        assert abs((x.mean() - out.mean()) / x.mean()) <= tol

    @pytest.mark.skipif(GPU == "cpu", reason="Shipping to a GPU makes only sense if there is one.")
    @pytest.mark.parametrize("input_device", ["cpu", GPU])
    @pytest.mark.parametrize("forward_device", ["cpu", GPU])
    def test_forward_backward_device(self, cam_fix, input_device, forward_device):
        exp_device = torch.rand(1).to(forward_device).device

        assert cam_fix.forward(torch.rand((1, 32, 32)).to(input_device), forward_device).device \
               == exp_device

        assert cam_fix.backward(torch.rand((1, 32, 32)).to(input_device), forward_device).device \
               == exp_device


class TestPerfectCamera(TestPhoton2Camera):

    @pytest.fixture()
    def cam_fix(self):
        return camera.PerfectCamera()

    def test_forward_backward_equal(self, cam_fix):
        x = torch.rand((32, 3, 64, 64)) * 10000

        x_out_cnt = cam_fix.forward(x.clone())
        x_out_phot = cam_fix.backward(x_out_cnt.clone())

        assert test_utils.tens_almeq(x_out_phot, x_out_cnt)


@pytest.mark.skip(reason="Implementation not ready.")
class TestSCMOS(TestPhoton2Camera):

    @pytest.fixture()
    def cam_fix(self):
        read_sigma = torch.meshgrid(torch.arange(256), torch.arange(256))[0]

        return camera.SCMOS(sample_mode='batch', qe=1.0, spur_noise=0.002, em_gain=300., e_per_adu=45.,
                            baseline=100, read_sigma=read_sigma, photon_units=False)
