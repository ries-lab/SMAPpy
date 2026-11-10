from abc import ABC, abstractmethod

import pytest
import torch

import decode_smlm.simulation.psf_kernel as psf_kernel

# DECODE's TestCubicSplinePSF tested its `spline` package on a downloaded bead calibration;
# the torch renderer's tests, against the v0.10.2 oracle, come with milestone (c).


psf_cuda_available = pytest.mark.skipif(not psf_kernel.CubicSplinePSF.cuda_is_available(),
                                        reason="Skipped because cuda not available for Spline PSF.")


class AbstractPSFTest(ABC):

    @abstractmethod
    @pytest.fixture()
    def psf(self):
        raise NotImplementedError

    def test_forward_indices(self, psf):
        """Tests whether the correct amount of frames are returned"""

        # No Emitters but indices
        out = psf.forward(torch.zeros((0, 3)), torch.ones(0), torch.zeros(0).long(), -5, 5)
        assert out.size(0) == 11

        # Emitters but off indices
        out = psf.forward(torch.zeros((10, 3)), torch.ones(10), torch.zeros(10).long(), 5, 10)
        assert out.size(0) == 6

        # Emitters and matching indices
        out = psf.forward(torch.zeros((10, 3)), torch.ones(10), torch.zeros(10).long(), -5, 5)
        assert out.size(0) == 11

    def test_forward_frame_index(self, psf):
        """
        Tests whether the correct frames are active, i.e. signal present.

        Args:
            abs_psf: fixture

        """
        """Setup"""
        xyz = torch.Tensor([[15., 15., 0.], [0.5, 0.3, 0.]])
        phot = torch.ones_like(xyz[:, 0])
        frame_ix = torch.Tensor([0, 0]).int()

        """Run"""
        frames = psf.forward(xyz, phot, frame_ix, -1, 1)

        """Asserts"""
        assert frames.size() == torch.Size([3, 64, 64]), "Wrong dimensions."
        assert (frames[[0, -1]] == 0.).all(), "Wrong frames active."
        assert (frames[1] != 0).any(), "Wrong frames active."
        assert frames[1].max() > 0, "Wrong frames active."


class TestSingleFrameImplementedPSF(AbstractPSFTest):

    @pytest.fixture()
    def psf(self):
        """
        Abstract PSF class fixture

        Returns:
            psf_kernel.PSF: abstract PSF
        """

        class PseudoAbsPSF(psf_kernel.PSF):
            def _forward_single_frame(self, xyz: torch.Tensor, weight: torch.Tensor):
                if xyz.numel() >= 1:
                    return torch.ones((64, 64)) * xyz[0, 0]
                else:
                    return torch.zeros((64, 64))

            def forward(self, xyz: torch.Tensor, weight: torch.Tensor, frame_ix: torch.Tensor, ix_low, ix_high):
                xyz, weight, frame_ix, ix_low, ix_high = super().forward(xyz, weight, frame_ix, ix_low, ix_high)
                return self._forward_single_frame_wrapper(xyz, weight, frame_ix, ix_low, ix_high)

        return PseudoAbsPSF(None, None, None, None)

    def test_single_frame_wrapper(self, psf):
        """
        Tests whether the general impl
        Args:
            abs_psf:

        Returns:

        """

        xyz = torch.Tensor([[15., 15., 0.], [0.5, 0.3, 0.]])
        phot = torch.ones_like(xyz[:, 0])
        frame_ix = torch.Tensor([1, -2]).int()

        frames = psf._forward_single_frame_wrapper(xyz, phot, frame_ix, -2, 1)

        assert frames.dim() == 3
        assert (frames[0] == 0.5).all()
        assert (frames[1:3] == 0).all()
        assert (frames[-1] == 15).all()


class TestDeltaPSF:

    @pytest.fixture()
    def delta_1px(self):
        return psf_kernel.DeltaPSF(xextent=(-0.5, 31.5),
                                   yextent=(-0.5, 31.5),
                                   img_shape=(32, 32))

    @pytest.fixture()
    def delta_05px(self):
        return psf_kernel.DeltaPSF(xextent=(-0.5, 31.5),
                                   yextent=(-0.5, 31.5),
                                   img_shape=(64, 64))

    def test_bin_ctr(self, delta_1px, delta_05px):

        """Assert"""
        assert (delta_1px._bin_ctr_x == torch.arange(32).float()).all()
        assert (delta_1px._bin_ctr_y == torch.arange(32).float()).all()

        assert (delta_05px._bin_ctr_x == torch.arange(64).float() / 2 - 0.25).all()
        assert (delta_05px._bin_ctr_y == torch.arange(64).float() / 2 - 0.25).all()

    def test_px_search(self, delta_05px):

        """Setup"""
        xyz = torch.tensor([[-0.6, -0.5, 0.],  # outside
                            [-0.5, -0.5, 0.],  # just inside
                            [20., 30., 500.],  # inside
                            [31.4, 31.49, 0],  # just inside
                            [31.5, 31.49, 0.],  # just outside (open interval to the right)
                            [50., 60., 0.]])  # clearly outside

        xtar = torch.tensor([0, 41, 63])  # elmnts 1 2 3
        ytar = torch.tensor([0, 61, 63])

        """Run"""
        with pytest.raises(ValueError):
            delta_05px.search_bin_index(xyz[:2])
        with pytest.raises(ValueError):
            delta_05px.search_bin_index(xyz[-3:-1])
        with pytest.raises(ValueError):
            delta_05px.search_bin_index(xyz[[-3, -1]])

        x, y = delta_05px.search_bin_index(xyz[1:4])

        """Assert"""
        assert isinstance(x, torch.LongTensor)
        assert isinstance(y, torch.LongTensor)
        assert (x == xtar).all()
        assert (y == ytar).all()

    def test_forward(self, delta_1px, delta_05px):
        """
        Tests the implementation
        Args:
            delta_1px:
            delta_05px:

        Returns:

        """

        # ToDo: Not such a nice test ...

        xyz = torch.Tensor([[0., 0., 0.], [15.0, 15.0, 200.], [4.8, 4.8, 200.], [4.79, 4.79, 500.]])
        phot = torch.ones_like(xyz[:, 0])
        frame_ix = torch.tensor([1, 0, 1, 1])

        """Run"""
        out_1px = delta_1px.forward(xyz, phot, frame_ix, 0, 2)
        out_05px = delta_05px.forward(xyz, phot, frame_ix, 0, 2)

        """Test"""
        assert out_1px.size() == torch.Size([3, 32, 32]), "Wrong output shape."
        assert out_05px.size() == torch.Size([3, 64, 64]), "Wrong output shape."

        assert out_1px[0, 15, 15] == 1.
        assert out_05px[0, 31, 31] == 1.

        assert out_1px[1, 0, 0] == 1.
        assert out_05px[1, 1, 1] == 1.

        assert (out_1px.unique() == torch.Tensor([0., 1.])).all()
        assert (out_05px.unique() == torch.Tensor([0., 1.])).all()

    def test_forward_border(self, delta_1px):
        _ = delta_1px.forward(torch.zeros((0, 3)), torch.zeros((0,)), torch.zeros(0).long(), 0, 0)

    def test_doubles(self, delta_1px):
        frames = delta_1px.forward(torch.zeros((2, 3)), torch.tensor([1., 2.]), torch.zeros(2).long(), 0, 0)

        """Assert"""
        assert frames[0, 0, 0] in (1., 2.)


class TestGaussianExpect(TestSingleFrameImplementedPSF):

    @pytest.fixture(scope='class', params=[None, (-5000., 5000.)])
    def psf(self, request):
        return psf_kernel.GaussianPSF((-0.5, 63.5), (-0.5, 63.5), request.param, img_shape=(64, 64), sigma_0=1.5)

    def test_norm(self, psf):
        xyz = torch.tensor([[32., 32., 0.]])
        phot = torch.tensor([1.])
        assert pytest.approx(psf.forward(xyz, phot).sum().item(), 0.05) == 1

    def test_peak_weight(self, psf):
        psf.peak_weight = True

        xyz = torch.tensor([[32., 32., 0.]])
        phot = torch.tensor([1.])
        assert pytest.approx(psf.forward(xyz, phot).max().item(), 0.05) == 1

    def test_single_frame_wrapper(self, psf):
        """
        Tests whether the general impl
        Args:
            abs_psf:

        Returns:

        """

        xyz = torch.Tensor([[15., 15., 0.], [0.5, 0.3, 0.]])
        phot = torch.ones_like(xyz[:, 0])
        frame_ix = torch.Tensor([1, -2]).int()

        frames = psf._forward_single_frame_wrapper(xyz, phot, frame_ix, -2, 1)

        assert frames.dim() == 3
        assert (frames[0] != 0).any()
        assert (frames[1:3] == 0).all()
        assert (frames[-1] != 0).any()
