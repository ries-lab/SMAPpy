import pytest
import torch

from decode_smlm.generic import emitter, test_utils
from decode_smlm.neuralfitter import post_processing


class TestPostProcessingAbstract:

    @pytest.fixture(params=["batch-set", "frame-set"], scope="function")
    def post(self, request):
        class PostProcessingMock(post_processing.PostProcessing):
            def forward(self):
                return emitter.EmptyEmitterSet()

        return PostProcessingMock(xy_unit=None, px_size=None, return_format=request.param)

    @pytest.mark.parametrize("return_format", [None, 'batch_set', 'frame_set', 'emitters'])
    def test_sanity(self, post, return_format):
        """
        Tests the sanity checks

        """
        with pytest.raises(ValueError):
            post.return_format = return_format
            post.sanity_check()

    def test_filter(self, post):
        assert not post.skip_if(torch.rand((1, 3, 32, 32)))


class TestNoPostProcessing(TestPostProcessingAbstract):

    @pytest.fixture()
    def post(self):
        return post_processing.NoPostProcessing()

    def test_forward(self, post):
        out = post.forward(torch.rand((256, 2, 64, 64)))
        assert isinstance(out, emitter.EmptyEmitterSet)


class TestLookUpPostProcessing(TestPostProcessingAbstract):

    @pytest.fixture()
    def post(self):
        return post_processing.LookUpPostProcessing(raw_th=0.1, xy_unit='px')

    @pytest.fixture()
    def pseudo_out_no_sigma(self):
        """Pseudo model output without sigma prediction"""

        detection = torch.tensor([[0.1, 0.0], [0.6, 0.05]]).unsqueeze(0).unsqueeze(0)
        features = torch.tensor([[1., 2.], [3., 4.]]).unsqueeze(0).unsqueeze(0).repeat(1, 5, 1, 1)
        features = features * torch.tensor([1., 2., 3., 4., 5.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        pseudo_net_ouput = torch.cat((detection, features), 1)

        return pseudo_net_ouput

    @pytest.fixture()
    def pseudo_out(self):
        """Pseudo model output with sigma prediction"""

        detection = torch.tensor([[0.1, 0.0], [0.6, 0.05]]).unsqueeze(0).unsqueeze(0)
        features = torch.tensor([[1., 2.], [3., 4.]]).unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1)
        features = features * torch.tensor([1., 2., 3., 4.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        sigma = torch.ones((1, 4, 2, 2))
        sigma *= torch.arange(1, 5).view(1, -1, 1, 1)
        sigma /= detection

        pseudo_net_ouput = torch.cat((detection, features, sigma, torch.rand_like(detection)), 1)

        return pseudo_net_ouput

    def test_filter(self, post):
        """Setup"""
        detection = torch.tensor([[0.1, 0.0], [0.6, 0.05]]).unsqueeze(0)

        """Run"""
        active_px = post._filter(detection)

        """Assertions"""
        assert (active_px == torch.tensor([[1, 0], [1, 0]]).unsqueeze(0).bool()).all()

        return active_px

    def test_lookup(self, post):
        """Setup"""
        active_px = self.test_filter(post)  # get the return value of the previous test

        features = torch.tensor([[1., 2.], [3., 4.]]).unsqueeze(0).unsqueeze(0).repeat(1, 5, 1, 1)
        features = features * torch.tensor([1., 2., 3., 4., 5.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        """Run"""
        batch_ix, features = post._lookup_features(features, active_px)

        """Assertions"""
        assert isinstance(batch_ix, torch.LongTensor), "Batch ix should be integer type."
        assert (batch_ix == 0).all()
        assert batch_ix.size()[0] == features.size()[1]

        # This is hard coded designed for the very specific test case
        assert ((features / (torch.arange(5).unsqueeze(1).float() + 1)).unique() == torch.tensor([1., 3.])).all()

    def test_forward_no_sigma(self, post, pseudo_out_no_sigma):
        """Setup"""
        post.photxyz_sigma_mapping = None  # because this test is without sigma (this is non-default)

        """Run"""
        emitter_out = post.forward(pseudo_out_no_sigma)

        """Assert"""
        assert isinstance(emitter_out, emitter.EmitterSet), "Output should be an emitter."
        assert (emitter_out.frame_ix == 0).all()
        assert (emitter_out.phot.unique() == torch.tensor([1., 3.])).all()

    def test_forward(self, post, pseudo_out):
        """Run"""
        emitter_out = post.forward(pseudo_out)

        """Assert"""
        assert isinstance(emitter_out, emitter.EmitterSet), "Output should be an emitter."
        assert (emitter_out.frame_ix == 0).all()
        assert (emitter_out.phot.unique() == torch.tensor([1., 3.])).all()

        assert not torch.isnan(emitter_out.xyz_sig).any(), "Sigma values for xyz should not be nan."
        assert not torch.isnan(emitter_out.phot_sig).any(), "Sigma values for phot should not be nan."
        assert torch.isnan(emitter_out.bg_sig).all()

        assert test_utils.tens_almeq(emitter_out.xyz_sig,
                                     torch.tensor([[20., 30., 40.], [2 / 0.6, 3 / 0.6, 4 / 0.6]]))

        assert test_utils.tens_almeq(emitter_out.phot_sig, torch.tensor([10., 1 / 0.6]))


class TestSpatialIntegration(TestLookUpPostProcessing):

    @pytest.fixture()
    def post(self):
        return post_processing.SpatialIntegration(raw_th=0.1, xy_unit='px')

    def test_forward_no_sigma(self, post, pseudo_out_no_sigma):
        """Setup"""
        post.photxyz_sigma_mapping = None  # because this test is without sigma (this is non-default)

        """Run"""
        emitter_out = post.forward(pseudo_out_no_sigma)

        """Assert"""
        assert isinstance(emitter_out, emitter.EmitterSet), "Output should be an emitter."
        assert len(emitter_out) == 1
        assert emitter_out.frame_ix == 0
        assert emitter_out.phot == pytest.approx(3.)
        assert emitter_out.prob == pytest.approx(0.75)

    def test_forward(self, post, pseudo_out):
        """Run"""
        emitter_out = post.forward(pseudo_out)

        """Assert"""
        assert isinstance(emitter_out, emitter.EmitterSet)
        assert len(emitter_out) == 1

        assert not torch.isnan(emitter_out.xyz_sig).any(), "Sigma values for xyz should not be nan."
        assert not torch.isnan(emitter_out.phot_sig).any(), "Sigma values for phot should not be nan."
        assert torch.isnan(emitter_out.bg_sig).all()

        assert test_utils.tens_almeq(emitter_out.xyz_sig,
                                     torch.tensor([[2 / 0.6, 3 / 0.6, 4 / 0.6]]))

        assert test_utils.tens_almeq(emitter_out.phot_sig, torch.tensor([1 / 0.6]))

    @pytest.mark.parametrize("aggr,expct", [('sum', ([0., 1.000001], [0., 0.501])),
                                            ('norm_sum', ([0., 1.], [0., 0.501]))
                                            ])
    def test_nms(self, post, aggr, expct):
        """Setup, Run, Assert"""
        post.p_aggregation = post.set_p_aggregation(aggr)

        p = torch.zeros((2, 32, 32))
        p[0, 4:6, 4] = 0.5
        p[0, 4, 4] = 0.5
        p[0, 5, 4] = 0.500001
        p[1, 6, 4] = 0.25
        p[1, 7, 4] = 0.251

        p_out = post._nms(p, post.p_aggregation, 0.2, 0.6)

        """Assertions"""
        assert test_utils.tens_almeq(p_out[0, 4:6, 4], torch.tensor(expct[0]))
        assert test_utils.tens_almeq(p_out[1, 6:8, 4], torch.tensor(expct[1]))
