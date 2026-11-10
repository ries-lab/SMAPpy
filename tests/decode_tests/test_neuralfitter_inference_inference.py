
import pytest
import torch

from decode_smlm.generic import emitter
from decode_smlm.generic.process import Identity
from decode_smlm.neuralfitter import post_processing
from decode_smlm.neuralfitter.inference import inference
from decode_smlm.utils.hardware import default_device

GPU = default_device()  # CUDA, else Apple's GPU, else 'cpu': DECODE tested CUDA only


class TestInfer:

    @pytest.fixture()
    def model(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.parameter.Parameter(torch.rand(1, 1, 1, 1))

            def forward(self, x):
                assert x.dim() == 4
                assert x.size(1) == 3

                return self.dummy * torch.rand_like(x[:, [0]])

        return DummyModel()

    @pytest.fixture()
    def infer(self, model):
        return inference.Infer(model=model, ch_in=3, frame_proc=None, post_proc=post_processing.NoPostProcessing(),
                               device=GPU)

    def test_forward_em(self, infer):
        """
        Tests the inference wrapper

        Args:
            infer:

        """

        frames = torch.rand((100, 64, 64))

        em = infer.forward(frames)

        assert isinstance(em, emitter.EmitterSet)

    @pytest.mark.parametrize("batch_size", [16, 'auto'])
    def test_forward_frames(self, infer, batch_size):
        # reinit because now we output frames
        infer = inference.Infer(model=infer.model, batch_size=batch_size, ch_in=3,
                                frame_proc=None, post_proc=Identity(), forward_cat='frames',
                                device=GPU)

        out = infer.forward(torch.rand((100, 64, 64)))

        assert isinstance(out, torch.Tensor)
        assert out.size() == torch.Size((100, 1, 64, 64))

    # CUDA only: on Apple's unified memory the search for the largest batch takes the machine's RAM;
    # web: torch.hub downloads the UNet
    @pytest.mark.web
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA.")
    def test_get_max_batch_size(self, infer):
        infer.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                     in_channels=3, out_channels=1, init_features=32, pretrained=False)

        bs = infer.get_max_batch_size(infer.model.cuda(), (3, 256, 256), 1, 5024)

        assert 16 <= bs <= 1024
