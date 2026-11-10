import time

import pytest
import torch

import decode_smlm.utils.model_io as io_model


# DECODE took a UNet from torch.hub (a download) and wrote into its test folder; any module
# does for what is tested here, and the files go to pytest's temporary folder.
@pytest.fixture
def unet():
    """Inits an arbitrary network."""
    return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(8, 1, 1))


@pytest.fixture
def model_interface(unet, tmp_path):
    """Inits the i/o interface."""
    return io_model.LoadSaveModel(unet, tmp_path / 'test_load_save.pt', None, 1, 1e-3)


def test_save(model_interface, unet, tmp_path):
    """Tests saving the model, and tests whether a new suffix is appended."""
    model_interface.save(unet, None)
    if not (tmp_path / 'test_load_save_0.pt').is_file():
        pytest.fail("Model could not be found after saving.")

    time.sleep(2)
    model_interface.save(unet, None)
    if not (tmp_path / 'test_load_save_1.pt').is_file():
        pytest.fail("Model could not be found after saving.")


def test_load_init(model_interface, unet, tmp_path):
    model_interface.save(unet, None)
    model_interface.warmstart_file = tmp_path / 'test_load_save_0.pt'
    model_interface.load_init()
