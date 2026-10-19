import math

import pytest
import torch

from uipsf.optics.kernels import bin_pixels, gaussian_otf, sphere_otf


def test_the_gaussian_blur_has_the_standard_deviation_it_is_given():
    shape, spacing = (101, 101), (20.0, 25.0)
    H = gaussian_otf(shape, spacing, torch.tensor([60.0, 90.0], dtype=torch.float64))
    delta = torch.zeros(shape, dtype=torch.float64)
    delta[50, 50] = 1
    img = torch.fft.ifft2(torch.fft.fft2(delta) * H).real
    yy, xx = torch.meshgrid(torch.arange(101.0) - 50, torch.arange(101.0) - 50, indexing="ij")
    assert abs(img.sum().item() - 1) < 1e-12
    assert abs(math.sqrt((img * (xx * 20) ** 2).sum()) - 60) < 0.1
    assert abs(math.sqrt((img * (yy * 25) ** 2).sum()) - 90) < 0.1


def test_the_sphere_transfer_function_is_one_at_zero_frequency_and_the_sphere_transform_elsewhere():
    H = sphere_otf((9, 8, 8), (50.0, 50.0, 50.0), 100.0, dtype=torch.float64)
    assert H[0, 0, 0] == 1
    t = 2 * math.pi * 100.0 * (1 / 8 / 50.0)          # one step along x
    assert abs(H[0, 0, 1].item() - 3 * (math.sin(t) - t * math.cos(t)) / t ** 3) < 1e-12
    assert torch.all(sphere_otf((5, 5, 5), (50.0, 50.0, 50.0), 1e-3) > 0.999)


def test_binning_sums_the_upsampled_samples_of_each_pixel():
    img = torch.arange(16.0).reshape(1, 4, 4)
    assert torch.equal(bin_pixels(img, 2), torch.tensor([[[10.0, 18.0], [42.0, 50.0]]]))
    with pytest.raises(ValueError):
        bin_pixels(torch.ones(5, 5), 2)
