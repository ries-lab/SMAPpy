"""The camera's read noise in the fit: its variance added to data and model.

The fitter's likelihood is Poisson.  An sCMOS adds Gaussian read noise of about
one electron per pixel, and the standard remedy (Huang et al. 2013) adds its
variance to both the data and the model -- which also floors every pixel's
weight, 1 / (model + variance).  With the background free, adding it to the
data is all it takes; it is taken back off the fitted background after.
"""
import numpy as np
import pytest
from scipy.special import erf

from smappy.metadata import CameraMetadata
from smappy.pipeline import with_readout, without_readout
from smappy.psf import GaussianPSF, GlobalGaussianPSF


def spots(n=400, photons=3000.0, background=1.0, read_noise=1.0, sz=11, seed=0):
    """Gaussian spots with Poisson noise and Gaussian read noise, in electrons."""
    rng = np.random.default_rng(seed)
    ii = np.arange(sz)
    x0, y0 = (sz - 1) / 2 + rng.uniform(-0.5, 0.5, (2, n))
    norm = np.sqrt(0.5) / 1.3

    def axis(centre):
        return .5 * (erf((ii - centre[:, None] + .5) * norm)
                     - erf((ii - centre[:, None] - .5) * norm))
    mean = background + photons * axis(y0)[:, :, None] * axis(x0)[:, None, :]
    data = rng.poisson(mean) + rng.normal(0, read_noise, mean.shape)
    return data.astype(np.float32), x0


def test_the_read_noise_is_one_electron_without_em_gain_and_none_with_it():
    assert CameraMetadata(em_on=False).readout_variance == 1.0
    assert CameraMetadata().readout_variance == 1.0
    assert CameraMetadata(em_on=True, emgain=100).readout_variance == 0.0
    assert CameraMetadata(em_on=False, read_noise_e=1.4).readout_variance == pytest.approx(1.96)
    assert CameraMetadata(em_on=False, read_noise_e=0.0).readout_variance == 0.0


def test_the_background_comes_back_without_the_variance():
    data, _ = spots(background=5.0)
    model = GaussianPSF(sigma=1.3)
    result = model.fit(with_readout(data, 1.0))
    without_readout(result, model, 1.0)
    assert np.median(model.unpack(result)["background"]) == pytest.approx(5.0, abs=0.15)


def test_a_global_fit_loses_the_variance_from_every_background_it_has():
    data, _ = spots(background=5.0)
    pair = np.ascontiguousarray(np.stack([data, data], axis=1))
    link = np.zeros((len(data), 2, 2, 5), np.float32)
    link[:, 1] = 1.0
    for shared in ((True, True, False, False, False), (True, True, False, True, False)):
        model = GlobalGaussianPSF(sigma=1.3, shared=shared)
        result = model.fit(with_readout(pair, 2.0), link)
        without_readout(result, model, 2.0)
        values = model.unpack(result, link)
        assert np.median(values["background"]) == pytest.approx(5.0, abs=0.15)


def test_read_noise_in_the_data_is_what_the_variance_is_for():
    """At a photon of background the Poisson likelihood cannot take read
    noise -- it treats a negative count as none -- and the background comes
    out high; with the variance it is the background."""
    data, x0 = spots(background=0.5, read_noise=1.0)
    model = GaussianPSF(sigma=1.3)
    plain = model.unpack(model.fit(data))
    result = model.fit(with_readout(data, 1.0))
    without_readout(result, model, 1.0)
    read = model.unpack(result)
    assert abs(np.median(read["background"]) - 0.5) < abs(np.median(plain["background"]) - 0.5)
    assert np.median(read["photons"]) == pytest.approx(3000, rel=0.03)
    assert np.std(read["x_roi"] - x0) <= np.std(plain["x_roi"] - x0) * 1.05
