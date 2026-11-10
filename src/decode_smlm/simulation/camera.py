from abc import ABC, abstractmethod  # abstract class
from typing import Union

import torch

from . import noise_distributions


class Camera(ABC):

    @abstractmethod
    def forward(self, x: torch.Tensor, device: Union[str, torch.device] = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def backward(self, x: torch.Tensor, device: Union[str, torch.device] = None) -> torch.Tensor:
        raise NotImplementedError


class Photon2Camera(Camera):
    """
    Simulates a physical EM-CCD camera device. Input are the theoretical photon counts as by the psf and background model,
    all the device specific things are modelled.

    """

    def __init__(self, *, qe: float, spur_noise: float, em_gain: Union[float, None], e_per_adu: float, baseline: float,
                 read_sigma: float, photon_units: bool, device: Union[str, torch.device] = None):
        """

        Args:
            qe: quantum efficiency :math:`0 ... 1'
            spur_noise: spurious noise
            em_gain: em gain
            e_per_adu: electrons per analog digital unit
            baseline: manufacturer baseline / offset
            read_sigma: readout sigma
            photon_units: convert back to photon units
            device: device (cpu / cuda)

        """
        self.qe = qe
        self.spur = spur_noise
        self._em_gain = em_gain
        self.e_per_adu = e_per_adu
        self.baseline = baseline
        self._read_sigma = read_sigma
        self.device = device

        self.poisson = noise_distributions.Poisson()
        self.gain = noise_distributions.Gamma(scale=self._em_gain)
        self.read = noise_distributions.Gaussian(sigma=self._read_sigma)
        self.photon_units = photon_units

    @classmethod
    def parse(cls, param):

        return cls(qe=param.Camera.qe, spur_noise=param.Camera.spur_noise,
                   em_gain=param.Camera.em_gain, e_per_adu=param.Camera.e_per_adu,
                   baseline=param.Camera.baseline, read_sigma=param.Camera.read_sigma,
                   photon_units=param.Camera.convert2photons,
                   device=param.Hardware.device_simulation)

    def __str__(self):
        return f"Photon to Camera Converter.\n" + \
               f"Camera: QE {self.qe} | Spur noise {self.spur} | EM Gain {self._em_gain} | " + \
               f"e_per_adu {self.e_per_adu} | Baseline {self.baseline} | Readnoise {self._read_sigma}\n" + \
               f"Output in Photon units: {self.photon_units}"

    def forward(self, x: torch.Tensor, device: Union[str, torch.device] = None) -> torch.Tensor:
        """
        Forwards frame through camera

        Args:
            x: camera frame of dimension *, H, W
            device: device for forward

        Returns:
            torch.Tensor
        """
        if device is not None:
            x = x.to(device)
        elif self.device is not None:
            x = x.to(self.device)

        """Clamp input to 0."""
        x = torch.clamp(x, 0.)

        """Poisson for photon characteristics of emitter (plus autofluorescence etc."""
        camera = self.poisson.forward(x * self.qe + self.spur)

        """Gamma for EM-Gain (EM-CCD cameras, not sCMOS)"""
        if self._em_gain is not None:
            camera = self.gain.forward(camera)

        """Gaussian for read-noise. Takes camera and adds zero centred gaussian noise."""
        camera = self.read.forward(camera)

        """Electrons per ADU, (floor function)"""
        camera /= self.e_per_adu
        camera = camera.floor()

        """Add Manufacturer baseline. Make sure it's not below 0."""
        camera += self.baseline
        camera = torch.max(camera, torch.tensor([0.]).to(camera.device))

        if self.photon_units:
            return self.backward(camera, device)

        return camera

    def backward(self, x: torch.Tensor, device: Union[str, torch.device] = None) -> torch.Tensor:
        """
        Calculates the expected number of photons from a noisy image.

        Args:
            x:
            device:

        Returns:

        """

        if device is not None:
            x = x.to(device)
        elif self.device is not None:
            x = x.to(self.device)

        out = (x - self.baseline) * self.e_per_adu
        if self._em_gain is not None:
            out /= self._em_gain
        out -= self.spur
        out /= self.qe

        return out


class PerfectCamera(Photon2Camera):
    def __init__(self, device: Union[str, torch.device] = None):
        """
        Convenience wrapper for perfect camera, i.e. only shot noise. By design in 'photon units'.

        Args:
            device: device for simulation

        """
        super().__init__(qe=1.0, spur_noise=0., em_gain=None, e_per_adu=1., baseline=0., read_sigma=0.,
                         photon_units=False, device=device)

    @classmethod
    def parse(cls, param):
        return cls(device=param.Hardware.device_simulation)
