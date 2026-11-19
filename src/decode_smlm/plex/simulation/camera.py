from abc import ABC, abstractmethod
from typing import Union

import torch

from . import noise as noise_distributions


class Camera(ABC):
    def __init__(self, sensor_size: tuple[int, int] | None, device: str | torch.device):
        """
        Abstract base class for camera models.

        Args:
            sensor_size: (max.) sensor size in pixels
            device: device
        """
        super().__init__()

        self.sensor_size = sensor_size
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

    @property
    def device(self):
        return self._device

    @abstractmethod
    def forward(
        self, x: torch.Tensor, device: Union[str, torch.device] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def backward(
        self, x: torch.Tensor, device: Union[str, torch.device] = None
    ) -> torch.Tensor:
        raise NotImplementedError


class CameraNoOp(Camera):
    # a do nothing camera
    def forward(
        self, x: torch.Tensor, device: Union[str, torch.device] = None
    ) -> torch.Tensor:
        return x.to(device)

    def backward(
        self, x: torch.Tensor, device: Union[str, torch.device] = None
    ) -> torch.Tensor:
        return x.to(device)


class CameraEMCCD(Camera):
    def __init__(
        self,
        *,
        qe: float,
        spur_noise: float,
        em_gain: Union[float, None],
        e_per_adu: float,
        baseline: float,
        read_sigma: float,
        photon_units: bool,
        sensor_size: tuple[int, int] | None = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Simulates a physical EM-CCD camera device.
        Inputs are the theoretical photon counts as by the psf and background model,
        all the device specific things are modelled.

        Args:
            qe: quantum efficiency :math:`0 ... 1'
            spur_noise: spurious noise
            em_gain: em gain
            e_per_adu: electrons per analog digital unit
            baseline: manufacturer baseline / offset
            read_sigma: readout sigma
            photon_units: convert back to photon units
            sensor_size: (max.) sensor size in pixels
            device: device
        """
        super().__init__(sensor_size=sensor_size, device=device)
        self.qe = qe
        self.spur = spur_noise
        self._em_gain = em_gain
        self.e_per_adu = e_per_adu
        self.baseline = baseline
        self._read_sigma = read_sigma

        self.poisson = noise_distributions.Poisson()
        self.gain = noise_distributions.Gamma(scale=self._em_gain)
        self.read = noise_distributions.Gaussian(sigma=self._read_sigma)
        self.photon_units = photon_units

    def __str__(self):
        return (
            f"Photon to Camera Converter.\n"
            + f"Camera: QE {self.qe} | Spur noise {self.spur} | EM Gain {self._em_gain} | "
            + f"e_per_adu {self.e_per_adu} | Baseline {self.baseline} | Readnoise {self._read_sigma}\n"
            + f"Output in Photon units: {self.photon_units}"
        )

    def forward(
        self, x: torch.Tensor, device: Union[str, torch.device] = None
    ) -> torch.Tensor:
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

        # clamp input to 0
        x = torch.clamp(x, 0.0)

        # Poisson for photon characteristics of emitter (plus autofluorescence etc
        camera = self.poisson.forward(x * self.qe + self.spur)

        # Gamma for EM-Gain (EM-CCD cameras, not sCMOS)
        if self._em_gain is not None:
            camera = self.gain.forward(camera)

        # Gaussian for read-noise. Takes camera and adds zero centred gaussian noise
        camera = self.read.forward(camera)

        # Electrons per ADU, (floor function)
        camera /= self.e_per_adu
        camera = camera.floor()

        # Add Manufacturer baseline. Make sure it's not below 0
        camera += self.baseline
        camera = torch.max(camera, torch.tensor([0.0]).to(camera.device))

        if self.photon_units:
            return self.backward(camera, device)

        return camera

    def backward(
        self, x: torch.Tensor, device: Union[str, torch.device] = None
    ) -> torch.Tensor:
        """Calculates the expected number of photons from a noisy image."""

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


class CameraSCMOS(CameraEMCCD):
    def __init__(
        self,
        *,
        qe: float,
        spur_noise: float,
        e_per_adu: float,
        baseline: float,
        read_sigma: float,
        photon_units: bool,
        sensor_size: tuple[int, int] | None = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Simulates a sCMOS camera device.
        Inputs are the theoretical photon counts as by the psf and background model,
        all the device specific things are modelled. The only difference to EMCCD is that
        sCMOS does not have EM-Gain.

        Args:
            qe: quantum efficiency :math:`0 ... 1'
            spur_noise: spurious noise
            e_per_adu: electrons per analog digital unit
            baseline: manufacturer baseline / offset
            read_sigma: readout sigma
            photon_units: convert back to photon units
            sensor_size: (max.) sensor size in pixels
            device: device
        """
        super().__init__(
            em_gain=None,
            qe=qe,
            spur_noise=spur_noise,
            e_per_adu=e_per_adu,
            baseline=baseline,
            read_sigma=read_sigma,
            photon_units=photon_units,
            sensor_size=sensor_size,
            device=device,
        )


class CameraPerfect(CameraEMCCD):
    def __init__(
        self,
        sensor_size: tuple[int, int] | None,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Convenience wrapper for perfect camera, i.e. only shot noise.
        By design in 'photon units'.
        """
        super().__init__(
            qe=1.0,
            spur_noise=0.0,
            em_gain=None,
            e_per_adu=1.0,
            baseline=0.0,
            read_sigma=0.0,
            photon_units=True,
            sensor_size=sensor_size,
            device=device,
        )
