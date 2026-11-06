"""A confocal raster scan: the excitation beam times the detection through a pinhole (C1 of
docs/uipsf/scanned-psfs.md §9, its S5).

    μ(r_s) = N · I_exc(r_e − r_s) · D(r_s − r_e) + b,      D(r) = ∫_pinhole I_em(r + r_ph + ρ) d²ρ.

I_exc is the beam an emitter at r_e sees when the beam is at r_s (`ScannedBeamPSF`, point-reflected); D is the
share of the emitter's collected photons that pass the pinhole: the emission PSF I_em (a widefield `VectorPSF` at
the emission wavelength, integrating to one photon over the image plane) summed over the pinhole's disc,
back-projected into the sample and descanned, so centred on the beam's focus, ``r_ph`` off it when misaligned.
The camera image is not mirrored (an image is drawn where the emitter is), so D is the widefield image of an
emitter at r_e − r_ph convolved with the disc.  Its size is typed in Airy units, 1.22 λ_em / NA (SimuFLUX's
unit); its offset is held unless freed, a diagnostic as a mask's centre is.

**One objective, two wavelengths.**  Excitation and detection pass the same objective, whose aberrations are a
difference of optical path, not of phase: the pupil learnt is the excitation's (rad at λ_exc), and the detection
sees the same map scaled by λ_exc / λ_em (`_SharedPath`).  The pupil coordinates are in units of the NA at both
wavelengths, so the two grids sample the same points.  A phase mask is in the excitation path only.  Paths that
differ after the dichroic are not modelled: scanned-psfs.md §4 keeps them for when the data ask.

**Normalisation.**  The excitation's (a flat beam integrates to one over the focal plane) times a fraction: a
pinhole wide open passes everything collected, D → 1, and the scan is S3's open detector; a point pinhole of
radius R passes π R² I_em(r), the classic confocal product I_exc·I_em.  Both limits are tested.

**The sample.**  A rotating dye absorbs |E|² and emits as the unpolarised widefield PSF: exact.  A fluorescent
bead is taken as the open detector's weights on the excitation times the unpolarised detection, which drops the
correlation of each fixed dye's absorption and emission dipoles -- small next to what a pinhole changes, and
exact for a dye.  Gold scatters coherently behind a pinhole, |Σ_d E_exc,d E_det,d|², and is refused here.

**A bead's size** is the product convolved with the sphere's projection, per plane (the thin-bead
approximation of `model.scanned`, exact while the beam changes little over the bead's depth: 20-40 nm beads).

**Derivatives** go through the whole model (`linearisable` False): the solver's fast pupil tangents assume an
image linear in one pupil field's |·|², and a confocal image is a product of two at different wavelengths.  A
product-rule linearisation is the speed-up to make if fits get slow.
"""
from __future__ import annotations

import dataclasses
import math

import torch

from ..optics.grid import PupilGrid
from ..optics.kernels import sphere_otf
from ..params import Bounded, ParamSpec
from .excitation import ExcitationBeam
from .psf import VectorPSF
from .scanned import ScannedBeamPSF

#: an Airy unit in λ_em / NA: the diameter of the Airy disc's first dark ring (SimuFLUX's)
AIRY = 1.22
#: how far the emission PSF's tails reach beyond the pinhole's rim into the FFT's margin, nm; its rings carry
#: ~1e-3 of the light there, which the wrap of a smaller margin would fold back onto the scan
TAIL_NM = 400.0


class _SharedPath:
    """The detection's view of the objective's pupil: the same optical path, its phase scaled to λ_em, and
    neither the mask (the excitation's) nor parameters of its own."""

    def __init__(self, pupil, factor: float):
        self.pupil, self.factor = pupil, float(factor)

    def __call__(self, p):
        magnitude, phase = self.pupil(p)
        return magnitude, phase * self.factor

    def specs(self):
        return []

    def describe(self, p) -> dict:
        return {}


class ConfocalPSF:
    #: the solver differentiates it through the whole model (module docstring)
    linearisable = False

    def __init__(self, beam: ExcitationBeam, roi_shape: tuple[int, int], *, emission_nm: float,
                 pinhole_au: float = 1.0, free: tuple[str, ...] = (), offset_nm: tuple[float, float] = (0.0, 0.0),
                 sample: str = "dye", bead_radius_nm: float = 0.0):
        """``beam`` the excitation (its grid's optics carry the scan step as the pixel size); ``roi_shape`` (Y, X)
        scan positions; ``emission_nm`` the detected wavelength; ``pinhole_au`` the pinhole's diameter in Airy
        units of λ_em / NA, back-projected into the sample; ``offset_nm`` its centre off the beam's focus, held
        unless ``free`` names "offset".  ``sample`` "dye" or "bead" (module docstring)."""
        if sample == "gold":
            raise ValueError("gold scatters coherently behind a pinhole: not modelled yet (scanned-psfs.md §3)")
        if set(free) - {"offset"}:
            raise ValueError(f"a pinhole can learn its offset, not {sorted(set(free) - {'offset'})}")
        if pinhole_au <= 0:
            raise ValueError("the pinhole needs a positive diameter")
        g = beam.grid
        o = g.optics
        self.beam, self.grid, self.sample = beam, g, sample
        self.roi_shape = (int(roi_shape[0]), int(roi_shape[1]))
        self.upsample = 1
        self.bead_radius_nm = float(bead_radius_nm)
        self.emission_nm, self.pinhole_au = float(emission_nm), float(pinhole_au)
        self.radius_nm = AIRY * self.emission_nm / o.NA * self.pinhole_au / 2
        self.free, self.offset_init = tuple(free), tuple(float(v) for v in offset_nm)
        self.spacing = tuple(float(s) for s in o.pixel_size_nm)
        step = min(self.spacing)
        # margins: the bead's around the product, the pinhole's and the PSF's tails around the detection
        self.bead_margin = math.ceil(self.bead_radius_nm / step) + 1 if self.bead_radius_nm > 0 else 0
        self.pinhole_margin = math.ceil((self.radius_nm + TAIL_NM) / step)
        Y, X = self.roi_shape
        mb, mp = self.bead_margin, self.pinhole_margin
        self.excitation = ScannedBeamPSF(beam, (Y + 2 * mb, X + 2 * mb), sample=sample)
        em_grid = PupilGrid.build(dataclasses.replace(o, wavelength_nm=self.emission_nm), g.u.shape[-1],
                                  device=g.device, dtype=g.dtype)
        shared = _SharedPath(beam.pupil, o.wavelength_nm / self.emission_nm)
        self.detection = VectorPSF(em_grid, shared, (Y + 2 * (mb + mp), X + 2 * (mb + mp)), fit_sigma=False)
        self._disc = self._disc_otf()

    def _disc_otf(self) -> torch.Tensor:
        """[Y, X/2+1]: the disc's transform per pixel area, rfft order -- Σ_j img_j·[|r − r_j| ≤ R] by FFT."""
        Y, X = self.detection.roi_shape
        qx = torch.fft.rfftfreq(X, d=self.spacing[0], dtype=torch.float64)
        qy = torch.fft.fftfreq(Y, d=self.spacing[1], dtype=torch.float64)
        t = 2 * math.pi * self.radius_nm * torch.sqrt(qx[None, :] ** 2 + qy[:, None] ** 2)
        small = t < 1e-6
        ts = torch.where(small, torch.ones_like(t), t)
        jinc = torch.where(small, torch.ones_like(t), 2 * torch.special.bessel_j1(ts) / ts)
        area = math.pi * self.radius_nm ** 2 / (self.spacing[0] * self.spacing[1])
        return (area * jinc).to(device=self.grid.device, dtype=self.grid.dtype)

    def specs(self) -> list[ParamSpec]:
        return self.beam.specs() + [ParamSpec("pinhole.offset", (2,), init=torch.tensor(self.offset_init),
                                              transform=Bounded(-1000.0, 1000.0), fixed="offset" not in self.free)]

    @property
    def pupil(self):
        return self.beam.pupil

    def describe(self, p) -> dict:
        out = dict(self.excitation.describe(p), modality="confocal", emission_nm=self.emission_nm,
                   pinhole_au=self.pinhole_au)
        if "pinhole.offset" in p:
            out["pinhole_offset_nm"] = [round(float(v), 2) for v in p["pinhole.offset"].detach().cpu()]
        return out

    def pinhole(self, p, xyz, depth_nm=None) -> torch.Tensor:
        """[E, K, Y', X']: D, the share of the collected photons through the pinhole, on the scan with the bead's
        margin."""
        off = p.get("pinhole.offset")
        pos = xyz if off is None else torch.cat([xyz[..., :2] - off.to(xyz.dtype), xyz[..., 2:]], -1)
        img = self.detection(_unblurred(p, xyz), pos, depth_nm)
        Y, X = img.shape[-2:]
        D = torch.fft.irfft2(torch.fft.rfft2(img) * self._disc, s=(Y, X))
        m = self.pinhole_margin
        return D[..., m:Y - m, m:X - m]

    def __call__(self, p, xyz: torch.Tensor, depth_nm: torch.Tensor | None = None,
                 z_step_nm: float | None = None) -> torch.Tensor:
        """[E, K, Y, X]: expected photons per scan position for N = 1 (module docstring)."""
        mu = self.excitation(_unblurred(p, xyz), xyz, depth_nm) * self.pinhole(p, xyz, depth_nm)
        if self.bead_radius_nm <= 0:
            return mu
        Y, X = mu.shape[-2:]
        H = sphere_otf((1, Y, X), (*self.spacing, 1.0), self.bead_radius_nm, device=mu.device, dtype=mu.dtype)[0]
        mu = torch.fft.irfft2(torch.fft.rfft2(mu) * H[..., :X // 2 + 1], s=(Y, X))
        m = self.bead_margin
        return mu[..., m:Y - m, m:X - m]


def _unblurred(p, xyz):
    """``p`` with the parts' blur at zero: `VectorPSF`'s is a camera's (pixels, drift), and a confocal scan's
    jitter would be one blur of the product, not of each part."""
    return p if "psf.sigma" in p else dict(p, **{"psf.sigma": xyz.new_zeros(2)})
