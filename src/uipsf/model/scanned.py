"""A beam scanned over an emitter behind an open detector: the scan image of a bead or a molecule (S3 of
docs/uipsf/scanned-psfs.md).

With the detector open (no pinhole, or one much wider than the beam's image) what is collected does not depend
on where the beam is, so the count at scan position r_s is the excitation the emitter absorbs there:

    μ(r_s) = N · I(r_e − r_s) + b,

I the focused beam's intensity (`ExcitationBeam`) weighted per field component by what the sample absorbs and
the detector collects (`sample_weights`), N the photons, b the background.  The scan's step is the "pixel size"
(``optics.pixel_size_nm``), and every count is a point sample: nothing is integrated over a pixel.

**A widefield PSF with the beam's spectrum mirrored.**  The scan image is the beam's intensity point-reflected
about the emitter, I(r_e − r_s), not I(r_s − r_e) as a camera images a PSF.  E(r_e − r_s) = Σ_k s(k)
exp(−2πi k·(r_s − r_e)) = Σ_k s(−k) exp(2πi k·(r_s − r_e)), so it is `VectorPSF`'s image of the spectrum
mirrored through the pupil's centre, s(−k): the grid is symmetric about k = 0, so the mirror is a flip of both
axes, and k_z is even, so the defocus is unchanged.  Everything downstream of the pupil -- the matrix DFT that
places the emitter, the z of a bead stack, the forward model, the solver's tangents, the CRLBs -- is then the
widefield code's, unchanged: this class replaces only the six dipole fields by the beam's three field
components and the pupil by the beam's.  The mirror is why a scan of a decentred mask shows the mask's offset
the other way from a camera image of it, and why a scanned image of coma points the other way.

**What the pupil holds** (`ExcitationBeam`): the objective's pupil and its mask, and the input beam's Gaussian;
the polarisation's Jones vector sits in the three fields.  Pupil and mask parameters reach the image through
`pupil_field` and get the solver's fast tangents; the beam's own (``beam.*``, held unless freed) through the
whole model, as a PSF's blur does.

**Normalisation** is the beam's: N·Σ_pixels I = N·(the share of a flat beam's power through a pixel), so N is the
counts a flat x-polarised beam would give an emitter integrated over a scan of the whole focal plane at this
step -- a brightness × power, shared by every pattern from one laser.  A dye integrates to the same N in every
pattern only where the patterns carry the same power (scanned-psfs.md §3, "Normalisation").

**A bead's size.**  In a z stack, the sphere is convolved in 3-D as for widefield beads (`VectorPSF._image`).  In a
single plane it is convolved with the sphere's projection, the 2-D kernel at zero axial frequency: exact while the
beam changes little over the bead's depth, which for 20-40 nm beads it does (the depth of focus is ~600 nm).
"""
from __future__ import annotations

import math

import torch

from ..optics.kernels import gaussian_otf, sphere_otf
from ..optics.mdft import MatrixDFT
from ..optics.vectorial import dipole_fields
from .excitation import ExcitationBeam
from .psf import VectorPSF

#: what scatters or fluoresces under the beam, by the name a model gives (`sample_weights`)
SAMPLES = ("dye", "bead", "gold")


def collection(grid) -> torch.Tensor:
    """(η_x, η_y, η_z): the power an open detector collects from a dipole along x, y and z, from the detection's
    dipole fields on ``grid`` (both analysers, the aperture's, apodised as detection is) -- a z dipole's is a
    fraction of an in-plane one's (0.38 at NA 1.4 in oil, most of it supercritical)."""
    h = grid.aperture * grid.apodisation * dipole_fields(grid, None)               # [6, P, P]
    e = (h.abs() ** 2).sum((-2, -1))
    return (e[:3] + e[3:]).real


def sample_weights(sample: str, grid=None) -> torch.Tensor:
    """(w_x, w_y, w_z), w_x = 1: what an emitter makes of the beam's three field components, Σ_d w_d |E_d|².

    - *dye*: a freely rotating molecule absorbs |E|² and emits whatever it absorbed: (1, 1, 1).
    - *gold*: a gold bead's induced dipole is α·E, and the open detector collects η_d of a dipole along d, the
      cross-terms vanishing over a round aperture: w_d = η_d.  At the excitation's wavelength (``grid`` the
      beam's).
    - *bead*: a fluorescent bead is many dyes fixed in random orientations μ, each absorbing |μ·E|² and emitting
      η(μ) = Σ_k μ_k² η_k.  Averaged over the sphere, ⟨μ_i μ_j μ_k²⟩ = δ_ij (1 + 2δ_ik)/15, so the signal is
      [(Σ_k η_k)|E|² + 2 Σ_d η_d |E_d|²]/15: w_d = Σ_k η_k + 2η_d.  At the emission's wavelength (``grid`` the
      detection's).  Between dye and gold: E_z's weight is 0.85 of the in-plane ones' where gold's is 0.65
      (NA 1.4, oil).

    E_z is what fills a vortex's zero when the circular polarisation's handedness is wrong, so the three see
    that zero differently (scanned-psfs.md §3)."""
    if sample not in SAMPLES:
        raise ValueError(f"sample must be one of {SAMPLES}, got {sample!r}")
    if sample == "dye":
        return torch.ones(3, dtype=torch.float64)
    eta = collection(grid).to(torch.float64)
    w = eta if sample == "gold" else eta.sum() + 2 * eta
    return w / w[0]


class ScannedBeamPSF(VectorPSF):
    def __init__(self, beam: ExcitationBeam, roi_shape: tuple[int, int], *, sample: str = "dye", weights=None,
                 emission_grid=None, bead_radius_nm: float = 0.0, sigma_nm: tuple[float, float] = (0.0, 0.0),
                 fit_sigma: bool = False):
        """``beam`` the excitation (`ExcitationBeam`: grid, the objective's pupil with its mask, the input beam);
        ``roi_shape`` (Y, X) scan positions, ``beam.grid.optics.pixel_size_nm`` apart.  ``sample`` one of
        `SAMPLES`, or ``weights`` (w_x, w_y, w_z) given; a fluorescent bead's weights need the detection's grid at
        the emission wavelength, ``emission_grid`` (the beam's if not given).  ``bead_radius_nm`` a bead's size;
        ``sigma_nm`` a Gaussian blur (scanner jitter, a finite pinhole), held at zero by default."""
        grid = beam.grid
        super().__init__(grid, beam.pupil, roi_shape, upsample=1, kind="vector", analyser="unpolarised",
                         bead_radius_nm=bead_radius_nm, sigma_nm=sigma_nm, fit_sigma=fit_sigma)
        self.beam, self.sample = beam, sample
        if weights is None:
            weights = sample_weights(sample, emission_grid if sample == "bead" and emission_grid is not None
                                     else grid)
        self.sample_weights = torch.as_tensor(weights, dtype=torch.float64)
        self.amplitudes = torch.sqrt(self.sample_weights).to(device=grid.device, dtype=grid.dtype)[:, None, None]
        start = {s.name: s.init for s in beam.specs() if s.name.startswith("beam.")}
        self._set_fields(self._beam_fields(start))
        # The beam's normalisation (`ExcitationBeam`): Σ_aperture |h_x|², not the fields' own energy, so that the
        # weights and the polarisation change the image's height as they change what the emitter absorbs
        self.energy = beam.reference
        self.norm = self.spacing[0] * self.spacing[1] / (self.energy * self.dft.scale)
        # A bead is convolved by FFT, which wraps the scan's edges into each other, and a donut is still bright at
        # the edge of a scan: so the scan is computed with a margin of the bead's radius (and the blur's 3σ) and
        # cropped (`_image`) -- without it the edges of a 100 nm bead's scan were 6 % of the peak off
        self.margin = 0
        if self.bead_radius_nm > 0:
            reach = self.bead_radius_nm + 3 * max(self.sigma_init)
            self.margin = math.ceil(reach / min(self.spacing)) + 1
            Y, X = self.roi_shape
            self.dft = MatrixDFT(grid, (Y + 2 * self.margin, X + 2 * self.margin), self.spacing)

    def _beam_fields(self, p) -> torch.Tensor:
        """[3, P, P]: the polarisation's (E_x, E_y, E_z) per pupil sample, each times √w_d, mirrored (k → −k)."""
        J = self.beam.jones({k: torch.as_tensor(p[k]) for k in ("beam.angle", "beam.ellipticity") if k in p})
        F = (J[0] * self.beam.hx + J[1] * self.beam.hy) * self.amplitudes
        return F.flip(-2, -1)

    def _set_fields(self, F):
        self.fields = F
        Ft = F.permute(1, 0, 2)
        self._fields_re = torch.cat([Ft.real, Ft.imag], 0).contiguous()
        self._fields_im = torch.cat([-Ft.imag, Ft.real], 0).contiguous()
        self.weight = (F.abs() ** 2).sum(0)

    def specs(self):
        return self.beam.specs() + [s for s in super().specs() if s.name == "psf.sigma"]

    def describe(self, p) -> dict:
        b = self.beam.beam
        out = dict(self.pupil.describe(p), wavelength_nm=float(self.grid.optics.wavelength_nm),
                   NA=float(self.grid.optics.NA), n_immersion=float(self.grid.optics.n_imm),
                   n_medium=float(self.grid.optics.n_med), modality="scanned beam, open detector",
                   sample=self.sample, sample_weights=[round(float(w), 4) for w in self.sample_weights],
                   polarisation=b.polarisation)
        for name in ("beam.angle", "beam.ellipticity", "beam.fill", "beam.centre"):
            if name in p:
                v = p[name].detach().cpu()
                out[name.split(".")[1]] = [round(float(x), 4) for x in v.reshape(-1)] if v.dim() else round(float(v), 4)
        return out

    def medium(self, p):
        if "optics.n_med" in p:
            raise ValueError("a scanned beam does not learn the sample's index yet")
        # from p every time, held or free: a held ellipticity given another value (the other handedness) is
        # honoured, and three [P, P] fields cost nothing beside the DFTs
        if "beam.angle" in p:
            self._set_fields(self._beam_fields(p))
        return self.fields, self._fields_re, self._fields_im, self.weight, self.grid.kz_med

    def pupil_field(self, p) -> torch.Tensor:
        """The beam's pupil -- the objective (magnitude rescaled), its mask and the input Gaussian -- mirrored
        (module docstring).  Not rescaled to the fields' energy as a widefield pupil is: what a mask scatters or
        an overfilled aperture clips is lost from the focus."""
        return (self.beam.objective(p) * self.beam.amplitude(p)).flip(-2, -1)

    def resized(self, roi_shape=None, upsample=None, *, bead_radius_nm=None, pupil=None) -> ScannedBeamPSF:
        if upsample not in (None, 1):
            raise ValueError("a scan is point samples: no upsampling")
        beam = self.beam if pupil is None else ExcitationBeam(self.grid, pupil, self.beam.beam)
        roi = self.roi_shape if roi_shape is None else (
            (int(roi_shape), int(roi_shape)) if isinstance(roi_shape, int) else tuple(int(n) for n in roi_shape))
        return ScannedBeamPSF(beam, roi, sample=self.sample, weights=self.sample_weights,
                              bead_radius_nm=self.bead_radius_nm if bead_radius_nm is None else bead_radius_nm,
                              sigma_nm=self.sigma_init, fit_sigma=self.fit_sigma)

    def for_emitters(self, roi_shape=None, upsample=None, height_nm=None) -> ScannedBeamPSF:
        return self.resized(roi_shape, upsample, bead_radius_nm=0.0)

    def _stack(self, xyz, z_step_nm, depth_nm=None):
        # one plane, or planes that are not a stack: the bead is its projection, per plane (module docstring)
        if self.bead_radius_nm > 0 and (z_step_nm is None or xyz.shape[1] < 2):
            return xyz, 0, None
        return super()._stack(xyz, z_step_nm, depth_nm)

    def _image(self, I, sigma, pad, dz):
        if self.bead_radius_nm <= 0:
            return super()._image(I, sigma, pad, dz)
        if dz is not None:                                  # a z stack: the sphere in 3-D, as for widefield beads
            out = super()._image(I, sigma, pad, dz)
        else:                                               # one plane: the sphere's projection
            H = gaussian_otf(I.shape[-2:], self.spacing, sigma) * sphere_otf(
                (1, *I.shape[-2:]), (*self.spacing, 1.0), self.bead_radius_nm, device=I.device, dtype=I.dtype)[0]
            X = I.shape[-1]
            out = torch.fft.irfft2(torch.fft.rfft2(I) * H[..., :X // 2 + 1], s=I.shape[-2:])
        m = self.margin
        return out[..., m:out.shape[-2] - m, m:out.shape[-1] - m]
