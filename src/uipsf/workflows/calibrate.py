"""A bead calibration in one call: the model built from the optics and a few choices (plan §7.5).

`learn_beads` takes a model a caller has composed component by component;
SMAPpy's *PSF calibration* window, the CLI and a notebook want to say
"this microscope, these choices" and get a `Calibration`.  `BeadSettings`
is those choices and `calibrate_beads` the call.

**The settings are SMAPpy's `param` fields**, so the window builds its form
from them as it does a plugin's; `smappy.plugins` imports no Qt, and uipsf
may use smappy, never its GUI (src/uipsf/CLAUDE.md).

**The model options are choices, not flags** (*magnitude*, *blur*): they
are shown while what they decide is under investigation (plan §10, 2 and
3), and a choice added there, or one taken out once the investigation
settles it, is a line here.  Everything not shown is the default decided
in M7 (`beads.py`), and the instrument -- NA, indices, wavelength, pixel
size -- is not a setting at all: it arrives as `Optics`, from SMAPpy's
microscope database, so that no form offers a default for it.

**Starting from a previous model** (`start_values`): its coefficients go
where the new pupil has the same Noll modes, so a model learnt with
another order or another magnitude choice still starts the new one; what
the new model fixes (a fixed blur) keeps the new settings' value.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from smappy.plugins import param  # GUI-free: the form's metadata, as a plugin's

from ..model import Channel, ForwardModel, VectorPSF, ZernikePupil
from ..model.channels import prefix
from ..optics import PupilGrid, choose_pupil_size
from ..optics.zernike import noll_to_nm, zernike_modes
from .beads import Calibration, learn_beads
from .rejection import RejectionSettings

#: the order a calibration learns up to unless told: n_max 6, with 8 the held-out check (plan §10)
AUTO_ORDER = 6

MAGNITUDES = (("free", "free"), ("symmetric", "symmetric"), ("fixed", "fixed (apodisation)"))
BLURS = (("free", "free"), ("fixed", "fixed"))


@dataclass
class BeadSettings:
    """The choices a bead calibration exposes; the instrument comes as `Optics`."""
    zernike_order: int | None = param(
        None, label="Zernike order", min=2, max=12,
        help=f"the highest radial order of the pupil's modes; auto: {AUTO_ORDER}")
    magnitude: str = param(
        "free", label="pupil magnitude", choices=MAGNITUDES,
        help="free: every mode up to the order; symmetric: the rotationally symmetric ones; "
             "fixed: uniform, the apodisation only")
    blur: str = param("free", label="blur", choices=BLURS,
                      help="an extra Gaussian blur of the image, learnt or held at its value")
    blur_nm: float = param(30.0, label="blur σ", unit="nm", min=0.0,
                           help="its start when free, its value when fixed")
    bead_diameter_nm: float = param(0.0, label="bead diameter", unit="nm", min=0.0,
                                    help="the beads' size, taken out of the model; 0: point-like")
    rejection_k: float = param(3.0, label="rejection (k)", min=1.0, advanced=True,
                               help="robust standard deviations above the median at which a bead is "
                                    "rejected")
    heldout: float = param(0.0, label="held out", min=0.0, max=0.5, advanced=True,
                           help="the fraction of beads kept out of learning and localised for the "
                                "held-out report")

    def __post_init__(self):
        if self.magnitude not in dict(MAGNITUDES):
            raise ValueError(f"magnitude must be one of {[k for k, _ in MAGNITUDES]}, got {self.magnitude!r}")
        if self.blur not in dict(BLURS):
            raise ValueError(f"blur must be one of {[k for k, _ in BLURS]}, got {self.blur!r}")

    @property
    def order(self) -> int:
        return AUTO_ORDER if self.zernike_order is None else int(self.zernike_order)


def magnitude_modes(n_max: int, kind: str) -> list[int]:
    """The Noll modes of the pupil magnitude for ``kind`` (`MAGNITUDES`): no piston, which the PSF's
    energy normalisation fixes (`model.pupil`)."""
    every = zernike_modes(n_max, exclude=("piston",))
    if kind == "free":
        return every
    if kind == "symmetric":
        return [j for j in every if noll_to_nm(j)[1] == 0]
    if kind == "fixed":
        return []
    raise ValueError(f"no magnitude choice {kind!r}")


def bead_model(optics, settings: BeadSettings, roi_shape, transforms=None, z_max_nm: float = 0.0,
               device=None, dtype=torch.float32) -> ForwardModel:
    """The model a bead calibration learns: one channel per `Optics` in ``optics`` (one, or a list, the
    main channel first), each with its own pupil, started at ``transforms`` (secondary → main chip
    pixels; SMAPpy's dual calibration).  The pupil sampling keeps defocus up to ``z_max_nm`` from
    aliasing (`choose_pupil_size`, never below 64)."""
    optics = [optics] if not isinstance(optics, (list, tuple)) else list(optics)
    transforms = list(transforms) if transforms is not None else [None] * len(optics)
    if len(transforms) != len(optics):
        raise ValueError(f"{len(optics)} channels but {len(transforms)} transforms")
    n_max = settings.order
    channels = []
    for o, H in zip(optics, transforms, strict=True):
        grid = PupilGrid.build(o, choose_pupil_size(o, z_max_nm), device=device, dtype=dtype)
        pupil = ZernikePupil(grid, n_max=n_max, magnitude_modes=magnitude_modes(n_max, settings.magnitude))
        psf = VectorPSF(grid, pupil, tuple(roi_shape), bead_radius_nm=settings.bead_diameter_nm / 2,
                        sigma_nm=(settings.blur_nm, settings.blur_nm), fit_sigma=settings.blur == "free")
        channels.append(Channel(psf, transform=H))
    return ForwardModel(channels)


def start_values(model: ForwardModel, previous) -> dict:
    """Shared start values for ``model`` from ``previous`` (a `LearnedPSF` or a `Calibration`).

    Zernike coefficients go to the same Noll modes, zero where the previous model had none; the blur,
    the channel transforms, shares and offsets are taken when the model fits them; anything else (a
    fixed blur, the stage's terms) keeps the model's own start."""
    shared = previous.shared() if callable(getattr(previous, "shared", None)) else previous.shared
    old_model = previous.model
    if old_model.n_channels != model.n_channels:
        raise ValueError(f"the previous model has {old_model.n_channels} channels, this one {model.n_channels}")
    out = {}
    for c, (new, old) in enumerate(zip(model.channels, old_model.channels, strict=True)):
        pre = prefix(c)
        for which in ("phase", "magnitude"):
            name = pre + f"pupil.{which}"
            if not isinstance(new.psf.pupil, ZernikePupil) or name not in shared:
                continue
            new_modes = getattr(new.psf.pupil, f"{which}_modes")
            old_modes = getattr(old.psf.pupil, f"{which}_modes")
            coeff = torch.as_tensor(shared[name]).detach().cpu().double()
            value = torch.zeros(len(new_modes), dtype=torch.float64)
            for i, j in enumerate(new_modes):
                if j in old_modes:
                    value[i] = coeff[old_modes.index(j)]
            out[name] = value
        if new.psf.fit_sigma and pre + "psf.sigma" in shared:
            out[pre + "psf.sigma"] = shared[pre + "psf.sigma"]
        if c > 0:
            for n in ("share", "dz") + (("transform",) if new.fit_transform else ()):
                if pre + n in shared:
                    out[pre + n] = shared[pre + n]
    return {n: torch.as_tensor(v).detach().cpu() for n, v in out.items()}


def calibrate_beads(batch, optics, settings: BeadSettings | None = None, *, transforms=None,
                    geometry: dict | None = None, start=None, noise=None, decisions: dict | None = None,
                    warm_start: Calibration | None = None, progress=None, cancel=None, device=None,
                    dtype=torch.float32) -> Calibration:
    """A `Calibration` learnt from the bead ROIs in ``batch``, for the microscope ``optics``.

    ``batch`` holds every channel's ROIs ([E, C, K, Y, X], photons; SMAPpy's `calibrate.pupil` makes
    it); ``optics`` one `Optics` per channel; ``transforms`` each secondary channel's starting map to
    the main one, ``geometry`` the split as SMAPpy records it.  ``start``, a previous model (a
    `LearnedPSF` or `Calibration` of the same microscope), replaces the first stage
    (`learn_beads`).  ``warm_start`` is the window's *Recalculate*: the same beads, the previous fit
    of this calibration.  ``decisions``, ``progress`` and ``cancel`` as in `learn_beads`."""
    settings = settings if settings is not None else BeadSettings()
    if warm_start is not None:
        model = warm_start.model
    else:
        z_max = float(batch.z_nm.abs().max()) if batch.z_nm.numel() else 0.0
        model = bead_model(optics, settings, tuple(batch.data.shape[-2:]), transforms, z_max, device, dtype)
    if model.n_channels != batch.data.shape[1]:
        raise ValueError(f"{model.n_channels} channels of optics for a batch of {batch.data.shape[1]}")
    values = None if start is None or warm_start is not None else start_values(model, start)
    rejection = RejectionSettings(k=settings.rejection_k)
    cal = learn_beads(model, batch, noise, decisions=decisions, warm_start=warm_start, rejection=rejection,
                      heldout=settings.heldout, progress=progress, cancel=cancel, geometry=geometry,
                      start=values)
    cal.options = settings
    return cal
