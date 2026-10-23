"""A bead calibration in one call: the model built from the optics and a few choices (plan §7.5).

`learn_beads` takes a model a caller has composed component by component;
SMAPpy's *PSF calibration* window, the CLI and a notebook want to say
"this microscope, these choices" and get a `Calibration`.  `BeadSettings`
is those choices and `calibrate_beads` the call.

**The settings are SMAPpy's** (`smappy.calibrate.pupil.BeadSettings`, `param`
fields), so the window builds its form from them as it does a plugin's, and
without torch: every `uipsf` import loads torch through the package.  uipsf
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
the new model fixes (a fixed blur) keeps the new settings' value.  It
starts stage 1 rather than replacing it (`learn_beads(start_stage1=True)`):
on the lab's M2, each acquisition from each other's model, that took 0.5-0.8
of a cold fit's time and always reached the optimum, while replacing stage 1
was slower from a model 0.26 rad off (another camera) and once ended 159 000
nats up (docs/uipsf/architecture.md, "How a bead fit starts").

**A pair's transform starts from the beads' Gaussian positions**
(`model.astig.gaussian_start`), not from the pairing's centroids: 2-3x
closer on simulated pairs (0.11-0.25 px against 0.37-0.59), where a start
2.5 px off already fails on mirrored pairs; on Zenodo 2ch it kept stage 1
from stalling.  It replaces a previous model's transform too, which a
splitter can lose between sessions.  The pairing's stays when too few beads
give positions in both channels or they fit badly (`GAUSS_MIN_PAIRS`,
`GAUSS_MAX_RMS_PX`).
"""
from __future__ import annotations

import torch

# The choices live in smappy, torch-free, so its window builds the form without torch installed.
from smappy.calibrate.pupil import AUTO_ORDER, BLURS, MAGNITUDES, BeadSettings  # noqa: F401

from ..model import Channel, ForwardModel, VectorPSF, ZernikePupil
from ..model.channels import prefix
from ..optics import PupilGrid, choose_pupil_size
from ..optics.zernike import noll_to_nm, zernike_modes
from .beads import Calibration, learn_beads
from .rejection import RejectionSettings

# The Gaussian transform needs enough beads to fix eight entries well, and positions that agree with it:
# simulated and Zenodo pairs fit it to 0.02-0.1 px, so half a pixel means a bead the Gaussians misread.
GAUSS_MIN_PAIRS = 6
GAUSS_MAX_RMS_PX = 0.5


def gaussian_transforms(model, batch) -> dict:
    """Each secondary channel's transform from the beads' Gaussian positions ({c: 3 × 3}), only for the
    channels where it is trustworthy (`GAUSS_MIN_PAIRS`, `GAUSS_MAX_RMS_PX`); {} for one channel."""
    if model.n_channels < 2:
        return {}
    from ..device import on_model_device
    from ..model.astig import gaussian_start
    from .beads import MAX_DZ_NM
    try:
        g = gaussian_start(model, on_model_device(batch, model).thinned(MAX_DZ_NM))
    except (ValueError, RuntimeError):          # a degenerate set of beads: the pairing's stays
        return {}
    return {c: H for c, H in g["transform"].items()
            if g["transform_fit"][c]["pairs"] >= GAUSS_MIN_PAIRS
            and g["transform_fit"][c]["rms_px"] <= GAUSS_MAX_RMS_PX}


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
               device=None, dtype=torch.float32, held_phase: dict | None = None,
               held_magnitude: dict | None = None) -> ForwardModel:
    """The model a bead calibration learns: one channel per `Optics` in ``optics`` (one, or a list, the
    main channel first), each with its own pupil, started at ``transforms`` (secondary → main chip
    pixels; SMAPpy's dual calibration).  The pupil sampling keeps defocus up to ``z_max_nm`` from
    aliasing (`choose_pupil_size`, never below 64).  ``held_phase`` / ``held_magnitude``: modes held at
    given values (`ZernikePupil`)."""
    optics = [optics] if not isinstance(optics, (list, tuple)) else list(optics)
    transforms = list(transforms) if transforms is not None else [None] * len(optics)
    if len(transforms) != len(optics):
        raise ValueError(f"{len(optics)} channels but {len(transforms)} transforms")
    n_max = settings.order
    channels = []
    for o, H in zip(optics, transforms, strict=True):
        grid = PupilGrid.build(o, choose_pupil_size(o, z_max_nm), device=device, dtype=dtype)
        pupil = ZernikePupil(grid, n_max=n_max, magnitude_modes=magnitude_modes(n_max, settings.magnitude),
                             held_phase=held_phase, held_magnitude=held_magnitude)
        psf = VectorPSF(grid, pupil, tuple(roi_shape), bead_radius_nm=settings.bead_diameter_nm / 2,
                        sigma_nm=(settings.blur_nm, settings.blur_nm), fit_sigma=settings.blur == "free")
        channels.append(Channel(psf, transform=H))
    return ForwardModel(channels)


def insitu_model(optics, settings: BeadSettings, roi_shape, focus_nm: float, z_max_nm: float = 1000.0,
                 device=None, dtype=torch.float32, held_phase: dict | None = None,
                 held_magnitude: dict | None = None, fit_focus: bool = True) -> ForwardModel:
    """The model in situ learning learns (plan §9, M5): one channel, the pupil and blur as `bead_model`
    makes them for a point (no bead's size), molecules in the sample seen through ``focus_nm``
    (`InSituGeometry`; held there with ``fit_focus`` off), one photon count each.  ``held_phase`` /
    ``held_magnitude`` hold modes at given values (`ZernikePupil`): a bead model's symmetric magnitude and
    high orders.  The pupil sampling keeps defocus up to ``z_max_nm`` from
    aliasing: the molecules' distance from the plane in focus, not their depth, since the depth's and the
    focus's waves largely cancel there and the light beyond the critical angle decays rather than
    aliases.  ±1 µm needs 64 samples at NA 1.43; 2 µm asked for 96, 2.25 times the work."""
    from dataclasses import replace as replaced

    from ..model import Bleaching, InSituGeometry
    beads = bead_model(optics, replaced(settings, bead_diameter_nm=0.0), roi_shape, z_max_nm=z_max_nm,
                       device=device, dtype=dtype, held_phase=held_phase, held_magnitude=held_magnitude)
    return ForwardModel(beads.channels, geometry=InSituGeometry(focus_nm, fit_focus=fit_focus),
                        intensity=Bleaching(order=0))


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
    the main one (refitted from the beads' Gaussian positions where they allow it: the module
    docstring), ``geometry`` the split as SMAPpy records it.  ``start``, a previous model (a
    `LearnedPSF` or `Calibration` of the same microscope), starts stage 1 (`learn_beads`).
    ``warm_start`` is the window's *Recalculate*: the same beads, the previous fit
    of this calibration.  ``decisions``, ``progress`` and ``cancel`` as in `learn_beads`."""
    settings = settings if settings is not None else BeadSettings()
    if warm_start is not None:
        model = warm_start.model
    else:
        z_max = float(batch.z_nm.abs().max()) if batch.z_nm.numel() else 0.0
        model = bead_model(optics, settings, tuple(batch.data.shape[-2:]), transforms, z_max, device, dtype)
    if model.n_channels != batch.data.shape[1]:
        raise ValueError(f"{model.n_channels} channels of optics for a batch of {batch.data.shape[1]}")
    found = {}
    if warm_start is None:
        found = gaussian_transforms(model, batch)
        for c, H in found.items():
            model.channels[c].transform = H
    values = None if start is None or warm_start is not None else start_values(model, start)
    if values is not None:
        for c in found:                         # the beads' own transform over the previous model's
            values.pop(prefix(c) + "transform", None)
    rejection = RejectionSettings(k=settings.rejection_k)
    cal = learn_beads(model, batch, noise, decisions=decisions, warm_start=warm_start, rejection=rejection,
                      heldout=settings.heldout, progress=progress, cancel=cancel, geometry=geometry,
                      start=values, start_stage1=values is not None)
    cal.options = settings
    if warm_start is None and model.n_channels > 1:
        cal.transform_start = {c: "gaussian" if c in found else "pairing" for c in range(1, model.n_channels)}
    return cal
