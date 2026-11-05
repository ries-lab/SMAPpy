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
from smappy.calibrate.pupil import AUTO_ORDER, BLURS, MAGNITUDES, BeadSettings, PSFModelSettings  # noqa: F401

from ..model import (
    BeadStackGeometry,
    Channel,
    FieldPolynomial,
    ForwardModel,
    MinimalPupil,
    VectorPSF,
    ZernikePupil,
)
from ..model.build import Held, magnitude_modes, make_psf  # noqa: F401  (magnitude_modes: callers import it here)
from ..model.channels import prefix
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


def bead_model(optics, settings: BeadSettings, roi_shape, transforms=None, z_max_nm: float = 0.0,
               device=None, dtype=torch.float32, held: Held | None = None, field_centre_px=None) -> ForwardModel:
    """The model a bead calibration learns: one channel per `Optics` in ``optics`` (one, or a list, the
    main channel first), each with its own pupil and PSF as ``settings.psf`` describes them (`make_psf`: the
    grid, its margin, the pixel's blur and the SAF damping are the builder's), started at ``transforms``
    (secondary → main chip pixels; SMAPpy's dual calibration).  The pupil sampling keeps defocus up to
    ``z_max_nm`` from aliasing.  ``held``: what each channel's pupil holds (`Held`).  ``settings.fit_tilt``
    learns the coverslip's tilt (`BeadStackGeometry`): seen by beads only through a coverslip of another index
    than the immersion's, a water or silicone objective's (`Optics.coverslip_matters`).

    Each channel learns its own γ (the user's decision, November 2026): two colours pass the objective at
    different angles near the edge.  To revisit: if the channels' γ come out alike, one shared γ is fewer
    numbers to learn (`docs/uipsf/roadmap.md`)."""
    optics = [optics] if not isinstance(optics, (list, tuple)) else list(optics)
    transforms = list(transforms) if transforms is not None else [None] * len(optics)
    if len(transforms) != len(optics):
        raise ValueError(f"{len(optics)} channels but {len(transforms)} transforms")
    channels = []
    for o, H in zip(optics, transforms, strict=True):
        psf = make_psf(o, settings.psf, roi_shape, held=held, bead_radius_nm=settings.bead_diameter_nm / 2,
                       z_max_nm=z_max_nm, device=device, dtype=dtype)
        field = None
        if settings.field_dependence and not channels:          # the main channel only, for now
            if field_centre_px is None:
                raise ValueError("a field-dependent model needs the field's centre (the beads' chip positions)")
            field = FieldPolynomial(field_centre_px, o.pixel_size_nm)
        channels.append(Channel(psf, transform=H, field=field))
    return ForwardModel(channels, geometry=BeadStackGeometry(drift=settings.drift, fit_tilt=settings.fit_tilt))


def insitu_model(optics, spec: PSFModelSettings, roi_shape, focus_nm: float, z_max_nm: float = 1000.0,
                 device=None, dtype=torch.float32, held: Held | None = None, fit_focus: bool = True,
                 fit_tilt: bool = False) -> ForwardModel:
    """The model in situ learning learns (plan §9, M5): one channel, the pupil and blur as ``spec``
    describes them for a point (`make_psf`; no bead's size, and no SAF damping, which is the beads': a
    molecule's light beyond the critical angle follows its own depth), molecules in the sample seen through
    ``focus_nm`` (`InSituGeometry`; held there with ``fit_focus`` off, the coverslip's tilt learnt with
    ``fit_tilt``), one photon count each.  ``held``: a bead model's γ (or symmetric magnitude) and high
    orders (`held_from`).  The pupil sampling keeps defocus up to ``z_max_nm`` from aliasing: the
    molecules' distance from the plane in focus, not their depth, since the depth's and the focus's waves
    largely cancel there and the light beyond the critical angle decays rather than aliases.  ±1 µm needs
    64 samples at NA 1.43; 2 µm asked for 96, 2.25 times the work."""
    from dataclasses import replace as replaced

    from ..model import Bleaching, InSituGeometry
    psf = make_psf(optics, replaced(spec, saf_damping=False), roi_shape, held=held, z_max_nm=z_max_nm,
                   device=device, dtype=dtype)
    return ForwardModel([Channel(psf)], geometry=InSituGeometry(focus_nm, fit_focus=fit_focus, fit_tilt=fit_tilt),
                        intensity=Bleaching(order=0))


def molecule_stack_model(optics, spec: PSFModelSettings, roi_shape, sweeps: int, focus_nm, centre_px,
                         height_nm: float | None = None, transforms=None, z_max_nm: float = 1100.0, device=None,
                         dtype=torch.float32) -> ForwardModel:
    """The model learnt from single molecules on the coverslip through z sweeps (`CoverslipStackGeometry`,
    `molecule_stack`): one channel per `Optics` in ``optics``, the main one first, secondaries started at
    ``transforms`` as `bead_model`'s; each a point's PSF as ``spec`` describes it, with no SAF damping of
    the pupil's own: the molecules' common height ``geometry.height`` (learnt, or held at ``height_nm``)
    decays the supercritical light, and only that (`VectorPSF.saf_only_depth`: the phase at that height is
    every molecule's, the pupil's and the foci's to take up; drawn, the pupil is the objective's with it taken
    out, `VectorPSF.for_emitters`).  One focus per sweep, shared by the channels; one photon
    count per molecule.  ``z_max_nm`` the sampling's defocus: the sweeps' ±1 µm and a margin."""
    from dataclasses import replace as replaced

    from ..model import Bleaching, CoverslipStackGeometry
    optics = [optics] if not isinstance(optics, (list, tuple)) else list(optics)
    transforms = list(transforms) if transforms is not None else [None] * len(optics)
    channels = []
    for o, H in zip(optics, transforms, strict=True):
        psf = make_psf(o, replaced(spec, saf_damping=False), roi_shape, z_max_nm=z_max_nm, device=device, dtype=dtype)
        psf.saf_only_depth = True
        channels.append(Channel(psf, transform=H))
    geometry = CoverslipStackGeometry(sweeps, focus_nm, centre_px, optics[0].pixel_size_nm, height_nm=height_nm)
    return ForwardModel(channels, geometry=geometry, intensity=Bleaching(order=0))


def start_values(model: ForwardModel, previous) -> dict:
    """Shared start values for ``model`` from ``previous`` (a `LearnedPSF` or a `Calibration`).

    Zernike coefficients go to the same Noll modes, zero where the previous model had none; the minimal
    pupil's γ, h and NA, and a learnt sample index, where both have them; the blur, the channel transforms, shares
    and offsets are taken when the model fits them, the blur converted where one model has the pixel's under it
    and the other not (the image's blur kept); anything else (a fixed blur, the stage's terms) keeps the model's
    own start."""
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
        if isinstance(new.psf.pupil, MinimalPupil) and isinstance(old.psf.pupil, MinimalPupil):
            learnt = {s.name for s in new.psf.pupil.specs()}          # not a held γ, nor h without a ring
            for n in ("pupil.gamma", "pupil.ring_h", "pupil.na"):
                if pre + n in shared and n in learnt:
                    out[pre + n] = shared[pre + n]
        if new.psf.fit_sigma and pre + "psf.sigma" in shared:
            # a round blur from a free one takes their mean, a free one from a round one its value twice
            s = torch.as_tensor(shared[pre + "psf.sigma"]).detach().cpu().double()
            s = pixel_converted(VectorPSF.blur({"psf.sigma": s}), old.psf, new.psf)
            out[pre + "psf.sigma"] = s.mean().reshape(1) if new.psf.round_sigma else VectorPSF.blur({"psf.sigma": s})
        if c > 0:
            for n in ("share", "dz") + (("transform",) if new.fit_transform else ()):
                if pre + n in shared:
                    out[pre + n] = shared[pre + n]
    if "geometry.tilt" in shared and getattr(model.geometry, "fit_tilt", False):
        out["geometry.tilt"] = shared["geometry.tilt"]       # the coverslip's, where this model learns one
    if "optics.n_med" in shared and model.psf.learn_n_med:
        out["optics.n_med"] = shared["optics.n_med"]         # the sample's index, where this model learns one
    return {n: torch.as_tensor(v).detach().cpu() for n, v in out.items()}


def pixel_converted(sigma, old, new):
    """A learnt σ (x, y) of the PSF ``old`` as ``new`` learns it, the same blur of the image: the pixel's
    taken out or put in where only one has it under the learnt one (`VectorPSF.pixel_blur`)."""
    def px(psf):
        return torch.tensor(psf.pixel_sigma, dtype=sigma.dtype)
    total = sigma ** 2 + px(old) ** 2
    # at least 1 nm: a learnt σ starting at 0 has no gradient through its magnitude transform
    return torch.sqrt(torch.clamp(total - px(new) ** 2, min=1.0))


def learning_batch(batch, settings=None):
    """The slices a bead calibration learns from: every n-th, n·Δz at most the settings' *largest z step*
    (`Batch.thinned`, the middle slice kept), then those within their *learn z range* of the focus
    (`Batch.within`).  Settings without these (a model file of before November 2026, or None) learnt
    every 50 nm over the whole stack.  Whatever reads a calibration's beads back takes the same slices,
    from ``cal.options``: per-slice parameters (a drift per slice) belong to them."""
    from .beads import MAX_DZ_NM
    max_dz = getattr(settings, "max_dz_nm", MAX_DZ_NM)
    return batch.thinned(max_dz).within(getattr(settings, "z_range_nm", None))


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
    batch = learning_batch(batch, settings)
    if warm_start is not None:
        model = warm_start.model
    else:
        z_max = float(batch.z_nm.abs().max()) if batch.z_nm.numel() else 0.0
        from smappy.calibrate.pupil import field_centre
        model = bead_model(optics, settings, tuple(batch.data.shape[-2:]), transforms, z_max, device, dtype,
                           field_centre_px=field_centre(batch))
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
                      heldout=settings.heldout, progress=progress, cancel=cancel, geometry=geometry, max_dz_nm=None,
                      start=values, start_stage1=values is not None)
    cal.options = settings
    if warm_start is None and model.n_channels > 1:
        cal.transform_start = {c: "gaussian" if c in found else "pairing" for c in range(1, model.n_channels)}
    return cal
