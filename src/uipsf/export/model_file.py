"""The model file: everything needed to recompute a learned PSF, in one HDF5 file (plan §5.8).

uipsf's only output.  SMAPpy loads it with `load_model`, samples it with
`LearnedPSF.psf` on a grid of its choosing and makes its spline from that
(plan §6), so the spline's format and normalisation have one owner.

**Layout** (``format = "uipsf-model"``, ``version`` = `VERSION`):

- ``/model``: the components as a JSON description (`describe`; `build`
  makes the same model again), and every shared parameter in physical
  units, ``/model/params/<name>``, with its CRLB beside it in
  ``/model/crlb/<name>`` where it was fitted.  The description has one
  entry per channel (``channels``, the main one first): its optics, pupil
  (the Zernike modes as Noll indices, in the order of the coefficients),
  PSF settings and starting transform.  Channel c ≥ 1's parameters carry
  the prefix ``ch<c>.`` (`model.channels`): ``ch1.pupil.phase``, and its
  transform to the main channel's chip pixels ``ch1.transform`` (the 3×3
  matrix row by row without H₂₂ = 1, SMAPpy's direction, secondary →
  main), photon share ``ch1.share`` and defocus offset ``ch1.dz`` (nm).

For channels split on one chip, ``/model`` also holds the split's
geometry (attribute ``geometry``, JSON) as SMAPpy records it with a dual
calibration (`DualColorCalibration.geometry`: layout, main channel, split
position, mirror axis, image shape, coordinate system, the stacks' camera
ROIs).  With it, SMAPpy turns the file into one of its own dual
calibrations, which its global fitter reads: each channel sampled with
`LearnedPSF.psf(…, channel=c)` on one z grid and normalised jointly as
`positive_pair_models` does (its ``photon_normalization`` is then each
channel's share), the secondary's spline in the camera's orientation, and
`LearnedPSF.transform(1)` as its ``transformation``.

**Versions.**  2 (M4): the ``channels`` list and the geometry.  1: one
channel, its optics, pupil and PSF at the top of the description; still
read.
- ``/fit``: objective, stop reason, iterations, condition, the history
  (JSON), the shared covariance with its layout, and the used beads' own
  parameters and CRLBs (``/fit/emitters/<name>``).
- ``/beads``: the bead table (JSON, `BeadTable.as_dicts`), its thresholds
  and the rejection settings.
- ``/diagnostics``: the residual atlas and the held-out report (JSON).

JSON rather than one dataset per number: the tables are small and their
fields grow; a reader of another language still finds plain text.  NaN is
written as JSON's ``NaN`` (Python's reading; a bead never scored has NaN
scores).  The data themselves are not stored: the stacks are the user's
files, and the model file stays small (≈ 100 kB).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

import h5py
import numpy as np
import torch

from ..model import (
    Bleaching,
    Constant,
    ForwardModel,
    PerObservation,
    PerSliceFactor,
    PixelPupil,
    Planar,
    VectorPSF,
    ZernikePupil,
)
from ..model.channels import Channel, prefix, unpack_transform, view
from ..model.geometry import BeadStackGeometry
from ..optics import Optics, PupilGrid

FORMAT = "uipsf-model"
VERSION = 2
CONVENTIONS = 1          # docs/uipsf/conventions.md: axes, signs and units of the stored numbers


def _describe_channel(ch: Channel) -> dict:
    psf, pupil = ch.psf, ch.psf.pupil
    if isinstance(pupil, ZernikePupil):
        pup = dict(kind="zernike", phase_modes=list(pupil.phase_modes),
                   magnitude_modes=list(pupil.magnitude_modes), high_na=bool(pupil.high_na))
    elif isinstance(pupil, PixelPupil):
        pup = dict(kind="pixel")
    else:
        raise TypeError(f"no model-file entry for the pupil {type(pupil).__name__}")
    return dict(name=ch.name, optics=dataclasses.asdict(psf.grid.optics), pupil_size=psf.grid.size, pupil=pup,
                psf=dict(kind=psf.kind, analyser=psf.analyser, roi_shape=list(psf.roi_shape),
                         upsample=psf.upsample, bead_radius_nm=psf.bead_radius_nm,
                         sigma_init_nm=list(psf.sigma_init), fit_sigma=psf.fit_sigma),
                transform=ch.transform.tolist(), share=ch.share, dz_nm=ch.dz_nm, fit_transform=ch.fit_transform)


def describe(model: ForwardModel) -> dict:
    """The components of ``model`` and their settings, as plain data: what `build` needs."""
    g = model.geometry
    geometry = dict(kind="bead stack", drift=g.drift, drift_prior_nm=g.drift_prior_nm, z_scale=g.z_scale,
                    z_scale_prior=g.z_scale_prior)
    i = model.intensity
    if isinstance(i, PerSliceFactor):
        intensity = dict(kind="per-slice factor", order=i.order, prior_sigma=i.prior_sigma)
    elif isinstance(i, Bleaching):
        intensity = dict(kind="bleaching", order=i.order)
    elif isinstance(i, PerObservation):
        intensity = dict(kind="per observation")
    else:
        raise TypeError(f"no model-file entry for the intensity {type(i).__name__}")
    background = dict(kind="planar") if isinstance(model.background, Planar) else dict(kind="constant")
    return dict(channels=[_describe_channel(ch) for ch in model.channels], geometry=geometry,
                intensity=intensity, background=background)


def _build_channel(config: dict, device, dtype) -> Channel:
    o = dict(config["optics"])
    o["pixel_size_nm"] = tuple(o["pixel_size_nm"])
    grid = PupilGrid.build(Optics(**o), config["pupil_size"], device=device, dtype=dtype)
    pup = config["pupil"]
    if pup["kind"] == "zernike":
        pupil = ZernikePupil(grid, phase_modes=pup["phase_modes"], magnitude_modes=pup["magnitude_modes"],
                             high_na=pup["high_na"])
    else:
        pupil = PixelPupil(grid)
    s = config["psf"]
    psf = VectorPSF(grid, pupil, tuple(s["roi_shape"]), upsample=s["upsample"], kind=s["kind"],
                    analyser=s["analyser"], bead_radius_nm=s["bead_radius_nm"], sigma_nm=tuple(s["sigma_init_nm"]),
                    fit_sigma=s.get("fit_sigma", True))
    return Channel(psf, transform=config.get("transform"), share=config.get("share"), dz_nm=config.get("dz_nm"),
                   name=config.get("name"), fit_transform=config.get("fit_transform", True))


def build(config: dict, device=None, dtype=torch.float32) -> ForwardModel:
    """The `ForwardModel` that ``config`` (`describe`) describes; a version-1 description (one channel,
    its entries at the top) too."""
    channels = [_build_channel(c, device, dtype) for c in config.get("channels", [config])]
    g = config["geometry"]
    geometry = BeadStackGeometry(g["drift"], g["drift_prior_nm"], g["z_scale"], g["z_scale_prior"])
    i = config["intensity"]
    intensity = {"bleaching": lambda: Bleaching(i["order"]),
                 "per-slice factor": lambda: PerSliceFactor(i["order"], i["prior_sigma"]),
                 "per observation": PerObservation}[i["kind"]]()
    pixel = channels[0].pixel_size_nm
    background = Planar(pixel) if config["background"]["kind"] == "planar" else Constant()
    return ForwardModel(channels, geometry, intensity, background)


def sample_psf(model: ForwardModel, shared: dict, xyz_nm, roi_px=None, depth_nm=None, include_bead=False,
               upsample: int | None = None, channel: int = 0) -> torch.Tensor:
    """The learned PSF of ``channel`` at positions ``xyz_nm`` [N, 3]: [N, Y, X] expected photons per pixel
    for one photon of the emitter, on the CPU.

    x and y are from the centre of the channel's own region, in its own (camera) orientation: SMAPpy
    samples every channel with the emitter on the central knot.  z is the emitter's height above the
    *main* channel's focus, for every channel: the channel's defocus offset is applied here, so both
    channels are sampled on one z grid and a biplane offset is in the PSFs (plan §6.4).  The channel's
    photon share is included, so the channels together hold the emitter's photon, as SMAPpy's
    `positive_pair_models` normalises a pair jointly; with one channel the share is 1.

    The blur is the learned one; the bead's size, part of the bead stacks and not of a point emitter,
    is left out unless ``include_bead`` (then ``xyz_nm`` must be one stack of equal z steps).  The
    stage's z scale is not applied: z is the emitter's, not an objective position.  Pixels are
    integrated by summing ``upsample``² sub-samples (default the model's); sample the light at more
    of them for a spline that must not inherit the sampling's error.  ``roi_px`` is the region,
    an int or (Y, X), default the model's.
    """
    base = model.channels[channel].psf
    roi = base.roi_shape if roi_px is None else (roi_px, roi_px) if np.isscalar(roi_px) else tuple(roi_px)
    up = base.upsample if upsample is None else int(upsample)
    psf = VectorPSF(base.grid, base.pupil, roi, upsample=up, kind=base.kind, analyser=base.analyser,
                    bead_radius_nm=base.bead_radius_nm if include_bead else 0.0)
    dev, dt = base.grid.rho.device, base.grid.rho.dtype
    xyz = torch.as_tensor(xyz_nm, dtype=dt, device=dev).reshape(1, -1, 3)
    shared = {k: torch.as_tensor(v, device=dev) for k, v in shared.items()}
    p = view(shared, channel)
    if channel > 0:
        xyz = xyz + torch.stack([torch.zeros((), dtype=dt, device=dev)] * 2 + [p[prefix(channel) + "dz"].to(dt)])
    step = None
    if include_bead and base.bead_radius_nm > 0:
        dz = torch.diff(xyz[0, :, 2])
        if len(dz) == 0 or not torch.allclose(dz, dz[0].expand_as(dz), rtol=1e-4, atol=1e-3):
            raise ValueError("include_bead needs one stack with equal z steps")
        step = float(dz[0])
    depth = None if depth_nm is None else torch.as_tensor(depth_nm, dtype=dt, device=dev).reshape(1)
    with torch.no_grad():
        # on the CPU, for NumPy, wherever it was computed
        return (model.share(shared, channel) * psf(p, xyz, depth, z_step_nm=step)[0]).cpu()


@dataclass
class LearnedPSF:
    """A loaded model file: the model, its shared parameters and the record of how they were learned."""
    model: ForwardModel
    shared: dict[str, torch.Tensor]                 # physical values
    crlb: dict[str, torch.Tensor]                   # of the fitted ones
    config: dict
    fit: dict = field(default_factory=dict)
    beads: list[dict] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    emitters: dict[str, np.ndarray] = field(default_factory=dict)
    geometry: dict | None = None                    # a split chip, as SMAPpy's dual calibration records it
    options: dict = field(default_factory=dict)     # the choices it was learnt with (`BeadSettings`)

    def psf(self, xyz_nm, roi_px=None, depth_nm=None, include_bead=False, upsample=None,
            channel: int = 0) -> torch.Tensor:
        """`sample_psf` on this model."""
        return sample_psf(self.model, self.shared, xyz_nm, roi_px, depth_nm, include_bead, upsample, channel)

    @property
    def n_channels(self) -> int:
        return self.model.n_channels

    def shares(self) -> list[float]:
        """Each channel's photon share, the main channel's first; they sum to 1."""
        return [float(self.model.share(self.shared, c)) for c in range(self.n_channels)]

    def transform(self, channel: int = 1) -> np.ndarray:
        """Channel ``channel``'s 3×3 projective map to the main channel's chip pixels (H₂₂ = 1), as SMAPpy's
        `DualColorCalibration.transformation`; the identity for the main channel."""
        if channel == 0:
            return np.eye(3)
        return unpack_transform(torch.as_tensor(_np(self.shared[prefix(channel) + "transform"]))).numpy()

    def dz_nm(self, channel: int = 1) -> float:
        """Channel ``channel``'s defocus offset: the emitter is that much higher above its focal plane."""
        return 0.0 if channel == 0 else float(_np(self.shared[prefix(channel) + "dz"]))

    def zernikes(self, which: str = "phase", channel: int = 0) -> list[tuple[int, float, float]]:
        """(Noll index, coefficient, CRLB) of every ``which`` ("phase", rad, or "magnitude") mode."""
        config = self.config["channels"][channel] if "channels" in self.config else self.config
        modes = config["pupil"].get(f"{which}_modes", [])
        name = prefix(channel) + f"pupil.{which}"
        c = _np(self.shared[name])
        s = _np(self.crlb[name]) if name in self.crlb else np.full(len(modes), np.nan)
        return [(int(j), float(v), float(e)) for j, v, e in zip(modes, c, s, strict=True)]


def _json(obj) -> str:
    def default(o):
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().double().numpy().tolist()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        raise TypeError(f"cannot write {type(o).__name__} to the model file")
    return json.dumps(obj, default=default)


def _atlas(atlas) -> dict | None:
    if atlas is None:
        return None
    return dict(mean_deviance=atlas.mean_deviance, checks=atlas.checks,
                findings=[dataclasses.asdict(f) for f in atlas.findings],
                groups={k: dataclasses.asdict(g) for k, g in atlas.groups.items()})


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().double().numpy()        # copy, then cast: MPS has no float64


def save_model(cal, path) -> None:
    """Write the `Calibration` ``cal`` to ``path`` (module docstring for the layout)."""
    from .. import __version__
    res = cal.result
    params = res.params
    with h5py.File(path, "w") as f:
        f.attrs.update(format=FORMAT, version=VERSION, conventions=CONVENTIONS, uipsf=__version__)
        m = f.create_group("model")
        m.attrs["config"] = _json(describe(cal.model))
        if getattr(cal, "geometry", None) is not None:
            m.attrs["geometry"] = _json(cal.geometry)
        if getattr(cal, "options", None) is not None:
            m.attrs["options"] = _json(cal.options)
        pg, cg = m.create_group("params"), m.create_group("crlb")
        for n, v in cal.shared().items():
            pg[n] = _np(v)
            if n in res.std:
                cg[n] = _np(res.std[n])
        fg = f.create_group("fit")
        fg.attrs.update(objective=float(res.objective), stop_reason=res.stop_reason, converged=bool(res.converged),
                        iterations=res.iterations, condition=float(res.condition), remaining=float(res.remaining),
                        history=_json(res.history),
                        covariance_layout=_json([[n, a, b] for n, a, b in params.layout("shared")]))
        fg["covariance_shared"] = _np(res.covariance_shared)
        eg = fg.create_group("emitters")
        for n, v in params.physical().items():
            if params.specs[n].scope == "emitter":
                eg[n] = _np(v)
                if n in res.std:
                    eg[n + ".crlb"] = _np(res.std[n])
        bg = f.create_group("beads")
        bg.attrs.update(table=_json(cal.beads.as_dicts()), thresholds=_json(cal.beads.thresholds),
                        settings=_json(cal.settings), used=_json(cal.beads.used), heldout=_json(cal.heldout))
        dg = f.create_group("diagnostics")
        dg.attrs["atlas"] = _json(_atlas(cal.atlas))
        rep = cal.heldout_report
        dg.attrs["heldout"] = _json(None if rep is None else dict(dataclasses.asdict(rep), summary=rep.summary()))


def _read(attr):
    return json.loads(attr if isinstance(attr, str) else attr.decode())


def load_model(path, device=None, dtype=torch.float32) -> LearnedPSF:
    """The model file at ``path``, ready for `LearnedPSF.psf`."""
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT:
            raise ValueError(f"{path} is not a uipsf model file")
        if int(f.attrs["version"]) > VERSION:
            raise ValueError(f"{path} is model-file version {int(f.attrs['version'])}; this uipsf reads up to "
                             f"{VERSION}")
        config = _read(f["model"].attrs["config"])
        geometry = _read(f["model"].attrs["geometry"]) if "geometry" in f["model"].attrs else None
        options = _read(f["model"].attrs["options"]) if "options" in f["model"].attrs else {}
        model = build(config, device=device, dtype=dtype)       # the fastest device unless given
        dev = model.psf.grid.rho.device
        shared = {n: torch.as_tensor(np.asarray(d), dtype=dtype, device=dev) for n, d in f["model/params"].items()}
        crlb = {n: torch.as_tensor(np.asarray(d), dtype=torch.float64) for n, d in f["model/crlb"].items()}
        fa = f["fit"].attrs
        fit = dict(objective=float(fa["objective"]), stop_reason=str(fa["stop_reason"]),
                   converged=bool(fa["converged"]), iterations=int(fa["iterations"]),
                   condition=float(fa["condition"]), remaining=float(fa["remaining"]),
                   history=_read(fa["history"]), covariance_layout=_read(fa["covariance_layout"]),
                   covariance_shared=np.asarray(f["fit/covariance_shared"]))
        emitters = {n: np.asarray(d) for n, d in f["fit/emitters"].items()}
        ba = f["beads"].attrs
        beads, thresholds = _read(ba["table"]), _read(ba["thresholds"])
        diagnostics = dict(atlas=_read(f["diagnostics"].attrs["atlas"]),
                           heldout=_read(f["diagnostics"].attrs["heldout"]),
                           rejection=_read(ba["settings"]), heldout_beads=_read(ba["heldout"]))
    return LearnedPSF(model, shared, crlb, config, fit, beads, thresholds,
                      diagnostics, emitters, geometry, options)
