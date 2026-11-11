"""A model file put to use: the fitter's side of a trained DECODE model (milestone (g)).

`load_predictor(path)` reads a ``model.safetensors`` -- the port's training writes one every epoch
(`training.write_model`), `export_original` makes one of a model the original trained -- and
gives a `Predictor`, which turns frames in photons into localizations.  It is the one interface
SMAPpy's fitter (`smappy.plugins.fit_decode`) talks to (plan §5): the fitter cuts the movie into
blocks and keeps the overlap, the predictor runs the network on what it is given.

**Context frames.**  The network sees each frame with ``window`` neighbours on either side.
`Predictor.__call__` takes the frames to localize *with* that many frames of context before and
after, so a block's edges see the same frames as one long run does; the caller repeats the first
and last frame of the movie where there is nothing beyond them, as DECODE's ``InferenceDataset``
did (plan §6, item 18).

**Nothing in the file is run.**  The weights are tensors and the rest JSON.  The architecture is
built by name from an allowed list, and the activation is a class of ``torch.nn`` -- DECODE's
own parse evaluates strings, which is not done with a file passed around.

**Photons.**  The caller converts the counts with its camera, ``(counts - offset) * e/ADU / EM
gain`` (SMAPpy's `camera.to_photons`, no excess-noise factor); the training's ``qe`` and
``spur_noise`` are then taken off here, as DECODE's ``Photon2Camera.backward`` did, so the
network sees what it was trained on.

**The crop.**  As in (d), the frames are cut to a multiple of 8 pixels with DECODE's
``AutoCenterCrop``, up to 7 edge pixels lost; the positions are put back on the frames as given.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .io import CROP_FOLD, RAW_TH, crop_of, pixel_columns

METHOD = "decode-v0.10.2"
ARCHITECTURES = ("SigmaMUNet",)
BATCH = 32


def _namespace(meta: dict):
    """The parts of DECODE's parameter tree its parse functions read, from a model file's metadata."""
    from .utils.types import RecursiveNamespace
    hyper = {"architecture": meta["architecture"], "arch_param": meta["arch_param"],
             "channels_in": int(meta["channels_in"]), "channels_out": int(meta.get("channels_out", 10))}
    return RecursiveNamespace(HyperParameter=hyper, Scaling=meta["scaling"], Camera=meta.get("camera") or {})


def trained_on(param) -> dict:
    """What the training drew, from DECODE's parameter tree: per-frame photons (mean and spread),
    the background range and the z range -- what the fitter compares the data's fit against."""
    sim = param.Simulation
    extent = list(getattr(sim, "emitter_extent", None) or [None, None, None])
    def plain(v):
        return None if v is None else [float(x) for x in v]
    return {"photons_mu_sig": plain(getattr(sim, "intensity_mu_sig", None)),
            "background": plain(getattr(sim, "bg_uniform", None)),
            "z_nm": plain(extent[2] if len(extent) > 2 else None)}


def check_metadata(meta: dict) -> None:
    """Refuse what this predictor cannot run, before torch builds anything."""
    if meta.get("method") != METHOD:
        raise ValueError(f"a {meta.get('method') or 'unknown'} model, not {METHOD}")
    if meta.get("architecture") not in ARCHITECTURES:
        raise ValueError(f"architecture {meta.get('architecture')!r}: this version runs {', '.join(ARCHITECTURES)}")
    channels = int(meta.get("channels_in", 0))
    if channels < 1 or channels % 2 != 1:
        raise ValueError(f"a frame window of {channels}: it must be odd")
    activation = (meta.get("arch_param") or {}).get("activation", "ReLU")
    cls = getattr(torch.nn, str(activation), None) if str(activation).isidentifier() else None
    if not (isinstance(cls, type) and issubclass(cls, torch.nn.Module)):
        raise ValueError(f"activation {activation!r} is not a layer of torch.nn")


@dataclass
class Predictor:
    """A trained network with what it needs to localize: scaling, training camera, post-processing."""
    model: torch.nn.Module
    param: object                   # the parse functions' parameter tree (`_namespace`)
    metadata: dict
    frames_mirrored: bool           # flip the frames along x to match the calibration the model learnt
    device: str
    raw_th: float = RAW_TH
    batch_size: int = BATCH

    @property
    def window(self) -> int:
        """Frames of context the network needs on either side of the frame it localizes."""
        return (int(self.metadata["channels_in"]) - 1) // 2

    def _prepared(self, photons: torch.Tensor) -> torch.Tensor:
        from .neuralfitter import frame_processing, scale_transform
        camera = self.metadata.get("camera") or {}
        x = photons
        spur, qe = float(camera.get("spur_noise") or 0.0), float(camera.get("qe") or 1.0)
        if spur:
            x = x - spur
        if qe != 1.0:
            x = x / qe
        x = frame_processing.AutoCenterCrop(CROP_FOLD).forward(x)
        if self.frames_mirrored:
            x = frame_processing.Mirror2D(dims=-1).forward(x)
        return scale_transform.AmplitudeRescale.parse(self.param).forward(x)

    def _post(self, h: int, w: int):
        from .neuralfitter.coord_transform import Offset2Coordinate
        from .neuralfitter.post_processing import SpatialIntegration
        from .neuralfitter.scale_transform import InverseParamListRescale
        from .neuralfitter.utils.processing import TransformSequence
        return TransformSequence([
            InverseParamListRescale.parse(self.param),
            Offset2Coordinate(xextent=(-0.5, h - 0.5), yextent=(-0.5, w - 0.5), img_shape=(h, w)),
            SpatialIntegration(raw_th=self.raw_th, xy_unit="px")])

    def raw(self, photons) -> torch.Tensor:
        """The network's output for the middle frames, (n, 10, rows, columns) of the cropped frames."""
        return self._forward(photons, None)

    def __call__(self, photons, first_frame: int = 0) -> dict:
        """Localizations of ``photons`` (n + 2 window, rows, columns), the first and last ``window``
        frames context only: pixel columns on the frames as given (``x_pix`` along the columns,
        pixel centres at whole numbers), z in nm, ``frame`` from ``first_frame`` for the first
        frame localized."""
        photons = np.asarray(photons, dtype=np.float32)
        if photons.ndim != 3 or len(photons) <= 2 * self.window:
            raise ValueError(f"frames (n + {2 * self.window}, rows, columns), not {photons.shape}")
        em = self._forward(photons, "emitter")
        cols = pixel_columns(em, crop=crop_of(photons.shape), frames_mirrored=self.frames_mirrored)
        cols["frame"] = cols["frame"] + int(first_frame)
        return cols

    def _forward(self, photons, post):
        from .generic.emitter import EmitterSet
        x = torch.as_tensor(np.asarray(photons, dtype=np.float32))
        k, n = self.window, len(x) - 2 * self.window
        x = self._prepared(x)
        _, h, w = x.shape
        post_proc = self._post(h, w) if post else None
        out = []
        # cuDNN's TF32 moved the network's output by 1.5 % of a channel's range (plan §8, (d))
        tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            with torch.no_grad():
                for start in range(0, n, self.batch_size):
                    stop = min(start + self.batch_size, n)
                    # frames start .. stop-1 of the middle, each with its neighbours: (b, 2k+1, h, w)
                    batch = torch.stack([x[i:i + 2 * k + 1] for i in range(start, stop)]).to(self.device)
                    y = self.model(batch)
                    if post_proc is None:
                        out.append(y.cpu())
                    else:
                        em = post_proc.forward(y)
                        em.frame_ix = em.frame_ix + start
                        out.append(em)
        finally:
            torch.backends.cudnn.allow_tf32 = tf32
        if post_proc is None:
            return torch.cat(out) if out else torch.zeros((0, 10, h, w))
        return EmitterSet.cat(out) if out else EmitterSet(
            xyz=torch.zeros((0, 3)), phot=torch.zeros(0), frame_ix=torch.zeros(0, dtype=torch.long), xy_unit="px")


def load_predictor(path, device=None, *, frames_mirrored: bool = False, raw_th: float = RAW_TH,
                   batch_size: int = BATCH) -> Predictor:
    """The model file at ``path`` ready to localize on ``device`` (auto: the fastest there is).

    ``frames_mirrored`` is the caller's: whether the data's frames are mirrored against the
    calibration the model learnt, which the file records (``calibration_em_on``) and the data's
    camera decides (SMAPpy's fitter, `smappy.plugins.fit_decode._mirrored`).
    """
    from smappy.training import model_file

    from .neuralfitter import models
    from .utils import hardware

    tensors, meta = model_file.read(path)
    check_metadata(meta)
    device = str(hardware.default_device() if device in (None, "auto") else hardware.available_device(device))
    param = _namespace(meta)
    model = getattr(models, meta["architecture"]).parse(param)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in tensors.items()})
    return Predictor(model=model.to(device).eval(), param=param, metadata=meta, frames_mirrored=bool(frames_mirrored),
                     device=device, raw_th=float(raw_th), batch_size=int(batch_size))


def export_original(param_path, model_path, out, *, calibration=None, calibration_em_on=None) -> Path:
    """A model the original DECODE trained (``param_run.yaml`` and ``model.pt``) as a model file.

    ``calibration`` -- the `SplineCalibration` it was trained on, as SMAPpy loads it -- gives the
    sha256 the fitter checks data against and the EM state of its beads; without it neither is
    known, and the fitter cannot check.
    """
    from smappy import __version__
    from smappy.training import model_file
    from smappy.training.job import coefficients_sha256

    from .io import load_original
    from .training import _plain

    original = load_original(param_path, model_path, frames_mirrored=False, device="cpu")
    p = original.param
    if calibration is not None and calibration_em_on is None:
        calibration_em_on = calibration.em_on
    meta = {"method": METHOD, "architecture": p.HyperParameter.architecture,
            "arch_param": _plain(p.HyperParameter.arch_param), "channels_in": int(p.HyperParameter.channels_in),
            "channels_out": 10, "scaling": _plain(p.Scaling), "camera": _plain(p.Camera),
            "calibration_em_on": None if calibration_em_on is None else bool(calibration_em_on),
            "calibration_sha256": coefficients_sha256(calibration) if calibration is not None else None,
            "post_processing": {"kind": "SpatialIntegration", "raw_th": RAW_TH},
            "ranges": {"from": str(Path(param_path).name)}, "trained_on": trained_on(p), "base_model": None,
            "trained_by": "DECODE v0.10.2", "smappy_version": __version__, "decode_version": "0.10.2"}
    tensors = {k: v.detach().cpu().numpy() for k, v in original.model.state_dict().items()}
    return model_file.write(out, tensors, meta)
