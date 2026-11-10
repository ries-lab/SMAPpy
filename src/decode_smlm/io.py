"""A model DECODE v0.10.2 trained, run in the port: milestone (d) of docs/decode/poc-plan.md.

``load_original(param_run.yaml, model.pt)`` reads the original's model with its parameters, and
`OriginalModel.localize` puts frames through the published Fit notebook's pipeline: counts to
photons (`Photon2Camera.backward`), `AutoCenterCrop(8)`, `Mirror2D(-1)` for a model trained on a
mirrored calibration, `AmplitudeRescale`, the network, and `SpatialIntegration` at the notebook's
``raw_th`` 0.1 (plan §6, item 3).  On the example package's first 500 frames this gives the
original's 137 351 localizations, in its order, on the CPU, MPS and CUDA
(``tests/decode_tests/test_original_model.py``).

**The camera is the frames', always given.**  The notebook meant to take it from
``frames_meta.yaml`` but filled the camera *with* the parameter tree instead, so it kept the
training's (EM gain 100 for frames taken at 50) and reported half the photons (plan §8, "(a) as
run").  Here nothing is taken from the parameter file unasked: `training_camera` gives what the
notebook in effect used, for comparing with it.

**The coordinates.**  DECODE's x is the frame's first axis, the rows -- SMAPpy's y (plan §6, item
4) -- in pixels of the cropped, possibly mirrored frame, with pixel centres at whole numbers.
`to_columns` puts them back on the frames as they came in and names them as SMAPpy does; matched to
Spline 3D's fit of the same frames, the medians differ by 0.01 px in x and y.  The pixel size is
the caller's, in SMAPpy's (x, y) order: DECODE's ``px_size`` is in its own (rows, columns) order,
and which order a parameter file meant is not recorded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CROP_FOLD = 8          # the U-Net's two poolings and the notebook's AutoCenterCrop(8)
RAW_TH = 0.1           # the Fit notebook's threshold; the training validates at 0.5
BATCH = 32


@dataclass
class OriginalModel:
    """An original v0.10.2 SigmaMUNet with the parameters it was trained with."""
    param: object                  # DECODE's RecursiveNamespace of param_run.yaml
    model: torch.nn.Module
    frames_mirrored: bool          # trained on a mirrored calibration: the frames are flipped to match
    device: str

    @property
    def frame_window(self) -> int:
        return int(self.param.HyperParameter.channels_in)

    def crop(self, shape) -> tuple[int, int, int, int]:
        """(first row, first column, rows, columns) that AutoCenterCrop keeps of frames of ``shape``."""
        h, w = int(shape[-2]), int(shape[-1])
        th, tw = h // CROP_FOLD * CROP_FOLD, w // CROP_FOLD * CROP_FOLD
        if th == 0 or tw == 0:
            raise ValueError(f"frames of {h} x {w} pixels are smaller than {CROP_FOLD}")
        return -(-(h - th) // 2), -(-(w - tw) // 2), th, tw      # the ceiling, as AutoCenterCrop

    def frame_processing(self, camera: dict):
        from .neuralfitter import frame_processing, scale_transform
        from .neuralfitter.utils.processing import TransformSequence, wrap_callable
        from .simulation.camera import Photon2Camera

        cam = Photon2Camera(qe=camera.get("qe", 1.0), spur_noise=camera.get("spur_noise", 0.0),
                            em_gain=camera.get("em_gain"), e_per_adu=camera["e_per_adu"],
                            baseline=camera["baseline"], read_sigma=0.0, photon_units=True, device="cpu")
        steps = [wrap_callable(cam.backward), frame_processing.AutoCenterCrop(CROP_FOLD)]
        if self.frames_mirrored:
            steps.append(frame_processing.Mirror2D(dims=-1))
        steps.append(scale_transform.AmplitudeRescale.parse(self.param))
        return TransformSequence(steps)

    def _infer(self, frames, camera: dict, post_proc, batch_size: int):
        from .neuralfitter.inference.inference import Infer

        frames = torch.as_tensor(np.asarray(frames, dtype=np.float32))
        if frames.ndim != 3:
            raise ValueError(f"frames are (n, rows, columns), not {tuple(frames.shape)}")
        infer = Infer(model=self.model, ch_in=self.frame_window, frame_proc=self.frame_processing(camera),
                      post_proc=post_proc, device=self.device, batch_size=batch_size, num_workers=0,
                      forward_cat="frames" if post_proc is None else "emitter")
        # Current torch lets cuDNN convolve in TF32 on Ampere and later, which moved the example
        # package's network output by 1.5 % of a channel's range on a 4090 (8e-5 without it).
        tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            return infer.forward(frames)
        finally:
            torch.backends.cudnn.allow_tf32 = tf32

    def raw(self, frames, camera: dict, batch_size: int = BATCH) -> torch.Tensor:
        """The network's output, (n, 10, rows, columns) of the cropped frames, on the CPU."""
        return self._infer(frames, camera, None, batch_size)

    def localize(self, frames, camera: dict, raw_th: float = RAW_TH, batch_size: int = BATCH):
        """The localizations of ``frames`` (n, rows, columns; counts) as DECODE's `EmitterSet`.

        x and y in pixels of the cropped and mirrored frames (`to_columns` puts them back), z in nm,
        ``frame_ix`` from 0.
        """
        from .neuralfitter.coord_transform import Offset2Coordinate
        from .neuralfitter.post_processing import SpatialIntegration
        from .neuralfitter.scale_transform import InverseParamListRescale
        from .neuralfitter.utils.processing import TransformSequence

        _, _, h, w = self.crop(np.shape(frames))
        post = TransformSequence([
            InverseParamListRescale.parse(self.param),
            Offset2Coordinate(xextent=(-0.5, h - 0.5), yextent=(-0.5, w - 0.5), img_shape=(h, w)),
            SpatialIntegration(raw_th=raw_th, xy_unit="px", px_size=self.param.Camera.px_size)])
        return self._infer(frames, camera, post, batch_size)

    def to_columns(self, em, frame_shape, pixel_size_nm, first_frame: int = 0) -> dict:
        """`to_columns` for frames of ``frame_shape`` (n, rows, columns), as `localize` was given."""
        return to_columns(em, pixel_size_nm, crop=self.crop(frame_shape),
                          frames_mirrored=self.frames_mirrored, first_frame=first_frame)


def load_original(param_path, model_path, *, frames_mirrored: bool, device=None) -> OriginalModel:
    """The original's ``model.pt`` (a state dict) with its ``param_run.yaml``.

    ``frames_mirrored`` is not in either file: it is whether the model was trained on a SMAP
    calibration of x-mirrored EM beads used as it was, which the Fit notebook matched by flipping
    the frames (plan §6, item 5).
    """
    from .neuralfitter import models
    from .utils import hardware, param_io

    device = str(device or hardware.default_device())
    param = param_io.load_params(Path(param_path))
    model = getattr(models, param.HyperParameter.architecture).parse(param)
    model.load_state_dict(torch.load(Path(model_path), map_location="cpu", weights_only=True))
    return OriginalModel(param=param, model=model.to(device).eval(), frames_mirrored=bool(frames_mirrored),
                         device=device)


def training_camera(param) -> dict:
    """The camera the parameter file trained with -- what the Fit notebook converted every frame by."""
    c = param.Camera
    return {"baseline": float(c.baseline), "e_per_adu": float(c.e_per_adu), "em_gain": c.em_gain,
            "spur_noise": float(c.spur_noise), "qe": float(c.qe)}


def to_columns(em, pixel_size_nm, *, crop, frames_mirrored: bool, first_frame: int = 0) -> dict:
    """An `EmitterSet` of `OriginalModel.localize` as SMAPpy's columns, on the frames as given.

    ``pixel_size_nm`` is (x, y) in SMAPpy's order, x along the columns; ``crop`` is
    `OriginalModel.crop` of the frames' shape.  ``prob`` keeps its name: it is what DECODE's
    detections are filtered by.
    """
    row0, col0, _, w = crop
    px, py = np.broadcast_to(np.asarray(pixel_size_nm, np.float64), 2)
    xyz = em.xyz.double().numpy()
    sig = em.xyz_sig.double().numpy() if em.xyz_sig is not None else None
    col = (w - 1) - xyz[:, 1] if frames_mirrored else xyz[:, 1]
    cols = {"frame": em.frame_ix.numpy().astype(np.int64) + int(first_frame),
            "x_nm": ((col + col0) * px).astype(np.float32),
            "y_nm": ((xyz[:, 0] + row0) * py).astype(np.float32),
            "z_nm": xyz[:, 2].astype(np.float32),
            "photons": em.phot.numpy().astype(np.float32),
            "background": em.bg.numpy().astype(np.float32),
            "prob": em.prob.numpy().astype(np.float32)}
    if sig is not None:
        cols.update(x_err_nm=(sig[:, 1] * px).astype(np.float32), y_err_nm=(sig[:, 0] * py).astype(np.float32),
                    z_err_nm=sig[:, 2].astype(np.float32))
    if em.phot_sig is not None:
        cols["photons_err"] = em.phot_sig.numpy().astype(np.float32)
    if em.bg_sig is not None:
        cols["background_err"] = em.bg_sig.numpy().astype(np.float32)
    return cols
