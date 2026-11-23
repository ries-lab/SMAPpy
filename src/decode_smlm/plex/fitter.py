"""DECODE-Plex's published models in SMAPpy's DECODE fitter: a model file, and a predictor behind the
fitter's interface (docs/decode/gui-plan.md, I5; plex-plan §9).

**The model file.**  A run of DECODE-Plex is a folder -- its ``param_run.yaml`` and a Lightning
checkpoint -- and the registration it was trained with, a ``*_trafo.mat`` whose path in the run's
settings is the training machine's.  `export` puts the three into one ``.safetensors``, the network's
weights as tensors and the settings and the registration's bytes in its header, method
``decode-plex``: the file the fitter takes, which runs wherever it is copied (and on a training host).

**The predictor.**  `PlexPredictor` runs the port's own inference (`neuralfitter.inference.functional.infer`)
on the frames the fitter's `WindowEngine` hands it, with their context: the camera's frames whole, cut
into the channels and turned into photons by DECODE-Plex's own steps, with its own registration, as
the paper's notebooks do (the port reproduces their localizations, ``tests/decode_tests/plex``).  The
engine hands photons, converted with SMAPpy's camera; they go back to counts with the run's camera
before DECODE-Plex converts them again, so the photons it sees are SMAPpy's whichever camera either
side names.  Its localizations come back in SMAPpy's columns: DECODE-Plex's ``xyz`` is (row, column)
on channel 0, the frame's main half, in pixels, and z in nm.

It is a baseline, not the tool: none of SMAPpy's checks (goodness, bound, the channels' move) run on
it, and a movie must be cut as the paper's were -- the channels the frame's halves, top and bottom.
"""
from __future__ import annotations

import base64
import json
import math
import tempfile
from pathlib import Path

import numpy as np

METHOD = "decode-plex"


def export(run_folder, trafo, out) -> Path:
    """A DECODE-Plex run (its folder, with its ``param_run.yaml`` and one ``*.ckpt``, or the checkpoint
    itself) and its registration (``*_trafo.mat``) as one model file for SMAPpy's DECODE fitter."""
    from smappy import __version__
    from smappy.training import model_file

    from .io import param
    from .io.model import load_state_dict
    run_folder = Path(run_folder)
    if run_folder.suffix == ".ckpt":                # a checkpoint named: its run is its folder
        ckpts, run_folder = [run_folder], run_folder.parent
    else:
        ckpts = sorted(run_folder.glob("*.ckpt"))
    if len(ckpts) != 1:
        raise ValueError(f"{run_folder}: one checkpoint (*.ckpt) expected, found {len(ckpts)}")
    cfg = param.load(run_folder / "param_run.yaml")
    state = load_state_dict(ckpts[0], device="cpu")
    meta = {"method": METHOD, "cfg": json.loads(json.dumps(cfg, default=str)),
            "trafo_name": Path(trafo).name, "trafo": base64.b64encode(Path(trafo).read_bytes()).decode("ascii"),
            "window": int(cfg["Trainer"]["frame_window"]), "run": run_folder.name, "checkpoint": ckpts[0].name,
            "trained_by": "DECODE-Plex", "smappy_version": __version__,
            "calibration_source": str((cfg.get("Paths") or {}).get("calibration"))}
    out = Path(out)
    model_file.write(out, {k: v.numpy() for k, v in state.items()}, meta)
    return out


class PlexPredictor:
    """`predict.Predictor`'s interface for the `WindowEngine` (the module's docstring)."""

    frames_mirrored = False

    def __init__(self, path, device=None, *, threshold: float | None = None, batch: int = 0):
        import torch

        from decode_smlm.utils.hardware import default_device
        from smappy.training import model_file

        from .neuralfitter.inference.functional import infer  # noqa: F401  (fails early without the port)
        tensors, meta = model_file.read(Path(path))
        if meta.get("method") != METHOD:
            raise ValueError(f"{Path(path).name} is not a DECODE-Plex model")
        self.metadata = {k: v for k, v in meta.items() if k != "trafo"}
        self.device = device if device not in (None, "auto") else default_device()
        self.cfg = meta["cfg"]
        for s in self.cfg["Hardware"]["device"]:
            self.cfg["Hardware"]["device"][s] = self.device
        for cam in self.cfg["Camera"]:
            cam["specs"]["flip"]["channel"] = -2             # as the paper's notebooks set it
        self._folder = tempfile.TemporaryDirectory()
        self.trafo = Path(self._folder.name) / meta["trafo_name"]
        self.trafo.write_bytes(base64.b64decode(meta["trafo"]))
        self.state = {k: torch.as_tensor(np.asarray(v)) for k, v in tensors.items()}
        self.window = (int(meta["window"]) - 1) // 2
        # DECODE-Plex's own post-processing decides what is a localization; this only keeps those above it
        self.raw_th = float(threshold) if threshold is not None else 0.0
        self.batch = int(batch) or "auto"
        self._model, self._shape = None, None

    def _network(self, crop):
        """The network for frames cut to ``crop`` (rows, columns a channel), built once per size."""
        if self._shape != tuple(crop):
            from .io import param
            from .neuralfitter.train import setup_cfg
            cfg = param.patch_img_size(json.loads(json.dumps(self.cfg)), crop, scenario=["Simulation", "Test"])
            model = setup_cfg.setup_model(cfg)
            model.load_state_dict(self.state)
            self._model, self._shape = model.to(self.device).eval(), tuple(crop)
        return self._model

    def _counts(self, photons):
        """Photons back to the counts DECODE-Plex's first camera converts from."""
        spec = self.cfg["Camera"][0]["specs"]
        gain = float(spec.get("em_gain") or 1.0)
        return photons * gain / float(spec["e_per_adu"]) + float(spec["baseline"])

    def __call__(self, photons, first_frame: int = 0) -> dict:
        import torch

        from .neuralfitter.inference.functional import infer
        photons = np.asarray(photons, np.float32)
        if photons.ndim != 3:
            raise ValueError("a DECODE-Plex model takes the camera's frames whole: (n, rows, columns)")
        k = self.window
        n = len(photons) - 2 * k
        # the channels the frame's halves, top and bottom, each cut to a multiple of 8 (the notebooks')
        crop = (math.floor(photons.shape[-2] / 2 / 8) * 8, math.floor(photons.shape[-1] / 8) * 8)
        frames = torch.as_tensor(self._counts(photons))
        em, _ = infer(frames, frame_crop=crop, cfg=json.loads(json.dumps(self.cfg)), model=self._network(crop),
                      mode="multi", trafo=self.trafo, mode_camera="rois", device=[self.device],
                      batch_size=self.batch, num_workers=0, logger=None)
        e = {f: getattr(em, f).cpu().numpy() for f in ("xyz", "xyz_sig", "phot", "phot_sig", "bg", "prob", "frame_ix")}
        keep = (e["frame_ix"] >= k) & (e["frame_ix"] < k + n) & (e["prob"] >= self.raw_th)
        e = {f: v[keep] for f, v in e.items()}
        phot = e["phot"] if e["phot"].ndim == 1 else e["phot"].sum(1)
        phot_err = (e["phot_sig"] if e["phot_sig"].ndim == 1 else np.sqrt((e["phot_sig"] ** 2).sum(1)))
        bg = e["bg"] if e["bg"].ndim == 2 else e["bg"][:, None]
        out = {"frame": e["frame_ix"].astype(np.int64) - k + int(first_frame),
               "x_pix": e["xyz"][:, 1], "y_pix": e["xyz"][:, 0], "z_nm": e["xyz"][:, 2],
               "x_err_pix": e["xyz_sig"][:, 1], "y_err_pix": e["xyz_sig"][:, 0], "z_err_nm": e["xyz_sig"][:, 2],
               "photons": phot, "photons_err": phot_err, "prob": e["prob"],
               "background": bg[:, 0], "background_ch0": bg[:, 0], "background_ch1": bg[:, -1]}
        if e["phot"].ndim == 2:                       # colour: each channel its own count
            out.update(photons_ch0=e["phot"][:, 0], photons_ch1=e["phot"][:, 1],
                       photons_ch0_err=e["phot_sig"][:, 0], photons_ch1_err=e["phot_sig"][:, 1])
        return out
