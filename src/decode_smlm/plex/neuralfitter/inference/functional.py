from pathlib import Path
from typing import Literal

import omegaconf
import torch

from . import debug
from . import inference
from .. import indicator
from ..train import setup_cfg
from ... import emitter
from ... import io
from ...simulation.trafo.pos import trafo as pos_trafo
from ...utils import dev
from ...utils import param_auto
from ... import validation as val


@dev.experimental(tested=False)
def infer(
    frames: torch.Tensor,
    frame_crop: tuple[int, int],
    cfg: dict | omegaconf.dictconfig.DictConfig | str | Path,
    model: torch.nn.Module | str | Path,
    mode: Literal["single", "multi"],
    trafo: Path | pos_trafo.XYZTransformation | None,
    device: torch.device | str = "cuda:0",
    roi_shift: tuple[int, int] | None = None,
    mode_camera: Literal["rois", "cameras"] = "rois",
    patches: dict | None = None,
    logger: Literal["debug"] | debug.InferenceLogger | None = None,
    validator: Literal["default"] | val.base.PipelineValidator | None = "default",
    batch_size: int | Literal["auto"] = "auto",
    num_workers: int = 8,
) -> tuple[emitter.EmitterSet, debug.InferenceLogger | None]:
    """
    Factory function to create inference object.

    Args:
        frames: frames to infer on
        frame_crop: crop of frames
        cfg: config object or path to config file
        model: model object or path to model checkpoint
        trafo: trafo object or path to trafo file
        device: device to run inference on
        roi_shift: roi_shift due to experimental data
        mode_camera: mode of input frames, either "rois" or "cameras"
        patches: patches to be applied to config
        logger: logger object or string to create logger

    Returns:
        emitter.EmitterSet: inferred emitter set
        logger: logger object or None

    """
    cfg = io.param.load(cfg) if isinstance(cfg, (str, Path)) else cfg
    cfg = io.param.patch_img_size(cfg, frame_crop, scenario=["Simulation", "Test"])

    # config patches
    if patches is not None:
        cfg = param_auto._autofill_dict(
            patches, cfg, mode_missing="include", cut_reference=True
        )

    model = (
        io.model.load_model(model, cfg, device=device[0])
        if isinstance(model, (str, Path))
        else model
    )

    if isinstance(validator, str):
        if validator != "default":
            raise ValueError(f"Unknown validator: {validator}")
        validator = setup_cfg.setup_inference_validator(cfg)

    match mode:
        case "single":
            if trafo is not None:
                raise ValueError("Trafo not needed in single mode")
            aux = None
        case "multi":
            if (trafo is None) or (isinstance(trafo, Path)):
                cfg["Paths"]["trafo"] = trafo
                trafo = setup_cfg.setup_trafo(
                    cfg,
                    device="cpu",
                    roi_shift_xy=roi_shift,
                )

            aux = indicator.trafo_aux_factory(trafo, frame_crop)
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    proc = setup_cfg.setup_processor_inference(
        cfg, cfg_sim=cfg["Test"], crop_to=frame_crop, mode_camera=mode_camera
    )

    inf = inference.Infer(
        model=model,
        window=cfg["Trainer"]["frame_window"],
        pre=proc.pre_inference,
        post=proc.post,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        logger=logger,
        validator=validator,
    )
    em_out = inf.forward(frames=frames, aux=aux)
    return em_out, inf.logger
