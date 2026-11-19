from __future__ import annotations

import copy
import contextlib
import pathlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Union

import torch
from decode_smlm.plex.generic.deprecation import deprecated

from . import param
from ..generic.logging import get_logger
from ..neuralfitter.train import setup_cfg

if TYPE_CHECKING:
    from ..neuralfitter import model as model_lightning
from ..utils import dev

logger = get_logger(__name__)


@contextlib.contextmanager
def _patch_posix_path_unpickling_on_windows():
    if not sys.platform.startswith("win"):
        yield
        return

    posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        pathlib.PosixPath = posix_path


@dev.experimental()
def load_model(
    path_weights: Union[str, Path],
    cfg: Union[dict, str, Path] = None,
    device: Union[str, torch.device] = "cpu",
    scenario: str = "Test",
    img_size: tuple[int, int] | None = None,
    frame_extent: dict[str, tuple[float, float]] | None = None,
) -> torch.nn.Module:
    cfg = param.load(cfg) if not isinstance(cfg, dict) else cfg

    if img_size is not None:
        logger.info(
            "Patching img_size in config",
            img_size_new=img_size,
            scenario=scenario,
        )
        cfg = param.patch_img_size(cfg, img_size, frame_extent, scenario)

    model = setup_cfg.setup_model(cfg)
    model.load_state_dict(load_state_dict(path_weights, device=device))
    return model.to(device)


def load_state_dict(path_ckpt: Union[str, Path], device: Union[str, torch.device] = "cpu") -> dict:
    """The network's weights from a Lightning checkpoint, without Lightning.

    decode_smlm.plex (docs/decode/plex-plan.md §3): the checkpoint's ``state_dict`` holds the
    Lightning module's, whose network is its ``_model``; that prefix is taken off.  It is read
    with ``weights_only``, so nothing in the file runs: of the classes it names beside tensors,
    only ``pathlib.PosixPath`` (a callback's directory) needs allowing, read as ``PurePosixPath``,
    which also reads on Windows, where the original had to patch ``pathlib`` instead.
    """
    with torch.serialization.safe_globals([(pathlib.PurePosixPath, "pathlib.PosixPath")]):
        ckpt = torch.load(path_ckpt, map_location=device, weights_only=True)
    state = ckpt["state_dict"]
    prefix = "_model."
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


@dev.experimental(level="warning")
def load_pipeline(
    path_ckpt: Union[str, Path],
    cfg: Union[dict, str, Path] = None,
    device: Union[str, torch.device] = "cpu",
    scenario: str = "Test",
    img_size: tuple[int, int] | None = None,
    frame_extent: dict[str, tuple[float, float]] | None = None,
) -> model_lightning.Model:
    cfg = param.load(cfg) if not isinstance(cfg, dict) else copy.deepcopy(cfg)

    if img_size is not None:
        logger.info(
            "Patching img_size in config",
            img_size_new=img_size,
            scenario=scenario,
        )
        cfg = param.patch_img_size(cfg, img_size, frame_extent, scenario)

    from ..neuralfitter import model as model_lightning  # Lightning: training only

    model = setup_cfg.setup_model(cfg)
    proc_train = setup_cfg.setup_processor(cfg, cfg["Simulation"])
    proc_val = setup_cfg.setup_processor(cfg, cfg["Test"])

    with _patch_posix_path_unpickling_on_windows():
        model_wrap = model_lightning.Model.load_from_checkpoint(
            path_ckpt,
            model=model,
            proc_train=proc_train,
            proc_val=proc_val,
            batch_size=None,
            map_location=device,
        )

    return model_wrap


@deprecated(version="0.11", reason="deprecated", action="error")
def hash_model(modelfile):
    pass


@deprecated(version="0.11", reason="deprecated", action="error")
class LoadSaveModel:
    pass
