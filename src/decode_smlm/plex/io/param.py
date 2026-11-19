import copy
from pathlib import Path
from typing import Any, IO, Union, Literal

import hydra
import pydantic
from deprecated import deprecated
from omegaconf import OmegaConf


def load(p: Union[str, Path, IO[Any]]) -> dict:
    return OmegaConf.to_container(OmegaConf.load(p))


def load_hydra(config_name: Literal["conf", "conf_dual"] = "conf") -> dict:
    try:
        hydra.initialize(config_path="../../config")
    except ValueError:
        pass
    cfg = hydra.compose(config_name=config_name)
    cfg = OmegaConf.to_container(cfg)
    return cfg


def load_reference(config_name: Literal["conf", "conf_dual"] = "conf") -> dict:
    # alias for backwards compatibility
    return load_hydra(config_name=config_name)


def patch_img_size(
    cfg: dict,
    img_size: tuple[int, int],
    frame_extent: dict[str, tuple[float, float]] | None = None,
    psf_extent: dict[str, tuple[float, float]] | None = None,
    scenario: str | tuple[str] | list[str] = "Test",
) -> dict:
    cfg = copy.deepcopy(cfg)
    if isinstance(scenario, str):
        scenario = [scenario]

    for s in scenario:
        cfg_sim = cfg[s]
        if img_size is not None:
            cfg_sim["img_size"] = img_size
            for k, v in zip(("frame_extent", "psf_extent"), (frame_extent, psf_extent)):
                if v is not None:
                    cfg_sim[k] = v
                else:
                    cfg_sim[k]["x"] = (-0.5, img_size[0] - 0.5)
                    cfg_sim[k]["y"] = (-0.5, img_size[1] - 0.5)
    return cfg


def patch_camera(cfg: dict, em_gain: int) -> dict:
    cfg = copy.deepcopy(cfg)

    for cfg_cam in cfg["Camera"]:
        cfg_cam["specs"]["em_gain"] = em_gain

    return cfg


@deprecated(reason="Absorbed by hydra loading", version="v0.11", action="error")
def copy_reference(path: pydantic.DirectoryPath) -> tuple[Path, Path]:
    ...
