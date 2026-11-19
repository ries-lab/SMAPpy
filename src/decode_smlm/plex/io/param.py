import copy
from pathlib import Path
from typing import Any, IO, Union, Literal

import re

import yaml

from decode_smlm.plex.generic import validated as pydantic
from decode_smlm.plex.generic.deprecation import deprecated


class _OmegaConfLoader(yaml.SafeLoader):
    """PyYAML's safe loader with OmegaConf's two changes to it (omegaconf 2.2, ``get_yaml_loader``):
    floats written without a dot (``1e3``) are floats, and dates stay strings.  ``load`` read a
    run's settings through OmegaConf; this reads them the same without it (decode_smlm.plex,
    docs/decode/plex-plan.md §3).  Interpolations (``${...}``) stay strings, as
    ``OmegaConf.to_container`` left them."""


_OmegaConfLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:
         [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)
_OmegaConfLoader.yaml_implicit_resolvers = {
    k: [(tag, regexp) for tag, regexp in v if tag != "tag:yaml.org,2002:timestamp"]
    for k, v in _OmegaConfLoader.yaml_implicit_resolvers.items()
}


def load(p: Union[str, Path, IO[Any]]) -> dict:
    if isinstance(p, (str, Path)):
        with open(p, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=_OmegaConfLoader)
    return yaml.load(p, Loader=_OmegaConfLoader)


def load_hydra(config_name: Literal["conf", "conf_dual"] = "conf") -> dict:
    import hydra
    from omegaconf import OmegaConf

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
