import copy
import statistics
from typing import Literal, Union
from typing import Optional, Sequence

import omegaconf

from . import dev
from ..io import param


def _autofill_dict(
    x: dict, reference: dict, mode_missing: str = "include", cut_reference: bool = True
) -> dict:
    """
    Fill dict `x` with keys and values of reference if they are not present in x.

    Args:
        x: dict to be filled
        reference: reference dict
        mode_missing: what to do if there are values in x that are not present in ref
         `exclude` means that key-value pairs present in x but not in ref will not be
         part of the output dict.
         `include` means that they are.
         `raise` will raise an error if x contains more keys than reference does
        cut_reference: allow x to cut deeper hierarchy of reference if elementary
         value is set True: x: {"a": 5}, ref: {"a": {"b": 6}} -> {"a": 5}, error if
         false
    """
    out = dict()
    if mode_missing == "exclude":  # create new dict and copy from ref
        pass
    elif mode_missing == "include":
        out = copy.deepcopy(x)
    elif mode_missing == "raise":
        if not set(x.keys()).issubset(set(reference.keys())):
            raise ValueError(
                f"There are more keys in `x` than in `reference`."
                f"\n{set(x.keys()) - set(reference.keys())}"
            )
    else:
        raise ValueError(f"Not supported mode_missing type: {mode_missing}")

    for k, v in reference.items():
        if k in x and not isinstance(x[k], dict):
            # x cuts off deeper dicts of reference
            if cut_reference:
                out[k] = x[k]
            else:
                raise ValueError(
                    f"x cuts away deeper nested values of reference but "
                    f"cutting is not allowed."
                )
        elif isinstance(v, dict):
            out[k] = _autofill_dict(
                x[k] if k in x else {},
                v,
                mode_missing=mode_missing,
                cut_reference=cut_reference,
            )
        elif k in x:  # below here never going to be a dict
            out[k] = x[k]
        else:
            out[k] = reference[k]

    return out


def auto_scale(cfg: dict) -> dict:
    """
    Automatically determine scale from simulation parameters if not manually set

    Args:
        cfg:

    Returns:

    """
    cfg = copy.deepcopy(cfg)
    cfg_scale = cfg["Scaling"]
    cfg_sim = copy.deepcopy(cfg["Simulation"])  # because we interpolate some types
    n_ch = cfg_sim["channels"]

    len_dist = len(cfg_sim["Photon"]["distribution"])

    if cfg_scale["input"]["frame"]["scale"] is None:
        if len_dist >= 2:
            raise NotImplementedError(f"Cannot infer scale from ambigous distribution")
        elif "Normal" in cfg_sim["Photon"]["distribution"]:
            cfg_scale["input"]["frame"]["scale"] = [
                # possibly change to / 10
                v / 50
                for v in cfg_sim["Photon"]["distribution"]["Normal"]["loc"]
            ]
        elif "Uniform" in cfg_sim["Photon"]["distribution"]:
            # ToDo: Evaluate this
            # logic: normally gaussian / mean over 50; here uniform/2 over 50
            cfg_scale["input"]["frame"]["scale"] = [
                v / 100 for v in cfg_sim["Photon"]["distribution"]["Uniform"]["high"]
            ]

        if (n_ch >= 2) and (len(cfg_scale["input"]["frame"]["scale"]) == 1):
            cfg_scale["input"]["frame"]["scale"] *= n_ch

    if cfg_scale["input"]["frame"]["offset"] is None:
        cfg_scale["input"]["frame"]["offset"] = [
            statistics.mean(v["uniform"]) for v in cfg_sim["bg"]
        ]

    if cfg_scale["output"]["phot"]["max"] is None:
        if len_dist >= 2:
            raise NotImplementedError(
                f"Cannot infer phot scale from ambigous distribution"
            )
        elif "Normal" in cfg_sim["Photon"]["distribution"]:
            cfg_scale["output"]["phot"]["max"] = [
                v + 8 * s
                for v, s in zip(
                    cfg_sim["Photon"]["distribution"]["Normal"]["loc"],
                    cfg_sim["Photon"]["distribution"]["Normal"]["scale"],
                )
            ]
        elif "Uniform" in cfg_sim["Photon"]["distribution"]:
            cfg_scale["output"]["phot"]["max"] = [
                v * 1.2 for v in cfg_sim["Photon"]["distribution"]["Uniform"]["high"]
            ]

    if cfg_scale["output"]["bg"]["max"] is None:
        cfg_scale["output"]["bg"]["max"] = [
            max(v["uniform"]) * 1.2 for v in cfg_sim["bg"]
        ]

    if cfg_scale["output"]["z"]["max"] is None:
        cfg_scale["output"]["z"]["max"] = cfg_sim["emitter_extent"]["z"][1] * 1.2

    cfg["Scaling"] = cfg_scale
    return cfg


class AutoConfig:
    def __init__(
        self,
        fill: bool = True,
        fill_test: bool = True,
        auto_scale: bool = True,
        auto_ch_maps: bool = True,
        ref: Optional[dict] = None,
        static: Sequence[Union[str, dict]] = ("Camera",),
        return_type: Optional[Literal[dict, omegaconf.DictConfig]] = None,
    ):
        """
        Automate config handling

        Args:
            fill: fill missing values by reference
            fill_test: fill test set values by training set / simulation
            auto_scale: infer scaling parameters
            auto_ch_maps: infer model channel maps automatically. The reason to do this
             here is to know the channel maps per config file other than doing it
             implicitly.
            ref: reference dict for automatic filling
            static: list of keys or dict that should not be changed.
             E.g. ["Camera", {"Simulation": "emitter_extent"}] will not change Camera at
             top level and emitter_extent in Simulation
            return_type: return type of parsing. If none, dictionary will be returned
        """
        self._do_fill = fill
        self._do_fill_test = fill_test
        self._do_auto_scale = auto_scale
        self._do_auto_ch_maps = auto_ch_maps
        self._reference = ref if ref is not None else dict(**param.load_reference())
        self._return_type = return_type if return_type is not None else dict
        self._static = static
        self._store = {}

    def parse(self, cfg: dict) -> Union[dict, omegaconf.DictConfig]:
        self._store_static(cfg)

        cfg = self._fill(cfg)  # if self._do_fill else cfg
        cfg = self._fill_test(cfg) if self._do_fill_test else cfg
        cfg = self._auto_scale(cfg) if self._do_auto_scale else cfg
        cfg = self._auto_channel_maps(cfg) if self._do_auto_ch_maps else cfg

        cfg = self._restore_static(cfg)

        return self._convert_return_type(cfg)

    def _convert_return_type(self, cfg: dict) -> Union[dict, omegaconf.DictConfig]:
        if self._return_type is dict:
            return cfg
        if self._return_type is omegaconf.DictConfig:
            return omegaconf.OmegaConf.create(cfg)

    def _store_static(self, cfg: dict) -> None:
        # store everything
        self._store = copy.deepcopy(cfg)

    def _restore_static(self, cfg: dict) -> dict:
        # restore values by str or nested dict
        for s in self._static:
            cfg = self._restore_static_kernel(cfg, self._store, s)
        return cfg

    @classmethod
    def _restore_static_kernel(cls, cfg, store, static):
        if isinstance(static, str):
            cfg[static] = store[static]
        elif isinstance(static, dict):
            for k, v in static.items():
                cfg[k] = cls._restore_static_kernel(cfg[k], store[k], v)

        return cfg

    def _fill(self, cfg: dict) -> dict:
        # fill config by reference
        # ToDo: More sophisticated checks other than blindly including everything
        return _autofill_dict(cfg, self._reference, mode_missing="include")

    def _fill_test(self, cfg: dict) -> dict:
        # fill test set by training set config
        cfg["Test"] = _autofill_dict(
            cfg["Test"], cfg["Simulation"], mode_missing="include"
        )
        return cfg

    def _auto_scale(self, cfg: dict) -> dict:
        # fill scaling parameters by simulation
        return auto_scale(cfg)

    @dev.experimental(False)
    def _auto_channel_maps(self, cfg: dict) -> dict:
        from ..neuralfitter import spec

        win = cfg["Trainer"]["frame_window"]
        n_code = cfg["Simulation"]["codes"]
        n_ch = cfg["Simulation"]["channels"]
        # ToDo: Remove duplication with setup_cfg
        n_phot = len(
            next(
                iter(
                    next(
                        iter(cfg["Simulation"]["Photon"]["distribution"].values())
                    ).values()
                )
            )
        )
        aux_shared = cfg["Model"]["aux_shared"]
        aux_union = cfg["Model"]["aux_union"]
        separate_ch = cfg["Model"]["separate_ch"]
        separate_win = cfg["Model"]["separate_win"]

        s_in = spec.ModelChannelMapInput(
            win=win,
            n_ch=n_ch,
            n_aux=2 * n_ch if n_ch >= 2 and (aux_shared or aux_union) else 0,
        )
        gmm_map_in = spec.ModelGMMMapInput(
            s_in, separate_win, separate_ch, aux_shared, aux_union
        )
        s_out = spec.ModelChannelMapGMM(
            n_codes=n_code,
            n_channels=n_ch,
            n_phot=n_phot,
            ch_map_out=cfg["Model"]["ch_out_heads"],
        )

        cfg["Model"]["ch_in_map"] = (
            gmm_map_in.stage_shared
            if cfg["Model"]["ch_in_map"] is None
            else cfg["Model"]["ch_in_map"]
        )
        cfg["Model"]["ch_out_heads"] = (
            s_out.ch_map_out
            if cfg["Model"]["ch_out_heads"] is None
            else cfg["Model"]["ch_out_heads"]
        )

        return cfg
