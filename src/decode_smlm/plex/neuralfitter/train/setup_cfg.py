import copy
import functools
from pathlib import Path
from typing import Optional, Sequence, Any, Union, Literal

import pydantic
import pytorch_lightning as pl
import structlog
import torch
from deprecated import deprecated
from pytorch_lightning import loggers
from torch import nn

import decode.simulation.trafo
import decode.simulation.trafo.photon.trafo
import decode.simulation.trafo.pos.sampler
from decode import emitter
from decode import evaluation
from decode import generic
from decode import neuralfitter
from decode import simulation
from decode import utils
from decode.neuralfitter import models
from decode.neuralfitter import scaling
from decode import validation as val

from decode.io.psf import load_h5
from pathlib import Path

logger = structlog.get_logger()

# common derived parameters


def n_codes(cfg_sim) -> int:
    return cfg_sim["codes"]


def codes(cfg_sim) -> list[int]:
    return list(range(n_codes(cfg_sim)))


def n_channels(cfg) -> int:
    return cfg["Simulation"]["channels"]


def n_aux(cfg) -> int:
    n_ch = n_channels(cfg)
    return 2 * n_ch if n_ch >= 2 else 0


def n_phot(cfg) -> int:
    return len(
        next(
            iter(
                next(
                    iter(cfg["Simulation"]["Photon"]["distribution"].values())
                ).values()
            )
        )
    )


def n_frames(cfg, cfg_sim) -> int:
    return cfg_sim["samples"]  # ToDo: Change on IxShift fix
    # + (cfg["Trainer"]["frame_window"] - 1)


def ch_range(cfg) -> tuple[int, int]:
    return 0, n_channels(cfg)


def px_size(cfg_cam):
    return cfg_cam[0]["specs"]["px_size"]


def setup_logger(
    cfg,
    version: Optional[str] = None,
) -> Union[loggers.logger.Logger, list[loggers.logger.Logger]]:
    """
    Set up logging.

    Args:
        cfg: config
    """
    if cfg["Logging"]["no_op"]:
        return loggers.logger.DummyLogger()

    log = []

    if "TensorBoardLogger" in cfg["Logging"]["logger"]:
        tb = neuralfitter.logger.TensorboardLogger(
            save_dir=cfg["Paths"]["logging"],
            name=None,
            version=version,
        )
        log.append(tb)
    if "WandB" in cfg["Logging"]["logger"]:
        wandb = neuralfitter.logger.WandbLogger(
            save_dir=cfg["Paths"]["logging"],
            name=None,
            version=version,
        )
        log.append(wandb)

    return log


def setup_psf(
    cfg, cfg_sim
) -> simulation.psf_kernel.CubicSplinePSF | list[simulation.psf_kernel.CubicSplinePSF]:
    psf = setup_psf_kernel(cfg, cfg_sim)

    cfg_psf = cfg["Simulation"]["PSF"]["CubicSpline"]
    ix_out = cfg_psf["channels_out"]
    n_ch = cfg_psf["channels"]

    if not isinstance(psf, (list, tuple)):
        ix_out = 0
        psf = [psf]

    # wrap flips
    if cfg_psf["flip"] is not None:
        for i, (p, fx, fy) in enumerate(
            zip(psf, cfg_psf["flip"]["x"], cfg_psf["flip"]["y"], strict=True)
        ):
            if fx:
                psf[i] = simulation.trafo.pos.shared.PipedTransformation(
                    trafo_pos=simulation.trafo.pos.trafo.XYZMirrorAt.from_frame_flip(
                        p.img_shape[0], 0
                    ),
                    psf=p,
                    trafo_frame=simulation.trafo.pos.frame.GenericFrameTransformation(
                        functools.partial(torch.flip, dims=(-2,)), device="cpu"
                    ),
                )
                logger.warning("Flipping PSF in x. This is experimental.", ix_psf=i)
            if fy:
                psf[i] = simulation.trafo.pos.shared.PipedTransformation(
                    trafo_pos=simulation.trafo.pos.trafo.XYZMirrorAt.from_frame_flip(
                        p.img_shape[1], 1
                    ),
                    psf=p,
                    trafo_frame=simulation.trafo.pos.frame.GenericFrameTransformation(
                        functools.partial(torch.flip, dims=(-1,)), device="cpu"
                    ),
                )
                logger.warning("Flipping PSF in y. This is experimental", ix_psf=i)

    if ix_out is not None:
        squeeze = True if isinstance(ix_out, int) else False
        ix_out = [ix_out] if squeeze else ix_out

        psf_in = copy.copy(psf)
        psf = [None] * len(ix_out)
        for i, ix in enumerate(ix_out):
            psf[i] = psf_in[ix]
            logger.info(
                "Reassigning or limiting PSF channels.",
                ix_in_sim=i,
                ix_nat=ix,
                channels_nat=n_ch,
            )

        psf = psf[0] if squeeze else psf

    return psf


def setup_psf_kernel(
    cfg, cfg_sim
) -> simulation.psf_kernel.CubicSplinePSF | list[simulation.psf_kernel.CubicSplinePSF]:
    from decode import io

    logger.info("Note: PSF normalization is with reference to the saved PSF.")

    n_ch = cfg_sim["PSF"]["CubicSpline"]["channels"]
    unsqueeze_pre = n_ch if n_ch >= 2 else None

    # switch between different psf
    if set(cfg_sim["PSF"].keys()) != {"CubicSpline", "format"}:
        raise NotImplementedError

    xextent = tuple(cfg_sim["psf_extent"]["x"])
    yextent = tuple(cfg_sim["psf_extent"]["y"])
    img_shape = tuple(cfg_sim["img_size"])

    if unsqueeze_pre is not None:
        xextent = (xextent,) * unsqueeze_pre
        yextent = (yextent,) * unsqueeze_pre
        img_shape = (img_shape,) * unsqueeze_pre

    kwargs_static = cfg_sim["PSF"]["CubicSpline"]["kwargs_static"]
    kwargs_static = {} if kwargs_static is None else kwargs_static

    psf = io.psf.load_spline(
        path=cfg["Paths"]["calibration"],
        xextent=xextent,
        yextent=yextent,
        img_shape=img_shape,
        norm=cfg_sim["PSF"]["CubicSpline"]["norm"],
        device=cfg["Hardware"]["device"]["simulation"],
        kwargs_static={
            "roi_size": cfg_sim["PSF"]["CubicSpline"]["roi_size"],
            "roi_auto_center": cfg_sim["PSF"]["CubicSpline"]["roi_auto_center"],
        }
        | kwargs_static,
        kwargs_multi=cfg_sim["PSF"]["CubicSpline"]["kwargs_per_channel"],
    )

    return psf


def setup_indicator(cfg_sim) -> neuralfitter.indicator.IndicatorChannelOffset:
    return neuralfitter.indicator.IndicatorChannelOffset(
        xextent=(cfg_sim["frame_extent"]["x"],),
        yextent=(cfg_sim["frame_extent"]["y"],),
        img_shape=(cfg_sim["img_size"],),
        xy_trafo=None,
    )


class _ValueSampler:
    def __init__(self, v):
        self._v = v

    def __call__(self, n):
        return torch.rand(n, len(self._v)) * self._v


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def setup_trafo(
    cfg,
    device,
    roi_shift_xy: tuple[int, int] | None = None,
):
    path_trafo = cfg["Paths"]["trafo"]
    if path_trafo is None:
        return simulation.trafo.pos.trafo.XYZTransformationNoOp(
            n_channels=n_channels(cfg), device=device
        )

    if n_channels(cfg) != 2:
        raise ValueError("Only implemented and tested for two channels.")

    reference = cfg["Simulation"]["Transformation"]["Pos"]["reference"]
    logger.info("Loading transformation.", reference=reference, path=path_trafo)

    # scale and shift because of SMAP standard
    trafo = decode.io.trafo.load_xyz_trafo(
        path_trafo,
        scale=1 / 1000.0,
        switch_xy=True,
        shift=(1.0, 1.0, 0.0),
        reference=reference,
        device=device,
    )

    # offset entire thing by ROI shift; this needs to be done before mirroring
    # because otherwise mirroring references something else
    if roi_shift_xy is not None:
        logger.info(
            "Offsetting transformation to account for ROI.", roi_shift_xy=roi_shift_xy
        )
        shift = [float(s) for s in roi_shift_xy] + [0.0]
        trafo = simulation.trafo.pos.trafo.offset_trafo(trafo, offset=shift)

    cfg_trafo_pos = cfg["Simulation"]["Transformation"]["Pos"]
    if cfg_trafo_pos["glob"]["flip"] is not None:
        size_ref = cfg_trafo_pos["glob"]["flip"]["frame_size"]
        dim = cfg_trafo_pos["glob"]["flip"]["dim"]
        ch = cfg_trafo_pos["glob"]["flip"]["channel"]

        logger.info(
            "De-mirroring transformation.", size_reference=size_ref, dim=dim, ch=ch
        )
        t_mirr = simulation.trafo.pos.trafo.XYZMirrorAt.from_frame_flip(
            size_ref,
            dim,
            device=device,
        )
        t_mirr = simulation.trafo.pos.trafo.XYZChanneledTransformation(
            t_mirr,
            ch=ch,
        )
        trafo.append(t_mirr)

    path_trafo = cfg["Paths"]["trafo"]
    path = path_trafo if isinstance(path_trafo, Path) else Path(path_trafo)
    
    if path.suffix == ".mat":
        shift_ch0 = [p[0] for p in cfg_trafo_pos["glob"]["offset"].values()]
        shift_ch1 = [p[1] for p in cfg_trafo_pos["glob"]["offset"].values()]
        logger.info("Offsets as specified in config.")
    elif path.suffix == ".h5":
        f, p = load_h5(path)
        offset = (f.res.xyshift[:,[1,0]]-0.5).astype(int).tolist()
        

        shift_ch0 = [0,0,0]
        shift_ch1 = offset[1] + [0]
        shift_ch1[1] = -shift_ch1[1]
        shift_ch1[0] = -shift_ch1[0]

        logger.info("Offsets as specified in .h5 file.")
    
    # # apply global shift of channel 1 to register both channels
    # shift_ch0 = [p[0] for p in cfg_trafo_pos["glob"]["offset"].values()]
    # shift_ch1 = [p[1] for p in cfg_trafo_pos["glob"]["offset"].values()]
    if not all([s == 0 for s in shift_ch0]):
        raise NotImplementedError("Only implemented for zero shift in channel 0.")

    trafo_shift = simulation.trafo.pos.trafo.XYZShiftTransformation(
        shift_ch1, device=device
    )
    trafo_shift = simulation.trafo.pos.trafo.XYZChanneledTransformation(trafo_shift, 1)

    # final trafo
    trafo.append(trafo_shift)
    logger.debug("Final transformation.", trafo=trafo)

    return trafo


class _DummySampler:
    def __init__(self, t, n):
        self._t = t
        self._n = n

    def sample(self):
        return [self._t] * self._n


def setup_trafo_sampler(trafo, cfg_trafo_sample, camera, roi_size, n, device):
    if cfg_trafo_sample is None or isinstance(
        trafo, decode.simulation.trafo.pos.trafo.XYZTransformationNoOp
    ):  # no-op
        return _DummySampler(trafo, n)

    # # safeguard camera sensor sizes
    # if not len(set(c.sensor_size for c in camera)) == 1:
    #     raise NotImplementedError(
    #         "Different sensor sizes for different channels not supported."
    #     )

    cfg_off = copy.deepcopy(cfg_trafo_sample["offset"])

    for dim_ix, dim in enumerate(["x", "y", "z"]):
        if (v := cfg_off[dim]) == "auto":
            cfg_off[dim] = (0, camera[0].sensor_size[dim_ix] - roi_size[dim_ix])
        elif v is not None:
            cfg_off[dim] = v
        elif v is None:
            cfg_off[dim] = (0, 1)  # no-op for int
        else:
            raise ValueError(f"Invalid offset value: {v}")

    offset = (cfg_off["x"], cfg_off["y"], cfg_off["z"])
    offset = utils.torch.UniformInt(
        low=torch.tensor([o[0] for o in offset], device=device),
        high=torch.tensor([o[1] for o in offset], device=device),
    )

    sampler = simulation.trafo.pos.sampler.TransformationOffsetSampler(
        trafo=trafo,
        offset=offset.sample,
        rebound=True,
        device=device,
        n=n,
    )
    return sampler


def setup_trafo_coord_sampler(cfg):
    raise NotImplementedError("Deprecated in favour of offset sampler.")

    if n_channels(cfg) == 1:
        return None, None

    angles = torch.as_tensor(
        list(cfg["Processing"]["coupled"]["random_angle"].values())
    )
    offsets = torch.as_tensor(
        list(cfg["Processing"]["coupled"]["random_offset"].values())
    )
    device = cfg["Hardware"]["device"]["simulation"]

    sampler = decode.simulation.trafo.pos.sampler.TransformationRotSampler(
        angle=_ValueSampler(angles),
        offset=_ValueSampler(offsets),
        n=n_frames(cfg, cfg["Simulation"]),
        device=device,
    )
    sampler_val = decode.simulation.trafo.pos.sampler.TransformationRotSampler(
        angle=_ValueSampler(angles),
        offset=_ValueSampler(offsets),
        n=n_frames(cfg, cfg["Simulation"]),
        device=device,
    )
    return sampler, sampler_val


def setup_trafo_phot(
    cfg, cfg_sim
) -> decode.simulation.trafo.photon.trafo.MultiChoricSplitter:
    cfg_choric = cfg_sim["Transformation"]["Photon"]["choric_split"]
    if cfg_choric["mean"] is None:
        return None

    t = torch.as_tensor(cfg_choric["mean"])
    t_sig = (
        torch.as_tensor(cfg_choric["std"]) if cfg_choric["std"] is not None else None
    )

    return decode.simulation.trafo.photon.trafo.MultiChoricSplitter(t, t_sig).to(
        cfg["Hardware"]["device"]["simulation"]
    )


def setup_roi(cfg, cfg_sim) -> simulation.roi.ROISampler:
    return simulation.roi.ROISampler.factory(
        img_shape=cfg_sim["img_size"],
        roi_shape=cfg["Processing"]["input"]["crop"]["roi_shape"],
        xextent=cfg_sim["frame_extent"]["x"],
        yextent=cfg_sim["frame_extent"]["y"],
        mode=cfg["Processing"]["input"]["crop"]["mode"],
        strategy=cfg["Processing"]["input"]["crop"]["strategy"],
        rref_pos=True,
    )


def setup_coupled_crop(
    cfg, cfg_sim
) -> Optional[neuralfitter.processing.coupled.CoupledCrop]:
    if cfg["Processing"]["input"] is None or "crop" not in cfg["Processing"]["input"]:
        return None

    return neuralfitter.processing.coupled.CoupledCrop(roi=setup_roi(cfg, cfg_sim))


def setup_background(
    cfg, cfg_sim
) -> simulation.background.Background | Sequence[simulation.background.Background]:
    bg = [_setup_background_core(cfg, cfg_sim, cfg_bg) for cfg_bg in cfg_sim["bg"]]

    # limit background to number of channels
    if len(bg) != n_channels(cfg):
        bg = bg[: n_channels(cfg)]
        logger.warning(
            "Limiting background to number of channels.",
            n_channels=n_channels(cfg),
            n_bg=len(bg),
        )

    if n_channels(cfg) == 1:
        bg = bg[0]

    return bg


def _setup_background_core(cfg, cfg_sim, cfg_bg) -> simulation.background.Background:
    return simulation.background.BackgroundUniform(
        bg=cfg_bg["uniform"],
        size=(n_frames(cfg, cfg_sim), *cfg_sim["img_size"]),
        device=cfg["Hardware"]["device"]["simulation"],
    )


def setup_cameras(
    cfg, device: str | torch.device
) -> Union[simulation.camera.CameraEMCCD, list[simulation.camera.CameraEMCCD]]:
    cam = []
    for cfg_cam in cfg["Camera"]:
        cam_type_str = cfg_cam["name"]

        # modify some kwargs
        cam_kwargs = copy.copy(cfg_cam["specs"])
        cam_kwargs["device"] = device
        cam_kwargs.pop("flip", None)
        cam_kwargs.pop("px_size", None)

        cam_type = getattr(simulation.camera, cam_type_str)

        cam.append(cam_type(**cam_kwargs))

    # limit to actually used channels
    if len(cam) != n_channels(cfg):
        logger.info(
            f"Using {n_channels(cfg)} of {len(cam)} cameras.",
            n_channels=n_channels(cfg),
            n_cameras=len(cam),
        )
    cam = cam[: n_channels(cfg)] if n_channels(cfg) >= 2 else cam[0]
    return cam


def setup_structure(cfg_sim) -> simulation.structures.StructurePrior:
    return simulation.structures.RandomStructure(
        xextent=cfg_sim["emitter_extent"]["x"],
        yextent=cfg_sim["emitter_extent"]["y"],
        zextent=cfg_sim["emitter_extent"]["z"],
    )


def setup_code(cfg_sim) -> Optional[simulation.code.Code]:
    if n_codes(cfg_sim) is None:
        return None
    return simulation.code.Code(codes=codes(cfg_sim))


def setup_model(cfg) -> torch.nn.Module:
    match cfg["Model"]["backbone"]:
        case "SigmaMUNet":
            return setup_model_sigmamunet(cfg)
        case "SigmaMUNetv2":
            return setup_model_sigmamunet_v2(cfg)

    raise NotImplementedError


@deprecated(version="1.0.0", reason="Use sigmamunet_v2 instead.")
def setup_model_sigmamunet(cfg) -> torch.nn.Module:
    specs = cfg["Model"]["backbone_specs"]
    activation = getattr(torch.nn, cfg["Model"]["backbone_specs"]["activation"])()
    disabled_attr = 3 if cfg["Trainer"]["train_dim"] == 2 else None
    ch_in_map = cfg["Model"]["ch_in_map"]
    if isinstance(ch_in_map[0][0], Sequence):
        ch_in_map = ch_in_map[0]

    model = neuralfitter.models.SigmaMUNet(
        ch_in_map=ch_in_map,
        ch_out_heads=cfg["Model"]["ch_out_heads"],
        ch_map=setup_ch_map_model_out(cfg),
        depth_shared=specs["depth_shared"],
        depth_union=specs["depth_union"],
        initial_features=specs["initial_features"],
        inter_features=specs["inter_features"],
        activation=activation,
        norm=specs["norm"],
        norm_groups=specs["norm_groups"],
        norm_head=specs["norm_head"],
        norm_head_groups=specs["norm_head_groups"],
        pool_mode=specs["pool_mode"],
        upsample_mode=specs["upsample_mode"],
        skip_gn_level=specs["skip_gn_level"],
        disabled_attributes=disabled_attr,
        kaiming_normal=specs["init_custom"],
    )
    return model


def setup_inference_validator(cfg) -> val.base.PipelineValidator:
    raw_in = setup_model_in_validator(cfg)
    raw_out = setup_model_out_validator(cfg)
    p = val.base.PipelineValidator.from_validator(
        steps=1, limit=3, val_model_in=[raw_in], val_model_out=[raw_out]
    )
    return p


def setup_model_in_validator(cfg) -> val.raw.generic.ChanneledValidator:
    ch_map = setup_ch_map_model_in(cfg)

    val_frame = {
        ch: val.raw.generic.LimitValidator(
            action=val.action.base.LogAction(
                level="warning", msg="Frame input above typical value range (-1, 5)"
            ),
            mass={(5.0, float("inf")): 1e-4},
        )
        for ch in ch_map.ix_frames
    }
    val_aux = {
        ch: val.raw.generic.LimitValidator(
            action=val.action.base.LogAction(
                level="warning", msg="Auxiliary outside typical value range (-2, 2)"
            ),
            mass={(float("-inf"), -2.0): 1e-6, (2.0, float("inf")): 1e-6},
        )
        for ch in ch_map.ix_aux
    }
    return val.raw.generic.ChanneledValidator(val_frame | val_aux)


def setup_model_out_validator(cfg) -> val.raw.generic.ChanneledValidator:
    ch_map = setup_ch_map_model_out(cfg)

    val_prob = {
        ch: val.raw.generic.ProbabilityMassValidator(
            action=val.action.base.LogAction(
                level="warning", msg="Unusual probability output"
            ),
            low=0.9,
            high=None,
        )
        for ch in ch_map.ix_prob
    }
    val_bg = {
        ch: val.raw.generic.LimitValidator(
            action=val.action.base.LogAction(
                level="warning", msg="Background output close to limits"
            ),
            mass={(0.9, 1.0): 0.05},
        )
        for ch in ch_map.ix_bg
    }

    return val.raw.generic.ChanneledValidator(val_prob | val_bg)


def setup_model_sigmamunet_v2(cfg) -> torch.nn.Module:
    specs = cfg["Model"]["backbone_specs"]

    # TODO: own function?
    win = cfg["Trainer"]["frame_window"]
    n_ch = cfg["Simulation"]["channels"]
    aux_shared = cfg["Model"]["aux_shared"]
    aux_union = cfg["Model"]["aux_union"]
    separate_ch = cfg["Model"]["separate_ch"]
    separate_win = cfg["Model"]["separate_win"]
    mcmi = neuralfitter.spec.ModelChannelMapInput(
        win=win,
        n_ch=n_ch,
        n_aux=2 * n_ch if n_ch >= 2 and (aux_shared or aux_union) else 0,
    )
    gmm_map_in = neuralfitter.spec.ModelGMMMapInput(
        mcmi, separate_win, separate_ch, aux_shared, aux_union
    )
    ch_map = setup_ch_map_model_out(cfg)

    # shared stage
    models_shared = nn.ModuleList(
        [
            models.factory.factory_model_core(
                in_channels=gmm_map_in.n_ch_shared,
                out_channels=specs["inter_features"],
                depth=specs["depth_shared"],
                pad_convs=True,
                initial_features=specs["initial_features"],
                activation=getattr(nn, specs["activation"])(),
                norm=specs["norm"],
                norm_groups=specs["norm_groups"],
                pool_mode=specs["pool_mode"],
                upsample_mode=specs["upsample_mode"],
                skip_gn_level=specs["skip_gn_level"],
            )
            for _ in range(gmm_map_in.n_models_shared)
        ]
    )

    # union stage
    model_union = models.factory.factory_model_core(
        in_channels=gmm_map_in.compute_ch_union(specs["inter_features"]),
        out_channels=specs["inter_features"],
        depth=specs["depth_union"],
        pad_convs=True,
        initial_features=specs["initial_features"],
        activation=getattr(nn, specs["activation"])(),
        norm=specs["norm"],
        norm_groups=specs["norm_groups"],
        pool_mode=specs["pool_mode"],
        upsample_mode=specs["upsample_mode"],
        skip_gn_level=specs["skip_gn_level"],
    )

    # heads stage
    models_heads = nn.ModuleList(
        [
            models.factory.factory_head(
                specs["inter_features"],
                out_channels=ch_out_head,
                last_kernel=1,
                norm=specs["norm_head"],
                norm_groups=specs["norm_head_groups"],
                padding=1,
                activation=getattr(nn, specs["activation"])(),
                activation_last=None,
                low_init_bias=specs["init_custom"] and i in ch_map.ix_prob,
            )
            for i, ch_out_head in enumerate(cfg["Model"]["ch_out_heads"])
        ]
    )

    # output stage
    ch_sigmoid = []
    ch_tanh = []
    for ch_name, idxs in (
        ("bg", ch_map.ix_bg),
        ("phot", ch_map.ix_phot),
        ("prob", ch_map.ix_prob),
        ("sig", ch_map.ix_sig),
        ("xyz", ch_map.ix_xyz),
    ):
        match cfg["Model"]["backbone_specs"][f"activation_{ch_name}"]:
            case "sigmoid":
                ch_sigmoid.extend(idxs)
            case "tanh":
                ch_tanh.extend(idxs)
    model_output = models.factory.factory_output(
        in_channels=sum(cfg["Model"]["ch_out_heads"]),
        ch_clamp=ch_map.ix_prob,
        ch_sigmoid=ch_sigmoid,
        ch_tanh=ch_tanh,
        ch_scale=ch_map.ix_sig,
    )

    model = models.aggregator.AggregateModel(
        models_shared,
        model_union,
        models_heads,
        model_output,
        gmm_map_in,
        specs["init_custom"],
    )
    return model


def setup_loss(cfg, cfg_sim) -> neuralfitter.loss.Loss:
    loss = neuralfitter.loss.GaussianMMLoss(
        xextent=cfg_sim["psf_extent"]["x"],
        yextent=cfg_sim["psf_extent"]["y"],
        img_shape=cfg_sim["img_size"],
        ch_map=setup_ch_map_model_out(cfg),
        device=cfg["Hardware"]["device"]["training"],
        chweight_stat=cfg["Loss"]["ch_weight"],
        forward_safety=True,
        reduction="mean",
        return_loggable=True,
    )
    return loss


def setup_optimizer(cfg) -> tuple[type, dict[str, Any]]:
    catalog = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}

    opt = catalog[cfg["Optimizer"]["name"]]
    return opt, cfg["Optimizer"]["specs"]


def setup_scheduler(cfg) -> type:
    catalog = {
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "StepLR": torch.optim.lr_scheduler.StepLR,
    }
    lr_sched = catalog[cfg["Trainer"]["schedulers"]["learning_rate"]["name"]]
    specs = cfg["Trainer"]["schedulers"]["learning_rate"]["specs"]
    specs = {} if specs is None else specs
    return lr_sched, specs


def setup_em_filter(cfg) -> emitter.process.EmitterProcess:
    if cfg["Target"]["filter"] is not None:
        f = emitter.process.EmitterFilterGeneric(**cfg["Target"]["filter"])
    else:
        f = None

    return f


def setup_input_scaling(cfg):
    ch_map = setup_ch_map_model_in(cfg)
    aux_scale = None if n_aux(cfg) == 0 else cfg["Scaling"]["input"]["aux"]["scale"]
    return decode.neuralfitter.scaling.factories.channeled_scaler_from_input(
        frame_scale=cfg["Scaling"]["input"]["frame"]["scale"],
        frame_offset=cfg["Scaling"]["input"]["frame"]["offset"],
        aux_scale=aux_scale,
        ch_map=ch_map,
    )


def setup_frame_scaling(cfg) -> neuralfitter.scaling.base.ScalerAmplitude:
    raise DeprecationWarning("Use setup_input_scaling instead.")


def setup_aux_scaling(cfg) -> Optional[scaling.base.ScalerAmplitude]:
    raise DeprecationWarning("Use setup_input_scaling instead.")


def setup_input_proc(
    cfg, cfg_sim
) -> neuralfitter.processing.model_input.ModelInputPostponed:
    if n_channels(cfg) >= 2:
        aux_gen = setup_indicator(cfg_sim).forward
    else:
        aux_gen = None

    return neuralfitter.processing.model_input.ModelInputPostponed(
        cam=None,
        aux=aux_gen,
        scaler=setup_input_scaling(cfg),
        merger_bg=simulation.background.MergerNoOp(),
    )


def setup_cam_backward_scaler(cfg, device: str | torch.device):
    cam = setup_cameras(cfg, device)
    if isinstance(cam, list):
        cam = [generic.lazy.forward_relay("backward")(c) for c in cam]
    else:
        cam = generic.lazy.forward_relay("backward")(cam)
    return cam


def setup_cam_mirr(
    cfg_cam_specs, src: Literal["gain", "channel"]
) -> neuralfitter.frame_processing.Mirror2D | None:
    if (dim := cfg_cam_specs["flip"][src]) is None:
        return None
    return neuralfitter.frame_processing.Mirror2D(dims=dim)


def setup_frame_pre_single_channel():
    pass


def setup_frame_pre_multi_roi(cfg, cfg_cam, device, scope, crop_to):
    
    path_trafo = cfg["Paths"]["trafo"]
    path = path_trafo if isinstance(path_trafo, Path) or path_trafo is None else Path(path_trafo)
    
    logger.info(
        "Setting up frame pre-processing in multi-roi mode, on a single camera."
    )
    if path is None:
        xy = None
    elif path.suffix == ".mat" and cfg[scope]["Transformation"]["Pos"]["glob"]["offset"] is not None:
        offset = cfg[scope]["Transformation"]["Pos"]["glob"]["offset"]
        if not all([o[0] == 0 for o in offset.values()]):
            raise NotImplementedError(f"Channel 0 must not be offset. Got {offset}")

        xy = offset["x"][1], offset["y"][1]
        logger.info(
            "Offsetted transformation, adjusting frame in 2nd channel.",
            offset=xy,
        )
    elif path.suffix == ".h5":
        f, p = load_h5(path)
        offset = (f.res.xyshift[:,[1,0]]-0.5).astype(int).tolist()
        offset[1][1] = -offset[1][1]
        offset[1][0] = -offset[1][0]
        xy = offset[1]
        logger.info("Offsets as specified in .h5 file.", offset=xy)
    else:
        xy = None

    pre = neuralfitter.processing.multichannel.dual_splitter_factory(
        mirr_dim_gain=cfg_cam["flip"]["gain"],
        mirr_dim_ch=cfg_cam["flip"]["channel"],
        crop_to_fold=8,
        crop_to=crop_to,
        camera=setup_cameras(cfg, device)[0],
        offset_xy=xy,
    )
    return pre


def setup_frame_pre_multi_camera(cfg, cfg_cams, device, scope, crop_to):
    logger.info("Setting up frame pre-processing in multi-camera mode.")
    
    path_trafo = cfg["Paths"]["trafo"]
    path = path_trafo if isinstance(path_trafo, Path) else Path(path_trafo)
    
    if path.suffix == ".mat" and cfg[scope]["Transformation"]["Pos"]["glob"]["offset"] is not None:
        offset = cfg[scope]["Transformation"]["Pos"]["glob"]["offset"]
        xy = [(x, y) for x, y in zip(offset["x"], offset["y"], strict=True)]
        logger.info("Offsets as specified in config.", offset=xy)
    elif path.suffix == ".h5":
        f, p = load_h5(path)
        offset = (f.res.xyshift[:,[1,0]]-0.5).astype(int).tolist()
        offset[1][1] = -offset[1][1]
        xy = [(x, y) for x, y in zip(offset[0], offset[1], strict=True)]
        logger.info("Offsets as specified in config.", offset=xy)
    else:
        xy = None

    pre = neuralfitter.processing.multichannel.multi_cam_factory(
        mirr_dim_gain=[c["specs"]["flip"]["gain"] for c in cfg_cams],
        mirr_dim_ch=[c["specs"]["flip"]["channel"] for c in cfg_cams],
        crop_to_fold=8,
        crop_to=crop_to,
        camera=setup_cameras(cfg, device),
        offset_xy=xy,
    )
    return pre


def setup_input_proc_inference(
    cfg: dict,
    crop_to: tuple[int, int] | None = None,
    scope: str = "Test",
    mode_camera: Literal["rois", "cameras"] = "rois",
) -> neuralfitter.processing.model_input.ModelInputPostponed:
    """
    Set up input processing for inference.

    Args:
        cfg: config dict
        crop_to: crop to specific size
        scope: scope of config to operate on
        mode_camera: roi

    Returns:

    """
    cfg_cam = cfg["Camera"]
    device = "cpu"
    mirr_gain = setup_cam_mirr(cfg_cam[0]["specs"], "gain")
    if mirr_gain is not None:
        logger.info("Flip due to camera specification", dim=mirr_gain.dims)
    else:
        logger.info("No flipping of frames due to camera.")
    cropper = neuralfitter.frame_processing.AutoLeftUpperCrop(8)

    match n_channels(cfg):
        case 1:
            # camera backward, de-mirror gain, and crop
            cam_back = setup_cam_backward_scaler(cfg, device=device)
            cam_back = cam_back[0] if isinstance(cam_back, list) else cam_back
            pre = (
                [cam_back] + ([mirr_gain] if mirr_gain is not None else []) + [cropper]
            )
        case 2:
            match mode_camera:
                case "rois":
                    pre = setup_frame_pre_multi_roi(
                        cfg, cfg["Camera"][0]["specs"], device, scope, crop_to
                    )
                case "cameras":
                    pre = setup_frame_pre_multi_camera(
                        cfg, cfg["Camera"], device, scope, crop_to
                    )
                case _:
                    raise NotImplementedError
        case _:
            raise NotImplementedError

    return neuralfitter.processing.model_input.ModelInputPostponed(
        frame_pre=pre,
        cam=None,
        aux=None,
        scaler=setup_input_scaling(cfg),
        merger_bg=simulation.background.MergerNoOp(),
    )


def setup_tar_scaling(cfg) -> neuralfitter.scaling.base.ScalerModelChannel:
    return neuralfitter.scaling.factories.channeled_scaler_from_tar(
        phot=cfg["Scaling"]["output"]["phot"]["max"],
        z=cfg["Scaling"]["output"]["z"]["max"],
        ch_map=setup_ch_map_tar(n_phot(cfg)),
    )


def setup_bg_scaling(cfg) -> scaling.base.ScalerModelChannel:
    return decode.neuralfitter.scaling.factories.channeled_scaler_from_tar_bg(
        bg=cfg["Scaling"]["output"]["bg"]["max"]
    )


def setup_ch_map_model_in(cfg):
    return neuralfitter.spec.ModelChannelMapInput(
        win=cfg["Trainer"]["frame_window"],
        n_ch=n_channels(cfg),
        n_aux=n_aux(cfg),
    )


def setup_ch_map_model_out(cfg):
    return neuralfitter.spec.ModelChannelMapGMM(
        n_codes=n_codes(cfg["Simulation"]),
        n_channels=n_channels(cfg),
        n_phot=n_phot(cfg),
        ch_map_out=cfg["Model"]["ch_out_heads"],
    )


def setup_ch_map_tar(n):
    return neuralfitter.spec.tar.MapListTar(
        n_phot=n,
    )


def setup_post_model_scaling(cfg) -> scaling.base.ScalerModelChannel:
    return decode.neuralfitter.scaling.factories.channeled_scaler_from_output(
        ch_map=setup_ch_map_model_out(cfg),
        phot=cfg["Scaling"]["output"]["phot"]["max"],
        z=cfg["Scaling"]["output"]["z"]["max"],
        bg=cfg["Scaling"]["output"]["bg"]["max"],
        sigma_eps=0.0,  # internalised in the model at the moment
        sigma_factor=1.0,
        device=cfg["Hardware"]["device"]["training"],
    )


def setup_post_process(cfg, cfg_sim) -> neuralfitter.processing.post.PostProcessing:
    post = neuralfitter.processing.post.PostProcessingGaussianMixture(
        scaler=setup_post_model_scaling(cfg),
        coord_convert=setup_post_process_offset(cfg, cfg_sim),
        frame_to_emitter=setup_post_process_frame_emitter(cfg, cfg_sim),
        ch_map=setup_ch_map_model_out(cfg),
    )
    return post


def _extent_on_crop(cfg, cfg_sim):
    xextent = cfg_sim["frame_extent"]["x"]
    yextent = cfg_sim["frame_extent"]["y"]
    img_shape = cfg_sim["img_size"]

    if cfg["Processing"]["input"] is not None and "crop" in cfg["Processing"]["input"]:
        raise NotImplementedError(f"Not used anymore.")
        crop = cfg["Processing"]["input"]["crop"]
        logger.info(
            "Inferring extent automatically from crop, assuming px_size of 1.",
            crop=crop,
        )

        xextent = (-0.5, crop["roi_shape"][0] - 0.5)
        yextent = (-0.5, crop["roi_shape"][1] - 0.5)
        img_shape = crop["roi_shape"]

    return xextent, yextent, img_shape


def setup_post_process_offset(
    cfg, cfg_sim
) -> neuralfitter.coord_transform.Offset2Coordinate:
    xextent, yextent, img_shape = _extent_on_crop(cfg, cfg_sim)

    return neuralfitter.coord_transform.Offset2Coordinate(
        xextent=xextent,
        yextent=yextent,
        img_shape=img_shape,
    )


def setup_post_process_frame_emitter(
    cfg, cfg_sim
) -> neuralfitter.processing.to_emitter.ToEmitter:
    # last bit that transforms frames to emitters

    if cfg["Processing"]["post"] is None:
        post = neuralfitter.processing.to_emitter.ToEmitterEmpty(
            # ToDo: Generalise
            xy_unit=cfg_sim["xy_unit"],
            px_size=px_size(cfg["Camera"]),
        )

    elif "LookUp" in cfg["Processing"]["post"]:
        raise NotImplementedError

    elif "SpatialIntegration" in cfg["Processing"]["post"]:
        post = neuralfitter.processing.to_emitter.ToEmitterSpatialIntegration(
            mask=cfg["Processing"]["post"]["SpatialIntegration"]["raw_th"],
            ch_map=setup_ch_map_model_out(cfg),
            xy_unit=cfg_sim["xy_unit"],
            px_size=px_size(cfg["Camera"]),
        )
    else:
        raise NotImplementedError(
            f"Post-processing method not implemented. "
            f"Post cfg is {cfg['Procesing']['post']}"
        )

    return post


def setup_eval_filter(cfg) -> emitter.process.EmitterFilterGeneric:
    return emitter.process.EmitterFilterGeneric(**cfg["Evaluation"]["filter"])


def setup_matcher(cfg) -> evaluation.match_emittersets.EmitterMatcher:
    matcher = evaluation.match_emittersets.GreedyHungarianMatching(
        match_dims=cfg["Evaluation"]["match_dims"],
        dist_lat=cfg["Evaluation"]["dist_lat"],
        dist_ax=cfg["Evaluation"]["dist_ax"],
        dist_vol=cfg["Evaluation"]["dist_vol"],
    )
    return matcher


def setup_evaluator(cfg) -> evaluation.evaluation.SMLMEvaluation:
    # matcher = setup_matcher(cfg)
    dist_eval = evaluation.evaluation.DistanceEvaluation(
        num_codes=cfg["Simulation"]["codes"]
    )
    evaluator = evaluation.evaluation.SMLMEvaluation(dist_eval=dist_eval)
    return evaluator


def setup_emitter_flux(cfg_sim):
    allowed_dist = {"Normal", "Uniform", "normal", "uniform"}

    if len(dist := cfg_sim["Photon"]["distribution"]) >= 2:
        raise ValueError(f"Setup not implemented for multiple distributions.")
    if (k := next(iter(dist))) not in allowed_dist:
        raise ValueError(f"Setup not implemented for {k} distribution.")

    dist = dist[k]
    if not k[0] == k[0].upper():
        k = k.capitalize()
        logger.warning("Fixing capitalisation of distribution. Please change", k=k)
    flux = getattr(torch.distributions, k)(
        **{kk: torch.as_tensor(vv, dtype=torch.float) for kk, vv in dist.items()}
    )

    return flux


def setup_emitter_sampler(
    cfg, cfg_sim, flux
) -> simulation.sampler.EmitterSamplerBlinking:
    """
    Get emitter samplers

    Args:
        cfg: config

    Returns:
        sampler
    """

    em_sampler = simulation.sampler.EmitterSamplerBlinking(
        structure=setup_structure(cfg_sim),
        code=setup_code(cfg_sim),
        flux=flux,
        em_num=cfg_sim["emitter_avg"],
        lifetime=cfg_sim["lifetime_avg"],
        frame_range=n_frames(cfg, cfg_sim),
        xy_unit=cfg_sim["xy_unit"],
        px_size=px_size(cfg["Camera"]),
    )
    return em_sampler


def setup_microscope(
    cfg, cfg_sim, cam, ignore_trafo: bool = True
) -> Union[
    simulation.microscope.Microscope, simulation.microscope.MicroscopeMultiChannel
]:
    if n_channels(cfg) == 1:
        return simulation.microscope.Microscope(
            psf=setup_psf(cfg, cfg_sim),
            noise=cam,
            frame_range=None,
        )
    device_trafo = cfg["Hardware"]["device"]["simulation"]
    if not ignore_trafo:
        trafo_xyz = setup_trafo(
            cfg,
            device=device_trafo,
            reference=cfg_sim["Transformation"]["Pos"]["reference"],
        )
    else:
        trafo_xyz = None

    return simulation.microscope.MicroscopeMultiChannel(
        psf=setup_psf(cfg, cfg_sim),
        noise=cam,
        trafo_xyz=trafo_xyz,
        trafo_phot=setup_trafo_phot(cfg, cfg_sim),
        frame_range=None,
        ch_range=ch_range(cfg),
    )


def setup_tar(cfg) -> neuralfitter.target_generator.TargetGenerator:
    scaler = setup_tar_scaling(cfg)
    filter = setup_em_filter(cfg)
    bg_lane = setup_bg_scaling(cfg)

    if (nc := n_codes(cfg["Simulation"])) == 1:
        range_code = None
    else:
        range_code = (0, nc)

    return neuralfitter.target_generator.TargetGaussianMixture(
        n_max=cfg["Target"]["max_emitters"],
        range_code=range_code,
        ix_low=None,
        ix_high=None,
        ignore_ix=True,
        scaler=scaler,
        filter=filter,
        aux_lane=bg_lane,
    )


def setup_processor(cfg, cfg_sim) -> neuralfitter.process.Processing:
    model_input = setup_input_proc(cfg, cfg_sim)
    tar = setup_tar(cfg)
    coupled = setup_coupled_crop(cfg, cfg_sim)
    filter_em = setup_em_filter(cfg)
    post_processor = setup_post_process(cfg, cfg_sim)

    return neuralfitter.process.Processing(
        m_input=model_input,
        coupled=coupled,
        tar=tar,
        tar_em=filter_em,
        post_model=None,
        post=post_processor,
    )


def setup_processor_inference(
    cfg,
    cfg_sim,
    crop_to: tuple[int, int] | None = None,
    mode_camera: Literal["rois", "cameras"] = "rois",
) -> neuralfitter.process.Processing:
    model_input = setup_input_proc_inference(
        cfg, crop_to=crop_to, mode_camera=mode_camera
    )
    post_processor = setup_post_process(cfg, cfg_sim)

    return neuralfitter.process.Processing(
        m_input=model_input,
        post=post_processor,
    )


def setup_sampler_microscope(
    cfg, cfg_sim, cam
) -> (
    neuralfitter.sampler.SamplerMicroscope
    | simulation.microscope.Microscope
    | simulation.microscope.MicroscopeMultiChannel
):
    if n_channels(cfg) == 1:
        return setup_microscope(cfg, cfg_sim, cam, ignore_trafo=True)

    return neuralfitter.sampler.SamplerMicroscope(
        mic_common=setup_microscope(cfg, cfg_sim, cam, ignore_trafo=True),
        win=cfg["Trainer"]["frame_window"],
    )


def setup_sampler_physical(
    cfg, cfg_sim, cam
) -> neuralfitter.sampler.SamplerIndependents:
    flux = setup_emitter_flux(cfg_sim)
    sampler_em = setup_emitter_sampler(cfg, cfg_sim, flux)
    sample_bg = setup_background(cfg, cfg_sim)

    device = cfg["Hardware"]["device"]["simulation"]

    if (n_ch := n_channels(cfg)) == 1:
        sampler_trafo = None
    elif cfg_sim["Transformation"]["Pos"]["model"] is None:
        trafo = simulation.trafo.pos.trafo.XYZTransformationNoOp(n_channels=n_ch)
        sampler_trafo = simulation.trafo.pos.sampler.TransformationSamplerNoOp(
            trafo, n=n_frames(cfg, cfg_sim)
        )
    else:
        trafo = setup_trafo(cfg, device)
        sampler_trafo = setup_trafo_sampler(
            trafo,
            cfg_sim["Transformation"]["Pos"]["sampled"],
            camera=cam,
            roi_size=cfg_sim["img_size"],
            n=n_frames(cfg, cfg_sim),
            device=device,
        )

    sampler = neuralfitter.sampler.SamplerIndependents(
        em=sampler_em,
        bg=sample_bg,
        trafo=sampler_trafo,
    )

    return sampler


def setup_sampler(cfg, cfg_sim) -> neuralfitter.sampler.SamplerTraining:
    proc = setup_processor(cfg, cfg_sim)
    cam = setup_cameras(cfg, cfg["Hardware"]["device"]["simulation"])
    sim = setup_sampler_physical(cfg, cfg_sim, cam=cam)
    mic = setup_sampler_microscope(cfg, cfg_sim, cam=cam)

    sampler = neuralfitter.sampler.SamplerTraining(
        proc=proc,
        sampler_physical=sim,
        sampler_microscope=mic,
        window=cfg["Trainer"]["frame_window"],
        ix_low=0,
        ix_high=n_frames(cfg, cfg_sim),
        device=cfg["Hardware"]["device"]["simulation"],
    )

    return sampler


def setup_callbacks(cfg, path_exp: Path) -> list[pl.callbacks.Callback]:
    m_ckpt = pl.callbacks.ModelCheckpoint(dirpath=path_exp)
    lr_mtr = pl.callbacks.LearningRateMonitor("epoch")
    callbacks = [m_ckpt, lr_mtr]
    for add_cbks in cfg.Trainer.callbacks or []:
        for cbk_mod, cbk_kwargs in add_cbks.items():
            # first take from own callbacks then from pytorch lightning
            if hasattr(neuralfitter.callbacks, cbk_mod):
                cb = getattr(neuralfitter.callbacks, cbk_mod)
            else:
                cb = getattr(pl.callbacks, cbk_mod)
            cb = cb(
                **{
                    # allow using path_exp in kwargs, e.g. for ModelCheckpoint dirpath
                    k: v.format(path_exp=path_exp) if isinstance(v, str) else v
                    for k, v in cbk_kwargs.items()
                }
            )
            callbacks.append(cb)
    return callbacks
