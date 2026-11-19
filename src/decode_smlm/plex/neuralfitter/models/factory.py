from torch import nn

from decode.neuralfitter import utils
from . import model_param, unet_param


def factory_model_core(
    in_channels: int,
    out_channels: int,
    depth: int = 3,
    pad_convs: bool = True,
    initial_features: int = 64,
    activation: int = nn.ReLU(),
    norm=None,
    norm_groups=None,
    pool_mode="StrideConv",
    upsample_mode="bilinear",
    skip_gn_level=None,
):
    return unet_param.UNet2d(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        pad_convs=pad_convs,
        initial_features=initial_features,
        activation=activation,
        norm=norm,
        norm_groups=norm_groups,
        pool_mode=pool_mode,
        upsample_mode=upsample_mode,
        skip_gn_level=skip_gn_level,
    )


def factory_head(
    in_channels: int,
    out_channels: int,
    last_kernel=1,
    norm=None,
    norm_groups=None,
    padding=1,
    activation=nn.ReLU(),
    activation_last=None,
    low_init_bias=False,
):
    return model_param.MLTHeads(
        in_channels,
        out_channels,
        last_kernel,
        norm,
        norm_groups,
        padding,
        activation,
        activation_last,
        low_init_bias,
    )


def factory_output(
    in_channels: int,
    ch_clamp: list | None = None,
    ch_sigmoid: list | None = None,
    ch_tanh: list | None = None,
    ch_scale: list | None = None,
    clamp_min: float = -8.0,
    clamp_max: float = 8.0,
    sigma_scale: float = 3.0,
    sigma_eps: float = 1e-3,
):
    activations = [nn.Sequential() for _ in range(in_channels)]
    for ix in ch_clamp if ch_clamp is not None else []:
        activations[ix].add_module(
            f"clamp_act_{ix}", model_param.Clamp(min_val=clamp_min, max_val=clamp_max)
        )
    for ix in ch_sigmoid if ch_sigmoid is not None else []:
        activations[ix].add_module(f"sigmoid_act_{ix}", nn.Sigmoid())
    for ix in ch_tanh if ch_tanh is not None else []:
        activations[ix].add_module(f"tanh_act_{ix}", nn.Tanh())
    for ix in ch_scale if ch_scale is not None else []:
        activations[ix].add_module(
            f"scale_act{ix}", model_param.Scale(scale=sigma_scale, eps=sigma_eps)
        )
    activation_layer = utils.activation.ChanneledActivation(activations)
    return activation_layer
