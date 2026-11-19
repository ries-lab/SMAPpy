from typing import Sequence

import torch

from torch import nn

from . import unet_param


class DoubleMUnet(nn.Module):
    p_nl = torch.sigmoid  # only in inference, during training
    phot_nl = torch.sigmoid
    xyz_nl = torch.tanh
    bg_nl = torch.sigmoid

    def __init__(
        self,
        ch_in_map: Sequence[Sequence[int]],
        ch_out: int,
        depth_shared: int = 3,
        depth_union: int = 3,
        initial_features: int = 64,
        inter_features: int = 64,
        activation=nn.ReLU(),
        use_last_nl=True,
        norm=None,
        norm_groups=None,
        norm_head=None,
        norm_head_groups=None,
        pool_mode="StrideConv",
        upsample_mode="bilinear",
        skip_gn_level=None,
        disabled_attributes=None,
    ):
        super().__init__()

        if len({len(m) for m in ch_in_map}) != 1:
            raise ValueError("All maps must have the same number of channels.")
        n_groups = len(ch_in_map)
        n_ch_group = len(ch_in_map[0])

        self.ch_in_map = ch_in_map
        self.ch_out = ch_out
        self._n_groups = n_groups
        self._n_ch_group = n_ch_group
        self._use_last_nl = use_last_nl

        self.unet_shared = unet_param.UNet2d(
            n_ch_group,
            inter_features,
            depth=depth_shared,
            pad_convs=True,
            initial_features=initial_features,
            activation=activation,
            norm=norm,
            norm_groups=norm_groups,
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            skip_gn_level=skip_gn_level,
        )

        self.unet_union = unet_param.UNet2d(
            n_groups * inter_features,
            inter_features,
            depth=depth_union,
            pad_convs=True,
            initial_features=initial_features,
            activation=activation,
            norm=norm,
            norm_groups=norm_groups,
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            skip_gn_level=skip_gn_level,
        )

        self.mt_heads = nn.ModuleList(
            [
                MLTHeads(
                    inter_features,
                    out_channels=1,
                    last_kernel=1,
                    norm=norm_head,
                    norm_groups=norm_head_groups,
                    padding=1,
                    activation=activation,
                )
                for _ in range(self.ch_out)
            ]
        )

        # convert to list
        if disabled_attributes is None or isinstance(
            disabled_attributes, (tuple, list)
        ):
            self.disabled_attr_ix = disabled_attributes
        else:
            self.disabled_attr_ix = [disabled_attributes]

    def apply_nonlin(self, o: torch.Tensor) -> torch.Tensor:
        """
        Apply non-linearity to all but the detection channel.

        Args:
            o:

        """
        raise NotImplementedError("Only implemented for single channel output")
        # Apply for phot, xyz
        p = o[:, [0]]  # leave unused
        phot = o[:, [1]]
        xyz = o[:, 2:5]

        phot = self.phot_nl(phot)
        xyz = self.xyz_nl(xyz)

        if self.ch_out == 5:
            o = torch.cat((p, phot, xyz), 1)
            return o
        elif self.ch_out == 6:
            bg = o[:, [5]]
            bg = self.bg_nl(bg)

            o = torch.cat((p, phot, xyz, bg), 1)
            return o

    def forward(self, x, force_no_p_nl=False):
        """

        Args:
            x:
            force_no_p_nl:

        Returns:

        """
        o = self._forward_core(x)

        o_head = []
        for i in range(self.ch_out):
            o_head.append(self.mt_heads[i].forward(o))
        o = torch.cat(o_head, 1)

        # apply the final non-linearities
        if not self.training and not force_no_p_nl:
            o[:, [0]] = self.p_nl(o[:, [0]])

        if self._use_last_nl:
            o = self.apply_nonlin(o)

        return o

    def _forward_core(self, x) -> torch.Tensor:
        # core, i.e. shared and union networks
        out_shared = [None] * self._n_groups

        # map input channels through shared network iteratively
        for i, ch_map in enumerate(self.ch_in_map):
            out_shared[i] = self.unet_shared.forward(x[:, ch_map, :, :])

        out_shared = torch.cat(out_shared, 1)
        out_union = self.unet_union.forward(out_shared)

        return out_union


class MLTHeads(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        last_kernel,
        norm,
        norm_groups,
        padding: int,
        activation: nn.Module,
        activation_last: nn.Module | None = None,
        low_init_bias: bool = False,
    ):
        super().__init__()
        self.norm = norm
        self.norm_groups = norm_groups

        if self.norm is not None:
            groups_1 = min(in_channels, self.norm_groups)
            groups_2 = min(1, self.norm_groups)
        else:
            groups_1 = None
            groups_2 = None

        padding = padding

        self.core = self._make_core(
            in_channels, groups_1, groups_2, activation, padding, self.norm
        )
        self.out_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=last_kernel, padding=0
        )
        self.out_act = activation_last

        if low_init_bias:
            nn.init.constant_(self.out_conv.bias, -6.0)

    def forward(self, x):
        o = self.core.forward(x)
        o = self.out_conv.forward(o)
        o = self.out_act.forward(o) if self.out_act is not None else o

        return o

    @staticmethod
    def _make_core(in_channels, groups_1, groups_2, activation, padding, norm):
        if norm == "GroupNorm":
            return nn.Sequential(
                nn.GroupNorm(groups_1, in_channels),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=padding),
                activation,
            )
        elif norm is None:
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=padding),
                activation,
            )
        else:
            raise NotImplementedError


class Clamp(nn.Module):
    def __init__(
        self,
        min_val: float | None = None,
        max_val: float | None = None,
    ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x.clone(), self.min_val, self.max_val)


class Scale(nn.Module):
    def __init__(self, scale: float, eps: float):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x):
        return self.scale * x + self.eps
