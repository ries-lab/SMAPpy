import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from deprecated import deprecated

from . import utils


def xy_kerneled(x, y, kde=True, ax=None, nan_okay=True):
    if ax is None:
        ax = plt.gca()

    if len(x) == 0:
        return ax

    if not torch.isnan(x).any():
        if kde:
            utils.kde_sorted(x, y, True, ax, sub_sample=10000, nan_inf_ignore=True)
        else:
            ax.plot(x, y, "x")
    else:
        if not nan_okay:
            raise ValueError(f"Some of the values are NaN.")

    return ax


def deviation_dist(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    residuals=False,
    kde=True,
    ax=None,
    nan_okay=True,
    xlabel="reference",
    ylabel="prediction",
):
    # ToDo: Deprecate this.
    """Plot predicted values over reference values"""
    if ax is None:
        ax = plt.gca()

    if len(x) == 0:
        ax.set_ylabel("no data")
        return ax

    if residuals:
        x = x - x_gt

    if not torch.isnan(x).any():
        if kde:
            utils.kde_sorted(x_gt, x, True, ax, sub_sample=10000, nan_inf_ignore=True)
        else:
            ax.plot(x_gt, x, "x")

    else:
        if not nan_okay:
            raise ValueError(f"Some of the values are NaN.")

    if residuals:
        ax.plot([x_gt.min(), x_gt.max()], [0, 0], "green")
        ax.set_ylabel(ylabel)

    else:
        ax.plot([x_gt.min(), x_gt.max()], [x_gt.min(), x_gt.max()], "green")
        ax.set_ylabel(ylabel)

    ax.set_xlabel(xlabel)
    return ax


@deprecated(
    reason="this should be superseded by inverse of Offset2Coordinate", version="0.11"
)
def px_pointer_dist(
    pointer, px_border: float, px_size: float, return_ix: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Get within pixel dist

    Args:
        pointer:
        px_border: lower limit of pixel (most commonly -0.5)
        px_size: size of pixel (most commonly 1.)

    Returns:

    """
    x = (pointer - px_border) % px_size + px_border
    # return index as well, such that ix + offset = pointer (for -0.5, 1)
    ix = (pointer - px_border) // px_size

    if return_ix:
        return x, ix

    return x


def emitter_deviations(
    tp, tp_match, px_border: float, px_size: float, axes, residuals=False, kde=True
):
    """Plot within px distribution"""
    assert len(axes) == 4

    # xy within
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.distplot(
            px_pointer_dist(tp.xyz_px[:, 0], px_border=px_border, px_size=px_size),
            norm_hist=True,
            ax=axes[0],
            bins=50,
        )
        sns.distplot(
            px_pointer_dist(tp.xyz_px[:, 1], px_border=px_border, px_size=px_size),
            norm_hist=True,
            ax=axes[1],
            bins=50,
        )

    # z and photons
    deviation_dist(
        tp.xyz_nm[:, 2], tp_match.xyz_nm[:, 2], residuals=residuals, kde=kde, ax=axes[2]
    )
    deviation_dist(tp.phot, tp_match.phot, residuals=residuals, kde=kde, ax=axes[3])
