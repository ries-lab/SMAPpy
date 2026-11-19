import math
import numpy as np
import torch
import torchmetrics
from scipy import stats

from torch import nn as nn


def rmse_generic(
    x: torch.Tensor,
    x_ref: torch.Tensor,
    dim: int = 0,
    re_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = x.float() - x_ref.float()
    if re_norm is not None:
        diff = diff / re_norm
    return torch.sqrt(torch.mean(diff**2, dim))


def rmse(xyz: torch.Tensor, xyz_ref: torch.Tensor) -> tuple[float, ...]:
    """
    Root mean squared distances

    Args:
        xyz:
        xyz_ref:

    Returns:
        - rmse lateral
        - rmse axial
        - rmse volumetric
    """
    num_tp = xyz.size(0)
    num_gt = xyz_ref.size(0)

    if num_tp != num_gt:
        raise ValueError("The number of points must match.")

    if xyz.size(1) not in (2, 3):
        raise NotImplementedError("Unsupported dimension")

    if num_tp == 0:
        return (torch.ones(1) * float("nan"),) * 3

    mse_loss = nn.MSELoss(reduction="sum")

    rmse_lat = (
        (mse_loss(xyz[:, 0], xyz_ref[:, 0]) + mse_loss(xyz[:, 1], xyz_ref[:, 1]))
        / num_tp
    ).sqrt()

    rmse_axial = (mse_loss(xyz[:, 2], xyz_ref[:, 2]) / num_tp).sqrt()
    rmse_vol = (mse_loss(xyz, xyz_ref) / num_tp).sqrt()

    return rmse_lat.item(), rmse_axial.item(), rmse_vol.item()


def mae(xyz: torch.Tensor, xyz_ref: torch.Tensor) -> tuple[float, ...]:
    """
    Mean absolute errors

    Args:
        xyz:
        xyz_ref:

    Returns:
        - mae lateral
        - mae axial
        - mae volumetric
    """
    num_tp = xyz.size(0)
    num_gt = xyz_ref.size(0)

    if num_tp != num_gt:
        raise ValueError("The number of points must match.")

    if xyz.size(1) not in (2, 3):
        raise NotImplementedError("Unsupported dimension")

    if num_tp == 0:
        return (torch.ones(1) * float("nan"),) * 3

    dist_fn = nn.PairwiseDistance(p=2, eps=1e-8, keepdim=False)

    mae_lat = dist_fn(xyz[:, :2], xyz_ref[:, :2]).sum() / num_tp
    mae_axial = dist_fn(xyz[:, [2]], xyz_ref[:, [2]]).sum() / num_tp
    mae_vol = dist_fn(xyz, xyz_ref).sum() / num_tp

    return mae_lat.item(), mae_axial.item(), mae_vol.item()


def mad(xyz: torch.Tensor, xyz_ref: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Mean absolute distances

    Args:
        xyz:
        xyz_ref:

    Returns:
        - mad lateral
        - mad axial
        - mad volumetric
    """
    num_tp = xyz.size(0)
    num_gt = xyz_ref.size(0)

    if num_tp != num_gt:
        raise ValueError("The number of points must match.")

    if xyz.size(1) not in (2, 3):
        raise NotImplementedError("Unsupported dimensions")

    if num_tp == 0:
        return (torch.ones(1) * float("nan"),) * 3

    mad_loss = nn.L1Loss(reduction="sum")

    mad_vol = mad_loss(xyz, xyz_ref) / num_tp
    mad_lat = (
        mad_loss(xyz[:, 0], xyz_ref[:, 0]) + mad_loss(xyz[:, 1], xyz_ref[:, 1])
    ) / num_tp
    mad_axial = mad_loss(xyz[:, 2], xyz_ref[:, 2]) / num_tp

    return mad_lat.item(), mad_axial.item(), mad_vol.item()


def precision(tp: int, fp: int) -> float:
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return math.nan


def recall(tp: int, fn: int) -> float:
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return math.nan


def jaccard(tp: int, fp: int, fn: int) -> float:
    try:
        return tp / (tp + fp + fn)
    except ZeroDivisionError:
        return math.nan


def f1(tp: int, fp: int, fn: int) -> float:
    prec = precision(tp=tp, fp=fp)
    rec = recall(tp=tp, fn=fn)
    try:
        return (2 * prec * rec) / (prec + rec)
    except ZeroDivisionError:
        return math.nan


def bootstrap_matching_se(fun: callable, **kwargs) -> tuple[float, ...]:
    """Bootstraps the standard error of a function that takes a prediction and a reference (ground truth)
    to compute a metric, e.g. rmse, by resampling matching pairs on the first axis.

    Args:
        func: function that takes a prediction and a reference (ground truth),
            indexed on the first axis
        **kwargs: keyword arguments to be passed to the function,
            typically 'xyz' and 'xyz_ref'

    Returns:
        metric standard error, or tuple thereof if multiple metrics are returned
    """
    if len(list(kwargs.values())[0]) == 0:
        return fun(**kwargs)  # same datatype as fun returns (None, nan, etc.)
    data = torch.stack(list(kwargs.values()), dim=1)
    # scipy.stats.bootstrap samples 1-dimensional data
    idxs = [torch.arange(len(list(kwargs.values())[0]))]

    def _adapted_fn(d):
        d = data[d, :, :]
        return np.array(fun(**{k: d[:, i, :] for i, k in enumerate(kwargs.keys())}))

    return tuple(
        stats.bootstrap(
            idxs, _adapted_fn, method="basic", n_resamples=1000
        ).standard_error
    )


def bootstrap_counts_se(fun: callable, **kwargs) -> float:
    """Bootstraps the standard error of a function that takes counts (e.g. true positives, false positives, etc.)
    to compute a metric, e.g. precision, by resampling counts from those probabilities.

    Args:
        func: function that takes counts arguments (e.g. true positives, false positives, etc.)
        **kwargs: keyword arguments to be passed to the function, e.g. 'tp', 'fp', 'fn'

    Returns:
        metric standard error
    """
    if all(v == 0 for v in kwargs.values()):
        return fun(**kwargs)  # same datatype as fun returns (None, nan, etc.)
    indicator_array = [np.repeat(list(kwargs.keys()), list(kwargs.values()), axis=0)]
    _adapted_fn = lambda data: fun(**{k: np.sum(data == k) for k in kwargs.keys()})
    return stats.bootstrap(
        indicator_array, _adapted_fn, method="basic", n_resamples=1000
    ).standard_error


class Metric:
    """Wrapper to have metric behave as float if no uncertainty is given,
    else as tuple of mean and standard error.
    Adds attributes 'mean' and 'se' to the metric.
    """

    def __init__(self, mean, se=None):
        self.mean = mean
        self.se = se

    def __new__(cls, mean, se=None):
        if se is None:
            return PointMetric(mean)
        return DistributionMetric(mean, se)


class PointMetric(Metric, float):
    """Wrapper to have point metric (i.e., no uncertainty) behave as float."""

    def __new__(cls, value):
        return float.__new__(cls, value)

    def __init__(self, mean, se=None):
        Metric.__init__(self, mean, se=None)

    def to_str(self, round=3):
        return f"{self.mean:.{round}f}"


class DistributionMetric(Metric, tuple):
    """Wrapper to have 'distribution' metric (i.e., with uncertainty) behave as tuple of mean and standard error."""

    def __new__(cls, mean, se):
        return tuple.__new__(cls, (mean, se))

    def __init__(self, mean, se):
        Metric.__init__(self, mean, se)

    def to_str(self, round=3):
        return f"{self.mean:.{round}f} +/- {self.se:.{round}f}"


def efficiency(jac: float, rmse: float, alpha: float) -> float:
    """
    Calculate Efficiency following Sage et al. 2019, superres fight club

    Args:
        jac (float): jaccard index 0-1
        rmse (float) RMSE value
        alpha (float): alpha value

    Returns:
        effcy (float): efficiency 0-1
    """
    return (100 - ((100 * (1 - jac)) ** 2 + alpha**2 * rmse**2) ** 0.5) / 100


def accuracy(
    c: torch.Tensor, c_ref: torch.LongTensor, num_classes: int
) -> torch.Tensor:
    """
    Accuracy for code

    Args:
        c: code out
        c_ref: code reference
        num_classes: number of codes

    Returns:
        accuracy
    """
    return torchmetrics.functional.accuracy(
        c, c_ref, task="multiclass", num_classes=num_classes
    )


def diff_scr(x: torch.Tensor, x_ref: torch.Tensor, scr: torch.Tensor) -> torch.Tensor:
    # square root crlb weighted error
    return (x - x_ref) / scr
