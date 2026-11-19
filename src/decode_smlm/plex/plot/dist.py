# some default plotting for deviations
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import torch

from typing import Optional


def deviation(
    diff: torch.Tensor,
    stat="density",
    fit: Optional[str] = "norm",
    kde=False,
    ax=None,
    alpha_hist: float = 0.4,
    kwargs_hist=None,
    kwargs_fit=None,
):
    kwargs_hist = {} if kwargs_hist is None else kwargs_hist
    kwargs_fit = {} if kwargs_fit is None else kwargs_fit

    ax = plt.gca() if ax is None else ax
    ax = sns.histplot(
        diff.numpy(), stat=stat, kde=kde, ax=ax, alpha=alpha_hist, **kwargs_hist
    )

    if fit:
        fit = getattr(scipy.stats, fit)
        params = fit.fit(diff, **kwargs_fit)
        x = torch.linspace(*ax.get_xlim(), 10000)
        y = fit.pdf(x, *params)

        params_str = ", ".join([f"{p:.3f}" for p in params])
        ax.plot(x, y, label=f"{fit.name} ({params_str})")

    return ax
