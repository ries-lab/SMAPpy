from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Union

import math
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import loggers
from pytorch_lightning.utilities import rank_zero_only
from structlog import get_logger

try:
    import wandb
except ImportError:
    wandb = None

from .. import emitter
from ..plot import plot
from ..evaluation import predict_dist


logger_console = get_logger(__name__)


def _channel_metrics(metrics: dict[str, float | Sequence[float]]) -> dict:
    m_out = {k: v for k, v in metrics.items() if not isinstance(v, Sequence)}
    m_out.update(
        {
            f"{k}_{i}": v[i]
            for k, v in metrics.items()
            if isinstance(v, Sequence)
            for i in range(len(v))
        }
    )
    return m_out


class PrefixDictMixin(ABC):
    @abstractmethod
    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None
    ) -> None:
        ...

    def log_group(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """

        Args:
            metrics: dictionary of metrics to log
            step:
            prefix: prefix to add before metric name
        """
        if prefix is not None:
            metrics = {prefix + k: v for k, v in metrics.items()}

        return self.log_metrics(metrics=metrics, step=step)


class LogTensorMixin(ABC):
    @abstractmethod
    def log_figure(self, figure: plt.figure, name: str, step: int, close: bool):
        ...

    def log_tensor(
        self,
        t: Union[torch.Tensor, Sequence[torch.Tensor]],
        name: str,
        step: Optional[int] = None,
        unbind: Optional[int] = None,
        colormap: str = "gray",
        colorbar: bool = True,
    ):
        if unbind is not None:
            t = torch.unbind(t, dim=unbind)
        t = [t] if not isinstance(t, Sequence) else t
        t = [tt.detach().cpu() for tt in t]

        for i, tt in enumerate(t):
            f, ax = plt.subplots()
            cax = ax.matshow(tt.numpy(), cmap=colormap)
            if colorbar:
                plt.colorbar(cax)
            self.log_figure(name=f"{name}/{i}", figure=f, step=step, close=True)


class Logger(PrefixDictMixin, loggers.logger.Logger, ABC):
    def log_figure(
        self,
        name: str,
        figure: plt.figure,
        step: Optional[int] = None,
        close: bool = True,
    ) -> None:
        """
        Logs a matplotlib figure.
        Args:
            name: name of the figure
            figure: plt figure handle
            step: step number at which the figure should be recorded
            close: close figure after logging
        """
        if close:
            plt.close(figure)

    def log_emitter(
        self,
        name: str,
        em: Optional[emitter.EmitterSet] = None,
        em_tar: Optional[emitter.EmitterSet] = None,
        frame: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ):
        plot.PlotFrameCoord(
            frame=frame.cpu(),
            pos_out=em.xyz_px if em is not None else None,
            pos_tar=em_tar.xyz_px if em_tar is not None else None,
        ).plot()
        self.log_figure(name=name, figure=plt.gcf(), step=step, close=True)

    def log_deviation(
        self,
        name: str,
        x: torch.Tensor,
        x_ref: torch.Tensor,
        xlabel: str = "reference",
        ylabel: str = "prediction",
        step: Optional[int] = None,
    ):
        f, ax = plt.subplots()
        predict_dist.deviation_dist(
            x=x, x_gt=x_ref, xlabel=xlabel, ylabel=ylabel, ax=ax
        )
        self.log_figure(name=name, figure=f, step=step, close=True)


class TensorboardLogger(loggers.TensorBoardLogger, LogTensorMixin, Logger):
    def __init__(
        self,
        save_dir: str | Path,
        name: str | None = "lightning_logs",
        version: int | str | None = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: str | Path | None = None,
        drop_nan: bool = True,
        drop_inf: bool = True,
        **kwargs,
    ):
        super().__init__(
            save_dir=save_dir,
            name=name,
            version=version,
            log_graph=log_graph,
            default_hp_metric=default_hp_metric,
            prefix=prefix,
            sub_dir=sub_dir,
            **kwargs,
        )
        self.drop_nan = drop_nan
        self.drop_inf = drop_inf

    @rank_zero_only
    def log_figure(
        self,
        name: str,
        figure: plt.figure,
        step: Optional[int] = None,
        close: bool = True,
    ) -> None:
        self.experiment.add_figure(
            tag=name, figure=figure, global_step=step, close=close
        )

    @rank_zero_only
    def log_hist(self, name: str, vals: torch.Tensor, step: Optional[int] = None):
        self.experiment.add_histogram(tag=name, values=vals, global_step=step)

    def log_metrics(
        self, metrics: dict[str, float | Sequence[float]], step: Optional[int] = None
    ) -> None:
        metrics = _channel_metrics(metrics)
        if self.drop_nan:
            metrics = {k: v for k, v in metrics.items() if not math.isnan(v)}
        if self.drop_inf:
            metrics = {k: v for k, v in metrics.items() if not math.isinf(v)}
        super().log_metrics(metrics=metrics, step=step)


class WandbLogger(loggers.WandbLogger, LogTensorMixin, Logger):
    @rank_zero_only
    def log_figure(
        self,
        name: str,
        figure: plt.figure,
        step: Optional[int] = None,
        close: bool = True,
    ) -> None:
        # we don't need to catch optional deps `wandb` as lightning does that already
        # wandb.Image fix because some plotly conversion fails apparently
        self.experiment.log({name: wandb.Image(figure), "trainer/global_step": step})
        plt.close(figure) if close else None

    @rank_zero_only
    def log_hist(self, name: str, vals: torch.Tensor, step: Optional[int] = None):
        logger_console.warning("WandbLogger has not yet implemented histograms.")
