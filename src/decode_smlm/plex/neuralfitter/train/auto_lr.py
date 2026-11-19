import copy
import logging
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import isolate_rng
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.utilities.parsing import lightning_setattr
from pytorch_lightning.tuner.lr_finder import _LRFinder
from typing import Literal


class _AltLRFinder(_LRFinder):
    def __init__(
        self,
        mode: str,
        lr_min: float,
        lr_max: float,
        num_lrs: int,
        num_steps: int,
        rel_tol: float,
    ):
        """Alternative LR finder that takes the biggest learning rate within a relative tolerance of the lowest loss."""
        self.num_lrs = num_lrs
        self.num_steps = num_steps
        self.rel_tol = rel_tol
        super().__init__(mode, lr_min, lr_max, num_lrs)
        self.results.update({"lr": [], "loss": []})

    def suggestion(self) -> float:
        """Suggests the biggest learning rate that lead to a loss within a relative tolerance of the best reached loss."""
        losses = np.array(self.results["loss"])
        best_loss = np.nanmin(losses)
        sugg_lr_idx = np.argwhere(
            (losses <= best_loss * (1 + self.rel_tol)) * (~np.isnan(losses))
        )
        sugg_lr_idx = sugg_lr_idx[-1][0]
        lr = self.results["lr"][sugg_lr_idx]
        if sugg_lr_idx in (0, len(self.results["lr"])):
            raise ValueError(
                f"Best learning rate found at the edge (lr={lr}). Increase or move the search range."
            )
        return lr


def alternative_lr_finder(
    trainer_specs: dict,
    model: pl.LightningModule,
    dm: pl.LightningDataModule,
    min_lr: float = 1e-6,
    max_lr: float = 1e-2,
    num_lrs: int = 15,
    num_steps: int = 10,
    rel_tol: float = 0.5,
    mode: Literal["linear", "exponential"] = "exponential",
    lr_attr_name: str = "_lr",
) -> _AltLRFinder:
    """Alternative to the PyTorch Lightning LR finder, specific for DECODE.
    Instead of running only one step per learning rate, it runs `num_steps` steps.
    This allows more stable results, especially considering that DECODE needs some steps to reach meaningful results.
    It suggests the biggest learning rate that lead to a loss within a relative tolerance of the best reached loss.
    """
    with isolate_rng():
        lr_finder = _AltLRFinder(
            mode=mode,
            lr_min=min_lr,
            lr_max=min_lr,
            num_lrs=num_lrs,
            num_steps=num_steps,
            rel_tol=rel_tol,
        )

        dl = dm.train_dataloader()
        num_workers = dl.num_workers
        dl.num_workers = 0
        trainer_specs = copy.deepcopy(trainer_specs)
        trainer_specs.update(
            {
                "callbacks": [],
                "logger": DummyLogger(),
                "enable_progress_bar": False,
                "enable_checkpointing": False,
                "max_steps": num_steps,
            }
        )

        lrs = (
            np.logspace(np.log10(min_lr), np.log10(max_lr), num_lrs)
            if mode == "exponential"
            else np.linspace(min_lr, max_lr, num_lrs)
        )

        for lr in lrs:
            # Disable logs
            log = logging.getLogger("pytorch_lightning")
            log_level = log.level
            log.propagate = False
            log.setLevel(logging.ERROR)

            trainer = pl.Trainer(**trainer_specs)

            m_new = copy.deepcopy(model)
            setattr(m_new, lr_attr_name, lr)

            trainer.fit(model=m_new, train_dataloaders=dl)

            loss = trainer.fit_loop.running_loss.last()
            if loss is not None:
                loss = loss.item()
            lr_finder.results["lr"].append(lr)
            lr_finder.results["loss"].append(loss)

        # Reset
        dl.num_workers = num_workers
        log.propagate = True
        log.setLevel(log_level)

        # Update lr attr
        lr = lr_finder.suggestion()
        if lr is not None:
            lightning_setattr(model, lr_attr_name, lr)

    return lr_finder
