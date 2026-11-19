from typing import Any, Optional, Protocol, Sequence

import pytorch_lightning as pl
import torch

from decode.emitter import emitter, process as em_process
from decode.neuralfitter import logger, process
from decode.evaluation import evaluation
from decode.evaluation import predict_dist


class Model(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        proc_train: process.Processing,
        proc_val: process.Processing,
        batch_size: int,
        optimizer: Optional[type] = None,
        optimizer_specs: Optional[dict[str, Any]] = None,
        lr: Optional[float] = None,
        lr_sched: Optional[type] = None,
        lr_sched_specs: Optional[dict[str, Any]] = None,
        loss_train: Optional[torch.nn.Module] = None,
        loss_val: Optional[torch.nn.Module] = None,
        matcher: Optional[evaluation.match_emittersets.EmitterMatcher] = None,
        evaluator: Optional[evaluation.SMLMEvaluation] = None,
        eval_filter: Optional[em_process.EmitterProcess] = None,
        ix_tar: int = 1,
    ):
        """

        Args:
            model:
            proc:
            batch_size:
            optimizer: Optimizer type, not instance because we may need to alter it
            optimizer_specs: Optimizer specs
            lr: Learning rate
            lr_sched:
            lr_sched_specs:
            loss:
            matcher:
            evaluator:
            ix_tar: index of the target frame to use for logging
        """
        super().__init__()

        self._model = model
        self._opt = optimizer
        self._opt_specs = optimizer_specs
        self._lr = lr
        self._lr_sched = lr_sched
        self._lr_sched_specs = lr_sched_specs
        self.auto_lr = lr is None
        self._proc_train = proc_train
        self._proc_val = proc_val
        self._loss_train = loss_train
        self._loss_val = loss_val
        self.batch_size = batch_size
        self._evaluator = evaluator
        self._eval_filter = eval_filter
        self._matcher = matcher
        self._em_val_out = []
        self._em_val_tar = []
        self._ix_tar = ix_tar

    @property
    def logger(self) -> Optional[logger.TensorboardLogger]:
        return super().logger

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]
    ]:
        opt = self._opt(self.parameters(), lr=self._lr or 1, **self._opt_specs)
        lr_sched = (
            self._lr_sched(opt, **self._lr_sched_specs)
            if self._lr_sched is not None
            else None
        )

        lr_sched_cfg = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_sched,
            "interval": "epoch",
            "frequency": 1,
            # Metric to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "loss/train_epoch",
            "strict": True,
        }

        return [opt], [lr_sched_cfg] if lr_sched is not None else []

    def forward(self, x: torch.Tensor) -> Any:
        y = self._model.forward(x)
        if self.training and self._proc_train is not None:
            y = self._proc_train.post(y)
        elif not self.training and self._proc_val is not None:
            y = self._proc_val.post(y)

        return y

    def training_step(self, batch, batch_ix: int):
        x, y = batch

        y_raw = self._model.forward(x)
        y_post = self._proc_train.post_model(y_raw)
        loss, loggable = self._loss_train.forward(y_post, y)

        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
            prog_bar=True,
        )
        self.logger.log_group(loggable, prefix="loss/train_", step=self.global_step)

        if batch_ix == 0:  # graphic logging
            ix = 0
            self.logger.log_tensor(
                x[ix], name="input_train", step=self.global_step, unbind=0
            )
            self.logger.log_tensor(
                y_raw[ix], name="output_raw_train", step=self.global_step, unbind=0
            )
            self.logger.log_tensor(
                y_post[ix],
                name="output_post_model_train",
                step=self.current_epoch,
                unbind=0,
            )

        return loss

    def on_validation_epoch_start(self) -> None:
        self._em_val_out = []
        self._em_val_tar = []

    def validation_step(self, batch, batch_ix: int):
        x, y_ref, em_tar = batch

        y_raw = self._model.forward(x)
        y_post = self._proc_val.post_model(y_raw)

        em_out = self._proc_val.post(y_raw.clone())
        em_out.frame_ix += batch_ix * self.batch_size  # frame ix in dataset

        self._em_val_out.append(em_out)
        self._em_val_tar.append(em_tar)

        ix_tar = self._ix_tar

        if batch_ix == 2:  # graphic logging
            ix = min(8, x.size(0) - 1)  # ix in batch
            ix_ds = self.batch_size * batch_ix + ix  # ix in dataset

            em_tar_log = em_tar.iframe[ix_ds]
            em_out_log = em_out.iframe[ix_ds]

            self.logger.log_tensor(
                x[ix], name="input_val", step=self.current_epoch, unbind=0
            )
            for i, x_ch in enumerate(torch.unbind(x[ix], dim=0)):
                self.logger.log_emitter(
                    name=f"input_em_val/ch{i}",
                    em_tar=em_tar_log,
                    frame=x_ch,
                    step=self.current_epoch,
                )
            self.logger.log_emitter(
                name="output_em_val/p010",
                em_tar=em_tar_log,
                em=em_out_log[em_out_log.prob >= 0.1],
                frame=x[ix, ix_tar],
                step=self.current_epoch,
            )
            self.logger.log_emitter(
                name="output_em_val/p050",
                em_tar=em_tar_log,
                em=em_out_log[em_out_log.prob >= 0.5],
                frame=x[ix, ix_tar],
                step=self.current_epoch,
            )
            self.logger.log_emitter(
                name="output_em_val/p090",
                em_tar=em_tar_log,
                em=em_out_log[em_out_log.prob >= 0.9],
                frame=x[ix, ix_tar],
                step=self.current_epoch,
            )
            self.logger.log_tensor(
                y_raw[ix], name="output_raw_val", step=self.current_epoch, unbind=0
            )
            self.logger.log_tensor(
                y_post[ix],
                name="output_post_model_val",
                step=self.current_epoch,
                unbind=0,
            )
        return

    def on_train_epoch_end(self) -> None:
        sched = self.lr_schedulers()

        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(self.trainer.callback_metrics["loss/train"])
        else:
            sched.step()

    def on_validation_epoch_end(self) -> None:
        if self._evaluator is None:
            return

        em_out = emitter.EmitterSet.cat(self._em_val_out)
        em_tar = emitter.EmitterSet.cat(self._em_val_tar)

        # emitter based metrics
        em_out_filtered = em_out
        if self._eval_filter is not None:
            em_out_filtered = self._eval_filter.forward(em_out)
        tp, fp, fn, tp_ref = self._matcher.forward(em_out_filtered, em_tar)
        metrics = self._evaluator.forward(tp, fp, fn, tp_ref)

        self.logger.log_group(metrics, prefix="eval/", step=self.current_epoch)
        self.logger.log_group(
            {"n_out": len(em_out), "n_tar": len(em_tar)},
            prefix="eval/",
            step=self.current_epoch,
        )

        self.logger.log_hist(
            name="tar_em_dist_val/frame_ix",
            vals=em_tar.frame_ix,
            step=self.current_epoch,
        )

        if len(em_out) >= 1:
            self.logger.log_hist(
                name="output_em_dist_val/prob",
                vals=em_out.prob,
                step=self.current_epoch,
            )
            self.logger.log_hist(
                name="output_em_dist_val/frame_ix",
                vals=em_out.frame_ix,
                step=self.current_epoch,
            )
            self.logger.log_hist(
                name="output_em_dist_val/x",
                vals=em_out.xyz[:, 0],
                step=self.current_epoch,
            )
            self.logger.log_hist(
                name="output_em_dist_val/y",
                vals=em_out.xyz[:, 1],
                step=self.current_epoch,
            )
            self.logger.log_hist(
                name="output_em_dist_val/z",
                vals=em_out.xyz[:, 2],
                step=self.current_epoch,
            )
            self.logger.log_hist(
                name="output_em_dist_val/x_offset",
                vals=predict_dist.px_pointer_dist(em_out.xyz[:, 0], -0.5, 1.0),
                step=self.current_epoch,
            )
            self.logger.log_hist(
                name="output_em_dist_val/y_offset",
                vals=predict_dist.px_pointer_dist(em_out.xyz[:, 1], -0.5, 1.0),
                step=self.current_epoch,
            )
            # handle multi-dim phot
            if (tp.phot.squeeze().dim() == 1) or (tp_ref.phot.squeeze().dim() == 1):
                self.logger.log_deviation(
                    name="output_em_dev_val/phot",
                    x=tp.phot.squeeze(-1),
                    x_ref=tp_ref.phot.squeeze(-1),
                    step=self.current_epoch,
                )
            else:
                for i in range(tp.phot.shape[1]):
                    self.logger.log_deviation(
                        name=f"output_em_dev_val/phot_{i}",
                        x=tp.phot[..., i],
                        x_ref=tp_ref.phot[..., i],
                        step=self.current_epoch,
                    )
                    if i == 0:
                        continue
                    self.logger.log_deviation(
                        name=f"em_ch_corr/tp_phot_{i}_0",
                        x=tp.phot[..., i],
                        x_ref=tp.phot[..., 0],
                        step=self.current_epoch,
                        xlabel="channel 0",
                        ylabel=f"channel {i}",
                    )
                    self.logger.log_deviation(
                        name=f"em_ch_corr/tp_ref_phot_{i}_0",
                        x=tp_ref.phot[..., i],
                        x_ref=tp_ref.phot[..., 0],
                        step=self.current_epoch,
                        xlabel="channel 0",
                        ylabel=f"channel {i}",
                    )
            self.logger.log_deviation(
                name="output_em_dev_val/z",
                x=tp.xyz_nm[:, 2],
                x_ref=tp_ref.xyz_nm[:, 2],
                step=self.current_epoch,
            )

        # ToDo: emitter based distributions
        # ToDo: graphical samples
