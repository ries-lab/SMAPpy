import pickle
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch.utils.data
from torch.utils import data

from decode_smlm.plex.generic.logging import get_logger

from . import dataset, experiment

logger = get_logger(__name__)


class DataModel(pl.LightningDataModule):
    def __init__(
        self,
        experiment_train: experiment.Experiment,
        experiment_val: experiment.Experiment,
        path_val: Path,
        num_workers: int,
        batch_size: int,
        pin_memory: bool = True,
    ):
        super().__init__()

        self._exp_train = experiment_train
        self._exp_val = experiment_val
        self._path_val = path_val
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self.batch_size = batch_size
        self._ds_val = None

    def prepare_data(self) -> None:
        # sample val set here, because we want to sample this only once
        if self._path_val.is_file():
            logger.info("Validation sampler already exists.", path=self._path_val)
            return

        self._exp_val.sample()
        with self._path_val.open("wb") as f:
            pickle.dump(self._exp_val, f)

        logger.info("Validation sampler sampled and pickled.")

    def train_dataloader(self) -> data.DataLoader:
        # call this every epoch to get new data
        # i.e. `reload_dataloaders_every_n_epochs` hook in trainer
        self._exp_train.sample()
        logger.info("Training sampled")

        ds = dataset.SequenceDataset(self._exp_train.train_samples())
        logger.info("Training dataset created", len_ds=len(ds))

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
        logger.info(
            "Training dataloader created",
            len_dl=len(dl),
            batch_size=self.batch_size,
            num_workers=self._num_workers,
        )
        return dl

    def val_dataloader(self) -> data.DataLoader:
        with self._path_val.open("rb") as f:
            self._exp_val = pickle.load(f)
        logger.info("Validation sampler loaded", path=self._path_val)

        self._ds_val = dataset.SequenceDataset(self._exp_val.val_samples())
        logger.info("Validation dataset created", len_ds=len(self._ds_val))

        dl = data.DataLoader(
            self._ds_val,
            batch_size=self.batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            pin_memory=False,  # because here we also have EmitterSet as return
        )
        logger.info(
            "Validation dataloader created",
            len_dl=len(dl),
            batch_size=self.batch_size,
            num_workers=self._num_workers,
        )
        return dl

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        # leave EmitterSet on CPU; it is not used for training, on batch_len = 3
        if len(batch) == 3:
            return super().transfer_batch_to_device(
                batch[:-1], device, dataloader_idx
            ) + [
                batch[-1],
            ]
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
