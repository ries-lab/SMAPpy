import math
import pytorch_lightning as pl


class ModelCheckpointExponential(pl.callbacks.ModelCheckpoint):
    """Model checkpointing with exponential increase of the saving interval."""

    def __init__(self, inc_factor: float, **kwargs):
        """
        Args:
            inc_factor: factor by which the saving interval is increased
            **kwargs: see pytorch_lightning.callbacks.ModelCheckpoint
        """
        # deciding parameter: not None, not 0, set
        controlling_params = [
            "every_n_train_steps",
            "train_time_interval",
            "every_n_epochs",
        ]
        controlling_params = [p for p in controlling_params if kwargs.get(p)]
        if len(controlling_params) != 1:
            raise ValueError(
                "Exactly one of every_n_train_steps, train_time_interval, every_n_epochs must be specified."
            )
        controlling_param = controlling_params[0]

        super().__init__(**kwargs)
        self.inc_factor = inc_factor
        self.controlling_param = f"_{controlling_param}"

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        # increase the saving interval every time the checkpoint is saved
        super()._save_checkpoint(trainer, filepath)
        new_val = getattr(self, self.controlling_param) * self.inc_factor
        if "_n_" in self.controlling_param:
            # float -> int, ceil required for inc_factor < 2
            new_val = math.ceil(new_val)
        setattr(self, self.controlling_param, new_val)
