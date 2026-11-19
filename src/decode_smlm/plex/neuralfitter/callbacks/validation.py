from pytorch_lightning import Callback


class LossValidation(Callback):
    def __init__(self, loss_max: float = 1e5, at: int = 15):
        """

        Args:
            loss_max: max loss
            at: at which step
        """
        self.loss_max = loss_max
        self.at = at

    def on_train_epoch_end(self, trainer, pl_module):
        # Access validation metrics from the trainer
        loss = trainer.callback_metrics.get("loss/train_epoch", None)

        if trainer.current_epoch >= self.at and loss > self.loss_max:
            raise ValueError("Abort training because loss too high.")
