from pathlib import Path

import torch


class CheckPoint:
    def __init__(self, path: str | Path):
        """
        Checkpointing intended to resume to an already started training.
        Warning:
            Checkpointing is not intended for long-term storage of models or other information.
            No version compatibility guarantees are given here at all.

        Args:
            path: filename / path where to dump the checkpoints

        """
        self.path = path

        self.model_state = None
        self.optimizer_state = None
        self.lr_sched_state = None
        self.step = None
        self.log = None

    @property
    def dict(self):
        return {
            'step': self.step,
            'model_state': self.model_state,
            'optimizer_state': self.optimizer_state,
            'lr_sched_state': self.lr_sched_state,
            'log': self.log
        }

    def update(self, model_state: dict, optimizer_state: dict, lr_sched_state: dict, step: int, log=None):
        self.model_state = model_state
        self.optimizer_state = optimizer_state
        self.lr_sched_state = lr_sched_state
        self.step = step
        self.log = log

    def save(self):
        torch.save(self.dict, self.path)

    @classmethod
    def load(cls, path: str | Path, path_out: str | Path | None = None,
             map_location: str | torch.device = 'cpu'):
        # map_location: a checkpoint written on CUDA does not open on a machine without it.
        # weights_only stays torch's default (True since 2.6): a v0.10.2 checkpoint, its log of
        # floats included, loads under it (tried on a 4090 run's, 2026-11-10).
        ckpt_dict = torch.load(path, map_location=map_location)

        if path_out is None:
            path_out = path
        ckpt = cls(path=path_out)
        ckpt.update(model_state=ckpt_dict['model_state'], optimizer_state=ckpt_dict['optimizer_state'],
                    lr_sched_state=ckpt_dict['lr_sched_state'], step=ckpt_dict['step'],
                    log=ckpt_dict['log'] if 'log' in ckpt_dict.keys() else None)

        return ckpt

    def dump(self, model_state: dict, optimizer_state: dict, lr_sched_state: dict, step: int, log=None):
        """Updates and saves to file."""
        self.update(model_state, optimizer_state, lr_sched_state, step, log)
        self.save()
