import time
import warnings
from functools import partial
from typing import Any, Callable, Optional, Union
from typing import Literal

import torch
from deprecated import deprecated
from tqdm import tqdm

from . import debug
from ...validation import base as val_base
from ..data import dataset
from ..sample import samplers
from ...emitter import emitter
from ...utils import hardware
from ...generic import logging

logger = logging.get_logger(__name__)


class Infer:
    def __init__(
        self,
        model,
        window: int,
        device: Union[str, torch.device, list],
        pre: Optional[Callable] = None,
        post: Optional[Callable] = None,
        batch_size: Union[int, Literal["auto"]] = "auto",
        num_workers: int = 0,
        pin_memory: bool = False,
        forward_cat: Union[str, Callable] = "emitter",
        logger: debug.InferenceLogger | Literal["debug"] | None = None,
        validator: val_base.PipelineValidator | None = None,
    ):
        """
        Convenience class for inference.

        Args:
            model: pytorch model
            post: post-processing callable
            device: device where to run inference
            batch_size: batch-size or 'auto' if the batch size should be determined automatically (only use in combination with cuda)
            num_workers: number of workers
            pin_memory: pin memory in dataloader
            forward_cat: method which concatenates the output batches. Can be string or Callable.
                Use 'em' when the post-processor outputs an EmitterSet, or 'frames' when you don't use post-processing or if
                the post-processor outputs frames.
            logger: logger for debugging; if 'debug' a default logger is used with
                interval 1 and max_steps 100
        """

        self.model = model
        self.window = window
        self.pre = pre
        self.post = post
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.forward_cat = None
        self._forward_cat_mode = forward_cat

        if logger is None:
            logger = debug.NoOpLogger()
        elif logger == "debug":
            logger = debug.InferenceLogger(interval=1, max_steps=5)
        self.logger = logger
        self.validator = (
            validator if validator is not None else val_base.PipelineValidator.no_op()
        )

        if str(self.device) == "cpu" and self.batch_size == "auto":
            warnings.warn(
                "Automatically determining the batch size does not make sense on cpu. "
                "Falling back to reasonable value."
            )
            self.batch_size = 16

    def forward(
        self,
        frames: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
        aux: Optional[Any] = None,
    ) -> emitter.EmitterSet | torch.Tensor:
        """
        Forward frames through model, pre- and post-processing and output EmitterSet

        Args:
            frames:
            aux:
        """
        frames = (
            [f.cpu() for f in frames]
            if isinstance(frames, (list, tuple))
            else frames.cpu()
        )
        aux = aux.to("cpu") if aux is not None else None

        frames = samplers.frame_samples(frames, window=self.window)
        samples = (
            samplers.inference_samples(frames, aux, self.pre)
            if self.pre is not None
            else frames
        )

        model = self.model.to(self.device[0])
        model = model.eval()

        
            
        def _normalized_device_list(dev):
            if isinstance(dev, (list, tuple)):
                return list(dev)
            if isinstance(dev, str) and "," in dev:
               return [s.strip() for s in dev.split(",")]
            return dev
        
        dev_spec = _normalized_device_list(self.device)
        is_multi = isinstance(dev_spec, (list, tuple)) and len(dev_spec) > 1
        
        if is_multi:
            # convert to integer device ids for DataParallel
            device_ids = []
            for d in dev_spec:
                if isinstance(d, torch.device):
                    if d.type != "cuda":
                        raise ValueError("DataParallel only supports cuda devices")
                    device_ids.append(d.index if d.index is not None else 0)
                else:
                    s = str(d)
                    if s.startswith("cuda"):
                        # "cuda:0"
                        try:
                            device_ids.append(int(s.split(":")[1]))
                        except Exception:
                            device_ids.append(0)
                    elif s.isdigit():
                        device_ids.append(int(s))
                    else:
                        raise ValueError(f"Unsupported device spec: {d}")

            primary = torch.device(f"cuda:{device_ids[0]}")
            _model = self.model.to(primary)
            if len(device_ids) > 1:
                model = torch.nn.DataParallel(_model, device_ids=device_ids)
            else:
                model = _model
        else:
            # single device
            model_device = (
                torch.device(self.device)
                if not isinstance(self.device, (list, tuple))
                else torch.device(self.device[0])
            )
            model = self.model.to(model_device)

        model = model.eval()
        input_device = primary if is_multi else model_device

        # if self.batch_size == "auto":
        #     # include safety factor of 20%
        #     bs = int(0.8 * self.get_max_batch_size(model, samples[0].size(), 1, 512))
        #     logger.info("Inferred batch size as", batch_size=bs)
        # else:
        #     bs = self.batch_size
        
        # batch size auto: compute per-GPU max and multiply by num gpus when using DataParallel
        if self.batch_size == "auto":
            # pick underlying module for get_max_batch_size if wrapped in DataParallel
            model_for_probe = model.module if isinstance(model, torch.nn.DataParallel) else model
            per_gpu_bs = int(0.8 * self.get_max_batch_size(model_for_probe, samples[0].size(), 1, 512))
            num_gpus = len(model.device_ids) if isinstance(model, torch.nn.DataParallel) else 1
            bs = max(1, per_gpu_bs * num_gpus)
            logger.info("Inferred batch size as", batch_size=bs, per_gpu=per_gpu_bs, num_gpus=num_gpus)
        else:
            bs = self.batch_size

        # generate concatenate function here because we need batch size for this
        self.forward_cat = self._setup_forward_cat(self._forward_cat_mode, bs)

        ds = dataset.SequenceDataset(samples)
        dl = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=bs,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        out = [None] * len(dl)

        with torch.no_grad():
            for ix_batch, sample in enumerate(tqdm(dl)):
                self.logger.log_model_in(sample, ix_batch)
                self.validator.validate_model_in(sample, ix_batch)
                x_in = sample.to(input_device)

                y_out = model(x_in)
                self.logger.log_model_out(y_out, ix_batch)
                self.validator.validate_model_out(y_out, ix_batch)

                if self.post is not None:
                    out[ix_batch] = self.post(y_out)
                    self.logger.log_post(out[ix_batch], ix_batch)
                    self.validator.validate_post(out[ix_batch], ix_batch)
                else:
                    out[ix_batch] = (
                        y_out.detach().cpu()
                        if isinstance(y_out, torch.Tensor)
                        else y_out
                    )

        out = self.forward_cat(out)
        return out

    @staticmethod
    def get_max_batch_size(
        model: torch.nn.Module,
        frame_size: Union[tuple, torch.Size],
        limit_low: int,
        limit_high: int,
    ):
        """
        Get maximum batch size for inference.

        Args:
            model: model on correct device
            frame_size: size of frames (without batch dimension)
            limit_low: lower batch size limit
            limit_high: upper batch size limit
        """

        def model_forward_no_grad(x: torch.Tensor):
            """
            Helper function because we need to account for torch.no_grad()
            """
            with torch.no_grad():
                o = model.forward(x)

            return o

        return hardware.get_max_batch_size(
            model_forward_no_grad,
            frame_size,
            next(model.parameters()).device,
            limit_low,
            limit_high,
        )

    @staticmethod
    def _setup_forward_cat(forward_cat, batch_size: int):
        if forward_cat is None:
            return lambda x: x

        elif isinstance(forward_cat, str):
            if forward_cat == "emitter":
                return partial(emitter.EmitterSet.cat, step_frame_ix=batch_size)

            elif forward_cat == "frames":
                return partial(torch.cat, dim=0)

        elif callable(forward_cat):
            return forward_cat

        else:
            raise TypeError(f"Specified forward cat method was wrong.")
        raise ValueError(f"Unsupported forward_cat value.")


@deprecated(reason="Change this implementation on use.", version="1.0")
class LiveInfer(Infer):
    def __init__(
        self,
        model,
        ch_in: int,
        *,
        stream,
        time_wait=5,
        safety_buffer: int = 20,
        frame_proc=None,
        post_proc=None,
        device: Union[str, torch.device] = "cuda:0"
        if torch.cuda.is_available()
        else "cpu",
        batch_size: Union[int, str] = "auto",
        num_workers: int = 0,
        pin_memory: bool = False,
        forward_cat: Union[str, Callable] = "emitter",
    ):
        """
        Inference from memmory mapped tensor, where the mapped file is possibly live being written to.

        Args:
            model: pytorch model
            ch_in: number of input channels
            stream: output stream. Will typically get emitters (along with starting and stopping index)
            time_wait: wait if length of mapped tensor has not changed
            safety_buffer: buffer distance to end of tensor to avoid conflicts when the file is actively being
            written to
            frame_proc: frame pre-processing pipeline
            post_proc: post-processing pipeline
            device: device where to run inference
            batch_size: batch-size or 'auto' if the batch size should be determined automatically (only use in combination with cuda)
            num_workers: number of workers
            pin_memory: pin memory in dataloader
            forward_cat: method which concatenates the output batches. Can be string or Callable.
            Use 'em' when the post-processor outputs an EmitterSet, or 'frames' when you don't use post-processing or if
            the post-processor outputs frames.
        """

        super().__init__(
            model=model,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            forward_cat=forward_cat,
        )

        self._stream = stream
        self._time_wait = time_wait
        self._buffer_length = safety_buffer

    def forward(self, frames: Union[torch.Tensor], aux: Any = None):
        # ToDo: Deprecate this in favor of a new inference method for streaming data
        # then impelment an input stream (incoming live data)
        # and an output stream
        n_fitted = 0
        n_waited = 0
        while n_waited <= 2:
            n = len(frames)

            if n_fitted == n - self._buffer_length:
                n_waited += 1
                time.sleep(self._time_wait)  # wait
                continue

            n_2fit = n - self._buffer_length
            out = super().forward(frames[n_fitted:n_2fit], aux=aux)
            self._stream(out, n_fitted, n_2fit)

            n_fitted = n_2fit
            n_waited = 0

        # fit remaining frames
        out = super().forward(frames[n_fitted:n], aux=aux)
        self._stream(out, n_fitted, n)
