from typing import Any, Optional, Sequence, Union

import torch
from deprecated import deprecated

from decode.emitter import emitter
from decode.generic import delay, protocols, indexing
from decode.neuralfitter import process


class IxShifter:
    _pad_modes = (None, "same")

    def __init__(self, mode: str, window: int, n: Optional[int] = None):
        """
        Shift index to allow for windowing without repeating samples
        Args:
            mode: either `None` (which will shift) or `same` (no-op)
            window: window size
            n: length of indexable object (to compute lenght after shifting)
        Examples:
            >>> IxShifter(None, 3)[0]
            1
            >>> IxShifter("same", 100000)[0]
            0
        """
        self._mode = mode
        self._window = window
        self._n_raw = n

        if mode not in self._pad_modes:
            raise NotImplementedError

        if window % 2 != 1:
            raise ValueError("Window must be odd.")

    def __len__(self) -> int:
        if self._n_raw is None:
            raise ValueError("Cannot compute len without specifying n.")
        if self._mode is None:
            n = self._n_raw - self._window + 1
        else:
            n = self._n_raw
        return n

    def __getitem__(self, item: int) -> int:
        if self._mode is None:
            # no padding means we need to shift indices, i.e. loose a few samples
            if item < 0:
                raise ValueError("Negative indexing not supported.")
            item = item + (self._window - 1) // 2

        return item


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, seq: Sequence[Any]):
        self._seq = seq

    def __len__(self) -> int:
        return len(self._seq)

    def __getitem__(self, item: int) -> Any:
        return self._seq[item]


class DatasetGenericInputTar(torch.utils.data.Dataset):
    def __init__(
        self,
        x: protocols.TypedSequence,
        y: protocols.TypedSequence,
        em: Optional[emitter.EmitterSet] = None,
    ):
        """
        Generic dataset consisting of input and target and optional EmitterSet.

        Args:
            x: input
            y: target
            em: optional EmitterSet that is returned as last return argument
        """
        self._x = x
        self._y = y
        self._em = em

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, item: int):
        if self._em is None:
            return self._x[item], self._y[item]
        else:
            return self._x[item], self._y[item], self._em.iframe[item]


@deprecated(reason="No longer used.", version="1.0")
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frames: Union[torch.Tensor, Sequence[torch.Tensor]],
        aux: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        window: int = 1,
        proc: Optional[process.Processing] = None,
    ):
        """
        Dataset for inference.
        This dataset will apply the pre_inference method of the processing, if given.


        Args:
            frames:
            aux:
            window:
            proc:
        """
        f = (
            delay._InterleavedSlicer(frames)
            if not isinstance(frames, torch.Tensor)
            else frames
        )
        f = indexing.IxWindow(window, None).attach(f, auto_recurse=False)
        self._frames = f
        self._aux = aux
        self._window = window
        self._proc = proc

        if self._proc is None and self._aux is not None:
            raise ValueError("Aux data is only supported if proc is given.")

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, item: int) -> torch.Tensor:
        if self._proc is None:
            return self._frames[item]
        return self._proc.pre_inference(self._frames[item], aux=self._aux)
