from typing import TypeVar, Callable, Optional, Sequence, Union

import torch

from decode.generic import protocols

TDelayed = TypeVar("TDelayed")


class _DelayedSlicer:
    def __init__(
        self,
        fn: Callable[..., TDelayed],
        args: Optional[Sequence] = None,
        kwargs: Optional[dict] = None,
        kwargs_static: Optional[dict] = None,
        indexer: protocols.TypedSequence[int] = None,
    ):
        """
        Returns a sliceable handle and executes a function on __getitem__ where input
        arguments are then sliced and passed on to the function. Useful for delaying
        function executions that are optionally batched.

        Args:
            fn:
            attr: registered attributes
            args: list of sliced positional arguments
            kwargs: list of sliced keyword arguments
            kwargs_static: list of non-sliced keyword arguments
        """
        self._fn = fn
        self._args = args if args is not None else []
        self._kwargs = kwargs if kwargs is not None else dict()
        self._kwargs_stat = kwargs_static if kwargs_static is not None else dict()
        self._indexer = indexer

        if indexer is not None:
            raise NotImplementedError("Currently issue with indexer.")

    def __len__(self) -> int:
        if self._indexer is not None:
            return len(self._indexer)
        elif self._args is None and self._kwargs is None:
            raise ValueError("Unable to return length because no arguments were set.")
        elif len(self._args) >= 1:
            return len(self._args[0])
        else:
            return len(next(iter(self._kwargs.values())))

    def __getitem__(self, item) -> TDelayed:
        if self._indexer is not None:
            item = self._indexer[item]
        args = [o[item] for o in self._args]
        kwargs = {k: v[item] for k, v in self._kwargs.items()}
        kwargs.update(self._kwargs_stat)

        return self._fn(*args, **kwargs)


class _DelayedTensor(_DelayedSlicer):
    def __init__(
        self,
        fn: Callable[..., torch.Tensor],
        size: Optional[torch.Size] = None,
        *,
        args: Optional[Union[list, tuple]] = None,
        kwargs: Optional[dict] = None,
        kwargs_static: Optional[dict] = None,
    ):
        """
        Delay a callable on a tensor.

        Args:
            fn: delayed callable
            size: output size of function given all arguments
            args: arbitrary positional arguments to pass on. Must be passed as explicit
             list, not implicitly.
            kwargs: arbitrary keyword arguments to pass on. Must be passed as explicit
             dict, not implicitly.
        """
        super().__init__(
            fn,
            args=args,
            kwargs=kwargs,
            kwargs_static=kwargs_static,
        )

        self._size = size

    def __len__(self) -> int:
        if self._size is None:
            raise ValueError("Unable to return length because size was not set.")

        return self._size[0]

    def to(self, device: str | torch.device) -> "_DelayedTensor":
        device = torch.device(device) if isinstance(device, str) else device
        if device != torch.device("cpu"):
            raise ValueError("Currently only supports CPU device.")
        return self

    def cpu(self) -> "_DelayedTensor":
        return self.to(torch.device("cpu"))

    def cuda(self) -> "_DelayedTensor":
        return self.to(torch.device("cuda:0"))

    def size(self, dim=None) -> torch.Size:
        if self._size is None:
            raise ValueError("Unable to return size because it was not set.")

        if dim is None:
            return self._size
        else:
            return self._size[dim]

    def auto_size(self, n: Optional[int] = None) -> "_DelayedTensor":
        """
        Automatically determine, by running the callable on the first element,
        inspecting output and concatenating this to batch dim.

        Args:
            n: manually specify first (batch) dim
        """
        if n is None:
            if len(self._args) >= 1:
                n = len(self._args[0])
            elif len(self._kwargs) >= 1:
                n = len(next(iter(self._kwargs.values())))
            else:
                raise ValueError(
                    "Cannot auto-determine size if neither arguments nor "
                    "keyword arguments were specified."
                )

        size_last_dims = self[0].size()
        self._size = torch.Size([n, *size_last_dims])

        return self


class _InterleavedSlicer:
    def __init__(self, x: Sequence[torch.Tensor]):
        """
        Helper to slice a sequence of tensors in a batched manner, i.e. slicing on the
        sequence will be forwarded to each tensor in the sequence.

        Args:
            x: sequence of tensors
        """
        self._x = x

    def __len__(self):
        if all(len(x) == len(self._x[0]) for x in self._x):
            return len(self._x[0])
        else:
            raise ValueError("Length is ill-defined if tensors are not of same length.")

    def __getitem__(self, item) -> tuple[torch.Tensor, ...]:
        return tuple(x[item] for x in self._x)
