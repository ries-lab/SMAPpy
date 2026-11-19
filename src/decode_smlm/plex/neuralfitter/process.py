from typing import Any, Optional, Protocol, Sequence, Union, Literal

import torch

from ..emitter import emitter
from ..generic import lazy


class _Forwardable(Protocol):
    def forward(self, x: Any) -> Any:
        ...


_on_coupled = Literal["ignore", "raise"]
_mode = Literal["train", "val"]


class Processing:
    def __init__(
        self,
        coupled: Optional[_Forwardable] = None,
        m_input: Optional[_Forwardable] = None,
        tar: Optional[_Forwardable] = None,
        tar_em: Optional[_Forwardable] = None,
        post_model: Optional[_Forwardable] = None,
        post: Optional[_Forwardable] = None,
        mode: _mode = "train",
        on_coupled: _on_coupled = "raise",
    ):
        """
        Processing container to organize flow for neural model training and inference.

        Args:
            m_input: input processing, forward has frame, emitter and auxiliary
             arguments; must return model compatible input
            tar: compute target
            tar_em: compute target emitters without actually computing the target,
             useful for validation
            post_model: intermediate post-processing
            post: post-processing, forward has model output as argument
            mode: `train` or `val`
            on_coupled: what to do for `tar, tar_em` if `coupled` is not None.
        """
        super().__init__()
        self.mode = mode
        self._coupled = coupled
        self._m_input = m_input
        self._tar = tar
        self._tar_em = tar_em
        self._post_model = post_model
        self._post = post
        self._on_coupled = on_coupled

    def pre_train(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        em: emitter.EmitterSet,
        bg: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        aux: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, Any]:
        """Preprocessing for training step. Returns input and target."""
        x, y, _ = self.pre_val(frame, em, bg, aux)
        return x, y

    def pre_val(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        em: emitter.EmitterSet,
        bg: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        aux: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, Any, emitter.EmitterSet]:
        """Preprocessing for validation step. Returns input, target and emitters."""
        if self._coupled is not None:
            em, frame, bg, aux = self._coupled.forward(
                em=em, frame=frame, bg=bg, aux=aux
            )
        x = self._m_input.forward(frame=frame, em=em, bg=bg, aux=aux)
        # ToDo: aux and bg naming
        y = self._tar.forward(em=em, aux=bg)
        return x, y, em

    def pre_inference(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        aux: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Preprocessing for inference step. Returns model input."""
        return self._m_input.forward(frame=frame, em=None, bg=None, aux=aux)

    def input(
        self,
        frame: torch.Tensor,
        em: emitter.EmitterSet,
        bg: torch.Tensor,
        aux: dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError

    def tar(self, em: emitter.EmitterSet, aux: dict[str, Any]) -> torch.Tensor:
        if self._coupled is not None and self._on_coupled == "raise":
            raise RuntimeError(
                "`tar` can not be independently computed if `coupled` is not None. "
                "Use entire pipeline via `pre_val` instead, "
                "which returns target as argument [1]."
            )

        return self._tar.forward(em, aux)

    def tar_em(self, em: emitter.EmitterSet) -> emitter.EmitterSet:
        if self._coupled is not None and self._on_coupled == "raise":
            raise RuntimeError(
                "`tar_em` can not be independently computed if `coupled` is not None. "
                "Use entire pipeline via `pre_val` instead, "
                "which returns target emitters as argument [2]."
            )

        return self._tar_em.forward(em)

    def post(self, x: torch.Tensor) -> emitter.EmitterSet:
        """
        Process model output through whole post-processing pipeline to get EmitterSet.

        Args:
            x: model output

        """
        return self._post.forward(x)

    @lazy.no_op_on("_post_model")
    def post_model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process model output only to post-processing necessary to compute the loss

        Args:
            x: model output

        """
        return self._post_model.forward(x)
