from typing import Literal

from ... import emitter


_on_missing = Literal["raise", "ignore"]


class ProcessingFrameSpecific:
    def __init__(
        self,
        proc: dict[int, emitter.process.EmitterProcess],
        on_missing: _on_missing = "raise",
    ):
        """
        Applies different processing to different frames.

        Args:
            proc: dict of frame_ix to processing
            on_missing: how to handle non-matching frame_ix
        """
        self._proc = proc
        self._on_missing = on_missing

        if self._on_missing not in ["raise", "ignore"]:
            raise ValueError(f"on_missing={self._on_missing} not supported.")

    def forward(self, em: emitter.EmitterSet) -> emitter.EmitterSet:
        if self._on_missing == "raise":
            if set(em.frame_ix) != set(self._proc.keys()):
                raise KeyError(
                    f"EmitterSet.frame_ix={em.frame_ix} does not match "
                    f"ProcessingFrameSpecific.proc={self._proc.keys()}"
                )

        return emitter.EmitterSet.cat(
            [v.forward(em.iframe[k]) for k, v in self._proc.items()]
        )
