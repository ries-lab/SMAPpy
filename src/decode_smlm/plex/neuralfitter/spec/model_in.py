from ...generic.logging import get_logger

logger = get_logger(__name__)


class ModelChannelMapInput:
    def __init__(self, win: int, n_ch: int, n_aux: int):
        """
        Helper to map the channels of an input tensor to semantic information.

        Args:
            win: window
            n_ch: number of channels
            n_aux: number of auxiliary channels
        """
        self._win = win
        self._n_ch = n_ch
        self._n_aux = n_aux

    @property
    def n(self) -> int:
        """Unique number of input channels"""
        return self._win * self._n_ch + self._n_aux

    @property
    def ix_frames(self) -> list[int, ...]:
        """Indices of frame channels"""
        return list(range(self._win * self._n_ch))

    @property
    def ix_ch(self) -> list[list[int, ...]]:
        """
        Returns the indices of the frame in the respective channel.
        E.g. win 3, n_ch 2 returns
            [[0, 1, 2], [3, 4, 5]]
        """
        return [
            self.ix_frames[i : i + self._win]
            for i in range(0, len(self.ix_frames), self._win)
        ]

    @property
    def ix_aux(self) -> list[int, ...]:
        """Indices of auxiliary channels"""
        return list(range(self._win * self._n_ch, self.n))
