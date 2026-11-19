from .model_in import ModelChannelMapInput


class ModelGMMMapInput:
    def __init__(
        self,
        mapper: ModelChannelMapInput,
        separate_win: bool,
        separate_ch: bool,
        aux_stage_shared: bool,
        aux_stage_union: bool,
    ):
        """
        Helper to map input semantic channels to the implementation of the model.

        Args:
            mapper: above mapper
            separate_win: if the frames are passed to the shared net separately, or concatenated
            separate_ch: if different channels have a different shared net
            aux_stage_shared: add auxiliaries to shared stage
            aux_stage_union: add auxiliaries to union stage
        """
        self._mapper = mapper
        self._separate_win = separate_win
        self._separate_ch = separate_ch
        self._aux_stage_shared = aux_stage_shared
        self._aux_stage_end = aux_stage_union

        self._stage_shared = self._compute_stage_shared()

    @property
    def stage_shared(
        self,
    ) -> list[list[list[int, ...], ...], ...]:
        """
        Indices for the 'shared' stage, i.e. the first stage where
        a model (one per channel if `separate_ch` is True, else one model)
        is applied to every frame individually.
        """
        return self._stage_shared

    @property
    def stage_union(self) -> list[int, ...]:
        return self._mapper.ix_aux if self._aux_stage_end else []

    @property
    def n_models_shared(self) -> int:
        return len(self._stage_shared)

    @property
    def n_ch_shared(self) -> int:
        return len(self._stage_shared[0][0])

    def compute_ch_union(self, inter_features: int) -> int:
        """
        Compute the number of channels in the union stage.

        Args:
            inter_features: number of features in the intermediate stage (output from shared stage)
        """
        return sum(
            [len(mod_in) for mod_in in self._stage_shared]
        ) * inter_features + len(self.stage_union)

    def _compute_stage_shared(
        self,
    ) -> list[list[list[int, ...], ...], ...]:
        if self._separate_win:
            out = [[[ii] for ii in ch] for ch in self._mapper.ix_ch]
        else:
            out = [[list(ch)] for ch in self._mapper.ix_ch]

        if not self._separate_ch:  # all channels input in same model
            out = [[ch_in for ch in out for ch_in in ch]]

        if self._aux_stage_shared:
            out = [[oo + self._mapper.ix_aux for oo in o] for o in out]

        return out
