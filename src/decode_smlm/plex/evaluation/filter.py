from typing import Sequence

from decode.emitter import emitter
from decode.emitter import process


class MatchedFilter:
    def __init__(
        self,
        filters: Sequence[process.EmitterProcess],
    ):
        """
        Filter matched emittersets, e.g. to compute more robust metrics. Makes sure,
        the model is not punished for `hard` emitters.
        This is (or should be) in accordance to the challenge matching algorithm.

        Args:
            filters: a sequence of filters to apply to the emittersets
        """
        self.filters = filters

    def forward(
        self,
        tp: emitter.EmitterSet,
        fp: emitter.EmitterSet,
        fn: emitter.EmitterSet,
        tp_ref: emitter.EmitterSet,
    ) -> tuple[
        emitter.EmitterSet, emitter.EmitterSet, emitter.EmitterSet, emitter.EmitterSet
    ]:
        """
        Filter the emitters

        Args:
            tp: true positives
            fp: false positives
            fn: false negatives
            tp_ref: ground truths that have been matched to the true positives

        Returns:
            (emitter.EmitterSet, emitter.EmitterSet, emitter.EmitterSet, emitter.EmitterSet)

                - **tp**: true positives
                - **fp**: false positives
                - **fn**: false negatives
                - **tp_ref**: ground truths that have been matched to the true positives
        """
        for f in self.filters:
            ix = f.arg_forward(tp_ref)
            tp = tp[ix]
            tp_ref = tp_ref[ix]

            fn = f.forward(fn)
            fp = f.forward(fp)  # ToDo: Double check, since this is post matching

        return tp, fp, fn, tp_ref
