from typing import Sequence

import torch

from decode.emitter import EmitterSet
from decode.evaluation import match_emittersets as matcher
from decode.neuralfitter import peakfinder


def check_aux(frames: Sequence[torch.Tensor], aux: Sequence[torch.Tensor]):
    """
    Check if auxiliary input matches the shift in frames.
    For this we run a simple peak finder on the frames, pick up the auxiliary
    values at the peak positions and compare the transformed coordinates.
    They should end up roughly be within 1 px.

    Args:
        frames:
        aux: auxiliary map which is used for model info

    """
    if len(frames) != 2:
        raise NotImplementedError

    pf = peakfinder.ModelPeakfinder(format="list")
    match = matcher.GreedyHungarianMatching(match_dims=2, dist_lat=10.0)

    em = [None] * 2

    for i, f in enumerate(frames):
        a = aux[i * 2 : (i * 2 + 2)]
        frame_ix, xyz, phot = pf.forward(f.unsqueeze(0))

        # look up auxiliary values and add
        x_off = a[0, xyz[:, 0], xyz[:, 1]]
        y_off = a[1, xyz[:, 0], xyz[:, 1]]
        xyz = xyz.float()
        xyz[:, 0] -= x_off
        xyz[:, 1] -= y_off

        em[i] = EmitterSet(
            xyz=xyz, phot=phot, frame_ix=frame_ix, px_size=(1.0, 1.0), xy_unit="nm"
        )

    # check if the emitters are roughly at the same position
    tp, *_, tp_match = match.forward(em[0], em[1])
    em_matched = EmitterSet(
        xyz=torch.stack([tp_match.xyz, tp.xyz], dim=1),
        phot=torch.stack([tp_match.phot, tp.phot], dim=1),
        frame_ix=tp.frame_ix,
        xy_unit="px",
    )
    return em_matched, em
