import torch.utils.data
from deprecated import deprecated

from ... import emitter


def collate_emitter(em_batch: list, *, collate_fn_map):
    # back cat emitters, this is a bit cumbersome because ds.__getitem__ first does
    # `.iframe` and then here we need to cat the emitters back
    return emitter.EmitterSet.cat([e for e in em_batch])


torch.utils.data._utils.collate.default_collate_fn_map.update(
    {emitter.EmitterSet: collate_emitter}
)


@deprecated(
    reason="Code duplication and not necessary anymore.",
    version="0.11.0",
    action="error",
)
def smlm_collate(batch):
    pass
