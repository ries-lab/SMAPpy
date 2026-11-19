import torch

from decode.emitter import emitter


def chall_to_decode(em: emitter.EmitterSet, flip_z: bool = True) -> emitter.EmitterSet:
    """
    Convert challenge standard to decode standard (without biases)

    Args:
        em: emitter set
        flip_z: flip z-axis

    Returns:
        emitter.EmitterSet: emitter set
    """
    em = em.clone()
    em.xyz = em.xyz[:, [1, 0, 2]]
    em.xyz_px = em.xyz_px - torch.tensor([[1.0, 0.0, 0.0]])
    if flip_z:
        em.xyz_px[:, 2] *= -1
    em.frame_ix -= 1
    return em


def decode_to_chall(em: emitter.EmitterSet, flip_z: bool = True) -> emitter.EmitterSet:
    """
    Convert decode standard to challenge standard (without biases)

    Args:
        em: emitter set
        flip_z: flip z-axis

    Returns:
        emitter.EmitterSet: emitter set
    """
    em = em.clone()
    em.xyz_px = em.xyz_px + torch.tensor([[1.0, 0.0, 0.0]])
    em.xyz = em.xyz[:, [1, 0, 2]]
    if flip_z:
        em.xyz_px[:, 2] *= -1
    em.frame_ix += 1
    return em
