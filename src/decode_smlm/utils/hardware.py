import os
import warnings

import torch


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def default_device() -> str:
    """CUDA, then Apple's GPU (MPS), then the CPU, as uipsf chooses (uipsf.device.default_device).

    ``DECODE_DEVICE=cpu`` (or ``cuda:1``, ``mps``) overrides.  DECODE knew CUDA and the CPU only.
    """
    requested = os.environ.get("DECODE_DEVICE")
    if requested:
        return requested
    if torch.cuda.is_available():
        return 'cuda:0'
    if _mps_available():
        return 'mps'
    return 'cpu'


def available_device(device: str) -> str:
    """`device` if this machine has it, else `default_device()`, with a warning.

    A parameter file from a CUDA workstation says ``cuda``; on a Mac its training runs on MPS.
    DECODE fell back to the CPU whenever CUDA was missing.
    """
    name, _ = _specific_device_by_str(str(device))
    if (name == 'cuda' and torch.cuda.is_available()) or (name == 'mps' and _mps_available()) or name == 'cpu':
        return str(device)
    fallback = default_device()
    warnings.warn(f"{device} is not available here; running on {fallback}.")
    return fallback


def _specific_device_by_str(device) -> tuple[str, str]:
    """Converts torch compatible device string to device name and device index"""
    if device != 'cpu' and device[:4] != 'cuda' and device[:3] != 'mps':
        raise ValueError(f"Unknown device {device!r}: cpu, cuda, cuda:<i> or mps.")

    if device == 'cpu':
        return 'cpu', None

    name = 'mps' if device[:3] == 'mps' else 'cuda'
    if ':' not in device:
        return name, None
    return name, int(device.split(':')[-1])


def get_device_capability() -> str:
    capability = torch.cuda.get_device_capability()
    return f'{capability[0]}.{capability[1]}'


def get_max_batch_size(callable, x_size: tuple, device: str | torch.device, size_low: int, size_high: int):
    if size_low > size_high:
        raise ValueError("Lower bound must be lower than upper bound.")

    bs = size_low
    bs_fail = size_high
    bs_pass = None

    while bs < size_high:

        try:
            x_try = torch.rand(bs, *x_size, device=device)
            callable(x_try)
            bs_pass = bs

        except RuntimeError as err:  # torch.OutOfMemoryError is one; CUDA's and MPS's messages differ
            if not isinstance(err, getattr(torch, 'OutOfMemoryError', ())) and 'out of memory' not in str(err):
                raise err

            bs_fail = bs

        if bs_pass is None:
            raise RuntimeError("Lowest possible batch size is outside of provided bounds.")

        bs_new = int((bs_fail + bs_pass) / 2)

        if bs_new == bs:
            break

        bs = bs_new

    del x_try
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    elif torch.device(device).type == 'mps':
        torch.mps.empty_cache()
    return bs
