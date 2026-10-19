"""Synthetic data from the forward model itself, with known parameters to recover.

The same `ForwardModel` simulates and fits, so a matched simulation tests
the solver and the statistics alone; the mismatch effects of review §B
(background ramps, piezo scale, polychromatic light, laser fluctuations,
speckle) come with M3 as variations of the truth, not of this code.

Noise: Poisson photon counts, plus Gaussian read noise of variance
``read_noise_var`` (photons²) when given, which the batch then carries as
its shifted-Poisson offset s.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from .data.dataset import Batch
from .device import to_cpu64
from .params import Params


def template(n_emitters: int, z_nm, roi_shape: tuple[int, int], dtype=torch.float32, device=None,
             channels: int = 1) -> Batch:
    """An empty batch with the right shape: what `parameters` needs to build the specs.  With several
    ``channels`` the caller sets ``roi_xy``."""
    z = torch.as_tensor(z_nm, dtype=dtype, device=device)
    K = z.shape[-1]
    data = torch.zeros(n_emitters, channels, K, *roi_shape, dtype=dtype, device=device)
    return Batch(data=data, z_nm=z)


def parameters(model, batch: Batch, values: dict, dtype=torch.float32, device=None) -> Params:
    """Every parameter of ``model`` for ``batch``, physical ``values`` overriding the inits."""
    E = batch.n_emitters
    C = batch.data.shape[1]
    start = {"xyz": values.get("emitter.xyz", torch.zeros(E, 3)),
             "photons": values.get("emitter.photons", torch.ones(E)),
             "background": values.get("emitter.background", torch.ones(E) if C == 1 else torch.ones(E, C))}
    return Params(model.specs(batch, start), n_emitters=E, dtype=dtype, device=device, values=values)


def simulate(model, params: Params, batch: Batch, seed: int = 0, read_noise_var: float = 0.0) -> Batch:
    """The batch with Poisson (+ Gaussian read noise) data drawn from the model's expected photons."""
    with torch.no_grad():
        mu = model(params.physical(), batch)
    g = torch.Generator().manual_seed(int(seed))
    # Copy, then cast: on MPS (torch 2.14) `.to("cpu", torch.float64)` in one call returns wrong values.
    counts = torch.poisson(to_cpu64(mu), generator=g)
    if read_noise_var > 0:
        counts = counts + torch.randn(counts.shape, generator=g, dtype=torch.float64) * read_noise_var ** 0.5
    return replace(batch, data=counts.to(batch.data.dtype).to(batch.data.device), s=float(read_noise_var))
