"""What happens to the intensity after the field: blur, the bead's size, the pixels.

All three act in Fourier space on the (upsampled) image region, as uiPSF
does, so the convolution is circular over the region; a margin against the
wrap-around is a phase-2 change (plan M7).  Frequencies are the FFT's own
(`torch.fft.fftfreq`).  uiPSF builds the blur on the grid
(i − L/2 + ½)/L, which is the same for odd L but half a sample off for
even L (its bin = 2 regions); there v2 differs from v1 on purpose.
"""
from __future__ import annotations

import math

import torch


def gaussian_otf(shape: tuple[int, int], spacing_nm: tuple[float, float], sigma_nm: torch.Tensor) -> torch.Tensor:
    """[Y, X] in FFT order: the transfer function of a Gaussian blur with sd (σx, σy) in nm."""
    Y, X = shape
    dev, dt = sigma_nm.device, sigma_nm.dtype
    qx = torch.fft.fftfreq(X, d=spacing_nm[0], device=dev, dtype=dt)
    qy = torch.fft.fftfreq(Y, d=spacing_nm[1], device=dev, dtype=dt)
    return torch.exp(-2 * math.pi ** 2 * (sigma_nm[0] ** 2 * qx[None, :] ** 2 + sigma_nm[1] ** 2 * qy[:, None] ** 2))


def blur_bin_matrices(shape: tuple[int, int], spacing_nm: tuple[float, float], sigma_nm: torch.Tensor,
                      factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(M_y [Y/f, Y], M_x [X/f, X]) with M_y · I · M_xᵀ = bin_pixels(ifft2(fft2(I) · gaussian_otf), f).

    The Gaussian's transfer function is a product of one in x and one in y, so the circular blur is a
    real circulant matrix on each axis, c[d] = (1/L) Σ_k h[k] cos(2πkd/L) (h is even in k), and the
    pixel sums fold into it.  Two small real products instead of two FFTs of the whole stack: on
    Apple's GPU the FFTs of 42-sample regions cost as much as the matrix DFT (plan §9 M7)."""
    out = []
    for L, d, s in ((shape[0], spacing_nm[1], sigma_nm[1]), (shape[1], spacing_nm[0], sigma_nm[0])):
        dev, dt = sigma_nm.device, sigma_nm.dtype
        q = torch.fft.fftfreq(L, d=d, device=dev, dtype=dt)
        h = torch.exp((s * q) ** 2 * (-2 * math.pi ** 2))     # tensor first: float * Tensor fails under vmap on MPS
        k = torch.arange(L, device=dev)
        diff = (k[:, None] - k[None, :]) % L                            # (m − n) mod L
        cos = torch.cos(2 * math.pi * (diff[..., None] * k).remainder(L).to(dt) / L)   # [L, L, L]
        C = cos @ h / L
        out.append(C.reshape(L // factor, factor, L).sum(1))
    return out[0], out[1]


def sphere_otf(shape: tuple[int, int, int], spacing_nm: tuple[float, float, float], radius_nm: float,
               device=None, dtype=torch.float32) -> torch.Tensor:
    """[Z, Y, X] in FFT order: the transform of a uniform sphere, 1 at zero frequency.

    3(sin t − t cos t)/t³ with t = 2π|q|R, which is uiPSF's J_{3/2}(t)/t^{3/2}
    up to its normalisation (review §A).
    """
    Z, Y, X = shape
    # Built in float64 on the CPU and moved afterwards: Apple's GPU (MPS) has no float64.
    q = [torch.fft.fftfreq(n, d=d, dtype=torch.float64)
         for n, d in zip((Z, Y, X), (spacing_nm[2], spacing_nm[1], spacing_nm[0]), strict=True)]
    t = 2 * math.pi * radius_nm * torch.sqrt(q[0][:, None, None] ** 2 + q[1][None, :, None] ** 2
                                             + q[2][None, None, :] ** 2)
    small = t < 1e-3
    ts = torch.where(small, torch.ones_like(t), t)
    f = torch.where(small, 1 - t ** 2 / 10, 3 * (torch.sin(ts) - ts * torch.cos(ts)) / ts ** 3)
    return f.to(device=device, dtype=dtype)


def bin_pixels(img: torch.Tensor, factor: int) -> torch.Tensor:
    """Sum factor × factor blocks of the last two axes: the upsampled samples of each camera pixel."""
    if factor == 1:
        return img
    *lead, Y, X = img.shape
    if Y % factor or X % factor:
        raise ValueError(f"region {Y}×{X} is not a multiple of the upsampling {factor}")
    return img.reshape(*lead, Y // factor, factor, X // factor, factor).sum((-3, -1))
