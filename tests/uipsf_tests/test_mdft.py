import math

import torch

import uipsf
from uipsf.optics import MatrixDFT


def _grid(optics, P=16):
    return uipsf.PupilGrid.build(optics, P, device="cpu", dtype=torch.float64)


def test_the_matrix_dft_is_the_direct_sum_over_the_pupil(optics):
    g = _grid(optics)
    dft = MatrixDFT(g, (5, 7), (40.0, 30.0))
    F = torch.randn(2, 16, 16, dtype=torch.complex128)
    x0, y0 = torch.tensor([13.0, -7.0]), torch.tensor([-4.0, 22.0])
    E = dft(F, x0, y0)
    for e in range(2):
        x = dft.x - x0[e]
        y = dft.y - y0[e]
        phase = 2 * math.pi * (y[:, None, None, None] * g.ky[None, None, :, None]
                               + x[None, :, None, None] * g.kx[None, None, None, :])
        direct = (F[e] * torch.exp(1j * phase)).sum((-2, -1)) * g.dk ** 2
        assert torch.allclose(E[e], direct, atol=1e-12)


def test_on_the_fft_grid_the_matrix_dft_equals_a_zero_padded_fft(optics):
    g = _grid(optics)
    N = 64                                            # image period 1/dk sampled N times
    dft = MatrixDFT(g, (N, N), (1 / (N * g.dk), 1 / (N * g.dk)))
    F = torch.randn(16, 16, dtype=torch.complex128)
    a, b = (16 - 1) / 2, (N - 1) / 2
    n = torch.arange(16, dtype=torch.float64)
    m = torch.arange(N, dtype=torch.float64)
    # Σ_n F_n e^{2πi(n−a)(m−b)/N} = e^{−2πi a(m−b)/N} · N·ifft(F_n e^{−2πi n b/N})_m, on both axes
    pre = torch.exp(-2j * math.pi * n * b / N)
    post = torch.exp(-2j * math.pi * a * (m - b) / N)
    padded = torch.zeros(N, N, dtype=torch.complex128)
    padded[:16, :16] = F * pre[:, None] * pre[None, :]
    via_fft = torch.fft.ifft2(padded) * N * N * post[:, None] * post[None, :] * g.dk ** 2
    assert torch.allclose(dft(F), via_fft, atol=1e-10)


def test_an_odd_region_has_its_centre_sample_on_the_emitter(optics):
    dft = MatrixDFT(_grid(optics), (5, 5), (50.0, 50.0))
    assert dft.x[2] == 0 and dft.y[2] == 0


def test_gradients_flow_through_the_matrix_dft(optics):
    g = _grid(optics, 8)
    dft = MatrixDFT(g, (3, 3), (60.0, 60.0))

    def f(re, im, x0):
        E = dft(torch.complex(re, im), x0)
        return (E.real ** 2 + E.imag ** 2)

    args = (torch.randn(8, 8, dtype=torch.float64, requires_grad=True),
            torch.randn(8, 8, dtype=torch.float64, requires_grad=True),
            torch.tensor(17.0, dtype=torch.float64, requires_grad=True))
    assert torch.autograd.gradcheck(f, args)
