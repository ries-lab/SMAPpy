import pytest
import torch

from uipsf.device import complex_dtype, default_device, resolve, to_cpu64


def test_the_environment_variable_chooses_the_device(monkeypatch):
    monkeypatch.setenv("UIPSF_DEVICE", "cpu")
    assert default_device() == torch.device("cpu")


def test_apple_gpu_refuses_float64_and_accepts_float32():
    with pytest.raises(ValueError, match="no float64"):
        resolve("mps", torch.float64)
    assert resolve("mps", torch.float32) == (torch.device("mps"), torch.float32)


def test_only_real_dtypes_are_accepted():
    with pytest.raises(ValueError):
        resolve("cpu", torch.complex64)


def test_complex_dtypes_match_real_ones_and_the_solver_copy_is_double():
    assert complex_dtype(torch.float32) == torch.complex64
    assert complex_dtype(torch.float64) == torch.complex128
    assert to_cpu64(torch.ones(2, dtype=torch.float32)).dtype == torch.float64
    assert to_cpu64(torch.ones(2, dtype=torch.complex64)).dtype == torch.complex128


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple's GPU")
def test_bounded_and_floored_scalars_take_jacobian_columns_on_the_apple_gpu():
    # A Python number meeting a 0-d tensor under vmap(jvp) became a float64 tensor, which MPS refuses: two
    # channels' photon share stopped every two-channel fit there (torch 2.14).
    from uipsf.params import Bounded, Softplus
    x = torch.zeros((), device="mps")
    for transform in (Bounded(0.0, 1.0), Softplus(1e-3)):
        cols = torch.func.vmap(lambda t, f=transform: torch.func.jvp(f.forward, (x,), (t,))[1])(
            torch.ones(3, device="mps"))
        assert cols.shape == (3,) and torch.isfinite(cols).all()
