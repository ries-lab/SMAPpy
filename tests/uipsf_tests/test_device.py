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
