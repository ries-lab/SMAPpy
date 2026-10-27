"""Which PyTorch the PSF calibration's Install button brings, and when it is offered (`smappy.torch_install`)."""
import sys

from smappy import torch_install


def test_each_machine_gets_its_own_build_into_this_python():
    mac = torch_install.command("Darwin", nvidia=False, start=torch_install.installer(has_pip=True))
    assert mac[:4] == [sys.executable, "-m", "pip", "install"] and "--index-url" not in mac
    assert "--index-url" not in torch_install.command("Linux", nvidia=True)          # PyPI's wheel is CUDA's
    assert torch_install.command("Linux", nvidia=False)[-1] == torch_install.CPU_INDEX
    assert torch_install.command("Windows", nvidia=True)[-1] == torch_install.CUDA_INDEX
    assert "--index-url" not in torch_install.command("Windows", nvidia=False)
    assert "torch>=2.4" in mac


def test_a_python_without_pip_installs_through_uvs_pip_aimed_at_itself():
    start = torch_install.installer(has_pip=False, uv="C:/uv/uv.exe")
    assert start == ["C:/uv/uv.exe", "pip", "install", "--python", sys.executable]
    windows = torch_install.command("Windows", nvidia=True, start=start)
    assert windows[:5] == start and windows[-2:] == ["--index-url", torch_install.CUDA_INDEX]


def test_an_old_or_missing_torch_is_a_problem_and_a_current_one_is_not(monkeypatch):
    from importlib import metadata
    monkeypatch.setattr(metadata, "version", lambda name: "2.14.1+cu126")
    assert torch_install.problem() is None
    monkeypatch.setattr(metadata, "version", lambda name: "2.1.0")
    assert "too old" in torch_install.problem()

    def missing(name):
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(metadata, "version", missing)
    assert torch_install.problem() == "PyTorch is not installed"


def test_the_window_offers_the_install_only_for_the_pupil_model_and_only_while_torch_is_missing(monkeypatch):
    import pytest
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from smappy.calibrate.psf_window import PSFCalibrationWindow
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(torch_install, "problem", lambda: "PyTorch is not installed")
    window = PSFCalibrationWindow()
    window.method_box.setCurrentText("bead average")
    assert not window.install_button.isVisibleTo(window) and window.run_button.isEnabled()
    window.method_box.setCurrentText("pupil model (uiPSF)")
    assert window.install_button.isVisibleTo(window) and not window.run_button.isEnabled()
    assert " ".join(torch_install.command()) in window.notice.text()
    monkeypatch.setattr(torch_install, "problem", lambda: None)
    window._show_method()
    assert not window.install_button.isVisibleTo(window) and window.run_button.isEnabled()
    window.close()
