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

    from smappy.calibrate.psf_window import PSFCalibrationWindow, shell_line
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(torch_install, "problem", lambda: "PyTorch is not installed")
    window = PSFCalibrationWindow()
    window.method_box.setCurrentText("bead average")
    assert not window.install_button.isVisibleTo(window) and window.run_button.isEnabled()
    window.method_box.setCurrentText("pupil model (uiPSF)")
    assert window.install_button.isVisibleTo(window) and not window.run_button.isEnabled()
    assert shell_line(torch_install.command()) in window.notice.text()
    monkeypatch.setattr(torch_install, "problem", lambda: None)
    window._show_method()
    assert not window.install_button.isVisibleTo(window) and window.run_button.isEnabled()
    window.close()


def test_a_cpu_build_on_a_machine_with_an_nvidia_card_is_found_from_its_version():
    found = torch_install.cpu_build_on_nvidia
    assert found("2.14.1", "Windows", nvidia=True)               # PyPI's Windows wheel is the CPU's
    assert not found("2.14.1+cu126", "Windows", nvidia=True)
    assert found("2.14.1+cpu", "Linux", nvidia=True)
    assert not found("2.14.1", "Linux", nvidia=True)             # PyPI's Linux wheel is CUDA's
    assert not found("2.14.1", "Windows", nvidia=False)          # no card: the CPU is all there is
    assert not found("2.14.1", "Darwin", nvidia=False)


def test_replacing_a_cpu_build_reinstalls_from_the_cuda_index():
    windows = torch_install.command("Windows", nvidia=True, replace=True)
    assert windows[-4:] == ["--force-reinstall", "--no-deps", "--index-url", torch_install.CUDA_INDEX]
    linux = torch_install.command("Linux", nvidia=True, replace=True)
    assert linux[-2:] == ["--index-url", torch_install.CUDA_INDEX] and "--no-deps" not in linux
    assert torch_install.command("Darwin", nvidia=False, replace=True) == torch_install.command("Darwin", nvidia=False)


def test_the_window_warns_of_a_cpu_build_beside_a_card_offers_the_install_and_still_runs(monkeypatch):
    import pytest
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from smappy.calibrate.psf_window import PSFCalibrationWindow
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(torch_install, "problem", lambda: None)
    monkeypatch.setattr(torch_install, "_installed", lambda: "2.14.1")
    monkeypatch.setattr(torch_install, "has_nvidia", lambda: True)
    monkeypatch.setattr(torch_install.platform, "system", lambda: "Windows")
    window = PSFCalibrationWindow()
    window.method_box.setCurrentText("pupil model (uiPSF)")
    assert window.install_button.isVisibleTo(window) and window.run_button.isEnabled()
    assert "built for the CPU only" in window.notice.text() and torch_install.CUDA_INDEX in window.notice.text()
    monkeypatch.setattr(torch_install, "has_nvidia", lambda: False)
    window._show_method()
    assert not window.install_button.isVisibleTo(window) and not window.notice.isVisibleTo(window)
    window.close()


def test_after_detect_the_window_says_where_the_pupil_model_learns(monkeypatch):
    import pytest
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from smappy.calibrate import psf_window
    from smappy.calibrate.psf_methods import Detected
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(torch_install, "problem", lambda: None)
    monkeypatch.setattr(psf_window, "learning_device", lambda: "on the graphics card (a test card)")
    window = psf_window.PSFCalibrationWindow()
    window.method_box.setCurrentText("pupil model (uiPSF)")
    window.method = type("M", (), {"key": "pupil"})()      # its own text, not a method's detected_text
    window.detected = Detected(None, "12 beads to learn from")
    window._show_detected_text()
    assert window.found_label.text() == "12 beads to learn from; learnt on the graphics card (a test card)"
    window.close()


def test_the_device_is_described_for_a_person():
    import pytest
    pytest.importorskip("torch")
    from uipsf.device import describe
    assert describe("cpu") == "on the CPU" and describe("mps") == "on Apple's GPU"
