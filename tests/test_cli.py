"""``smappy-gui``'s options, checked before any window opens."""
import pytest

from smappy.cli import gui as gui_cli
from smappy.session import Session
from test_sml_files import AVERAGE, CAMERA, _rows, write_sml


def _two_files(tmp_path):
    """A SMAP file of two positions and an average, with a site on Pos1."""
    from test_sml_sites import _site
    rows = _rows([1] * 30 + [2] * 20 + [3] * 5)
    rows["frame"] *= 100                 # apart in time: no two merged into a blink
    site = _site(4, (1000.0, 2000.0))
    site["info"]["filenumber"] = 2.0
    files = [("C:\\data\\Pos0_sml.mat", CAMERA), ("C:\\data\\Pos1_sml.mat", CAMERA),
             ("all_average_sites_sml.mat", AVERAGE)]
    return write_sml(tmp_path / "experiment_sml.mat", files, rows, sites=[site])


# ------------------------------------------------------------- smappy gui

def test_a_flag_is_never_taken_for_a_file_to_open(capsys):
    """`smappy-gui -h` used to open the windows and fail to read a file "-h"."""
    with pytest.raises(SystemExit) as exit:
        gui_cli.main(["-h"])
    assert exit.value.code == 0
    assert "--workspace" in capsys.readouterr().out


@pytest.mark.parametrize("line, said", [
    (["missing.hdf5"], "no file missing.hdf5"),
    (["--workspace", "missing.yaml"], "no workspace file"),
    (["--restore"], "--restore takes one smappy HDF5 file"),
    (["--open", "RCX"], "perhaps Analysis/Drift/RCC"),
])
def test_what_cannot_be_done_is_said_before_any_window_opens(line, said, capsys):
    with pytest.raises(SystemExit) as exit:
        gui_cli.main(line)
    assert exit.value.code == 2
    assert said in capsys.readouterr().err


def test_a_plugin_is_found_by_its_path_its_name_or_the_end_of_its_path():
    from smappy import plugins
    refs = plugins.refs()
    path = "Analysis/Drift/COMET"
    assert gui_cli.find_plugin(path, refs) == path
    assert gui_cli.find_plugin("comet", refs) == path
    assert gui_cli.find_plugin(refs[path].name.upper(), refs) == path


def test_the_windows_start_with_the_workspace_files_and_plugins_asked_for(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    monkeypatch.setenv("SMAPPY_CONFIG_DIR", str(tmp_path / "config"))
    from smappy import config
    config.load(reload=True)
    from smappy.gui.app import ControlWindow, RenderWindow, start_with
    app = QApplication.instance() or QApplication([])
    session = Session()
    control = ControlWindow(session, RenderWindow(session))
    workspace = tmp_path / "mine.yaml"
    control.save_gui_to(workspace)
    path = _two_files(tmp_path)
    start_with(control, files=[str(path)], workspace=str(workspace),
               open_plugins=["Analysis/Drift/RCC"])
    for _ in range(200):                 # the file is read on a worker
        app.processEvents()
        if session.path:
            break
        import time
        time.sleep(0.05)
    assert control.workspace.path == workspace
    assert session.file_names() == ["Pos0_sml.mat", "Pos1_sml.mat",
                                    "all_average_sites_sml.mat"]
    assert control._plugin_windows["Analysis/Drift/RCC"].isVisible()
    for window in control._plugin_windows.values():
        window.close()
    control.close()

