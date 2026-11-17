"""``smappy-gui``'s options, ``smappy info`` and ``smappy convert``."""
from pathlib import Path

import numpy as np
import pytest

from smappy.cli import convert as convert_cli
from smappy.cli import gui as gui_cli
from smappy.cli import info as info_cli
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


# ------------------------------------------------------------ smappy info

def test_info_says_each_file_its_camera_its_columns_and_its_rois(tmp_path):
    session = Session()
    session.load(_two_files(tmp_path))
    text = "\n".join(info_cli.summary(session))
    assert "55 localizations in 3 files" in text
    rows = [line.split() for line in text.splitlines()]
    assert ["1", "Pos1_sml.mat", "20", "127", "nm", "35.31", "ms"] in rows
    # the two positions share a camera, the average has its own
    assert "camera of files 0-1: conversion 5 e-/ADU, offset 398.6 ADU, EM gain 200" in text
    assert "camera of file 2: conversion 1 e-/ADU" in text
    assert "5 columns: frame photons x_nm y_nm filenumber" in text
    assert "1 ROI" in text and "ROIs" not in text.split("columns")[1]


def test_info_goes_on_past_a_file_it_cannot_read(tmp_path, capsys):
    good = _two_files(tmp_path)
    assert info_cli.main([str(tmp_path / "missing.hdf5"), str(good)]) == 1
    out = capsys.readouterr()
    assert "missing.hdf5: no such file" in out.err
    assert "experiment_sml.mat  (SMAP)" in out.out


# --------------------------------------------------------- smappy convert

def test_a_smap_file_converts_to_smappy_hdf5_keeping_its_files_and_rois(tmp_path):
    source = _two_files(tmp_path)
    said = []
    written = convert_cli.convert(source, say=said.append)
    assert written == tmp_path / "experiment.hdf5"      # the reader's suffix off
    before, after = Session(), Session()
    before.load(source)
    after.load(written)
    assert after.file_names() == before.file_names()
    assert [f.pixelsize_nm for f in after.files] == pytest.approx([127, 127, 100])
    assert np.array_equal(after.locs["x_nm"], before.locs["x_nm"])
    assert [r.number for r in after.rois.rois.values()] == [4]
    assert said == [f"experiment_sml.mat -> {written} (smappy): 55 localizations in 3 files"]


def test_a_csv_keeps_every_localization_and_says_what_it_cannot_hold(tmp_path):
    source = _two_files(tmp_path)
    said = []
    written = convert_cli.convert(source, tmp_path / "out.csv", say=said.append)
    # read as it is: opened as one file, its rows would all be that file's
    table = np.genfromtxt(written, delimiter=",", names=True)
    assert len(table) == 55
    assert np.array_equal(np.sort(table["filenumber"]), [0] * 30 + [1] * 20 + [2] * 5)
    assert "not in a csv file: the files' cameras, the ROI" in said[1]


def test_convert_never_writes_over_a_file_unless_told(tmp_path):
    source = _two_files(tmp_path)
    out = convert_cli.convert(source, say=lambda _: None)
    with pytest.raises(FileExistsError):
        convert_cli.convert(source, say=lambda _: None)
    assert convert_cli.convert(source, force=True, say=lambda _: None) == out
    with pytest.raises(ValueError, match="the file being read"):
        convert_cli.convert(out, out, say=lambda _: None)
    # read as smappy, written as Picasso's: a name of its own, not the file
    assert convert_cli.default_output(out, "Picasso") == tmp_path / "experiment_picasso.hdf5"


def test_an_output_no_format_writes_is_refused():
    with pytest.raises(ValueError, match="no format writes .png"):
        convert_cli.output_format(Path("picture.png"), None)
    assert convert_cli.output_format(Path("x.tsv"), None) == "csv"
    assert convert_cli.output_format(Path("x.hdf5"), None) == "smappy"
    assert convert_cli.output_format(Path("x.hdf5"), "Picasso") == "Picasso"
