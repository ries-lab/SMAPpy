"""The folder file dialogs start in: the last one a file was opened or saved in."""
from smappy import config


def test_a_file_remembers_its_own_folder(tmp_path):
    (tmp_path / "run").mkdir()
    config.remember_folder(tmp_path / "run" / "locs.h5")
    assert config.last_folder() == (tmp_path / "run").resolve()


def test_a_folder_remembers_itself(tmp_path):
    config.remember_folder(tmp_path)
    assert config.last_folder() == tmp_path.resolve()


def test_a_camera_stack_remembers_the_folder_above_its_own(tmp_path):
    from smappy.gui import folders
    stack = tmp_path / "cell1" / "cell1_MMStack.ome.tif"
    stack.parent.mkdir()
    stack.write_bytes(b"")
    folders.remember(stack, stack=True)
    assert config.last_folder() == tmp_path.resolve()
    recipe = tmp_path / "cell1" / "beads.sim.yaml"     # the same field, not a TIFF
    folders.remember(recipe, stack=True)
    assert config.last_folder() == (tmp_path / "cell1").resolve()


def test_a_folder_that_has_gone_gives_its_nearest_parent(tmp_path):
    gone = tmp_path / "a" / "b"
    gone.mkdir(parents=True)
    config.remember_folder(gone)
    gone.rmdir()
    (tmp_path / "a").rmdir()
    assert config.last_folder() == tmp_path.resolve()


def test_a_dialog_starts_in_the_last_folder_with_the_suggested_name(tmp_path):
    from smappy.gui import folders
    config.remember_folder(tmp_path)
    assert folders.start() == str(tmp_path.resolve())
    assert folders.start("image.png") == str(tmp_path.resolve() / "image.png")


def test_the_data_folder_is_beside_a_movie_one_up_for_a_movie_in_its_own_folder(tmp_path):
    from smappy.gui import folders
    acquisition = tmp_path / "Pos1_cells"
    acquisition.mkdir()
    movie = acquisition / "Pos1_cells_MMStack_Pos0.ome.tif"
    movie.write_bytes(b"")
    assert folders.data_folder_of(movie) == tmp_path
    loose = tmp_path / "cells.tif"
    loose.write_bytes(b"")
    assert folders.data_folder_of(loose) == tmp_path
    assert folders.data_folder_of("") is None


def test_choosing_a_fitters_movie_makes_its_folder_where_empty_path_fields_open(tmp_path,
                                                                               monkeypatch):
    """Bead movies, a transformation, a calibration: looked for beside the data, whatever
    dialog was used last (2026-11)."""
    import pytest
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QFileDialog
    QApplication.instance() or QApplication([])
    from smappy.gui import folders
    from smappy.gui.plugin_panel import PluginPanel
    from smappy.plugins.fit import GaussianFit
    from smappy.session import Session
    monkeypatch.setattr(folders, "_data", None)
    elsewhere = tmp_path / "calibrations"
    elsewhere.mkdir()
    config.remember_folder(elsewhere)                       # the last dialog's folder
    movie = tmp_path / "cells.tif"
    movie.write_bytes(b"")
    panel = PluginPanel(GaussianFit, Session())
    source = panel.form.fields["source"].fields["path"]
    source.widget.setText(str(movie))
    source.changed.emit()
    opened = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda parent, title, start: opened.append(start) or ""))
    panel.form.fields["finish"].fields["overlay_channels"].fields["beads"]._browse()
    assert opened == [str(tmp_path)]
