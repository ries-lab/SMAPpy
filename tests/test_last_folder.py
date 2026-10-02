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
