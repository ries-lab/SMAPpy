"""The data sets in files and folders, one engine for the batch and the
calibration windows (`smappy.datasets`), and the filter that narrows them."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from smappy import batch, datasets
from smappy.batch import Input, Job
from smappy.chain import ChainSpec


def _tif(path, n=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.zeros((n, 4, 4), np.uint16))
    return path


def _h5(path):
    from smappy.io.formats import save
    from smappy.simulate import simulate
    path.parent.mkdir(parents=True, exist_ok=True)
    save(simulate(n_frames=50, seed=1), path)
    return path


@pytest.fixture
def day(tmp_path):
    """A day's folder: a Micro-Manager series in two files, a plain stack, an
    NDTiff folder, a bead stack in a subfolder, and fits beside them."""
    _tif(tmp_path / "cell1_MMStack_Pos0.ome.tif")
    _tif(tmp_path / "cell1_MMStack_Pos0_1.ome.tif")
    _tif(tmp_path / "cell2_640.tif")
    nd = tmp_path / "cell3_NDTiff"
    _tif(nd / "NDTiffStack.tif")
    (nd / "NDTiff.index").write_bytes(b"")
    _tif(tmp_path / "beads" / "beads_640.tif")
    _h5(tmp_path / "cell2_640_locs.h5")
    _h5(tmp_path / "beads" / "beads_locs.h5")
    return tmp_path


def names(found, folder):
    return [datasets.relative(p, folder) for p in found]


def test_a_series_is_one_data_set_an_ndtiff_folder_is_one_and_fits_are_another_kind(day):
    assert names(datasets.find([day], "images"), day) == [
        "beads/beads_640.tif", "cell1_MMStack_Pos0.ome.tif", "cell2_640.tif", "cell3_NDTiff"]
    assert names(datasets.find([day], "localizations"), day) == [
        "beads/beads_locs.h5", "cell2_640_locs.h5"]
    assert names(datasets.find([day], "images", recursive=False), day) == [
        "cell1_MMStack_Pos0.ome.tif", "cell2_640.tif", "cell3_NDTiff"]


def test_a_part_of_a_series_names_the_series_and_its_parts_are_every_file(day):
    second = day / "cell1_MMStack_Pos0_1.ome.tif"
    assert datasets.canonical(second) == day / "cell1_MMStack_Pos0.ome.tif"
    assert [p.name for p in datasets.parts(day / "cell1_MMStack_Pos0.ome.tif")] == [
        "cell1_MMStack_Pos0.ome.tif", "cell1_MMStack_Pos0_1.ome.tif"]
    assert datasets.canonical(day / "cell3_NDTiff" / "NDTiffStack.tif") == day / "cell3_NDTiff"
    assert {p.name for p in datasets.parts(day / "cell3_NDTiff")} == {"NDTiffStack.tif",
                                                                     "NDTiff.index"}


def test_the_calibration_windows_search_is_the_same_engine(day):
    from smappy.calibrate.input import discover_acquisitions
    assert discover_acquisitions([day]) == datasets.find([day], "images")


@pytest.mark.parametrize("text, kept", [
    ("", ["a_640.tif", "a_647.tif", "a_640_647.tif", "b_600.tif", "beads/c_640.tif"]),
    ("*640*, *600*", ["a_640.tif", "a_640_647.tif", "b_600.tif", "beads/c_640.tif"]),
    ("*640* | *600*", ["a_640.tif", "a_640_647.tif", "b_600.tif", "beads/c_640.tif"]),
    ("(*640* & !*647*), *600*", ["a_640.tif", "b_600.tif", "beads/c_640.tif"]),
    ("(*640* AND NOT *647*) OR *600*", ["a_640.tif", "b_600.tif", "beads/c_640.tif"]),
    ("!beads/*", ["a_640.tif", "a_647.tif", "a_640_647.tif", "b_600.tif"]),
    ("!*BEAD*", ["a_640.tif", "a_647.tif", "a_640_647.tif", "b_600.tif"]),
    ("a_6[!4]*", []),
    ("a_64[!0]*", ["a_647.tif"]),
])
def test_the_filter_reads_or_and_not_and_parentheses_as_a_search_box_does(text, kept):
    paths = ["a_640.tif", "a_647.tif", "a_640_647.tif", "b_600.tif", "beads/c_640.tif"]
    keep = datasets.parse_filter(text)
    assert [p for p in paths if keep(p)] == kept


def test_a_name_with_spaces_needs_no_quotes_and_one_with_brackets_is_quoted():
    assert datasets.parse_filter("my cell*")("my cell 1.tif")
    assert datasets.parse_filter('"cell (2)*"')("cell (2).tif")


@pytest.mark.parametrize("text, says", [
    ("(*640*", "not closed"), ("*640* &", "ends where a pattern"),
    ("& *640*", "where a pattern should be"), ('"*640*', "quote"),
    ("*640* (*647*)", "follows a complete filter")])
def test_a_filter_that_cannot_be_read_says_where(text, says):
    assert says in datasets.filter_error(text)


def test_a_folder_input_runs_its_data_sets_of_the_chains_kind_filtered_and_unticked(day):
    job = Job(chain=ChainSpec(), inputs=[Input(folder=str(day), filter="*640*, *cell1*",
                                               exclude=["beads/beads_640.tif"])])
    images = batch.entry_datasets(job.inputs[0], "images", day)
    assert names(images[0], day) == ["cell1_MMStack_Pos0.ome.tif", "cell2_640.tif"]
    assert names(images[1], day) == ["beads/beads_640.tif"]
    # the same folder, for a chain of localizations: the fits
    assert names(batch.entry_datasets(job.inputs[0], "localizations", day)[0], day) == [
        "cell2_640_locs.h5"]


def test_a_file_of_the_other_kind_is_skipped_with_a_warning_not_an_error(day):
    from smappy.chain import Step
    spec = ChainSpec(steps=[Step(plugin="Analysis/Measure/Localization Statistics",
                                 label="stats")])
    job = Job(chain=spec, inputs=[Input(file=str(day / "cell2_640.tif")),
                                  Input(file=str(day / "cell2_640_locs.h5"))])
    assert [i.path.name for i in batch.resolve_inputs(job, "localizations")] == [
        "cell2_640_locs.h5"]
    problems = batch.validate(job)
    assert [p.level for p in problems] == ["warning"]
    assert "cell2_640.tif is skipped" in problems[0].message


def test_a_job_with_a_filter_that_cannot_be_read_does_not_run(day):
    job = Job(chain=ChainSpec(), inputs=[Input(folder=str(day), filter="(*640*")])
    assert any("not closed" in str(p) for p in batch.validate(job))


def test_an_ndtiff_folder_is_complete_once_none_of_its_files_changes(day):
    seen = {}
    folder = day / "cell3_NDTiff"
    import os
    import time
    old = time.time() - 100
    for f in folder.iterdir():
        os.utime(f, (old, old))
    assert batch.is_complete(folder, seen, settle=5, now=old + 6)
    _tif(folder / "NDTiffStack_1.tif")                  # the writer begins the next file
    assert not batch.is_complete(folder, seen, settle=5)
