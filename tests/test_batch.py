"""A chain over many files: resolved, validated strictly, each file processed
alone, its numbers and figures written, and a re-run skipping what is done."""
import csv
import json
from dataclasses import dataclass, field

import numpy as np
import pytest
import yaml

from smappy import batch, chain
from smappy.batch import BatchError, Input, Job, Output
from smappy.chain import ChainSpec, Step
from smappy.io.hdf5 import load_localizations, save_localizations
from smappy.plugins import Plugin, Result, param, register

from test_batch_foundations import blinks


def chain_spec(**extra):
    return ChainSpec(steps=[
        Step(plugin="Chain/Layers", label="filter",
             values={"layers": [{"grouped": False, "start": "empty",
                                 "bounds": [{"field": "loc_precision_nm", "hi": 20}]}],
                     "remove": True}),
        Step(plugin="Analysis/Process/Math Parser", label="flag",
             values={"field": "bright", "expression": "photons > 500"}),
        Step(plugin="Analysis/Measure/Localization Statistics", label="stats",
             values={"source": "all", "bins": 20}),
    ], **extra)


@pytest.fixture
def folder(tmp_path):
    data = tmp_path / "data"
    for n, (a, b) in enumerate([(5, 40), (5, 25), (10, 60)]):
        locs = blinks(seed=n)
        locs.columns["loc_precision_nm"] = np.repeat(np.linspace(a, b, 40), 4)
        (data / f"cell{n}").mkdir(parents=True)
        save_localizations(data / f"cell{n}" / "locs.h5", locs)
    return data


def write(tmp_path, job):
    path = tmp_path / "job.batch.yaml"
    batch.write_job(job, path)
    return batch.read_job(path)


def test_a_job_file_reads_back_as_it_was_written(tmp_path, folder):
    job = Job(chain=chain_spec(), inputs=[Input(folder=str(folder), exclude=["*bead*"]),
                                          Input(file=str(folder / "cell0" / "locs.h5"),
                                                overrides={"stats.bins": 10})],
              output=Output(figures="svg"), overrides={"flag.field": "b"}, name="t")
    back = write(tmp_path, job)
    assert back.to_dict() == job.to_dict()
    # a chain given by file is written by file
    chain.write(chain_spec(), tmp_path / "c.chain.yaml")
    raw = yaml.safe_load((tmp_path / "job.batch.yaml").read_text())
    raw["chain"] = "c.chain.yaml"
    (tmp_path / "j2.batch.yaml").write_text(yaml.safe_dump(raw))
    assert batch.read_job(tmp_path / "j2.batch.yaml").to_dict()["chain"].endswith("c.chain.yaml")


def test_inputs_resolve_to_one_list_with_names_that_do_not_collide(tmp_path, folder):
    job = Job(chain=chain_spec(), inputs=[Input(folder=str(folder)),
                                          Input(file=str(folder / "cell1" / "locs.h5"))],
              output=Output(folder=str(folder / "out")))
    (folder / "out" / "old").mkdir(parents=True)
    save_localizations(folder / "out" / "old" / "locs_proc.h5", blinks())
    items = batch.resolve_inputs(job, "localizations")
    assert [i.name for i in items] == ["cell0_locs", "cell1_locs", "cell2_locs"]
    job.inputs[0].exclude = ["cell2/*"]
    assert [i.name for i in batch.resolve_inputs(job, "localizations")] == \
        ["cell0_locs", "cell1_locs"]


def test_validation_is_strict_where_reading_is_tolerant(tmp_path, folder):
    job = Job(chain=chain_spec(), inputs=[Input(folder=str(folder))])
    assert batch.validate(job) == []
    job.chain.steps[2].values["binz"] = 3
    job.chain.steps[1].values["where"] = "everywhere"
    job.overrides = {"stats.bins": "many", "nope.x": 1}
    messages = [str(p) for p in batch.validate(job)]
    assert any("no field 'binz'" in m for m in messages)
    assert any("'everywhere' is not one of" in m for m in messages)
    assert any("stats.bins: 'many' is not a number" in m for m in messages)
    assert any("no field 'nope.x'" in m for m in messages)
    empty = Job(chain=chain_spec(), inputs=[Input(folder=str(tmp_path / "empty_dir"))])
    (tmp_path / "empty_dir").mkdir()
    assert any("nothing to process" in str(p) for p in batch.validate(empty))
    missing = Job(chain=ChainSpec(steps=[Step(plugin="Analysis/Gone")]),
                  inputs=[Input(folder=str(folder))])
    assert any("not installed" in str(p) for p in batch.validate(missing))


def test_a_batch_processes_each_file_alone_and_writes_what_it_found(tmp_path, folder):
    job = write(tmp_path, Job(
        chain=chain_spec(), name="t",
        inputs=[Input(folder=str(folder)),
                Input(file=str(folder / "cell2" / "locs.h5"),
                      overrides={"filter": {"remove": False}})]))
    lines = []
    report = batch.run(job, emit=lambda *f: lines.append(f))
    out = tmp_path / "job_output"
    assert report["status"] == "complete"
    assert [f["status"] for f in report["files"]] == ["done"] * 3
    assert lines[0] == ("JOBS", 3) and lines[-1][0] == "REPORT"

    # the processed table, with the chain's entry at the end of its log
    kept = load_localizations(out / "cell0_locs" / "locs_proc.h5")
    assert len(kept) == 4 * np.sum(np.linspace(5, 40, 40) <= 20)
    assert "bright" in kept
    history = kept.metadata["history"]
    assert history[-1]["what"] == "Analysis/Chains/Unsaved chain"
    assert [s["label"] for s in history[-1]["steps"]] == ["filter", "flag", "stats"]
    # a file's own override: cell2 was filtered but nothing was removed
    all_of_it = load_localizations(out / "cell2_locs" / "locs_proc.h5")
    assert len(all_of_it) == 160

    record = json.loads((out / "cell0_locs" / "results.json").read_text())
    assert record["status"] == "done" and record["n_localizations"] == len(kept)
    assert record["steps"][2]["data"]["n"] == len(kept)
    assert any(p.endswith(".png") for p in record["output"]["figures"])
    with (out / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["status"] for r in rows] == ["done"] * 3
    assert int(rows[0]["stats.n"]) == len(kept)
    assert json.loads((out / "batch_report.json").read_text())["status"] == "complete"


def test_a_rerun_skips_what_is_done_and_redoes_what_changed(tmp_path, folder):
    job = write(tmp_path, Job(chain=chain_spec(), inputs=[Input(folder=str(folder))],
                              output=Output(figures="none")))
    batch.run(job)
    again = batch.run(job)
    assert [f["status"] for f in again["files"]] == ["unchanged"] * 3
    job.inputs.append(Input(file=str(folder / "cell1" / "locs.h5"),
                            overrides={"stats.bins": 30}))
    # a second mention of cell1 gives it overrides: that file, and only it, again
    assert [f["status"] for f in batch.run(job)["files"]] == ["unchanged", "done",
                                                              "unchanged"]
    # the job's override reaches the other two; cell1 has had 30 of its own
    job.overrides = {"stats.bins": 30}
    assert [f["status"] for f in batch.run(job)["files"]] == ["done", "unchanged", "done"]
    # the summary still has every row after a run that skipped them all
    batch.run(job)
    with (tmp_path / "job_output" / "summary.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 3


def test_one_bad_file_costs_that_file(tmp_path, folder):
    (folder / "cell1" / "locs.h5").write_bytes(b"not hdf5")
    job = write(tmp_path, Job(chain=chain_spec(), inputs=[Input(folder=str(folder))],
                              output=Output(figures="none")))
    report = batch.run(job)
    assert [f["status"] for f in report["files"]] == ["done", "failed", "done"]
    assert report["status"] == "partial"
    record = json.loads((tmp_path / "job_output" / "cell1_locs" / "results.json").read_text())
    assert record["status"] == "failed" and "Traceback" in record["traceback"]


@register("Test/Batch/Slow")
class Slow(Plugin):
    def preflight(self, ctx, settings):
        return "this takes an afternoon"

    def run(self, ctx, settings):
        return Result(text="ran")


def test_a_preflight_question_skips_the_file_unless_the_job_says_otherwise(tmp_path, folder):
    spec = ChainSpec(steps=[Step(plugin="Test/Batch/Slow")])
    job = write(tmp_path, Job(chain=spec, inputs=[Input(file=str(folder / "cell0" / "locs.h5"))],
                              output=Output(figures="none")))
    report = batch.run(job)
    assert report["files"][0]["status"] == "skipped"
    assert "afternoon" in report["files"][0]["message"]
    job.preflight = "proceed"
    assert batch.run(job)["files"][0]["status"] == "done"
    job.preflight = "abort"
    assert batch.run(job, force=True)["status"] == "aborted"


# A stand-in for a fit: a plugin that makes its table from a source file and
# writes its own file, as `Localize/*` do -- without fitting anything.

@dataclass
class _Source:
    path: str = param("", kind="open_file")


@dataclass
class _Out:
    save: bool = True
    path: str = ""


@dataclass
class FakeFitSettings:
    source: _Source = field(default_factory=_Source)
    output: _Out = field(default_factory=_Out)


@register("Localize/Test Fake")
class FakeFit(Plugin):
    Settings = FakeFitSettings

    def run(self, ctx, settings):
        from pathlib import Path
        from smappy.plugins.fit import default_output_path
        n = len(Path(settings.source.path).read_text())      # "images" of any size
        out = Path(settings.output.path or default_output_path(settings.source.path))
        locs = blinks(n_emitters=n)
        save_localizations(out, locs)
        return Result(locs=locs, text=f"{len(locs)} fitted", data={"path": out})


def test_images_in_means_the_first_step_is_given_each_file(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for n in (3, 5):
        (images / f"acq{n}.tif").write_text("x" * n)
    spec = ChainSpec(steps=[Step(plugin="Localize/Test Fake", label="fit"),
                            Step(plugin="Analysis/Process/Math Parser", label="flag",
                                 values={"field": "one", "expression": "photons > 0"})])
    job = write(tmp_path, Job(chain=spec, inputs=[Input(folder=str(images))],
                              output=Output(figures="none", fitted="output",
                                            delete_fitted=True)))
    assert batch.validate(job) == []
    report = batch.run(job)
    assert [f["status"] for f in report["files"]] == ["done", "done"]
    out = tmp_path / "job_output"
    processed = load_localizations(out / "acq3" / "acq3_proc.h5")
    assert len(processed) == 12 and "one" in processed
    assert not (out / "acq3" / "acq3_locs.hdf5").exists()       # deleted, as asked
    record = json.loads((out / "acq5" / "results.json").read_text())
    assert record["output"]["fitted_deleted"] and record["n_localizations"] == 20


def test_the_command_line_validates_runs_and_describes(tmp_path, folder, capsys):
    from smappy.cli.batch import main
    job = Job(chain=chain_spec(), inputs=[Input(folder=str(folder))],
              output=Output(figures="none"))
    path = tmp_path / "job.batch.yaml"
    batch.write_job(job, path)
    assert main(["validate", str(path)]) == 0
    assert main(["run", str(path), "--limit", "1"]) == 0
    out = capsys.readouterr().out
    assert "SMAPPY_JOBS\t1" in out and "SMAPPY_DONE\t1/1" in out
    assert main(["describe", "Chain/Layers"]) == 0
    assert "remove: bool" in capsys.readouterr().out
    chain.write(chain_spec(), tmp_path / "c.chain.yaml")
    assert main(["describe", str(tmp_path / "c.chain.yaml")]) == 0
    assert "stats.bins = 20" in capsys.readouterr().out
    job.chain.steps[0].values["nonsense"] = 1
    batch.write_job(job, path)
    assert main(["validate", str(path)]) == 2
    assert main(["run", str(path)]) == 2


def test_a_real_fit_then_an_analysis_over_two_stacks(tmp_path):
    import tifffile
    from smappy.plugins.fit import GaussianFit, GaussianFitSettings, CameraSettings, \
        DetectionSettings
    from smappy.plugins import settings_values
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:48, 0:48]
    for name in ("a", "b"):
        frames = []
        for _ in range(12):
            clean = np.zeros((48, 48))
            for _ in range(3):
                cx, cy = rng.uniform(8, 40, 2)
                clean += 3000.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 2.0)
            frames.append((rng.poisson(clean) + 100).astype(np.uint16))
        (tmp_path / "raw").mkdir(exist_ok=True)
        tifffile.imwrite(tmp_path / "raw" / f"{name}.tif", np.stack(frames))
    fit = GaussianFitSettings(
        camera=CameraSettings(conversion=1.0, offset=100.0, pixelsize_um=0.1),
        detection=DetectionSettings(cutoff_mode="absolute", cutoff=40.0))
    values = {k: v for k, v in settings_values(fit).items()
              if k.startswith(("camera.", "detection."))}
    spec = ChainSpec(steps=[
        Step(plugin="Localize/Gaussian 2D", label="fit", values=values),
        Step(plugin="Analysis/Measure/Localization Statistics", label="stats",
             values={"source": "all"})])
    job = write(tmp_path, Job(chain=spec, inputs=[Input(folder=str(tmp_path / "raw"))],
                              output=Output(fitted="beside_image")))
    assert batch.validate(job) == []
    report = batch.run(job)
    assert [f["status"] for f in report["files"]] == ["done", "done"], report["files"]
    assert (tmp_path / "raw" / "a_locs.hdf5").exists()           # the fit's own file
    processed = load_localizations(tmp_path / "job_output" / "a" / "a_proc.h5")
    assert 30 <= len(processed) <= 36
    assert [e["what"] for e in processed.metadata["history"]][-1] == \
        "Analysis/Chains/Unsaved chain"


def test_the_documented_examples_are_valid():
    from pathlib import Path
    examples = Path(__file__).resolve().parents[1] / "docs" / "examples"
    for path in sorted(examples.glob("*.chain.yaml")):
        _, problems = batch.validate_chain(chain.read(path))
        assert problems == [], (path.name, [str(p) for p in problems])
    job = batch.read_job(examples / "cells.batch.yaml")
    # everything but the example's own data folder, which is not on this machine
    assert [p for p in batch.validate(job) if "inputs" not in p.where] == []
