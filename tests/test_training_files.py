"""A training's model file and progress file (`smappy.training.model_file`, `.progress`): torch-free."""
import json
import subprocess
import sys

import numpy as np
import pytest

from smappy.training import model_file, progress


def _tensors():
    rng = np.random.default_rng(0)
    return {"conv.weight": rng.normal(size=(8, 3, 3, 3)).astype(np.float32), "conv.bias": np.zeros(8, np.float32),
            "steps": np.array(12, np.int64), "flags": np.array([True, False])}


def test_a_model_file_written_and_read_gives_back_its_tensors_and_metadata(tmp_path):
    meta = {"method": "decode-v0.10.2", "epoch": 3, "scaling": {"phot_max": 15000.0}, "frames_mirrored": False}
    path = model_file.write(tmp_path / "model.safetensors", _tensors(), meta)
    tensors, back = model_file.read(path)
    assert back == meta and model_file.read_metadata(path) == meta
    for name, a in _tensors().items():
        assert tensors[name].dtype == a.dtype and np.array_equal(tensors[name], a)


def test_the_header_is_safetensors_own_and_its_package_reads_the_file(tmp_path):
    path = model_file.write(tmp_path / "model.safetensors", _tensors(), {"method": "x"})
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    assert (8 + n) % 8 == 0 and header["conv.weight"]["dtype"] == "F32" and header["conv.weight"]["shape"] == [8, 3, 3, 3]
    st = pytest.importorskip("safetensors.numpy")
    loaded = st.load_file(str(path))
    assert np.array_equal(loaded["conv.weight"], _tensors()["conv.weight"])


def test_reading_a_model_files_metadata_loads_no_torch(tmp_path):
    path = model_file.write(tmp_path / "m.safetensors", _tensors(), {"method": "x"})
    code = (f"import sys; from smappy.training import model_file; model_file.read_metadata({str(path)!r}); "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_progress_lines_are_appended_with_time_and_epoch_and_read_back(tmp_path):
    progress.append(tmp_path, "start", 0, device="cpu")
    progress.append(tmp_path, "epoch", 0, loss_test=float("nan"), metrics={"jaccard": 0.5, "rmse_lat_nm": float("inf")})
    lines = progress.read(tmp_path)
    assert [line["kind"] for line in lines] == ["start", "epoch"]
    assert lines[1]["loss_test"] is None and lines[1]["metrics"] == {"jaccard": 0.5, "rmse_lat_nm": None}
    assert all("time" in line and "epoch" in line for line in lines)
    with pytest.raises(ValueError):
        progress.append(tmp_path, "nonsense", 0)


def test_a_line_cut_short_is_skipped_and_the_state_is_the_last_word(tmp_path):
    progress.append(tmp_path, "start", 0)
    progress.append(tmp_path, "epoch", 0, loss_test=1.0)
    with open(tmp_path / progress.NAME, "a") as f:
        f.write('{"kind": "epoch", "epo')                       # a crash in mid-line
    lines = progress.read(tmp_path)
    assert len(lines) == 2 and progress.state(lines) == "running"
    progress.append(tmp_path, "stopped", 0)
    assert progress.state(progress.read(tmp_path)) == "stopped"
    assert progress.state([]) == "new"
