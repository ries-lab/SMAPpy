"""Abberior `.msr` files: the OBF container, the MINFLUX archive inside it,
blosc and LZ4, the records of both Imspector versions, the images placed in
nm, and opening it all in a session.

The files are written here, from known localizations, in the layout the
m2410 files looked at have (`io.obf`, `io.zarr_lite`): no external data is
needed.  `SMAPPY_TEST_MSR=dir` adds a check on every real `.msr` in it.
"""
import json
import os
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from smappy.io import blosc, minflux, msr, obf, zarr_lite
from smappy.io.formats import load, reader_for

# ------------------------------------------------------------------ writers

M2410 = [["vld", "|b1"], ["fnl", "|b1"], ["bot", "|b1"], ["eot", "|b1"], ["sta", "|u1"],
         ["tim", "<f8"], ["tid", "<u4"], ["gri", "<u4"], ["thi", "|u1"], ["sqi", "|u1"],
         ["itr", "<i4"], ["loc", "<f8", [3]], ["lnc", "<f8", [3]], ["eco", "<u4"],
         ["ecc", "<u4"], ["efo", "<f4"], ["efc", "<f4"], ["fbg", "<f4"], ["cfr", "<f2"],
         ["dcr", "<f2", [2]]]


def _string(b: bytes) -> bytes:
    return struct.pack("<I", len(b)) + b


def _stack(name, res, length, offset, dtype_code, data, tags, next_pos=0, compress=True):
    """One OBF stack, version 7, as bytes; `next_pos` patched in later."""
    rank = len(res)
    pad = lambda v, fmt: struct.pack(f"<15{fmt}", *(list(v) + [0] * (15 - len(v))))
    disk = zlib.compress(data) if compress else data
    head = (obf.STACK_MAGIC + struct.pack("<II", 7, rank) + pad(res, "I")
            + pad(length, "d") + pad(offset, "d")
            + struct.pack("<IIIIIQQQ", dtype_code, int(compress), 1, len(name), 0, 0,
                          len(disk), next_pos) + name.encode())
    fixed = (struct.pack("<15I", *[0] * 15) * 2 + struct.pack("<I", 0) + bytes(1280)
             + struct.pack("<QQQ", 0, 0, 0) + bytes(20) + struct.pack("<QQ", 0, 0))
    extra = bytes(60)                       # a newer version's fields: skipped
    footer = struct.pack("<I", 4 + len(fixed) + len(extra)) + fixed + extra
    after = b"".join(_string(b"x") for _ in range(rank))
    for key, value in tags.items():
        after += _string(key.encode()) + _string(value.encode())
    after += _string(b"")
    return head, disk + footer + after


def write_obf(path, stacks):
    """``stacks``: (name, res, length_m, offset_m, dtype code, bytes, tags)."""
    out = bytearray(obf.FILE_MAGIC + struct.pack("<IQ", 2, 0) + struct.pack("<I", 0)
                    + struct.pack("<Q", 0))
    struct.pack_into("<Q", out, 14, len(out))
    previous = None
    for spec in stacks:
        start = len(out)
        if previous is not None:            # the last stack's next pointer
            struct.pack_into("<Q", out, previous, start)
        head, rest = _stack(*spec)
        previous = start + len(head) - 8 - len(spec[0].encode())
        out += head + rest
    Path(path).write_bytes(bytes(out))


def archive(files: dict, t=1781190507) -> bytes:
    head = b"MFXDTA\0" + struct.pack("<II", 3, t)
    out = bytearray(head + bytes(zarr_lite.ARCHIVE_HEADER - len(head)))
    dirs = sorted({str(Path(p).parent) for p in files} - {"."})
    for d in dirs:
        out += struct.pack("<Q", len(d)) + d.encode() + b"\x01"
    for p, data in files.items():
        out += struct.pack("<Q", len(p)) + p.encode() + b"\x00" + struct.pack("<Q", len(data)) + data
    return bytes(out + struct.pack("<Q", 0))


def zarr_array(node, rec, spec, chunk=7, compressor="zlib"):
    files = {f"{node}/.zarray": json.dumps({
        "chunks": [chunk], "shape": [len(rec)], "dtype": spec, "order": "C",
        "filters": None, "fill_value": None, "zarr_format": 2,
        "compressor": {"id": "zlib", "level": 1} if compressor == "zlib" else
        {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}}).encode()}
    for k in range(0, len(rec), chunk):
        part = np.zeros(chunk, rec.dtype)          # the last chunk padded
        part[:len(rec[k:k + chunk])] = rec[k:k + chunk]
        raw = part.tobytes()
        files[f"{node}/{k // chunk}"] = (zlib.compress(raw) if compressor == "zlib" else
                                         _blosc_memcpy(raw))
    return files


def _blosc_memcpy(raw: bytes) -> bytes:
    return struct.pack("<BBBBIII", 2, 1, blosc.BLOSC_MEMCPYED, 1, len(raw), len(raw),
                       len(raw) + 16) + raw


TRUTH = {  # per trace: position (nm), cfr, how many localizations
    5: ((1200.0, 2100.0), 0.25, 4),
    6: ((1500.0, 2300.0), 0.50, 3),
    9: ((1800.0, 2050.0), 0.75, 5),
}


def m2410_records(t0=0.0, invalid=True):
    """2D, four iterations: 0-2 once at a trace's start (cfr in 2), then the
    final iteration 3 once per localization -- as the 2D files have it."""
    rows = []
    t = t0
    for tid, ((x, y), cfr, n) in TRUTH.items():
        for itr in (0, 1, 2):
            t += 1e-3
            rows.append(dict(vld=True, fnl=False, itr=itr, tid=tid, tim=t, sta=2,
                             cfr=cfr if itr == 2 else 0.0, loc=(np.nan,) * 3, eco=10))
        for k in range(n):
            t += 1e-2
            rows.append(dict(vld=True, fnl=True, itr=3, tid=tid, tim=t, sta=1, cfr=0.0,
                             loc=(x * 1e-9, y * 1e-9, 0.0), eco=100 + k, dcr=(0.8, 0.2)))
    if invalid:
        t += 1e-2
        rows.append(dict(vld=False, fnl=True, itr=3, tid=9, tim=t, sta=11, cfr=0.0,
                         loc=(9e-6, 9e-6, 0.0), eco=3))
    rec = np.zeros(len(rows), zarr_lite._dtype(M2410))
    for i, r in enumerate(rows):
        for k, v in r.items():
            rec[k][i] = v
        rec["lnc"][i] = rec["loc"][i]
    # a 0 cfr where it is not measured, and a value that varies where it is:
    # the iteration is found by that
    rec["cfr"][(rec["itr"] == 2)] += np.arange((rec["itr"] == 2).sum()) * 0
    return rec


def measurement_stack(label, rec, date="2026-06-11T15:08:27+02:00", beads=False):
    files = {"zarr/.zattrs": json.dumps({"rois": [{"corners": [[1e-6, 2e-6], [2e-6, 3e-6]],
                                                   "type": "ROI", "name": ""}]}).encode()}
    files.update(zarr_array("zarr/mfx", rec, M2410))
    attrs = {"acquisition_date": date, "version": "2.1",
             "measurement": {"dimensionality": 2, "threads": [{"sequences": [{
                 "Itr": [{"Mode": {"pattern": "hexagon", "dim": 2}, "phtLimit": 100 + i,
                          "_excitation": {"laser": "MINFLUX640", "power": 0.8},
                          "estCoeff": [[1, 2, 3, 4, 5]]} for i in range(4)],
                 "locLimit": 200}], "lasers": [{"id": "MINFLUX640"}]}]}}
    files["zarr/mfx/.zattrs"] = json.dumps(attrs).encode()
    if beads:
        dt = [["gri", "<u4"], ["xyz", "<f8", [3]], ["tim", "<f4"], ["str", "<f4"]]
        pts = np.zeros(3, zarr_lite._dtype(dt))
        pts["gri"] = [61, 123, 61]
        pts["xyz"] = [[1e-6, 2e-6, 3e-8], [4e-6, 5e-6, 6e-8], [1.001e-6, 2e-6, 3e-8]]
        pts["tim"] = [1.0, 1.5, 2.0]
        files.update(zarr_array("zarr/grd/mbm/points", pts, dt, chunk=10, compressor="blosc"))
        files["zarr/mbm/.zattrs"] = json.dumps({"used": ["R0", "R1"]}).encode()
    blob = archive(files)
    side = int(np.ceil(np.sqrt(len(blob))))
    tag = json.dumps({"type": "data", "label": label, "did": f"did-{label}"})
    return ("", (side, side, 1, 1), (1.0, 1.0, 1.0, 1.0), (0.0,) * 4, 0x01, blob,
            {"minflux": tag, "imspector": "<root/>"})


def image_stack(name="Ch1 {0}", nx=6, ny=4, px=40e-9, x0=1000e-9, y0=2000e-9):
    data = np.arange(ny * nx, dtype="<i2").reshape(ny, nx)
    return (name, (nx, ny, 1, 1), (nx * px, ny * px, 1.0, 1.0), (x0, y0, -0.5, 0.0),
            0x08, data.tobytes(), {"imspector": "<root/>"})


def render_stack(label):
    data = np.ones((3, 5, 4), "<u2")
    return (f"MF({label})/density/loc", (4, 5, 3, 1), (4e-8, 5e-8, 3e-8, 1.0),
            (1e-6, 2e-6, -1e-8, 0.0), 0x04, data.tobytes(),
            {"minflux": json.dumps({"type": "density", "source": label})})


@pytest.fixture
def two(tmp_path):
    """An image, two measurements a minute apart, a render."""
    path = tmp_path / "cell.msr"
    write_obf(path, [image_stack(),
                     measurement_stack("first", m2410_records(), beads=True),
                     render_stack("first"),
                     measurement_stack("second", m2410_records(t0=5.0),
                                       date="2026-06-11T15:09:27+02:00")])
    return path


# -------------------------------------------------------------------- tests

def test_lz4_reads_literals_and_overlapping_matches():
    data = b"abc" * 40 + bytes(range(20))
    # literals 'abc' (3), then a match at offset 3 of 117 bytes (overlapping:
    # a run), then the last 20 literals; lengths past 15 extended by bytes
    block = (bytes([0x3F]) + b"abc" + struct.pack("<H", 3) + bytes([117 - 4 - 15])
             + bytes([0xF0, 20 - 15]) + bytes(range(20)))
    assert blosc.lz4_block(block, len(data)) == data
    with pytest.raises(ValueError):
        blosc.lz4_block(block, len(data) + 1)


def test_blosc_undoes_the_split_streams_and_the_shuffle():
    items = np.arange(128, dtype="<u4") * 1000 + 7       # typesize 4, 512 bytes
    shuffled = np.frombuffer(items.tobytes(), np.uint8).reshape(-1, 4).T  # byte j of each
    streams = b""
    for j in range(4):
        s = shuffled[j].tobytes()
        if j == 1:                                    # one stream as an LZ4 block
            lz = bytes([0xF0, 128 - 15]) + s
            streams += struct.pack("<i", len(lz)) + lz
        else:                                         # the others as they are
            streams += struct.pack("<i", 128) + s
    flags = blosc.BLOSC_SHUFFLE | (blosc.LZ4 << 5)
    frame = struct.pack("<BBBBIII", 2, 1, flags, 4, 512, 512, 0) + struct.pack("<I", 20) + streams
    assert np.array_equal(np.frombuffer(blosc._decode_python(frame), "<u4"), items)


def test_contents_lists_measurements_images_and_renders_without_decoding(two):
    found = msr.contents(two)
    assert [m.label for m in found.measurements] == ["first", "second"]
    first = found.measurements[0]
    assert first.layout == "m2410" and first.dimensionality == 2
    assert first.rows == len(m2410_records()) and first.n_beads == 2
    assert first.date.startswith("2026-06-11T15:08:27")
    assert [im.name for im in found.images if not im.render] == ["Ch1 {0}"]
    assert [im.shape for im in found.renders] == [(3, 5, 4)]
    assert reader_for(two).name == "Abberior MSR"


def test_a_measurement_reads_as_its_final_valid_iterations_with_their_own_cfr(two):
    locs, info = load(two, measurements=[1])
    n = sum(t[2] for t in TRUTH.values())
    assert len(locs) == n and "z_nm" not in locs        # 2D; the invalid one dropped
    for tid, ((x, y), cfr, k) in TRUTH.items():
        rows = locs["tid"] == tid
        assert rows.sum() == k
        assert np.allclose(locs["x_nm"][rows], x) and np.allclose(locs["y_nm"][rows], y)
        # measured in iteration 2, once at the trace's start: the first
        # localization has it, the others were not measured and are NaN
        assert np.isclose(locs["cfr"][rows][0], cfr, atol=1e-3)
        assert np.isnan(locs["cfr"][rows][1:]).all()
    assert np.all(np.diff(locs["time_s"]) > 0) and np.array_equal(locs["frame"], np.arange(n))
    assert np.allclose(locs["dcr"], 0.8, atol=1e-3) and np.allclose(locs["dcr2"], 0.2, atol=1e-3)
    assert set(np.unique(locs["sta"])) == {1} and set(np.unique(locs["iteration"])) == {3}
    meta = info.metadata["minflux"]["measurements"][0]
    assert meta["label"] == "first" and meta["localizations"] == n
    assert meta["attempts"] == n + 1
    assert [r["phtLimit"] for r in meta["sequence"]["iterations"]] == [100, 101, 102, 103]
    assert "estCoeff" not in meta["sequence"]["iterations"][0]      # long tables left out
    assert meta["roi"][0]["corners_nm"] == [[1000.0, 2000.0], [2000.0, 3000.0]]
    beads = meta["mbm"]
    assert beads["used"] == ["R0", "R1"] and beads["points"]["gri"] == [61, 123, 61]
    assert np.allclose(beads["points"]["x_nm"], [1000, 4000, 1001])


def test_a_cfr_measured_before_every_localization_is_each_ones_own():
    rows = []
    for tid, cfrs in ((1, [0.2, 0.4, 0.6]), (2, [0.8])):
        for k, c in enumerate(cfrs):
            rows += [dict(itr=6, fnl=False, cfr=c, tid=tid),
                     dict(itr=7, fnl=True, cfr=0.0, tid=tid, loc=(1e-6, 1e-6, 1e-8))]
    rec = np.zeros(len(rows), zarr_lite._dtype(M2410))
    rec["vld"], rec["loc"] = True, np.nan
    for i, r in enumerate(rows):
        for key, v in r.items():
            rec[key][i] = v
        rec["tim"][i] = i
    locs, _ = minflux.table(rec, dimensionality=3)
    assert np.allclose(locs["cfr"], [0.2, 0.4, 0.6, 0.8], atol=1e-3)


def test_keep_invalid_keeps_every_final_iteration_and_says_which(two):
    locs, _ = load(two, measurements="first", valid_only=False)
    assert len(locs) == sum(t[2] for t in TRUTH.values()) + 1
    assert set(np.unique(locs["vld"])) == {0, 1}
    assert locs["sta"][locs["vld"] == 0].tolist() == [11]


def test_measurements_merge_one_after_another_in_time_with_their_own_traces(two):
    locs, info = load(two)                                 # all, merged
    n = sum(t[2] for t in TRUTH.values())
    assert len(locs) == 2 * n and info.name == "cell.msr"
    first, second = (locs["measurement"] == 0), (locs["measurement"] == 1)
    assert first.sum() == second.sum() == n
    # the second started 60 s after the first: its clock is moved by that
    one = load(two, measurements=[2])[0]["time_s"]
    assert np.allclose(np.sort(locs["time_s"][second]), np.sort(one) + 60)
    assert not set(locs["tid"][first]) & set(locs["tid"][second])
    assert np.array_equal(locs["frame"], np.arange(2 * n))
    offsets = [m["time_offset_s"] for m in info.metadata["minflux"]["measurements"]]
    assert offsets == [0.0, 60.0]
    files = msr.read_files(two)                            # a file each
    assert [i.name for _, i in files] == ["cell: first", "cell: second"]


def test_a_measurement_is_chosen_by_number_label_did_or_pattern(two):
    found = msr.contents(two)
    pick = lambda w: [m.label for m in msr.select(found, w)]
    assert pick(None) == pick("") == ["first", "second"]
    assert pick([2]) == pick("second") == pick("did-second") == pick("sec.*") == ["second"]
    assert pick("2, first") == ["second", "first"]
    with pytest.raises(ValueError, match="first, second"):
        msr.select(found, "third")


def test_an_image_is_placed_in_nm_with_its_first_row_at_its_offset(two):
    found = msr.contents(two)
    image = msr.image(two, found.images[0])
    assert image.data.shape == (4, 6) and image.data[0, 1] == 1 and image.data[1, 0] == 6
    assert np.isclose(image.pixelsize, 40) and image.pixelsize_y is None
    assert np.isclose(image.x0, 1000) and np.isclose(image.y0, 2000)
    render = msr.image(two, found.renders[0])
    assert render.data.shape == (3, 5, 4) and np.isclose(render.metadata["z0_nm"], -10)


def test_an_obf_file_is_listed_from_its_headers_and_damage_is_refused(two, tmp_path):
    stacks = obf.stacks(two)
    assert [s.dtype_code for s in stacks] == [0x08, 0x01, 0x04, 0x01]
    assert stacks[0].shape == (4, 6) and stacks[0].tags["imspector"] == b"<root/>"
    bad = tmp_path / "bad.msr"
    bad.write_bytes(b"not an obf file at all")
    with pytest.raises(ValueError, match="not an OBF"):
        obf.stacks(bad)


def test_an_m2410_npy_export_reads_as_the_msr_does(tmp_path):
    rec = m2410_records()
    np.save(tmp_path / "export.npy", rec)
    locs, info = load(tmp_path / "export.npy")
    assert info.format == "MINFLUX" and len(locs) == sum(t[2] for t in TRUTH.values())
    assert np.isclose(locs["cfr"][locs["tid"] == 6][0], 0.5, atol=1e-3)


def test_m2205_takes_cfr_from_the_last_iteration_where_it_varies():
    n, k = 20, 4
    sub = np.zeros((n, k), dtype=[("loc", "f8", (3,)), ("eco", "f8"), ("cfr", "f8"), ("efo", "f8")])
    sub["loc"][:, -1, :2] = 1e-6
    sub["cfr"][:, 2] = np.linspace(0.1, 0.9, n)          # measured in iteration 2 only
    rec = np.zeros(n, dtype=[("itr", sub.dtype, (k,)), ("tim", "f8"), ("tid", "i8"), ("vld", "?")])
    rec["itr"], rec["tim"], rec["vld"] = sub, np.arange(n), True
    locs, _ = minflux.table(rec)
    assert np.allclose(locs["cfr"], np.linspace(0.1, 0.9, n))


def test_the_loader_plugin_opens_files_and_images_kept_through_a_save(two, tmp_path):
    from smappy.metadata import file_metadata
    from smappy.plugins import Context
    from smappy.plugins.file import LoadAuto, LoadSettings, MsrSettings
    from smappy.session import Session
    session = Session()
    plugin = LoadAuto()
    settings = LoadSettings(path=str(two), msr=MsrSettings())
    assert plugin.describe(None, settings).shown == {"csv": False, "msr": True}
    result = plugin.run(Context(session=session), settings)
    session.apply(plugin, result)
    assert [f.name for f in session.files] == ["cell: first", "cell: second"]
    # the image is a source for an image layer, not a layer of its own
    assert not any(l.is_image for l in session.layers)
    assert [im.name for im in session.images] == ["cell: Ch1 {0}"]        # not the render
    label = lambda m, n: file_metadata(m, n)["minflux"]["measurements"][0]["label"]
    assert label(session.locs.metadata, 1) == "second"
    saved = tmp_path / "cell.h5"
    session.save(saved)
    again = Session()
    again.load(saved)
    assert [f.name for f in again.files] == ["cell: first", "cell: second"]
    assert label(again.locs.metadata, 0) == "first" and label(again.locs.metadata, 1) == "second"
    assert again.images == []          # an image no layer showed is not saved


@pytest.mark.skipif(not os.environ.get("SMAPPY_TEST_MSR"),
                    reason="SMAPPY_TEST_MSR=dir of real .msr files")
def test_real_msr_files_list_and_read_every_measurement():
    paths = sorted(Path(os.environ["SMAPPY_TEST_MSR"]).glob("*.msr"))
    assert paths
    for path in paths:
        found = msr.contents(path)
        for m in found.measurements:
            locs, meta = msr.read_measurement(path, m)
            assert len(locs) and np.isfinite(locs["x_nm"]).all(), (path.name, m.label)
            assert np.isfinite(locs["cfr"]).any(), (path.name, m.label)
        for im in found.images:
            assert msr.image(path, im).data.size


def test_the_open_dialog_queues_what_is_ticked_and_opens_its_images(two, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication, QDialog
    from smappy.gui.app import ControlWindow, RenderWindow
    from smappy.gui.msr_import import MsrImportDialog
    from smappy.session import Session
    app = QApplication.instance() or QApplication([])
    dialog = MsrImportDialog(two)
    assert dialog.merge.isEnabled() and dialog.z_scale.isHidden()         # 2D only
    choice = dialog.choice()
    assert [a["measurements"] for a in choice["loads"]] == [[1], [2]]
    assert [im.name for im in choice["images"]] == ["Ch1 {0}"]            # not the render
    dialog.merge.setChecked(True)
    assert [a["measurements"] for a in dialog.choice()["loads"]] == [[1, 2]]
    group = dialog.tree.topLevelItem(0)
    group.child(1).setCheckState(0, Qt.Unchecked)
    assert not dialog.merge.isEnabled()
    assert [a["measurements"] for a in dialog.choice()["loads"]] == [[1]]

    monkeypatch.setattr(MsrImportDialog, "exec", lambda self: QDialog.Accepted)
    session = Session()
    control = ControlWindow(session, RenderWindow(session))
    control.load_paths([str(two)])
    QTimer.singleShot(20_000, app.quit)
    done = QTimer(control)
    done.timeout.connect(lambda: app.quit() if control._task is None
                         and not control._queue else None)
    done.start(10)
    app.exec()
    assert [f.name for f in session.files] == ["cell: first", "cell: second"]
    assert not any(l.is_image for l in session.layers)
    assert [im.name for im in session.images] == ["cell: Ch1 {0}"]
