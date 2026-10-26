"""SMAP's ROI manager sites in an `_sml.mat` come in as the ROI manager's ROIs.

SMAP keeps them as MATLAB objects, so the test writes the object store the
way MATLAB's ``save -v7.3`` lays it out (`io.mcos`) and reads it back.  A
real file is read when ``SMAPPY_TEST_SML`` names one with sites.
"""
import os

import h5py
import numpy as np
import pytest

from smappy.io.formats import load, sml_sites_project
from smappy.session import Session

MAGIC = 0xDD000000


class _Writer:
    """The parts of a v7.3 file MATLAB writes for an array of objects of one
    class: values under ``#refs#``, the store at ``#subsystem#/MCOS``."""

    def __init__(self, f):
        self.f = f
        self.refs = f.create_group("#refs#")
        self.n = 0

    def _name(self):
        self.n += 1
        return f"v{self.n}"

    def value(self, v, where=None, name=None):
        """A MATLAB value: a dict a struct, a str a char array, a number a
        double array -- stored transposed, as MATLAB does."""
        where = self.refs if where is None else where
        name = name or self._name()
        if isinstance(v, dict):
            g = where.create_group(name)
            g.attrs["MATLAB_class"] = np.bytes_("struct")
            for k, x in v.items():
                self.value(x, g, k)
            return g
        if isinstance(v, str):
            d = where.create_dataset(name, data=np.array([[ord(c)] for c in v] or [[0]],
                                                         np.uint16))
            d.attrs["MATLAB_class"] = np.bytes_("char")
            return d
        a = np.atleast_2d(np.asarray(v, float))
        d = where.create_dataset(name, data=a.T)
        d.attrs["MATLAB_class"] = np.bytes_("double")
        return d

    def objects(self, path, class_name, objects):
        """``objects``, each a dict of properties, as one array at ``path``."""
        names = ["interfaces", class_name]
        for o in objects:
            names += [k for k in o if k not in names]
        index = {n: i + 1 for i, n in enumerate(names)}
        cells = [None, None]                       # the metadata, then an empty cell
        props = [0, 0]
        for o in objects:
            entry = [len(o)]
            for k, v in o.items():
                cells.append(self.value(v))
                entry += [index[k], 1, len(cells) - 3]
            if len(entry) % 2:
                entry.append(0)
            props += entry
        classes = [0, 0, 0, 0, index["interfaces"], index[class_name], 0, 0]
        table = [0] * 6
        for i in range(len(objects)):
            table += [1, 0, 0, 0, i + 1, i + 1]
        text = b"".join(n.encode() + b"\0" for n in names)
        text += b"\0" * (-(40 + len(text)) % 8)
        regions = [np.array(r, "<u4").tobytes()
                   for r in (classes, [0, 0], table, props, [0, 0], [], [], [])]
        offsets, at = [], 40 + len(text)
        for r in regions:
            at += len(r)
            offsets.append(at)
        offsets = [40 + len(text)] + offsets[:-1]
        blob = (np.array([4, len(names)] + offsets, "<u4").tobytes() + text
                + b"".join(regions))
        meta = self.refs.create_dataset("meta", data=np.frombuffer(blob, np.uint8)[None])
        meta.attrs["MATLAB_class"] = np.bytes_("uint8")
        empty = self.refs.create_dataset("empty", data=np.array([0, 0], np.uint64))
        empty.attrs["MATLAB_empty"] = np.uint8(1)
        cells[0], cells[1] = meta, empty
        sub = self.f.create_group("#subsystem#")
        store = sub.create_dataset("MCOS", (1, len(cells)), dtype=h5py.ref_dtype)
        for i, c in enumerate(cells):
            store[0, i] = c.ref
        store.attrs["MATLAB_class"] = np.bytes_("FileWrapper__")
        store.attrs["MATLAB_object_decode"] = np.int32(3)
        ref = self.f.create_dataset(path, data=np.array(
            [[MAGIC, 2, 1, len(objects), *range(1, len(objects) + 1), 1]], np.uint32))
        ref.attrs["MATLAB_class"] = np.bytes_("interfaces." + class_name)
        ref.attrs["MATLAB_object_decode"] = np.int32(3)


def _line(p0=(0, 0), p1=(0, 0)):
    """A SMAP line annotation, in µm; zeros is a line never drawn."""
    return {"pos": [list(p0), list(p1)], "angle": 0.0, "length": 0.0, "value": 0.0}


def _site(ident, pos_nm, **annotation):
    note = {"line1": _line(), "line2": _line(), "rotationpos": _line(),
            "list1": {"string": "empty", "value": 1.0}, "use": 1.0}
    note.update(annotation)
    return {"pos": [*pos_nm, 0.0], "ID": float(ident),
            "info": {"filenumber": 1.0, "cell": 0.0},
            "annotation": note, "name": f"{ident}.S{ident}C0F1"}


def write_sml(path, sites, n=200):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        loc = f.create_group("saveloc/loc")
        loc["xnm"] = rng.uniform(0, 5000, (1, n)).astype(np.float32)
        loc["ynm"] = rng.uniform(0, 5000, (1, n)).astype(np.float32)
        loc["frame"] = np.arange(1, n + 1, dtype=np.float64)[None]
        f["saveloc"].create_group("siteexplorer")
        _Writer(f).objects("saveloc/siteexplorer/sites", "SEsites", sites)


def test_smap_sites_come_in_as_rois_with_their_lines_keywords_and_use(tmp_path):
    path = tmp_path / "cells_sml.mat"
    write_sml(path, [
        _site(3, (1000.0, 2000.0),
              rotationpos=_line((0.9, 2.0), (1.1, 2.0)),
              line1=_line((1.0, 1.9), (1.0, 2.15)),
              list1={"string": "good", "value": 1.0}),
        _site(7, (3000.0, 1500.0), use=0.0),
    ])
    locs, _ = load(path)
    project = Session(locs).rois
    rois = sorted(project.rois.values(), key=lambda r: r.number)
    assert [r.number for r in rois] == [3, 7]
    first, second = rois
    assert first.center == [1000.0, 2000.0] and first.use
    assert np.allclose(first.lines["direction"], [[900, 2000], [1100, 2000]])
    assert np.allclose(first.lines["line1"], [[1000, 1900], [1000, 2150]])
    assert "line2" not in first.lines                 # never drawn: all zeros
    assert first.annotations == {"list1": "good"}
    assert not second.use and second.lines == {} and second.annotations == {}
    assert project.line_names == ["direction", "line1"]
    assert project.next_number == 8                   # a new ROI does not take 3 or 7


def test_a_site_without_objects_leaves_the_file_without_rois(tmp_path):
    path = tmp_path / "plain_sml.mat"
    with h5py.File(path, "w") as f:
        f["saveloc/loc/xnm"] = np.zeros((1, 5), np.float32)
        f["saveloc/loc/ynm"] = np.zeros((1, 5), np.float32)
    locs, _ = load(path)
    assert "roi_project" not in locs.metadata


def test_the_shape_drawn_in_a_site_is_the_rois_outline():
    def site(shape, function="drawpolygon"):
        line3 = {"pos": np.asarray(shape, float),
                 "roifun": [None, {"function_handle": {"function": function}}]}
        return {"pos": np.array([[1000.0, 1000.0, 0.0]]), "ID": np.array([[1.0]]),
                "annotation": {"line3": line3}}

    polygon = [[0.9, 0.9], [1.1, 0.9], [1.0, 1.1]]
    (roi,) = sml_sites_project([site(polygon)])["rois"]
    assert np.allclose(roi["polygon"], np.array(polygon) * 1000)
    (roi,) = sml_sites_project([site([[0.9, 0.8, 0.2, 0.4]])])["rois"]   # x y w h
    assert np.allclose(roi["polygon"], [[900, 800], [1100, 800], [1100, 1200], [900, 1200]])
    (roi,) = sml_sites_project([site([[1.0, 1.0, 0.2, 0.1, 90.0]])])["rois"]
    xy = np.array(roi["polygon"])                    # turned a quarter: tall, not wide
    assert len(xy) == 64
    assert np.allclose(np.ptp(xy, axis=0), [200, 400], atol=1)
    (roi,) = sml_sites_project([site(polygon, "drawpolyline")])["rois"]
    assert roi["polygon"] is None                    # a polyline is not a shape


@pytest.mark.skipif(not os.environ.get("SMAPPY_TEST_SML"),
                    reason="SMAPPY_TEST_SML names a SMAP file with sites")
def test_a_real_smap_files_sites_are_read():
    locs, _ = load(os.environ["SMAPPY_TEST_SML"])
    project = Session(locs).rois
    assert project.rois
    for roi in project.rois.values():
        assert roi.origin["method"] == "SMAP"
        assert len(project.indices(roi))           # the site is where the data are
