"""A simulated MINFLUX measurement, written as an `.msr` (`simulate.minflux`,
`io.msr_write`): it opens as a measurement from the instrument does, and
the motors' steps, the dwells and the second molecules come back."""
import numpy as np

from smappy.io import msr
from smappy.simulate.minflux import motor_msr, simulate_attempts
from smappy.steps import autostepfinder, split_merge


def test_a_simulated_msr_opens_with_its_measurements_and_their_sequence(tmp_path):
    path = motor_msr(tmp_path / "kinesin.msr", n_events=600)
    found = msr.contents(path)
    assert [m.label for m in found.measurements] == ["kinesin 1000uM ATP",
                                                     "kinesin 20uM ATP"]
    locs, _ = msr.load(path, measurements=[1], every=True)
    meta = locs.metadata["minflux"]["measurements"][0]
    rows = meta["sequence"]["iterations"]
    assert [r["patGeoFactor"] for r in rows] == [0.79, 0.84, 0.42, 0.28]
    assert rows[2]["ccrLimit"] == 0.8 and meta["sequence"]["excitation_nm"] == 640.0
    assert set(np.unique(np.asarray(locs["final"]))) == {0, 1}


def test_a_simulated_motor_steps_by_its_step_size():
    rec = simulate_attempts(400, passed=(0.2, 0.95, 0.9), locs_per_burst=1500.0,
                            precision_nm=2.0, dwell_s=5e-4, step_nm=8.0,
                            step_dwell_s=0.03, seed=1)
    final = rec["fnl"] & rec["vld"]
    tid = rec["tid"][final]
    longest = np.bincount(tid).argmax()
    xy = rec["loc"][final][tid == longest][:, :2] * 1e9
    along = (xy - xy[0]) @ np.linalg.svd(xy - xy.mean(0))[2][0]
    along *= np.sign(along[-1])
    boundaries = split_merge(along, autostepfinder(along).boundaries, 8.0)
    levels = [along[a:b].mean() for a, b in zip(boundaries[:-1], boundaries[1:])]
    assert abs(np.median(np.diff(levels)) - 8.0) < 1.0


def test_two_molecules_at_once_double_the_photon_rate_and_raise_cfr():
    rec = simulate_attempts(3000, passed=(0.3, 0.9, 0.9), two_molecules=0.3,
                            two_cfr=0.6, cfr_limit=0.8, seed=2)
    final = rec["fnl"] & rec["vld"]
    cfr = rec["cfr"][(rec["itr"] == 2) & rec["vld"]].astype(float)
    two = cfr > 0.45
    assert 0.2 < two.mean() < 0.4
    efo_of = {t: e for t, e in zip(rec["tid"][final], rec["efo"][final])}
    tids = rec["tid"][(rec["itr"] == 2) & rec["vld"]]
    one = np.median([efo_of[t] for t in tids[~two] if t in efo_of])
    both = np.median([efo_of[t] for t in tids[two] if t in efo_of])
    assert 1.7 < both / one < 2.3
    assert not (rec["cfr"][rec["vld"]] > 0.8).any()       # the sequence's limit aborts


def test_a_tracking_measurement_opens_as_its_tracks_and_an_imaging_one_as_a_picture(tmp_path):
    """Told by the sequence's name (`minflux.is_tracking`): Abberior's
    ``TDs_Trk_2D_...`` tracks, ``imaging_sequence...`` images, an MBM
    sequence neither."""
    import pytest

    from smappy.io.minflux import is_tracking
    from smappy.session import Session
    from smappy.simulate.minflux import sequence_attrs, write_msr

    def meta(*names):
        return {"minflux": {"measurements": [
            {"sequence": {"globals": {"id": n}}} for n in names]}}
    assert is_tracking(meta("TDs_Trk_2D_L75_pho010_dt900us"))
    assert is_tracking(meta("seqMBM2d_v8", "Tracking_2D"))
    assert not is_tracking(meta("imaging_sequence_toolkit_test", "seqMBM2d_v8"))
    assert not is_tracking({}) and not is_tracking(None)

    rec = simulate_attempts(200, passed=(0.2, 0.95, 0.9), locs_per_burst=50.0,
                            precision_nm=3.0, dwell_s=5e-4, seed=2)
    date = "2026-06-11T15:08:27+02:00"
    shown = {}
    for name in ("Tracking_2D", "imaging_sequence_2d"):
        path = tmp_path / f"{name}.msr"
        write_msr(path, [("one", rec, sequence_attrs(6.0, date, name=name))])
        session = Session()
        session.load(path)
        layer = session.layers[0]
        shown[name] = (layer.state.settings.mode, layer.grouped, layer.colour_tracks)
    assert shown["Tracking_2D"] == ("tracks", False, True)
    assert shown["imaging_sequence_2d"][0] != "tracks"
    assert not shown["imaging_sequence_2d"][2]

    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui.render_tab import RenderTab
    session = Session()
    session.load(tmp_path / "Tracking_2D.msr")
    tab = RenderTab(session)                 # shows the layer: coloured by track, once
    settings = session.layers[0].state.settings
    assert settings.mode == "tracks" and settings.color_field == "tid"
    assert not session.layers[0].colour_tracks
    tab.deleteLater()
