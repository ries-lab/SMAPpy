"""*Sequence table*: the sequences of the open measurements, and what
differs between them."""
from smappy.plugins.sequence_table import differing, measured_sequences, rows_of
from smappy.io.formats import FileInfo


def seq(power, geo=0.28, lemd=1e-8):
    return {"globals": {"id": "imaging"},
            "iterations": [{"Mode.pattern": "hexagon", "patGeoFactor": 0.42,
                            "_excitation.power": power, "_lemd": lemd},
                           {"Mode.pattern": "hexagon", "patGeoFactor": geo,
                            "_excitation.power": power}]}


def files():
    meta = lambda label, s: {"minflux": {"measurements": [{"label": label, "sequence": s}]}}
    return [FileInfo("a.msr", "/a", "Abberior MSR", metadata=meta("m1", seq(2.0))),
            FileInfo("b.msr", "/b", "Abberior MSR",
                     metadata=meta("m2", seq(4.0, lemd=2e-8)))]


def test_each_measurement_and_thread_is_an_entry_and_an_iteration_a_row():
    entries = measured_sequences(files())
    assert [e["file"] for e in entries] == ["a.msr", "b.msr"]
    rows = rows_of(entries, ["patGeoFactor"])
    assert len(rows) == 4 and rows[1]["patGeoFactor"] == 0.28


def test_only_what_differs_is_listed_and_not_the_estimators_internals():
    diff = differing(measured_sequences(files()))
    assert set(diff) == {"itr 0 _excitation.power", "itr 1 _excitation.power"}
    assert diff["itr 0 _excitation.power"] == [2.0, 4.0]
