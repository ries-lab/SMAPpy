"""The shipped DNA-PAINT chain (`plugins/dna_paint.chain.yaml`): drifting
DNA origami in, drift-corrected binding sites with their kinetics and a file
of molecules out."""
import numpy as np
import pytest

from smappy import plugins
from smappy.io.formats import FileInfo
from smappy.session import Session

PATH = "Analysis/DNA-PAINT/Workflow"


def drifting_origami(field_um=5.0, seed=0):
    """The *Simulate* preset "DNA-PAINT origami (20 nm grid)" in a small
    field, with a slow drift of some 100 nm added; its sites' count."""
    from smappy.locs import Localizations
    from smappy.plugins import settings_from
    from smappy.plugins.simulate import SimulateSettings, preset_values
    from smappy.simulate import ground_truth, localizations
    values = preset_values("DNA-PAINT origami (20 nm grid)")
    values.update(field_um=field_um, seed=seed)
    locs = localizations(ground_truth(settings_from(SimulateSettings, values)))
    frame = np.asarray(locs["frame"])
    t = frame / 20000.0
    columns = dict(locs.columns)
    columns["x_nm"] = np.asarray(locs["x_nm"], float) + 100 * t + 20 * np.sin(5 * t)
    columns["y_nm"] = np.asarray(locs["y_nm"], float) - 60 * t ** 2
    columns = {k: v for k, v in columns.items() if not k.startswith("z_")}
    emitters = len(np.unique(np.asarray(locs["emitter"])))
    return Localizations(columns, dict(locs.metadata)), emitters


def test_the_chain_is_shipped_and_asks_four_settings():
    found = plugins.get(PATH)
    assert set(found.Settings().__dataclass_fields__) >= {"sites", "molecules"}
    assert len(found.interface()) == 4


@pytest.mark.slow
def test_the_chain_maps_drifting_origami_to_their_sites():
    locs, sites = drifting_origami()
    session = Session()
    session.add_file(locs, FileInfo(name="origami", path="origami", format="test",
                                    n=len(locs)))
    chain = plugins.get(PATH)
    session.run(chain(), chain.Settings())
    assert len(session.files) == 2                       # the molecules beside
    molecules = session.locs[np.asarray(session.locs["filenumber"]) == 1]
    # one molecule per site, a few lost to the frame analysis or merged
    assert 0.85 * sites <= len(molecules) <= 1.05 * sites
    # sharp: a molecule's position known to well under a nanometre, which
    # the drift left uncorrected would have smeared over 100 nm
    assert np.median(np.asarray(molecules["xy_err_nm"])) < 1.0
    # each site's binding kinetics, kept per cluster
    tables = session.id_regions.get("cluster_id") or {}
    assert any("Binding kinetics" in label for label in tables)
