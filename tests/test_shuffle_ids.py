"""Analysis/Process/Shuffle ids: a permutation of an integer column's ids."""
import numpy as np
import pytest

from smappy import plugins
from smappy.locs import Localizations
from smappy.mathparse import recipes
from smappy.plugins import Context
from smappy.plugins.shuffle_ids import shuffle_ids

PLUGIN = "Analysis/Process/Shuffle ids"


def table(n=600):
    rng = np.random.default_rng(0)
    ids = np.repeat(np.arange(0, 30), n // 30).astype(np.int64)    # 0 is noise
    return Localizations({"x_nm": rng.uniform(0, 1e4, n), "y_nm": rng.uniform(0, 1e4, n),
                          "frame": np.arange(n), "cluster_id": ids})


def run(locs, **settings):
    plugin = plugins.get(PLUGIN)
    return plugin().run(Context(locs=locs), plugin.Settings(**settings))


def test_the_ids_are_a_permutation_and_none_stays_none():
    ids = np.array([0, 3, 3, 7, -1, 9, 7, 0])
    out = shuffle_ids(ids, seed=1)
    assert out[0] == out[7] == 0 and out[4] == -1
    assert sorted(set(out[ids > 0])) == [3, 7, 9]           # the same ids
    assert out[1] == out[2] and out[3] == out[6]           # one id, one new id
    many = np.arange(1, 1001)
    assert (shuffle_ids(many, seed=2) != many).mean() > 0.9


def test_the_plugin_writes_a_new_column_kept_for_merged_blinks():
    locs = table()
    result = run(locs, column="cluster_id", seed=5)
    out = result.locs
    assert "cluster_id_shuffled" in out and np.array_equal(out["cluster_id"], locs["cluster_id"])
    assert np.array_equal(out["cluster_id_shuffled"],
                          shuffle_ids(np.asarray(locs["cluster_id"]), 5))
    rule = next(r for r in recipes(out) if r["field"] == "cluster_id_shuffled")
    assert rule["grouped"] == "max"
    assert result.settings.seed == 5 and "29 ids" in result.text


def test_a_new_seed_each_run_and_it_is_recorded():
    a, b = run(table(), column="cluster_id"), run(table(), column="cluster_id")
    assert a.settings.seed is not None and a.settings.seed != b.settings.seed
    again = run(table(), column="cluster_id", seed=a.settings.seed)
    assert np.array_equal(again.locs["cluster_id_shuffled"], a.locs["cluster_id_shuffled"])


def test_refused_on_a_column_of_floats_or_over_the_ids_themselves():
    with pytest.raises(ValueError, match="whole numbers"):
        run(table(), column="x_nm")
    with pytest.raises(ValueError, match="new column"):
        run(table(), column="cluster_id", name="cluster_id")
    with pytest.raises(ValueError, match="integer ones are.*cluster_id"):
        run(table(), column="tid")
