"""A chain's interface: the few settings it asks, the steps folded away."""
from dataclasses import replace

from smappy import chain
from smappy.chain import ChainSpec, Step, chain_class


def stats_chain(interface=None):
    return ChainSpec(steps=[Step(plugin="Analysis/Drift/RCC"),
                            Step(plugin="Analysis/Measure/Localization Statistics")],
                     interface=dict(interface or {}))


def test_the_interface_is_written_with_the_chain_and_read_back(tmp_path):
    spec = stats_chain({"rcc.n_timepoints": "drift windows", "rcc.max_drift_nm": ""})
    path = chain.write(spec, tmp_path / "simple.chain.yaml")
    assert "interface:" in path.read_text()
    assert chain.read(path).interface == spec.interface
    # names alone take the fields' own labels
    listed = ChainSpec.from_dict({"steps": [{"plugin": "Analysis/Drift/RCC"}],
                                  "interface": ["rcc.n_timepoints"]})
    assert chain_class(listed).interface() == {"rcc.n_timepoints": "time windows"}


def test_the_form_asks_only_the_interface_until_the_steps_are_shown():
    cls = chain_class(stats_chain({"rcc.n_timepoints": "drift windows"}))
    state = cls().describe(None, cls.Settings())
    assert state.keys == {"rcc.n_timepoints": "drift windows"}
    assert state.shown == {"rcc": False, "localization_statistics": False}
    shown = cls().describe(None, replace(cls.Settings(), show_steps=True)).shown
    assert shown == {"rcc": True, "localization_statistics": True}


def test_a_chain_without_an_interface_is_as_it_was():
    cls = chain_class(stats_chain())
    assert not hasattr(cls.Settings(), chain.STEPS_FIELD)
    state = cls().describe(None, cls.Settings())
    assert not state.keys and "rcc" not in state.shown


def test_a_name_the_chain_no_longer_has_is_left_out_and_a_rename_is_followed():
    cls = chain_class(stats_chain({"rcc.n_timepoints": "", "rcc.gone": "x", "rcc": "a part",
                                   "removed_step.bins": ""}))
    assert list(cls.interface()) == ["rcc.n_timepoints"]
    before = ["rcc", "localization_statistics"]
    renamed = chain.renamed_interface({"rcc.n_timepoints": "w"}, before, ["drift", before[1]])
    assert renamed == {"drift.n_timepoints": "w"}


def test_the_choices_are_every_steps_settings_down_through_its_parts():
    cls = chain_class(stats_chain())
    choices = {key: dict(rows) for key, _, rows in chain.interface_choices(cls)}
    assert "rcc.n_timepoints" in choices["rcc"]
    assert not any(path.endswith(chain.GROUPING_FIELD) for rows in choices.values()
                   for path in rows)
