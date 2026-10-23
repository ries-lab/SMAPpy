"""What a settings form does with a number that will not do: refused, naming the field and the bound."""
import dataclasses
from dataclasses import dataclass
from typing import Optional

import pytest

pytest.importorskip("PySide6")

from smappy import plugins                                  # noqa: E402
from smappy.plugins import param, param_specs               # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@dataclass
class Settings:
    radius_nm: float = param(50.0, label="radius", unit="nm", min=0.1, max=500.0)
    count: Optional[int] = param(None, label="count", min=1)


def _typed(form, name, text):
    field = form.fields[name]
    if field.auto is not None:
        field.auto.setChecked(False)
    field.widget.setText(text)
    field.widget.editingFinished.emit()
    return field


@pytest.mark.parametrize("name, text, said", [
    ("radius_nm", "abc", "radius: 'abc' is not a number"),
    ("radius_nm", "", "radius: empty, a number is needed"),
    ("radius_nm", "-5", "radius: -5 is below its least, 0.1 nm"),
    ("radius_nm", "1e4", "radius: 1e4 is above its most, 500 nm"),
    ("count", "2.7", "count: '2.7' is not a whole number"),
    ("count", "0", "count: 0 is below its least, 1"),
])
def test_a_word_a_decimal_or_a_value_beyond_its_bounds_is_refused_naming_the_field(app, name, text, said):
    from smappy.gui.params import SettingsForm
    form = SettingsForm(Settings, param_specs(Settings))
    field = _typed(form, name, text)
    assert field.problem() == said and field.widget.toolTip() == said
    assert "d62728" in field.widget.styleSheet()                  # red when typed, not only at the run
    with pytest.raises(ValueError, match=said.split(":")[0]):
        form.value()
    # put right, it is taken and no longer red
    _typed(form, name, "3")
    assert field.problem() == "" and "d62728" not in field.widget.styleSheet()
    assert getattr(form.value(), name) == 3


def _settings_classes():
    """Every settings dataclass of the shipped plugins and the PSF calibration window, parts included."""
    from smappy.calibrate.insitu import InSituSettings
    from smappy.calibrate.insitu_run import InSituModelSettings
    from smappy.calibrate.pupil import OpticsSettings
    roots = [plugins.get(p).Settings for p, r in plugins.refs().items() if r.root == "builtin"]
    roots += [InSituSettings, InSituModelSettings, OpticsSettings]
    seen, todo = [], [c for c in roots if c is not None]
    while todo:
        cls = todo.pop()
        if cls in seen:
            continue
        seen.append(cls)
        todo += [f.type for f in param_specs(cls).values() if dataclasses.is_dataclass(f.type)]
    return seen


def test_every_shipped_numeric_default_is_within_its_bounds():
    # a default the form itself would refuse turns the field red the moment it is shown
    wrong = []
    for cls in _settings_classes():
        for name, spec in param_specs(cls).items():
            if spec.type not in (int, float) or spec.default is None or not isinstance(spec.default, (int, float)):
                continue
            lo, hi = spec.info.min, spec.info.max
            if (lo is not None and spec.default < lo) or (hi is not None and spec.default > hi):
                wrong.append(f"{cls.__name__}.{name} = {spec.default} not in [{lo}, {hi}]")
    assert not wrong, wrong


def test_a_value_set_from_code_beyond_the_bounds_is_shown_red_but_passed_on(app):
    # a chain, a script or MicroClaw's live fit sets what it needs; the bounds are the GUI's, for typing
    from smappy.gui.params import SettingsForm
    form = SettingsForm(Settings, param_specs(Settings))
    form.set(Settings(radius_nm=0.01))
    field = form.fields["radius_nm"]
    assert form.value().radius_nm == 0.01
    assert "below its least" in field.problem() and "d62728" in field.widget.styleSheet()
