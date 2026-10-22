"""Reading an EMU configuration (``.uicfg``) into microscope devices.

EMU, the Micro-Manager plugin framework, runs the Ries lab's microscopes
through its ht-SMLM interface, and its configuration says what the
acquisition's metadata does not: which Micro-Manager property is which
control, and what the control's states are called.  One plugin
configuration has three blocks:

* ``properties`` -- each control's property and the value of each state:
  ``Two-state device 2 = Servos-Position1``, ``- On value = 28000``;
  ``Filter wheel position state 3 = 32000``;
* ``parameters`` -- the names: ``Controls - Two-state device 2 name = 3DA``,
  ``Filters - Filter names = 440/40,525/50,...`` (one per state, in order),
  ``2C - Slider 1 names`` and ``2C - Slider 1 title = Channel 1
  (transmitted)``;
* ``settings`` -- which panels exist (``Additional FW tab title = 2C``).

So a state is decoded by a chain -- the property's value, the control's
state with that value, the state's name -- and an import turns each
allocated control into a `microscope_db.Device` with its states named.
What no configuration says is what a name means: that *3DA* is the
cylindrical lens, the NA, the pixel.  `draft` leaves those to the editor,
filling in only what follows from the names (a wavelength from a filter's
``676/37``) and the 3D lens when the caller says which device it is: EMU has
no role for it.

The focus lock, the QPD, the triggers, laser powers and the linear stages
are left out: no parameter of the optics follows them.  A linear stage that
is also a two-state device (M5's 3D lens) comes in as the two-state device.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..camera_db import read_number
from ..microscope_db import FROM_NAME, Channel, Device, Microscope

UNALLOCATED = ("", "Unallocated", "Enter value")

#: the Mojo FPGA became MicroFPGA, whose devices lost the prefix
#: (``Mojo-Servos-Position1`` -> ``Servos-Position1``)
RENAMED_PREFIXES = ("Mojo-",)


@dataclass
class EmuConfig:
    properties: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, str] = field(default_factory=dict)
    settings: Dict[str, str] = field(default_factory=dict)
    name: str = ""
    path: Optional[Path] = None
    modified: Optional[_dt.date] = None

    def prop(self, key: str) -> str:
        value = str(self.properties.get(key, "")).strip()
        return "" if value in UNALLOCATED else value

    def names(self, key: str) -> List[str]:
        return [n.strip() for n in str(self.parameters.get(key, "")).split(",")
                if str(self.parameters.get(key, "")).strip()]


def read(path, configuration: str = "") -> EmuConfig:
    """One plugin configuration of an EMU file: the one named, else the
    file's default, else the first ht-SMLM one, else the first."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    found = raw.get("pluginConfigurations") or []
    wanted = configuration or raw.get("defaultConfigurationName", "")
    chosen = (next((c for c in found if c.get("configurationName") == wanted), None)
              or next((c for c in found if c.get("pluginName") == "ht-SMLM"), None)
              or (found[0] if found else {}))
    try:
        modified = _dt.date.fromtimestamp(path.stat().st_mtime)
    except OSError:
        modified = None
    return EmuConfig(properties={str(k): str(v) for k, v in (chosen.get("properties") or {}).items()},
                     parameters={str(k): str(v) for k, v in (chosen.get("parameters") or {}).items()},
                     settings={str(k): str(v) for k, v in (chosen.get("settings") or {}).items()},
                     name=str(chosen.get("configurationName", "")), path=path, modified=modified)


def numbers(config: EmuConfig, pattern: str) -> List[int]:
    """The control numbers a configuration has, in order: ``Laser (\\d+)
    enable`` finds 0 to 3 -- or only 3, since nothing makes them consecutive."""
    found = (re.fullmatch(pattern, key) for key in config.properties)
    return sorted({int(m.group(1)) for m in found if m})


def tags_for(prop: str) -> List[str]:
    """The tags a property is looked up by: as configured, and without a
    prefix the device has since dropped."""
    out = [prop]
    for prefix in RENAMED_PREFIXES:
        if prop.startswith(prefix):
            out.append(prop[len(prefix):])
    return out


def _device(name: str, prop: str, states: Dict[str, Any], emu: str) -> Device:
    numeric = all(read_number(v) is not None for v in states.values())
    return Device(name=name, tags=tags_for(prop), read="nearest" if numeric else "exact",
                  states=states, emu=emu)


def _named_states(config: EmuConfig, base: str, names: List[str], warnings: List[str],
                  device: str) -> Dict[str, Any]:
    """``{name: value}`` of a multi-position control, ``base state k``.

    A name given twice gets a number; a value given to two states keeps
    both names but only the first is ever read, which is said.
    """
    states: Dict[str, Any] = {}
    k = 0
    while f"{base} state {k}" in config.properties:
        value = config.prop(f"{base} state {k}")
        name = names[k] if k < len(names) and names[k] else f"state {k}"
        if name in states:
            name = f"{name} ({k})"
        if value:
            twin = next((n for n, v in states.items() if str(v) == value), None)
            if twin is not None:
                warnings.append(f"{device}: {name} and {twin} are both at {value}; "
                                f"a file reading {value} is taken as {twin}")
            states[name] = value
        k += 1
    return states


def devices(config: EmuConfig) -> tuple:
    """``({name: Device}, [warnings])``: every allocated control that a
    parameter of the optics can follow."""
    out: Dict[str, Device] = {}
    warnings: List[str] = []

    def add(device: Device):
        if device.name in out:
            device.name = f"{device.name} ({device.emu})"
        out[device.name] = device

    # filter wheels: "Filter wheel position", "Filter wheel 2 position"
    for number, suffix in (("", ""), (" 2", " 2")):
        base = f"Filter wheel{number} position"
        prop = config.prop(base)
        if not prop:
            continue
        name = f"filter wheel{number}"
        names = config.names(f"Filters - Filter names{suffix}")
        add(_device(name, prop, _named_states(config, base, names, warnings, name),
                    f"Filter wheel{number}"))

    # the sliders of the additional tab, "<tab> - Slider N names"
    tab = config.settings.get("Additional FW tab title", "")
    for n in numbers(config, r"Slider (\d+) position"):
        prop = config.prop(f"Slider {n} position")
        if prop:
            title = config.parameters.get(f"{tab} - Slider {n} title", "").strip()
            name = title or f"slider {n}"
            names = config.names(f"{tab} - Slider {n} names")
            add(_device(name, prop, _named_states(
                config, f"Slider {n} position", names, warnings, name), f"Slider {n}"))

    # two-state devices, named in the Controls panel
    for n in numbers(config, r"Two-state device (\d+)"):
        prop = config.prop(f"Two-state device {n}")
        enabled = config.parameters.get(f"Controls - Enable two-state device {n}", "true")
        if prop and enabled.strip().lower() != "false":
            name = config.parameters.get(f"Controls - Two-state device {n} name",
                                         "").strip() or f"two-state device {n}"
            states = {"on": config.prop(f"Two-state device {n} - On value"),
                      "off": config.prop(f"Two-state device {n} - Off value")}
            add(_device(name, prop, {k: v for k, v in states.items() if v},
                        f"Two-state device {n}"))

    # lasers on or off: which one lit the sample
    for n in numbers(config, r"Laser (\d+) enable"):
        prop = config.prop(f"Laser {n} enable")
        if prop:
            label = config.parameters.get(f"Laser {n} - Name", "").strip() or str(n)
            states = {"on": config.prop(f"Laser {n} enable - On value"),
                      "off": config.prop(f"Laser {n} enable - Off value")}
            add(_device(f"laser {label}", prop, {k: v for k, v in states.items() if v},
                        f"Laser {n}"))
    return out, warnings


def channels(found: Dict[str, Device]) -> List[Channel]:
    """The arms of a split, from the sliders' titles: *Channel 1
    (transmitted)* and *Channel 2 (reflected)*, transmitted first, because
    it is the one that applies with the dichroic out.  The sides are not in
    the configuration."""
    out = []
    for arm in ("transmitted", "reflected"):
        device = next((name for name in found if arm in name.lower()), None)
        if device is not None:
            out.append(Channel(name=arm, filter=device))
    return out


def draft(config: EmuConfig, name: str = "", three_d: str = "",
          three_d_in: str = "on") -> tuple:
    """``(Microscope, [warnings])``: a new entry from a configuration.

    `three_d` names the device that is the cylindrical lens, and `three_d_in`
    its state with the lens in -- EMU has neither.  The rest that is not in
    the configuration (the identifying tags, the NA, the pixel) is left to
    the editor.
    """
    found, warnings = devices(config)
    parameters: Dict[str, Any] = {}
    wheel = next((n for n, d in found.items() if d.emu == "Filter wheel"), None)
    if wheel is not None:
        parameters["wavelength_nm"] = {"device": wheel, "values": FROM_NAME}
    if three_d:
        if three_d not in found:
            raise ValueError(f"no device {three_d!r} in {config.path}: "
                             f"{', '.join(found)}")
        other = [s for s in found[three_d].states if s != three_d_in]
        parameters["modality"] = {"device": three_d, "values": dict(
            {three_d_in: "astigmatic"}, **{s: "2D" for s in other})}
    entry = Microscope(name=name or config.name, devices=found, parameters=parameters,
                       channels=channels(found))
    return entry, warnings


def update(entry: Microscope, config: EmuConfig) -> tuple:
    """``(Microscope, [warnings])``: `entry` with its imported devices read
    again from a newer configuration.

    A device is matched by the EMU control it came from and keeps its name,
    so every parameter that follows it still does; its tag becomes the new
    one, the old kept behind it for older files, and its states take the
    new values under their names.  Devices the configuration does not have
    are kept as they are; new ones are added.  `microscope_db.save_microscope`
    keeps what the entry said before as a version.
    """
    import copy
    entry = copy.deepcopy(entry)
    found, warnings = devices(config)
    by_control = {d.emu: name for name, d in entry.devices.items() if d.emu}
    for new in found.values():
        name = by_control.get(new.emu)
        if name is None:
            if new.name in entry.devices:
                new.name = f"{new.name} ({new.emu})"
            entry.devices[new.name] = new
            continue
        old = entry.devices[name]
        tags = new.tags + [t for t in old.tags if t not in new.tags]
        states = dict(old.states)
        for state, value in new.states.items():
            states[state] = value
        entry.devices[name] = Device(name=name, tags=tags, read=new.read, states=states,
                                     tolerance=old.tolerance, emu=old.emu)
    return entry, warnings
