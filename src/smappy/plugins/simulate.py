"""*Simulate*: a scene of components -- structures, a homogeneous background,
beads, diffusing particles -- as localizations or camera frames, one camera
or a split one (`smappy.simulate.scene`; the design is docs/simulation-redesign.md).

It replaces *Simulate structure* and *Simulate tracks*: a single structure
is a scene of one component, and tracks are a homogeneous component that
diffuses.  What a person sets first is a preset -- a whole scene, saved as
YAML in ``data/simulation_presets`` or the config folder -- which fills
every setting and names its **key parameters**, the few fields shown again
at the top (`FormState.keys`); everything else is folded below, where any
of it can still be changed.  A component can be saved on its own and added
to another scene (the list's *save...* and *+*).

The form follows the scene: a component's structure fields are those of its
kind, its placement those of its mode, its photophysics folded away while
it takes another component's, and the camera's settings shown only for a
camera.  The *same as* lists offer the other components by name.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from ..io.formats import FileInfo
from ..simulate.settings import SceneSettings, preset_folders
from . import Context, FormState, Notice, Plugin, Result, param, register

#: the key parameters of a scene that names none -- the default, or one of
#: the user's saved before keys were written with it
DEFAULT_KEYS = {"components.0.structure.preset": "structure",
                "components.0.labelling.efficiency": "labelling",
                "components.0.blinking.photons": "photons per blink",
                "n_frames": "frames"}
#: added to any preset's keys while the output is a camera: where the recipe
#: goes is what a test of a fitting workflow sets first
CAMERA_KEYS = {"camera.path": "simulation file"}
#: added too while the scene has two colours: the split camera is how they
#: reach the two-colour fit, and folded under *camera frames* it was missed
SPLIT_KEYS = {"camera.split": "split camera"}


@dataclass
class SimulateSettings(SceneSettings):
    """The scene as the plugin's form shows it; *add as a new file* is the
    scene's own field (`SceneSettings.append`), placed beside *add drift*."""


def auto_recipe(settings: SceneSettings):
    """Where a camera simulation is written when no file is given: named
    after its preset, in the folder last opened (or the home folder)."""
    from pathlib import Path
    from .. import config
    folder = config.last_folder() or Path.home()
    name = "".join(c if c.isalnum() or c in " -_" else "_" for c in
                   (settings.preset or "simulation")).strip() or "simulation"
    return Path(folder) / f"{name}.sim.yaml"


def recipe_of(settings: SceneSettings):
    """The recipe a camera run writes: the file given, else `auto_recipe`."""
    from ..simulate.source import recipe_path
    return recipe_path(settings.camera.path) if settings.camera.path \
        else auto_recipe(settings)


# ------------------------------------------------------------------ presets

def presets() -> List[str]:
    from .. import presets as store
    return store.names(preset_folders("simulation_presets"))


def preset_values(name: str) -> Dict[str, Any]:
    """A preset's settings over the defaults, so what it leaves out is reset
    rather than kept from the scene before; without its key list."""
    from .. import presets as store
    from . import settings_from, settings_values
    found = store.read(preset_folders("simulation_presets"), name)
    found = {k: v for k, v in found.items() if not k.startswith("key.")}
    found["preset"] = name
    # through the settings, so the values are spelt as the form spells them
    # (a component a flat map) and what the preset leaves out is the default
    values = settings_values(settings_from(SimulateSettings, found))
    values.pop("append", None)
    return values


def preset_keys(name: str) -> Dict[str, str]:
    """The key parameters a preset names (``key: {dotted name: label}``)."""
    from .. import presets as store
    if not name:
        return dict(DEFAULT_KEYS)
    try:
        found = store.read(preset_folders("simulation_presets"), name)
    except (ValueError, OSError):
        return dict(DEFAULT_KEYS)
    keys = {k[len("key."):]: str(v) for k, v in found.items() if k.startswith("key.")}
    return keys or dict(DEFAULT_KEYS)


def modified(settings: SimulateSettings) -> bool:
    """Whether the settings have moved from the preset they started from."""
    from . import settings_values
    if not settings.preset:
        return False
    try:
        wanted = preset_values(settings.preset)
    except (ValueError, OSError):
        return False
    now = settings_values(settings)
    now.pop("append", None)
    return any(now.get(k) != v for k, v in wanted.items())


# ------------------------------------------------------------------ the form

def component_state(scene: SceneSettings) -> FormState:
    """What each component's part of the form shows (`FormState`, by
    ``components.<i>.``): the fields of its structure's kind and its
    placement's mode, its photophysics folded away while it takes another's,
    and the other components offered to its *same as* lists."""
    state = FormState()
    names = [c.name for c in scene.components]
    split = scene.output == "camera" and scene.camera.split
    for i, c in enumerate(scene.components):
        at = f"components.{i}."
        others = [("", "-")] + [(n, n) for n in names if n != c.name]
        state.choices[at + "placement.same_as"] = others
        state.choices[at + "photophysics_from"] = [("", "its own")] + others[1:]
        kind = c.structure.kind
        structure = kind == "structure"
        state.shown.update({
            at + "structure.preset": structure,
            at + "structure.file": structure and c.structure.preset == "file",
            at + "structure.density_um2": not structure,
            at + "structure.z_range_nm": kind == "homogeneous",
            at + "placement": structure,
            at + "blinking": not c.photophysics_from,
            at + "split_ratio": split,
        })
        mode = c.placement.mode
        for name in ("n", "density_um2", "rotation", "tilt_deg", "jitter_nm",
                     "min_distance_nm"):
            state.shown[at + "placement." + name] = mode == "copies"
        state.shown[at + "placement.same_as"] = mode == "same"
        state.active[at + "placement.density_um2"] = c.placement.n == 0
        activation = c.blinking.activation
        state.active.update({
            # every blink until bleached: 1 / blinks is the bleaching; always
            # on: one emission, no blinks, no off time
            at + "blinking.bleaching": activation in ("constant", "decay"),
            at + "blinking.blinks": activation != "always",
            at + "blinking.on_time": activation != "always",
            at + "blinking.off_time": activation in ("constant", "decay")})
        moving = c.motion.diffusion
        two = c.motion.slow_fraction > 0
        state.active.update({at + "motion.d_um2_s": moving,
                             at + "motion.slow_fraction": moving,
                             at + "motion.d2_um2_s": moving and two,
                             at + "motion.switch_probability": moving and two})
    return state


def key_labels(scene: SceneSettings, keys: Dict[str, str]) -> Dict[str, str]:
    """The keys' labels, a component's prefixed with its name when the scene
    has several: "NPC: photons per blink"."""
    out = {}
    several = len(scene.components) > 1
    for path, label in keys.items():
        parts = path.split(".")
        if parts[0] == "components" and len(parts) > 2 and parts[1].isdigit():
            i = int(parts[1])
            if i >= len(scene.components):
                continue                     # a key of a component since removed
            if several:
                label = f"{scene.components[i].name}: {label}"
        out[path] = label
    return out


def present_density(scene: SceneSettings, c) -> float:
    """Roughly how many of a homogeneous component's labels are on in a
    frame, per µm²: all of them always on, else each for its blinks times
    its on time of the frames.  0 for a structure, where it means little."""
    if c.structure.kind == "structure":
        return 0.0
    b = c.blinking
    on = 1.0 if b.activation == "always" else min(1.0, b.blinks * b.on_time / scene.n_frames)
    return c.structure.density_um2 * c.labelling.efficiency * on


def colours(scene: SceneSettings) -> int:
    """How many colours (`channel`) the switched-on components are in."""
    return len({c.channel for c in scene.components if c.enabled})


def motion_notice(scene: SceneSettings) -> Optional[Notice]:
    """For diffusing components, the jump in a frame against the distance
    between particles: what *Link tracks* is set by."""
    from ..simulate import jump_nm, spacing_nm
    bits = []
    for c in scene.components:
        if not (c.enabled and c.motion.diffusion):
            continue
        d = max(c.motion.d_um2_s, c.motion.d2_um2_s if c.motion.slow_fraction > 0 else 0.0)
        jump = jump_nm(d, scene.frame_time_ms)
        density = present_density(scene, c)
        apart = (f", about {spacing_nm(density):.0f} nm apart when on" if density > 0
                 else "")
        bits.append(f"{c.name}: rms jump {jump:.0f} nm{apart}")
    return Notice("; ".join(bits), kind="info") if bits else None


# ------------------------------------------------------------------ the plugin

@register("File/Simulate/Simulate")
class Simulate(Plugin):
    """A scene -- structures, a background, beads, diffusing particles, in
    one colour or several -- as localizations, camera frames to fit, or a
    split camera for the two-colour fit (`smappy.simulate`)."""

    Settings = SimulateSettings
    changes_locs = True
    # 2: diffusion in 3D by default; the split camera a camera option, with
    # its transformation written beside the recipe
    version = "2"

    def describe(self, ctx: Optional[Context], settings: SimulateSettings) -> FormState:
        """The preset's key parameters at the top; each component's fields
        those it reads; the camera's settings for a camera only; and, for
        a camera, *Fit this*."""
        state = component_state(settings)
        camera = settings.output == "camera"
        state.shown.update({"camera": camera, "localizations": not camera,
                            "camera.split_sigma_nm": camera and settings.camera.split})
        keys = preset_keys(settings.preset)
        if camera:
            keys.update(CAMERA_KEYS)
            state.hints["camera.path"] = str(recipe_of(settings))
            state.hints["camera.split_sigma_nm"] = settings.optics.sigma_nm
            if colours(settings) > 1:
                keys.update(SPLIT_KEYS)
        # a key whose field the form hides (a split ratio without a split
        # camera, photophysics taken from another) is not shown at the top
        hidden = [path for path, on in state.shown.items() if not on]
        keys = {k: v for k, v in keys.items()
                if not any(k == h or k.startswith(h + ".") for h in hidden)}
        state.keys = key_labels(settings, keys)
        state.hints["preset"] = "modified" if modified(settings) else ""
        state.notice = motion_notice(settings)
        # two colours on one unsplit camera are one colour to any fitter
        unsplit = (f"{colours(settings)} colours drawn on one camera, as one: tick "
                   f"split camera for frames the two-colour fitter can tell apart"
                   if camera and colours(settings) > 1 and not settings.camera.split
                   else "")
        if camera:
            # Fit this writes the recipe from the settings as they are and
            # hands it to a fitter (the GUI's "fit simulation" window), as Run
            # writes it: a file left by an earlier run is never what is fitted
            what = "the two-colour fitter" if settings.camera.split else "a fitter"
            text = (f"Fit this writes {recipe_of(settings).name} with these settings "
                    f"and opens it in {what}")
            motion = state.notice.text + ".  " if state.notice is not None else ""
            state.notice = Notice(unsplit or motion + text, action="Fit this",
                                  window="fit simulation",
                                  kind="warning" if unsplit else "info")
        return state

    def react(self, changed: str, settings: SimulateSettings) -> Optional[Dict[str, Any]]:
        """Choosing a preset fills every setting from it; choosing a
        component's built-in structure forgets a file chosen before, so a
        file no longer shown cannot decide the structure."""
        if changed == "preset" and settings.preset:
            return preset_values(settings.preset)
        parts = changed.split(".")
        if (len(parts) == 4 and parts[0] == "components" and parts[2:] == ["structure", "preset"]
                and parts[1].isdigit()):
            i = int(parts[1])
            if i < len(settings.components) and settings.components[i].structure.preset != "file":
                return {f"components.{i}.structure.file": ""}
        return None

    def run(self, ctx: Context, settings: SimulateSettings) -> Result:
        from ..simulate import ground_truth, localizations
        if not isinstance(settings, SceneSettings):
            # one structure's settings, as a script says it: a scene of one
            from ..simulate.scene import as_scene
            settings = SimulateSettings(**{f: getattr(as_scene(settings), f)
                                           for f in SceneSettings.__dataclass_fields__})
        on = [c for c in settings.components if c.enabled]
        if not on:
            raise ValueError("no component is switched on: tick one in the list")
        ctx.report(f"simulating {settings.n_frames} frames of "
                   f"{', '.join(c.name for c in on)}...")
        truth = ground_truth(settings)
        if settings.output == "camera":
            return self._camera(ctx, settings, truth)
        locs = localizations(truth)
        name = f"simulation: {settings.preset}" if settings.preset else "simulation"
        info = FileInfo(name=name, path=name, format="simulated", n=len(locs),
                        frame_time_ms=settings.frame_time_ms)
        return Result(files=[(locs, info, None)], text=summary(truth, locs),
                      settings=settings, append=settings.append)

    def _camera(self, ctx, settings: SimulateSettings, truth) -> Result:
        from ..simulate.camera import camera_truth, render
        from ..simulate.source import dual_of, transform_path, write_recipe
        cam = settings.camera
        path = write_recipe(settings, recipe_of(settings))
        dual = dual_of(settings)
        n_spots = len(truth.emission)
        text = (f"{n_spots} spots of {len(truth.fluorophores)} emitters over "
                f"{settings.n_frames} frames of {cam.size_px} x {cam.size_px} pixels"
                f"{' per half' if dual else ''}: {path} -- open it in a fitter as "
                f"its file")
        if dual:
            text += (f"; the map between the halves in {transform_path(path).name}, "
                     f"which a two-colour fitter takes with it")
        data = {}
        if cam.tiff:
            import tifffile
            tif = transform_path(path).with_name(
                transform_path(path).name[:-len("_2ct.h5")] + ".tif")
            ctx.report(f"writing {tif}...")
            with tifffile.TiffWriter(tif, bigtiff=True) as writer:
                for a in range(0, settings.n_frames, 500):
                    writer.write(render(truth, a, a + 500, dual=dual), contiguous=True)
            data["tiff"] = str(tif)
            text += f"; the frames in {tif.name}"
        files = []
        if cam.load_truth:
            locs = camera_truth(truth, dual=dual)
            name = f"{path.name} (truth)"
            files = [(locs, FileInfo(name=name, path=str(path), format="simulated",
                                     n=len(locs), frame_time_ms=settings.frame_time_ms),
                      None)]
        return Result(files=files, text=text, data=data, written=str(path),
                      settings=settings, append=settings.append)


def summary(truth, locs) -> str:
    """What was made, per component when there are several."""
    import numpy as np
    md = locs.metadata
    names = truth.components()
    text = (f"{len(locs)} localizations of {md['n_emitters']} emitters "
            f"({md['n_labels']} labels), {md['n_blinks']} blinks over "
            f"{md['n_frames']} frames")
    if len(set(np.asarray(locs["component"]).tolist())) > 1 or len(names) > 1:
        counts = np.bincount(np.asarray(locs["component"]), minlength=len(names))
        text += " -- " + ", ".join(f"{n}: {c}" for n, c in zip(names, counts) if c)
    out = truth.settings.localizations
    close = "removed" if out.close == "remove" else "averaged"
    return text + f"; {md['n_close']} too close together, {close}"
