"""A scene: several components in one sample, each with its own labelling,
photophysics and motion, drawn into one ground truth.

A component is one kind of thing -- a structure (nuclear pores, a ring),
labels spread evenly over the field (a background, or particles that
diffuse), fiducial beads.  Colour is a component's, not the structure
file's: a two-colour pore is two components, the second placed *same as*
the first (each copy at the first's place and turn, its own geometry), and
two components of one dye share their photophysics with *photophysics as*.
The design is docs/simulation-redesign.md.

Every component runs the stages `localizations.ground_truth` always ran --
labels, labelling, blinking, the free linkage error, emission -- in that
order and from one generator, so a scene of one component draws exactly the
numbers the single-structure simulation drew and its data are unchanged.
What a component adds, it draws after: motion, then the next component.
The emission of all of them is merged and sorted by frame, and the drift,
the localizations and the camera frames are drawn from that as before.

**Motion** moves each label by Brownian steps of ``sqrt(2 D dt)`` per axis
and frame, as `tracks` did, with diffusive states drawn again with
``switch_probability`` per frame.  Only the frames a fluorophore emits in
are needed: with one state the step between two of them is one Gaussian of
the variance of all the frames between; with two, the frames between are
walked one by one, which is what a state that may switch every frame needs.
No motion blur: a frame sees the position at that frame.  In the plane
unless *in z* is ticked.  With motion the table also carries the truth the
tracking tools read: ``track_id_true`` (the emitter, from 1; 0 for a label
that does not move) and ``state_true`` (-1 for one that does not).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np

from .kinetics import Blinks, Emission, Fluorophores, blink, blink_offsets, emission, label
from .settings import (ComponentSettings, MotionSettings, SceneSettings,
                       SimulationSettings, StructureSettings)
from .structure import Labels, load_structure


def as_scene(settings: SimulationSettings) -> SceneSettings:
    """The single-structure settings as a scene of one component."""
    component = ComponentSettings(name="structure", structure=settings.structure,
                                  labelling=settings.labelling,
                                  blinking=settings.blinking)
    shared = {f: getattr(settings, f) for f in
              ("output", "n_frames", "frame_time_ms", "background", "background_std",
               "drift", "seed", "optics", "localizations", "camera")}
    return SceneSettings(components=[component], **shared)


def active(scene: SceneSettings) -> List[ComponentSettings]:
    """The components switched on, in an order where every component comes
    after the one it is placed *same as*; a link that cannot be followed is
    refused, naming it."""
    on = [c for c in scene.components if c.enabled]
    names = [c.name for c in scene.components]
    for c in on:
        if names.count(c.name) > 1:
            raise ValueError(f"two components are called {c.name!r}: a link "
                             f"would not know which")
    by_name = {c.name: c for c in on}
    for c in on:
        for link in (_placed_as(c), c.photophysics_from):
            if link and link not in by_name:
                what = "switched off" if link in names else "not in the scene"
                raise ValueError(f"{c.name}: {link!r} is {what}")
    order: List[ComponentSettings] = []
    placed = set()
    pending = list(on)
    while pending:
        ready = [c for c in pending if not _placed_as(c) or _placed_as(c) in placed]
        if not ready:
            raise ValueError("these components are placed as one another in a circle: "
                             + ", ".join(c.name for c in pending))
        for c in ready:
            order.append(c)
            placed.add(c.name)
            pending.remove(c)
    return order


def _placed_as(c: ComponentSettings) -> str:
    return c.placement.same_as if c.placement.mode == "same" else ""


def photophysics(scene: SceneSettings, c: ComponentSettings):
    """The blinking ``c`` uses: its own, or that of the component it names,
    followed along a chain of *as*."""
    seen = []
    while c.photophysics_from:
        if c.name in seen:
            raise ValueError("photophysics as one another in a circle: "
                             + ", ".join(seen))
        seen.append(c.name)
        c = scene.component(c.photophysics_from)
    return c.blinking


def component_labels(scene: SceneSettings, c: ComponentSettings, rng,
                     placements: Dict[str, list]) -> Labels:
    """Where ``c``'s labels are, and its copies' places for those *same as* it."""
    s = c.structure
    field = scene.field_um * 1000.0
    if s.kind == "structure":
        structure = load_structure(s.source())
        if c.placement.mode == "same":
            where = placements[c.placement.same_as]
            return structure.sample(rng, placements=where)
        copies = c.placement.copies(field) if c.placement.mode == "copies" else None
        return structure.sample(rng, copies=copies)
    if s.kind in ("homogeneous", "beads"):
        n = rng.poisson(s.density_um2 * scene.field_um ** 2)
        xy = rng.uniform(0, field, (n, 2))
        z = (rng.uniform(-s.z_range_nm / 2, s.z_range_nm / 2, n) if s.kind == "homogeneous"
             else np.zeros(n))
        # each label its own copy: a particle, a bead
        return Labels(np.column_stack([xy, z]), np.arange(n, dtype=np.int32))
    raise ValueError(f"{c.name}: labels {s.kind!r} are a structure, homogeneous or beads")


def motion(settings: MotionSettings, em: Emission, frame_time_ms: float,
           rng) -> Optional[tuple]:
    """Each emission row's displacement from where its fluorophore started,
    (len(em), 3) nm, and the diffusive state it was in -- or None without
    motion.  ``em`` is one component's, sorted by frame."""
    if not settings.diffusion or not len(em):
        return None
    two = settings.slow_fraction > 0
    d = np.array([settings.d_um2_s, settings.d2_um2_s] if two else [settings.d_um2_s])
    fractions = (np.array([1 - settings.slow_fraction, settings.slow_fraction]) if two
                 else np.array([1.0]))
    sigma = np.sqrt(2.0 * d * 1e6 * frame_time_ms * 1e-3)      # nm per axis and frame
    order = np.lexsort((em.frame, em.owner))                   # by fluorophore, in time
    owner, frame = em.owner[order], em.frame[order]
    start = np.concatenate([[True], owner[1:] != owner[:-1]])
    gap = np.diff(frame, prepend=frame[0])
    gap[start] = 0
    if not two or settings.switch_probability <= 0:
        # one state each, for good: a Gaussian step over all the frames between
        state = rng.choice(len(d), int(owner.max()) + 1, p=fractions)[owner]
        step = rng.normal(size=(len(owner), 3)) * (sigma[state] * np.sqrt(gap))[:, None]
    else:
        step, state = _switching_steps(owner, frame, start, sigma, fractions,
                                       settings.switch_probability, rng)
    if not settings.in_z:
        step[:, 2] = 0.0                  # diffusion in the plane, as a membrane
    walked = np.cumsum(step, axis=0)
    first = np.maximum.accumulate(np.where(start, np.arange(len(owner)), 0))
    displacement = np.empty_like(walked)
    displacement[order] = walked - walked[first]
    states = np.empty(len(owner), np.int32)
    states[order] = state
    return displacement, states


def _switching_steps(owner, frame, start, sigma, fractions, p_switch, rng):
    """Steps between successive emitting frames of each fluorophore, every
    frame between walked with the state of the frame before it."""
    n = len(owner)
    gap = np.diff(frame, prepend=frame[0])
    gap[start] = 0
    steps = np.zeros((n, 3))
    state = np.empty(n, np.int64)
    current = rng.choice(len(sigma), n, p=fractions)       # drawn per row, used at starts
    for i in range(n):
        if start[i]:
            s = current[i]
        else:
            k = int(gap[i])
            # a state per frame of the gap: redrawn with p_switch, else kept
            redraw = rng.random(k) < p_switch
            draws = rng.choice(len(sigma), k, p=fractions)
            path = np.empty(k, np.int64)
            for j in range(k):
                path[j] = s
                if redraw[j]:
                    s = draws[j]
            steps[i] = (rng.normal(size=(k, 3)) * sigma[path][:, None]).sum(axis=0)
        state[i] = s
    return steps, state


def scene_truth(scene: SceneSettings):
    """The ground truth of every component switched on, merged."""
    from .localizations import GroundTruth, drift_trace
    rng = np.random.default_rng(scene.seed)
    parts, placements, poses, names = [], {}, {}, []
    n_labels = 0
    for index, c in enumerate(active(scene)):
        labels = component_labels(scene, c, rng, placements)
        placements[c.name] = labels.placements
        poses[c.name] = labels.poses
        fluorophores = label(labels, c.labelling, rng)
        blinking = photophysics(scene, c)
        blinks = blink(len(fluorophores), scene.n_frames, blinking, rng)
        blinks.offset = blink_offsets(len(blinks), c.labelling.linkage_free_nm, rng)
        emitted = emission(blinks, scene.n_frames)
        moved = motion(c.motion, emitted, scene.frame_time_ms, rng)
        k = scene.components.index(c)
        n = len(fluorophores)
        fluorophores.component = np.full(n, k, np.int32)
        fluorophores.channel = np.full(n, c.channel, np.int32)
        fluorophores.ratio = np.full(n, c.split_ratio)
        parts.append((fluorophores, blinks, emitted, moved))
        n_labels += len(labels)
        names.append(c.name)
    fluorophores, blinks, emitted, moved = _merge(parts, scene.n_frames)
    drift = drift_trace(scene.n_frames, rng) if scene.drift else None
    one = len(parts) == 1
    displacement, state = moved if moved is not None else (None, None)
    return GroundTruth(scene, names[0] if one else ", ".join(names), n_labels,
                       fluorophores, blinks, emitted, drift,
                       next(iter(poses.values())) if one else poses, displacement, state)


def _merge(parts, n_frames):
    """One set of fluorophores, blinks and emission rows from the
    components', indices shifted, the emission sorted by frame."""
    if len(parts) == 1:
        return parts[0]
    if not parts:
        empty = np.zeros(0, np.int64)
        return (Fluorophores(np.zeros((0, 3)), empty, empty),
                Blinks(empty, np.zeros(0), np.zeros(0), np.zeros(0), float("nan"),
                       np.zeros((0, 3))),
                Emission(empty, empty, np.zeros(0), empty), None)
    xyz, copy, lab, comp, chan, ratio = [], [], [], [], [], []
    b_owner, b_start, b_end, b_rate, b_offset = [], [], [], [], []
    e_owner, e_frame, e_photons, e_blink, moved = [], [], [], [], []
    n_fl = n_bl = 0
    any_motion = any(m is not None for *_, m in parts)
    for fl, bl, em, mv in parts:
        xyz.append(fl.xyz)
        copy.append(fl.copy)
        lab.append(fl.label)
        comp.append(fl.component)
        chan.append(fl.channel)
        ratio.append(fl.ratio)
        b_owner.append(bl.owner + n_fl)
        b_start.append(bl.start)
        b_end.append(bl.end)
        b_rate.append(bl.rate)
        b_offset.append(bl.offset if bl.offset is not None else np.zeros((len(bl), 3)))
        e_owner.append(em.owner + n_fl)
        e_frame.append(em.frame)
        e_photons.append(em.photons)
        e_blink.append(em.blink + n_bl)
        if any_motion:
            moved.append(mv if mv is not None
                         else (np.zeros((len(em), 3)), np.full(len(em), -1, np.int32)))
        n_fl += len(fl)
        n_bl += len(bl)
    offs = [bl.off_time for _, bl, _, _ in parts if np.isfinite(bl.off_time)]
    fluorophores = Fluorophores(np.vstack(xyz), np.concatenate(copy), np.concatenate(lab),
                                np.concatenate(comp), np.concatenate(chan),
                                np.concatenate(ratio))
    blinks = Blinks(np.concatenate(b_owner), np.concatenate(b_start),
                    np.concatenate(b_end), np.concatenate(b_rate),
                    float(np.mean(offs)) if offs else float("nan"),
                    np.vstack(b_offset))
    frame = np.concatenate(e_frame)
    order = np.argsort(frame, kind="stable")
    emitted = Emission(np.concatenate(e_owner)[order], frame[order],
                       np.concatenate(e_photons)[order], np.concatenate(e_blink)[order])
    if any_motion:
        motion_out = (np.vstack([m for m, _ in moved])[order],
                      np.concatenate([st for _, st in moved])[order])
    else:
        motion_out = None
    return fluorophores, blinks, emitted, motion_out


def two_colour_scene(n_frames: int, seed: int, ratios=(0.25, 0.75),
                     structures=("ring", "cross"), **shared) -> SceneSettings:
    """The demo in two colours: the ring one component, the lines and the
    scattered points the other, each with its split-camera ratio."""
    components = [ComponentSettings(name=name, channel=k + 1, split_ratio=float(r),
                                    structure=StructureSettings(preset=name))
                  for k, (name, r) in enumerate(zip(structures, ratios))]
    return SceneSettings(n_frames=n_frames, seed=seed, components=components, **shared)


def settings_of(values: dict):
    """The settings a simulated table or a recipe recorded (`asdict`): a
    scene when they have components, else one structure."""
    from ..plugins import settings_from
    if isinstance(values, dict) and "components" in values:
        return settings_from(SceneSettings, values)
    return settings_from(SimulationSettings, values or {})


def with_components(scene: SceneSettings, **changes) -> SceneSettings:
    """Every component changed alike: ``with_components(s, labelling=...)``."""
    return replace(scene, components=[replace(c, **changes) for c in scene.components])
