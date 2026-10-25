# The simulator, redesigned: a scene of components

Status: built (2026-10).  *Simulate* (`File/Simulate/Simulate`,
`plugins/simulate.py`) replaces *Simulate structure* and *Simulate tracks*.
Backwards compatibility is not kept: the structure YAML's `dye` is gone and
saved settings of the old plugins are not read.  Where the build departs
from the plan below, it says so.  `simulate_tracks()`, the tracking tests'
model, is kept as it was rather than rebuilt on scenes: its per-frame
blinking and fixed precision are what those tests and benchmarks are
calibrated on.

## The model

A **scene** is a list of **components** plus what they share.

**Shared (the scene's):** frames, frame time, drift, camera background,
optics (pixel size, PSF, astigmatism), detection (limit, close emitters),
output, seed.

**A component** is one kind of thing in the sample, with its own:

| part | settings |
| --- | --- |
| name, enabled | the name the list shows; a tick to leave it out (not `on`: YAML reads a bare `on:` as true) |
| structure | a built-in, a file, *homogeneous* (random positions at a density over the field and a z range) or *beads* |
| placement | copies: number or density, field, rotation, tilt, jitter, minimum distance; or **same as** another component |
| labelling | efficiency, fluorophores per label (Poisson), linkage error fixed and free |
| photophysics | photons ± std, on time, blinks or bleaching, activation; or *always on* (beads); or **same as** another component |
| motion | off, or diffusion: D per state, fractions, switching; moves single labels |
| colour | the output `channel`; the split camera's ratio |

The structure YAML is geometry only: points, lines, circles, polygons,
images, copies.  Its `dye` is removed; a colour is a component's.  A
structure file with several `parts:` stays one component.

**Same as.**  Two links, each a choice of another component:

* *placement same as X*: this component's copies are put where X's are and
  turned as X's are -- the same poses (`Labels.poses`) -- with this
  component's own geometry.  A two-colour pore: Nup96 in colour 1, a second
  Nup in colour 2, each pore at a random place and orientation, the two
  colours of a pore together.  The two must have the same number of copies;
  the linked one's copy settings are not shown.
* *photophysics same as X*: X's photons, blinking and activation.  Two
  components in one dye stay in step.

A link to a component that is itself linked follows the chain; a cycle or a
link to a missing or switched-off component is refused with its name.

**Tracks** are a homogeneous component with motion and long-lived
photophysics (long on time, short off times).  Precision comes from the
photons, as for everything else.  `simulate_tracks()` stays, as a wrapper
that builds such a scene, for scripts and the tracking tests.

**Beads** are a component of structure *beads* (copies at a density or a
number) with photophysics *always on*, bright, no bleaching.

**Motion** moves single labels by Brownian steps, frame to frame, the state
held over the interval.  No motion blur: a camera frame draws the position
at the frame.  Moving whole copies is left for later.

## Outputs

* **localizations**: one table, `channel` = the component's colour; truth
  columns `component`, `emitter`, `copy`, and with motion `track_id_true`
  and `state_true`, the names the tracking tools read.
* **camera frames**: one camera, all components drawn.
* **split camera** (a camera option, *camera frames > split camera*): two
  halves, each component's photons split by its ratio; the transformation
  between the halves (`camera.dual_transformation`) is written beside the
  recipe as `<name>_2ct.h5`, which a two-colour fitter given the recipe
  takes for its *transformation* (`_FitPlugin.react`) -- unless a dual-colour
  *PSF calibration* is set, whose halves, registration and two PSFs it draws
  instead (`camera.split_layout`), for Spline 3D 2C -- and *Fit this* hands
  a split recipe to Gaussian 2D 2C.  So simulated two-colour pores go
  straight into the 2D two-colour workflow.

Each output can load the truth as a file.

## Presets and components

* A **scene preset** is a YAML file of the whole settings, shipped in
  `smappy/data/simulation_presets`, the user's in
  `<config dir>/simulation_presets`.  Choosing one fills every setting;
  any can then be changed, and the list says *(modified)*.  *Save...* writes
  the current scene as a preset of the user's; *delete* removes one.
* A preset names its **key parameters** -- the few fields shown at the top,
  across components (`NPC: labelling efficiency`, `background: density`).
  Everything else is folded below.
* A **component** can be saved on its own (*save component...*) and added
  to any scene (*add > from saved*): "NPC", "beads", "background".
* Shipped scene presets: Nuclear pores (Nup96), Two-colour pores, Ring and
  cross (two colours), Filaments, Diffusion, Two-state diffusion, Pores with
  beads.

## The interface

```
preset   [Nuclear pores (Nup96)   v]  [save...] [delete]
- key parameters --------------------------------------
NPC: structure      [npc v]
frames              [20000]
NPC: photons        [5000] +- [2500]
NPC: labelling      [0.6]
output              [localizations v]   [ ] add as new file
- components ------------------------------- (folded)
  [x] NPC          colour 1
  [x] background   colour 1    [+ v] [duplicate] [-] [save...]
  > the selected component's form: structure, placement,
    labelling, photophysics, motion, colour
> optics   > detection   > camera   > more (seed, frame time, drift)
```

## What the settings framework gains

1. **List-valued settings**: a field `List[ComponentSettings]` is a list of
   parts.  `settings_values` writes it as one key holding a list of flat
   maps; `settings_from` reads it back; forms show a list with a form for
   the selected item; chains, batches and workspaces store it as it is.
   `FormState` addresses an item as `components.<i>.<field>`.
2. **`beside=True`** on a `param`: the field on the row of the one before,
   for "photons +- std".
3. **A preset list** with save and delete, generalising *Segment regions*'
   presets, for scene presets and saved components alike.

## The work, in order

1. The framework: list-valued settings, `beside`, presets -- with tests.
2. The model: components, homogeneous and beads, per-component labelling,
   photophysics and motion, the two *same as* links, `channel`, the tracks
   wrapper.
3. The outputs: localizations, camera, split camera, the truth.
4. The plugin: *Simulate* replacing both, the presets, its page, the
   tutorials that use the simulator (seven), the tests.
