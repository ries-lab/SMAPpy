"""Simulated SMLM data: localizations, or the camera frames to fit.

One model with two outputs, so what is simulated as localizations can be
fitted from its frames by changing nothing but the output:

1. **structure** -- label positions, from a YAML of points, lines, circles,
   polygons and images at a density, and copies of them (`structure`);
2. **labelling** -- a label carries a fluorophore with some efficiency,
   one by default, optionally a Poisson number, off by a linkage error;
3. **blinking** -- on, off, bleached, in continuous time; the off-time set
   from the number of blinks in the measurement, the activation spread evenly
   or decaying; photons per blink from a distribution (`kinetics`);
4. **drift**, optionally;
5. either **localizations** -- Poisson photons, a detection limit, the
   Mortensen precision, close emitters removed or averaged
   (`localizations`) -- or **camera frames** -- pixel-integrated Gaussian
   spots, astigmatic if asked, on a flat background with shot and read noise,
   nothing removed (`camera`), saved as a recipe a fitter opens (`source`).

A **scene** (`scene`) is several such pipelines in one sample: components,
each a structure, homogeneous labels or beads with its own labelling,
photophysics, colour and optional diffusion, one placed where another's
copies are or blinking as another does -- a two-colour pore, a background,
fiducials, diffusing particles.  A single structure is a scene of one.

Single-particle tracks are a model of their own (`tracks`): Brownian
particles with diffusive states and blinking, the truth in ``track_id_true``.
Fiducial beads too (`beads`): bright, still but for the drift, with gaps and
late arrivals, the truth in ``bead_true``.

In the package rather than in ``scripts/`` because it is also a plugin
(``File/Simulate/Simulate``): something to try the whole pipeline
on without a microscope.
"""
from .beads import simulate_bead_tracks, simulate_beads, with_beads
from .camera import (ASTIGMATISM, ISOLATED_NM, astigmatic_sigmas, bead_stacks,
                     camera_frames, camera_truth, dual_bead_stacks, dual_camera_frames,
                     dual_transformation, render)
from .kinetics import blink, emission, expected_blinks, label, off_time_for
from .localizations import (GroundTruth, ground_truth, localizations, mortensen,
                            simulate, unresolvable)
from .scene import as_scene, scene_truth, settings_of, two_colour_scene
from .settings import (BlinkingSettings, CameraOutputSettings, ComponentSettings,
                       LabellingSettings, LocalizationOutputSettings, MotionSettings,
                       OpticsSettings, PlacementSettings, SceneSettings,
                       SimulationSettings, StructureSettings)
from .structure import Labels, Structure, load_structure, presets
from .tracks import TrackSimulationSettings, jump_nm, simulate_tracks, spacing_nm

__all__ = [
    "ASTIGMATISM", "ISOLATED_NM", "astigmatic_sigmas", "bead_stacks", "camera_frames",
    "camera_truth", "dual_bead_stacks", "dual_camera_frames", "dual_transformation",
    "render", "blink", "emission", "expected_blinks", "label", "off_time_for",
    "GroundTruth", "ground_truth", "localizations", "mortensen", "simulate",
    "unresolvable", "BlinkingSettings", "CameraOutputSettings", "LabellingSettings",
    "LocalizationOutputSettings", "OpticsSettings", "SimulationSettings",
    "StructureSettings", "ComponentSettings", "MotionSettings", "PlacementSettings",
    "SceneSettings", "as_scene", "scene_truth", "settings_of", "two_colour_scene",
    "Labels", "Structure", "load_structure", "presets",
    "TrackSimulationSettings", "jump_nm", "simulate_tracks", "spacing_nm",
    "simulate_beads", "simulate_bead_tracks", "with_beads",
]
