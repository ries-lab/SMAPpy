"""The settings of a simulation, one dataclass per stage.

Shared by the two outputs -- localizations and camera frames -- so a
simulation that was looked at as localizations can be fitted from its frames
with nothing changed but the output, and by the plugin, whose form is built
from these fields.  A camera-frame simulation is also saved as exactly this
(`smappy.simulate.source`), which is what lets a fitter open one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..plugins import param
from .structure import preset_choices

STRUCTURE_FILTER = "Structures (*.yaml *.yml);;All files (*)"
SIMULATION_FILTER = "Simulations (*.sim.yaml);;All files (*)"


#: the structure *preset* that means "the file below"
FROM_FILE = "file"
#: what a component's labels are (`StructureSettings.kind`)
STRUCTURE_KINDS = (("structure", "a structure"), ("homogeneous", "homogeneous"),
                   ("beads", "beads"))


@dataclass
class StructureSettings:
    """What is labelled (`smappy.simulate.structure`): a structure from a
    YAML, labels spread evenly (*homogeneous*: a background, or particles
    to diffuse), or fiducial beads."""
    kind: str = param("structure", label="labels", choices=STRUCTURE_KINDS,
                      help="a structure from a YAML; homogeneous: labels at "
                           "random over the field; beads: bright fiducials")
    preset: str = param("demo", label="structure",
                        choices=lambda: preset_choices() + [(FROM_FILE, "from file")],
                        help="a built-in structure; its YAML in smappy/data/structures "
                             "is an example of the syntax")
    file: str = param("", label="structure file", kind="open_file",
                      file_filter=STRUCTURE_FILTER,
                      help="a structure YAML: label positions, or lines, circles, "
                           "polygons and images at a density, and copies of them")
    density_um2: float = param(1.0, label="density", unit="/µm²", min=0,
                               help="homogeneous labels, or beads, per µm² of the "
                                    "scene's field")
    z_range_nm: float = param(0.0, label="z range", unit="nm", min=0,
                              help="homogeneous labels spread evenly over this "
                                   "range of z about 0")

    def source(self) -> str:
        """The YAML read: the file when *from file* is chosen (or a file is
        given, as a script says it), else the built-in."""
        if self.preset == FROM_FILE:
            if not self.file:
                raise ValueError("structure from file: choose the structure file")
            return self.file
        return self.file or self.preset


@dataclass
class LabellingSettings:
    """Labels -> fluorophores."""
    efficiency: float = param(1.0, label="labelling efficiency", min=0, max=1,
                              help="the probability that a label carries a fluorophore")
    fluorophores: float = param(1.0, label="fluorophores per label", min=0,
                                help="exactly this many on a label that is labelled "
                                     "(rounded), or its mean with Poisson ticked")
    poisson: bool = param(False, label="Poisson number",
                          help="the number of fluorophores per label is Poisson")
    linkage_nm: float = param(0.0, label="linkage error, fixed", unit="nm", min=0,
                              help="each fluorophore is off its label by a Gaussian "
                                   "of this width per axis, the same for all its "
                                   "blinks: a rigid linker or antibody")
    linkage_free_nm: float = param(0.0, label="linkage error, free", unit="nm", min=0,
                                   help="a Gaussian offset of this width per axis "
                                        "drawn anew for every blink: a dye that "
                                        "turns freely on a flexible linker")


@dataclass
class BlinkingSettings:
    """On, off, bleached: one model for STORM, PALM and PAINT."""
    on_time: float = param(1.5, label="on time", unit="frames", min=0.01,
                           help="mean of an exponential on-time; switching happens "
                                "at any time within a frame")
    blinks: float = param(3.0, label="blinks", min=0.01,
                          help="mean number of blinks of a fluorophore within the "
                               "measurement, which sets the off time; with "
                               "activation 'every blink', the mean number before "
                               "it bleaches")
    off_time: Optional[float] = param(None, label="off time", unit="frames",
                                      min=0.01, advanced=True,
                                      help="mean of an exponential off-time; auto: "
                                           "from the number of blinks")
    bleaching: float = param(0.1, label="bleaching probability", min=0, max=1,
                             help="the probability to bleach after each blink: at "
                                  "most 1/p blinks on average, however long one "
                                  "measures (not used with 'every blink', where it "
                                  "is 1 / blinks)")
    activation: str = param("constant", label="activation",
                            choices=(("constant", "constant: ramped against bleaching"),
                                     ("decay", "decaying: all dark at the start"),
                                     ("all", "every blink until bleached, spread evenly"),
                                     ("always", "always on: a bead")),
                            advanced=True,
                            help="constant: the blinks are spread evenly over the "
                                 "measurement, as an activation raised while the "
                                 "fluorophores bleach keeps them; decaying: a fixed "
                                 "rate, so most blinks come early; every blink: each "
                                 "fluorophore shows all its blinks before it bleaches "
                                 "(1 / blinks per blink), whatever the number of "
                                 "frames, spread evenly -- SMAP's 'Dye' model; "
                                 "always on: on from the first frame to the last, "
                                 "never bleached, photons per frame")
    photons: float = param(5000.0, label="photons per blink", min=0,
                           help="mean over a whole blink; a frame gets its share "
                                "by the time the fluorophore was on in it")
    photons_std: float = param(2500.0, label="std", min=0, beside=True,
                               help="spread between blinks (a gamma distribution); "
                                    "0: every blink the same")


@dataclass
class OpticsSettings:
    """The PSF and the pixel, for the precision and for the frames."""
    pixelsize_nm: float = param(100.0, label="pixel size", unit="nm", min=1,
                                help="the camera pixel in the sample: enters the "
                                     "precision and sets the frames' scale")
    sigma_nm: float = param(130.0, label="PSF sigma", unit="nm", min=1,
                            help="the width (standard deviation) of the Gaussian "
                                 "spot in focus")
    astigmatism: bool = param(False, label="astigmatic",
                              help="a cylindrical lens: the spot's widths follow z")
    focal_offset_nm: float = param(300.0, label="focal offset", unit="nm",
                                   advanced=True,
                                   help="astigmatic: x is in focus at +this z, "
                                        "y at -this")
    depth_nm: float = param(400.0, label="focal depth", unit="nm", min=1, advanced=True,
                            help="astigmatic: how far from its focus a width has "
                                 "grown by sqrt(2)")
    calibration: str = param("", label="PSF calibration", kind="open_file",
                             file_filter="Calibrations (*_3dcal.mat *_3Dcal.mat *.h5);;"
                                         "All files (*)",
                             help="camera frames drawn with a measured spline PSF "
                                  "(a bead calibration) instead of a Gaussian: a fit "
                                  "then meets a PSF it did not make itself")


@dataclass
class LocalizationOutputSettings:
    """What a fit would have made of the frames."""
    min_photons: float = param(10.0, label="detection limit", unit="photons", min=0,
                               help="fewer photons than this in a frame and there is "
                                    "no localization; the dim ones that remain are "
                                    "kept, as SMAP keeps them, and Ground Truth leaves "
                                    "them out of its score")
    close: str = param("remove", label="close emitters",
                       choices=(("remove", "remove both"),
                                ("average", "one localization, averaged")),
                       help="emitters on in one frame closer than the separation "
                            "below are one spot: dropped, or fitted as one at "
                            "their photon-weighted mean")
    min_separation_nm: float = param(250.0, label="separation", unit="nm", min=0,
                                     help="closer than this in one frame, two "
                                          "emitters are one spot; 0: never")
    z_factor: float = param(3.0, label="z precision", unit="x lateral", min=0,
                            advanced=True,
                            help="the z precision as a multiple of the lateral one")
    emccd: bool = param(False, label="EMCCD", advanced=True,
                        help="localizations only: the excess noise of EM gain, "
                             "the precision worse by sqrt(2); the camera frames' "
                             "is EM gain there")


@dataclass
class CameraOutputSettings:
    """Camera frames, for a fitter to fit."""
    path: str = param("", label="simulation file", kind="save_file",
                      file_filter=SIMULATION_FILTER,
                      help="written as the recipe (*.sim.yaml); open it in a "
                           "fitter as its file, and the frames are made as they "
                           "are read")
    tiff: bool = param(False, label="also write a TIFF",
                       help="the frames themselves, beside the recipe, for other "
                            "software")
    size_px: int = param(100, label="size", unit="px", min=8,
                         help="the side of the square frame, which covers x and y "
                              "from 0 to this times the pixel size")
    conversion: float = param(0.5, unit="e-/ADU", min=1e-6,
                              help="electrons per camera count")
    offset: float = param(100.0, unit="ADU",
                          help="the camera count of a dark pixel")
    read_noise: float = param(1.5, label="read noise", unit="ADU", min=0,
                              help="Gaussian noise per pixel and frame, in counts")
    split_sigma_nm: Optional[float] = param(None, label="second half's PSF",
                                            unit="nm", min=1,
                                            help="split camera: the PSF width in the "
                                                 "second half, which sees another "
                                                 "wavelength; auto: the first's")
    em_gain: float = param(0.0, label="EM gain", min=0,
                           help="camera frames only.  0: an sCMOS; otherwise an "
                                "EMCCD with this gain, its multiplication noise "
                                "included (the excess factor of 2)")
    load_truth: bool = param(False, label="load the truth",
                             help="the true positions of every spot drawn, as a "
                                  "localization table")


@dataclass
class SimulationSettings:
    output: str = param("localizations", label="simulate",
                        choices=(("localizations", "localizations"),
                                 ("camera", "camera frames")),
                        help="a localization table directly, or camera frames "
                             "for a fitter to fit")
    n_frames: int = param(20000, label="frames", min=1,
                          help="the length of the measurement")
    frame_time_ms: float = param(10.0, label="frame time", unit="ms", min=0.001,
                                 advanced=True,
                                 help="what one frame is in seconds: recorded with "
                                      "the file, for what reads time")
    background: float = param(20.0, label="background", unit="photons/px", min=0,
                              help="per pixel and frame: in the precision, or "
                                   "drawn into the frames")
    background_std: float = param(0.0, label="std", min=0, beside=True,
                                  help="spread between localizations, or between "
                                       "frames for the camera; 0: always the same")
    drift: bool = param(False, label="add drift",
                        help="a random walk plus a slow creep of ~100 nm -- over "
                             "a long acquisition the walk is as large; the truth "
                             "is kept in the metadata as drift_truth")
    seed: int = param(0, label="seed", min=0,
                      help="the same seed and settings give the same simulation")
    structure: StructureSettings = param(default_factory=StructureSettings,
                                         label="structure")
    labelling: LabellingSettings = param(default_factory=LabellingSettings,
                                         label="labelling", collapsed=True)
    blinking: BlinkingSettings = param(default_factory=BlinkingSettings,
                                       label="blinking")
    optics: OpticsSettings = param(default_factory=OpticsSettings, label="optics",
                                   collapsed=True)
    localizations: LocalizationOutputSettings = param(
        default_factory=LocalizationOutputSettings, label="localizations",
        collapsed=True)
    camera: CameraOutputSettings = param(default_factory=CameraOutputSettings,
                                         label="camera frames", collapsed=True)


# ------------------------------------------------------------------ a scene

@dataclass
class PlacementSettings:
    """Where a structure's copies go (`smappy.simulate.structure`)."""
    mode: str = param("file", label="copies",
                      choices=(("file", "as the structure file says"),
                               ("copies", "set here"),
                               ("same", "where another component's are")),
                      help="the structure file's own copies; copies set here over "
                           "the scene's field; or the poses of another "
                           "component's copies, this one's geometry put there")
    n: int = param(0, label="number", min=0,
                   help="how many copies; 0: by the density")
    density_um2: float = param(0.5, label="density", unit="/µm²", min=0,
                               help="copies per µm² of the field, when the number is 0")
    rotation: str = param("random", label="rotation",
                          choices=(("none", "none"), ("random", "in the plane"),
                                   ("random_3d", "in 3D")),
                          help="how each copy is turned")
    tilt_deg: float = param(0.0, label="tilt", unit="°", min=0, max=90,
                            help="each copy tilted by up to this, in a random direction")
    jitter_nm: float = param(0.0, label="jitter", unit="nm", min=0,
                             help="each copy moved by up to this per axis")
    min_distance_nm: float = param(0.0, label="minimum distance", unit="nm", min=0,
                                   help="between the copies' centres")
    same_as: str = param("", label="component", choices=lambda: [("", "-")],
                         help="the component whose copies' places and turns "
                              "this one takes; it must have as many copies")

    def copies(self, field_nm: float) -> dict:
        """As a structure file's ``copies:`` block, over the scene's field."""
        out = {"field": [0.0, 0.0, field_nm, field_nm], "rotation": self.rotation,
               "tilt": self.tilt_deg, "jitter": self.jitter_nm,
               "min_distance": self.min_distance_nm}
        if self.n > 0:
            out["n"] = self.n
        else:
            out["density"] = self.density_um2
        return out


@dataclass
class MotionSettings:
    """Brownian motion of each label, frame to frame (no motion blur)."""
    diffusion: bool = param(False, label="diffusion", help="the labels diffuse")
    d_um2_s: float = param(0.5, label="D", unit="µm²/s", min=0,
                           help="the diffusion coefficient of the first state")
    slow_fraction: float = param(0.0, label="second state", min=0, max=1,
                                 help="the fraction of the time in a second state; "
                                      "0: one state")
    d2_um2_s: float = param(0.02, label="D of the second", unit="µm²/s", min=0,
                            help="the diffusion coefficient of the second state")
    switch_probability: float = param(0.0, label="switching", min=0, max=1,
                                      help="per frame: the state is drawn again")
    in_z: bool = param(False, label="in z too",
                       help="diffuse in 3D; off: in the plane, as in a membrane")


@dataclass
class ComponentSettings:
    """One kind of thing in the sample, with its own labelling, photophysics
    and motion: a structure, a homogeneous background, beads."""
    name: str = param("structure", label="name",
                      help="what the list calls it, and what another "
                           "component's 'same as' names")
    # not `on`: YAML reads a bare `on:` as true, and a hand-written preset broke
    enabled: bool = param(True, label="on", help="untick to leave it out")
    channel: int = param(1, label="colour", min=1,
                         help="the localizations' channel; a split camera sends "
                              "the ratio below of its light to the second half")
    split_ratio: float = param(0.5, label="split ratio", min=0, max=1,
                               help="split camera: the fraction of this "
                                    "component's photons in the second half")
    structure: StructureSettings = param(default_factory=StructureSettings,
                                         label="structure")
    placement: PlacementSettings = param(default_factory=PlacementSettings,
                                         label="placement", collapsed=True)
    labelling: LabellingSettings = param(default_factory=LabellingSettings,
                                         label="labelling", collapsed=True)
    photophysics_from: str = param("", label="photophysics as",
                                   choices=lambda: [("", "its own")],
                                   help="another component whose photophysics "
                                        "this one takes; empty: its own, below")
    blinking: BlinkingSettings = param(default_factory=BlinkingSettings,
                                       label="photophysics")
    motion: MotionSettings = param(default_factory=MotionSettings, label="motion",
                                   collapsed=True)


def preset_folders(kind: str) -> list:
    """Where saved scenes or components are: the user's, then the shipped."""
    from pathlib import Path
    from ..config import config_dir
    return [config_dir() / kind, Path(__file__).resolve().parent.parent / "data" / kind]


def default_components() -> list:
    return [ComponentSettings()]


@dataclass
class SceneSettings:
    """A scene: components, each with its own labelling, photophysics and
    motion, and what they share -- frames, background, optics, output."""
    preset: str = param("", label="preset", kind="preset",
                        store=lambda: preset_folders("simulation_presets"),
                        help="a scene to start from: it fills every setting, and "
                             "any can then be changed")
    output: str = param("localizations", label="simulate",
                        choices=(("localizations", "localizations"),
                                 ("camera", "camera frames"),
                                 ("split", "split camera")),
                        help="a localization table directly; camera frames for a "
                             "fitter; or a split camera, two halves, for the "
                             "two-colour fit")
    n_frames: int = param(20000, label="frames", min=1,
                          help="the length of the measurement")
    background: float = param(20.0, label="background", unit="photons/px", min=0,
                              help="per pixel and frame: in the precision, or "
                                   "drawn into the frames")
    background_std: float = param(0.0, label="std", min=0, beside=True,
                                  help="spread between localizations, or between "
                                       "frames for the camera; 0: always the same")
    drift: bool = param(False, label="add drift",
                        help="a random walk plus a slow creep of ~100 nm; the "
                             "truth is kept in the metadata as drift_truth")
    field_um: float = param(10.0, label="field", unit="µm", min=0.1, advanced=True,
                            help="the side of the square field copies set in a "
                                 "component, homogeneous labels and beads are in")
    frame_time_ms: float = param(10.0, label="frame time", unit="ms", min=0.001,
                                 advanced=True,
                                 help="what one frame is in seconds: recorded with "
                                      "the file, and the time diffusion has per frame")
    seed: int = param(0, label="seed", min=0, advanced=True,
                      help="the same seed and settings give the same simulation")
    components: List[ComponentSettings] = param(
        default_factory=default_components, label="components", collapsed=True,
        item_on="enabled",
        store=lambda: preset_folders("simulation_components"),
        help="the things in the sample; select one to set it")
    optics: OpticsSettings = param(default_factory=OpticsSettings, label="optics",
                                   collapsed=True)
    localizations: LocalizationOutputSettings = param(
        default_factory=LocalizationOutputSettings, label="localizations",
        collapsed=True)
    camera: CameraOutputSettings = param(default_factory=CameraOutputSettings,
                                         label="camera frames", collapsed=True)

    def component(self, name: str) -> "ComponentSettings":
        for c in self.components:
            if c.name == name:
                return c
        raise ValueError(f"no component {name!r}: the scene has "
                         + (", ".join(repr(c.name) for c in self.components) or "none"))
