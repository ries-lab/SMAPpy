"""What a training is: the simulation, the camera, and the parts, each by name.

A model file records all of it, so that a benchmark report says exactly what was trained
(`docs/decode/benchmark.md`, "A run"); the benchmark's run files name the same parts.  The defaults
are version 1: DECODE's method as published, on the frames DECODE trains on (40 x 40, 3 frames),
with the example package's ranges -- what the port's `decode_B` was trained with.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SimulationSettings:
    """How training frames are drawn: a window of frames, emitters that blink over it, a background
    per frame.  Fresh for every batch (decided 2026-11-12): no pool of frames to overfit."""
    frame_px: int = 40                        # square frames, DECODE's size
    window: int = 3                           # frames a sample holds; the middle one is localized
    margin_px: float = 0.0                    # emitters drawn this far beyond the frame (DECODE: 0)
    # molecules on in a frame, per 40 x 40 pixels: a density, so that frames of another size hold as
    # many per area (DECODE's emitter_av is per frame, of its 40 x 40; the example package's 50)
    density_per_40px: float = 50.0
    lifetime_frames: float = 1.0              # mean on-time, exponential (DECODE's lifetime_avg)
    photons_mean: float = 7000.0              # flux while on, a full frame's photons
    photons_sd: float = 3000.0
    photons_min: float = 0.0                  # the flux drawn no lower (DECODE's intensity_th, unset there)
    background: tuple = (20.0, 200.0)         # photons a pixel, uniform per frame
    z_nm: tuple = (-800.0, 800.0)
    target_photons_min: float = 0.0           # a target is a molecule with at least this many photons in the frame


@dataclass
class CameraModel:
    """The camera the frames are drawn for, and back to photons as the fitter converts them
    (`CameraMetadata.adu_to_photons`), so the network sees what it will be given."""
    conversion: float = 5.0                   # e- per ADU
    offset: float = 398.6                     # ADU
    em_gain: float = 100.0                    # 0: no EM register
    read_noise_e: float = 58.8
    qe: float = 1.0
    pixel_nm: tuple = (117.0, 127.0)          # x (columns), y (rows): for the metrics and the output

    @property
    def adu_to_photons(self) -> float:
        return self.conversion / self.em_gain if self.em_gain else self.conversion


@dataclass
class Part:
    """One part of the method, by name, with its settings."""
    name: str
    settings: dict = field(default_factory=dict)


@dataclass
class Parts:
    """The swappable parts (`docs/decode/benchmark.md`): version 1's are DECODE's."""
    backbone: Part = field(default_factory=lambda: Part("unet2", {"features": 48, "depth_frame": 2, "depth_union": 2}))
    time_model: Part = field(default_factory=lambda: Part("frame_stack"))
    output: Part = field(default_factory=lambda: Part("offsets"))
    extraction: Part = field(default_factory=lambda: Part("lookahead"))
    loss: Part = field(default_factory=lambda: Part("mixture", {"terms": ["count", "mixture", "background"]}))
    # DECODE's: AdamW, the rate times 0.9 every 10 epochs, gradients clipped (the norm 0.03 is
    # recalled from DECODE's training, not read: a setting to revisit if training is unstable);
    # an epoch is 10 000 samples, DECODE's pseudo_ds_size.  The rate is the port's decode_B's.
    schedule: Part = field(default_factory=lambda: Part("adamw_step", {
        "lr": 6e-4, "weight_decay": 0.1, "step_epochs": 10, "gamma": 0.9, "clip_norm": 0.03,
        "batch": 64, "samples_per_epoch": 10000, "epochs": 1000, "test_samples": 512, "threshold": 0.5}))


@dataclass
class TrainingSettings:
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    camera: CameraModel = field(default_factory=CameraModel)
    parts: Parts = field(default_factory=Parts)
    calibration: str | None = None
    test_set: str | None = None               # a fixed test set (`testset.py`); none: drawn from the seed
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
