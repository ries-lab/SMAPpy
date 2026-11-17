"""Simulated single-particle tracks: Brownian motion with diffusive states,
blinking, a localization precision and a density, the truth kept.

What the tracking linker (`smappy.tracking.link`) is tested and benchmarked
on.  Single-particle tracking here is fitted from single molecules, so the
regime that matters is sparse -- a particle's jump well below the distance
to its neighbours -- with frames where the fluorophore was dark, dim or
badly fitted: those are the frames a linker has to bridge.  So the model is:

* **particles** appear at a constant rate anywhere in a square field and
  last a geometric number of frames (bleaching), the
  rate set so that ``density_um2`` particles are present in a frame on
  average; the first frame starts with that many already there.  They are
  free to wander out of the field: the field is where they start;
* **diffusive states**: a coefficient per state, and per frame with
  ``switch_probability`` the state is drawn again from ``fractions`` (which
  may give the same one), so ``fractions`` is exactly the time spent in each
  and a state is left with ``switch_probability * (1 - fraction)`` per frame.
  A frame's step is drawn with the state the particle had in the frame
  before it: the state holds over the interval;
* **blinking**: a fluorophore goes dark with ``p_dark`` per frame and stays
  dark a geometric number of frames, ``dark_frames`` on average; a dark
  frame gives no localization, as a frame missed by the peak finder or
  dropped by a filter would.  A track starts bright;
* **localizations**: the true position plus a Gaussian error of
  ``precision_nm`` per axis (``z_precision_nm`` in z), recorded as
  ``xy_err_nm`` (and ``z_err_nm``); optionally spurious localizations at
  ``noise_density_um2`` per frame, uniform over the field, in no track;
* **the focus** (``focal_range_nm`` > 0): a particle is seen only while
  its z is within half that of the focal plane, as in 2D tracking with a
  thin focus (sptPALM).  Every particle then diffuses in z too, in a slab
  twice the focal range thick whose top and bottom are joined (periodic),
  so the particles in focus are spread evenly in z at every frame, as
  those of a cell are, and what leaves at the top comes back at the
  bottom as a particle of a neighbourhood would.  Each run of frames in
  focus is a track of its own: a particle that leaves is lost, and one
  that comes back is, to a linker, a new one.  This is the loss the jump
  fit's *axial range* corrects (`tracking.analysis.in_focus`): fast
  particles leave sooner, so they make fewer and shorter tracks.  With it
  ``density_um2`` counts the particles in focus, and ``track_frames`` is
  the photophysics' lifetime alone.

The truth is ``track_id_true`` (1, 2, ...; -1 for a spurious localization)
and ``state_true`` (the state at that frame, -1 for spurious).  No motion
blur and no two particles merged into one spot: what is simulated is what a
linker is given, not the camera.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional, Tuple

import numpy as np

from ..locs import Localizations


@dataclass
class TrackSimulationSettings:
    n_frames: int = 500
    field_um: float = 20.0                  # the side of the square field
    density_um2: float = 0.05               # particles present per um^2 in a frame
    track_frames: float = 20.0              # the mean frames a particle lasts
    frame_time_ms: float = 10.0
    d_um2_s: Tuple[float, ...] = (0.5,)     # a diffusion coefficient per state
    fractions: Tuple[float, ...] = (1.0,)   # the time spent in each state
    switch_probability: float = 0.0         # per frame: the state drawn again
    p_dark: float = 0.0                     # per bright frame: going dark
    dark_frames: float = 2.0                # the mean length of a dark run
    precision_nm: float = 20.0              # per lateral axis
    use_z: bool = False                     # 3D: diffusion and an error in z too
    z_range_nm: float = 800.0               # where particles start in z
    z_precision_nm: float = 50.0
    noise_density_um2: float = 0.0          # spurious localizations per um^2 per frame
    focal_range_nm: float = 0.0             # > 0: seen only within half this of focus
    seed: int = 0


def simulate_tracks(settings: Optional[TrackSimulationSettings] = None,
                    **values) -> Localizations:
    """A localization table of tracks: `TrackSimulationSettings`, or its
    fields as keywords (``simulate_tracks(density_um2=0.2, p_dark=0.1)``).
    Sorted by frame, as a fit writes it."""
    s = replace(settings or TrackSimulationSettings(), **values)
    d = np.atleast_1d(np.asarray(s.d_um2_s, float))
    fractions = np.atleast_1d(np.asarray(s.fractions, float))
    if len(fractions) != len(d):
        raise ValueError(f"{len(d)} diffusion coefficients but {len(fractions)} fractions")
    if (d < 0).any() or (fractions < 0).any() or fractions.sum() <= 0:
        raise ValueError("diffusion coefficients and fractions must not be negative")
    fractions = fractions / fractions.sum()
    if s.track_frames < 1 or s.dark_frames < 1:
        raise ValueError("track_frames and dark_frames are at least 1")
    rng = np.random.default_rng(s.seed)
    side = s.field_um * 1000.0
    area_um2 = s.field_um ** 2

    focus = float(s.focal_range_nm) > 0
    # with a focus, half the particles are in it at any frame (the slab is
    # twice as thick), so twice as many are made
    made = s.density_um2 * area_um2 * (2.0 if focus else 1.0)

    # particles: those present at the start, then a steady stream; geometric
    # lifetimes are memoryless, so the ones already there are drawn alike
    present = rng.poisson(made)
    arriving = rng.poisson(made / s.track_frames, max(s.n_frames - 1, 0))
    start = np.concatenate([np.zeros(present, np.int64),
                            np.repeat(np.arange(1, s.n_frames), arriving)])
    n = len(start)
    length = rng.geometric(1.0 / s.track_frames, n)
    length = np.minimum(length, s.n_frames - start)
    first = np.concatenate([[0], np.cumsum(length)[:-1]]) if n else np.zeros(0, np.int64)
    rows = int(length.sum())
    pid = np.repeat(np.arange(n), length)
    step = np.arange(rows) - first[pid]                    # frames since it appeared
    frame = start[pid] + step

    # the states and the blinking, one frame of every track at a time
    state = np.empty(rows, np.int64)
    dark = np.zeros(rows, bool)
    state[first] = rng.choice(len(d), n, p=fractions)
    q_on = 1.0 / s.dark_frames
    for k in range(1, int(length.max(initial=0))):
        at = first[length > k] + k
        redraw = rng.random(len(at)) < s.switch_probability
        state[at] = np.where(redraw, rng.choice(len(d), len(at), p=fractions), state[at - 1])
        flip = rng.random(len(at)) < np.where(dark[at - 1], q_on, s.p_dark)
        dark[at] = dark[at - 1] ^ flip

    # Brownian steps with the state of the frame before; the first row of
    # each particle is where it starts
    dims = 3 if s.use_z else 2
    walk = 3 if (s.use_z or focus) else 2               # z walks behind a 2D focus too
    half = s.focal_range_nm / 2.0
    z_start = s.focal_range_nm if focus else s.z_range_nm / 2   # the slab's half-thickness
    sigma = np.sqrt(2.0 * d * 1e6 * s.frame_time_ms * 1e-3)        # nm per axis
    moves = rng.normal(size=(rows, walk))
    moves[1:] *= sigma[state[:-1]][:, None]
    origin = np.column_stack([rng.uniform(0, side, (n, 2))]
                             + ([rng.uniform(-z_start, z_start, (n, 1))]
                                if walk == 3 else []))
    moves[first] = 0.0
    walked = np.cumsum(moves, axis=0)
    xyz = walked - walked[first][pid] + origin[pid]

    seen = ~dark
    track = pid
    if focus:
        # periodic in z over the slab; a run of frames in focus is a track
        xyz[:, 2] = np.mod(xyz[:, 2] + z_start, 2 * z_start) - z_start
        inside = np.abs(xyz[:, 2]) <= half
        begins = inside.copy()
        begins[1:] &= ~inside[:-1] | (pid[1:] != pid[:-1])
        track = np.cumsum(begins) - 1
        seen &= inside
    frame, track, state, xyz = frame[seen], track[seen], state[seen], xyz[seen, :dims]
    if focus:
        # numbered 1, 2, ... over the runs that left a localization
        track = np.unique(track, return_inverse=True)[1].reshape(-1)
    error = np.array([s.precision_nm, s.precision_nm, s.z_precision_nm][:dims])
    xyz = xyz + rng.normal(size=xyz.shape) * error
    truth = track + 1
    n_tracks = int(truth.max(initial=0)) if focus else n

    n_noise = rng.poisson(s.noise_density_um2 * area_um2 * s.n_frames)
    if n_noise:
        frame = np.concatenate([frame, rng.integers(0, s.n_frames, n_noise)])
        noise = np.column_stack([rng.uniform(0, side, (n_noise, 2))]
                                + ([rng.uniform(-s.z_range_nm / 2, s.z_range_nm / 2,
                                                (n_noise, 1))] if s.use_z else []))
        xyz = np.concatenate([xyz, noise])
        truth = np.concatenate([truth, np.full(n_noise, -1, np.int64)])
        state = np.concatenate([state, np.full(n_noise, -1)])

    order = np.argsort(frame, kind="stable")
    m = len(order)
    columns = {"frame": frame[order].astype(np.int64),
               "x_nm": xyz[order, 0].astype(np.float32),
               "y_nm": xyz[order, 1].astype(np.float32),
               "xy_err_nm": np.full(m, s.precision_nm, np.float32)}
    if s.use_z:
        columns["z_nm"] = xyz[order, 2].astype(np.float32)
        columns["z_err_nm"] = np.full(m, s.z_precision_nm, np.float32)
    columns["track_id_true"] = truth[order].astype(np.int64)
    columns["state_true"] = state[order].astype(np.int32)
    metadata = {"units": "nm", "simulation": "smappy tracks", "seed": s.seed,
                "n_frames": s.n_frames, "n_tracks": n_tracks, "n_localizations": m,
                # where every reader of seconds looks (`metadata.file_frame_time`)
                "camera": {"frame_time_ms": s.frame_time_ms},
                "track_simulation": _plain(asdict(s))}
    return Localizations(columns, metadata)


def _plain(values: dict) -> dict:
    """The settings with tuples as lists, as a saved file keeps them."""
    return {k: list(v) if isinstance(v, tuple) else v for k, v in values.items()}


def jump_nm(d_um2_s: float, frame_time_ms: float, dims: int = 2) -> float:
    """The root-mean-square jump in one frame of diffusion at ``d_um2_s``:
    ``sqrt(2 dims D t)``, nm."""
    return float(np.sqrt(2.0 * dims * d_um2_s * 1e6 * frame_time_ms * 1e-3))


def spacing_nm(density_um2: float) -> float:
    """The mean distance to the nearest neighbour of points at this density
    (per um^2), ``1 / (2 sqrt(density))``, nm: what a jump is compared with."""
    return float(500.0 / np.sqrt(density_um2)) if density_um2 > 0 else float("inf")
