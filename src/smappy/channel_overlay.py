"""Two channels of a split camera put on top of each other, one dye in each.

The other two-channel workflow.  In a ratiometric experiment every molecule is
seen in both halves of the chip, and the two-colour fitters pair the spots and
fit one emitter (`multichannel.MultiChannelEngine`).  Here each dye is seen in
one half only -- spectrally well separated, no crosstalk to speak of -- so
there is nothing to pair.  The movie is fitted as *two channels*, each half on
its own (`multichannel.SplitEngine`, ``camera_channel``), and the secondary
half's localizations are then moved onto the main half by a registration
measured on beads, which *are* seen in both (`calibrate.transform`).

This module is that last step, on arrays and tables, and where the
registration comes from:

* `overlay_table` moves the ``camera_channel == 1`` rows through a
  `ChannelTransform` (chip pixels, read with the table's pixel size by
  `transformation.from_channel_transform`), and writes ``channel`` -- the
  dye -- from ``camera_channel``.  Through `transformation.transform_table`,
  so the move is recorded in the table's metadata like any other.
* `fit_bead_movies` fits bead acquisitions with Gaussian 2D on *two channels*,
  in the arrangement the sample was fitted with; `plugins.registration.
  register_table` registers that table, as *Calibrate transform* does.
* `bead_key` / `cached_transform`: a registration saved beside the beads
  remembers which files it was measured on, so a batch over fifty samples
  registers the bead folder once.
* `refine`: the beads were imaged before or after the sample, and between the
  two the splitter, the focus or the temperature may have moved the channels by
  some tens of nm.  A little crosstalk -- a bright molecule of one dye leaking
  a few percent of its light into the other half -- is a pair measured *in the
  sample*, and enough of them measure that change.  They cannot re-measure the
  field distortion (a few hundred dim pairs do not carry a cubic over the
  field), so they correct the bead map by a shift or an affine map on top of
  it, never replace it, and the correction is applied only when it is larger
  than its own error (`Refinement.significant`).

Coordinates are chip pixels, as in `calibrate.transform`, until the table is
moved.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from .locs import Localizations

#: the name the move is recorded under in ``metadata["transformations"]``; a table that has it is
#: not moved again
RECORD_NAME = "overlay channels"
SECONDARY = 1
#: how many times the correction is refitted on resampled pairs for its error
BOOTSTRAP = 200
#: a correction is applied when it moves the field by more than this many times its error.
#: Three, not two: for a shift that is only noise, its size over its error in two dimensions
#: passes 2 in 14 % of samples (exp(-2^2/2)) and 3 in 1 %.
SIGNIFICANCE = 3.0
#: at most this many pair-and-fit rounds for the crosstalk correction (`refine`) ...
PAIRING_ROUNDS = 10
#: ... which stop when a round moves the correction by less than this, px
CONVERGED_PX = 1e-3


# ----------------------------------------------------------------- the table
def secondary_rows(locs: Localizations) -> np.ndarray:
    """The localizations of the secondary half (``camera_channel == 1``)."""
    if "camera_channel" not in locs:
        raise ValueError("the table was not fitted as two channels: fit it with channels > "
                         "read: two channels, so that each localization keeps the half it "
                         "was found in (camera_channel)")
    return np.asarray(locs["camera_channel"]) == SECONDARY


def is_overlaid(locs: Localizations) -> bool:
    """True when the table's secondary half has already been moved."""
    return any(record.get("name") == RECORD_NAME
               for record in locs.metadata.get("transformations") or [])


def overlay_table(locs: Localizations, transform, correction: Optional[np.ndarray] = None,
                  what: str = "") -> Localizations:
    """``locs`` with its secondary half moved onto the main half, and ``channel`` written.

    ``transform`` is a `ChannelTransform` (secondary to main, chip pixels);
    ``correction`` a 3x3 in chip pixels applied after it (`refine`), or None.
    Refused for a table already moved: twice is never what is meant.
    """
    from .locs import table_pixel_sizes
    from .transformation import Transformation, from_channel_transform, transform_table
    if is_overlaid(locs):
        raise ValueError("the secondary channel has already been put onto the main one; "
                         "undo that first to overlay it again")
    rows = secondary_rows(locs)
    sizes = table_pixel_sizes(locs)
    moved = transform_table(locs, from_channel_transform(transform, sizes), rows,
                            name=RECORD_NAME, what=what or "camera_channel 1")
    if correction is not None and not np.allclose(correction, np.eye(3)):
        # the same correction in nm: S C S^-1, an affine of x and y with z kept
        s = np.diag([sizes[0], sizes[1], 1.0])
        c = s @ np.asarray(correction, float) @ np.linalg.inv(s)
        m = np.eye(4)
        m[:2, :2] = c[:2, :2]
        m[:2, 3] = c[:2, 2]
        t = Transformation("affine", m, description="crosstalk correction")
        moved = transform_table(moved, t, rows, name=RECORD_NAME,
                                what="camera_channel 1, crosstalk correction")
    columns = dict(moved.columns)
    # the dye: in this workflow, the half it was seen in
    columns["channel"] = np.asarray(locs["camera_channel"]).astype(np.int32)
    return Localizations(columns, moved.metadata)


def check_layout(locs: Localizations, transform) -> Optional[str]:
    """Why the registration may not be this table's, or None: the two record
    where on the chip their channels are, and they should agree."""
    recorded = locs.metadata.get("channels")
    theirs = (transform.geometry or {}).get("channels")
    if not recorded or not theirs:
        return None
    if [list(r) for r in recorded.get("regions", [])] != [list(r) for r in theirs.get("regions", [])]:
        return (f"the table's channels lie at {recorded.get('regions')} on the chip and the "
                f"registration's at {theirs.get('regions')}: fit the sample and the beads in "
                "the same arrangement")
    return None


# ------------------------------------------------------------------- crosstalk
@dataclass
class Refinement:
    """A correction on top of the bead registration, from crosstalk pairs in the sample."""
    model: str
    matrix: np.ndarray               # 3x3, chip pixels, applied after the bead map
    n_pairs: int
    size_px: float                   # how far it moves the field at most, at its corners
    error_px: float                  # the bootstrap error of that
    reference: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    before: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))   # residuals, (n, 2)
    after: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    reason: str = ""

    @property
    def significant(self) -> bool:
        return (self.n_pairs > 0 and not self.reason
                and self.size_px > SIGNIFICANCE * self.error_px)

    @property
    def applied(self) -> np.ndarray:
        """The correction to apply: the matrix where it is significant, else none."""
        return self.matrix if self.significant else np.eye(3)


def crosstalk_pairs(x, y, frame, channel, mapping: Callable, tolerance_px: float,
                    photons=None, max_ratio: Optional[float] = None):
    """``(main ids, secondary ids)``: a molecule and its crosstalk, in the same frame, where
    ``mapping`` (secondary to main, chip pixels) puts them within ``tolerance_px``.

    Two molecules of different dyes can also meet within a pixel in one frame.  Crosstalk is a
    few percent of a molecule's light, so a pair whose dimmer partner has more than
    ``max_ratio`` of the brighter one's photons is taken for such a meeting and left out.
    """
    from .calibrate.transform import _pair
    x, y = np.asarray(x, float), np.asarray(y, float)
    frame, channel = np.asarray(frame), np.asarray(channel)
    order = np.argsort(frame, kind="stable")
    ref, sec = _pair(x[order], y[order], frame[order], mapping, channel[order] == 0,
                     channel[order] == SECONDARY, tolerance_px)
    ref, sec = order[ref], order[sec]
    if photons is not None and max_ratio is not None and len(ref):
        p = np.asarray(photons, float)
        a, b = p[ref], p[sec]
        ratio = np.minimum(a, b) / np.maximum(np.maximum(a, b), 1e-12)
        keep = ratio <= max_ratio
        ref, sec = ref[keep], sec[keep]
    return ref, sec


def fit_correction(reference, mapped, model: str = "shift", loss_scale: float = 0.1,
                   weights=None) -> np.ndarray:
    """The 3x3 that takes ``mapped`` (secondaries through the bead map) onto ``reference``.

    A shift or an affine map, by least squares reweighted with soft-L1 (as `calibrate.dual`'s
    refinement) so that a few pairs that are two molecules after all do not pull it.
    """
    reference, mapped = np.asarray(reference, float), np.asarray(mapped, float)
    n = len(reference)
    w0 = np.ones(n) if weights is None else np.asarray(weights, float)
    centre = mapped.mean(axis=0)
    design = (np.ones((n, 1)) if model == "shift"
              else np.c_[mapped - centre, np.ones(n)])
    target = reference - (mapped if model == "shift" else centre)
    w = w0.copy()
    for _ in range(6):
        sw = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(design * sw, target * sw, rcond=None)
        residual = np.linalg.norm(design @ coef - target, axis=1)
        w = w0 / np.sqrt(1.0 + (residual / max(loss_scale, 1e-6)) ** 2)
    out = np.eye(3)
    if model == "shift":
        out[:2, 2] = coef[0]
        return out
    a = coef[:2].T                     # (2, 2): reference = a (m - c) + c + t
    out[:2, :2] = a
    out[:2, 2] = coef[2] + centre - a @ centre
    return out


def _moves(matrix, probes) -> np.ndarray:
    """How far ``matrix`` moves each probe point, px."""
    from .calibrate.dual import map_points
    return np.linalg.norm(map_points(matrix, probes) - probes, axis=1)


def refine(x, y, frame, channel, transform, model: str = "shift", tolerance_px: float = 1.0,
           photons=None, max_ratio: Optional[float] = 0.3, min_pairs: int = 50,
           precision=None, seed: int = 1729) -> Refinement:
    """A correction of the bead registration from the sample's own crosstalk (module docstring).

    ``x``, ``y`` are chip pixels of a two-channel fit, ``frame`` kept apart per file
    (`locs.frames_apart`), ``channel`` the ``camera_channel``.  Its size is how far it moves the
    corners of the paired region, and its error the spread of that over refits on resampled
    pairs -- so a shift of 3 nm measured on twenty pairs is reported as the noise it is.
    """
    from .calibrate.dual import map_points
    from .calibrate.transform import pair_weights
    if model not in ("shift", "affine"):
        raise ValueError(f"no correction {model!r}: shift or affine")
    x, y = np.asarray(x, float), np.asarray(y, float)
    need = max(int(min_pairs), 3 if model == "affine" else 1)
    loss = max(tolerance_px / 4, 0.05)
    matrix = np.eye(3)
    # Paired around where the map puts a partner, the pairing distance cuts the far side of
    # whatever offset is there, and the correction comes out short -- by 4.6 of 30 nm at a
    # pairing distance of 1 px on the page's simulation, five times its error.  So the pairs
    # are found again through the corrected map until they stop changing.
    for _ in range(PAIRING_ROUNDS):
        def mapping(xy, matrix=matrix):
            return map_points(matrix, transform.to_reference(xy))
        ref, sec = crosstalk_pairs(x, y, frame, channel, mapping, tolerance_px, photons,
                                   max_ratio)
        if len(ref) < need:
            return Refinement(model, np.eye(3), len(ref), 0.0, 0.0,
                              reason=f"{len(ref)} crosstalk pairs within {tolerance_px:g} px, "
                                     f"{need} wanted")
        reference = np.c_[x[ref], y[ref]]
        mapped = transform.to_reference(np.c_[x[sec], y[sec]])
        weights = pair_weights(precision, ref, sec)
        again = fit_correction(reference, mapped, model, loss, weights)
        lo, hi = reference.min(axis=0), reference.max(axis=0)
        corners = np.array([[lo[0], lo[1]], [hi[0], hi[1]]])
        moved = float(np.max(np.linalg.norm(map_points(again, corners)
                                            - map_points(matrix, corners), axis=1)))
        matrix = again
        if moved < CONVERGED_PX:
            break
    lo, hi = reference.min(axis=0), reference.max(axis=0)
    probes = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]],
                       (lo + hi) / 2])
    rng = np.random.default_rng(seed)
    spread = []
    for _ in range(BOOTSTRAP):
        take = rng.integers(0, len(ref), len(ref))
        again = fit_correction(reference[take], mapped[take], model, loss,
                               None if weights is None else weights[take])
        spread.append(map_points(again, probes))
    spread = np.asarray(spread)                                 # (B, probes, 2)
    error = float(np.max(np.linalg.norm(spread.std(axis=0), axis=1)))
    return Refinement(model, matrix, len(ref), float(np.max(_moves(matrix, probes))), error,
                      reference, reference - mapped, reference - map_points(matrix, mapped))


# ---------------------------------------------------------------- the beads
def bead_files(path) -> List[Path]:
    """The bead acquisitions at ``path``: the movie itself, or every one in a folder."""
    from .calibrate.input import discover_acquisitions
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no bead movies at {path}")
    found = discover_acquisitions([path], recursive=False)
    if not found:
        raise ValueError(f"no image stacks in {path}")
    return [Path(p) for p in found]


def bead_key(paths: Sequence[Path], **settings) -> dict:
    """What a registration measured on these beads depends on: the files as they are now (size
    and time of change) and the settings it was measured with."""
    files = []
    for p in paths:
        p = Path(p)
        st = os.stat(p)
        files.append({"path": str(p.resolve()), "size": int(st.st_size),
                      "mtime_ns": int(st.st_mtime_ns)})
    return {"files": files, "settings": _plain(settings)}


def _plain(value):
    """``value`` as JSON has it, so that a key read back compares equal to one made."""
    import json
    from dataclasses import asdict, is_dataclass
    if is_dataclass(value):
        value = asdict(value)
    return json.loads(json.dumps(value, default=str))


def transform_path(path) -> Path:
    """Where the registration of the beads at ``path`` is kept: ``<name>_2ct.h5`` beside the
    movie or the folder -- not inside the folder, which stays images only."""
    path = Path(path)
    stem = path.name.split(".")[0] if path.is_file() else path.name
    return path.parent / f"{stem}_2ct.h5"


def cached_transform(path, key: dict):
    """The registration saved at ``path`` if it was measured on the same beads with the same
    settings (`bead_key`), else None."""
    from .calibrate.transform import load_transform
    path = Path(path)
    if not path.exists():
        return None
    try:
        transform = load_transform(path)
    except (OSError, ValueError, KeyError):
        return None
    return transform if transform.parameters.get("beads") == _plain(key) else None


def fit_bead_movies(paths: Sequence[Path], camera, channels, cutoff: float = 1.7,
                    max_frames: Optional[int] = None, report=None, stop=None) -> Localizations:
    """The bead movies fitted as two channels, one table, a ``filenumber`` per movie so that
    their frames are never paired across movies (`locs.frames_apart`).

    ``camera`` is a fitter's `CameraSettings` (the sample's, so the beads are read alike) and
    ``channels`` its `ChannelsSettings` or an `ArrangementSettings`: where the halves lie.
    Gaussian 2D whatever fitted the sample: a bead's centre is all a 2D registration needs, and a
    bead is bright enough for any model.
    """
    from dataclasses import replace
    from .locs import concat
    from .plugins import Context
    from .plugins.after_fit import AfterFitSettings
    from .plugins.fit import (ChannelsSettings, DetectionSettings, GaussianFit,
                              GaussianFitSettings, OutputSettings, SourceSettings)
    split = ChannelsSettings(split="two channels", layout=channels.layout,
                             main_channel=channels.main_channel,
                             split_position=channels.split_position)
    nothing = AfterFitSettings(astigmatic_z=False, fourpi_z=False, drift="none",
                               assign_colors=False, chain="")
    tables = []
    for k, path in enumerate(paths):
        if report:
            report(f"fitting beads: {Path(path).name} ({k + 1} of {len(paths)})")
        settings = GaussianFitSettings(
            source=SourceSettings(path=str(path), stop=max_frames),
            camera=replace(camera), channels=split,
            detection=DetectionSettings(cutoff_mode="dynamic", cutoff=float(cutoff)),
            output=OutputSettings(save=False, raw_frames=0, show_tags=""))
        settings.finish = nothing
        locs = GaussianFit().run(Context(progress=report, stop=stop), settings).locs
        if locs is None or not len(locs):
            continue
        locs.columns["filenumber"] = np.full(len(locs), k, np.int32)
        tables.append(locs)
    if not tables:
        raise ValueError("no beads found in " + ", ".join(Path(p).name for p in paths))
    out = concat(tables) if len(tables) > 1 else tables[0]
    out.metadata.setdefault("source", str(paths[0]))
    return out


def camera_of(locs: Localizations, given: Optional[dict] = None):
    """The `CameraSettings` to read the beads with: ``given`` (a fitter's own, filled when this
    runs as its step), else what the sample's table recorded of its camera, else nothing --
    the bead files' metadata and the databases, as for any fit."""
    from dataclasses import fields
    from .plugins.fit import CameraSettings
    known = {f.name for f in fields(CameraSettings)}
    values = dict(given or {})
    if not values:
        recorded = locs.metadata.get("camera") or {}
        for name in ("conversion", "offset", "em_on", "emgain", "read_noise_e"):
            if recorded.get(name) is not None:
                values[name] = recorded[name]
        pixel = recorded.get("pixelsize_um")
        if isinstance(pixel, (list, tuple)):
            values["pixelsize_um"], values["pixelsize_y_um"] = float(pixel[0]), float(pixel[1])
        elif pixel is not None:
            values["pixelsize_um"] = float(pixel)
    if "pixelsize_um" not in values and "pixelsize_um" not in (given or {}):
        from .locs import table_pixel_sizes
        try:
            px, py = table_pixel_sizes(locs)
            values["pixelsize_um"] = px / 1e3
            if py != px:
                values["pixelsize_y_um"] = py / 1e3
        except Exception:
            pass
    return CameraSettings(**{k: v for k, v in values.items() if k in known})
