"""A transformation of a file's coordinates, and applying one.

Wave 5's first topic (`docs/smap_porting/wave5_design.md`, "Topic 1"): one
idea under shifting, rotating, mirroring and scaling a file, the projective
map between two cameras, and another file's drift.  This module holds the
constant-in-time ones -- `Transformation` -- and the one place that applies
them to a table (`transform_table`); a drift curve stays a `drift.Drift`,
applied by `Drift.apply`, so that nothing applies a transformation twice in
code.  Measuring one is somebody else's job (*Calibrate transform*, *Register
layers*, the drift plugins); they hand what they measured to *Transform*.

Three kinds, by what they can do:

* **affine** -- a 4x4 matrix on ``(x, y, z, 1)`` in nm.  Shift, rotation,
  scale and mirror are all of this kind, and so is their composition, so the
  typed operations are multiplied into one matrix rather than kept as a list.
  The only kind that can move an image layer (`ImageData.placement`): an
  affine map of a pixel grid is still a grid, so nothing is resampled.
* **projective** -- a 3x3 matrix on ``(x, y, 1)`` in nm, z unchanged: the map
  of a `_2ct.h5` (*Calibrate transform*), converted from chip pixels.  A
  projective map bends the pixel grid's straight spacing, so an image would
  have to be resampled to follow it, and is refused instead.
* **polynomial** -- the third-order map of a `_2ct.h5` fitted that way, kept
  in chip pixels with the pixel size it is to be read with.

The rotation is SMAP's (TransformLocsManually): ``x' = cos(a) x + sin(a) y``,
``y' = -sin(a) x + cos(a) y``, which is counterclockwise as the picture shows
it, y pointing down.  The typed operations act about a centre ``c`` in the
order ``x' = R S M (x - c) + c + t``: mirror, scale, rotate, then shift --
SMAP shifted first, which made the centre a point of the *shifted* data, and
a centre one has to compute is not one anyone types.

What is not changed: precisions and widths (`xy_err_nm`, `sigma_nm`) stay as
they are, as in SMAP, also under a scale -- a scale here corrects where the
localizations are, not how well they were measured; the detection positions
(`peak_*`), the record of what was found where, as `drift._POSITION_COLUMNS`
leaves them.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .locs import Localizations

FORMAT = "smappy-transformation"
VERSION = 1
# the suffix the plugin proposes, after SMAP's `_T.mat`
SUFFIX = "_T.h5"
KINDS = ("affine", "projective", "polynomial")

# Another file's drift is for two cameras recording together, which see the
# same frames.  Their last localizations need not fall in the same frame --
# the last few frames of a movie may hold none in one channel -- so the last
# frames may differ by this fraction of the drift's frames (at least
# `FRAME_SLACK_MIN`); a larger difference is another acquisition.
FRAME_SLACK = 0.01
FRAME_SLACK_MIN = 10


@dataclass
class Transformation:
    """A map of x, y (and z) in nm.  See the module docstring for the kinds."""
    kind: str = "affine"
    # affine: (4, 4) on (x, y, z, 1); projective: (3, 3) on (x, y, 1)
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    # polynomial: the `calibrate.dual.Polynomial`, secondary -> reference in
    # chip pixels, read with ``pixelsize_nm`` (x, y)
    polynomial: Any = None
    pixelsize_nm: Optional[Tuple[float, float]] = None
    # what it is, in words, and where it came from: the typed values, a path
    description: str = ""
    origin: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind of transformation {self.kind!r}")
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        want = (4, 4) if self.kind == "affine" else (3, 3)
        if self.matrix.shape != want:
            raise ValueError(f"a {self.kind} transformation is a {want} matrix, "
                             f"not {self.matrix.shape}")
        if not np.isfinite(self.matrix).all():
            raise ValueError("the transformation is not finite")
        if self.kind == "polynomial" and (self.polynomial is None
                                          or self.pixelsize_nm is None):
            raise ValueError("a polynomial map needs its polynomial and the "
                             "pixel size it is read with")

    # ------------------------------------------------------------ applying
    def apply(self, x, y, z=None):
        """``(x, y, z)`` mapped; z is None when it was not given."""
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        if self.kind == "affine":
            m = self.matrix
            zz = np.zeros_like(x) if z is None else np.asarray(z, np.float64)
            nx = m[0, 0] * x + m[0, 1] * y + m[0, 2] * zz + m[0, 3]
            ny = m[1, 0] * x + m[1, 1] * y + m[1, 2] * zz + m[1, 3]
            nz = None if z is None else m[2, 0] * x + m[2, 1] * y + m[2, 2] * zz + m[2, 3]
            return nx, ny, nz
        if self.kind == "projective":
            from .calibrate.dual import map_points
            out = map_points(self.matrix, np.column_stack([x, y]))
            return out[:, 0], out[:, 1], None if z is None else np.asarray(z, np.float64)
        px, py = self.pixelsize_nm
        out = self.polynomial.forward(np.column_stack([x / px, y / py]))
        return (out[:, 0] * px, out[:, 1] * py,
                None if z is None else np.asarray(z, np.float64))

    def inverse(self) -> "Transformation":
        """The map back.  A polynomial's is the one fitted the other way
        (`calibrate.dual.Polynomial`), not an exact inverse."""
        description = f"inverse of {self.description}" if self.description else ""
        if self.kind == "polynomial":
            from .calibrate.dual import Polynomial
            q = self.polynomial
            back = Polynomial(q.order, q.inverse_centre, q.inverse_scale,
                              q.inverse_coef, q.forward_centre,
                              q.forward_scale, q.forward_coef)
            return Transformation("polynomial", np.linalg.inv(self.matrix), back,
                                  self.pixelsize_nm, description, dict(self.origin))
        return Transformation(self.kind, np.linalg.inv(self.matrix),
                              description=description, origin=dict(self.origin))

    @property
    def moves_z(self) -> bool:
        if self.kind != "affine":
            return False
        return not np.allclose(self.matrix[2], [0, 0, 1, 0]) \
            or not np.allclose(self.matrix[:2, 2], 0)

    def planar(self) -> np.ndarray:
        """The 3x3 affine of x and y, for moving an image (`ImageData.placement`).

        Refused for what an image cannot follow without being resampled --
        the projective and polynomial maps -- and for a move in z, which an
        image does not have.
        """
        if self.kind != "affine":
            raise ValueError(f"a {self.kind} map cannot move an image: it bends "
                             "the pixel grid, and the image would have to be "
                             "resampled.  Images take shift, rotation, scale, "
                             "mirror and affine maps")
        if self.moves_z:
            raise ValueError("an image has no z: this transformation moves "
                             "localizations in z")
        m = self.matrix
        return np.array([[m[0, 0], m[0, 1], m[0, 3]],
                         [m[1, 0], m[1, 1], m[1, 3]],
                         [0.0, 0.0, 1.0]])

    # ------------------------------------------------------------- showing
    def __str__(self) -> str:
        if self.kind != "affine":
            return self.description or f"{self.kind} map"
        a = self.matrix[:2, :2]
        angle = np.degrees(np.arctan2(a[0, 1], a[0, 0]))
        sx, sy = np.hypot(a[0, 0], a[1, 0]), np.hypot(a[0, 1], a[1, 1])
        flipped = np.linalg.det(a) < 0
        t = self.matrix[:3, 3]
        parts = [f"shift ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) nm"]
        if abs(angle) > 1e-9:
            parts.append(f"rotation {angle:.3f} deg")
        if abs(sx - 1) > 1e-9 or abs(sy - 1) > 1e-9:
            parts.append(f"scale {sx:.6g}" if abs(sx - sy) < 1e-9
                         else f"scale {sx:.6g} x {sy:.6g}")
        if flipped:
            parts.append("mirrored")
        return ", ".join(parts)

    # ------------------------------------------------------------- as data
    def to_dict(self) -> Dict[str, Any]:
        out = {"kind": self.kind, "matrix": self.matrix.tolist(),
               "description": self.description, "origin": dict(self.origin)}
        if self.kind == "polynomial":
            q = self.polynomial
            out["pixelsize_nm"] = [float(v) for v in self.pixelsize_nm]
            out["polynomial"] = {
                "order": int(q.order),
                "forward_centre": np.asarray(q.forward_centre, float).tolist(),
                "forward_scale": float(q.forward_scale),
                "forward_coef": np.asarray(q.forward_coef, float).tolist(),
                "inverse_centre": np.asarray(q.inverse_centre, float).tolist(),
                "inverse_scale": float(q.inverse_scale),
                "inverse_coef": np.asarray(q.inverse_coef, float).tolist()}
        return out

    @classmethod
    def from_dict(cls, saved: Dict[str, Any]) -> "Transformation":
        polynomial = None
        q = saved.get("polynomial")
        if q:
            from .calibrate.dual import Polynomial
            polynomial = Polynomial(
                int(q["order"]), np.asarray(q["forward_centre"], float),
                float(q["forward_scale"]), np.asarray(q["forward_coef"], float),
                np.asarray(q["inverse_centre"], float), float(q["inverse_scale"]),
                np.asarray(q["inverse_coef"], float))
        sizes = saved.get("pixelsize_nm")
        return cls(kind=str(saved.get("kind", "affine")),
                   matrix=np.asarray(saved["matrix"], float), polynomial=polynomial,
                   pixelsize_nm=None if sizes is None else tuple(float(v) for v in sizes),
                   description=str(saved.get("description") or ""),
                   origin=dict(saved.get("origin") or {}))


# ------------------------------------------------------------ the typed ones
def shift(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = dx, dy, dz
    return m


def rotation(angle_deg: float) -> np.ndarray:
    """SMAP's rotation: counterclockwise as the picture shows it (y down)."""
    a = np.radians(angle_deg)
    m = np.eye(4)
    m[:2, :2] = [[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]
    return m


def scaling(sx: float, sy: Optional[float] = None) -> np.ndarray:
    return np.diag([float(sx), float(sx if sy is None else sy), 1.0, 1.0])


MIRRORS = ("none", "x", "y", "both")


def mirroring(which: str) -> np.ndarray:
    """``x``: left-right (x -> -x), ``y``: up-down, ``both``."""
    if which not in MIRRORS:
        raise ValueError(f"mirror is one of {', '.join(MIRRORS)}, not {which!r}")
    return np.diag([-1.0 if which in ("x", "both") else 1.0,
                    -1.0 if which in ("y", "both") else 1.0, 1.0, 1.0])


def typed(shift_nm: Sequence[float] = (0.0, 0.0, 0.0), angle_deg: float = 0.0,
          scale: float = 1.0, scale_y: Optional[float] = None,
          mirror: str = "none", centre: Sequence[float] = (0.0, 0.0)
          ) -> Transformation:
    """``x' = R S M (x - c) + c + t``: mirror, scale and rotate about the
    centre ``c`` (x, y), then shift by ``t``."""
    if scale <= 0 or (scale_y is not None and scale_y <= 0):
        raise ValueError("a scale is positive; mirror to turn the data over")
    cx, cy = float(centre[0]), float(centre[1])
    dx, dy, dz = (list(shift_nm) + [0.0, 0.0, 0.0])[:3]
    m = (shift(cx + dx, cy + dy, dz) @ rotation(angle_deg)
         @ scaling(scale, scale_y) @ mirroring(mirror) @ shift(-cx, -cy))
    origin = {"shift_nm": [float(dx), float(dy), float(dz)],
              "angle_deg": float(angle_deg), "scale": float(scale),
              "scale_y": None if scale_y is None else float(scale_y),
              "mirror": mirror, "centre_nm": [cx, cy]}
    t = Transformation("affine", m, origin={"typed": origin})
    t.description = str(t)
    return t


def from_channel_transform(transform, pixelsize_nm: Tuple[float, float],
                           origin: Optional[Dict[str, Any]] = None
                           ) -> Transformation:
    """A `_2ct.h5`'s map (`calibrate.dual.ChannelTransform`), secondary to
    reference, in the nm of a table with this camera pixel.

    The projective map is ``S H S^-1`` with ``S`` the pixel size; one that
    turns out affine (its last row 0, 0, 1) is kept as affine, so it can move
    an image too.
    """
    px, py = float(pixelsize_nm[0]), float(pixelsize_nm[1])
    s = np.diag([px, py, 1.0])
    h = s @ np.asarray(transform.transformation, float) @ np.linalg.inv(s)
    h = h / h[2, 2]
    origin = dict(origin or {})
    if getattr(transform, "polynomial", None) is not None:
        return Transformation("polynomial", h, transform.polynomial, (px, py),
                              "third-order polynomial channel map", origin)
    if np.allclose(h[2, :2], 0.0, atol=1e-14):
        m = np.eye(4)
        m[:2, :2] = h[:2, :2]
        m[:2, 3] = h[:2, 2]
        t = Transformation("affine", m, origin=origin)
        t.description = str(t)
        return t
    return Transformation("projective", h, description="projective channel map",
                          origin=origin)


# ------------------------------------------------------------- the table
_LATERAL = (("x_nm", 0), ("y_nm", 1))


def transform_table(locs: Localizations, transformation: Transformation,
                    rows=None, name: str = "", what: str = "") -> Localizations:
    """A copy of ``locs`` with ``rows`` (a mask; None: all) transformed.

    x_nm, y_nm and, where the table has it, z_nm; a table that also keeps
    `x_pix` / `y_pix` (a file saved in pixels) has them moved with the nm
    ones.  The transformation is recorded in ``metadata["transformations"]``
    -- a list, since a file may be moved more than once -- under ``name``,
    with ``what`` saying which rows.
    """
    n = len(locs)
    mask = np.ones(n, bool) if rows is None else np.asarray(rows, bool)
    if mask.shape != (n,):
        raise ValueError("rows is a mask over the table")
    for column, _ in _LATERAL:
        if column not in locs:
            raise ValueError(f"the table has no {column}; it has "
                             + ", ".join(sorted(locs.keys())))
    columns = dict(locs.columns)
    x = np.asarray(locs["x_nm"], np.float64)[mask]
    y = np.asarray(locs["y_nm"], np.float64)[mask]
    z = np.asarray(locs["z_nm"], np.float64)[mask] if "z_nm" in locs else None
    nx, ny, nz = transformation.apply(x, y, z)
    moved = {"x_nm": nx - x, "y_nm": ny - y}
    for column, new in (("x_nm", nx), ("y_nm", ny), ("z_nm", nz)):
        if new is None:
            continue
        values = np.array(columns[column], copy=True)
        values[mask] = new
        columns[column] = values
    if "x_pix" in locs or "y_pix" in locs:
        from .locs import table_pixel_sizes
        sizes = table_pixel_sizes(locs)
        for (column, axis), pix in zip(_LATERAL, ("x_pix", "y_pix")):
            if pix in columns:
                values = np.array(columns[pix], copy=True)
                values[mask] = values[mask] + moved[column] / sizes[axis]
                columns[pix] = values
    metadata = dict(locs.metadata)
    record = {"time": datetime.now().isoformat(timespec="seconds"),
              "name": name, "rows": what, "n": int(mask.sum()),
              "transformation": transformation.to_dict()}
    metadata["transformations"] = list(metadata.get("transformations") or []) + [record]
    return Localizations(columns, metadata)


def check_same_frames(drift_frames: int, frames: np.ndarray, source: str = "the drift",
                      target: str = "the file") -> None:
    """Refuse another file's drift for a file of other frames.

    Another file's drift is for simultaneous acquisition: both files cover
    the same frames.  The target's frames must lie within the drift's, and
    its last frame must be within `FRAME_SLACK` of the drift's last.
    """
    frames = np.asarray(frames)
    if not len(frames):
        raise ValueError(f"{target} has no localizations")
    first, last = int(frames.min()), int(frames.max())
    slack = max(FRAME_SLACK_MIN, int(np.ceil(FRAME_SLACK * drift_frames)))
    if first < 0 or last >= drift_frames or last < drift_frames - 1 - slack:
        raise ValueError(
            f"{source} covers frames 0..{drift_frames - 1} and {target} frames "
            f"{first}..{last}: another file's drift is applied frame by frame "
            f"to a file of the same frames (two cameras recording together), "
            f"and these are different acquisitions")


# ---------------------------------------------------------------- the file
def save_transformation(path, transformation: Transformation, name: str = "",
                        overwrite: bool = False) -> Path:
    """Write ``transformation`` to its own file: an HDF5 file whose
    attributes say what it is and whose one dataset is the transformation as
    JSON (`Transformation.to_dict`), so a script reads it with h5py and json
    alone."""
    import h5py
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp",
                                     dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as f:
            f.attrs.update(format=FORMAT, version=VERSION, name=name,
                           kind=transformation.kind)
            f["transformation_json"] = json.dumps(transformation.to_dict())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def is_transformation_file(path) -> bool:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return f.attrs.get("format") == FORMAT
    except OSError:
        return False


def load_transformation(path, pixelsize_nm: Optional[Tuple[float, float]] = None
                        ) -> Transformation:
    """A transformation file -- this module's, or a `_2ct.h5` (a channel
    transformation of *Calibrate transform*, or a dual-colour bead
    calibration), which is in chip pixels and needs ``pixelsize_nm``."""
    import h5py
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no file {path}")
    if is_transformation_file(path):
        with h5py.File(path, "r") as f:
            if int(f.attrs.get("version", 0)) > VERSION:
                raise ValueError(f"{path.name} was written by a newer smappy")
            saved = json.loads(f["transformation_json"][()])
        t = Transformation.from_dict(saved)
        t.origin.setdefault("file", str(path))
        return t
    from .calibrate.transform import load_transform
    try:
        channel = load_transform(path)
    except (KeyError, ValueError, OSError) as e:
        raise ValueError(f"{path.name} is neither a transformation file nor a "
                         f"channel transformation (_2ct.h5): {e}") from None
    if pixelsize_nm is None:
        raise ValueError(f"{path.name} is in camera pixels, and the pixel size "
                         "to read it with is not known")
    return from_channel_transform(channel, pixelsize_nm, {"file": str(path)})
