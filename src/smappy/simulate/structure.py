"""What is labelled: a structure, as label positions in nm.

One description serves both simulators, the localizations and the camera
frames, and it is a YAML file rather than code, so a structure someone needs
is written, not programmed.  A file is a list of *elements*, each one kind of
geometry, and optionally *copies* of the whole:

.. code-block:: yaml

    elements:
      - points: [[0, 0, 0], [12, 5, 0]]     # label positions, nm
      - file: labels.csv                    # x, y[, z][, dye] per row, nm
      - line: [[0, 0, 0], [1000, 0, 0]]     # a polyline, `density` per nm
        density: 0.05
        width: 15                           # labels scatter by this about it
      - circle: 50                          # the radius; `centre`, `normal`
        density: 0.2
      - polygon: [[0, 0], [500, 0], [0, 500]]   # an area, `density` per nm^2
        density: 0.001
        z: [-100, 100]                      # spread evenly over this z range
      - image: actin.png                    # grey values scale the density
        pixelsize: 10                       # nm per image pixel
        density: 0.01                       # per nm^2 at the brightest pixel
    copies:                                 # optional
      n: 300                                # or  density: 0.5  (per um^2)
      field: [0, 0, 10000, 10000]           # x0, y0, x1, y1 in nm
      placement: random                     # random | grid
      rotation: random                      # none | random | random_3d
      min_distance: 300                     # between copies, random placement

Every element takes ``dye`` (default 1) and ``width``.  Explicit positions
(``points``, ``file``) are the same in every copy; the geometric elements are
sampled afresh for each, Poisson in number, so no two copies of a filament
network are alike.  A structure made of different things -- pores and some
scattered background -- is ``parts:``, a list of such blocks, each with its
own copies.  Paths are relative to the YAML.

This is the geometry only: which labels carry a fluorophore, and how many, is
the labelling (`smappy.simulate.kinetics.label`), set with the simulation and
not with the structure, because the same structure is labelled well or badly.

The built-in structures are YAML files like any other, in
``smappy/data/structures``, so each is also an example of the syntax.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

PRESETS_DIR = Path(__file__).resolve().parent.parent / "data" / "structures"
KINDS = ("points", "file", "line", "circle", "polygon", "image")


@dataclass
class Labels:
    """Label positions and what each belongs to."""
    xyz: np.ndarray          # (n, 3) nm
    dye: np.ndarray          # (n,) int, 1-based
    copy: np.ndarray         # (n,) int, which copy of the structure, 0-based

    def __len__(self) -> int:
        return len(self.xyz)


def presets() -> List[str]:
    """The names of the built-in structures."""
    return sorted(p.name[:-len(".yaml")] for p in PRESETS_DIR.glob("*.yaml"))


def preset_choices() -> List[tuple]:
    """``(preset, its name)`` for a menu: the YAML's own ``name``, which says
    what the structure is where the file name only labels it."""
    import yaml
    out = []
    for preset in presets():
        try:
            with open(PRESETS_DIR / f"{preset}.yaml", encoding="utf-8") as fh:
                name = (yaml.safe_load(fh) or {}).get("name") or preset
        except (OSError, yaml.YAMLError):
            name = preset
        out.append((preset, f"{name} ({preset})" if name != preset else preset))
    return out


def load_structure(source: Union[str, Path, Dict[str, Any]]) -> "Structure":
    """A structure from a preset name, a YAML path or an already parsed dict."""
    if isinstance(source, dict):
        return Structure(source, Path.cwd())
    text = str(source)
    path = Path(text)
    if not path.suffix and (PRESETS_DIR / f"{text}.yaml").exists():
        path = PRESETS_DIR / f"{text}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no structure {text!r}: neither a file nor one of "
                                f"the built-in {', '.join(presets())}")
    import yaml
    with open(path, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh) or {}
    if not isinstance(spec, dict):
        raise ValueError(f"{path}: a structure is a mapping with `elements`")
    return Structure(spec, path.parent)


class Structure:
    """A parsed structure file; `sample` draws its labels."""

    def __init__(self, spec: Dict[str, Any], base: Path):
        self.spec, self.base = spec, Path(base)
        self.name = str(spec.get("name", ""))
        parts = spec.get("parts")
        if parts is None:
            parts = [spec]
        elif not isinstance(parts, list):
            raise ValueError("`parts` is a list of blocks, each with `elements`")
        self.parts = []
        for part in parts:
            elements = part.get("elements")
            if not isinstance(elements, list) or not elements:
                raise ValueError("a structure needs a list of `elements`")
            # read the files now: a missing one should fail when the structure
            # is chosen, not a thousand copies later
            self.parts.append(([_Element(e, self.base) for e in elements],
                               part.get("copies")))

    def sample(self, rng) -> Labels:
        xyz, dye, copy = [], [], []
        n_copies = 0
        for elements, copies in self.parts:
            placements = _placements(copies, rng)
            for position, rotation in placements:
                one = [e.sample(rng) for e in elements]
                pts = np.vstack([p for p, _ in one]) if one else np.zeros((0, 3))
                if rotation is not None:
                    pts = pts @ rotation.T
                xyz.append(pts + position)
                dye.append(np.concatenate([d for _, d in one]))
                copy.append(np.full(len(pts), n_copies))
                n_copies += 1
        if not xyz:
            return Labels(np.zeros((0, 3)), np.zeros(0, np.int32), np.zeros(0, np.int32))
        return Labels(np.vstack(xyz), np.concatenate(dye).astype(np.int32),
                      np.concatenate(copy).astype(np.int32))


class _Element:
    def __init__(self, spec: Dict[str, Any], base: Path):
        if not isinstance(spec, dict):
            raise ValueError(f"an element is a mapping such as `line: [...]`, not {spec!r}")
        kinds = [k for k in KINDS if k in spec]
        if len(kinds) != 1:
            raise ValueError(f"an element has exactly one of {', '.join(KINDS)}; "
                             f"this one has {kinds or sorted(spec)}")
        self.kind = kinds[0]
        self.spec = spec
        self.dye = int(spec.get("dye", 1))
        self.width = float(spec.get("width", 0.0))
        self.density = float(spec.get("density", 0.0))
        value = spec[self.kind]
        if self.kind in ("line", "circle", "polygon", "image") and self.density <= 0:
            raise ValueError(f"a {self.kind} needs a `density` above 0")
        if self.kind == "points":
            self.points = _xyz(value, "points")
        elif self.kind == "file":
            self.points, dyes = read_labels(base / str(value))
            self.dyes = dyes if dyes is not None else np.full(len(self.points), self.dye)
        elif self.kind == "line":
            self.vertices = _xyz(value, "line")
            if len(self.vertices) < 2:
                raise ValueError("a line needs at least two vertices")
        elif self.kind == "circle":
            self.radius = float(value)
            self.centre = _xyz([spec.get("centre", [0, 0, 0])], "centre")[0]
            normal = np.asarray(spec.get("normal", [0, 0, 1]), float)
            self.normal = normal / np.linalg.norm(normal)
        elif self.kind == "polygon":
            self.vertices = np.asarray(value, float)[:, :2]
            if len(self.vertices) < 3:
                raise ValueError("a polygon needs at least three vertices")
            self.z = _z_range(spec.get("z", 0.0))
        elif self.kind == "image":
            self.image = _read_image(base / str(value))
            self.pixelsize = float(spec.get("pixelsize", 0.0))
            if self.pixelsize <= 0:
                raise ValueError("an image needs its `pixelsize` in nm")
            self.origin = np.asarray(spec.get("origin", [0, 0]), float)
            self.z = _z_range(spec.get("z", 0.0))

    def sample(self, rng):
        if self.kind == "points":
            pts, dye = self.points, np.full(len(self.points), self.dye)
        elif self.kind == "file":
            pts, dye = self.points, self.dyes
        elif self.kind == "line":
            pts = _sample_polyline(self.vertices, self.density, rng)
        elif self.kind == "circle":
            pts = _sample_circle(self.centre, self.radius, self.normal, self.density, rng)
        elif self.kind == "polygon":
            pts = _sample_polygon(self.vertices, self.density, self.z, rng)
        else:
            pts = _sample_image(self.image, self.pixelsize, self.origin, self.density,
                                self.z, rng)
        if self.kind not in ("points", "file"):
            dye = np.full(len(pts), self.dye)
        if self.width > 0 and len(pts):
            pts = pts + rng.normal(0, self.width, pts.shape)
        return np.asarray(pts, float).reshape(-1, 3), np.asarray(dye, int)


# ------------------------------------------------------------------ sampling
def _sample_polyline(vertices, density, rng):
    seg = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    n = rng.poisson(density * lengths.sum())
    s = rng.uniform(0, lengths.sum(), n)
    edges = np.concatenate([[0], np.cumsum(lengths)])
    k = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(seg) - 1)
    t = (s - edges[k]) / np.where(lengths[k] > 0, lengths[k], 1)
    return vertices[k] + t[:, None] * seg[k]


def _sample_circle(centre, radius, normal, density, rng):
    n = rng.poisson(density * 2 * np.pi * radius)
    phi = rng.uniform(0, 2 * np.pi, n)
    # two unit vectors in the circle's plane
    helper = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    if np.allclose(normal, [0, 0, 1]):
        u, v = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    return centre + radius * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)


def _sample_polygon(vertices, density, z, rng):
    from matplotlib.path import Path as MplPath
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    x, y = vertices[:, 0], vertices[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    n = rng.poisson(density * area)
    path, out = MplPath(vertices), []
    # rejection from the bounding box: its area over the polygon's is the cost
    while sum(len(o) for o in out) < n:
        need = n - sum(len(o) for o in out)
        box_fraction = area / max(np.prod(hi - lo), 1e-12)
        cand = rng.uniform(lo, hi, (int(need / max(box_fraction, 1e-3)) + 16, 2))
        out.append(cand[path.contains_points(cand)])
    xy = np.vstack(out)[:n] if out else np.zeros((0, 2))
    return np.column_stack([xy, rng.uniform(z[0], z[1], len(xy))])


def _sample_image(image, pixelsize, origin, density, z, rng):
    counts = rng.poisson(density * pixelsize ** 2 * image)
    iy, ix = np.nonzero(counts)
    reps = counts[iy, ix]
    iy, ix = np.repeat(iy, reps), np.repeat(ix, reps)
    # row index is y, as in a camera frame; pixel k spans k-1/2 .. k+1/2
    x = origin[0] + (ix + rng.uniform(-0.5, 0.5, len(ix))) * pixelsize
    y = origin[1] + (iy + rng.uniform(-0.5, 0.5, len(iy))) * pixelsize
    return np.column_stack([x, y, rng.uniform(z[0], z[1], len(x))])


def _placements(copies: Optional[Dict[str, Any]], rng):
    """``[(position, rotation or None)]`` for every copy of a part."""
    if not copies:
        return [(np.zeros(3), None)]
    field = np.asarray(copies.get("field", [0, 0, 10000, 10000]), float)
    x0, y0, x1, y1 = field
    area_um2 = (x1 - x0) * (y1 - y0) / 1e6
    if "n" in copies:
        n = int(copies["n"])
    elif "density" in copies:
        n = rng.poisson(float(copies["density"]) * area_um2)
    else:
        raise ValueError("`copies` needs `n` or `density` (per um^2)")
    placement = copies.get("placement", "random")
    if placement == "grid":
        side = max(int(np.ceil(np.sqrt(n * (x1 - x0) / max(y1 - y0, 1e-9)))), 1)
        rows = int(np.ceil(n / side))
        gx = x0 + (np.arange(side) + 0.5) * (x1 - x0) / side
        gy = y0 + (np.arange(rows) + 0.5) * (y1 - y0) / rows
        xy = np.array([(a, b) for b in gy for a in gx])[:n]
    elif placement == "random":
        xy = _random_positions(n, field, float(copies.get("min_distance", 0.0)), rng)
    else:
        raise ValueError(f"placement {placement!r}: random or grid")
    rotation = copies.get("rotation", "none")
    out = []
    for p in xy:
        if rotation in ("none", None, False):
            r = None
        elif rotation in ("random", True):
            a = rng.uniform(0, 2 * np.pi)
            r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        elif rotation == "random_3d":
            from scipy.spatial.transform import Rotation
            r = Rotation.random(random_state=rng).as_matrix()
        else:
            raise ValueError(f"rotation {rotation!r}: none, random or random_3d")
        out.append((np.array([p[0], p[1], float(copies.get("z", 0.0))]), r))
    return out


def _random_positions(n, field, min_distance, rng):
    x0, y0, x1, y1 = field
    if min_distance <= 0:
        return np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])
    # sequential rejection on a grid of cells one min_distance wide: each new
    # point looks only at its own and the eight neighbouring cells
    cell = min_distance
    grid: Dict[tuple, list] = {}
    accepted = []
    tries = 0
    while len(accepted) < n and tries < 100 * n + 1000:
        tries += 1
        p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        cx, cy = int(p[0] // cell), int(p[1] // cell)
        near = [q for i in (-1, 0, 1) for j in (-1, 0, 1) for q in grid.get((cx + i, cy + j), ())]
        if any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < min_distance ** 2 for q in near):
            continue
        grid.setdefault((cx, cy), []).append(p)
        accepted.append(p)
    if len(accepted) < n:
        raise ValueError(f"only {len(accepted)} of {n} copies fit {min_distance:g} nm apart "
                         f"in the field")
    return np.asarray(accepted).reshape(-1, 2)


# ------------------------------------------------------------------ reading
def _xyz(value, what) -> np.ndarray:
    a = np.asarray(value, float)
    if a.ndim != 2 or a.shape[1] not in (2, 3):
        raise ValueError(f"`{what}` is a list of [x, y] or [x, y, z] in nm")
    if a.shape[1] == 2:
        a = np.column_stack([a, np.zeros(len(a))])
    return a


def _z_range(value) -> Sequence[float]:
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    return float(value), float(value)


def read_labels(path: Path):
    """``(xyz, dye or None)`` from a text file: ``x, y[, z][, dye]`` per row,
    commas or white space, header lines and ``#`` comments skipped."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                rows.append([float(v) for v in line.replace(",", " ").split()])
            except ValueError:
                continue                          # a header
    if not rows:
        raise ValueError(f"{path}: no rows of numbers")
    width = min(len(r) for r in rows)
    if width < 2:
        raise ValueError(f"{path}: each row needs at least x and y")
    a = np.array([r[:width] for r in rows])
    xyz = _xyz(a[:, :min(width, 3)], str(path))
    dye = a[:, 3].astype(int) if width >= 4 else None
    return xyz, dye


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        img = tifffile.imread(path)
    else:
        from matplotlib.image import imread
        img = imread(path)
    img = np.asarray(img, float)
    if img.ndim == 3:                           # colour, perhaps with alpha
        img = img[..., :3].mean(axis=2)
    top = img.max()
    if top <= 0:
        raise ValueError(f"{path}: the image is black")
    return img / top
