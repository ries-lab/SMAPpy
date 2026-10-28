"""A path of the 3D view through key views: the player's *view path*.

Called a view path and not a camera path: in this package a camera is the
one that took the images (`smappy.camera`).

The player (`smappy.scan`) moves a window through the data; this moves the
view, and the two run on one time line, so a structure can turn while it
builds up, or turn and zoom with all of it shown.  A path is a list of key
views, each the time after the one before it; between them the view is
interpolated, and a recorded movie is the 3D window's picture at each
point (`scan.scan_pictures`).

**What a key is.**  The view as `view3d.Projection` holds it -- a turn,
a centre of rotation (``pivot``), a pan (``offset``), a zoom -- is not what
should be interpolated: halfway between two views with different pans and
turns, the pan of the one is applied in the turn of the other, and what
was in the middle of the screen at both ends swings out of it on the way.
So a key is the view *about the point in the middle of the screen*
(`Projection.pivot_at_centre`, after which the pan is zero): a turn, the
point looked at, a zoom.  Those three interpolate independently and the
point looked at stays in the middle all the way.  The rest of a
projection -- the engine, the opacity, the depth cue -- is not the view
and is the window's own (`Viewpoint.apply`).

**The box.**  A key also keeps the 3D window's box (`view3d.Slab`: its
centre, its size, its turn), and a path whose keys differ in it moves the
box: a box made shallower at the next key, its back face where it was,
moves the front face through the sample.

**How it is interpolated.**  The turn on unit quaternions, smoothly
through the keys: each key has an angular velocity, the weighted mean of
the turns to the keys on either side (as Catmull-Rom takes its tangents),
and each segment is a cubic Bezier curve on the sphere of rotations
(Shoemake's, by repeated slerp) with its inner control points a third of
a segment along those velocities.  The turn neither stops nor jerks at a
key, and keys spaced evenly about one axis turn at a constant rate, as a
slerp from key to key would.  Interpolating the azimuth and elevation
angles instead jumps where they wrap at 360 degrees and swings around the
pole of a top view.  Each segment is the shorter of the two ways round, so
a turn of more than 180 degrees is several keys, which the turns made here
(`turn_keys`) are, a key every 90 degrees.  The point looked at and the
zoom by a cubic through the keys (Catmull-Rom with the keys' own times),
the zoom on a log scale, so that zooming in by four looks as fast at the
end as at the start, and the box's centre, size and turn the same way; a cubic reproduces a straight line exactly, so keys
spaced evenly along one are passed at a constant speed.  Perspective's
eye distance linearly, where both keys have one.

*ease* slows the start and the end of the whole path (a smoothstep of
the time), not every key: a turn made of four keys would otherwise stop at
each of them.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

#: the axes a turn can be about: the data's own, or the screen's
AXES = ("z", "x", "y", "screen vertical", "screen horizontal")
#: a key every so many degrees of a turn: slerp takes the shortest way, so
#: a key must be less than 180 degrees from the next
TURN_STEP = 90.0


def _rot(axis: int, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


def quaternion(R: np.ndarray) -> np.ndarray:
    """The unit quaternion (w, x, y, z) of a rotation matrix (Shepperd's
    method: from the largest of the four squares, never a small divisor)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    d = np.array([t, R[0, 0], R[1, 1], R[2, 2]])
    k = int(np.argmax(d))
    if k == 0:
        w = math.sqrt(1.0 + t) / 2
        q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
             (R[1, 0] - R[0, 1]) / (4 * w)]
    else:
        i = k - 1
        j, m = (i + 1) % 3, (i + 2) % 3
        r = math.sqrt(max(1.0 + R[i, i] - R[j, j] - R[m, m], 0.0)) / 2
        q = [0.0] * 4
        q[0] = (R[m, j] - R[j, m]) / (4 * r)
        q[1 + i] = r
        q[1 + j] = (R[j, i] + R[i, j]) / (4 * r)
        q[1 + m] = (R[m, i] + R[i, m]) / (4 * r)
    q = np.array(q)
    return q / np.linalg.norm(q)


def matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                     w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                     w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])


def qconj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qlog(q: np.ndarray) -> np.ndarray:
    """The vector ``v`` with ``qexp(v) == q``: half the turn's angle, along
    its axis."""
    v = np.asarray(q[1:], float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    return v / n * math.atan2(n, float(q[0]))


def qexp(v: np.ndarray) -> np.ndarray:
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[math.cos(a)], np.sin(a) * np.asarray(v) / a])


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """From ``q0`` (u = 0) to ``q1`` (u = 1) the shortest way, at a constant
    rate."""
    dot = float(q0 @ q1)
    if dot < 0:                      # q and -q are one rotation: go the short way
        q1, dot = -q1, -dot
    if dot > 0.9995:                 # nearly one: the arc is a line
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    return (math.sin((1 - u) * theta) * q0 + math.sin(u * theta) * q1) / math.sin(theta)


def bezier(q0, c0, c1, q1, u: float) -> np.ndarray:
    """A cubic Bezier curve on the sphere of rotations, by repeated slerp
    (de Casteljau; Shoemake 1985): from ``q0`` to ``q1``, leaving towards
    ``c0`` and arriving from ``c1``."""
    a, b, c = slerp(q0, c0, u), slerp(c0, c1, u), slerp(c1, q1, u)
    return slerp(slerp(a, b, u), slerp(b, c, u), u)


def smooth_turn(times: np.ndarray, qs: Sequence[np.ndarray], s: float, i: int) -> np.ndarray:
    """The turn at ``s`` in segment ``i``, smooth through the keys ``qs``
    (already on one hemisphere) at ``times``.

    Each key's angular velocity is the mean of the turns to its neighbours
    weighted by their times, as `_hermite`'s tangents are; an end key's is
    the turn to its one neighbour.  The turn from one key to the next,
    ``qlog(conj(q_i) q_j)``, is the same vector seen from either key, so the
    two can be averaged."""
    n = len(qs)

    def rate(k):
        """Half the angular velocity at key ``k``, per second."""
        slopes, weights = [], []
        for a, b in ((k - 1, k), (k, k + 1)):
            if 0 <= a and b < n:
                h = times[b] - times[a]
                slopes.append(qlog(qmul(qconj(qs[a]), qs[b])) / h)
                weights.append(h)
        return sum(w * d for w, d in zip(weights, slopes)) / sum(weights)

    h = times[i + 1] - times[i]
    u = (s - times[i]) / h
    c0 = qmul(qs[i], qexp(rate(i) * h / 3))
    c1 = qmul(qs[i + 1], qexp(-rate(i + 1) * h / 3))
    return bezier(qs[i], c0, c1, qs[i + 1], u)


@dataclass
class Viewpoint:
    """A view about the point in the middle of the screen: ``rotation``
    (data -> view, `Projection.matrix`), the point ``centre`` (nm), the
    ``zoom`` (nm per screen pixel), perspective's ``focal`` (None:
    orthographic) and the 3D window's box, ``slab`` (None: none kept)."""

    rotation: np.ndarray
    centre: np.ndarray
    zoom: float
    focal: Optional[float] = None
    slab: Optional[object] = None        # a `view3d.Slab`

    @classmethod
    def of(cls, projection, slab=None) -> "Viewpoint":
        """What ``projection`` shows, as a view about the screen's centre,
        and the box ``slab``."""
        p = copy.deepcopy(projection)
        p.pivot_at_centre()
        return cls(p.matrix.copy(), p.pivot.copy(), float(p.zoom), p.focal,
                   copy.deepcopy(slab))

    def apply(self, projection):
        """A copy of ``projection`` looking from this viewpoint; its engine,
        opacity and the rest are kept."""
        p = copy.deepcopy(projection)
        p.set_matrix(self.rotation)
        p.pivot = np.asarray(self.centre, float).copy()
        p.offset = np.zeros(2)
        p.zoom = float(self.zoom)
        p.focal = self.focal
        return p

    def same(self, other: "Viewpoint", tol: float = 1e-6) -> bool:
        a, b = self.slab, other.slab
        same_box = (a is None) == (b is None) and (a is None or (
            np.allclose(a.center, b.center) and np.allclose(a.size, b.size)
            and math.isclose(a.angle % 360, b.angle % 360)))
        return (np.allclose(self.rotation, other.rotation, atol=tol)
                and np.allclose(self.centre, other.centre, atol=tol * max(1.0, self.zoom))
                and math.isclose(self.zoom, other.zoom, rel_tol=tol)
                and self.focal == other.focal and same_box)


@dataclass
class Key:
    """A key view and ``seconds``, the time from the key before it (the
    first key's is not used)."""

    view: Viewpoint
    seconds: float = 2.0

    def to_dict(self) -> dict:
        c = self.view
        return {"rotation": np.round(c.rotation, 12).tolist(), "centre": c.centre.tolist(),
                "zoom": c.zoom, "focal": c.focal,
                "box": None if c.slab is None else c.slab.to_dict(),
                "seconds": self.seconds}

    @classmethod
    def from_dict(cls, d: dict) -> "Key":
        from .view3d import Slab
        box = d.get("box")
        return cls(Viewpoint(np.array(d["rotation"], float), np.array(d["centre"], float),
                             float(d["zoom"]), d.get("focal"),
                             None if box is None else Slab.from_dict(box)),
                   float(d.get("seconds", 2.0)))


def _hermite(times: np.ndarray, values: np.ndarray, s: float, i: int) -> np.ndarray:
    """The cubic through ``values`` at ``times`` (Catmull-Rom with the keys'
    own spacing), at ``s`` in segment ``i``.  The end tangents are the
    chords, so evenly spaced keys on a line are passed at a constant speed."""
    n = len(times)

    def tangent(k):
        a, b = max(k - 1, 0), min(k + 1, n - 1)
        return (values[b] - values[a]) / (times[b] - times[a])

    h = times[i + 1] - times[i]
    u = (s - times[i]) / h
    h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
    h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    return (h00 * values[i] + h10 * h * tangent(i)
            + h01 * values[i + 1] + h11 * h * tangent(i + 1))


def _box(times, views, s: float, i: int, u: float):
    """The box at ``s``: centre, size and turn through the keys' boxes as
    the centre is; None unless every key keeps one."""
    boxes = [v.slab for v in views]
    if any(b is None for b in boxes):
        return boxes[i] if u < 1 else boxes[i + 1]
    from .view3d import Slab
    # the turn unwrapped, so 350 to 10 degrees is 20 and not 340 back
    angles = np.degrees(np.unwrap(np.radians([b.angle for b in boxes])))
    return Slab(_hermite(times, np.array([b.center for b in boxes]), s, i),
                np.maximum(_hermite(times, np.array([b.size for b in boxes]), s, i), 1e-3),
                float(_hermite(times, angles, s, i)) % 360)


@dataclass
class ViewPath:
    """Key views and how to move between them."""

    keys: List[Key] = field(default_factory=list)
    ease: bool = True

    @property
    def seconds(self) -> float:
        """How long the path takes, at its keys' times."""
        return float(sum(k.seconds for k in self.keys[1:]))

    @property
    def moves(self) -> bool:
        return len(self.keys) >= 2 and self.seconds > 0

    @property
    def closed(self) -> bool:
        """It ends where it began (a whole turn): a loop must not show that
        view twice running."""
        return self.moves and self.keys[0].view.same(self.keys[-1].view)

    def times(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum([k.seconds for k in self.keys[1:]])])

    def at(self, t: float) -> Viewpoint:
        """The viewpoint at ``t``, 0 the first key and 1 the last."""
        if not self.keys:
            raise ValueError("the view path has no keys")
        if not self.moves:
            return self.keys[0].view
        t = float(np.clip(t, 0.0, 1.0))
        if self.ease:
            t = t * t * (3 - 2 * t)
        times = self.times()
        keep = np.concatenate([[True], np.diff(times) > 0])    # a key of 0 s replaces
        times = times[keep]
        views = [k.view for k, on in zip(self.keys, keep) if on]
        s = t * times[-1]
        i = int(np.clip(np.searchsorted(times, s, side="right") - 1, 0, len(times) - 2))
        u = (s - times[i]) / (times[i + 1] - times[i])
        # one sign per key, each the nearer of q and -q to the one before
        qs = [quaternion(c.rotation) for c in views]
        for k in range(1, len(qs)):
            if qs[k] @ qs[k - 1] < 0:
                qs[k] = -qs[k]
        rotation = matrix(smooth_turn(times, qs, s, i))
        centre = _hermite(times, np.array([c.centre for c in views], float), s, i)
        zoom = float(np.exp(_hermite(times, np.log([c.zoom for c in views]), s, i)))
        f0, f1 = views[i].focal, views[i + 1].focal
        focal = f0 + u * (f1 - f0) if f0 and f1 else (f0 if u < 1 else f1)
        return Viewpoint(rotation, centre, zoom, focal, _box(times, views, s, i, u))

    @property
    def moves_box(self) -> bool:
        """Every key keeps a box, and not all the same one."""
        boxes = [k.view.slab for k in self.keys]
        if len(boxes) < 2 or any(b is None for b in boxes):
            return False
        return any(not (np.allclose(b.center, boxes[0].center)
                        and np.allclose(b.size, boxes[0].size)
                        and math.isclose(b.angle, boxes[0].angle)) for b in boxes[1:])

    def samples(self, n: int, loop: bool = False) -> np.ndarray:
        """``n`` times from 0 to 1 for the frames of a movie; a closed path
        played in a loop leaves its last view out, which is its first."""
        if n <= 1:
            return np.zeros(max(n, 1))
        if loop and self.closed:
            return np.arange(n) / n
        return np.linspace(0.0, 1.0, n)

    def to_dict(self) -> dict:
        return {"ease": self.ease, "keys": [k.to_dict() for k in self.keys]}

    @classmethod
    def from_dict(cls, d: dict) -> "ViewPath":
        return cls([Key.from_dict(k) for k in d.get("keys", [])], bool(d.get("ease", True)))


def turned(rotation: np.ndarray, axis: str, degrees: float) -> np.ndarray:
    """``rotation`` turned by ``degrees`` about one of `AXES`: the data's
    z, x or y through the point looked at, or the screen's vertical or
    horizontal.  Positive about z or the screen's vertical turns as a drag
    to the right does in the 3D window."""
    if axis == "screen vertical":
        return _rot(1, degrees) @ rotation
    if axis == "screen horizontal":
        return _rot(0, degrees) @ rotation
    i = "xyz".index(axis)
    return rotation @ _rot(i, -degrees)


def turn_keys(start: Viewpoint, axis: str = "z", degrees: float = 360.0,
              zoom_by: float = 1.0, seconds: float = 12.0) -> List[Key]:
    """The keys after ``start`` of a turn by ``degrees`` about ``axis`` and
    a zoom in by ``zoom_by`` (2: twice as large at the end) over ``seconds``:
    a key every `TURN_STEP` degrees, so slerp turns all the way, and the zoom
    spread evenly over them on a log scale.  No turn is a zoom alone."""
    if zoom_by <= 0:
        raise ValueError("the zoom must be above 0")
    n = max(1, int(math.ceil(abs(degrees) / TURN_STEP - 1e-9)))
    keys = []
    for k in range(1, n + 1):
        f = k / n
        keys.append(Key(Viewpoint(turned(start.rotation, axis, degrees * f),
                                  np.asarray(start.centre, float).copy(),
                                  start.zoom / zoom_by ** f, start.focal,
                                  copy.deepcopy(start.slab)), seconds / n))
    return keys


#: the ⟳'s axis left to `motion_axis`, beside the fixed ones of `AXES`
FROM_VIEW = "from the view"


def motion_axis(rotation: np.ndarray, axis: str = FROM_VIEW) -> str:
    """What the 3D window's ⟳ turns about: the data's z where z stands
    within 45 degrees of upright on the screen -- a turntable, the horizon
    level -- else the screen's vertical.  Depth is seen from a turn about an
    axis in the plane of the screen; about z seen from the top, the picture
    would only turn flat.  An ``axis`` of `AXES` is taken as it is: a rock
    about x, say, from whatever view."""
    if axis != FROM_VIEW:
        if axis not in AXES:
            raise ValueError(f"no axis {axis!r}: one of {', '.join((FROM_VIEW,) + AXES)}")
        return axis
    upright = abs(float(np.asarray(rotation)[1, 2]))       # data z along screen y
    return "z" if upright >= math.cos(math.radians(45.0)) else "screen vertical"


def motion_angle(t: float, motion: str, amplitude: float, period: float,
                 speed: float) -> float:
    """The turn ``t`` seconds into the ⟳'s motion, in degrees: a rock,
    ``amplitude`` either way and back over ``period`` seconds -- a sine, so
    it slows into each turnaround -- or a spin at ``speed`` degrees a
    second."""
    if motion == "rock":
        return amplitude * math.sin(2.0 * math.pi * t / max(period, 1e-3))
    return speed * t


def motion_path(start: Viewpoint, motion: str, amplitude: float, period: float,
                speed: float, axis: str = FROM_VIEW) -> "ViewPath":
    """The ⟳'s motion from ``start`` as a path for the player, which records
    it: one rock (to one side, back, to the other, back) or one whole turn,
    without *ease*, so that played in a loop it does not stop at the seam.
    The rock's keys at the far sides have no turn on either side of them,
    so the path slows into each turnaround as the window's sine does."""
    axis = motion_axis(start.rotation, axis)
    if motion == "rock":
        if not 0 < abs(amplitude) < 180:
            raise ValueError("a rock turns less than 180 degrees either way")
        keys = [Key(start)]
        for angle in (amplitude, 0.0, -amplitude, 0.0):
            keys.append(Key(Viewpoint(turned(start.rotation, axis, angle),
                                      np.asarray(start.centre, float).copy(), start.zoom,
                                      start.focal, copy.deepcopy(start.slab)),
                            period / 4.0))
    else:
        if speed == 0:
            raise ValueError("a spin needs a speed")
        keys = [Key(start)] + turn_keys(start, axis, math.copysign(360.0, speed), 1.0,
                                        360.0 / abs(speed))
    return ViewPath(keys, ease=False)


def projections(path: ViewPath, base, times: Sequence[float]) -> list:
    """The projection of each time on ``path``, as ``base`` (the 3D window's
    projection) with the path's viewpoint."""
    return [path.at(t).apply(base) for t in times]


def save(path: ViewPath, filename) -> None:
    """A path as a YAML file (``*.view.yaml``), to be played again."""
    import yaml
    with open(filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(path.to_dict(), f, sort_keys=False)


def load(filename) -> ViewPath:
    import yaml
    with open(filename, encoding="utf-8") as f:
        return ViewPath.from_dict(yaml.safe_load(f) or {})
