"""Look-up tables: 256-entry colour ramps.

A LUT is an ``(256, 3)`` float32 array of RGB in [0, 1].  Two places use them
and they are not the same operation:

* an intensity image is *recoloured* -- one LUT lookup per output pixel, after
  normalisation, so changing the LUT costs nothing and never touches the
  localizations;
* a field-coloured image needs the colour of every localization *before*
  accumulation, because two localizations in one pixel must mix.

Most ramps are defined by a handful of anchor colours interpolated to 256
entries, which is what SMAP's ``usercolormap`` does.  ``red hot``, ``hue
circle`` and ``gray`` follow MATLAB's hot, hsv and gray so images look the
same as SMAP's.  The LUTs come in three groups -- one colour and several
colours for intensity, attribute for a column (`groups`) -- and the names
say the colours rather than where a table came from.
No matplotlib dependency: the tables are small and the interpolation is three
lines.
"""

from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np

N = 256  # entries in every LUT


def from_anchors(*colors: Sequence[float]) -> np.ndarray:
    """A ramp through equally spaced anchor colours (SMAP's ``usercolormap``)."""
    anchors = np.asarray(colors, dtype=np.float64)
    at = np.linspace(0.0, 1.0, len(anchors))
    x = np.linspace(0.0, 1.0, N)
    return np.stack([np.interp(x, at, anchors[:, k]) for k in range(3)],
                    axis=1).astype(np.float32)


def _matlab_hot() -> np.ndarray:
    n = int(np.ceil(N * 3 / 8))
    ramp = np.arange(1, n + 1) / n
    r = np.concatenate([ramp, np.ones(N - n)])
    g = np.concatenate([np.zeros(n), ramp, np.ones(N - 2 * n)])
    b = np.concatenate([np.zeros(2 * n), np.arange(1, N - 2 * n + 1) / (N - 2 * n)])
    return np.stack([r, g, b], axis=1).astype(np.float32)


def _matlab_hsv() -> np.ndarray:
    h = np.linspace(0.0, 1.0, N, endpoint=False) * 6.0
    i, f = np.floor(h).astype(int) % 6, h - np.floor(h)
    up, down = f, 1.0 - f
    one, zero = np.ones_like(f), np.zeros_like(f)
    r = np.choose(i, [one, down, zero, zero, up, one])
    g = np.choose(i, [up, one, one, down, zero, zero])
    b = np.choose(i, [zero, zero, up, one, one, down])
    return np.stack([r, g, b], axis=1).astype(np.float32)


def _single(r: float, g: float, b: float) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, N)
    return np.stack([ramp * r, ramp * g, ramp * b], axis=1).astype(np.float32)


_gray = _single(1, 1, 1)

# Turbo, sampled from SMAP's copy of the table every 8th entry.  It is defined
# by a table rather than a formula, and 33 anchors reproduce it to 3/255 -- less
# than a step of the 256-entry ramp -- so it is stored as anchors like the rest.
# Worth having for z: unlike `hsv` it does not wrap, so the two ends of the
# range cannot be confused, and unlike `jet` its lightness increases
# monotonically, so structure does not appear where the colour map bunches up.
_TURBO_ANCHORS = [
    [0.1900, 0.0718, 0.2322],
    [0.2250, 0.1635, 0.4510],
    [0.2511, 0.2524, 0.6337],
    [0.2682, 0.3382, 0.7805],
    [0.2763, 0.4212, 0.8912],
    [0.2754, 0.5011, 0.9659],
    [0.2586, 0.5796, 0.9988],
    [0.2138, 0.6589, 0.9796],
    [0.1584, 0.7355, 0.9231],
    [0.1117, 0.8057, 0.8452],
    [0.0927, 0.8655, 0.7623],
    [0.1201, 0.9119, 0.6866],
    [0.1966, 0.9490, 0.5947],
    [0.3051, 0.9770, 0.4899],
    [0.4278, 0.9942, 0.3857],
    [0.5466, 0.9991, 0.2958],
    [0.6436, 0.9900, 0.2336],
    [0.7260, 0.9647, 0.2064],
    [0.8047, 0.9245, 0.2046],
    [0.8753, 0.8727, 0.2155],
    [0.9330, 0.8124, 0.2267],
    [0.9732, 0.7468, 0.2254],
    [0.9931, 0.6741, 0.2035],
    [0.9959, 0.5870, 0.1690],
    [0.9836, 0.4929, 0.1285],
    [0.9580, 0.3996, 0.0883],
    [0.9211, 0.3149, 0.0548],
    [0.8742, 0.2453, 0.0330],
    [0.8161, 0.1846, 0.0181],
    [0.7462, 0.1310, 0.0085],
    [0.6645, 0.0844, 0.0042],
    [0.5710, 0.0447, 0.0053],
    [0.4796, 0.0158, 0.0106],
]

# Inferno and plasma (matplotlib's, CC0), sampled at 21 points: 4/255 from the
# tables, below a step of the ramp.
_INFERNO_ANCHORS = [
    [0.0015, 0.0005, 0.0139], [0.0258, 0.0193, 0.1059], [0.0874, 0.0446, 0.2248],
    [0.1696, 0.0425, 0.3409], [0.2582, 0.0386, 0.4065], [0.3415, 0.0623, 0.4294],
    [0.4163, 0.0902, 0.4329], [0.4973, 0.1194, 0.4245], [0.5783, 0.1480, 0.4044],
    [0.6585, 0.1790, 0.3727], [0.7357, 0.2159, 0.3302], [0.8019, 0.2587, 0.2831],
    [0.8650, 0.3168, 0.2261], [0.9165, 0.3875, 0.1649], [0.9545, 0.4687, 0.0999],
    [0.9784, 0.5579, 0.0349], [0.9876, 0.6453, 0.0399], [0.9832, 0.7438, 0.1385],
    [0.9644, 0.8438, 0.2734], [0.9464, 0.9372, 0.4586], [0.9884, 0.9984, 0.6449],
]
_PLASMA_ANCHORS = [
    [0.0504, 0.0298, 0.5280], [0.1641, 0.0202, 0.5775], [0.2546, 0.0139, 0.6154],
    [0.3377, 0.0056, 0.6430], [0.4176, 0.0006, 0.6584], [0.4949, 0.0120, 0.6579],
    [0.5627, 0.0515, 0.6415], [0.6310, 0.1077, 0.6083], [0.6928, 0.1651, 0.5645],
    [0.7483, 0.2227, 0.5168], [0.7982, 0.2802, 0.4695], [0.8402, 0.3336, 0.4275],
    [0.8814, 0.3925, 0.3832], [0.9181, 0.4536, 0.3395], [0.9492, 0.5178, 0.2957],
    [0.9734, 0.5858, 0.2515], [0.9883, 0.6523, 0.2114], [0.9946, 0.7287, 0.1716],
    [0.9886, 0.8096, 0.1454], [0.9684, 0.8946, 0.1470], [0.9400, 0.9752, 0.1313],
]
# Our own blue -> teal -> green -> yellow, after MATLAB's parula (whose table
# is MathWorks' and is not copied): nine anchors drawn by eye.
_BLUE_GREEN_YELLOW_ANCHORS = [
    [0.208, 0.166, 0.529], [0.012, 0.388, 0.882], [0.078, 0.522, 0.831],
    [0.024, 0.646, 0.765], [0.209, 0.719, 0.604], [0.553, 0.749, 0.396],
    [0.851, 0.729, 0.337], [0.988, 0.808, 0.180], [0.976, 0.984, 0.081],
]


def full_value(table: np.ndarray) -> np.ndarray:
    """Each entry scaled so its brightest channel is one.

    A LUT that codes a column is multiplied by the density afterwards
    (`render._to_rgb`), so any darkness of its own reads as fewer
    localizations: turbo's two ends (brightest channel 0.23 and 0.48) made the
    lowest and highest z look sparse where they were not.  At full value the
    colour says the value and only the brightness says the density.
    """
    table = np.asarray(table, dtype=np.float32)
    return table / np.maximum(table.max(axis=1, keepdims=True), 1e-6)


def cyclic(*colors: Sequence[float]) -> np.ndarray:
    """A closed ramp through the anchors and back to the first, the last
    entry one step short of it -- so 0 and 360 degrees are the same colour
    and no colour is drawn twice."""
    anchors = np.asarray(colors + (colors[0],), dtype=np.float64)
    at = np.linspace(0.0, 1.0, len(anchors))
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    return np.stack([np.interp(x, at, anchors[:, k]) for k in range(3)],
                    axis=1).astype(np.float32)


# Three groups, which the Render tab's list shows under their headings.
#
# * one colour -- for intensity: black to the colour, brightness the only
#   dimension;
# * several colours -- for intensity: black through colours to white, the
#   lightness rising all the way, so more steps can be told apart than on one
#   colour (about three times as many as on gray, summing the colour
#   differences along the ramp in CIE Lab);
# * attribute -- to colour by a column: the colour is the value and the
#   density multiplies it, so every one is at full value (`full_value`) or, the
#   diverging ones, goes through black or white on purpose.  Ordered, cyclic
#   (a hue circle, for angles -- but nothing stops z on one) and diverging
#   (about a middle value) are all in it.
#
# White paper turns the picture over (`on_white`), which is its own inverse
# and swaps black with white while keeping saturated colours: so a diverging
# ramp through black is the same ramp through white on white paper, and one
# through white goes through black there.  Both are offered, in different
# colours, so either background has a ramp through its own colour.
_GROUPS = {
    "one colour": {
        "gray": _gray,
        "red": _single(1, 0, 0),
        "green": _single(0, 1, 0),
        "blue": _single(0, 0, 1),
        "cyan": _single(0, 1, 1),
        "magenta": _single(1, 0, 1),
        "yellow": _single(1, 1, 0),
    },
    "several colours": {
        "red hot": _matlab_hot(),           # MATLAB's hot, as SMAP's images
        "blue cold": from_anchors([0, 0, 0], [0, .2, 1], [0, .5, 1], [0, 1, 1],
                                  [.5, 1, 1], [1, 1, 1]),
        "green cold": from_anchors([0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]),
        # inferno, closed to white so that it is a black-to-white ramp like the
        # rest (white paper draws those backwards, `intensity_table`)
        "purple fire": from_anchors(*_INFERNO_ANCHORS, [1, 1, 1]),
    },
    "attribute": {
        # turbo: unlike jet its lightness rises monotonically, unlike hsv the
        # two ends cannot be confused
        "rainbow": full_value(from_anchors(*_TURBO_ANCHORS)),
        "blue-green-yellow": full_value(from_anchors(*_BLUE_GREEN_YELLOW_ANCHORS)),
        "blue-magenta-yellow": full_value(from_anchors(*_PLASMA_ANCHORS)),
        "hue circle": _matlab_hsv(),
        # four hues a quarter apart, so the quadrants of an angle read at a
        # glance; hsv's bright yellow, cyan and magenta bands make edges where
        # the data have none
        "four-hue circle": full_value(cyclic([1, 0, 1], [1, .85, 0], [0, .9, .3],
                                             [0, .45, 1])),
        # no blue, which is dark on black (lightness 32 of 100)
        "magenta-black-green": from_anchors([1, 0, 1], [0, 0, 0], [0, 1, 0]),
        "purple-white-orange": from_anchors([.6, .2, 1], [1, 1, 1], [1, .5, 0]),
    },
}
_TABLES: Dict[str, np.ndarray] = {name: table for group in _GROUPS.values()
                                  for name, table in group.items()}

LUT = Union[str, np.ndarray]


def names() -> list:
    """Every LUT, group by group."""
    return list(_TABLES)


def groups(attribute: bool = False) -> list:
    """``[(heading, names), ...]`` in the order a list shows them: the groups
    for intensity first, or with ``attribute`` the one for a column first."""
    order = list(_GROUPS)
    if attribute:
        order = order[-1:] + order[:-1]
    return [(heading, list(_GROUPS[heading])) for heading in order]


def kind(lut: LUT) -> str:
    """The heading a LUT's name is under ("" for one that was registered or
    passed as an array)."""
    for heading, group in _GROUPS.items():
        if isinstance(lut, str) and lut in group:
            return heading
    return ""


def invert_sum(table: np.ndarray) -> np.ndarray:
    """SMAP's inversion: ``sum(c) - c`` per entry, clipped.

    `lutinvert.m`, verbatim.  On a ramp of one hue it is the opposite colour
    at the same brightness -- ``(r, 0, 0)`` gives ``(0, r, r)`` -- and the
    sum counts a second channel twice, so `hot` inverted this way runs black
    through teal and cyan to white and saturates there.

    The clipping is ours; MATLAB leaves the entries above one and the display
    clips them, which is the same picture.

    It is the inversion of an intensity ramp only.  A LUT that codes a column
    is turned round instead (`get` with ``invert="reverse"``), as SMAP does:
    the sum of a rainbow's channels is no colour of it, and turbo inverted
    this way was a muddle of whites and greys.
    """
    table = np.asarray(table, dtype=np.float32)
    return np.clip(table.sum(axis=1, keepdims=True) - table, 0.0, 1.0)


#: what ``invert`` asks of a LUT that codes a column (`colors`)
REVERSE = "reverse"


INTENSITY_LUT = "red hot"   # an intensity ramp: brightness is the only dimension
FIELD_LUT = "rainbow"       # a full-hue ramp, which is what a coded field needs


def on_white(rgb: np.ndarray) -> np.ndarray:
    """The same colour at the opposite brightness: the picture on white paper.

    Hue and saturation are kept and the HSL lightness is flipped, which works
    out to ``c + (1 - max - min)``: chroma is unchanged by ``L -> 1 - L``, so
    only the level moves.  Black becomes white, white becomes black, a fully
    saturated colour (red, yellow) stays exactly itself, and a faint red spot
    becomes a pale pink one -- ink on paper, at the same contrast.

    So a ``red`` ramp runs white to red, ``hot`` runs white through red and
    yellow to black, and a grey one is the black-on-white ramp.  Takes a LUT
    ``(n, 3)`` or a whole image ``(ny, nx, 3)``; on an intensity image the two
    are the same thing, and on a field-coloured one, where the colour is baked
    into the accumulation, only the image can be turned over.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    level = rgb.max(axis=-1, keepdims=True) + rgb.min(axis=-1, keepdims=True)
    return np.clip(rgb + (1.0 - level), 0.0, 1.0)


def get(lut: LUT, invert=False) -> np.ndarray:
    """Resolve a LUT name (or pass an array through), inverted if asked.

    ``invert`` true is SMAP's inversion of an intensity ramp (`invert_sum`);
    `REVERSE` turns the table round, which is what inverting means for a
    LUT that codes a column (`colors`).
    """
    if isinstance(lut, str):
        try:
            table = _TABLES[lut]
        except KeyError:
            raise KeyError(f"unknown LUT {lut!r}; have {', '.join(names())}") from None
    else:
        table = np.asarray(lut, dtype=np.float32)
        if table.ndim != 2 or table.shape[1] != 3:
            raise ValueError("a LUT must have shape (n, 3)")
    if invert == REVERSE:
        return np.ascontiguousarray(table[::-1])
    return invert_sum(table) if invert else table


def intensity_table(lut: LUT, invert=False, white: bool = False) -> np.ndarray:
    """The table an intensity image is looked up in, on a black ground.

    On white paper the finished picture is turned over (`on_white`).  That
    suits a ramp of one colour -- white to red, ink on paper -- but on a ramp
    through several colours it keeps the saturated middle as it is, and red
    hot on white ran white, red, *yellow*, black: the brightness no longer in
    the order of the intensity.  A ramp from black to white is instead drawn
    backwards there, white for nothing and black for the most, through the
    same colours in the opposite order.  Since `on_white` is its own inverse,
    that is the table ``on_white(t[::-1])`` on the black ground the layers are
    summed on, and the turn-over of the sum makes it ``t[::-1]``.  On gray it
    is the same table either way.
    """
    table = get(lut, invert)
    if white and _black_to_white(table):
        return on_white(table[::-1])
    return table


def _black_to_white(table: np.ndarray) -> bool:
    return bool(np.all(table[0] < 0.02) and np.all(table[-1] > 0.98))


def field_inversion(invert) -> object:
    """``invert`` as a LUT that codes a column takes it: turned round."""
    return REVERSE if invert else False


def register(name: str, table: np.ndarray) -> None:
    """Add a LUT, e.g. one taken from matplotlib."""
    _TABLES[name] = get(table)


def colors(values: np.ndarray, lut: LUT, vmin: float, vmax: float,
           invert: bool = False) -> np.ndarray:
    """Map field values to ``(n, 3)`` colours, clamped to the range; inverted,
    the low values take the LUT's top end (`field_inversion`).

    A value that is not a number -- a MINFLUX localization's `cfr` where its
    iterations did not measure one -- is drawn grey (`NO_VALUE`), not in a
    colour of the table: converted to an index it was the first colour on
    ARM and an index out of range on x86, and either way a value it is not.
    """
    table = get(lut, field_inversion(invert))
    scale = (len(table) - 1) / (vmax - vmin) if vmax > vmin else 0.0
    values = np.asarray(values, dtype=np.float32)
    missing = ~np.isfinite(values)
    index = np.clip((np.where(missing, vmin, values) - vmin) * scale,
                    0, len(table) - 1).astype(np.int32)
    out = table[index]
    if missing.any():
        out[missing] = np.asarray(NO_VALUE, np.float64) * (255 if out.dtype == np.uint8 else 1)
    return out


# the colour of a localization whose colour column has no value: grey
NO_VALUE = (0.5, 0.5, 0.5)
