#!/usr/bin/env python3
"""Draw smappy's icon: a map pin whose head holds a pixelated PSF.

The pin is the "map" in SMAP; the head is what a camera records of one
emitter -- an idealised Gaussian integrated over 7 x 7 pixels, in smappy's own
``hot`` LUT (`smappy.lut._matlab_hot`, MATLAB's), with no noise -- and the
crosshair is the localization.  Writes ``smappy.svg`` and PNGs of it into
``src/smappy/data/icons``; the PNGs are rendered by Qt, so the SVG is the
source and the PNGs are never edited by hand.  Change the constants and rerun.
"""
import math
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "smappy" / "data" / "icons"
SIZES = (16, 24, 32, 48, 64, 128, 256, 512)

N_PIX = 7            # pixels across the head's grid
SIGMA_PIX = 1.35     # PSF width in pixels
CENTRE_PIX = (3.5, 3.5)   # the emitter, in pixels from the grid's corner
PEAK = 0.97          # the brightest pixel's place on the LUT
FLOOR = 0.06         # dimmer than this is left black
PIN = "#bb0000"
PIN_PATH = "M50 92 C41 78 19 62 19 40 A31 31 0 1 1 81 40 C81 62 59 78 50 92Z"
HEAD = (50, 40, 27)  # the circle the PSF is cut to: cx, cy, r
GRID = (22, 12, 56)  # the pixel grid: x0, y0, size, in a 100 x 100 box


def hot(f):
    """MATLAB's hot at ``f`` in [0, 1]: red, then green, then blue ramps in."""
    f = min(1.0, max(0.0, f))
    a = 3 / 8
    rgb = (min(1, f / a), min(1, max(0, (f - a) / a)),
           max(0, (f - 2 * a) / (1 - 2 * a)))
    return "#%02x%02x%02x" % tuple(round(255 * v) for v in rgb)


def pixel(i, c, s):
    """The fraction of a 1D Gaussian at ``c`` that falls in pixel ``i``."""
    e = lambda x: math.erf((x - c) / (s * math.sqrt(2)))  # noqa: E731
    return 0.5 * (e(i + 1) - e(i))


def psf_rects():
    """The pixels, each already cut to the head's circle.

    Not an SVG clip-path: Qt's renderer does not antialias a clip, and the
    circle came out stepped while the pin's filled outline was smooth.  A
    pixel the circle crosses is drawn as its intersection with the circle
    instead, a filled polygon, antialiased like every other shape.
    """
    x0, y0, size = GRID
    w = size / N_PIX
    cx, cy = CENTRE_PIX
    v = [[pixel(i, cx, SIGMA_PIX) * pixel(j, cy, SIGMA_PIX)
          for i in range(N_PIX)] for j in range(N_PIX)]
    top = max(map(max, v))
    hx, hy, r = HEAD
    rects = []
    for j in range(N_PIX):
        for i in range(N_PIX):
            f = v[j][i] / top * PEAK
            if f < FLOOR:
                continue
            # 0.3 of overlap, so no hairline of background shows between
            # neighbouring pixels when the icon is antialiased
            x, y, s = x0 + i * w, y0 + j * w, w + 0.3
            corners = [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
            if all(math.hypot(px - hx, py - hy) <= r for px, py in corners):
                rects.append(f'  <rect x="{x:.2f}" y="{y:.2f}" width="{s:.2f}" '
                             f'height="{s:.2f}" fill="{hot(f)}"/>')
                continue
            cut = _clip(corners, _circle(hx, hy, r))
            if len(cut) >= 3:
                d = "M" + "L".join(f"{px:.2f} {py:.2f}" for px, py in cut) + "Z"
                rects.append(f'  <path d="{d}" fill="{hot(f)}"/>')
    return rects


def _circle(cx, cy, r, n=180):
    """The circle as a polygon, counter-clockwise in SVG's y-down frame."""
    return [(cx + r * math.cos(2 * math.pi * k / n),
             cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def _clip(subject, clip):
    """Sutherland-Hodgman: ``subject`` cut to the convex polygon ``clip``."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def meet(p, q, a, b):
        dx, dy = q[0] - p[0], q[1] - p[1]
        ex, ey = b[0] - a[0], b[1] - a[1]
        t = (ex * (p[1] - a[1]) - ey * (p[0] - a[0])) / (ey * dx - ex * dy)
        return p[0] + t * dx, p[1] + t * dy

    out = subject
    for a, b in zip(clip, clip[1:] + clip[:1]):
        points, out = out, []
        for p, q in zip(points, points[1:] + points[:1]):
            if inside(q, a, b):
                if not inside(p, a, b):
                    out.append(meet(p, q, a, b))
                out.append(q)
            elif inside(p, a, b):
                out.append(meet(p, q, a, b))
        if not out:
            break
    return out


def crosshair():
    x = GRID[0] + CENTRE_PIX[0] * GRID[2] / N_PIX
    y = GRID[1] + CENTRE_PIX[1] * GRID[2] / N_PIX
    a, b = 5.5, 13
    d = (f"M{x:g} {y - b:g}V{y - a:g}M{x:g} {y + a:g}V{y + b:g}"
         f"M{x - b:g} {y:g}H{x - a:g}M{x + a:g} {y:g}H{x + b:g}")
    return (f'  <path d="{d}" stroke="#000" stroke-width="2.2" '
            f'stroke-linecap="round" fill="none"/>\n'
            f'  <circle cx="{x:g}" cy="{y:g}" r="2.1" fill="#000"/>')


def svg():
    cx, cy, r = HEAD
    return "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">',
        f'  <path d="{PIN_PATH}" fill="{PIN}"/>',
        f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="#000"/>',
        *psf_rects(),
        crosshair(),
        '</svg>', ""])


def render_pngs(svg_path):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    app = QGuiApplication.instance() or QGuiApplication(["make_icon"])  # noqa: F841
    renderer = QSvgRenderer(str(svg_path))
    for n in SIZES:
        image = QImage(n, n, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        renderer.render(painter)
        painter.end()
        image.save(str(OUT / f"smappy_{n}.png"))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "smappy.svg"
    path.write_text(svg())
    if "--svg-only" not in sys.argv:
        render_pngs(path)
    print(f"wrote {path}" + ("" if "--svg-only" in sys.argv else f" and {len(SIZES)} PNGs"))
