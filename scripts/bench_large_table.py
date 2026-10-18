"""Where the time goes on a large table of several files.

    python scripts/bench_large_table.py --files 10 --per-file 10000000

builds one table of ``files x per_file`` localizations (a 2D fit's columns,
blinks of a few frames, each file 50 000 frames over the same 40 um field),
opens it in a `Session` as that many files, and times what a user pays for
through the session's own paths: the layer's filter, choosing one file, a
slider, a plugin's context and selection, a plugin, the render, the spatial
index, linking.  The numbers in `docs/performance.md` ("Large tables") are
this script's.  Peak memory is printed with each step (``ru_maxrss``).
"""
from __future__ import annotations

import argparse
import gc
import resource
import time

import numpy as np

from smappy.io.formats import FileInfo
from smappy.locs import Localizations
from smappy.session import Session

FIELD_NM = 40_000.0
FRAMES = 50_000


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def timed(label, fn, *args, repeat=1, **kwargs):
    best, out = np.inf, None
    for _ in range(repeat):
        gc.collect()
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t)
    print(f"{label:<58s} {best * 1e3:10.1f} ms   peak {peak_gb():5.1f} GB", flush=True)
    return out


def fill_file(columns, start, n, seed):
    """One file's localizations, written into the preallocated columns:
    emitters blinking for a geometric number of frames, sorted by frame."""
    rng = np.random.default_rng(seed)
    n_emitters = max(1, n // 50)
    ex = (rng.random(n_emitters) * FIELD_NM).astype(np.float32)
    ey = (rng.random(n_emitters) * FIELD_NM).astype(np.float32)
    lengths = rng.geometric(0.5, size=n)            # blinks of ~2 frames
    blinks = int(np.searchsorted(np.cumsum(lengths), n)) + 1
    lengths = lengths[:blinks]
    emitter = rng.integers(0, n_emitters, blinks)
    first = rng.integers(0, FRAMES, blinks)
    owner = np.repeat(np.arange(blinks), lengths)[:n]
    offset = np.arange(len(owner)) - np.repeat(np.cumsum(lengths) - lengths, lengths)[:n]
    frame = np.minimum(first[owner] + offset, FRAMES - 1).astype(np.int32)
    order = np.argsort(frame, kind="stable")
    owner, frame = owner[order], frame[order]
    sl = slice(start, start + n)
    photons = rng.lognormal(7.0, 0.6, n).astype(np.float32)
    err = (150.0 / np.sqrt(photons)).astype(np.float32)
    columns["x_nm"][sl] = ex[emitter[owner]] + rng.standard_normal(n, np.float32) * err
    columns["y_nm"][sl] = ey[emitter[owner]] + rng.standard_normal(n, np.float32) * err
    columns["frame"][sl] = frame
    columns["photons"][sl] = photons
    columns["bg"][sl] = rng.normal(20.0, 3.0, n).astype(np.float32)
    columns["xy_err_nm"][sl] = err
    columns["sigma_nm"][sl] = rng.normal(130.0, 10.0, n).astype(np.float32)
    columns["logl_rel"][sl] = rng.normal(0.0, 1.0, n).astype(np.float32)


def build(files: int, per_file: int) -> Localizations:
    n = files * per_file
    columns = {name: np.empty(n, np.float32) for name in
               ("x_nm", "y_nm", "photons", "bg", "xy_err_nm", "sigma_nm", "logl_rel")}
    columns["frame"] = np.empty(n, np.int32)
    columns["filenumber"] = np.repeat(np.arange(files, dtype=np.int32), per_file)
    for k in range(files):
        fill_file(columns, k * per_file, per_file, seed=k)
    return Localizations(columns, {"camera": {"pixelsize": 0.1}})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--files", type=int, default=10)
    parser.add_argument("--per-file", type=int, default=10_000_000)
    parser.add_argument("--no-group", action="store_true", help="skip linking")
    args = parser.parse_args()
    files, per_file = args.files, args.per_file
    n = files * per_file
    print(f"{files} files x {per_file:,} = {n:,} localizations")

    locs = timed("simulate", build, files, per_file)
    print(f"table {sum(v.nbytes for v in locs.columns.values()) / 1e9:.1f} GB "
          f"in {len(locs.columns)} columns")
    infos = [FileInfo(f"file{k}", "", "bench", n=per_file,
                      columns=[c for c in locs.columns if c != "filenumber"])
             for k in range(files)]
    locs.metadata["files"] = [f.to_dict() for f in infos]

    def open_session():
        s = Session(locs)
        s.files = infos
        return s
    s = timed("Session(locs): table, spatial index, one layer", open_session)
    layer = s.layers[0]
    layer.show_grouped(False)          # localizations first; blinks at the end
    state = layer.state
    filt = timed("layer filter: built from the default bounds",
                 lambda: state.filter_for(state.current))
    timed("  ... its mask (AND)", lambda: filt.mask)
    print(f"    filters on: {', '.join(filt._masks)}")

    chosen = min(3, files - 1)          # the file a user looks at
    timed("show one file (set_files: isin over the table, AND)",
          lambda: (layer.set_files([chosen]), state.filter_for(state.current).mask))
    timed("move a slider (photons >= 500) and AND", lambda: (
        layer.set_bound("photons", 500.0, None), state.filter_for(state.current).mask))

    ctx = timed("plugin context for the layer (Session.context)", s.context, 0)
    sel = ctx.selection
    print(f"    selection {len(sel):,} rows of {len(ctx.locs):,}")
    timed("selection.apply: every column of the selected rows", sel.apply, ctx.locs)
    rows = np.flatnonzero(sel.mask)
    timed("  the same, three columns (x, y, photons)",
          lambda: {c: np.asarray(ctx.locs[c])[rows] for c in ("x_nm", "y_nm", "photons")})
    timed("  np.flatnonzero(mask) alone", np.flatnonzero, sel.mask)
    first, last = chosen * per_file, (chosen + 1) * per_file
    timed("  one file as a slice: a view of every column (no copy)",
          lambda: Localizations({k: v[first:last] for k, v in ctx.locs.columns.items()}))

    from smappy.plugins.statistics import LocalizationStatistics, StatisticsSettings
    timed("Localization Statistics on the selection",
          LocalizationStatistics().run, ctx, StatisticsSettings())
    del ctx, sel, rows

    from smappy.layer_state import snapshot_layers
    from smappy.picture import render_picture
    from smappy.render import FieldOfView
    full = FieldOfView(0.0, 0.0, 20.0, 2000, 2000)
    zoom = FieldOfView(18_000.0, 18_000.0, 2.0, 2000, 2000)
    snap = snapshot_layers(s.layers)
    timed("render one file, full field 2000x2000", render_picture, snap, full,
          what="intensity", repeat=2)
    timed("render one file, zoomed 10x", render_picture, snap, zoom,
          what="intensity", repeat=2)
    layer.set_files(None)
    layer.set_bound("photons", None, None)
    snap = snapshot_layers(s.layers)
    timed(f"render all {files} files, full field 2000x2000", render_picture, snap,
          full, what="intensity", repeat=2)
    timed(f"render all {files} files, zoomed 10x", render_picture, snap, zoom,
          what="intensity", repeat=2)
    del snap

    if not args.no_group:
        from smappy.group import GroupSettings, group
        one = Localizations({k: v[first:last] for k, v in locs.columns.items()}, {})
        timed(f"link one file ({per_file:,}) into blinks", group, one,
              GroupSettings(dx=50.0, dt=1))
        del one
        layer.show_grouped(True)
        timed(f"link the whole table ({n:,}): the layer shows blinks",
              lambda: state.current)
        layer.set_files([chosen])
        ctx = timed("grouped: plugin context, one file", s.context, 0)
        print(f"    selection {len(ctx.selection):,} blinks of {len(ctx.locs):,}")
        timed("grouped: selection.apply", ctx.selection.apply, ctx.locs)


if __name__ == "__main__":
    main()
