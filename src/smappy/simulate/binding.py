"""Simulated DNA-PAINT: spots of binding sites, each visited by an imager.

What *Binding kinetics* and its qPAINT summary are tested and illustrated
on.  The blinking model of `simulate.kinetics` is a dye's photophysics in
frames; DNA-PAINT is an exchange, and what qPAINT reads from it is the
time between bindings, so the model here is that exchange in continuous
time:

* **a spot** (a DNA origami, a protein) carries ``sites`` binding sites at
  one place, within ``spread_nm`` of its centre;
* **a binding site** alternates between free and bound for exponential
  times, ``tau_dark`` and ``tau_bright`` frames on average, starting in
  its stationary state (bound with probability ``tau_bright / (tau_bright
  + tau_dark)``).  ``tau_dark`` is one site's: a spot of n sites is bound
  n times as often, which is what qPAINT counts;
* **a frame** shows a site when the imager was bound for at least half of
  it (`dark_times`' convention in Picasso, under which the frames counted
  bright and dark are unbiased estimates of the continuous times), and
  gives one localization at the site's place, ``precision_nm`` per axis,
  for each such site;
* **a sticky event** (``sticky``): a spot held bright from one frame to
  another, as an imager stuck to the surface or to a damaged strand is --
  the step in a trace that curation removes.

The truth is ``spot_true`` (1, 2, ...) and ``site_true`` (1, 2, ...,
across the spots; 0 for a sticky event's localizations); the metadata
keeps each spot's number of sites (``sites_true``), the rates and a frame
time, so seconds can be read.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from ..locs import Localizations


def bound_intervals(n_sites: int, n_frames: int, tau_bright: float, tau_dark: float,
                    rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(site, start, end)`` of every binding of ``n_sites`` independent
    sites in ``[0, n_frames)``, continuous time in frames, the sites
    alternating free and bound from their stationary state."""
    sites, starts, ends = [], [], []
    cycle = tau_bright + tau_dark
    # enough cycles that every site outlasts the acquisition, drawn at once
    k = int(np.ceil(n_frames / cycle * 1.5 + 6 * np.sqrt(n_frames / cycle) + 10))
    for s in range(n_sites):
        bound_first = rng.random() < tau_bright / cycle
        dark = rng.exponential(tau_dark, k)
        bright = rng.exponential(tau_bright, k)
        if bound_first:
            # bound at the start; by memorylessness what is left of that
            # binding is a whole one
            dark[0] = 0.0
        t = np.cumsum(np.column_stack([dark, bright]).ravel())
        begin, finish = t[0::2], t[1::2]
        while finish[-1] < n_frames:          # rare: more cycles than drawn
            more = np.cumsum(np.column_stack([rng.exponential(tau_dark, k),
                                              rng.exponential(tau_bright, k)]).ravel())
            more += finish[-1]
            begin, finish = np.append(begin, more[0::2]), np.append(finish, more[1::2])
        keep = begin < n_frames
        sites.append(np.full(int(keep.sum()), s))
        starts.append(begin[keep])
        ends.append(np.minimum(finish[keep], n_frames))
    return (np.concatenate(sites).astype(np.int64), np.concatenate(starts),
            np.concatenate(ends))


def bright_frames(site, start, end) -> Tuple[np.ndarray, np.ndarray]:
    """``(site, frame)`` of the frames each site was bound for at least half
    of; a frame's coverage summed over the site's bindings in it."""
    first = np.floor(start).astype(np.int64)
    last = np.ceil(end).astype(np.int64) - 1
    span = np.maximum(last - first + 1, 0)
    which = np.repeat(np.arange(len(first)), span)
    frame = first[which] + (np.arange(len(which)) - np.repeat(np.cumsum(span) - span,
                                                                span))
    cover = (np.minimum(end[which], frame + 1.0) - np.maximum(start[which], frame))
    owner = site[which]
    if not len(frame):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    key = owner * (int(frame.max()) + 1) + frame
    unique, inverse = np.unique(key, return_inverse=True)
    total = np.bincount(inverse.reshape(-1), weights=cover)
    lit = total >= 0.5
    width = int(frame.max()) + 1
    return unique[lit] // width, unique[lit] % width


def simulate_binding_sites(sites: Sequence[int], n_frames: int = 20000, *,
                           tau_bright: float = 5.0, tau_dark: float = 2000.0,
                           spacing_nm: float = 500.0, spread_nm: float = 5.0,
                           precision_nm: float = 5.0, photons: float = 3000.0,
                           sticky: Sequence[Tuple[int, int, int]] = (),
                           frame_time_ms: float = 100.0,
                           seed: int = 0) -> Localizations:
    """Spots on a square grid ``spacing_nm`` apart, spot ``i`` with
    ``sites[i]`` binding sites.  ``tau_bright`` and ``tau_dark`` are one
    site's mean bound and free times in frames; ``sticky`` holds ``(spot,
    first, last)``: spot (0-based) bright in every frame from first to last.
    """
    rng = np.random.default_rng([seed, 11])
    sites = [int(n) for n in sites]
    side = int(np.ceil(np.sqrt(max(len(sites), 1))))
    centre = np.array([[(i % side + 0.5) * spacing_nm, (i // side + 0.5) * spacing_nm]
                       for i in range(len(sites))]).reshape(-1, 2)
    owner = np.repeat(np.arange(len(sites)), sites)          # each site's spot
    where = centre[owner] + rng.normal(0.0, spread_nm, (len(owner), 2))
    site, start, end = bound_intervals(len(owner), n_frames, tau_bright, tau_dark, rng)
    site, frame = bright_frames(site, start, end)
    spot = owner[site]
    xy = where[site]
    site_id = site + 1
    for which, first, last in sticky:
        f = np.arange(int(first), int(last) + 1)
        frame = np.append(frame, f)
        spot = np.append(spot, np.full(len(f), int(which)))
        xy = np.vstack([xy, np.repeat(centre[[int(which)]], len(f), axis=0)])
        site_id = np.append(site_id, np.zeros(len(f), np.int64))
    order = np.argsort(frame, kind="stable")
    frame, spot, xy, site_id = frame[order], spot[order], xy[order], site_id[order]
    n = len(frame)
    noise = rng.normal(0.0, precision_nm, (n, 2))
    columns = {
        "frame": frame.astype(np.int64),
        "x_nm": (xy[:, 0] + noise[:, 0]).astype(np.float32),
        "y_nm": (xy[:, 1] + noise[:, 1]).astype(np.float32),
        "photons": rng.poisson(photons, n).astype(np.float32),
        "xy_err_nm": np.full(n, precision_nm, np.float32),
        "spot_true": (spot + 1).astype(np.int32),
        "site_true": site_id.astype(np.int32),
    }
    return Localizations(columns, {
        "units": "nm", "simulation": "smappy binding sites", "n_frames": int(n_frames),
        "seed": seed, "sites_true": sites, "tau_bright_frames": float(tau_bright),
        "tau_dark_frames": float(tau_dark), "spacing_nm": float(spacing_nm),
        # where every reader of seconds looks (`metadata.file_frame_time`)
        "camera": {"frame_time_ms": float(frame_time_ms)}})


def spot_centres(locs: Localizations) -> np.ndarray:
    """Each spot's centre, (n, 2) in nm, in the order of ``spot_true``: where
    a test or a page puts its ROIs."""
    sites = locs.metadata["sites_true"]
    spacing = float(locs.metadata["spacing_nm"])
    side = int(np.ceil(np.sqrt(max(len(sites), 1))))
    return np.array([[(i % side + 0.5) * spacing, (i // side + 0.5) * spacing]
                     for i in range(len(sites))]).reshape(-1, 2)
