"""Small plugin contracts: find(locs, parameters, existing),
evaluate(locs, geometry, parameters), and analyze(result_rows, parameters).

Plugins expose a stable name and version. They return data, never GUI handles;
the same plugins therefore work in scripts and the interactive manager.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, label, find_objects
from scipy.spatial import cKDTree

from .core import positive


class DensityPeaks:
    name = "density_peaks"
    version = "1"
    defaults = {"bin_nm": 20.0, "sigma_nm": 50.0, "separation_nm": 150.0,
                "count_radius_nm": 75.0, "min_count": 10}

    def find(self, locs, parameters, existing=()):
        unknown = set(parameters) - set(self.defaults)
        if unknown:
            raise ValueError(f"Unknown finder settings: {sorted(unknown)}")
        p = {**self.defaults, **parameters}
        for key in p:
            p[key] = positive(p[key], key)
        if int(p['min_count']) != p['min_count']:
            raise ValueError("min_count must be an integer")
        xy = np.column_stack((locs['x_nm'], locs['y_nm'])).astype(float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < p['min_count']:
            return []
        pixel = p['bin_nm']
        # Padding keeps peaks at the data extent from being reflected/omitted.
        pad = 4 * p['sigma_nm'] + pixel
        origin = np.floor((xy.min(axis=0) - pad) / pixel) * pixel
        shape = np.ceil((xy.max(axis=0) + pad - origin) / pixel).astype(int) + 1
        if int(shape[0]) * int(shape[1]) > 16_000_000:
            raise ValueError("Finder image exceeds 16 million pixels; increase bin_nm")
        bins = np.floor((xy - origin) / pixel).astype(int)
        density = np.zeros(tuple(shape[::-1]), dtype=float)
        np.add.at(density, (bins[:, 1], bins[:, 0]), 1)
        density = gaussian_filter(density, p['sigma_nm'] / pixel, mode='constant')
        # Small local-max neighborhood, followed by radial suppression in nm.
        maxima = (density == maximum_filter(density, size=3, mode='constant')) & (density > 0)
        labels, _ = label(maxima)
        peaks = []
        for index, region in enumerate(find_objects(labels), 1):
            if region is None:
                continue
            yy, xx = np.nonzero(labels[region] == index)
            yy, xx = yy + region[0].start, xx + region[1].start
            peaks.append((float(density[yy[0], xx[0]]), float(xx.mean()), float(yy.mean())))
        peaks.sort(key=lambda v: (-v[0], v[2], v[1]))
        tree = cKDTree(xy)
        accepted = [np.asarray(c, dtype=float) for c in existing]
        result = []
        for _, x, y in peaks:
            center = origin + (np.array([x, y]) + 0.5) * pixel
            neighbors = tree.query_ball_point(center, p['count_radius_nm'])
            if len(neighbors) < p['min_count']:
                continue
            # Recenter using the compact structure, not the full analysis ROI.
            center = xy[neighbors].mean(axis=0)
            if len(tree.query_ball_point(center, p['count_radius_nm'])) < p['min_count']:
                continue
            if any(np.linalg.norm(center - other) < p['separation_nm'] for other in accepted):
                continue
            accepted.append(center)
            result.append(center.tolist())
        return result


class Statistics:
    name = "statistics"
    version = "1"

    def evaluate(self, locs, geometry, parameters):
        if parameters:
            raise ValueError("Statistics has no adjustable parameters")
        out = {"n_localizations": len(locs)}
        for column, name in (("loc_precision_nm", "mean_precision_nm"),
                             ("photons", "mean_photons")):
            if column not in locs:
                raise ValueError(f"Statistics requires {column}")
            values = np.asarray(locs[column], dtype=float)
            finite = values[np.isfinite(values)]
            out[name] = float(finite.mean()) if len(finite) else float('nan')
            out[name + '_n'] = len(finite)
        return out


class Histograms:
    name = "histograms"
    version = "1"
    fields = ("n_localizations", "mean_precision_nm", "mean_photons")

    def analyze(self, rows, parameters=None):
        parameters = parameters or {}
        bins = parameters.get('bins', 'auto')
        out = {}
        for field in self.fields:
            values = np.asarray([r[field] for r in rows], dtype=float)
            finite = np.isfinite(values)
            counts, edges = np.histogram(values[finite], bins=bins)
            out[field] = {"counts": counts, "edges": edges,
                          "roi_ids": [r['roi_id'] for r, ok in zip(rows, finite) if ok],
                          "values": values[finite], "missing": int((~finite).sum())}
        return out
