"""Single-particle tracking: linking localizations into tracks (`link`).

A track is a region: ``track_id`` is an id column like ``cluster_id``, so
every ``per_region`` plugin runs per track (`docs/regions.md`).
"""
from .link import (ALGORITHMS, Candidates, candidates, link_global, link_simple,
                   link_tracks, score, track_ids)

__all__ = ["ALGORITHMS", "Candidates", "candidates", "link_global", "link_simple",
           "link_tracks", "score", "track_ids"]
