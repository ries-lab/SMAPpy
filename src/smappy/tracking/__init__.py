"""Single-particle tracking: linking localizations into tracks (`link`), how
they move (`analysis`), and which column names them (`columns`).

A track is a region: ``track_id`` is an id column like ``cluster_id``, so
every ``per_region`` plugin runs per track (`docs/regions.md`).
"""
from .columns import (AUTO, TRACK_COLUMNS, auto_track_column, integer_columns,
                      track_column, track_column_choices)
from .link import (ALGORITHMS, Candidates, candidates, link_global, link_simple,
                   link_tracks, score, track_ids)

__all__ = ["ALGORITHMS", "AUTO", "Candidates", "TRACK_COLUMNS", "auto_track_column",
           "candidates", "integer_columns", "link_global", "link_simple",
           "link_tracks", "score", "track_column", "track_column_choices",
           "track_ids"]
