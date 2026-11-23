"""The network a model file holds, ready to localize: one loader for every method the DECODE fitter runs.

A model file names its method (``method`` in its header, `model_file`): ``decode-v0.10.2``, the port
of DECODE (`decode_smlm.predict`), ``netfit-v1``, SMAPpy's own (`smappy.netfit.predict`), or
``decode-plex``, a published DECODE-Plex model exported (`decode_smlm.plex.fitter`: the baseline,
its frames cut and converted by DECODE-Plex's own steps).  The
fitter (`plugins.fit_decode`) and the server on a training host (`decode_smlm.serve`) both load
through `load`, so a model fits the same wherever it runs.  Both predictors give the
`window_engine.WindowEngine` one interface: ``window``, a call on frames in photons with their
context giving SMAPpy's pixel columns, and ``device``, ``raw_th``, ``frames_mirrored``, ``metadata``
for the run's record.

Nothing here imports torch: the form reads a model's header to say what it is (`describe`).
"""
from __future__ import annotations

from pathlib import Path

DECODE = "decode-v0.10.2"
NETFIT = "netfit-v1"
PLEX = "decode-plex"
METHODS = (DECODE, NETFIT, PLEX)


def describe(meta: dict) -> str:
    """What a model is, in the GUI's words: DECODE, whatever trained it (docs/decode/gui-plan.md §1)."""
    method = meta.get("method")
    if method == NETFIT:
        return "DECODE, aligned channels" if (meta.get("channels") or {}) else "DECODE, optimised"
    if method == PLEX:
        return "DECODE-Plex"
    return "DECODE"


def two_channel(meta: dict) -> bool:
    """Whether a model reads a split camera's two channels (plex-plan §16)."""
    return bool(meta.get("channels"))


def field_dependent(meta: dict) -> bool:
    """Whether a model was trained on a field-dependent PSF and reads u, v maps (`netfit.field`)."""
    return bool(meta.get("field"))


def load(path, device=None, *, frames_mirrored: bool = False, threshold: float | None = None, batch: int = 0,
         corner=(0, 0)):
    """The model file at ``path`` as a predictor on ``device`` (None or "auto": the fastest there is).
    ``threshold``: the detection probability a localization starts from (DECODE's ``raw_th``; None:
    the model's own); ``batch``: frames through the network at once, 0 as many as memory allows;
    ``corner`` (x, y): the frames' corner on the chip, the camera ROI's offset, which a field-dependent
    model's maps are made from (`netfit.field`)."""
    from . import model_file
    meta = model_file.read_metadata(Path(path))
    method = meta.get("method")
    if method == DECODE:
        from decode_smlm.predict import RAW_TH, load_predictor
        return load_predictor(path, device, frames_mirrored=frames_mirrored,
                              raw_th=RAW_TH if threshold is None else threshold, batch_size=batch)
    if method == NETFIT:
        if two_channel(meta):
            raise ValueError(f"{Path(path).name} reads two channels: its frames are cut into them first "
                             f"(the DECODE fitter's two channels, `netfit.fit_two`); this loads one-channel models")
        from ..netfit.predict import Predictor
        return Predictor(path, None if device in (None, "auto") else device, frames_mirrored=frames_mirrored,
                         threshold=threshold, batch=batch, corner=corner)
    if method == PLEX:
        from decode_smlm.plex.fitter import PlexPredictor
        return PlexPredictor(path, None if device in (None, "auto") else device, threshold=threshold, batch=batch)
    raise ValueError(f"{Path(path).name} is not a DECODE model" + (f" (it is {method})" if method else ""))
