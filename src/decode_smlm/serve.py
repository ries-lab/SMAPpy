"""``smappy-decode serve``: a model's predictor for a SMAPpy on another computer, over stdin and stdout.

SMAPpy's DECODE fitter, set to run its network on a training host, starts this over ssh
(`smappy.decode_remote.RemotePredictor`) and keeps it for one fit: the model file comes first, then
blocks of frames in photons, each answered with its localizations, as `predict.Predictor` gives
them.  The messages are `smappy.decode_remote`'s.

stdout carries the messages alone: whatever the libraries print goes to stderr, which ssh brings back
for the fit's error message.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def serve(stdin=None, stdout=None) -> int:
    from smappy.decode_remote import pack_columns, read_message, unpack_frames, write_message

    if stdout is None:                      # the real stdout for the messages; print() to stderr
        stdout = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    stdin = stdin if stdin is not None else sys.stdin.buffer
    hello, predictor = {}, None
    while True:
        tag, payload = read_message(stdin)
        if tag in (None, b"QUIT"):
            return 0
        try:
            if tag == b"HELO":
                hello = json.loads(payload)
            elif tag == b"MODL":
                predictor = _load(payload, hello)
                write_message(stdout, b"INFO", json.dumps({"window": predictor.window, "device": predictor.device,
                                                           "metadata": predictor.metadata}).encode())
            elif tag == b"FRMS":
                if predictor is None:
                    raise ValueError("frames before a model")
                photons, first = unpack_frames(payload)
                write_message(stdout, b"COLS", pack_columns(predictor(photons, first)))
            else:
                raise ValueError(f"unknown message {tag!r}")
        except Exception as error:          # said to the fitter, which stops; the worker ends with it
            write_message(stdout, b"ERRR", f"{type(error).__name__}: {error}".encode())


def _load(model: bytes, hello: dict):
    from .predict import load_predictor
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.safetensors"
        path.write_bytes(model)
        device = hello.get("device", "auto")
        return load_predictor(path, None if device == "auto" else device,
                              frames_mirrored=bool(hello.get("frames_mirrored", False)),
                              raw_th=float(hello.get("raw_th", 0.1)), batch_size=int(hello.get("batch_size", 32)))
