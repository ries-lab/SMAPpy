"""The DECODE fitter's network on a workstation: frames streamed there over ssh, localizations back.

Inference stays SMAPpy's fitter (plan, "The fitter run on a workstation"): the movie is read here,
cut to the camera ROI, turned into photons with this computer's camera, cut into blocks with their
context by `window_engine.WindowEngine` -- and only the network, the one step a CPU is too slow for
(0.7 frames a second of 512 x 512 on a laptop's, 50 -- 70 on a workstation's GPU), runs there.
`RemotePredictor` stands in for `decode_smlm.predict.Predictor`: the engine calls it the same way.

**The worker** is ``<host command> serve`` (`decode_smlm.serve`) started over ``ssh`` with the
training hosts' settings (`training.jobs.Host`), so a workstation that trains also localizes.  It
gets the model file at the start -- the model need not be there -- and then a block of frames at a
time, and answers each with the block's localizations in the predictor's columns.

**The messages** on the ssh pipe are a 4-byte tag, an 8-byte little-endian length and the payload:
``HELO`` (JSON: device, mirror, threshold, batch) and ``MODL`` (the model file's bytes) from here,
``INFO`` (JSON: window, metadata, device) back; then ``FRMS`` (the first frame's number, 8 bytes,
and the frames as ``.npy``, float32 photons) and ``COLS`` (the columns as ``.npz``) back, or ``ERRR``
(the worker's error, as text); ``QUIT``, or the pipe closed, ends the worker.  Frames go as float32:
512 x 512 is 1 MB a frame, 50 MB a second at a GPU's pace, which a gigabit network carries.
"""
from __future__ import annotations

import io
import json
import struct
import subprocess
import threading
import weakref
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<4sQ")


class RemoteFitError(RuntimeError):
    """The worker's error, or the connection's."""


def write_message(stream, tag: bytes, payload: bytes = b"") -> None:
    stream.write(HEADER.pack(tag, len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exactly(stream, n: int) -> bytes:
    chunks, left = [], n
    while left:
        chunk = stream.read(left)
        if not chunk:
            raise EOFError(f"the stream ended {left} bytes short")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def read_message(stream):
    """``(tag, payload)``; ``(None, b"")`` at the end of the stream."""
    head = stream.read(HEADER.size)
    if not head:
        return None, b""
    if len(head) < HEADER.size:
        head += _read_exactly(stream, HEADER.size - len(head))
    tag, length = HEADER.unpack(head)
    return tag, _read_exactly(stream, length)


def pack_frames(photons: np.ndarray, first_frame: int) -> bytes:
    buffer = io.BytesIO()
    buffer.write(struct.pack("<q", int(first_frame)))
    np.save(buffer, np.ascontiguousarray(photons, np.float32), allow_pickle=False)
    return buffer.getvalue()


def unpack_frames(payload: bytes):
    first = struct.unpack("<q", payload[:8])[0]
    return np.load(io.BytesIO(payload[8:]), allow_pickle=False), first


def pack_columns(cols: dict) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **{k: np.asarray(v) for k, v in cols.items()})
    return buffer.getvalue()


def unpack_columns(payload: bytes) -> dict:
    with np.load(io.BytesIO(payload), allow_pickle=False) as found:
        return {k: found[k] for k in found.files}


class RemotePredictor:
    """`decode_smlm.predict.Predictor`'s interface, run by a worker on a training host."""

    def __init__(self, host, model_path, device: str | None = None, *, frames_mirrored: bool = False,
                 raw_th: float = 0.1, batch_size: int = 32, popen=None):
        from .training.remote import SSH_OPTIONS
        self.host = host
        self.frames_mirrored, self.raw_th, self.batch_size = bool(frames_mirrored), float(raw_th), int(batch_size)
        command = ["ssh", *SSH_OPTIONS, host.ssh, f"{host.command} serve"]
        popen = popen or subprocess.Popen      # looked up at the call: a test replaces it
        try:
            self.process = popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as e:
            raise RemoteFitError("no ssh on this computer") from e
        self._said = []                    # the worker's stderr, for an error message
        self._reader = threading.Thread(target=self._collect, args=(self.process.stderr,), daemon=True)
        self._reader.start()
        self._closer = weakref.finalize(self, _close, self.process)
        hello = {"device": device or "auto", "frames_mirrored": self.frames_mirrored, "raw_th": self.raw_th,
                 "batch_size": self.batch_size}
        info = self._ask([(b"HELO", json.dumps(hello).encode()), (b"MODL", Path(model_path).read_bytes())], b"INFO")
        told = json.loads(info)
        self.window, self.metadata, self.device = int(told["window"]), told["metadata"], f"{host.name}: {told['device']}"

    def _collect(self, stream) -> None:
        for line in iter(stream.readline, b""):
            self._said = (self._said + [line.decode("utf-8", "replace").rstrip()])[-20:]

    def _ask(self, messages, expect: bytes) -> bytes:
        try:
            for tag, payload in messages:
                write_message(self.process.stdin, tag, payload)
            tag, payload = read_message(self.process.stdout)
        except (BrokenPipeError, EOFError, OSError, ValueError) as e:
            raise RemoteFitError(self._why(str(e))) from e
        if tag == b"ERRR":
            raise RemoteFitError(f"{self.host.name}: {payload.decode('utf-8', 'replace')}")
        if tag != expect:
            raise RemoteFitError(self._why(f"expected {expect.decode()}, got {tag!r}"))
        return payload

    def _why(self, what: str) -> str:
        self._reader.join(timeout=2)
        said = [line for line in self._said if line.strip()]
        return f"{self.host.name}: " + (said[-1] if said else what)

    def __call__(self, photons, first_frame: int = 0) -> dict:
        photons = np.asarray(photons, np.float32)
        if photons.ndim != 3 or len(photons) <= 2 * self.window:
            raise ValueError(f"frames (n + {2 * self.window}, rows, columns), not {photons.shape}")
        return unpack_columns(self._ask([(b"FRMS", pack_frames(photons, first_frame))], b"COLS"))

    def close(self) -> None:
        self._closer()


def _close(process) -> None:
    try:
        write_message(process.stdin, b"QUIT")
        process.stdin.close()
        process.wait(timeout=10)
    except Exception:                       # a worker already gone, or one that would not stop
        process.kill()
