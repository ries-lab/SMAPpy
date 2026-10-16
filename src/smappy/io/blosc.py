"""Blosc1 frames with LZ4 inside: what Imspector compresses zarr chunks with.

Imspector m2410 compresses its MINFLUX table with zlib but the beam-monitoring
(MBM) beads and the search grid with blosc and LZ4; m2205 compressed the
MINFLUX table itself that way.  Neither needs a library to read:

* **blosc1** is a 16-byte header (version, flags, the type size, the
  uncompressed size, the block size, the compressed size), a table of where
  each block starts, and the blocks -- each split into one stream per byte of
  the type when the type is small enough, each stream a 32-bit size and
  either the bytes as they are (size = the stream's) or one LZ4 block;
* the **shuffle** filter stores byte 0 of every item, then byte 1, ...: it is
  undone by a transpose.  With a structured dtype wider than 255 bytes, as
  m2205's 1030-byte records are, blosc records ``typesize = 1`` and the
  shuffle is a no-op;
* an **LZ4 block** is a sequence of tokens: literals to copy, then an offset
  and a length to repeat from what is already out.

`decode` uses ``numcodecs`` when it is installed (C, about 1 GB/s); the
decoder here is pure Python, about 2.5 MB/s -- nothing for the MBM tables,
about a second per measurement of an m2205 file.
"""
from __future__ import annotations

import struct

import numpy as np

BLOSC_MEMCPYED = 0x2
BLOSC_SHUFFLE = 0x1
BLOSC_BITSHUFFLE = 0x4
BLOSC_DONT_SPLIT = 0x10
LZ4 = 1                                   # the codec number in the flags' top bits


def lz4_block(src: bytes, out_size: int) -> bytes:
    """One LZ4 block (the block format, not the frame) of ``out_size`` bytes."""
    dst = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        literals = token >> 4
        if literals == 15:
            while True:
                b = src[i]
                i += 1
                literals += b
                if b != 255:
                    break
        dst += src[i:i + literals]
        i += literals
        if i >= n:                         # the last sequence has literals only
            break
        offset = src[i] | (src[i + 1] << 8)
        i += 2
        length = token & 15
        if length == 15:
            while True:
                b = src[i]
                i += 1
                length += b
                if b != 255:
                    break
        length += 4
        start = len(dst) - offset
        if offset <= 0 or start < 0:
            raise ValueError("damaged LZ4 block: an offset before the start")
        if offset >= length:
            dst += dst[start:start + length]
        else:                              # overlapping: a run, copied in turns
            while length > 0:
                piece = dst[start:start + min(offset, length)]
                dst += piece
                length -= len(piece)
                start += len(piece)
    if len(dst) != out_size:
        raise ValueError(f"damaged LZ4 block: {len(dst)} bytes, expected {out_size}")
    return bytes(dst)


def unshuffle(buf: bytes, typesize: int) -> bytes:
    """Undo blosc's byte shuffle; the bytes past the last whole item are as
    they were."""
    a = np.frombuffer(buf, np.uint8)
    n = len(a) // typesize
    body = a[:n * typesize].reshape(typesize, n).T.reshape(-1)
    return body.tobytes() + a[n * typesize:].tobytes()


def _decode_python(frame: bytes) -> bytes:
    (_version, _versionlz, flags, typesize, nbytes, blocksize,
     _cbytes) = struct.unpack_from("<BBBBIII", frame, 0)
    if flags & BLOSC_MEMCPYED:
        return bytes(frame[16:16 + nbytes])
    if flags & BLOSC_BITSHUFFLE:
        raise ValueError("blosc bit-shuffle is not read without numcodecs")
    codec = flags >> 5
    if codec != LZ4:
        raise ValueError(f"blosc codec {codec} is not read without numcodecs "
                         "(pip install numcodecs)")
    n_blocks = (nbytes + blocksize - 1) // blocksize
    starts = struct.unpack_from(f"<{n_blocks}I", frame, 16)
    out = bytearray()
    for b, start in enumerate(starts):
        size = blocksize if b < n_blocks - 1 else nbytes - blocksize * (n_blocks - 1)
        split = (not flags & BLOSC_DONT_SPLIT and typesize <= 16
                 and size // typesize >= 128 and size % typesize == 0)
        n_streams = typesize if split else 1
        stream_size = size // n_streams
        p, block = start, bytearray()
        for _ in range(n_streams):
            csize, = struct.unpack_from("<i", frame, p)
            p += 4
            if csize == stream_size:       # stored as it is
                block += frame[p:p + csize]
            else:
                block += lz4_block(frame[p:p + csize], stream_size)
            p += csize
        if flags & BLOSC_SHUFFLE and typesize > 1:
            block = unshuffle(bytes(block), typesize)
        out += block
    return bytes(out)


def decode(frame: bytes) -> bytes:
    """A blosc1 frame's bytes, by numcodecs where it is installed."""
    try:
        from numcodecs import Blosc
    except ImportError:
        return _decode_python(frame)
    return bytes(Blosc().decode(frame))
