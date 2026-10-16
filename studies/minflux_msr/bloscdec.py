"""Minimal pure-Python blosc1 + LZ4 block decoder, to validate the msr-embedded zarr (research only)."""
import struct, sys, json, os
import numpy as np

def lz4_block(src, out_size):
    dst = bytearray(); i = 0; n = len(src)
    while i < n:
        tok = src[i]; i += 1
        lit = tok >> 4
        if lit == 15:
            while True:
                b = src[i]; i += 1; lit += b
                if b != 255: break
        dst += src[i:i + lit]; i += lit
        if i >= n: break
        off = src[i] | (src[i + 1] << 8); i += 2
        ml = tok & 15
        if ml == 15:
            while True:
                b = src[i]; i += 1; ml += b
                if b != 255: break
        ml += 4
        start = len(dst) - off
        if off >= ml:
            dst += dst[start:start + ml]
        else:
            for k in range(ml): dst.append(dst[start + k])
    assert len(dst) == out_size, (len(dst), out_size)
    return bytes(dst)

def unshuffle(buf, typesize):
    a = np.frombuffer(buf, np.uint8)
    n = len(a) // typesize
    body = a[:n * typesize].reshape(typesize, n).T.reshape(-1)
    return body.tobytes() + a[n * typesize:].tobytes()

def blosc1(c):
    ver, verlz, flags, typesize, nbytes, blocksize, cbytes = struct.unpack_from('<BBBBIII', c, 0)
    shuffle = flags & 0x1; memcpyed = flags & 0x2; bitshuffle = flags & 0x4; dont_split = flags & 0x10
    codec = flags >> 5
    info = dict(ver=ver, verlz=verlz, flags=hex(flags), typesize=typesize, nbytes=nbytes, blocksize=blocksize, cbytes=cbytes, codec=codec)
    if memcpyed:
        return c[16:16 + nbytes], info
    nblocks = (nbytes + blocksize - 1) // blocksize
    bstarts = struct.unpack_from(f'<{nblocks}I', c, 16)
    out = bytearray()
    for bi, bs in enumerate(bstarts):
        bsize = blocksize if bi < nblocks - 1 else nbytes - blocksize * (nblocks - 1)
        nstreams = typesize if (not dont_split and typesize <= 16 and bsize // typesize >= 128 and bsize % typesize == 0) else 1
        ssize = bsize // nstreams
        p = bs; blk = bytearray()
        for _ in range(nstreams):
            cs, = struct.unpack_from('<i', c, p); p += 4
            if cs == ssize:
                blk += c[p:p + cs]
            else:
                blk += lz4_block(c[p:p + cs], ssize)
            p += cs
        if shuffle and typesize > 1:
            blk = unshuffle(bytes(blk), typesize)
        out += blk
    return bytes(out), info

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mfxdump import stacks, blob
    from mfxparse import parse
    p = sys.argv[1]
    for i, s in stacks(p):
        if s['dt'] == 1 and s['res'][1:] == (1, 1, 1):
            b = blob(p, s); ents, _ = parse(b, 147)
            files = {e[0]: b[e[3]:e[3] + e[2]] for e in ents if e[1] == 'file'}
            za = json.loads(files['zarr\\mfx\\.zarray'])
            def todt(d):
                return np.dtype([(f[0], todt(f[1]) if isinstance(f[1], list) else f[1], *( [tuple(f[2])] if len(f) > 2 else [])) for f in d])
            dt = todt(za['dtype'])
            print('itemsize', dt.itemsize, 'shape', za['shape'])
            parts = []
            for k in sorted((k for k in files if k.startswith('zarr\\mfx\\') and k.split('\\')[-1].isdigit()), key=lambda k: int(k.split('\\')[-1])):
                raw, info = blosc1(files[k]); print(k, info)
                parts.append(np.frombuffer(raw, dt))
            arr = np.concatenate(parts)[:za['shape'][0]]
            print('vld', arr['vld'].sum(), 'of', len(arr), 'tid range', arr['tid'].min(), arr['tid'].max(), 'tim', arr['tim'][:3], arr['tim'][-1])
            v = arr[arr['vld']]
            print('last-iter loc (nm) sample', (v['itr']['loc'][:3, -1] * 1e9).round(1), 'efo', v['itr']['efo'][:3, -1], 'cfr', v['itr']['cfr'][:3])
            break
