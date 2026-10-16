"""Inflate the uint8 MFXDTA stacks of an .msr and dump them (research only)."""
import struct, sys, zlib, re

def rs(f):
    n, = struct.unpack('<I', f.read(4))
    return f.read(n)

def stacks(path):
    f = open(path, 'rb')
    f.seek(14); first, = struct.unpack('<Q', f.read(8))
    pos = first; i = 0
    while pos:
        f.seek(pos + 16)
        sv, rank = struct.unpack('<II', f.read(8))
        res = struct.unpack('<15I', f.read(60))[:rank]
        f.read(240)
        dt, ct, cl, nl, dl, rsv, dlen_disk, nxt = struct.unpack('<IIIIIQQQ', f.read(44))
        name = f.read(nl).decode(); descr = f.read(dl).decode()
        data_pos = f.tell(); f.seek(data_pos + dlen_disk)
        fs, = struct.unpack('<I', f.read(4)); hcp = struct.unpack('<15I', f.read(60)); hcl = struct.unpack('<15I', f.read(60))
        mlen, = struct.unpack('<I', f.read(4)); f.read(1280)
        nfp, fbs, tdl = struct.unpack('<QQQ', f.read(24)); f.read(20)
        sw, ncp = struct.unpack('<QQ', f.read(16))
        f.seek(data_pos + dlen_disk + fs)
        [rs(f) for _ in range(rank)]
        f.read(mlen); f.read(8 * nfp)
        tags = {}
        while True:
            k = rs(f)
            if not k: break
            tags[k.decode()] = rs(f).decode('utf8', 'replace')
        yield i, dict(rank=rank, res=res, dt=dt, ct=ct, name=name, descr=descr, data_pos=data_pos, dlen=dlen_disk, sw=sw, ncp=ncp, tags=tags)
        pos = nxt; i += 1

def blob(path, s):
    with open(path, 'rb') as f:
        f.seek(s['data_pos']); raw = f.read(s['dlen'])
    return zlib.decompress(raw) if s['ct'] else raw

if __name__ == '__main__':
    path, out = sys.argv[1], sys.argv[2]
    for i, s in stacks(path):
        if s['dt'] == 1 and s['res'][1:] == (1, 1, 1):
            b = blob(path, s)
            print(i, s['descr'], len(b), s['sw'])
            print('  minflux tag:', s['tags'].get('minflux'))
            print('  imspector tag head:', s['tags'].get('imspector', '')[:300])
            open(f'{out}/stack{i}.bin', 'wb').write(b)
        elif i < 3 or 'MF(' in s['name']:
            print(i, s['name'], 'minflux tag:', s['tags'].get('minflux'))
