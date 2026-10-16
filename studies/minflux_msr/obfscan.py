"""Walk the OBF stacks of an .msr file and print what is where (research only)."""
import struct, sys, zlib

def rs(f):
    n, = struct.unpack('<I', f.read(4))
    return f.read(n)

def scan(path, dump_dir=None):
    f = open(path, 'rb')
    size = f.seek(0, 2); f.seek(0)
    magic = f.read(10)
    ver, first, dlen = struct.unpack('<IQI', f.read(16))
    descr = f.read(dlen)
    meta_pos, = struct.unpack('<Q', f.read(8)) if ver >= 2 else (0,)
    print(f'file {path} size {size} magic {magic!r} ver {ver} first {first} descr_len {dlen} meta_pos {meta_pos} hdr_end {f.tell()}')
    if meta_pos:
        f.seek(meta_pos)
        keys = []
        while True:
            k = rs(f)
            if not k: break
            v = rs(f); keys.append((k, len(v), v[:120]))
        print('  file meta:', [(k, n) for k, n, _ in keys], 'ends at', f.tell())
    pos = first; i = 0; regions = [(0, f.tell() if not meta_pos else 0)]
    while pos:
        f.seek(pos)
        m = f.read(16)
        sv, rank = struct.unpack('<II', f.read(8))
        res = struct.unpack('<15I', f.read(60))[:rank]
        ln = struct.unpack('<15d', f.read(120))[:rank]
        off = struct.unpack('<15d', f.read(120))[:rank]
        dt, ct, cl, nl, dl, rsv, dlen_disk, nxt = struct.unpack('<IIIIIQQQ', f.read(44))
        name = f.read(nl).decode('utf8', 'replace'); d = f.read(dl)
        data_pos = f.tell()
        fpos = data_pos + dlen_disk
        f.seek(fpos)
        fs, = struct.unpack('<I', f.read(4))
        hcp = struct.unpack('<15I', f.read(60)); hcl = struct.unpack('<15I', f.read(60))
        mlen, = struct.unpack('<I', f.read(4))
        si = f.read(80 * 16)
        nfp, fbs, tdl = struct.unpack('<QQQ', f.read(24))
        sed, mfv, seud = struct.unpack('<QIQ', f.read(20))
        sw, ncp = struct.unpack('<QQ', f.read(16)) if sv >= 6 else (0, 0)
        f.seek(fpos + fs)
        labels = [rs(f).decode() for _ in range(rank)]
        for dim in range(rank):
            if hcp[dim]: f.read(8 * res[dim])
        for dim in range(rank):
            if hcl[dim]:
                for _ in range(res[dim]): rs(f)
        f.read(mlen)
        f.read(8 * nfp)
        tags = {}
        while True:
            k = rs(f)
            if not k: break
            tags[k.decode()] = rs(f)
        chunks = [struct.unpack('<QQ', f.read(16)) for _ in range(ncp)]
        end = f.tell()
        print(f'[{i}] @{pos} v{sv} rank {rank} res {res} dt 0x{dt:x} comp {ct}/{cl} name {name!r} descr {len(d)}B data@{data_pos} len_disk {dlen_disk} footer_size {fs} samples_written {sw} chunks {ncp} flush {nfp} tagdict {tdl} stack_end_disk {sed} min_fmt {mfv} stack_end_used {seud} next {nxt} meta_end {end}')
        print('     labels', labels, 'tags', {k: len(v) for k, v in tags.items()}, 'descr head', d[:80])
        if chunks: print('     chunk positions', chunks[:5])
        # peek at the data
        f.seek(data_pos)
        raw = f.read(min(dlen_disk, 64))
        print('     data head', raw[:48])
        if ct and dlen_disk:
            f.seek(data_pos)
            try:
                dec = zlib.decompressobj().decompress(f.read(min(dlen_disk, 1 << 20)), 256)
                print('     inflated head', dec[:96])
            except Exception as e:
                print('     inflate error', e)
        if dump_dir and 'MFX' in name.upper() or (dump_dir and b'MFXDTA' in raw):
            pass
        regions.append((pos, end))
        pos = nxt; i += 1
    regions.sort()
    # gaps not covered by stack header..metadata end
    last = 0
    for a, b in regions:
        if a > last + 16:
            f.seek(last); g = f.read(min(a - last, 64))
            print(f'  gap {last}..{a} ({a-last} B) head {g[:48]!r}')
        last = max(last, b)
    if last < size:
        f.seek(last); g = f.read(64)
        print(f'  tail {last}..{size} ({size-last} B) head {g[:48]!r}')

if __name__ == '__main__':
    for p in sys.argv[1:]:
        scan(p)
