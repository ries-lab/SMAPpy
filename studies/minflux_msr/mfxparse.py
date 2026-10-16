"""Parse the MFXDTA archive found inside an msr stack (research only, hypothesis-driven)."""
import struct, sys, json

def parse(b, start):
    o = start; entries = []
    while o < len(b):
        if o + 8 > len(b): break
        n, = struct.unpack_from('<Q', b, o)
        if n == 0 or n > 4096:
            entries.append(('STOP', o, n)); break
        name = b[o + 8:o + 8 + n].decode('utf8', 'replace'); o += 8 + n
        kind = b[o]; o += 1
        if kind == 1:
            entries.append((name, 'dir', None, None))
        else:
            size, = struct.unpack_from('<Q', b, o); o += 8
            entries.append((name, 'file', size, o)); o += size
    return entries, o

if __name__ == '__main__':
    b = open(sys.argv[1], 'rb').read()
    start = b.index(b'sync') - 8
    print('header', b[:start].hex())
    ents, end = parse(b, start)
    print('end', end, 'len', len(b))
    for e in ents:
        if e[1] == 'file':
            name, _, size, off = e
            head = b[off:off + min(size, 16)]
            print(f'{name:40s} file {size:9d} @{off} {head[:16]!r}')
            if name.endswith(('.zarray', '.zgroup')) or (name.endswith('.zattrs') and size < 3000):
                print(b[off:off + size].decode())
        else:
            print(e)
