"""Survey every MFXDTA stack in the given .msr files (research only)."""
import sys, json, struct, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mfxdump import stacks, blob
from mfxparse import parse

def keys(d, depth=0, maxd=2):
    if isinstance(d, dict) and depth < maxd:
        return {k: keys(v, depth + 1, maxd) for k, v in d.items()}
    if isinstance(d, list):
        return f'list[{len(d)}]' + (('<' + str(keys(d[0], depth + 1, maxd)) + '>') if d and depth < maxd else '')
    return type(d).__name__ if not isinstance(d, (int, float, str, bool)) else repr(d)[:40]

for path in sys.argv[1:]:
    print('=====', os.path.basename(path), os.path.getsize(path))
    kinds = {}
    for i, s in stacks(path):
        t = s['tags'].get('minflux', '')
        typ = json.loads(t).get('type') if t.startswith('{') else None
        kinds.setdefault((s['dt'], s['res'] if s['dt'] != 1 else 'blob', typ, s['name'].split('/')[-1] if s['name'].startswith('MF(') else s['name'][:12]), []).append(i)
        if s['dt'] == 1 and s['res'][1:] == (1, 1, 1):
            b = blob(path, s)
            if not b.startswith(b'MFXDTA'):
                print('  non-MFXDTA uint8 stack', i, b[:16]); continue
            start = b.index(b'sync') - 8 if b'sync' in b[:400] else 147
            print(f'  [{i}] tag={t[:160]}')
            print('     hdr', b[:24].hex(), 'first entry at', start, 'u32@7', struct.unpack_from('<I', b, 7)[0], 'u32@11', struct.unpack_from('<I', b, 11)[0])
            ents, end = parse(b, start)
            print('     end', end, 'len', len(b), 'trailer', b[end:].hex())
            for e in ents:
                if e[1] == 'file' and e[2]:
                    name, _, size, off = e
                    txt = b[off:off + size]
                    if name.endswith('.zarray'):
                        z = json.loads(txt)
                        dt = z['dtype']
                        dts = [f[0] + ('' if isinstance(f[1], str) else '{' + ','.join(g[0] for g in f[1]) + '}') + (str(f[2]) if len(f) > 2 else '') for f in dt] if isinstance(dt, list) else dt
                        print(f'     {name}: shape {z["shape"]} chunks {z["chunks"]} comp {z["compressor"]} filters {z["filters"]} dtype {dts}')
                    elif name.endswith('.zattrs'):
                        print(f'     {name} ({size} B): {keys(json.loads(txt), 0, 3)}'[:1500])
                    elif not name.endswith('.zgroup'):
                        print(f'     {name}: {size} B head {txt[:16].hex()}')
                elif e[1] == 'dir':
                    print('     dir', e[0])
    print('  stack kinds:', {k: v[:4] for k, v in kinds.items()})
