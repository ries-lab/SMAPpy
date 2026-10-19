"""Fetch parts of the uiPSF paper's example data from Zenodo without the whole 9.3 GB zip.

The record (doi:10.5281/zenodo.8267520, latest version 10602740) is a single zip.
Zip members can be read with HTTP range requests, so one dataset folder can be
fetched on its own.  Most members use Deflate64, which Python's zipfile cannot
read; the `zipfile-deflate64` package adds it.

    pip install zipfile-deflate64
    python scripts/uipsf/fetch_zenodo.py --list
    python scripts/uipsf/fetch_zenodo.py 1ch_40nm_bead/ results/1ch_40nm_bead/

Files go to $UIPSF_DATA (default ~/.cache/uipsf/zenodo).  They must never be
committed.  Existing files of the right size are skipped, so an interrupted
run can be repeated.
"""
import argparse
import io
import os
import shutil
import time
import urllib.request
import zipfile

import zipfile_deflate64  # noqa: F401  (registers Deflate64 with zipfile)

URL = "https://zenodo.org/api/records/10602740/files/example%20data%20for%20uiPSF.zip/content"
TOP = "example data for uiPSF/"


class RemoteFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests, retrying cut-off transfers."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.size = int(head.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0:
            return b""
        headers = {"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        for attempt in range(6):
            try:
                data = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=120).read()
                break
            except OSError as e:
                print(f"  retry ({e})", flush=True)
                time.sleep(2 ** attempt)
        else:
            raise OSError(f"giving up on bytes {headers['Range']}")
        self.pos += len(data)
        return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="*", help="paths inside the archive, e.g. 1ch_40nm_bead/")
    ap.add_argument("--list", action="store_true", help="list the archive's files and sizes")
    ap.add_argument("--out", default=os.environ.get("UIPSF_DATA", os.path.expanduser("~/.cache/uipsf/zenodo")))
    args = ap.parse_args()

    archive = zipfile.ZipFile(RemoteFile(URL))
    for info in archive.infolist():
        name = info.filename.removeprefix(TOP)
        if info.is_dir() or not name:
            continue
        if args.list:
            print(f"{info.file_size / 1e6:10.1f} MB  {name}")
            continue
        if not any(name.startswith(p) for p in args.prefixes):
            continue
        dst = os.path.join(args.out, name)
        if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with archive.open(info) as src, open(dst + ".part", "wb") as f:
            shutil.copyfileobj(src, f, 16 << 20)
        os.replace(dst + ".part", dst)
        print("got", name, flush=True)


if __name__ == "__main__":
    main()
