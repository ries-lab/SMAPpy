# Abberior .msr: research scripts

Throwaway scripts written while working out how Imspector stores MINFLUX data
inside an `.msr` (OBF) file, for `docs/minflux-reader-plan.md`.  They were run
on the m2205 `.msr` files of `jonatanalvelid/etMINFLUX-analysis-public`
(`exampledata/`; that repository has no licence, so the files are not
committed -- clone it to run these).  Research code, not part of the package:
the reader is written fresh in `src/smappy/io/`.

* `obfscan.py FILE.msr` -- walks the OBF stacks and the gaps between them.
* `mfxdump.py FILE.msr` -- lists the stacks with their tags (`minflux` JSON).
* `mfxparse.py` -- parses the `MFXDTA` archive inside a MINFLUX stack.
* `mfxsurvey.py FILE.msr` -- every MINFLUX dataset of a file, summarised.
* `bloscdec.py` -- a pure-Python blosc1 + LZ4 decoder and the full decode of
  one dataset into its structured array (about 1 s per 2.4 MB compressed).
