# decode_smlm: where this code comes from

`decode_smlm` is a port of [DECODE](https://github.com/TuragaLab/DECODE)
(Speiser, Müller et al., *Nat. Methods* 2021,
[doi:10.1038/s41592-021-01236-x](https://doi.org/10.1038/s41592-021-01236-x)) to current
PyTorch, to train and run DECODE's networks from SMAPpy (`docs/decode.md`,
`docs/decode/poc-plan.md`).

The modules here began as verbatim copies of DECODE **v0.10.2** (tag `v0.10.2`, commit
`cc03ce4cf1aa95e0b2fd8498763ced100afdba7e`, 2023-03-15) by Lucas-Raphael Müller, Artur
Speiser and the DECODE contributors, licensed under the **GNU General Public License,
version 3** (`LICENSE.txt` beside this file, DECODE's own).  The first commit that adds
them changes nothing; every change since is a visible diff in this repository's history.
The tests in `tests/decode_tests/` came the same way from `decode/test/`.

The rest of SMAPpy is BSD 3-Clause; this package and its tests are DECODE's licence until
its authors' relicensing says otherwise.

## `decode_smlm.plex`: DECODE-Plex

`plex/` is a port of [DECODE-Plex](https://github.com/ries-lab/DECODE-Plex) (Sha et al. 2026), the
multi-channel DECODE (biplane, two colours), to train and run its networks from SMAPpy as the
baseline for two channels (`docs/decode/plex-plan.md`).  Its modules began as verbatim copies of
DECODE-Plex at commit `6138472576d239eb6e9e7a5477e3b6d70e9f18d2` (2026-10-20): the package
`decode/` as `plex/`, and its Hydra configuration `config/` as `plex/config/`.  DECODE-Plex is
Hao Sha's work in Jonas Ries's group, built on DECODE 1.0.0dev0 by Lucas-Raphael Müller, Artur
Speiser and the DECODE contributors, and licensed under the same **GNU General Public License,
version 3** (`LICENSE.txt`); keeping it GPL-3 here was decided by Jonas Ries (2026-11-19).  The
first commit that adds it changes nothing; every change since is a visible diff in this
repository's history.
