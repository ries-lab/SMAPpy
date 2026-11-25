# 4Pi in situ on the Ries-lab Nup96 movie (docs/4pi.md §27)

The whole movie (zenodo 10823308, `~/.cache/uipsf/zenodo/4pi_smlm_nup96`, 70 files, 140 000 frames) and its
26 bead scans (`4pi_40nm_bead`); outputs in `~/.cache/uipsf/p2/`.  Run from the repository with `PYTHONPATH=src`
(a worktree's own code), in this order:

    python benchmarks/fourpi/insitu/bead_model.py                     # v2 bead model, ~10 min on a Mac's GPU
    python benchmarks/fourpi/insitu/fit_movie.py ~/.cache/uipsf/p2/bead_v2_4pi.h5 ~/.cache/uipsf/p2/fit_bead_v2.hdf5
    python benchmarks/fourpi/insitu/learn_nup96.py ~/.cache/uipsf/p2/fit_bead_v2.hdf5 \
        ~/.cache/uipsf/p2/bead_v2_4pi.h5 ~/.cache/uipsf/p2/insitu_v2b_4pi.h5 3000    # odd files, ~25 min
    python benchmarks/fourpi/insitu/fit_movie.py ~/.cache/uipsf/p2/insitu_v2b_4pi.h5 ~/.cache/uipsf/p2/fit_insitu_v2b.hdf5
    python benchmarks/fourpi/insitu/evaluate.py bead_v2=~/.cache/uipsf/p2/fit_bead_v2.hdf5 \
        insitu_v2b=~/.cache/uipsf/p2/fit_insitu_v2b.hdf5                          # even files
    python benchmarks/fourpi/insitu/heldout.py bead=bead insitu_v2b=~/.cache/uipsf/p2/insitu_v2b_4pi.h5.pkl

The cached bead model was saved before the bead export kept its recipe, which was then added to it by hand (the
window's defaults, as the script uses them); a fresh `bead_model.py` keeps it itself.  Each movie fit takes ~17 min on all cores.

## The 2017-11-28 tubulin (docs/4pi.md §29)

Gabin's 4Pi-SMS data in `~/.cache/uipsf/4pi_tubulin_2017-11-28` (P1's, §26); outputs in `smappy_insitu/` beside
it (`TUBULIN_OUT` moves P1's scripts' outputs there, never over `smappy_p1/`).  With `O` that folder:

    TUBULIN_OUT=$O python benchmarks/fourpi/tubulin_pupil.py 680 100 held        # P1's bead model, with its recipe
    TUBULIN_OUT=$O python benchmarks/fourpi/tubulin_fit.py $O/pupil_680_100_held_4pi.h5      # -> cell01_pupil_...hdf5
    python benchmarks/fourpi/insitu/tubulin_learn.py $O/cell01_bead.hdf5 $O/pupil_680_100_held_4pi.h5 $O/insitu_4pi.h5
    python benchmarks/fourpi/insitu/tubulin_learn.py $O/cell01_bead.hdf5 $O/pupil_680_100_held_4pi.h5 $O/tied_4pi.h5 3000 0.2
    TUBULIN_OUT=$O python benchmarks/fourpi/tubulin_fit.py $O/insitu_4pi.h5               # and tied_4pi.h5
    python benchmarks/fourpi/insitu/tubulin_evaluate.py bead=$O/cell01_bead.hdf5 insitu=$O/cell01_insitu.hdf5 ...
    python benchmarks/fourpi/tubulin_compare.py $O/cell01_bead.hdf5 $O/cell01_insitu.hdf5 $O/cell01_tied.hdf5
    python benchmarks/fourpi/insitu/tubulin_gauss.py $O/pupil_680_100_held_4pi.h5 $O/insitu_4pi.h5

The bead model's fit was renamed `cell01_bead.hdf5` after `tubulin_fit.py`.  Each learning is ~20 min on a Mac's
GPU and its held-out judging as long again; each movie fit 1 -- 3 min.
