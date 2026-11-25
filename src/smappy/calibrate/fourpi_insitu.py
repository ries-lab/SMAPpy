"""4Pi in situ: a movie's own molecules, chosen from a Spline 4Pi fit and cut in every channel, for uiPSF's
in-situ 4Pi learning (`uipsf.workflows.fourpi_insitu`; docs/4pi.md §27).  No torch and no Qt here: the PSF
calibration window and a script call it alike.

* **The molecules** come from a fit of the movie with the bead model (*Spline 4Pi*): its table says where each
  molecule is, how bright, at which z.  `choose` keeps the bright, alone and in-range ones, spread over z by
  equal quotas in bins and over the movie at random, so that no height and no part of the movie dominates.
* **The boxes** are cut as the fitter cuts its own (`multichannel.channel_candidates`, `cut_channel_rois`):
  the molecule's position in channel 0's own pixels mapped into each channel by the layout, each box
  centred on the nearest pixel, in photons.  Each box's centre is uiPSF's ``roi_xy`` and the remainder in
  channel 0 its start x, y, so the model's maps start from the layout's -- the bead model's, with which the
  movie was fitted.
* **The backgrounds** start from each box's edge, per channel (the fitter's table has one for all).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..channels import FourPiChannelsSettings
from ..plugins import param

MODEL_FILTER = "4Pi models (*.h5)"
LOCS_FILTER = "Localizations (*.hdf5 *.h5)"


@dataclass
class FourPiInSituSettings:
    """The PSF calibration window's *4Pi in situ* section: the bead model to start from, the molecules (a fit
    of the movie with it, given or made here), which of them are learnt from and how (docs/4pi.md §27)."""
    bead_model: str = param("", label="bead model", kind="open_file", file_filter=MODEL_FILTER,
                            help="the 4Pi pupil model learnt from beads, which in situ starts from")
    localizations: str = param("", label="localizations", kind="open_file", file_filter=LOCS_FILTER,
                               help="a Spline 4Pi fit of these movies with the bead model; empty: they are "
                                    "fitted here")
    fit_frames: int = param(20000, label="frames fitted", unit="frames", min=100,
                            help="without localizations: how many frames from the start are fitted")
    # where the channels are, as Spline 4Pi reads the movie: read only to fit it here
    channels: FourPiChannelsSettings = param(default_factory=FourPiChannelsSettings, label="channels",
                                             collapsed=True)
    molecules: int = param(3000, label="molecules", min=100,
                           help="how many molecules are learnt from, chosen over z and the movie")
    min_photons: float = param(0.0, label="min photons", min=0.0,
                               help="the dimmest molecule taken, all channels' photons; 0: the brighter half")
    roi_size: int = param(13, label="box", unit="px", min=9, help="each molecule's box per channel")
    z_margin_nm: float = param(100.0, label="z margin", unit="nm", min=0.0, advanced=True,
                               help="molecules this close to the model's z range's ends are left out")
    z_bins: int = param(8, label="z bins", min=1, advanced=True,
                        help="the molecules are spread over z in this many bins of equal quota")
    rounds: int = param(2, label="rounds", min=1, max=5, advanced=True,
                        help="localise, leave out the poor fits, learn: this many times")
    held_out: float = param(0.2, label="held out", min=0.0, max=0.5, advanced=True,
                            help="this share of the molecules is not learnt from, and judges the in-situ "
                                 "model against the bead model")
    seed: int = param(0, label="seed", advanced=True, help="which molecules are drawn, where there are more")
    cavity_sigma_rad: float = param(0.2, label="cavity tie", unit="rad", min=0.0, advanced=True,
                                    help="while learning, each molecule's phase is held this near the cavity's "
                                         "phase at its frame; 0: each its own")


def _column(locs, name):
    return np.asarray(locs[name], float)


def choose(locs, z_range_nm: float, settings: FourPiInSituSettings, frames=None) -> np.ndarray:
    """Indices of the molecules to learn from (module docstring): fitted to a finite position and phase,
    at least *min photons* (the median on 0), inside ±(``z_range_nm`` − *z margin*), with no other
    localization within a box in the same frame; then *molecules* of them spread over *z bins* equal bins.
    ``frames`` limits them to those frames (a set or a boolean function of the frame numbers)."""
    z = _column(locs, "z_env_nm" if "z_env_nm" in locs else "z_nm")
    x, y = _column(locs, "x_nm"), _column(locs, "y_nm")
    photons = _column(locs, "photons")
    frame = np.asarray(locs["frame"], np.int64)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(photons)
    if "phase_rad" in locs:
        ok &= np.isfinite(_column(locs, "phase_rad"))
    floor = settings.min_photons or float(np.median(photons[ok])) if ok.any() else 0.0
    ok &= photons >= floor
    limit = z_range_nm - settings.z_margin_nm
    ok &= np.abs(z) <= limit
    if frames is not None:
        ok &= frames(frame) if callable(frames) else np.isin(frame, list(frames))
    ok &= alone(x, y, frame, settings.roi_size * _pixel_nm(locs))
    index = np.flatnonzero(ok)
    rng = np.random.default_rng(settings.seed)
    if len(index) <= settings.molecules:
        return index
    edges = np.linspace(-limit, limit, settings.z_bins + 1)
    which = np.clip(np.digitize(z[index], edges) - 1, 0, settings.z_bins - 1)
    pools = [rng.permutation(index[which == b]) for b in range(settings.z_bins)]
    # equal quotas; what a thin bin cannot fill goes to the others
    sizes = np.array([len(p) for p in pools])
    quota, want = np.zeros(len(pools), int), settings.molecules
    while want > 0 and (quota < sizes).any():
        room = np.flatnonzero(quota < sizes)
        share = max(1, want // len(room))
        for b in room:
            take = min(share, sizes[b] - quota[b], want)
            quota[b] += take
            want -= take
    return np.sort(np.concatenate([p[:q] for p, q in zip(pools, quota)]))


def _pixel_nm(locs) -> float:
    px = (getattr(locs, "metadata", None) or {}).get("pixelsize_nm")
    if px is None:
        return 100.0
    return float(np.mean(px))


def alone(x, y, frame, distance) -> np.ndarray:
    """Which localizations have no other within ``distance`` (the same units) in their frame."""
    from scipy.spatial import cKDTree
    x, y, frame = np.asarray(x, float), np.asarray(y, float), np.asarray(frame)
    out = np.ones(len(x), bool)
    order = np.argsort(frame, kind="stable")
    bounds = np.flatnonzero(np.diff(frame[order])) + 1
    for group in np.split(order, bounds):
        if len(group) < 2:
            continue
        tree = cKDTree(np.c_[x[group], y[group]])
        n = np.array([len(v) for v in tree.query_ball_point(np.c_[x[group], y[group]], distance)])
        out[group[n > 1]] = False
    return out


@dataclass
class InSituMolecules:
    """The chosen molecules cut for learning: ``data`` (E, C, R, R) photons, ``roi_xy`` (E, C, 2) each box's
    centre in its channel's own pixels, ``start`` (xyz nm from channel 0's box centre, photons, backgrounds
    (E, C), phase), and where each came from: its ``frame`` and row ``index`` in the table."""
    data: np.ndarray
    roi_xy: np.ndarray
    start: dict
    frame: np.ndarray
    index: np.ndarray

    def __len__(self) -> int:
        return len(self.index)


def cut(source, layout, locs, index, camera, roi_size: int, pixel_nm, drift=None, progress=None) -> InSituMolecules:
    """The molecules ``index`` of the table ``locs`` cut from ``source`` (a `channels.ChannelSource`, or any
    source of whole frames, which is wrapped) in every channel of ``layout``, in photons by ``camera`` (module
    docstring).  ``pixel_nm`` (x, y); ``drift`` a `multichannel.ChannelDrift` the fit measured.  Molecules
    whose box leaves a channel are dropped."""
    from types import SimpleNamespace

    from ..camera import to_photons
    from ..channels import ChannelSource
    from ..multichannel import channel_candidates
    if not isinstance(source, ChannelSource):
        source = ChannelSource.wrap(source, layout.on_camera(camera.roi_offset, source.shape)
                                    if getattr(camera, "roi_offset", None) else layout)
    px = np.broadcast_to(np.asarray(pixel_nm, float), (2,))
    index = np.asarray(index)
    roi = np.asarray(getattr(camera, "roi_offset", (0, 0)) or (0, 0), float)
    chip = np.c_[_column(locs, "x_nm")[index] / px[0], _column(locs, "y_nm")[index] / px[1]] - roi
    own = source.layout.from_frame(0, chip)
    frame = np.asarray(locs["frame"], np.int64)[index]
    cands = SimpleNamespace(x=own[:, 0], y=own[:, 1], frame=frame)
    positions, residual, keep = channel_candidates(cands, source.layout, roi_size, drift)
    half = (roi_size - 1) // 2
    span = np.arange(-half, half + 1)
    C = source.layout.n_channels
    data = np.zeros((len(index), C, roi_size, roi_size), np.float32)
    order = np.argsort(frame, kind="stable")
    done = 0
    for f in np.unique(frame[keep]):
        rows = order[(frame[order] == f)]
        rows = rows[keep[rows]]
        image = to_photons(source.frame(int(f)), camera)                 # (C, h, w)
        for c in range(C):
            ys = positions[rows, c, 1][:, None] + span
            xs = positions[rows, c, 0][:, None] + span
            data[rows, c] = image[c][ys[:, :, None], xs[:, None, :]]
        done += len(rows)
        if progress is not None and done % 500 < len(rows):
            progress(f"cut {done} of {int(keep.sum())} molecules")
    edge = np.concatenate([data[..., 0, :], data[..., -1, :], data[..., 1:-1, 0], data[..., 1:-1, -1]], -1)
    background = np.median(edge, -1).clip(min=0.1)
    xyz = np.c_[residual[:, 0, 0] * px[0], residual[:, 0, 1] * px[1],
                _column(locs, "z_env_nm" if "z_env_nm" in locs else "z_nm")[index]]
    photons = _column(locs, "photons")[index]
    start = {"xyz": xyz[keep], "photons": photons[keep], "background": background[keep],
             "phase": np.zeros(int(keep.sum()))}
    return InSituMolecules(data[keep], positions[keep].astype(float), start, frame[keep], index[keep])


# ---------------------------------------------------------------- the run, as the window and a script make it

@dataclass
class FourPiInSituRun:
    """What a run of in situ left: the learnt fit (`uipsf.workflows.fourpi_insitu.FourPiInSitu`), the bead
    model it started from (model, shared values, calibration), the molecules, which of them were held out,
    the held-out comparison (objective of each model, nats, lower is better), and the calibration saved."""
    fit: object
    bead: tuple
    bead_calibration: object
    molecules: InSituMolecules
    held: np.ndarray
    heldout: dict
    layout: object
    calibration: object = None
    saved: Path | None = None
    messages: list = field(default_factory=list)
    #: each held-out molecule's z on each model, by the names of ``heldout``
    held_z: dict = field(default_factory=dict)

    @property
    def gain_nats(self) -> float | None:
        """How much better the in-situ model explains the held-out molecules than the bead model, nats per
        molecule; None without any held out."""
        if not self.heldout:
            return None
        return (self.heldout["bead"] - self.heldout["in situ"]) / max(1, len(self.held))

    @property
    def other_fringe(self) -> float | None:
        """The share of the held-out molecules the two models put on different fringes: their z apart by more
        than half the bead model's period, about the median difference; None without any held out."""
        if "bead" not in self.held_z or "in situ" not in self.held_z:
            return None
        d = np.asarray(self.held_z["in situ"]) - np.asarray(self.held_z["bead"])
        if not len(d):
            return None
        return float(np.mean(np.abs(d - np.median(d)) > self.bead_calibration.period_nm / 2))


#: more of the held-out molecules than this on another fringe, or a period moved by more than this share: the
#: in-situ model measures z differently from the beads', whatever its likelihood (docs/4pi.md §29)
OTHER_FRINGE_WARNING = 0.05
PERIOD_WARNING = 0.02


def warnings_of(result: FourPiInSituRun) -> list:
    """What the window warns of after a run: a held-out likelihood no better than the beads', and -- since a
    better likelihood is not enough (on the 2017-11-28 tubulin in situ explained the held-out molecules by 97
    nats each better and measured z worse, docs/4pi.md §29) -- z that moved to other fringes, or a period that
    moved, which the molecules alone cannot fix."""
    out = []
    gain = result.gain_nats
    if gain is not None and gain <= 0:
        out.append(f"the in-situ model explains the held-out molecules no better than the bead model "
                   f"({gain:+.1f} nats per molecule): keep the bead model")
    other = result.other_fringe
    if other is not None and other > OTHER_FRINGE_WARNING:
        out.append(f"the in-situ model puts {other:.0%} of the held-out molecules on another fringe than the "
                   f"bead model does: fit the movie with both and compare their uncertain fringes before using it")
    if result.calibration is not None:
        before, after = result.bead_calibration.period_nm, result.calibration.period_nm
        if abs(after / before - 1) > PERIOD_WARNING:
            out.append(f"the fringe period moved from {before:.1f} to {after:.1f} nm: single molecules do not "
                       f"measure it, the beads' scan does; the in-situ z may be scaled")
    return out


def bead_model_of(path, device=None):
    """(calibration, model, shared) of a bead model file, or a ValueError saying why it cannot start in situ."""
    from ..fourpi import load_fourpi_calibration
    from .fourpi_beads import model_from_calibration
    if not path:
        raise ValueError("give the bead model: in situ starts from a 4Pi pupil model learnt from beads")
    cal = load_fourpi_calibration(path)
    made = model_from_calibration(cal, device=device)
    if made is None:
        if "uiPSF v2" in str((cal.parameters or {}).get("method", "")):
            # a pupil model of this program's, saved before the model's recipe was kept with it (2026-11-24):
            # the same beads and settings learn it again, recipe and all (docs/4pi.md §29)
            raise ValueError(f"{Path(path).name} is a pupil model saved before its recipe was kept with it: "
                             "learn it again from the same beads with the same settings, and in situ can "
                             "start from the new file")
        raise ValueError(f"{Path(path).name} keeps no pupil model to start from: in situ needs a 4Pi pupil "
                         "model saved by this program's PSF calibration (a bead average or uiPSF v1's will not do)")
    return (cal,) + made


def z_range_of(cal) -> float:
    """The z the calibration covers either side of focus, nm."""
    ends = cal.z_index_to_nm(np.array([0.0, float(cal.shape[0])]))      # the first and the last knot
    return float(np.min(np.abs(ends)))


def fit_movie(paths, settings: FourPiInSituSettings, camera, out_path, progress=None):
    """The localizations in situ chooses from: the given fit, or *Spline 4Pi* on the first *frames fitted*
    frames of the movie with the bead model and the section's channels (and *4Pi z*'s phase), saved to
    ``out_path``."""
    from ..io.formats import load
    if settings.localizations:
        return load(settings.localizations)[0]
    from ..plugins import Context
    from ..plugins.fit import CameraSettings, OutputSettings, SourceSettings
    from ..plugins.fit_4pi import FourPiFit, FourPiFitSettings, FourPiModelSettings
    px = _pixel(camera)
    fit = FourPiFitSettings(source=SourceSettings(path=str(paths[0]), stop=int(settings.fit_frames)),
                            camera=CameraSettings(conversion=camera.conversion, offset=camera.offset,
                                                  pixelsize_um=float(np.mean(px)) / 1000.0),
                            model=FourPiModelSettings(calibration=str(settings.bead_model)),
                            output=OutputSettings(path=str(out_path)))
    fit.channels = settings.channels
    ctx = Context()
    if progress is not None:
        ctx.report = lambda text: progress({"message": f"fitting with the bead model: {text}", "fraction": 0.05})
    return FourPiFit().run(ctx, fit).locs


def run_insitu(paths, settings: FourPiInSituSettings, camera, save_to=None, progress=None, cancel=None,
               device=None, molecules: InSituMolecules | None = None, layout=None) -> FourPiInSituRun:
    """In situ from the movie ``paths`` (module docstring; docs/4pi.md §27): the molecules (``molecules``, a
    detection's, or chosen and cut here), *held out* of them kept aside, the rest learnt from, both models
    judged on those held out, and the model saved to ``save_to`` (None: not saved)."""
    import torch

    from uipsf.workflows import fourpi_insitu as fi
    tell = progress or (lambda entry: None)
    cal, bead_model, bead_shared = bead_model_of(settings.bead_model, device)
    if molecules is None:
        molecules, layout = detect(paths, settings, camera, progress)
    rng = np.random.default_rng(settings.seed)
    n = len(molecules)
    held = np.sort(rng.permutation(n)[:round(settings.held_out * n)])
    learn = np.setdiff1d(np.arange(n), held)
    batch = fi.molecule_batch(molecules.data, molecules.roi_xy, frame=molecules.frame)
    start = molecules.start

    def part(index):
        return fi.subset(batch, index), {k: v[index] for k, v in start.items()}

    sub, sub_start = part(learn)
    fit = fi.learn_insitu(bead_model, bead_shared, sub, sub_start, cal.period_nm,
                          settings=fi.InSituSettings(rounds=settings.rounds,
                                                     cavity_sigma_rad=settings.cavity_sigma_rad or None),
                          progress=tell, cancel=cancel)
    heldout, held_z = {}, {}
    if len(held) and fit.result is not None and fit.result.stop_reason != "cancelled":
        tell({"message": f"judging both models on {len(held)} held-out molecules", "fraction": 0.95})
        hb, hs = part(held)
        model = fi.molecule_model(bead_model, tuple(batch.data.shape[-2:]))
        for name, shared in (("bead", bead_shared), ("in situ", fit.shared)):
            found = fi.localise(model, shared, hb, hs, cal.period_nm)
            heldout[name] = float(found["objective"].sum())
            held_z[name] = found["xyz"][:, 2]
    out = FourPiInSituRun(fit, (bead_model, {k: torch.as_tensor(v) for k, v in bead_shared.items()}), cal,
                          molecules, held, heldout, layout, held_z=held_z)
    if save_to is not None and fit.result is not None and fit.result.stop_reason != "cancelled":
        save(out, save_to)
    return out


def detect(paths, settings: FourPiInSituSettings, camera, progress=None):
    """(the chosen molecules cut, the layout) from the movie ``paths``: the fit, the choice, the cut."""
    import tempfile

    from ..channels import ChannelLayout
    from ..io.tiff import open_stack
    tell = progress or (lambda entry: None)
    cal, _, _ = bead_model_of(settings.bead_model)
    if camera is None:
        raise ValueError("4Pi in situ needs the camera's conversion, offset and pixel size: give them under "
                         "microscope")
    out = Path(tempfile.mkdtemp()) / "insitu_fit_locs.hdf5"
    locs = fit_movie(paths, settings, camera, out, progress)
    record = (getattr(locs, "metadata", None) or {}).get("channel_layout")
    if record is None:
        raise ValueError("the localizations keep no channel layout: give a Spline 4Pi fit of these movies")
    layout = ChannelLayout.from_dict(record)
    tell({"message": "choosing the molecules", "fraction": 0.1})
    index = choose(locs, z_range_of(cal), settings)
    if not len(index):
        raise ValueError("no molecule passes: lower *min photons* or check that the fit is of these movies")
    px = _pixel(camera)
    mol = cut(open_stack(paths[0]), layout, locs, index, camera, settings.roi_size, px,
              progress=lambda text: tell({"message": text, "fraction": 0.12}))
    return mol, layout


def _pixel(camera) -> tuple:
    px = camera.pixelsize_nm_xy
    if px is None:
        raise ValueError("4Pi in situ needs the camera's pixel size: give it under microscope")
    return px


def save(result: FourPiInSituRun, path) -> object:
    """The in-situ model as a 4Pi calibration for *Spline 4Pi*, beside its layout and its recipe and values --
    so in situ can start from it again -- at ``path``; the calibration."""
    from uipsf.workflows import fourpi_insitu as fi

    from ..fourpi import save_fourpi_calibration
    from .fourpi_beads import recipe_record, shared_record
    bead_cal = result.bead_calibration
    roi = bead_cal.shape[1] + 1                     # the bead model's box: its knots per side
    dz = float(bead_cal.dz)
    z = z_range_of(bead_cal)
    params = {"bead_model": str(bead_cal.source), "heldout_nats": result.heldout,
              "shared": shared_record(result.fit.shared)}
    recipe = getattr(result.bead[0], "recipe", None)
    if recipe is not None:
        params["model"] = recipe_record(recipe)
    cal = fi.calibration(result.fit, z_range_nm=z, dz_nm=dz, roi_px=roi, parameters=params)
    if result.layout is not None:
        cal.layout = result.layout.to_dict()
    save_fourpi_calibration(path, cal, overwrite=True)
    result.calibration, result.saved = cal, Path(path)
    return cal


# ---------------------------------------------------------------- the window's pages

def draw_molecules(figure, result: FourPiInSituRun) -> str:
    """The molecules learnt from: their z and photons as the fit with the bead model gave them, and as the
    in-situ model localises them; those left out as poor fits apart."""
    m = result.molecules
    learnt = np.setdiff1d(np.arange(len(m)), result.held)
    kept = learnt[result.fit.kept]
    out = np.setdiff1d(learnt, kept)
    z_start = m.start["xyz"][:, 2]
    z_found = np.full(len(m), np.nan)
    z_found[learnt] = result.fit.found["xyz"][:, 2]
    a, b = figure.subplots(1, 2)
    bins = np.linspace(np.nanmin(z_start), np.nanmax(z_start), 25)
    a.hist(z_start[kept], bins, histtype="step", label="bead model's fit")
    a.hist(z_found[kept], bins, histtype="step", label="in situ")
    if len(out):
        a.hist(z_found[out], bins, histtype="step", label="left out")
    a.set_xlabel("z (nm)")
    a.set_ylabel("molecules")
    a.legend(fontsize="small")
    b.scatter(z_start[kept], z_found[kept] - z_start[kept], s=3, alpha=0.4)
    b.axhline(0, color="0.5", lw=0.8)
    b.set_xlabel("z, bead model's fit (nm)")
    b.set_ylabel("in situ − bead model (nm)")
    return f"{len(kept)} molecules used, {len(out)} left out as poor fits, {len(result.held)} held out"


def parameter_rows(result: FourPiInSituRun) -> tuple:
    """(columns, rows): the shared numbers of the bead model and of the in-situ model side by side, and the
    held-out molecules' objective under each."""
    bead, insitu = result.bead[1], result.fit.shared

    def text(v):
        return ", ".join(f"{x:.3g}" for x in np.atleast_1d(np.asarray(v, float)))

    rows = []
    for name, label in (("fourpi.coherence", "coherence"), ("fourpi.balance", "upper arm's share"),
                        ("fourpi.share", "channels' shares (1 - 3)"), ("fourpi.delta", "phase offsets (rad)"),
                        ("fourpi.offset", "misalignment x, y, z (nm)"), ("psf.sigma", "blur σ (nm)")):
        if name in bead and name in insitu:
            rows.append((label, text(bead[name]), text(insitu[name].cpu())))
    if result.calibration is not None:
        rows.append(("fringe period (nm)", f"{result.bead_calibration.period_nm:.1f}",
                     f"{result.calibration.period_nm:.1f}"))
    if result.heldout:
        rows.append((f"held out: objective of {len(result.held)} molecules (nats)",
                     f"{result.heldout['bead']:.0f}", f"{result.heldout['in situ']:.0f}"))
    return ("", "bead model", "in situ"), rows
