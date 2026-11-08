"""Camera noise calibration: an sCMOS's offset and read noise per pixel, from a dark movie.

The work is `smappy.scmos` (`dark_map`, `NoiseMap`); this reads the movie,
says what was measured, saves the map and enters it in the camera database
as the noise map of the movie's readout mode (`camera_db.with_noise_map`),
where a fit with *per-pixel noise* on finds it (`FitSettings.per_pixel_noise`).
Only dark frames: why no gain and no second exposure is in `smappy.scmos`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import Context, Plot, Plugin, Result, param, register

TIFF_FILTER = "Image stacks (*.tif *.tiff *.ome.tif);;All files (*)"

#: a dark movie whose pixels sit this many electrons above the camera's offset
#: saw light: the shutter was open, or the room was not dark
LIGHT_E = 3.0


def _camera_choices():
    from .fit import _camera_choices as choices
    return choices()


@dataclass
class CameraNoiseSettings:
    path: str = param("", label="dark frames", kind="open_file", file_filter=TIFF_FILTER,
                      help="a movie with the shutter closed, in the readout mode and "
                           "binning of the data: 1000 frames or more")
    start: int = param(0, label="first frame", min=0, advanced=True,
                       help="frames before this one are skipped")
    stop: Optional[int] = param(None, label="last frame", min=1, advanced=True,
                                help="auto: to the end")
    camera: str = param("", label="camera", choices=lambda: _camera_choices(),
                        help="auto: identified from the file's metadata")
    enter: bool = param(True, label="enter in the camera database",
                        help="the map becomes this readout mode's noise map, which a fit "
                             "with per-pixel noise reads")
    output: str = param("", label="save as", kind="save_file", advanced=True,
                        file_filter="Noise maps (*.h5)",
                        help="auto: the configuration folder's noise_maps, named by the "
                             "camera, its readout mode and the day")


@register("Localize/Camera noise calibration")
class CameraNoiseCalibration(Plugin):
    """An sCMOS's offset and read noise per pixel, from dark frames, for the fit."""
    description = ("measure each pixel's offset and read noise of an sCMOS from a dark "
                   "movie, and keep it as the camera's noise map for fits with per-pixel "
                   "noise")
    changes_locs = False
    Settings = CameraNoiseSettings
    version = "1"

    def run(self, ctx: Context, settings: CameraNoiseSettings) -> Result:
        from .. import scmos
        from ..io.tiff import open_stack, resolve_camera
        if not settings.path:
            raise ValueError("choose a movie of dark frames")
        source = open_stack(settings.path)
        resolved = resolve_camera(source, camera=settings.camera)
        camera = resolved.camera_metadata()
        if camera.is_em:
            raise ValueError("these frames were taken with EM gain: a noise map is for an sCMOS")
        stop = settings.stop or source.n_frames
        info = {"camera": resolved.camera_name or camera.camera_name or "",
                "readout_mode": resolved.state_name, "exposure_ms": camera.exposure_ms,
                "source": str(settings.path), "date": _today(),
                "conversion": camera.conversion}
        blocks = (block for _, block in source.frames(chunk=200, start=settings.start,
                                                      stop=stop))
        noise = scmos.dark_map(blocks, region=camera.roi, info=info,
                               report=ctx.report)
        path = Path(settings.output) if settings.output else scmos.default_path(
            info["camera"], info["readout_mode"])
        noise.save(path)

        found = scmos.summary(noise, camera.conversion)
        unit = "e-" if camera.conversion else "ADU"
        lines = [f"{noise.n_frames} dark frames, {noise.shape[1]} x {noise.shape[0]} px at "
                 f"chip x {noise.region[0]}, y {noise.region[1]}: saved as {path}",
                 f"read noise: median {found['read_noise_median']:.2f} {unit}, rms "
                 f"{found['read_noise_mean_rms']:.2f}, 99th percentile "
                 f"{found['read_noise_p99']:.2f}, worst {found['read_noise_max']:.1f}; "
                 f"{100 * found['hot_fraction']:.2f} % of the pixels above "
                 f"{scmos.HOT_FACTOR:g}x the median",
                 f"offset: median {found['offset_median_adu']:.1f} ADU, 98 % within "
                 f"{found['offset_spread_adu']:.1f} ADU; each variance known to "
                 f"{100 * found['variance_precision']:.1f} %"]
        if noise.n_frames < scmos.GOOD_FRAMES:
            ctx.warn(f"{noise.n_frames} frames: each pixel's variance is known to "
                     f"{100 * found['variance_precision']:.0f} %; "
                     f"{scmos.GOOD_FRAMES} or more make it 4.5 %")
        if camera.conversion and camera.offset is not None:
            light = (found["offset_median_adu"] - float(camera.offset)) * float(camera.conversion)
            if light > LIGHT_E:
                ctx.warn(f"the frames sit {light:.1f} e- above the camera's offset of "
                         f"{camera.offset:g} ADU: were they dark?")

        if settings.enter:
            lines.append(self._enter(ctx, resolved, path))

        def maps(figure) -> None:
            draw_maps(figure, noise, camera.conversion)

        def histogram(ax) -> None:
            draw_histogram(ax, noise, camera.conversion)

        return Result(text="\n".join(lines), settings=settings,
                      data={"path": str(path), **found},
                      plot=Plot(maps, panels=2, size=(11, 4.5)),
                      plots={"read noise": Plot(histogram)})

    @staticmethod
    def _enter(ctx: Context, resolved, path: Path) -> str:
        """Enter the map in the camera database as the readout mode's; what was done, in words."""
        from .. import camera_db
        if resolved.camera is None:
            ctx.warn("the file's camera is not in the camera database, so the map was not "
                     "entered: add the camera, then give it this file as its noise map")
            return "not entered in the camera database"
        try:
            entry = camera_db.with_noise_map(resolved.camera, resolved.metadata, str(path))
        except ValueError as error:
            ctx.warn(str(error))
            return "not entered in the camera database"
        camera_db.save_camera(entry)
        mode = f", readout mode {resolved.state_name}" if resolved.state_name else ""
        return f"entered as the noise map of {entry.name}{mode}"


def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()


def draw_maps(figure, noise, conversion: Optional[float] = None) -> None:
    """The offset and the read noise over the chip, the read noise's scale
    clipped at its 99.5th percentile so that a few hot pixels do not hide
    the pattern of the rest."""
    left, right = figure.subplots(1, 2)
    x, y, w, h = noise.region
    extent = (x - 0.5, x + w - 0.5, y + h - 0.5, y - 0.5)
    shown = left.imshow(noise.offset, cmap="gray", extent=extent,
                        vmin=np.percentile(noise.offset, 0.5),
                        vmax=np.percentile(noise.offset, 99.5))
    figure.colorbar(shown, ax=left, fraction=0.046)
    left.set(title="offset (ADU)", xlabel="chip x (px)", ylabel="chip y (px)")
    scale = float(conversion) if conversion else 1.0
    read = noise.read_noise_adu() * scale
    shown = right.imshow(read, cmap="magma", extent=extent, vmin=0,
                         vmax=np.percentile(read, 99.5))
    figure.colorbar(shown, ax=right, fraction=0.046)
    right.set(title=f"read noise ({'e-' if conversion else 'ADU'} rms)", xlabel="chip x (px)")


def draw_histogram(ax, noise, conversion: Optional[float] = None, truth=None) -> None:
    """How the pixels' read noise is spread: a log scale on both axes shows the
    tail of hot pixels, which is what a map is for.  ``truth``, the noise the
    pixels were simulated with, in the same unit, drawn beside it."""
    scale = float(conversion) if conversion else 1.0
    read = (noise.read_noise_adu() * scale).ravel()
    read = read[read > 0]
    bins = np.geomspace(read.min(), read.max(), 60)
    ax.hist(read, bins=bins, histtype="stepfilled", alpha=0.6, label="measured")
    if truth is not None:
        ax.hist(np.ravel(truth), bins=bins, histtype="step", color="k", label="true")
        ax.legend(fontsize=8)
    median = float(np.median(read))
    ax.axvline(median, color="#c06000", lw=1)
    ax.set(xscale="log", yscale="log", xlabel=f"read noise ({'e-' if conversion else 'ADU'})",
           ylabel="pixels", title=f"median {median:.2f}")
