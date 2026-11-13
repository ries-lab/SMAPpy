"""The sample's refractive index, one choice wherever a pupil model is learnt or drawn at a depth.

The index of the medium the emitters sit in (n_med) sets two things in a pupil model: the light beyond the
critical angle that reaches the objective only from near the coverslip (SAF), and the spherical aberration
that grows with the depth when it differs from the immersion's.  It was a number under *more*, on *auto*
water's 1.33, and a fixed-cell dSTORM sample in a blinking buffer (about 1.345) was drawn as in water
without anyone deciding so.  So it is a choice of what the sample is in, shown, with the number beside
it (November 2026):

* *water*, 1.33: beads in water or PBS, live cells in medium;
* *blinking buffer*, 1.345: a GLOX dSTORM buffer with about 10 % w/v glucose at 690 nm (1.340 at 5 %).
  PFA-fixed cells in PBS measure about 1.340 on average, nucleoli 1.35 (doi 10.1002/cyto.a.24229).  On
  AF647 molecules on glass in that buffer a model learnt at 1.345 fitted better than at 1.33 (236 nats on
  ~4500 molecules) and moved γ and the coma toward the same day's beads; learnt freely in situ the index
  ran to 1.41, so learning it is no substitute for a sensible value here;
* *index matched*: the immersion's -- a mounting medium matched to oil, or 4Pi's sample;
* *typed*: the number beside it, which on the others shows, in grey, the index they stand for; typing over
  it makes the choice *typed*.

*auto* (the empty value) keeps each place's old default: in a calibration water's, or the immersion's for
4Pi; drawn in the sample, the index the calibration was learnt with.  The number fields keep their old
names (`OpticsSettings.n_medium`, the sample parts' ``sample_index``), so that a saved setting, a chain or
a GUI file restored into a form field by field (`SettingsForm.restore`) still sets them; a number with the
choice on *auto* -- what those make, and a number typed over *auto*'s -- reads as *typed* (`settle`, inside
`resolve`; the settings keep the pair as given, so that a form can tell a choice made from a number left).
A form drops the number when the choice is changed, and makes the choice *typed* when a number is typed
over another's (`cleared`), so that going back to *auto* does not read the old number as typed.

Imports nothing heavy: the fitters' forms read it without torch."""
from __future__ import annotations

from typing import Optional, Union

from ..params import param

#: water's index, and a buffer's (module docstring)
WATER = 1.33
BUFFER = 1.345

#: the choice: auto first, with "" as its value, so a form says what auto stands for ("auto (water)"); the
#: number beside it shows the index each stands for, greyed but on *typed*
CHOICES = (("", "auto"), ("water", "water"), ("buffer", "blinking buffer"), ("matched", "index matched"),
           ("typed", "typed"))
PRESETS = {"water": WATER, "buffer": BUFFER}
#: the two fields in `OpticsSettings`
FIELDS = ("medium", "n_medium")

#: the tooltip's common part, the same in every place
_HELP = ("what the sample is in: water 1.33 (beads, live cells), blinking buffer 1.345 (fixed cells in a "
         "dSTORM buffer), index matched (the immersion's), or typed")


def medium_field(auto: str, **info):
    """The choice, ``auto`` saying what *auto* is in this place."""
    return param("", label="sample index", choices=CHOICES, help=f"{_HELP}; auto: {auto}", **info)


def index_field(**info):
    """The number beside the choice: in grey the index the choice stands for; typed over, the choice is
    *typed* (`cleared`)."""
    return param(None, label="n", min=1.0, max=2.0, beside=True, type_over=True,
                 help="the sample's refractive index the choice stands for; type over it for another", **info)


def settle(medium: str, number: Optional[float]) -> str:
    """The choice a pair stands for: a number with *auto* is a typed one (module docstring); an unknown
    choice is refused."""
    if medium not in dict(CHOICES):
        raise ValueError(f"the sample index must be one of {[k or 'auto' for k, _ in CHOICES]}, not {medium!r}")
    return "typed" if medium == "" and number is not None else medium


def cleared(changed: str, medium: str, number: Optional[float], choice: str, number_path: str) -> dict:
    """What a form sets after the field ``changed`` (`Plugin.react`'s updates): the number at ``number_path``
    dropped when the choice at ``choice`` leaves *typed*, and the choice made *typed* when a number is typed
    over another's."""
    if changed == choice and medium != "typed" and number is not None:
        return {number_path: None}
    if changed == number_path and number is not None and medium != "typed":
        return {choice: "typed"}
    return {}


def resolve(medium: str, number: Optional[float], auto: Optional[float],
            n_immersion: Optional[float] = None) -> Optional[float]:
    """The index ``medium`` and ``number`` stand for; ``auto`` on *auto* (None: the caller's own) and on
    *typed* without a number; *index matched* ``n_immersion``, or "matched" without one, for the caller
    to resolve (`depth.resolved_medium`, the model's immersion)."""
    medium = settle(medium, number)
    if medium in PRESETS:
        return PRESETS[medium]
    if medium == "matched":
        return float(n_immersion) if n_immersion is not None else "matched"
    if medium == "typed" and number is not None:
        return float(number)
    return auto


def for_drawing(medium: str, number: Optional[float]) -> Union[None, float, str]:
    """What a sample part hands `depth` as ``n_med``: None on *auto* (the calibration's own), a number, or
    "matched" (the model's immersion)."""
    return resolve(medium, number, None)


def differs(drawn: Optional[float], learnt: Optional[float]) -> bool:
    """Whether a PSF is drawn for another sample index than its calibration was learnt with."""
    return drawn is not None and learnt is not None and abs(float(drawn) - float(learnt)) > 5e-4


def mismatch_text(drawn: float, learnt: float) -> str:
    """The notice a fitter shows when it draws the PSF at another index than its calibration's."""
    return (f"the PSF is drawn for a sample index of {drawn:g}, the calibration learnt at {learnt:g}: expected "
            f"for beads in water and a sample in buffer; a calibration from molecules in this sample's "
            f"medium is drawn at its own index (sample index on auto)")
