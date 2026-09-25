"""A spoken track: each subtitle read aloud by Piper, one small clip per step.

Piper (``pip install piper-tts``) is a neural text-to-speech engine that runs
offline on the CPU, so the clips are made where the screenshots are -- on a
laptop, or in CI -- with no API key and nothing sent anywhere.  The voice is
a model file (``.onnx`` with its ``.onnx.json`` beside it); the one used so
far is ``en-us-lessac-medium`` from Piper's v0.0.2 release on GitHub::

    https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz

Clips are MP3 at 32 kbit/s mono, which is plenty for one voice at Piper's
22 kHz: about 4 kB a second of speech, so a five-minute tutorial is about a
megabyte -- less than its screenshots.  MP3 rather than Opus or AAC because it
is the one every browser plays *and* every host serves: Safari was late to
Opus, and the artifact host refuses ``.m4a`` outright.  The page fetches one
clip per step as it is reached.

A clip's length decides its step's (`player.timing`), so the voice and the
subtitle cannot drift apart.  What is *said* is the subtitle put through
`spoken`: the screen says "Ctrl+O" and "24 416", the voice says "control O"
and "twenty-four thousand ...".
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List, Optional

# The voice's own pace.  1.1 (slower, for a lab listening in its second
# language) was the first try, and the review found the whole thing slow.
LENGTH_SCALE = 1.0
BITRATE = "32k"

# How the screen's words are read.  Whole words, case as written; add to it
# when the voice stumbles on a name.
SAY_AS = {
    "SMAPpy": "smap pie",
    "SMAP": "smap",
    "ThunderSTORM": "thunder storm",
    "MINFLUX": "min flux",
    "ROI": "R O I",
    "ROIs": "R O Is",
    "LUT": "lookup table",
    "PSF": "P S F",
    "SMLM": "S M L M",
    "nm": "nanometres",
    "z": "zed",
}


def spoken(text: str) -> str:
    """The subtitle as it should be read aloud."""
    # shortcuts: Ctrl+Shift+P -> "control shift P"
    text = re.sub(r"\bCtrl\+", "control ", text)
    text = re.sub(r"\bShift\+", "shift ", text)
    text = re.sub(r"\bCmd\+", "command ", text)
    text = re.sub(r"\b(control|shift|command) 0\b", r"\1 zero", text)
    # "Add file..." is a menu item, not a trailing-off
    text = text.replace("...", "")
    # 24 416 (thin or plain space as a thousands separator) -> 24416
    text = re.sub(r"(?<=\d)[   ](?=\d{3}\b)", "", text)
    for word, said in SAY_AS.items():
        text = re.sub(rf"(?<![\w-]){re.escape(word)}(?![\w-])", said, text)
    return re.sub(r"\s+", " ", text).strip()


def model_path(model: Optional[str] = None) -> Path:
    """The voice to use: given, or ``$SMAPPY_PIPER_VOICE``."""
    found = model or os.environ.get("SMAPPY_PIPER_VOICE")
    if not found:
        raise ValueError("no voice: pass --voice path/to/voice.onnx or set "
                         "SMAPPY_PIPER_VOICE (see smappy.tutorial.voice)")
    path = Path(found)
    if not path.is_file():
        raise FileNotFoundError(f"no voice model at {path}")
    return path


def narrate(out: Path, steps: List[dict], model: Optional[str] = None) -> List[dict]:
    """A clip per step in ``out``, named after it; each step gets ``audio``
    (the file) and ``audio_seconds`` (how long it speaks)."""
    from piper import PiperVoice, SynthesisConfig
    voice = PiperVoice.load(str(model_path(model)))
    config = SynthesisConfig(length_scale=LENGTH_SCALE)
    out = Path(out)
    with tempfile.TemporaryDirectory() as scratch:
        for i, step in enumerate(steps, 1):
            wav = Path(scratch) / f"{i:03d}.wav"
            with wave.open(str(wav), "wb") as w:
                voice.synthesize_wav(spoken(step["say"]), w, syn_config=config)
            with wave.open(str(wav), "rb") as r:
                seconds = r.getnframes() / float(r.getframerate())
            name = f"{i:03d}.mp3"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                            "-ac", "1", "-c:a", "libmp3lame", "-b:a", BITRATE,
                            str(out / name)], check=True)
            step["audio"] = name
            step["audio_seconds"] = round(seconds, 2)
    return steps
