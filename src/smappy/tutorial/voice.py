"""A spoken track: each subtitle read aloud by Kokoro, one small clip per step.

Kokoro (``pip install kokoro``, hexgrad/Kokoro-82M, Apache 2.0) is a neural
text-to-speech model of 82 M parameters that runs offline on the CPU, so the
clips are made where the screenshots are -- on a laptop, or in CI -- with no
API key and nothing sent anywhere.  Its weights (330 MB) and voices come from
the Hugging Face hub on first use and are cached there.  It replaced Piper
(``en-us-lessac-medium``) in 2026-10, whose delivery was flat enough that the
tutorials sounded like a machine whatever the wording; Kokoro reads at
about real time on a laptop's CPU, Piper ten times faster.

A voice is a name, ``bf_emma`` by default (British, as the subtitles'
"colours" and "zed" are; chosen by ear from all 28 English voices, over
``af_heart``, the best graded in Kokoro's VOICES.md); the first letter is the accent (``a`` American, ``b`` British),
the second the speaker (``f``, ``m``).

Clips are MP3 at 32 kbit/s mono, which is plenty for one voice at Kokoro's
24 kHz: about 4 kB a second of speech, so a five-minute tutorial is about a
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

DEFAULT_VOICE = "bf_emma"
REPO = "hexgrad/Kokoro-82M"
RATE = 24000                 # Kokoro's sample rate

# The voice's own pace.  Piper at 1/1.1 (slower, for a lab listening in its
# second language) was the first try, and the review found the whole thing slow.
SPEED = 1.0
BITRATE = "32k"

# How the screen's words are read.  Whole words, case as written; add to it
# when the voice stumbles on a name.
SAY_AS = {
    "SMAPpy": "Smappy",          # like "snappy" (espeak: s m 'a p i)
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
    # Kokoro's phonemiser takes a letter before a full stop for an initial
    # ("John O. Smith") and drops the stop, so "control O. Smappy reads"
    # runs on as one sentence; "O ." keeps it (a 0.26 s pause, 0.4 s after
    # a word).  The same for "y.", "R O I." and "2D."
    text = re.sub(r"\b(\d?[A-Za-z])\.(?=\s|$)", r"\1 .", text)
    return re.sub(r"\s+", " ", text).strip()


def voice_name(voice: Optional[str] = None) -> str:
    """The voice to use: given, or ``$SMAPPY_VOICE``, or `DEFAULT_VOICE`."""
    name = voice or os.environ.get("SMAPPY_VOICE") or DEFAULT_VOICE
    if not re.fullmatch(r"[ab][fm]_[a-z]+", name):
        raise ValueError(f"{name!r} is not an English Kokoro voice (af_heart, "
                         "bm_george, ...: see hexgrad/Kokoro-82M's VOICES.md)")
    return name


def narrate(out: Path, steps: List[dict], voice: Optional[str] = None) -> List[dict]:
    """A clip per step in ``out``, named after it; each step gets ``audio``
    (the file) and ``audio_seconds`` (how long it speaks)."""
    import numpy as np
    from kokoro import KPipeline
    name = voice_name(voice)
    pipeline = KPipeline(lang_code=name[0], repo_id=REPO)
    out = Path(out)
    with tempfile.TemporaryDirectory() as scratch:
        for i, step in enumerate(steps, 1):
            # Kokoro hands a long text back in pieces, a sentence or so each
            audio = np.concatenate([r.audio.numpy() for r in
                                    pipeline(spoken(step["say"]), voice=name, speed=SPEED)])
            pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
            wav = Path(scratch) / f"{i:03d}.wav"
            with wave.open(str(wav), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(pcm.tobytes())
            clip = f"{i:03d}.mp3"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                            "-ac", "1", "-c:a", "libmp3lame", "-b:a", BITRATE,
                            str(out / clip)], check=True)
            step["audio"] = clip
            step["audio_seconds"] = round(len(pcm) / RATE, 2)
    return steps
