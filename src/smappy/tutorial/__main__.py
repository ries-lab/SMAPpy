"""python -m smappy.tutorial TOPIC [-o DIR] [--voice VOICE.onnx] [--video]"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Optional


def build(topic: str, out: Path, video: bool = False,
          voice: Optional[str] = None) -> Path:
    """Run a storyboard and write its page to ``out/topic``: with a spoken
    clip per step if given a Piper ``voice``, and a video if asked."""
    module = importlib.import_module(f"smappy.tutorial.topics.{topic}")
    target = Path(out) / topic
    from .director import Director
    from . import player
    director = Director(target)
    try:
        module.make(director)
    finally:
        director.close()
    steps = director.manifest()
    if voice:
        from .voice import narrate
        narrate(target, steps, voice)
    page = player.write(target, steps, module.TITLE,
                        getattr(module, "DESCRIPTION", ""), director.size)
    if video:
        player.record(target, director.size)
    return page


def main(argv=None) -> int:
    from .topics import TOPICS
    parser = argparse.ArgumentParser(prog="python -m smappy.tutorial",
                                     description="Build a tutorial from its storyboard.")
    parser.add_argument("topic", choices=TOPICS)
    parser.add_argument("-o", "--out", default="build/tutorials", type=Path)
    parser.add_argument("--voice", metavar="VOICE.onnx",
                        help="read every subtitle aloud with this Piper voice "
                             "(needs piper-tts and ffmpeg; see smappy.tutorial.voice)")
    parser.add_argument("--video", action="store_true",
                        help="also record tutorial.mp4 (needs playwright and ffmpeg)")
    args = parser.parse_args(argv)
    page = build(args.topic, args.out, args.video, args.voice)
    print(page)
    return 0


if __name__ == "__main__":
    sys.exit(main())
