import json
import subprocess
import sys

import pytest

from smappy.tutorial import player


def _step(say, **extra):
    step = {"say": say, "image": "001.webp", "spot": [], "point": None, "click": False,
            "zoom": None, "card": None, "chapter": None, "duration": 0.0}
    step.update(extra)
    return step


def test_a_subtitle_stays_up_long_enough_to_read_and_a_card_longer():
    short = _step("Click Run.")
    long = _step(" ".join(["word"] * 28))
    pointed = _step(" ".join(["word"] * 28), point=(10, 10))
    card = _step("Filters.", image=None,
                 card={"title": "Filters", "body": " ".join(["word"] * 70), "figure": ""})
    player.timing([short, long, pointed, card])
    assert short["duration"] == player.MIN_SECONDS
    assert long["duration"] == pytest.approx(1.2 + 28 / player.WORDS_PER_SECOND, abs=0.01)
    assert pointed["duration"] == pytest.approx(long["duration"] + player.MOVE_SECONDS, abs=0.01)
    assert card["duration"] > 70 / player.CARD_WORDS_PER_SECOND


def test_the_subtitle_track_follows_the_steps_end_to_end():
    steps = player.timing([_step("one two three"), _step("four five six seven")])
    track = player.vtt(steps)
    total = sum(s["duration"] for s in steps)
    assert track.startswith("WEBVTT")
    assert "00:00:00.000 --> " in track
    m, s = divmod(total, 60)
    assert f"--> 00:{int(m):02d}:{s:06.3f}" in track
    assert "four five six seven" in track


def test_the_page_carries_its_steps_and_cannot_be_closed_early_by_them(tmp_path):
    steps = [_step("a </script> in a subtitle")]
    page = player.write(tmp_path, steps, "Title", "what it is").read_text()
    start = page.index('<script id="data" type="application/json">')
    embedded = page[start:].split(">", 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded)["steps"][0]["say"] == "a </script> in a subtitle"
    assert json.loads((tmp_path / "steps.json").read_text())[0]["duration"] > 0


@pytest.mark.parametrize("topic", ["quickstart", "layout", "fitting"])
def test_every_tutorial_builds_from_the_real_gui(tmp_path, topic):
    """The whole storyboard, run as a user would, on the offscreen platform.

    A subprocess because a tutorial needs a QApplication of its own -- the
    screen size and the scale are fixed when Qt starts, and the rest of the
    suite has usually started one already.  A storyboard that stops running
    is a GUI that no longer does what its tutorial says.
    """
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from smappy.tutorial.topics import TOPICS
    assert topic in TOPICS                  # a new tutorial gets a line above
    done = subprocess.run([sys.executable, "-m", "smappy.tutorial", topic,
                           "-o", str(tmp_path)], capture_output=True, text=True,
                          timeout=600)
    if done.returncode and "libEGL" in done.stderr:
        pytest.skip("Qt's libraries are not installed")
    assert done.returncode == 0, done.stderr[-3000:]
    out = tmp_path / topic
    steps = json.loads((out / "steps.json").read_text())
    assert len(steps) > 10
    assert steps[0]["card"] and steps[-1]["card"]
    width, height = 1600, 900
    for step in steps:
        if step["image"]:
            assert (out / step["image"]).stat().st_size > 10_000
        for x, y, w, h in step["spot"] + ([step["zoom"]] if step["zoom"] else []):
            assert w > 0 and h > 0
            assert -10 <= x and x + w <= width + 10, step["say"]
            assert -10 <= y and y + h <= height + 10, step["say"]
        if step["point"]:
            assert 0 <= step["point"][0] <= width and 0 <= step["point"][1] <= height
    # STYLE.md: point at numbers, never read them -- no count in any subtitle
    import re
    for step in steps:
        assert not re.search(r"\d[\d\u2009 ,]{3,}", step["say"]), step["say"]
    assert "**1.**" in (out / "script.md").read_text()


def test_the_voice_reads_shortcuts_numbers_and_names_as_they_are_said():
    from smappy.tutorial.voice import spoken
    assert spoken("Open with Ctrl+O, or Ctrl+Shift+P.") == \
        "Open with control O, or control shift P."
    assert spoken("Reset view, or Ctrl+0.") == "Reset view, or control zero."
    assert spoken("Add file... puts") == "Add file puts"
    assert spoken("25 888 of 29 088 kept") == "25888 of 29088 kept"
    assert spoken("click ROI, 4 nm, z and PSF") == "click R O I, 4 nanometres, zed and P S F"
    assert spoken("SMAPpy opens two windows") == "smappy opens two windows"
    # names inside other words are left alone
    assert spoken("ROIManager and smappy-batch") == "ROIManager and smappy-batch"


def test_a_spoken_step_lasts_as_long_as_its_clip():
    spoken_step = _step("one two", audio="001.mp3", audio_seconds=6.0)
    card = _step("Filters.", image=None, audio="002.mp3", audio_seconds=1.0,
                 card={"title": "Filters", "body": " ".join(["word"] * 70), "figure": ""})
    player.timing([spoken_step, card])
    assert spoken_step["duration"] == pytest.approx(
        player.VOICE_LEAD + 6.0 + player.VOICE_TAIL)
    # a spoken card holds a moment, not for as long as its body takes to read
    assert card["duration"] == pytest.approx(
        player.VOICE_LEAD + 1.0 + player.VOICE_TAIL + player.CARD_HOLD)


def test_the_voice_makes_a_clip_per_step(tmp_path):
    import os
    pytest.importorskip("piper")
    model = os.environ.get("SMAPPY_PIPER_VOICE")
    if not model:
        pytest.skip("set SMAPPY_PIPER_VOICE to a Piper voice to test it")
    from smappy.tutorial.voice import narrate
    steps = narrate(tmp_path, [_step("Click Run."), _step("The picture is drawn.")], model)
    for step in steps:
        assert (tmp_path / step["audio"]).stat().st_size > 1000
        assert 0.3 < step["audio_seconds"] < 5


def test_the_index_lists_what_was_built_and_what_is_planned(tmp_path):
    built = tmp_path / "layout"
    built.mkdir()
    player.write(built, [_step("one"), _step("two", chapter="Two")], "The tour", "what it is")
    page = player.write_index(tmp_path, ["Fitting"]).read_text()
    assert page.startswith("<!doctype html>")
    assert 'href="layout/"' in page and "The tour" in page and "what it is" in page
    assert "Fitting" in page
    assert "--ground:" in page                      # the player's tokens came along
    assert (built / "index.html").read_text().startswith("<!doctype html>")


def test_the_script_numbers_the_steps_as_the_page_does():
    steps = [_step("Welcome.", image=None, chapter="Start",
                   card={"title": "Hello", "body": "<p>One</p><ul><li>a</li><li>b</li></ul>",
                         "figure": ""}),
             _step("Click Run.")]
    text = player.script(steps, "The tour")
    assert "## Start" in text and "**1.** *card:* **Hello**" in text
    assert "> One" in text and "> - a" in text and "**2.** Click Run." in text
