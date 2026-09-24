"""The page that plays a tutorial, and the video recorded from it.

The page is one self-contained HTML file beside the screenshots: the steps
are inlined as JSON, so it opens from the disk as well as from a web server,
and it needs nothing but a browser.  Everything that moves -- the spotlight,
the pointer, the zoom -- is CSS over a still screenshot, which is what keeps a
four-minute tutorial a few megabytes of pictures rather than a video.

The video is the same page in its *record* mode (no controls, the subtitle on
the picture), played by headless Chromium through Playwright and encoded by
ffmpeg.  One renderer for both means the MP4 and the page cannot disagree.

Timing is decided here, once, and written into the steps: the page, the
recording and the WebVTT track all read the same durations.  Subtitles are
read at 2.8 words a second -- a little under the 3 of broadcast subtitles, for
a lab that reads English as its second language -- with a floor, so a two-word
step is still on screen long enough to find the pointer.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

WORDS_PER_SECOND = 2.8
MIN_SECONDS = 3.5
CARD_WORDS_PER_SECOND = 3.5  # a card's body, read once its title has been seen
MOVE_SECONDS = 0.9          # the pointer's travel and the zoom; the click follows
VOICE_LEAD = 0.4            # the picture changes, then the voice starts
VOICE_TAIL = 0.7            # and a breath before the next step


def _words(text: str) -> int:
    return len(re.sub(r"<[^>]+>", " ", text).split())


def timing(steps: List[dict]) -> List[dict]:
    """Fill in each step's duration, in seconds.

    A step with a spoken clip (`voice.narrate`) lasts as long as the clip, with
    a beat either side; the pointer travels while it is spoken.  Without one,
    the reading speed decides.  A card is never shorter than its body takes
    to read, voice or not.
    """
    for step in steps:
        if step.get("audio_seconds"):
            seconds = VOICE_LEAD + step["audio_seconds"] + VOICE_TAIL
            if step.get("card"):
                seconds = max(seconds, 2.5 + _words(step["card"]["body"]) / CARD_WORDS_PER_SECOND)
            step["duration"] = round(max(MIN_SECONDS, seconds), 2)
            continue
        seconds = 1.2 + _words(step["say"]) / WORDS_PER_SECOND
        if step.get("card"):
            # the subtitle says what the card says, so they are read side by
            # side: the longer of the two, plus a beat for the figure
            seconds = max(seconds, 2.5 + _words(step["card"]["body"]) / CARD_WORDS_PER_SECOND)
        if step.get("point"):
            seconds += MOVE_SECONDS
        step["duration"] = round(max(MIN_SECONDS, seconds), 2)
    return steps


def vtt(steps: List[dict]) -> str:
    """The subtitles as a WebVTT track, for the video."""
    def stamp(t: float) -> str:
        h, rest = divmod(t, 3600)
        m, s = divmod(rest, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    lines, t = ["WEBVTT", ""], 0.0
    for i, step in enumerate(steps, 1):
        lines += [str(i), f"{stamp(t)} --> {stamp(t + step['duration'])}", step["say"], ""]
        t += step["duration"]
    return "\n".join(lines)


def write(out: Path, steps: List[dict], title: str, description: str = "",
          size=(1600, 900)) -> Path:
    """The page, ``index.html``, beside the screenshots in ``out``."""
    out = Path(out)
    steps = timing(steps)
    data = json.dumps({"title": title, "description": description,
                       "size": list(size), "steps": steps}, ensure_ascii=False)
    page = (_PAGE.replace("__TITLE__", html.escape(title))
                 .replace("__VOICE_LEAD__", str(VOICE_LEAD))
                 .replace("__DESCRIPTION__", html.escape(description))
                 .replace("__DATA__", data.replace("</", "<\\/")))
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "subtitles.vtt").write_text(vtt(steps), encoding="utf-8")
    (out / "steps.json").write_text(json.dumps(steps, indent=1, ensure_ascii=False))
    return out / "index.html"


def record(out: Path, size=(1600, 900), browser: Optional[str] = None) -> Path:
    """Play the page in record mode and keep it as ``tutorial.mp4``.

    Chromium is Playwright's; ``browser`` (or ``$SMAPPY_CHROMIUM``) points at
    another build when the pinned one is not installed.
    """
    import os
    from playwright.sync_api import sync_playwright
    out = Path(out).resolve()
    steps = json.loads((out / "steps.json").read_text())
    total = sum(s["duration"] for s in steps)
    raw = out / "_video"
    shutil.rmtree(raw, ignore_errors=True)
    browser = browser or os.environ.get("SMAPPY_CHROMIUM")
    w, h = size
    with sync_playwright() as p:
        chromium = p.chromium.launch(executable_path=browser) if browser \
            else p.chromium.launch()
        context = chromium.new_context(viewport={"width": w, "height": h},
                                       record_video_dir=str(raw),
                                       record_video_size={"width": w, "height": h})
        page = context.new_page()
        page.goto((out / "index.html").as_uri())
        page.wait_for_function("window.tutorial && window.tutorial.ready")
        lead = 0.5                      # the page settling before the first step
        page.evaluate("window.tutorial.record()")
        page.wait_for_timeout(int((total + lead + 1.0) * 1000))
        video = page.video.path()
        context.close()
        chromium.close()
    target = out / "tutorial.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.3", "-i", str(video),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-movflags", "+faststart", str(target)], check=True)
    shutil.rmtree(raw, ignore_errors=True)
    return target


# The page.  Tokens first, both themes; the stage is a fixed 1600 x 900
# coordinate system scaled to fit, so the steps' rectangles are used as they
# came out of the director.
_PAGE = r"""<title>__TITLE__</title>
<meta name="description" content="__DESCRIPTION__">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --ground: #eef1f4;
  --surface: #ffffff;
  --ink: #18202b;
  --muted: #5b6878;
  --line: #d3d9e1;
  --accent: #e8900c;          /* the 'hot' LUT's amber: the spotlight, the pointer */
  --accent-ink: #8a4f00;
  --roi: #1a8f7a;
  --stage: #1f2833;
  --sans: "IBM Plex Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground: #10151b; --surface: #19212b; --ink: #e4e9ef; --muted: #93a0b0;
    --line: #2c3743; --accent: #f5a524; --accent-ink: #f5b95a; --roi: #3cc9ad;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground: #10151b; --surface: #19212b; --ink: #e4e9ef; --muted: #93a0b0;
  --line: #2c3743; --accent: #f5a524; --accent-ink: #f5b95a; --roi: #3cc9ad;
}
* { box-sizing: border-box; }
body { background: var(--ground); color: var(--ink); font-family: var(--sans);
       font-size: 15px; line-height: 1.5; }
.wrap { max-width: 1180px; margin: 0 auto; padding-inline: 16px; padding-block: 20px 32px;
        display: grid; gap: 14px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 16px; }
header h1 { font-size: 22px; font-weight: 600; margin: 0; text-wrap: balance; }
header .kicker { font-family: var(--mono); font-size: 12px; letter-spacing: .08em;
                 text-transform: uppercase; color: var(--muted); }
header p { margin: 0; color: var(--muted); flex-basis: 100%; }

.viewport { position: relative; width: 100%; aspect-ratio: 16 / 9; max-width: 100%;
            background: var(--stage); border-radius: 10px; overflow: hidden;
            box-shadow: 0 1px 2px rgba(0,0,0,.08), 0 8px 28px rgba(15,25,40,.14); }
.stage { position: absolute; left: 0; top: 0; width: 1600px; height: 900px;
         transform-origin: 0 0; }
.world { position: absolute; inset: 0; transform-origin: 0 0;
         transition: transform .9s cubic-bezier(.45,.05,.25,1); }
.world img { position: absolute; inset: 0; width: 1600px; height: 900px;
             transition: opacity .45s ease; }
.shade { position: absolute; inset: 0; pointer-events: none; transition: opacity .4s ease; }
.shade .dim { fill: rgba(8,12,18,.55); }
.shade .ring { fill: none; stroke: var(--accent); stroke-width: 3;
               vector-effect: non-scaling-stroke; }
.pointer { position: absolute; left: 0; top: 0; width: 28px; height: 28px;
           margin: -3px 0 0 -4px; opacity: 0; pointer-events: none;
           transition: left .9s cubic-bezier(.45,.05,.25,1), top .9s cubic-bezier(.45,.05,.25,1), opacity .3s;
           filter: drop-shadow(0 2px 3px rgba(0,0,0,.45)); }
.pointer svg { width: 28px; height: 28px; }
.pointer .ripple { position: absolute; left: 4px; top: 3px; width: 44px; height: 44px;
                   margin: -22px 0 0 -22px; border-radius: 50%;
                   border: 3px solid var(--accent); opacity: 0; }
.pointer.clicking .ripple { animation: ripple .7s ease-out .9s 2; }
@keyframes ripple { from { transform: scale(.3); opacity: 1; } to { transform: scale(1.4); opacity: 0; } }

.card { position: absolute; inset: 0; display: grid; place-items: center;
        background: rgba(12,17,24,.72); opacity: 0; transition: opacity .45s ease;
        pointer-events: none; }
.card.on { opacity: 1; }
.card .panel-box { width: 1080px; max-height: 800px; background: var(--surface); color: var(--ink);
                   border-radius: 14px; padding: 44px 60px; display: grid; gap: 14px;
                   box-shadow: 0 20px 60px rgba(0,0,0,.35); }
.card .eyebrow { font-family: var(--mono); font-size: 18px; letter-spacing: .1em;
                 text-transform: uppercase; color: var(--accent-ink); }
.card h2 { font-size: 50px; line-height: 1.1; margin: 0; font-weight: 600; text-wrap: balance; }
.card .body { font-size: 25px; line-height: 1.45; max-width: 62ch; }
.card .body p, .card .body ul { margin: 0 0 12px; }
.card .body li { margin-bottom: 6px; }
.card .figure svg { width: 100%; height: auto; max-height: 250px; color: var(--ink); }
.figure .panel { fill: var(--ground); stroke: var(--line); stroke-width: 1.5; }
.figure .dot { fill: var(--accent); }
.figure .dot.strong { stroke: var(--accent-ink); stroke-width: 2; }
.figure .faint { fill: var(--muted); opacity: .45; }
.figure .arrow { stroke: currentColor; stroke-width: 2; fill: none; }
.figure .roi { fill: none; stroke: var(--roi); stroke-width: 2.5; stroke-dasharray: 7 4; }
.figure .dot-text { fill: var(--accent); font-weight: 600; }
.figure .faint-text { fill: var(--muted); }
.figure .roi-text { fill: var(--roi); font-weight: 600; }

.caption { position: absolute; left: 0; right: 0; bottom: 0; display: none;
           padding: 18px 120px 26px; text-align: center; font-size: 30px; line-height: 1.35;
           color: #fff; background: linear-gradient(transparent, rgba(0,0,0,.78) 30%); }

.subtitle { min-height: 3.2em; font-size: 20px; line-height: 1.45; margin: 0;
            max-width: 70ch; text-wrap: pretty; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; }
.controls button { font: inherit; font-size: 14px; color: var(--ink); background: var(--surface);
                   border: 1px solid var(--line); border-radius: 7px; padding: 6px 12px;
                   cursor: pointer; min-width: 44px; }
.controls button:hover { border-color: var(--accent); }
.controls button:focus-visible, .chapters button:focus-visible, .track:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px; }
.controls #play { min-width: 84px; font-weight: 600; }
.counter { font-family: var(--mono); font-size: 13px; color: var(--muted);
           font-variant-numeric: tabular-nums; }
.track { flex: 1 1 260px; display: flex; gap: 3px; height: 10px; cursor: pointer; }
.track .seg { position: relative; height: 100%; background: var(--line); border-radius: 3px; overflow: hidden; }
.track .seg i { position: absolute; inset: 0; width: 0; background: var(--accent); }
.chapters { display: flex; flex-wrap: wrap; gap: 6px; margin: 0; padding: 0; list-style: none; }
.chapters button { font: inherit; font-size: 13px; color: var(--muted); background: none;
                   border: 1px solid var(--line); border-radius: 999px; padding: 3px 11px; cursor: pointer; }
.chapters button[aria-current="true"] { color: var(--ink); border-color: var(--accent);
                                        background: var(--surface); }
.hint { font-size: 13px; color: var(--muted); margin: 0; }
kbd { font-family: var(--mono); font-size: 12px; border: 1px solid var(--line);
      border-bottom-width: 2px; border-radius: 4px; padding: 0 5px; background: var(--surface); }

body.recording { background: #000; }
body.recording .wrap { max-width: none; padding: 0; gap: 0; }
body.recording header, body.recording .below { display: none; }
body.recording .viewport { border-radius: 0; box-shadow: none; }
body.recording .caption { display: block; }
/* the subtitle takes the bottom of the frame: a card sits above it */
body.recording .card { padding-bottom: 110px; }
body.recording .card .panel-box { max-height: 690px; }
@media (prefers-reduced-motion: reduce) {
  .world, .pointer, .shade, .card, .world img { transition: none; }
  .pointer.clicking .ripple { animation: none; }
}
@media (max-width: 560px) {
  .subtitle { font-size: 17px; }
  header h1 { font-size: 19px; }
}
</style>

<div class="wrap">
  <header>
    <span class="kicker">smappy tutorial</span>
    <h1 id="title">__TITLE__</h1>
    <p>__DESCRIPTION__</p>
  </header>
  <div class="viewport" id="viewport">
    <div class="stage" id="stage">
      <div class="world" id="world">
        <img id="imgA" alt="">
        <img id="imgB" alt="" style="opacity:0">
        <svg class="shade" id="shade" viewBox="0 0 1600 900" preserveAspectRatio="none"></svg>
        <div class="pointer" id="pointer"><span class="ripple"></span>
          <svg viewBox="0 0 28 28"><path d="M4 3 L4 22 L9 17.5 L12.5 25 L16 23.5 L12.6 16 L19.5 16 Z"
            fill="#fff" stroke="#111" stroke-width="1.6" stroke-linejoin="round"/></svg></div>
      </div>
      <div class="card" id="card"><div class="panel-box">
        <div class="eyebrow" id="cardEyebrow"></div>
        <h2 id="cardTitle"></h2>
        <div class="body" id="cardBody"></div>
        <div class="figure" id="cardFigure"></div>
      </div></div>
      <div class="caption" id="caption"></div>
    </div>
  </div>
  <div class="below" style="display:grid; gap:12px">
    <p class="subtitle" id="subtitle" aria-live="polite"></p>
    <div class="controls">
      <button id="prev" type="button" aria-label="previous step">&#9664;</button>
      <button id="play" type="button">Play</button>
      <button id="next" type="button" aria-label="next step">&#9654;</button>
      <button id="sound" type="button" aria-pressed="true" hidden>Voice on</button>
      <div class="track" id="track" role="slider" tabindex="0" aria-label="progress"></div>
      <span class="counter" id="counter"></span>
    </div>
    <ul class="chapters" id="chapters"></ul>
    <p class="hint"><kbd>space</kbd> play or pause &nbsp; <kbd>&larr;</kbd> <kbd>&rarr;</kbd> step &nbsp; <span id="soundHint" hidden><kbd>m</kbd> voice on or off &nbsp;</span> click a chapter to jump to it</p>
  </div>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  var data = JSON.parse(document.getElementById("data").textContent);
  var steps = data.steps, W = data.size[0], H = data.size[1];
  var $ = function (id) { return document.getElementById(id); };
  var stage = $("stage"), world = $("world"), shade = $("shade"), pointer = $("pointer");
  var imgs = [$("imgA"), $("imgB")], front = 0, shown = null;
  var index = 0, playing = false, timer = null, started = 0, elapsed = 0, recording = false;
  // The voice: one clip per step, fetched when the step is reached.  A browser
  // plays sound only after a click, which Play is, so it starts with Play.
  var voice = new Audio(), voiceStep = -1, leadTimer = null, LEAD = __VOICE_LEAD__;
  var sound = steps.some(function (s) { return s.audio; });

  // chapters: a step without one belongs to the last one named
  var chapters = [];
  steps.forEach(function (s, i) {
    if (s.chapter || !chapters.length) chapters.push({ name: s.chapter || "", start: i });
    s._chapter = chapters.length - 1;
  });
  chapters.forEach(function (c, k) {
    var li = document.createElement("li"), b = document.createElement("button");
    b.type = "button"; b.textContent = c.name; b.addEventListener("click", function () { go(c.start); });
    li.appendChild(b); $("chapters").appendChild(li); c.button = b;
  });
  steps.forEach(function (s) {
    var seg = document.createElement("div");
    seg.className = "seg"; seg.style.flexGrow = s.duration; seg.appendChild(document.createElement("i"));
    $("track").appendChild(seg); s._seg = seg.firstChild;
    if (s.image) { var pre = new Image(); pre.src = s.image; }
  });

  function fit() {
    var v = $("viewport"), k = v.clientWidth / W;
    stage.style.transform = "scale(" + k + ")";
  }
  window.addEventListener("resize", fit);

  function zoomTransform(z) {
    if (!z) return "none";
    var k = Math.min(W / z[2], H / z[3]);
    var x = z[0] + z[2] / 2 - W / (2 * k), y = z[1] + z[3] / 2 - H / (2 * k);
    return "scale(" + k + ") translate(" + (-x) + "px," + (-y) + "px)";
  }

  function spotlight(rects) {
    if (!rects || !rects.length) { shade.style.opacity = 0; return; }
    var holes = "", rings = "";
    rects.forEach(function (r) {
      holes += '<rect x="' + r[0] + '" y="' + r[1] + '" width="' + r[2] + '" height="' + r[3] + '" rx="6" fill="#000"/>';
      rings += '<rect class="ring" x="' + r[0] + '" y="' + r[1] + '" width="' + r[2] + '" height="' + r[3] + '" rx="6"/>';
    });
    shade.innerHTML = '<defs><mask id="m"><rect width="1600" height="900" fill="#fff"/>' + holes +
      '</mask></defs><rect class="dim" width="1600" height="900" mask="url(#m)"/>' + rings;
    shade.style.opacity = 1;
  }

  function show(i) {
    var s = steps[i];
    // a card sits over the last screenshot, so the program stays in view
    var picture = s.image || shown;
    if (picture && picture !== shown) {
      var back = imgs[1 - front];
      back.src = picture; back.style.opacity = 1; imgs[front].style.opacity = 0;
      front = 1 - front; shown = picture;
    }
    world.style.transform = s.card ? "none" : zoomTransform(s.zoom);
    spotlight(s.card ? [] : s.spot);
    if (s.point && !s.card) {
      pointer.style.opacity = 1;
      pointer.style.left = s.point[0] + "px"; pointer.style.top = s.point[1] + "px";
      pointer.classList.remove("clicking"); void pointer.offsetWidth;
      if (s.click) pointer.classList.add("clicking");
    } else {
      pointer.style.opacity = 0; pointer.classList.remove("clicking");
    }
    var card = $("card");
    if (s.card) {
      $("cardEyebrow").textContent = i === 0 ? "smappy tutorial" : chapters[s._chapter].name;
      $("cardTitle").textContent = s.card.title;
      $("cardBody").innerHTML = s.card.body;
      $("cardFigure").innerHTML = s.card.figure || "";
      card.classList.add("on");
    } else card.classList.remove("on");
    $("subtitle").textContent = s.say;
    $("caption").textContent = s.say;
    $("counter").textContent = (i + 1) + " / " + steps.length;
    chapters.forEach(function (c, k) { c.button.setAttribute("aria-current", k === s._chapter ? "true" : "false"); });
    steps.forEach(function (t, k) { t._seg.style.width = k < i ? "100%" : "0"; });
  }

  function speak(i, fresh) {
    clearTimeout(leadTimer);
    var s = steps[i];
    if (!sound || !s.audio) { voice.pause(); return; }
    if (fresh || voiceStep !== i) {
      voice.pause(); voice.src = s.audio; voiceStep = i;
      leadTimer = setTimeout(function () {
        if (playing && index === i) voice.play().catch(function () {});
      }, LEAD * 1000);
    } else if (!voice.ended) {
      voice.play().catch(function () {});          // resumed where it was paused
    }
    var next = steps[i + 1];                        // the next clip, ready in time
    if (next && next.audio) { var pre = new Audio(); pre.preload = "auto"; pre.src = next.audio; }
  }
  function speaking() { return sound && voiceStep === index && !voice.paused && !voice.ended; }

  function tick() {
    var s = steps[index];
    var t = elapsed + (playing ? (performance.now() - started) / 1000 : 0);
    s._seg.style.width = Math.min(100, 100 * t / s.duration) + "%";
    // a clip that runs long (a slow start, a slow machine) is not cut off
    if (playing && t >= s.duration && !speaking()) {
      if (index + 1 < steps.length) {
        index++; elapsed = 0; started = performance.now(); show(index); speak(index, true);
      }
      else { pause(); s._seg.style.width = "100%"; if (recording) window.tutorial.done = true; return; }
    }
    if (playing) timer = requestAnimationFrame(tick);
  }
  function play() {
    if (index === steps.length - 1 && elapsed >= steps[index].duration) go(0);
    playing = true; started = performance.now(); $("play").textContent = "Pause";
    speak(index, elapsed === 0);
    cancelAnimationFrame(timer); timer = requestAnimationFrame(tick);
  }
  function pause() {
    if (playing) elapsed += (performance.now() - started) / 1000;
    playing = false; $("play").textContent = "Play"; cancelAnimationFrame(timer);
    clearTimeout(leadTimer); voice.pause();
  }
  function go(i) {
    index = Math.max(0, Math.min(steps.length - 1, i)); elapsed = 0; started = performance.now();
    show(index); if (playing) speak(index, true); else { voice.pause(); voiceStep = -1; }
    tick();
  }
  function toggleSound() {
    sound = !sound;
    $("sound").textContent = sound ? "Voice on" : "Voice off";
    $("sound").setAttribute("aria-pressed", sound ? "true" : "false");
    if (sound && playing) speak(index, true); else { clearTimeout(leadTimer); voice.pause(); }
  }
  if (sound) { $("sound").hidden = false; $("soundHint").hidden = false; }
  $("sound").addEventListener("click", toggleSound);

  $("play").addEventListener("click", function () { playing ? pause() : play(); });
  $("prev").addEventListener("click", function () { go(index - 1); });
  $("next").addEventListener("click", function () { go(index + 1); });
  $("track").addEventListener("click", function (e) {
    var r = this.getBoundingClientRect(), f = (e.clientX - r.left) / r.width;
    var total = steps.reduce(function (a, s) { return a + s.duration; }, 0), t = f * total, k = 0;
    while (k < steps.length - 1 && t > steps[k].duration) { t -= steps[k].duration; k++; }
    go(k);
  });
  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "INPUT") return;
    if (e.key === " ") { e.preventDefault(); playing ? pause() : play(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); go(index + 1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); go(index - 1); }
    else if ((e.key === "m" || e.key === "M") && !$("sound").hidden) toggleSound();
  });

  window.tutorial = {
    ready: true, done: false,
    record: function () {
      recording = true; document.body.classList.add("recording"); fit(); go(0); play();
    }
  };
  fit(); show(0);
})();
</script>
"""
