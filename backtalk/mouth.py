# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
# Modified 2026-11 by Ethan: Voicebox engine, lead-in split, prefetch, GPU lock, captions, inter-chunk pause.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The mouth — streaming sentence-chunked TTS, played through one
long-lived output stream.

Default engine: Kokoro, in-process. Local, free, no server, no API key,
~0.2s to first audio once warm. Optional premium engine: ElevenLabs on
YOUR key — read from the system keychain, never from a file (see
_get_elevenlabs_key) — with Kokoro as the automatic fallback: the voice
degrades instead of going mute if the cloud fails.

Optional PRIMARY engine (checked before ElevenLabs): Voicebox, a local
voice clone running on your own hardware via its FastAPI server
(github.com/jamiepine/voicebox), see voicebox.* in backtalk.json. Unlike
ElevenLabs and Kokoro, Voicebox renders the ENTIRE chunk server-side
before sending anything back, so there is no progressive stream to hide
latency behind. Same fallback chain on any failure: Voicebox ->
ElevenLabs -> Kokoro. Degrade, never mute.

VOICEBOX LATENCY FLOOR (measured with chatterbox_turbo, warm model, on an
RTX 4070 Ti): render time scales with text length on top of a ~1.1-1.3s
floor that never goes lower however short the text is (one word ~1.2s, a
9-word sentence ~2.8s, a 20-word sentence ~6.3s). Two mitigations hide
that latency rather than remove it:

1. LEAD-IN SPLIT (say_chunk, cold start only): the first chunk of a
   fresh reply, with nothing queued and nothing playing, gets a short
   lead-in clause peeled off (see _split_lead_in) so the first sound
   reaches the speaker at the ~1.2s floor instead of after the whole
   first sentence renders.
2. TWO-AHEAD PREFETCH (_maybe_prefetch_ahead): while the current chunk
   plays, the next two are rendered in background threads, so playback
   usually finds them already done. If a prefetch didn't finish or never
   started, the chunk is synthesized inline exactly as without it: this
   can only shorten a wait, never add one.

CAPTIONS: every chunk can carry per-word {word, start, end} timing,
published on the signal bus (signals.caption) and shown live in the
console (caption_tty). ElevenLabs supplies real alignment; Voicebox and
Kokoro audio is timed by running the already-loaded Whisper model over
the rendered clip (ears.align_words), with a length-proportional estimate
(_estimate_words) only as the last fallback.

Sentences are synthesized one at a time and queued for playback, so the
first sentence is audible while later ones are still rendering. Playback
is cancellable mid-word: set the stop event and the speaker goes silent
within one audio block plus the device buffer (~0.15s).

HARD-WON AUDIO LAW #1 — ONE long-lived OutputStream, reused for every
sentence for the life of the process. A fresh stream per sentence gives
an audible onset blip or a beat of dead air on plenty of audio setups
(USB interfaces, Bluetooth, streaming mixers that latch onto each new
stream late). Proven by A/B test; do not "simplify" this away.

HARD-WON AUDIO LAW #2 — buffer ~0.75s of synthesized audio before a
sentence starts playing, so a slower machine never underruns into
slow-motion garble.
"""
import base64
import io
import itertools
import json
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import wave

import numpy as np
import sounddevice as sd

from backtalk.config import CFG
from backtalk.vlog import log

KOKORO_RATE = 24000
EL_RATE = 44100
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_ATEMPO_RE = re.compile(r"atempo=([0-9.]+)")


def _mastering_atempo(master: str) -> float:
    """The leading atempo= value in the ElevenLabs mastering chain: pace
    applied AFTER synthesis. Raw alignment timestamps come off the
    unmastered audio, so they need scaling by 1/atempo to land on the
    audio actually played. 1.0 (no change) if the chain doesn't set one."""
    m = _ATEMPO_RE.search(master or "")
    try:
        return float(m.group(1)) if m else 1.0
    except ValueError:
        return 1.0


def _words_from_alignment(chars, starts, ends) -> list:
    """Group ElevenLabs' character-level alignment into words (split on
    whitespace), each carrying start/end in seconds. ElevenLabs' streamed
    alignment is one continuous timeline across the whole response, not
    restarted per line, so the times are used as-is."""
    words = []
    cur = ""
    w_start = w_end = None
    for ch, s, e in zip(chars, starts, ends):
        if ch.isspace():
            if cur:
                words.append({"word": cur, "start": round(w_start, 3),
                              "end": round(w_end, 3)})
                cur = ""
                w_start = None
        else:
            if w_start is None:
                w_start = s
            w_end = e
            cur += ch
    if cur:
        words.append({"word": cur, "start": round(w_start, 3),
                      "end": round(w_end, 3)})
    return words


def _spoken_weight(word: str) -> float:
    """How long `word` takes to SAY, relative to an ordinary word.

    Character count is a fair stand-in for an ordinary word, but a numeral
    breaks it: "12345" is 5 characters yet is spoken as several words.
    Each digit counts as one word-equivalent (~4.5 chars, matching the
    plain character-count weight for ordinary words), otherwise the
    caption races ahead of the audio through number sequences."""
    digits = sum(1 for ch in word if ch.isdigit())
    if not digits:
        return max(len(word), 1)
    return digits * 4.5


def _estimate_words(text: str, duration: float) -> list:
    """Fallback word timing when there is no real alignment: distribute the
    sentence proportionally by spoken weight across `duration`, the ACTUAL
    measured length of the rendered audio (samples / rate), never a guessed
    speaking rate. Anchoring to the real duration means the last word
    always ends exactly where the audio does, so the error can't compound
    across a long chunk the way a guessed chars-per-second rate does."""
    words = text.split()
    if not words or duration <= 0:
        return []
    weights = [_spoken_weight(w) for w in words]
    total = sum(weights) + 0.5 * (len(words) - 1)   # + inter-word gaps
    if total <= 0:
        return []
    scale = duration / total
    t = 0.0
    out = []
    for w, weight in zip(words, weights):
        dur = weight * scale
        out.append({"word": w, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur + 0.5 * scale
    return out

_pipe = None
_pipe_lock = threading.Lock()


def _ensure_espeak():
    """kokoro phonemizes through system espeak-ng (its bundled loader
    ships a broken build path — found the hard way; upstream's own docs
    say install the system package). Help phonemizer find it in the
    usual homes when the env isn't already set."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    candidates = (
        "/opt/homebrew/lib/libespeak-ng.dylib",       # macOS arm64 (brew)
        "/usr/local/lib/libespeak-ng.dylib",          # macOS intel (brew)
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",  # debian/ubuntu
        "/usr/lib/libespeak-ng.so.1",                 # other linux
        "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll",       # windows
        "C:\\Program Files (x86)\\eSpeak NG\\libespeak-ng.dll",
    )
    for lib in candidates:
        if os.path.exists(lib):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib
            break


# Every espeak library filename phonemizer might copy, on any platform. A
# directory holding exactly one of these and nothing else is a phonemizer
# scratch dir and is not plausibly anything else.
_ESPEAK_LIB_NAMES = (
    "espeak-ng.dll",
    "libespeak-ng.dll",
    "libespeak-ng.so",
    "libespeak-ng.so.1",
    "libespeak-ng.dylib",
)


def _is_orphan_espeak_tempdir(path: str) -> bool:
    """True only for a directory whose ENTIRE contents are one espeak
    library. That signature is what makes it safe to point a delete at a
    shared temp folder: one file, and its name is one of five."""
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return len(entries) == 1 and entries[0] in _ESPEAK_LIB_NAMES


def _sweep_orphan_espeak_tempdirs():
    """Delete espeak scratch dirs left behind by previous runs.

    phonemizer copies the espeak shared library into a fresh temp dir for
    every backend it builds, because espeak-ng keeps its state in globals
    and the loader refuses the same file twice. Kokoro builds several
    backends, so ONE start leaves several behind.

    On POSIX that cleanup rides a finalizer and usually happens. On
    Windows phonemizer can only register it with atexit, and atexit does
    not run when a process is KILLED rather than exited -- so anything
    stopping the voice line by terminating it, which is most launchers and
    every supervisor, leaks every directory it ever made. Sixty had piled
    up on the machine where this was found, and fifteen were sitting on
    the author's own Mac when it was reviewed: the POSIX path is not as
    reliable as it looks either. The count only ever grows.

    Patching phonemizer where it is installed is not a fix, because the
    launcher runs a dependency sync that would overwrite it. Sweeping at
    our own startup bounds the total at one run's worth instead.

    Two things make deleting from a shared temp folder safe, and only the
    first is ours: the signature above is narrow enough that nothing else
    matches it, and anything we are not permitted to remove raises and is
    skipped. On Windows a loaded library cannot be deleted at all, so a
    live instance is protected by the OS rather than by us noticing it.
    POSIX does not work that way, but a process that has already mapped
    the library keeps it after the unlink, so a running instance is
    unharmed either way.
    """
    root = tempfile.gettempdir()
    swept = 0
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not _is_orphan_espeak_tempdir(path):
            continue
        try:
            shutil.rmtree(path)
            swept += 1
        except OSError:
            pass          # in use, or not ours. Leaving it is correct.
    if swept:
        log(f"[mouth] swept {swept} orphaned espeak temp dir(s)")


def warm():
    """Load the Kokoro pipeline (first call downloads the model to the
    HF cache). Called at startup while the greeting text is composed."""
    global _pipe
    with _pipe_lock:
        if _pipe is None:
            _ensure_espeak()
            # Before kokoro makes this run's scratch dirs, clear the ones
            # earlier runs could not clean up on their way out.
            _sweep_orphan_espeak_tempdirs()
            from kokoro import KPipeline
            # The voice name's first letter IS the language pipeline:
            # a=American English, b=British English, e/f/h/i/j/p/z = the
            # other shipped languages. bm_lewis -> 'b'.
            lang = (CFG["voice"] or "bm_lewis")[0]
            log(f"[mouth] loading kokoro (lang '{lang}', "
                f"voice {CFG['voice']})...")
            _pipe = KPipeline(lang_code=lang)
            log("[mouth] voice ready")
    return _pipe


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


# Cold-start lead-in split (module docstring, mitigation #1). 45 chars:
# the rest of the sentence renders in the background while the lead-in
# PLAYS, so the lead-in has to be long enough to cover that render, and
# 30 chars often wasn't (an audible pause between the two). Below
# _LEAD_IN_MIN_REST_CHARS there is nothing worth splitting off.
_LEAD_IN_MAX_CHARS = 45
_LEAD_IN_MIN_REST_CHARS = 12
_LEAD_IN_SPLIT_RE = re.compile(r"[,;:—-]\s")


def _split_lead_in(text: str) -> tuple[str, str | None]:
    """Peel a short lead-in clause off the front of *text*.

    Splits at the last comma/semicolon/colon/dash within _LEAD_IN_MAX_CHARS,
    else the last word boundary within it. Returns (text, None), meaning
    "don't bother", when splitting wouldn't leave a meaningful remainder."""
    text = text.strip()
    if len(text) <= _LEAD_IN_MAX_CHARS + _LEAD_IN_MIN_REST_CHARS:
        return text, None
    window = text[:_LEAD_IN_MAX_CHARS]
    split = -1
    for m in _LEAD_IN_SPLIT_RE.finditer(window):
        split = m.end()
    if split == -1:
        split = window.rfind(" ")
    if split <= 0:
        return text, None
    lead, rest = text[:split].strip(), text[split:].strip()
    if not lead or not rest or len(rest) < _LEAD_IN_MIN_REST_CHARS:
        return text, None
    return lead, rest


def _stream_kokoro(text: str):
    """One sentence -> int16 PCM chunks at 24kHz, in-process."""
    pipe = warm()
    try:
        speed = float(CFG.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    for _, _, audio in pipe(text, voice=CFG["voice"], speed=speed):
        a = np.asarray(audio, dtype=np.float32)
        if a.size:
            yield (np.clip(a, -1.0, 1.0) * 32767).astype(np.int16)


_voicebox_profile_cache: dict[tuple[str, str], str] = {}
_voicebox_launch_attempted = False
_voicebox_launch_lock = threading.Lock()

# The prefetch pipeline can fire two background renders at once, and
# Voicebox's generate has no server-side queueing: two concurrent requests
# both hit the GPU and each slows the other down. Measured: a request that
# took ~3.5s alone took ~8.6s under real session load, and timeouts
# followed as the backlog compounded across a multi-sentence reply.
# Serializing our own requests keeps prefetch's early start without
# letting two renders contend for the same GPU.
_voicebox_generate_lock = threading.Lock()


def _voicebox_ready() -> bool:
    vb = CFG["voicebox"]
    return bool(vb.get("enabled") and vb.get("profile") and vb.get("base_url"))


def _voicebox_root() -> str:
    """The Voicebox source checkout, for auto-launching a downed backend.
    Defaults to a `voicebox` folder next to this repo; override with
    voicebox.repo_dir in backtalk.json."""
    configured = CFG["voicebox"].get("repo_dir")
    if configured:
        return os.path.expanduser(configured)
    from pathlib import Path
    return str(Path(__file__).resolve().parent.parent.parent / "voicebox")


def _voicebox_port(base_url: str) -> str:
    from urllib.parse import urlparse
    try:
        return str(urlparse(base_url).port or 17493)
    except ValueError:
        return "17493"


def _try_autostart_voicebox():
    """Self-heal: the Voicebox backend is a separate long-running process
    this repo does not supervise, so a reboot or a killed process would
    otherwise drop the voice to the fallback for the rest of the session
    with nothing ever bringing it back. Fire-and-forget, once per process
    lifetime. The sentence that triggered this still falls back (the
    backend takes several seconds to boot and load a model), but the NEXT
    synth_stream call finds it up.

    Only works for a Voicebox SOURCE checkout with its backend venv set
    up. If you run the Voicebox desktop app instead, keep the app open:
    it serves the same API, and this logs once and stays out of the way."""
    global _voicebox_launch_attempted
    with _voicebox_launch_lock:
        if _voicebox_launch_attempted:
            return
        _voicebox_launch_attempted = True
    import subprocess
    root = _voicebox_root()
    if sys.platform == "win32":
        uvicorn = os.path.join(root, "backend", "venv", "Scripts", "uvicorn.exe")
    else:
        uvicorn = os.path.join(root, "backend", "venv", "bin", "uvicorn")
    if not os.path.exists(uvicorn):
        log(f"[mouth] voicebox backend not found at {uvicorn}, staying on "
            f"the fallback voice (set voicebox.repo_dir, or keep the "
            f"Voicebox app running)")
        return
    port = _voicebox_port(CFG["voicebox"]["base_url"])
    log("[mouth] voicebox backend unreachable, launching it")
    try:
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(
            [uvicorn, "backend.main:app", "--port", port],
            cwd=root, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
    except Exception as e:
        log(f"[mouth] could not launch voicebox backend: {e}")


def ensure_voicebox_warm(deadline: float = 10.0):
    """Wake Voicebox the moment backtalk starts, so the very FIRST line of
    the session doesn't gamble on a cold backend and fall back for just
    that one line. A no-op when Voicebox isn't configured; one quick
    reachability check when it's already running (the common case). Only
    a genuinely cold backend costs real time here, capped at `deadline`
    seconds so a broken backend can never hang startup. If it's still not
    up by then, synth_stream's per-call fallback covers it."""
    if not _voicebox_ready():
        return
    import httpx
    base_url = CFG["voicebox"]["base_url"].rstrip("/")

    def _reachable() -> bool:
        try:
            httpx.get(f"{base_url}/profiles", timeout=1.5).raise_for_status()
            return True
        except Exception:
            return False

    if _reachable():
        return
    _try_autostart_voicebox()
    started = time.monotonic()
    while time.monotonic() - started < deadline:
        time.sleep(0.4)
        if _reachable():
            log("[mouth] voicebox warm")
            return
    log(f"[mouth] voicebox still not reachable after {deadline:.0f}s, "
        f"first line(s) may use the fallback voice")


def _resolve_voicebox_profile(base_url: str, name: str, timeout: float) -> str:
    """Voicebox's /generate/stream wants a profile ID, not the display
    name shown in the app, so resolve once and cache. Cleared and
    re-resolved automatically on a 404 (profile renamed or recreated)."""
    import httpx
    key = (base_url, name)
    cached = _voicebox_profile_cache.get(key)
    if cached:
        return cached
    r = httpx.get(f"{base_url}/profiles", timeout=timeout)
    r.raise_for_status()
    for p in r.json():
        if p.get("name") == name:
            _voicebox_profile_cache[key] = p["id"]
            return p["id"]
    raise RuntimeError(f"voicebox profile {name!r} not found")


def _stream_voicebox(text: str, timeout: float, words_out: list | None = None):
    """Voicebox -> int16 PCM. /generate/stream renders the whole chunk
    server-side and only then sends the finished WAV, so this yields once.

    `words_out`, if given, is filled with real per-word timing (Whisper over
    the rendered clip, see ears.align_words) BEFORE the audio is yielded.
    That ordering matters: since this yields once, _play_stream pulls the
    whole chunk on its first pull and starts playing immediately, so words
    filled after the yield would not exist yet when the caption is first
    published. Falls back to _estimate_words if alignment fails, so the
    caption degrades to approximate, never goes dark."""
    import httpx
    vb = CFG["voicebox"]
    base_url = vb["base_url"].rstrip("/")
    engine = vb.get("engine") or "chatterbox_turbo"
    try:
        profile_id = _resolve_voicebox_profile(base_url, vb["profile"], timeout)
        payload = {"profile_id": profile_id, "text": text, "engine": engine}
        with _voicebox_generate_lock:
            r = httpx.post(f"{base_url}/generate/stream", json=payload, timeout=timeout)
            if r.status_code == 404:
                # Profile likely recreated with a new id: refresh once, retry.
                _voicebox_profile_cache.pop((base_url, vb["profile"]), None)
                profile_id = _resolve_voicebox_profile(base_url, vb["profile"], timeout)
                payload["profile_id"] = profile_id
                r = httpx.post(f"{base_url}/generate/stream", json=payload, timeout=timeout)
        r.raise_for_status()
    except (httpx.ConnectError, httpx.ConnectTimeout):
        # ConnectError: nothing listening. ConnectTimeout: the backend is
        # hung, or a firewall silently drops the connection. Either way the
        # request still falls through to the next engine via synth_stream;
        # this just gives the self-heal a chance to fire for both.
        _try_autostart_voicebox()
        raise
    with wave.open(io.BytesIO(r.content), "rb") as w:
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    if not pcm.size:
        raise RuntimeError("voicebox returned empty audio")
    if words_out is not None:
        from backtalk import ears
        aligned = ears.align_words(r.content, text)
        words_out.extend(aligned if aligned else _estimate_words(text, len(pcm) / rate))
    yield rate, pcm


def _stream_elevenlabs(text: str, timeout: float, words_out: list | None = None):
    """ElevenLabs -> ffmpeg streaming decode -> int16 PCM at 44.1kHz.

    THE ELEVENLABS DOCTRINE, learned the expensive way:
    - fetch mp3_44100_128 and decode locally (raw 44.1k PCM needs their
      Pro tier; the mp3 decode hides inside network wait anyway)
    - turbo model, stability 0.5, similarity 0.75
    - never the multilingual model for English, never style above 0 —
      both make delivery slow and dull
    - their site previews are MASTERED demo clips; raw API output never
      matches them, so master locally (the ffmpeg chain in config)
    ffmpeg reads stdin as we feed it, so playback still starts before
    synthesis finishes.

    When `words_out` is passed (captions), this uses the
    /stream/with-timestamps endpoint: one JSON object per line carrying
    audio_base64 plus character alignment, decoded and fed to ffmpeg line
    by line. The alignment is one continuous timeline for the whole
    response, arriving in a few large bursts, so times are used exactly as
    sent (re-basing them per line double-counts and freezes the caption).
    Words are appended to `words_out` as each burst lands."""
    import subprocess

    import httpx

    el = CFG["elevenlabs"]
    key = _get_elevenlabs_key()
    want_words = words_out is not None
    suffix = "/with-timestamps" if want_words else ""
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/"
           f"{el['voice_id']}/stream{suffix}?output_format=mp3_44100_128")
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "quiet", "-i", "pipe:0",
         "-af", el["master"],
         "-f", "s16le", "-ar", str(EL_RATE), "-ac", "1", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    feed_error: list = []
    atempo = _mastering_atempo(el["master"])

    def _feed():
        try:
            with httpx.stream("POST", url, headers={"xi-api-key": key},
                              json={"text": text, "model_id": el["model"],
                                    "voice_settings": {
                                        "stability": 0.5,
                                        "similarity_boost": 0.75}},
                              timeout=timeout) as r:
                r.raise_for_status()
                if not want_words:
                    for chunk in r.iter_bytes(chunk_size=4096):
                        proc.stdin.write(chunk)
                    return
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    b64 = obj.get("audio_base64")
                    if b64:
                        proc.stdin.write(base64.b64decode(b64))
                    align = obj.get("alignment") or obj.get("normalized_alignment")
                    if not align:
                        continue
                    try:
                        words = _words_from_alignment(
                            align.get("characters") or [],
                            align.get("character_start_times_seconds") or [],
                            align.get("character_end_times_seconds") or [])
                        for w in words:
                            # raw synthesis time -> post-mastering playback time
                            w["start"] = round(w["start"] / atempo, 3)
                            w["end"] = round(w["end"] / atempo, 3)
                        words_out.extend(words)
                    except Exception:
                        pass   # alignment is a bonus; never break the audio for it
        except Exception as e:
            feed_error.append(e)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    t = threading.Thread(target=_feed, daemon=True)
    t.start()
    carry = b""
    got_audio = False
    while True:
        data = proc.stdout.read(8820)
        if not data:
            break
        data = carry + data
        usable = len(data) - (len(data) % 2)
        carry = data[usable:]
        if usable:
            got_audio = True
            yield np.frombuffer(data[:usable], dtype=np.int16)
    proc.wait(timeout=10)
    if feed_error and not got_audio:
        raise feed_error[0]


_el_key_cache: str | None = None


def _key_slot() -> str:
    """The credential-store entry name, so someone who already keeps a key
    under their own name points at it instead of storing a second copy."""
    return str(CFG["elevenlabs"].get("key_slot") or "backtalk-elevenlabs")


def _get_elevenlabs_key() -> str:
    """The API key, from the most secure store available — NEVER from a
    file in this repo. Lookup order:
      1. macOS Keychain, item `backtalk-elevenlabs` by default (change it
         with elevenlabs.key_slot) — seed it once with:
         security add-generic-password -a "$USER" -s backtalk-elevenlabs -T /usr/bin/security -w
         (it prompts for the secret; -T lets this code read it without a
         GUI prompt every launch)
      2. Linux secret-tool (libsecret):
         secret-tool store --label backtalk service backtalk-elevenlabs
      3. the ELEVENLABS_API_KEY environment variable — the last-resort
         fallback, and the only option on Windows for now. Know the
         tradeoff: an export line in a shell profile is a plaintext key
         on disk, which is exactly what the keychain path avoids."""
    global _el_key_cache
    if _el_key_cache is not None:
        return _el_key_cache
    import subprocess
    key = ""
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["security", "find-generic-password",
                                "-s", _key_slot(), "-w"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                key = r.stdout.strip()
        elif sys.platform.startswith("linux"):
            from shutil import which
            if which("secret-tool"):
                r = subprocess.run(["secret-tool", "lookup", "service",
                                    _key_slot()],
                                   capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    key = r.stdout.strip()
    except Exception:
        pass
    _el_key_cache = key or os.environ.get("ELEVENLABS_API_KEY", "")
    return _el_key_cache


def _elevenlabs_ready() -> bool:
    el = CFG["elevenlabs"]
    return bool(el.get("enabled") and el.get("voice_id")
                and _get_elevenlabs_key())


def synth_stream(text: str, timeout: float = 30.0, words_out: list | None = None):
    """One sentence -> yields (sample_rate, pcm_chunk) as the TTS
    renders. Voicebox when configured, else ElevenLabs when configured,
    else Kokoro, and Kokoro as the fallback on ANY failure upstream of it.
    Degrade, never mute.

    `words_out`, if given, is filled with per-word {word, start, end}
    timing for captions: ElevenLabs' real alignment, Whisper alignment of
    the rendered clip for Voicebox, or a length-proportional estimate for
    Kokoro. A partial real alignment from an engine that failed is
    discarded first, so a caption is never a mix of two timelines."""
    if _voicebox_ready():
        try:
            vb_timeout = float(CFG["voicebox"].get("timeout") or timeout)
            for rate, pcm in _stream_voicebox(text, vb_timeout, words_out):
                yield rate, pcm
            return
        except Exception as e:
            log(f"[mouth] voicebox failed ({str(e)[:60]}), falling back to "
                f"{'elevenlabs' if _elevenlabs_ready() else CFG['voice']}")
            if words_out is not None:
                words_out.clear()
    if _elevenlabs_ready():
        try:
            for pcm in _stream_elevenlabs(text, timeout, words_out):
                yield EL_RATE, pcm
            return
        except Exception as e:
            log(f"[mouth] elevenlabs failed ({str(e)[:60]}) — "
                f"falling back to {CFG['voice']}")
            if words_out is not None:
                words_out.clear()
    total_samples = 0
    for pcm in _stream_kokoro(text):
        total_samples += len(pcm)
        yield KOKORO_RATE, pcm
    if words_out is not None and not words_out:
        words_out.extend(_estimate_words(text, total_samples / KOKORO_RATE))


_PREFETCH_FAILED = object()


class Mouth:
    def __init__(self):
        from backtalk.ducking import Ducker
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._speaking = threading.Event()
        # Two-ahead prefetch pipeline (module docstring, mitigation #2):
        # every queued item gets a unique, increasing seq so a prefetch
        # result is matched to the exact item it was rendered for, never
        # an earlier or later one with the same text.
        self._seq = itertools.count()
        self._prefetch_lock = threading.Lock()
        self._prefetch: dict[int, dict] = {}   # seq -> {"thread", "result"}
        # The one persistent output stream (audio law #1).
        # Worker-thread-only — never touch from other threads.
        self._out: sd.OutputStream | None = None
        self._out_rate: int | None = None
        self.ducker = Ducker()  # public: PTT ducks for the USER's voice too
        # In-process twin of signals.caption(): (sentence, play_ts, words)
        # for the chunk audible right now, or None. Read directly by the
        # console caption renderer (caption_tty) with no file round-trip.
        self.current_caption: tuple | None = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    def say(self, text: str):
        """Queue text (split to sentences) for speech."""
        for s in split_sentences(text):
            self._q.put((next(self._seq), s, None))

    def say_chunk(self, text: str, directions=None):
        """Queue text as ONE TTS request, no sentence splitting — fuller
        chunks get livelier prosody (single short sentences come out
        dull).

        On a cold start (nothing playing, nothing queued) a short lead-in
        clause is peeled off first (see _split_lead_in) so the first sound
        arrives at the engine's latency floor instead of after the whole
        chunk renders; the remainder gets its own slot and is prefetched
        while the lead-in plays.

        `directions` are the stage directions this chunk carried. They are
        published on the signal bus when this chunk's audio STARTS, which
        is why they travel with it instead of firing at parse time."""
        text = text.strip()
        if not text:
            return
        if not self._speaking.is_set() and self._q.empty():
            lead, rest = _split_lead_in(text)
            if rest:
                self._q.put((next(self._seq), lead, directions or None))
                self._q.put((next(self._seq), rest, None))
                return
        self._q.put((next(self._seq), text, directions or None))

    def shut_up(self):
        """Barge-in: stop current playback and flush everything queued."""
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        # Orphan in-flight prefetches rather than chase them: they finish
        # harmlessly into slots nobody reads anymore.
        with self._prefetch_lock:
            self._prefetch.clear()

    def shutdown(self):
        """Exit path: stop playback and restore the music SYNCHRONOUSLY
        (the debounced restore timer dies with the process otherwise)."""
        self.shut_up()
        self.ducker.restore_now()

    def wait_done(self, timeout: float | None = None):
        """Block until the queue is drained and playback finished."""
        deadline = None if timeout is None else time.time() + timeout
        while (not self._q.empty()) or self._speaking.is_set():
            time.sleep(0.05)
            if deadline and time.time() > deadline:
                return

    def _run(self):
        from backtalk import signals
        while True:
            seq, sentence, directions = self._q.get()
            if not sentence:
                continue
            self._stop.clear()
            self._speaking.set()
            self.ducker.speech_start()
            signals.static_stop()     # thinking sound dies when speech starts
            signals.set_state("speaking")
            try:
                self._play_stream(sentence, directions, seq=seq)
            except Exception as e:
                log(f"[mouth] synth/play error: {e}")
            finally:
                if self._q.empty():
                    self._speaking.clear()
                    # The reply has genuinely stopped talking, as opposed to
                    # the gap between two sentences of the same reply.
                    signals.reply_done()
                    self.ducker.speech_end()
                    signals.set_state("idle")
                    self.current_caption = None   # nothing left to highlight

    def _get_out(self, rate: int) -> sd.OutputStream:
        """The long-lived stream (audio law #1). Reopened only when the
        sample rate changes (ElevenLabs 44.1k <-> Kokoro 24k fallback:
        rare, costs at most one blip on the switch)."""
        if self._out is not None and self._out_rate == rate:
            # Guarded, because the stream can die UNDER us: the ears
            # rebuild the whole audio system to recover from a device
            # change (see ears._reopen_after_device_change), and that
            # closes every open stream including this one. Touching a
            # dead stream raises rather than returning False, so the
            # check has to be the try, not an `if`. Falling through
            # rebuilds it, which is what the rest of this method does.
            try:
                if not self._out.active:
                    self._out.start()
                return self._out
            except Exception:
                log("[mouth] the output stream went away, reopening")
        self._drop_out()
        self._out = sd.OutputStream(samplerate=rate, channels=1, dtype="int16")
        self._out_rate = rate
        self._out.start()
        return self._out

    def _cut(self):
        """Barge-in cut: stop feeding audio and pad the line with a beat
        of silence — the stream itself NEVER stops (an abort+restart here
        re-triggers the onset blip on latch-happy audio setups). Cost:
        the device buffer (~0.1s) plays out after the kill order — half a
        syllable of tail."""
        try:
            zeros = np.zeros(2205, dtype=np.int16)
            for _ in range(3):
                self._out.write(zeros)
        except Exception:
            self._drop_out()

    def _drop_out(self):
        """Close and forget the stream — the next sentence reopens
        fresh. The self-heal path for device errors (interface
        unplugged, audio mixer restarted)."""
        if self._out is not None:
            try:
                self._out.close(ignore_errors=True)
            except Exception:
                pass
        self._out = None
        self._out_rate = None

    def _maybe_prefetch_ahead(self, depth: int = 2):
        """Peek up to `depth` queued items (without dequeuing them) and
        start a background render for any that don't already have one.
        Called from _play_stream only AFTER this item's own synth is done,
        so the lookahead overlaps this item's playback, not its render.

        Depth 2, not 1: a short line's playback often isn't long enough to
        hide a slow line's render behind it; a second slot gives a slow
        render an extra line's worth of head start."""
        with self._q.mutex:
            upcoming = list(self._q.queue)[:depth]
        with self._prefetch_lock:
            for seq, sentence, _ in upcoming:
                if seq in self._prefetch:
                    continue
                slot = {"thread": None, "result": None}
                self._prefetch[seq] = slot
                t = threading.Thread(target=self._prefetch_worker,
                                     args=(sentence, slot), daemon=True)
                slot["thread"] = t
                t.start()

    def _prefetch_worker(self, sentence, slot):
        try:
            words: list = []
            chunks = list(synth_stream(sentence, words_out=words))
            slot["result"] = (chunks, words)
        except Exception as e:
            log(f"[mouth] background prefetch failed ({str(e)[:60]}), "
                f"will synthesize inline when this chunk's turn comes")
            slot["result"] = _PREFETCH_FAILED

    def _play_stream(self, sentence: str, directions=None, seq: int | None = None,
                     block: int = 2205, prebuffer_s: float = 0.75):
        """Stream-synthesize and play with the head-start buffer (audio
        law #2). stop() reacts ~50ms. The sample rate comes from
        whichever engine actually answered.

        If a background prefetch already rendered this exact item (`seq`),
        its result is used, waiting for it to land if it hasn't quite
        finished, which is never slower than not prefetching at all."""
        from backtalk import signals
        words: list = []
        prefetched = None
        if seq is not None:
            with self._prefetch_lock:
                slot = self._prefetch.pop(seq, None)
            if slot is not None:
                if slot["thread"] is not None:
                    slot["thread"].join()
                if slot["result"] is not None and slot["result"] is not _PREFETCH_FAILED:
                    prefetched = slot["result"]
        if prefetched is not None:
            chunks, words = prefetched
            gen = iter(chunks)
        else:
            gen = synth_stream(sentence, words_out=words)
        head: list = []
        banked = 0
        rate = None
        for rate_, pcm in gen:
            rate = rate_
            head.append(pcm)
            banked += len(pcm)
            if banked >= int(rate * prebuffer_s):
                break
        if rate is None:
            return
        try:
            out = self._get_out(rate)
            # This item's own synth is done (Voicebox delivers the whole
            # chunk in the head-buffer loop above) and playback is about to
            # start, so NOW is when to start rendering the next two items:
            # they overlap the write loop below, not this item's synth.
            self._maybe_prefetch_ahead()
            # AUDIO STARTS HERE: the head buffer is full and the first write
            # is next. Publishing now is what puts a screen cue on the spoken
            # word rather than seconds ahead of it.
            play_ts = time.time()
            if directions:
                from backtalk import signals as _sig
                _sig.direction(directions)
            if words:
                # play_ts is what a watcher's elapsed-time math is keyed to,
                # so republishing below never makes the highlight jump.
                signals.caption(sentence, play_ts, list(words))
                self.current_caption = (sentence, play_ts, list(words))
            published = len(words)

            def _republish_if_grown():
                # ElevenLabs keeps extending `words` from a background
                # thread as alignment bursts land; without this a long
                # chunk's caption froze on whatever partial list the first
                # publish caught.
                nonlocal published
                if words and len(words) != published:
                    published = len(words)
                    signals.caption(sentence, play_ts, list(words))
                    self.current_caption = (sentence, play_ts, list(words))

            def _write(pcm):
                for i in range(0, len(pcm), block):
                    if self._stop.is_set():
                        return False
                    out.write(pcm[i:i + block])
                    # Re-check after the blocking write: a barge-in
                    # landing mid-block must not let feed_waveform
                    # re-assert "speaking" over a fresh "listening".
                    if self._stop.is_set():
                        return False
                    signals.feed_waveform(pcm[i:i + block])
                return True
            for pcm in head:
                if not _write(pcm):
                    self._cut()
                    return
                _republish_if_grown()
            for _, pcm in gen:
                if not _write(pcm):
                    self._cut()
                    return
                _republish_if_grown()
            # gen is fully drained, so words is complete: one final publish
            # in case the tail grew after the last republish.
            _republish_if_grown()
            if not self._stop.is_set():
                # A deliberate breath between this chunk and whatever plays
                # next -- without it, back-to-back chunks run sample into
                # sample with zero gap, which reads as rushed/run-on.
                out.write(np.zeros(int(rate * 0.18), dtype=np.int16))
        except Exception:
            self._drop_out()
            raise


if __name__ == "__main__":
    m = Mouth()
    m.say(sys.argv[1] if len(sys.argv) > 1 else
          "Voice check. The mouth is alive, and it is very good to be heard.")
    m.wait_done(timeout=60)
