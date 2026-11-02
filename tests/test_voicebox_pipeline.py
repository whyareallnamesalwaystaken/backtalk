# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Ethan
# Added to backtalk (Copyright (C) 2026 Jared Rhodenizer) under the same license.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline tests for the Voicebox engine, latency hiding and captions.

No audio device, no GPU, no Voicebox server, no network: every engine and
the output stream are faked. Run from the repo root with either

    uv run python tests/test_voicebox_pipeline.py
    uv run pytest tests/test_voicebox_pipeline.py      (if pytest is installed)
"""
import io
import os
import sys
import threading
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from backtalk import ears, mouth, signals, vlog  # noqa: E402

RATE = 24000


def _wav_bytes(seconds: float, rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(int(rate * seconds), dtype=np.int16).tobytes())
    return buf.getvalue()


class _Patch:
    """Tiny monkeypatch: setattr now, restore everything on exit."""

    def __init__(self):
        self._undo = []

    def set(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)


def _vb_cfg(**over):
    cfg = {"enabled": True, "base_url": "http://127.0.0.1:1", "profile": "test",
           "engine": "chatterbox_turbo", "timeout": 2.0, "repo_dir": ""}
    cfg.update(over)
    return cfg


# --- lead-in split ---------------------------------------------------------

def test_lead_in_splits_long_text_at_a_clause():
    text = "Okay so here is the thing, we need to sort out the queue before lunch today."
    lead, rest = mouth._split_lead_in(text)
    assert lead == "Okay so here is the thing,"
    assert rest == "we need to sort out the queue before lunch today."
    assert len(lead) <= mouth._LEAD_IN_MAX_CHARS
    assert len(rest) >= mouth._LEAD_IN_MIN_REST_CHARS


def test_lead_in_falls_back_to_a_word_boundary():
    text = "This sentence has no punctuation at all until the very end of it right here."
    lead, rest = mouth._split_lead_in(text)
    assert rest is not None and lead + " " + rest == text
    assert len(lead) <= mouth._LEAD_IN_MAX_CHARS


def test_lead_in_leaves_short_text_alone():
    assert mouth._split_lead_in("Short and sweet.") == ("Short and sweet.", None)


# --- caption timing --------------------------------------------------------

def test_estimate_words_ends_exactly_on_the_audio():
    words = mouth._estimate_words("one two three four", 2.0)
    assert [w["word"] for w in words] == ["one", "two", "three", "four"]
    assert abs(words[-1]["end"] - 2.0) < 0.01
    assert all(a["end"] <= b["start"] for a, b in zip(words, words[1:]))


def test_estimate_words_weights_numbers_by_digits():
    words = mouth._estimate_words("call 12345 now", 3.0)
    assert (words[1]["end"] - words[1]["start"]) > 3 * (words[0]["end"] - words[0]["start"])


def test_fill_gaps_anchors_unmatched_edges():
    real = ["Alright", "boss", "here", "we", "go"]
    partial = [None,
               {"word": "boss", "start": 0.6, "end": 0.9},
               {"word": "here", "start": 1.0, "end": 1.2},
               {"word": "we", "start": 1.3, "end": 1.4},
               None]
    out = ears._fill_gaps(partial, real, speech_start=0.1, speech_end=2.0)
    assert out[0]["start"] == 0.1 and out[0]["end"] <= 0.6
    assert out[-1]["start"] >= 1.4 and out[-1]["end"] <= 2.0 + 1e-6
    assert [w["word"] for w in out] == real


def test_align_words_keeps_real_timing_when_first_word_mismatches():
    class _W:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    class _Seg:
        words = [_W("All", 0.10, 0.25), _W("right", 0.25, 0.45), _W("boss,", 0.50, 0.80),
                 _W("here", 0.90, 1.10), _W("we", 1.15, 1.25), _W("go.", 1.30, 1.60)]

    class _Model:
        def transcribe(self, *a, **k):
            return [_Seg()], None

    with _Patch() as p:
        p.set(ears, "_backend", "faster-whisper")
        p.set(ears, "warm", lambda: _Model())
        out = ears.align_words(_wav_bytes(2.0), "Alright boss, here we go.")
    assert out is not None, "fell back to the estimate over one edge mismatch"
    assert [w["word"] for w in out] == ["Alright", "boss,", "here", "we", "go."]
    assert out[1]["start"] == 0.50 and out[-1]["end"] == 1.60
    assert out[0]["start"] == 0.10


# --- Voicebox engine -------------------------------------------------------

def test_voicebox_hung_connection_triggers_self_heal_and_reraises():
    called = []
    with _Patch() as p:
        p.set(mouth, "CFG", {**mouth.CFG, "voicebox": _vb_cfg()})
        p.set(mouth, "_resolve_voicebox_profile", lambda *a: "id")
        p.set(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectTimeout("hang")))
        p.set(mouth, "_try_autostart_voicebox", lambda: called.append(True))
        try:
            list(mouth._stream_voicebox("hi", 2.0))
            raised = False
        except httpx.ConnectTimeout:
            raised = True
    assert raised and called


class _Resp:
    def __init__(self, content, status=200):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


def test_voicebox_empty_audio_is_a_failure_not_silence():
    with _Patch() as p:
        p.set(mouth, "CFG", {**mouth.CFG, "voicebox": _vb_cfg()})
        p.set(mouth, "_resolve_voicebox_profile", lambda *a: "id")
        p.set(httpx, "post", lambda *a, **k: _Resp(_wav_bytes(0.0)))
        try:
            list(mouth._stream_voicebox("hi", 2.0))
            raised = False
        except RuntimeError:
            raised = True
    assert raised


def test_voicebox_words_filled_before_the_audio_is_yielded():
    words = []
    seen_at_yield = []
    with _Patch() as p:
        p.set(mouth, "CFG", {**mouth.CFG, "voicebox": _vb_cfg()})
        p.set(mouth, "_resolve_voicebox_profile", lambda *a: "id")
        p.set(httpx, "post", lambda *a, **k: _Resp(_wav_bytes(1.0)))
        p.set(ears, "align_words", lambda wav, text: None)   # force the estimate
        for rate, pcm in mouth._stream_voicebox("hello there", 2.0, words):
            seen_at_yield.append(len(words))
    assert seen_at_yield == [2] and rate == RATE and len(pcm) == RATE


def test_synth_stream_falls_back_and_never_mixes_timelines():
    def boom(text, timeout, words_out=None):
        if words_out is not None:
            words_out.append({"word": "stale", "start": 0, "end": 1})
        raise RuntimeError("voicebox down")
        yield  # pragma: no cover

    def fake_kokoro(text):
        yield np.zeros(RATE, dtype=np.int16)

    words = []
    with _Patch() as p:
        p.set(mouth, "CFG", {**mouth.CFG, "voicebox": _vb_cfg()})
        p.set(mouth, "_stream_voicebox", boom)
        p.set(mouth, "_elevenlabs_ready", lambda: False)
        p.set(mouth, "_stream_kokoro", fake_kokoro)
        out = list(mouth.synth_stream("hello there", words_out=words))
    assert out and out[0][0] == mouth.KOKORO_RATE
    assert [w["word"] for w in words] == ["hello", "there"]


def test_generate_lock_serializes_concurrent_renders():
    active, peak = [0], [0]
    lock = threading.Lock()

    def slow_post(*a, **k):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.1)
        with lock:
            active[0] -= 1
        return _Resp(_wav_bytes(0.2))

    with _Patch() as p:
        p.set(mouth, "CFG", {**mouth.CFG, "voicebox": _vb_cfg()})
        p.set(mouth, "_resolve_voicebox_profile", lambda *a: "id")
        p.set(httpx, "post", slow_post)
        ts = [threading.Thread(target=lambda: list(mouth._stream_voicebox("x", 2.0)))
              for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert peak[0] == 1


def test_warm_is_a_no_op_when_voicebox_is_disabled():
    with _Patch() as p:
        p.set(mouth, "CFG", {**mouth.CFG, "voicebox": _vb_cfg(enabled=False)})
        p.set(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network touched")))
        t = time.time()
        mouth.ensure_voicebox_warm(deadline=5)
    assert time.time() - t < 0.5


# --- prefetch pipeline + caption publishing --------------------------------

class _FakeStream:
    """Plays in real time so playback duration is honest, no device."""

    def __init__(self, rate):
        self.rate, self.active = rate, True

    def write(self, pcm):
        time.sleep(len(pcm) / self.rate)

    def start(self):
        pass

    def close(self, ignore_errors=True):
        pass


def test_next_chunk_renders_while_the_current_one_plays():
    events = []
    t0 = time.time()

    def fake_synth(text, timeout=30.0, words_out=None):
        events.append(("start", text, time.time() - t0))
        time.sleep(0.4)                                   # render cost
        if words_out is not None:
            words_out.extend(mouth._estimate_words(text, 0.8))
        events.append(("done", text, time.time() - t0))
        yield RATE, np.zeros(int(RATE * 0.8), dtype=np.int16)  # 0.8s of audio

    captions = []
    with _Patch() as p:
        p.set(mouth, "synth_stream", fake_synth)
        p.set(mouth.Mouth, "_get_out", lambda self, rate: _FakeStream(rate))
        p.set(signals, "caption", lambda text, ts, words: captions.append((text, ts, len(words))))
        p.set(signals, "feed_waveform", lambda pcm: None)
        m = mouth.Mouth()
        for s in ("First chunk here.", "Second chunk here.", "Third chunk here."):
            m._q.put((next(m._seq), s, None))
        m._speaking.set()
        m.wait_done(timeout=10)
    starts = {text: t for kind, text, t in events if kind == "start"}
    dones = {text: t for kind, text, t in events if kind == "done"}
    # chunk 2 must start rendering before chunk 1 finishes PLAYING (~1.2s)
    assert starts["Second chunk here."] < dones["First chunk here."] + 0.8
    # every chunk got a caption, keyed to its own start time, in order
    assert [c[0] for c in captions] == ["First chunk here.", "Second chunk here.", "Third chunk here."]
    assert all(c[2] == 3 for c in captions)
    assert captions[0][1] < captions[1][1] < captions[2][1]
    assert m.current_caption is None   # cleared once the reply finished


def test_shut_up_orphans_prefetches():
    m = mouth.Mouth.__new__(mouth.Mouth)
    m._q = __import__("queue").Queue()
    m._stop = threading.Event()
    m._prefetch_lock = threading.Lock()
    m._prefetch = {1: {"thread": None, "result": None}}
    m.shut_up()
    assert m._prefetch == {}


# --- console hooks ---------------------------------------------------------

def test_log_runs_the_pre_write_hook_and_bumps_generation():
    hits = []
    gen = vlog.generation
    vlog.set_pre_write_hook(lambda: hits.append(1))
    try:
        vlog.log("[test] hook check")
    finally:
        vlog.set_pre_write_hook(None)
    assert hits == [1] and vlog.generation == gen + 1


if __name__ == "__main__":
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
