# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Ethan
# Added to backtalk (Copyright (C) 2026 Jared Rhodenizer) under the same license.
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
"""Karaoke word-highlight for THIS console -- the transcript a user
actually reads while a voice session runs (e.g. on a second screen),
as opposed to the ai-visualizer browser face, which has its own
independent copy of the same feature (core.js's caption bar). Both
read the same per-word timing from the signal bus (see
signals.caption); they just render it differently.

Purely additive. main.py already prints the plain reply line the
moment a chunk is queued for speech (via log()) -- that line is never
touched, so the full transcript is always there even if this feature
is off, broken, or captions never arrive for a given chunk. This
module only ever adds a SECOND, EPHEMERAL line directly beneath it:
the live line lights the word audible right now as the chunk plays,
and is erased (not committed to scrollback) the moment that chunk
finishes -- so nothing ever ends up duplicated in the permanent
transcript, only a live tracker while each chunk is actually speaking.

Guards against other console writers via vlog.generation: a counter
bumped on every log() call. If it changes between two of this task's
frames, our cursor bookkeeping from the previous frame is stale (some
other line landed in between), so we drop it and re-anchor at the
current cursor position instead of doing cursor-relative math against
a position that no longer exists. Cosmetic-only even when it does
fire -- the permanent transcript (both on screen and in
logs/backtalk.log) is unaffected either way.
"""
import asyncio
import shutil
import sys
import time

from backtalk import vlog

_DIM = "\x1b[2m"
_HI = "\x1b[1;96m"
_RESET = "\x1b[0m"
_CLEAR_LINE = "\x1b[2K"


def _wrap(tokens: list[str], width: int) -> list[list[int]]:
    """Greedy word-wrap, same rule a terminal uses for plain text, so we
    know exactly how many rows our own last render occupied and can
    move the cursor back up that far before redrawing or erasing."""
    rows: list[list[int]] = []
    cur: list[int] = []
    cur_len = 0
    for i, tok in enumerate(tokens):
        piece = len(tok) + (1 if cur else 0)
        if cur and cur_len + piece > width:
            rows.append(cur)
            cur, cur_len = [i], len(tok)
        else:
            cur.append(i)
            cur_len += piece
    if cur:
        rows.append(cur)
    return rows or [[]]


def _render_rows(words: list[dict], active: int, rows: list[list[int]]) -> list[str]:
    lines = []
    for row in rows:
        parts = []
        for i in row:
            w = words[i]["word"]
            parts.append(f"{_HI}{w}{_RESET}" if i == active else f"{_DIM}{w}{_RESET}")
        lines.append(" ".join(parts))
    return lines


def _redraw(lines: list[str], prev_rows: int):
    buf = []
    if prev_rows:
        if prev_rows > 1:
            buf.append(f"\x1b[{prev_rows - 1}A")
        buf.append("\r")
    for idx, line in enumerate(lines):
        buf.append(_CLEAR_LINE)
        buf.append(line)
        if idx < len(lines) - 1:
            buf.append("\r\n")
    # If the previous frame had more rows than this one, its leftover
    # tail rows were never touched above -- wipe them too, then return
    # the cursor to right after the new content (where callers expect it).
    extra = prev_rows - len(lines)
    if extra > 0:
        for _ in range(extra):
            buf.append("\r\n")
            buf.append(_CLEAR_LINE)
        buf.append(f"\x1b[{extra}A")
        buf.append("\r")
    sys.stdout.write("".join(buf))
    sys.stdout.flush()


def _erase(prev_rows: int):
    """Wipe the last render and leave the cursor exactly where it
    started, as if this line was never printed."""
    if prev_rows <= 0:
        return
    buf = []
    if prev_rows > 1:
        buf.append(f"\x1b[{prev_rows - 1}A")
    buf.append("\r")
    for idx in range(prev_rows):
        buf.append(_CLEAR_LINE)
        if idx < prev_rows - 1:
            buf.append("\r\n")
    if prev_rows > 1:
        buf.append(f"\x1b[{prev_rows - 1}A")
    buf.append("\r")
    sys.stdout.write("".join(buf))
    sys.stdout.flush()


# Row count of whatever this module currently has drawn on screen, if
# anything. Module-level (not local to run()) so _erase_now() -- called
# synchronously from vlog.log(), possibly between two of run()'s own
# frames -- can see and clear it before a plain log line prints.
_prev_rows = 0


def _erase_now():
    """Synchronous pre-write hook registered with vlog: wipe our
    ephemeral line right now, if one is up. Runs on vlog's caller (the
    same asyncio thread, never preempted mid-call), so there's no race
    with run()'s own reads of _prev_rows between awaits."""
    global _prev_rows
    if _prev_rows:
        _erase(_prev_rows)
        _prev_rows = 0


async def run(mouth, stop_event: asyncio.Event, hz: float = 25.0):
    """Background task, one per voice session: poll mouth.current_caption
    and animate it in place beneath the static reply line. Cancelled from
    amain()'s shutdown path alongside the other per-session tasks."""
    global _prev_rows
    interval = 1.0 / hz
    _prev_rows = 0
    last_gen = vlog.generation
    vlog.set_pre_write_hook(_erase_now)
    try:
        while not stop_event.is_set():
            await asyncio.sleep(interval)
            try:
                if vlog.generation != last_gen:
                    # Something else wrote to the console since our last
                    # frame (e.g. a burst of short reply chunks) -- our
                    # cursor bookkeeping no longer matches reality. Drop
                    # it and re-anchor at wherever the cursor sits now,
                    # rather than moving it based on a position that's
                    # gone stale. (Normally _erase_now() already cleared
                    # this via the pre-write hook; this is the fallback
                    # for console writes that don't go through
                    # vlog.log().)
                    last_gen = vlog.generation
                    _prev_rows = 0
                cap = mouth.current_caption
                if not cap or not mouth.speaking:
                    if _prev_rows:
                        _erase(_prev_rows)
                        _prev_rows = 0
                    continue
                sentence, ts, words = cap
                if not words:
                    continue
                elapsed = time.time() - ts
                if elapsed >= words[-1]["end"] + 0.15:
                    # this chunk's audio has finished -- clear and wait
                    # for the next one rather than leave a stale
                    # highlight up
                    if _prev_rows:
                        _erase(_prev_rows)
                        _prev_rows = 0
                    continue
                width = max(20, shutil.get_terminal_size((100, 20)).columns - 1)
                tokens = [w["word"] for w in words]
                rows = _wrap(tokens, width)
                active = -1
                for i, w in enumerate(words):
                    if elapsed >= w["start"]:
                        active = i
                    else:
                        break
                lines = _render_rows(words, active, rows)
                _redraw(lines, _prev_rows)
                _prev_rows = len(lines)
            except Exception as e:
                # A single bad frame must never take the whole feature
                # down for the rest of the session -- this task runs
                # once and is never restarted, so an uncaught exception
                # here previously meant captions went dark for good the
                # moment one frame hit a case the rendering math didn't
                # expect. Log it (this is the one place that evidence
                # can land -- an unhandled exception in a fire-and-forget
                # asyncio task prints to the console, not to
                # backtalk.log, so it was invisible after the fact) and
                # drop the current on-screen state so the next good
                # frame starts clean instead of building on a corrupted
                # cursor position.
                vlog.log(f"[caption] frame error: {e!r}")
                _prev_rows = 0
    finally:
        if _prev_rows:
            _erase(_prev_rows)
            _prev_rows = 0
        vlog.set_pre_write_hook(None)
