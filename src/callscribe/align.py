"""Word-to-speaker alignment and utterance grouping."""

from __future__ import annotations

import bisect

from .types import UNKNOWN, Turn, Utterance, Word


def assign_speakers(words: list[Word], turns: list[Turn], max_gap: float = 0.5) -> list[Word]:
    """Give each word the speaker whose turns overlap the word most.

    A word with no overlap gets the speaker of the nearest turn within ``max_gap`` seconds.
    Else it gets ``UNKNOWN``. For equal overlap, the earlier turn wins.
    """
    turns = sorted(turns, key=lambda t: t.start)
    starts = [t.start for t in turns]
    out = []
    for w in words:
        hi = bisect.bisect_right(starts, w.end)
        overlap: dict[str, float] = {}
        best_gap, nearest = max_gap, None
        for t in turns[max(0, hi - 50):hi + 1]:
            ov = min(w.end, t.end) - max(w.start, t.start)
            if ov > 0:
                overlap[t.speaker] = overlap.get(t.speaker, 0.0) + ov
            else:
                gap = max(t.start - w.end, w.start - t.end)
                if gap <= best_gap:
                    best_gap, nearest = gap, t.speaker
        if overlap:
            speaker = max(overlap.items(), key=lambda kv: kv[1])[0]
        else:
            speaker = nearest or UNKNOWN
        out.append(w.with_speaker(speaker))
    return out


def utterances(words: list[Word], max_pause: float = 1.0) -> list[Utterance]:
    """Group words into utterances. A new utterance starts at a speaker change or a long pause."""
    out: list[Utterance] = []
    cur: list[Word] = []
    for w in words:
        if cur and (w.speaker != cur[-1].speaker or w.start - cur[-1].end > max_pause):
            out.append(_close(cur))
            cur = []
        cur.append(w)
    if cur:
        out.append(_close(cur))
    return out


def _close(ws: list[Word]) -> Utterance:
    return Utterance(ws[0].start, ws[-1].end, ws[0].speaker or UNKNOWN, " ".join(w.text for w in ws))
