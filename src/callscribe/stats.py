"""Speaker statistics and a text timeline for a transcript."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Transcript


@dataclass(frozen=True)
class SpeakerStats:
    speaker: str
    talk_time_s: float
    share: float
    turns: int
    words: int
    words_per_minute: float
    longest_turn_s: float


def speaker_stats(t: Transcript) -> list[SpeakerStats]:
    total = sum(x.duration for x in t.turns) or 1.0
    words: dict[str, int] = {}
    for w in t.words:
        words[w.speaker or "UNKNOWN"] = words.get(w.speaker or "UNKNOWN", 0) + 1
    out = []
    for spk in sorted({x.speaker for x in t.turns}):
        mine = [x for x in t.turns if x.speaker == spk]
        talk = sum(x.duration for x in mine)
        n_words = words.get(spk, 0)
        out.append(SpeakerStats(spk, round(talk, 2), round(talk / total, 4), len(mine), n_words,
                                round(60 * n_words / talk, 1) if talk else 0.0,
                                round(max(x.duration for x in mine), 2)))
    return sorted(out, key=lambda s: -s.talk_time_s)


def timeline(t: Transcript, width: int = 60) -> str:
    """One row for each speaker. ``#`` marks a time cell in which the speaker talks."""
    end = max((x.end for x in t.turns), default=0.0)
    if end <= 0:
        return "(no speech)"
    rows = []
    for spk in sorted({x.speaker for x in t.turns}):
        cells = [" "] * width
        for x in t.turns:
            if x.speaker != spk:
                continue
            a, b = int(x.start / end * width), max(int(x.start / end * width) + 1, int(x.end / end * width))
            for c in range(a, min(b, width)):
                cells[c] = "#"
        rows.append(f"{spk:>12} |{''.join(cells)}|")
    rows.append(f"{'':>12}  0s{'':>{width - 8}}{end:6.0f}s")
    return "\n".join(rows)
