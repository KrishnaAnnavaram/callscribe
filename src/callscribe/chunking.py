"""VAD-aware chunks with overlap, and a merge that keeps each word exactly once.

Each chunk has an OWN region. The own regions cover the timeline with no gap and no overlap.
The audio window of a chunk is the own region plus ``overlap`` seconds on each side. A word is
kept only by the chunk whose own region contains the word midpoint. A cut goes into a silence
when one exists, so a word is not cut in two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Word
from .vad import VadParams, detect


@dataclass(frozen=True)
class Chunk:
    own_start: float
    own_end: float
    start: float  # audio window start
    end: float  # audio window end


def plan_chunks(duration: float, speech: list[tuple[float, float]] | None, max_len: float = 30.0,
                overlap: float = 1.0, min_len: float = 10.0) -> list[Chunk]:
    """Cut the timeline into chunks of at most ``max_len`` seconds of own region."""
    if max_len <= 0 or duration <= 0:
        raise ValueError("max_len and duration must be positive")
    gaps = []
    if speech:
        ends = [e for _, e in speech[:-1]]
        starts = [s for s, _ in speech[1:]]
        gaps = [(a + b) / 2 for a, b in zip(ends, starts) if b > a]
    cuts, t = [], 0.0
    while duration - t > max_len:
        options = [g for g in gaps if t + min(min_len, max_len / 2) <= g <= t + max_len]
        cut = max(options) if options else t + max_len
        cuts.append(cut)
        t = cut
    bounds = [0.0, *cuts, duration]
    return [
        Chunk(a, b, max(0.0, a - overlap), min(duration, b + overlap))
        for a, b in zip(bounds[:-1], bounds[1:])
    ]


def merge_words(per_chunk: list[tuple[Chunk, list[Word]]]) -> list[Word]:
    """Keep each word in the chunk that owns its midpoint. Input word times are absolute."""
    kept = []
    last = len(per_chunk) - 1
    for n, (chunk, words) in enumerate(per_chunk):
        for w in words:
            in_own = chunk.own_start <= w.mid < chunk.own_end or (n == last and w.mid == chunk.own_end)
            if in_own:
                kept.append(w)
    return sorted(kept, key=lambda w: (w.start, w.end))


class ChunkedASR:
    """Run a short-input ASR over VAD-aware, overlapping chunks."""

    def __init__(self, inner, max_len: float = 30.0, overlap: float = 1.0, vad: VadParams = VadParams()):
        self.inner, self.max_len, self.overlap, self.vad = inner, max_len, overlap, vad
        self.name = f"chunked({getattr(inner, 'name', 'asr')})"

    def transcribe(self, x: np.ndarray, sr: int, offset: float = 0.0) -> list[Word]:
        duration = len(x) / sr
        chunks = plan_chunks(duration, detect(x, sr, self.vad), self.max_len, self.overlap)
        results = []
        for c in chunks:
            seg = x[int(c.start * sr):int(c.end * sr)]
            words = self.inner.transcribe(seg, sr, offset=offset + c.start)
            results.append((c, [w.shifted(c.start) for w in words]))
        return merge_words(results)
