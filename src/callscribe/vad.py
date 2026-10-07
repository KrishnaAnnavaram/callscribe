"""Energy voice activity detection (VAD). It needs no model and no download.

A frame is speech if its energy is above an adaptive threshold. Short gaps are closed and short
speech islands are removed. The real ASR backend uses its own Silero VAD in addition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import frame_rms

HOP = 0.010


@dataclass(frozen=True)
class VadParams:
    threshold_db: float = 18.0  # above the noise floor (10th percentile of frame energy)
    min_speech: float = 0.10
    min_silence: float = 0.25


def speech_mask(x: np.ndarray, sr: int, p: VadParams = VadParams()) -> np.ndarray:
    """Boolean speech flag for each 10 ms frame."""
    rms = frame_rms(x, sr, hop=HOP)
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    db = 20 * np.log10(rms + 1e-10)
    floor = np.percentile(db, 10)
    mask = db > max(floor + p.threshold_db, db.max() - 60)
    mask = _close_gaps(mask, int(p.min_silence / HOP))
    return _drop_short(mask, int(p.min_speech / HOP))


def regions(mask: np.ndarray, hop: float = HOP) -> list[tuple[float, float]]:
    """Speech regions (start, end) in seconds from a frame mask."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return [(s * hop, e * hop) for s, e in zip(edges[::2], edges[1::2])]


def detect(x: np.ndarray, sr: int, p: VadParams = VadParams()) -> list[tuple[float, float]]:
    return regions(speech_mask(x, sr, p))


def _runs(mask: np.ndarray, value: bool):
    padded = np.concatenate([[not value], mask == value, [not value]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return zip(edges[::2], edges[1::2])


def _close_gaps(mask: np.ndarray, n: int) -> np.ndarray:
    out = mask.copy()
    for s, e in _runs(mask, False):
        if s > 0 and e < len(mask) and e - s < n:
            out[s:e] = True
    return out


def _drop_short(mask: np.ndarray, n: int) -> np.ndarray:
    out = mask.copy()
    for s, e in _runs(mask, True):
        if e - s < n:
            out[s:e] = False
    return out
