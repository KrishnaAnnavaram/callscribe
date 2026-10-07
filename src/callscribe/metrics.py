"""Transcript and diarization metrics: WER, CER, cpWER and DER.

- WER counts substitutions, deletions and insertions after text normalisation. Word order and
  repeats count. Set overlap and TF-IDF cosine do not measure a transcript, so they are not here.
- cpWER is the WER of the speaker streams after the best mapping of hypothesis speakers to
  reference speakers.
- DER is (missed speech + false alarm + speaker confusion) / reference speech, on 10 ms frames,
  with a forgiveness collar around each reference boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Turn, Word

FILLERS = frozenset({"uh", "um", "hmm", "mm", "mhm", "ah", "er", "erm"})
_PUNCT = re.compile(r"[^\w\s']")


def normalize(text: str, drop_fillers: bool = True) -> list[str]:
    """Lower case, ``%`` to ``percent``, ``&`` to ``and``, no punctuation, optional filler removal."""
    t = text.lower().replace("%", " percent ").replace("&", " and ")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # 1,000 -> 1000
    t = _PUNCT.sub(" ", t)
    toks = [w.strip("'") for w in t.split()]
    toks = [w for w in toks if w]
    if drop_fillers:
        toks = [w for w in toks if w not in FILLERS]
    return toks


def _rows(ref: list, hyp: list, keep: bool):
    """Levenshtein DP. Each row is computed with vectorised NumPy. Return the last row (and all rows)."""
    vocab = {tok: n for n, tok in enumerate(set(ref) | set(hyp))}
    r = np.array([vocab[t] for t in ref], dtype=np.int64)
    h = np.array([vocab[t] for t in hyp], dtype=np.int64)
    m = len(h)
    j = np.arange(m + 1)
    prev = j.copy()
    table = np.empty((len(r) + 1, m + 1), dtype=np.int32) if keep else None
    if keep:
        table[0] = prev
    for i in range(1, len(r) + 1):
        tmp = np.empty(m + 1, dtype=np.int64)
        tmp[0] = i
        tmp[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (h != r[i - 1]))
        # Insertions: cur[j] = min over k <= j of (tmp[k] + j - k).
        prev = np.minimum.accumulate(tmp - j) + j
        if keep:
            table[i] = prev
    return prev, table


def edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein distance with O(len(hyp)) memory."""
    if not ref or not hyp:
        return max(len(ref), len(hyp))
    return int(_rows(ref, hyp, keep=False)[0][-1])


@dataclass(frozen=True)
class ErrorCounts:
    errors: int
    ref_len: int
    substitutions: int | None = None
    deletions: int | None = None
    insertions: int | None = None

    @property
    def rate(self) -> float:
        return self.errors / self.ref_len if self.ref_len else float(self.errors > 0)


def error_counts(ref: list, hyp: list, max_cells: int = 20_000_000) -> ErrorCounts:
    """Total edits, plus S, D and I from a backtrace when ``len(ref) * len(hyp) <= max_cells``."""
    n, m = len(ref), len(hyp)
    if n == 0 or m == 0:
        return ErrorCounts(max(n, m), n, 0, n, m)
    if n * m > max_cells:
        return ErrorCounts(edit_distance(ref, hyp), n)
    _, d = _rows(ref, hyp, keep=True)
    s = de = ins = 0
    i, jj = n, m
    while i > 0 or jj > 0:
        if i > 0 and jj > 0 and d[i, jj] == d[i - 1, jj - 1] + (ref[i - 1] != hyp[jj - 1]):
            s += ref[i - 1] != hyp[jj - 1]
            i, jj = i - 1, jj - 1
        elif i > 0 and d[i, jj] == d[i - 1, jj] + 1:
            de += 1
            i -= 1
        else:
            ins += 1
            jj -= 1
    return ErrorCounts(int(d[n, m]), n, int(s), de, ins)


def wer(ref_text: str, hyp_text: str) -> float:
    return error_counts(normalize(ref_text), normalize(hyp_text)).rate


def cer(ref_text: str, hyp_text: str, max_chars: int = 20_000) -> float:
    """Character error rate on normalised text. NaN if a text is longer than ``max_chars``."""
    r, h = " ".join(normalize(ref_text)), " ".join(normalize(hyp_text))
    if max(len(r), len(h)) > max_chars:
        return float("nan")
    return edit_distance(list(r), list(h)) / max(1, len(r))


def speaker_streams(words: list[Word]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for w in sorted(words, key=lambda w: w.start):
        out.setdefault(w.speaker or "UNKNOWN", []).extend(normalize(w.text))
    return out


@dataclass(frozen=True)
class CpWerResult:
    errors: int
    ref_len: int
    mapping: dict[str, str]

    @property
    def rate(self) -> float:
        return self.errors / self.ref_len if self.ref_len else float("nan")


def cpwer(ref_words: list[Word], hyp_words: list[Word]) -> CpWerResult:
    """Concatenated minimum-permutation WER. Unmatched streams count as deletions or insertions."""
    ref, hyp = speaker_streams(ref_words), speaker_streams(hyp_words)
    rk, hk = sorted(ref), sorted(hyp)
    n = max(len(rk), len(hk))
    cost = np.zeros((n, n), dtype=np.int64)
    for a in range(n):
        for b in range(n):
            r = ref[rk[a]] if a < len(rk) else []
            h = hyp[hk[b]] if b < len(hk) else []
            cost[a, b] = edit_distance(r, h)
    rows, cols = linear_sum_assignment(cost)
    mapping = {hk[c]: rk[r] for r, c in zip(rows, cols) if r < len(rk) and c < len(hk)}
    return CpWerResult(int(cost[rows, cols].sum()), sum(len(v) for v in ref.values()), mapping)


@dataclass(frozen=True)
class DerResult:
    missed: float
    false_alarm: float
    confusion: float
    total: float
    mapping: dict[str, str]

    @property
    def rate(self) -> float:
        return (self.missed + self.false_alarm + self.confusion) / self.total if self.total else float("nan")


def _activity(turns: list[Turn], names: list[str], n_frames: int, res: float) -> np.ndarray:
    act = np.zeros((len(names), n_frames), dtype=bool)
    pos = {s: k for k, s in enumerate(names)}
    for t in turns:
        act[pos[t.speaker], int(round(t.start / res)):int(round(t.end / res))] = True
    return act


def der(ref: list[Turn], hyp: list[Turn], collar: float = 0.25, skip_overlap: bool = False,
        resolution: float = 0.01) -> DerResult:
    """Diarization error rate. Times are in seconds. Results are in seconds."""
    end = max([t.end for t in ref + hyp], default=0.0)
    n = int(np.ceil(end / resolution)) + 1
    rn, hn = sorted({t.speaker for t in ref}), sorted({t.speaker for t in hyp})
    r_act, h_act = _activity(ref, rn, n, resolution), _activity(hyp, hn, n, resolution)
    keep = np.ones(n, dtype=bool)
    if collar > 0:
        c = int(round(collar / resolution))
        for t in ref:
            for edge in (t.start, t.end):
                f = int(round(edge / resolution))
                keep[max(0, f - c):f + c] = False
    n_ref = r_act.sum(axis=0)
    if skip_overlap:
        keep &= n_ref <= 1
    r_act, h_act, n_ref = r_act[:, keep], h_act[:, keep], n_ref[keep]
    n_hyp = h_act.sum(axis=0)
    mapping, correct = {}, 0
    if rn and hn:
        co = r_act.astype(np.int64) @ h_act.T.astype(np.int64)
        rows, cols = linear_sum_assignment(-co)
        mapping = {hn[c]: rn[r] for r, c in zip(rows, cols) if co[r, c] > 0}
        correct = int(co[rows, cols].sum())
    missed = np.maximum(0, n_ref - n_hyp).sum()
    fa = np.maximum(0, n_hyp - n_ref).sum()
    conf = np.minimum(n_ref, n_hyp).sum() - correct
    return DerResult(missed * resolution, fa * resolution, conf * resolution, n_ref.sum() * resolution, mapping)
