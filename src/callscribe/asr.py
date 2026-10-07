"""Speech recognition adapters behind one small interface.

``transcribe(x, sr, offset)`` returns words with times relative to the start of ``x``. ``offset``
is the absolute start time of ``x`` in the call. Only the scripted ASR uses it.

- ``FasterWhisperASR``: faster-whisper with word timestamps and its Silero VAD (extra ``asr``).
  It handles long audio itself. There is no token cap for each chunk.
- ``ScriptedASR``: an offline simulator for tests and the demo. It returns the reference words
  of the audio window with seeded substitutions, deletions and insertions.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .types import Word

FILLER_VOCAB = ("the", "and", "uh", "so", "we", "that", "of", "in")


class ASR(Protocol):
    name: str

    def transcribe(self, x: np.ndarray, sr: int, offset: float = 0.0) -> list[Word]: ...


def _unit(seed: int, index: int, salt: str) -> float:
    """A deterministic number in [0, 1) for one word. Chunking does not change it."""
    h = hashlib.blake2b(f"{seed}:{index}:{salt}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


class ScriptedASR:
    """Simulated recognition from reference words. It is NOT a speech recogniser."""

    name = "scripted"

    def __init__(self, reference: list[Word], sub_rate: float = 0.05, del_rate: float = 0.03,
                 ins_rate: float = 0.02, seed: int = 0):
        if min(sub_rate, del_rate, ins_rate) < 0 or sub_rate + del_rate > 1:
            raise ValueError("bad error rates")
        self.reference = sorted(reference, key=lambda w: w.start)
        self.rates = (sub_rate, del_rate, ins_rate)
        self.seed = seed
        vocab = sorted({w.text for w in reference}) or ["word"]
        self.vocab = vocab

    def transcribe(self, x: np.ndarray, sr: int, offset: float = 0.0) -> list[Word]:
        end = offset + len(x) / sr
        sub, dele, ins = self.rates
        out = []
        for n, w in enumerate(self.reference):
            if w.start < offset or w.end > end:
                continue  # a word that is not fully in the window is not heard
            r = _unit(self.seed, n, "op")
            text = w.text
            if r < dele:
                continue
            if r < dele + sub:
                text = self.vocab[int(_unit(self.seed, n, "sub") * len(self.vocab))]
                if text == w.text:
                    text = FILLER_VOCAB[n % len(FILLER_VOCAB)]
            out.append(Word(w.start - offset, w.end - offset, text, prob=0.9))
            if _unit(self.seed, n, "ins") < ins:
                gap = min(0.05, (w.end - w.start) / 4)
                out.append(Word(w.end - offset, w.end - offset + gap, FILLER_VOCAB[n % 3], prob=0.3))
        return out


class FasterWhisperASR:
    """faster-whisper backend. Install with ``pip install -e ".[asr]"``."""

    name = "faster-whisper"

    def __init__(self, model: str = "large-v3", device: str = "cpu", compute_type: str = "int8",
                 language: str | None = "en", beam_size: int = 5):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError('faster-whisper is not installed: pip install -e ".[asr]"') from exc
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language, self.beam_size, self.model_name = language, beam_size, model

    def transcribe(self, x: np.ndarray, sr: int, offset: float = 0.0) -> list[Word]:  # pragma: no cover
        if sr != 16_000:
            raise ValueError("faster-whisper needs 16 kHz audio")
        segments, _ = self.model.transcribe(
            x.astype(np.float32), language=self.language, beam_size=self.beam_size,
            word_timestamps=True, vad_filter=True, condition_on_previous_text=False,
        )
        words = []
        for seg in segments:
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    words.append(Word(float(w.start), float(w.end), text, prob=float(w.probability)))
        return words
