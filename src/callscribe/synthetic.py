"""Synthetic calls: tone-burst "words" from speakers with different voices, plus exact references.

Each speaker has a fundamental frequency and a formant. Each word is a short harmonic burst.
The references (word times, words, speakers, turns) are exact. The audio is not speech, so a
real speech recogniser cannot read it. The scripted ASR simulates the recognition step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE
from .audio import write_wav
from .references import ManifestItem, write_manifest, write_nlp, write_rttm
from .types import Turn, Word

VOCAB = (
    "revenue growth quarter margin guidance customers cloud demand pricing operating cash flow "
    "billion million percent year over increase decline segment strong outlook expect costs "
    "investment product market share earnings question analyst thank you next fiscal").split()

VOICES = ((110, 600), (190, 1500), (150, 2400), (240, 900), (95, 1900), (280, 2800))


@dataclass(frozen=True)
class CallSpec:
    n_speakers: int = 3
    n_turns: int = 14
    min_words: int = 4
    max_words: int = 14
    noise_db: float = -45.0
    seed: int = 0


@dataclass
class SyntheticCall:
    audio: np.ndarray
    words: list[Word]
    turns: list[Turn]
    sr: int = SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sr


def _burst(f0: float, formant: float, dur: float, sr: int, rng) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    f = f0 * (1 + rng.normal(0, 0.02))
    sig = np.zeros_like(t)
    for h in range(1, 16):
        amp = np.exp(-(((h * f) - formant) / 500.0) ** 2) + 0.05 / h
        sig += amp * np.sin(2 * np.pi * h * f * t + rng.uniform(0, 2 * np.pi))
    return sig * np.hanning(len(t)) / 3


def synth_call(spec: CallSpec = CallSpec()) -> SyntheticCall:
    if not 1 <= spec.n_speakers <= len(VOICES):
        raise ValueError(f"n_speakers must be 1..{len(VOICES)}")
    rng = np.random.default_rng(spec.seed)
    sr = SAMPLE_RATE
    voices = [VOICES[k] for k in rng.permutation(len(VOICES))[:spec.n_speakers]]
    pieces, words, turns = [], [], []
    pos = 0  # samples written so far; every time comes from this count, so there is no drift

    def add(x: np.ndarray) -> None:
        nonlocal pos
        pieces.append(x)
        pos += len(x)

    add(np.zeros(int(0.5 * sr)))
    speaker = 0
    for turn in range(spec.n_turns):
        if turn > 0 and spec.n_speakers > 1:
            speaker = (speaker + 1 + rng.integers(0, spec.n_speakers - 1)) % spec.n_speakers
        name = f"spk{speaker + 1}"
        f0, formant = voices[speaker]
        turn_start = pos / sr
        for _ in range(rng.integers(spec.min_words, spec.max_words + 1)):
            start = pos / sr
            add(_burst(f0, formant, float(rng.uniform(0.18, 0.42)), sr, rng))
            words.append(Word(start, pos / sr, str(rng.choice(VOCAB)), name))
            add(np.zeros(int(rng.uniform(0.04, 0.12) * sr)))
        turns.append(Turn(turn_start, words[-1].end, name))
        add(np.zeros(int(rng.uniform(0.4, 0.9) * sr)))
    audio = np.concatenate(pieces)
    audio = audio + rng.normal(0, 10 ** (spec.noise_db / 20), size=len(audio))
    return SyntheticCall(audio.astype(np.float32), words, turns)


def write_dataset(out_dir: str | Path, n_calls: int = 8, seed: int = 0) -> Path:
    """Write WAV, .nlp, .rttm files and ``manifest.json``. Return the manifest path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    items = []
    for k in range(n_calls):
        cid = f"synth{k:03d}"
        spec = CallSpec(n_speakers=int(rng.integers(2, 5)), n_turns=int(rng.integers(10, 18)),
                        seed=int(rng.integers(0, 2**31)))
        call = synth_call(spec)
        write_wav(out / f"{cid}.wav", call.audio)
        write_nlp(call.words, out / f"{cid}.nlp")
        write_rttm(call.turns, cid, out / f"{cid}.rttm")
        items.append(ManifestItem(cid, f"{cid}.wav", f"{cid}.nlp", f"{cid}.rttm", "dev" if k % 4 == 0 else "test"))
    return write_manifest(items, out / "manifest.json")
