"""Speaker diarization adapters behind one interface: ``diarize(x, sr, num_speakers) -> turns``.

- ``SpectralDiarizer``: offline. Log band energies of short windows, cosine distance and
  average-linkage clustering. It separates clearly different voices (for example the synthetic
  calls). It is not a replacement for a neural speaker embedding on real calls.
- ``PyannoteDiarizer``: pyannote.audio 3.x (extra ``diarize``). The hyperparameters come from a
  JSON file that ``callscribe tune`` writes from a dev split. No value is set by hand in code.
- ``ReferenceDiarizer``: returns the reference turns. Use it to measure the ASR alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .types import Turn
from .vad import HOP, VadParams, speech_mask


class Diarizer(Protocol):
    name: str

    def diarize(self, x: np.ndarray, sr: int, num_speakers: int | None = None) -> list[Turn]: ...


def merge_turns(turns: list[Turn], min_off: float = 0.0, min_on: float = 0.0) -> list[Turn]:
    """Join same-speaker turns with a gap below ``min_off``. Then drop turns shorter than ``min_on``."""
    out: list[Turn] = []
    for t in sorted(turns, key=lambda t: (t.start, t.end)):
        if out and out[-1].speaker == t.speaker and t.start - out[-1].end < min_off:
            out[-1] = Turn(out[-1].start, max(out[-1].end, t.end), t.speaker)
        else:
            out.append(t)
    return [t for t in out if t.duration >= min_on]


def relabel(turns: list[Turn]) -> list[Turn]:
    """Name the speakers SPEAKER_00, SPEAKER_01, ... in the order of the first turn."""
    names: dict[str, str] = {}
    for t in sorted(turns, key=lambda t: t.start):
        names.setdefault(t.speaker, f"SPEAKER_{len(names):02d}")
    return [Turn(t.start, t.end, names[t.speaker]) for t in turns]


@dataclass(frozen=True)
class SpectralParams:
    window: float = 1.0
    step: float = 0.5
    threshold: float = 0.15  # cosine distance at which clusters stop merging
    min_duration_off: float = 0.3
    min_duration_on: float = 0.2
    n_bands: int = 32
    min_speech_fraction: float = 0.3

    def with_values(self, **kw) -> "SpectralParams":
        return replace(self, **kw)


def band_energies(x: np.ndarray, sr: int, n_bands: int, frame: float = 0.025) -> np.ndarray:
    """Log energy in ``n_bands`` log-spaced bands (60 Hz to 4 kHz) for each 10 ms frame."""
    n, h = int(frame * sr), int(HOP * sr)
    if len(x) < n:
        return np.zeros((0, n_bands))
    idx = np.arange(0, len(x) - n + 1, h)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[idx] * np.hanning(n)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    edges = np.geomspace(60, min(4000, sr / 2 - 1), n_bands + 1)
    band = np.digitize(freqs, edges) - 1
    out = np.zeros((len(frames), n_bands))
    for b in range(n_bands):
        sel = band == b
        if sel.any():
            out[:, b] = power[:, sel].sum(axis=1)
    return np.log(out + 1e-10)


class SpectralDiarizer:
    name = "spectral"

    def __init__(self, params: SpectralParams = SpectralParams(), vad: VadParams = VadParams()):
        self.params, self.vad = params, vad

    def embeddings(self, x: np.ndarray, sr: int):
        p = self.params
        mask = speech_mask(x, sr, self.vad)
        feats = band_energies(x, sr, p.n_bands)
        n = min(len(mask), len(feats))
        mask, feats = mask[:n], feats[:n]
        win, step = int(p.window / HOP), max(1, int(p.step / HOP))
        centers, embs = [], []
        for s in range(0, max(1, n - win + 1), step):
            m = mask[s:s + win]
            if m.size == 0 or m.mean() < p.min_speech_fraction:
                continue
            e = feats[s:s + win][m].mean(axis=0)
            e = e - e.mean()
            norm = np.linalg.norm(e)
            if norm > 0:
                centers.append(s + win // 2)
                embs.append(e / norm)
        return mask, np.array(centers, dtype=int), np.array(embs)

    def diarize(self, x: np.ndarray, sr: int, num_speakers: int | None = None) -> list[Turn]:
        p = self.params
        mask, centers, embs = self.embeddings(x, sr)
        if len(embs) == 0:
            return []
        if len(embs) == 1:
            labels = np.array([1])
        else:
            tree = linkage(embs, method="average", metric="cosine")
            labels = (fcluster(tree, t=num_speakers, criterion="maxclust") if num_speakers
                      else fcluster(tree, t=p.threshold, criterion="distance"))
        speech_frames = np.flatnonzero(mask)
        nearest = np.abs(speech_frames[:, None] - centers[None, :]).argmin(axis=1)
        frame_label = labels[nearest]
        turns = []
        start = prev = speech_frames[0]
        cur = frame_label[0]
        for f, lab in zip(speech_frames[1:], frame_label[1:]):
            if lab != cur or f != prev + 1:
                turns.append(Turn(start * HOP, (prev + 1) * HOP, str(cur)))
                start, cur = f, lab
            prev = f
        turns.append(Turn(start * HOP, (prev + 1) * HOP, str(cur)))
        return relabel(merge_turns(turns, p.min_duration_off, p.min_duration_on))


class ReferenceDiarizer:
    name = "reference"

    def __init__(self, turns: list[Turn]):
        self.turns = turns

    def diarize(self, x, sr, num_speakers=None) -> list[Turn]:
        return list(self.turns)


class PyannoteDiarizer:  # pragma: no cover - needs the optional extra and a gated model
    """pyannote.audio 3.x pipeline. The token is passed to the hub and never printed."""

    name = "pyannote"

    def __init__(self, pipeline: str, token: str, device: str = "cpu", params: dict | None = None):
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError('pyannote.audio is not installed: pip install -e ".[diarize]"') from exc
        if not token:
            raise RuntimeError("HF_TOKEN is not set. Accept the model terms on Hugging Face first.")
        try:
            self.pipeline = Pipeline.from_pretrained(pipeline, token=token)
        except TypeError:
            self.pipeline = Pipeline.from_pretrained(pipeline, use_auth_token=token)
        if params:
            current = self.pipeline.parameters(instantiated=True)
            self.pipeline.instantiate(merge_nested(current, params))
        self.pipeline.to(torch.device(device))
        self._torch = torch

    def diarize(self, x, sr, num_speakers=None) -> list[Turn]:
        waveform = self._torch.from_numpy(np.asarray(x, dtype=np.float32))[None, :]
        kw = {"num_speakers": num_speakers} if num_speakers else {}
        result = self.pipeline({"waveform": waveform, "sample_rate": sr}, **kw)
        annotation = getattr(result, "speaker_diarization", result)
        turns = [Turn(float(seg.start), float(seg.end), str(spk))
                 for seg, _, spk in annotation.itertracks(yield_label=True)]
        return relabel(turns)


def merge_nested(base: dict, flat: dict) -> dict:
    """Merge ``{"clustering.threshold": 0.7}`` style keys into a nested parameter dict."""
    out = json.loads(json.dumps(base))
    for key, value in flat.items():
        node = out
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return out


def save_params(params: dict, path: str | Path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": params, "meta": meta or {}}, indent=2), encoding="utf-8")
    return path


def load_params(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["params"]


def spectral_from_params(params: dict | None) -> SpectralDiarizer:
    base = SpectralParams()
    if params:
        unknown = set(params) - set(asdict(base))
        if unknown:
            raise ValueError(f"unknown spectral parameters: {sorted(unknown)}")
        base = base.with_values(**params)
    return SpectralDiarizer(base)
