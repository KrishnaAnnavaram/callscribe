"""Reference files: Earnings-21 ``.nlp`` tokens, RTTM turns, words JSON and the eval manifest.

An ``.nlp`` file is pipe separated with the header
``token|speaker|ts|endTs|punctuation|case|tags|wer_tags``. ``ts`` and ``endTs`` are often empty.
Thus an ``.nlp`` file gives the reference words and speakers (for WER and cpWER), but DER needs
an RTTM file with turn times.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .types import Turn, Word

NLP_HEADER = "token|speaker|ts|endTs|punctuation|case|tags|wer_tags"
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a")


class ReferenceError(ValueError):
    pass


def read_nlp(path: str | Path) -> list[Word]:
    """Reference words and speakers. A missing time becomes 0.0 (Earnings-21 often has no times)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].lower().startswith("token|speaker"):
        raise ReferenceError(f"{path}: expected the header {NLP_HEADER!r}")
    words = []
    for n, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            raise ReferenceError(f"{path}:{n}: expected at least token|speaker")
        token, speaker = parts[0].strip(), parts[1].strip()
        start = _float(parts[2]) if len(parts) > 2 else None
        end = _float(parts[3]) if len(parts) > 3 else None
        if token:
            words.append(Word(start if start is not None else 0.0,
                              end if end is not None else (start or 0.0), token, speaker or None))
    return words


def _float(s: str) -> float | None:
    s = s.strip()
    return float(s) if s else None


def write_nlp(words: list[Word], path: str | Path) -> Path:
    rows = [NLP_HEADER] + [f"{w.text}|{w.speaker or ''}|{w.start:.3f}|{w.end:.3f}|||[]|[]" for w in words]
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return Path(path)


def read_rttm(path: str | Path) -> list[Turn]:
    turns = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts or parts[0] != "SPEAKER":
            continue
        if len(parts) < 8:
            raise ReferenceError(f"{path}:{n}: an RTTM SPEAKER line has at least 8 fields")
        start, dur = float(parts[3]), float(parts[4])
        turns.append(Turn(start, start + dur, parts[7]))
    return turns


def rttm_lines(turns: list[Turn], file_id: str) -> list[str]:
    return [f"SPEAKER {file_id} 1 {t.start:.3f} {t.duration:.3f} <NA> <NA> {t.speaker} <NA> <NA>"
            for t in sorted(turns, key=lambda t: t.start)]


def write_rttm(turns: list[Turn], file_id: str, path: str | Path) -> Path:
    Path(path).write_text("\n".join(rttm_lines(turns, file_id)) + "\n", encoding="utf-8")
    return Path(path)


@dataclass
class ManifestItem:
    id: str
    audio: str
    ref_words: str  # .nlp file
    ref_rttm: str | None = None
    split: str = "test"


def write_manifest(items: list[ManifestItem], path: str | Path) -> Path:
    Path(path).write_text(json.dumps([asdict(i) for i in items], indent=2), encoding="utf-8")
    return Path(path)


def read_manifest(path: str | Path, split: str | None = None) -> list[ManifestItem]:
    base = Path(path).parent
    items = []
    for d in json.loads(Path(path).read_text(encoding="utf-8")):
        item = ManifestItem(**d)
        for attr in ("audio", "ref_words", "ref_rttm"):
            value = getattr(item, attr)
            if value and not Path(value).is_absolute():
                setattr(item, attr, str(base / value))
        if split in (None, "all") or item.split == split:
            items.append(item)
    return items


def dev_or_test(file_id: str, dev_fraction: float = 0.25) -> str:
    """A stable split from a hash of the file ID. The split does not change when files are added."""
    h = int(hashlib.sha256(file_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < dev_fraction else "test"


def earnings21_manifest(root: str | Path, dev_fraction: float = 0.25) -> list[ManifestItem]:
    """Find each audio file with a matching ``.nlp`` file (and ``.rttm`` if present) under ``root``."""
    root = Path(root)
    nlp = {p.stem: p for p in root.rglob("*.nlp")}
    rttm = {p.stem: p for p in root.rglob("*.rttm")}
    items = []
    for audio in sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS):
        if audio.stem in nlp:
            items.append(ManifestItem(audio.stem, str(audio), str(nlp[audio.stem]),
                                      str(rttm[audio.stem]) if audio.stem in rttm else None,
                                      dev_or_test(audio.stem, dev_fraction)))
    if not items:
        raise ReferenceError(f"no audio file with a matching .nlp file under {root}")
    return items
