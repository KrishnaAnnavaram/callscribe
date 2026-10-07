"""Transcript files: JSON, SRT, TXT and RTTM."""

from __future__ import annotations

import json
from pathlib import Path

from .references import rttm_lines
from .types import Transcript

FORMATS = ("json", "srt", "txt", "rttm")


def clock(seconds: float, sep: str = ".") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_txt(t: Transcript) -> str:
    return "\n".join(f"{u.speaker} [{clock(u.start)} - {clock(u.end)}]: {u.text}" for u in t.utterances) + "\n"


def to_srt(t: Transcript) -> str:
    blocks = []
    for n, u in enumerate(t.utterances, 1):
        blocks.append(f"{n}\n{clock(u.start, ',')} --> {clock(u.end, ',')}\n{u.speaker}: {u.text}\n")
    return "\n".join(blocks)


def to_json(t: Transcript) -> str:
    return json.dumps(t.to_dict(), indent=2)


def to_rttm(t: Transcript, file_id: str) -> str:
    return "\n".join(rttm_lines(t.turns, file_id)) + "\n"


def write_outputs(t: Transcript, out_dir: str | Path, stem: str, formats=FORMATS) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    unknown = set(formats) - set(FORMATS)
    if unknown:
        raise ValueError(f"unknown formats {sorted(unknown)}. Known: {FORMATS}")
    writers = {"json": to_json, "srt": to_srt, "txt": to_txt, "rttm": lambda tr: to_rttm(tr, stem)}
    paths = []
    for fmt in formats:
        p = out / f"{stem}.{fmt}"
        p.write_text(writers[fmt](t), encoding="utf-8")
        paths.append(p)
    return paths
