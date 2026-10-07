"""The evaluation harness: run the pipeline on each manifest item and score the real output.

For each file: WER, CER, cpWER (from the reference words and speakers) and DER (if an RTTM file
exists). The pooled numbers add the errors of all files and divide by the pooled reference size.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import SAMPLE_RATE
from .audio import load_audio
from .config import Settings
from .metrics import cer, cpwer, der, error_counts, normalize
from .pipeline import build_pipeline
from .references import ManifestItem, read_nlp, read_rttm
from .types import Transcript


@dataclass
class FileResult:
    id: str
    split: str
    ref_words: int
    wer: float
    substitutions: int | None
    deletions: int | None
    insertions: int | None
    cer: float
    cpwer: float
    der: float | None
    der_missed_s: float | None = None
    der_false_alarm_s: float | None = None
    der_confusion_s: float | None = None
    der_total_s: float | None = None
    ref_speakers: int = 0
    hyp_speakers: int = 0
    word_errors: int = 0
    cp_errors: int = 0


@dataclass
class Report:
    pipeline: dict
    split: str
    collar: float
    files: list[FileResult] = field(default_factory=list)

    def pooled(self) -> dict:
        n = sum(f.ref_words for f in self.files)
        out = {
            "files": len(self.files),
            "ref_words": n,
            "wer": sum(f.word_errors for f in self.files) / n if n else math.nan,
            "cpwer": sum(f.cp_errors for f in self.files) / n if n else math.nan,
        }
        with_der = [f for f in self.files if f.der is not None]
        total = sum(f.der_total_s for f in with_der)
        if total:
            err = sum(f.der_missed_s + f.der_false_alarm_s + f.der_confusion_s for f in with_der)
            out["der"] = err / total
            out["der_files"] = len(with_der)
        return out

    def to_dict(self) -> dict:
        return {"pipeline": self.pipeline, "split": self.split, "collar": self.collar,
                "pooled": self.pooled(), "files": [asdict(f) for f in self.files]}

    def markdown(self) -> str:
        p = self.pooled()
        lines = [
            f"# callscribe evaluation ({self.split})", "",
            f"Pipeline: ASR `{self.pipeline.get('asr')}`, diarizer `{self.pipeline.get('diarizer')}`. "
            f"DER collar: {self.collar} s.", "",
            "| File | Ref words | WER | S/D/I | CER | cpWER | DER | Speakers ref/hyp |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for f in self.files:
            sdi = "/".join("-" if v is None else str(v) for v in (f.substitutions, f.deletions, f.insertions))
            d = "-" if f.der is None else f"{f.der:.3f}"
            lines.append(f"| {f.id} | {f.ref_words} | {f.wer:.3f} | {sdi} | {f.cer:.3f} | {f.cpwer:.3f} | "
                         f"{d} | {f.ref_speakers}/{f.hyp_speakers} |")
        d = f"{p['der']:.3f}" if "der" in p else "-"
        lines.append(f"| **pooled** | {p['ref_words']} | {p['wer']:.3f} | | | {p['cpwer']:.3f} | {d} | |")
        return "\n".join(lines) + "\n"


def score(transcript: Transcript, item: ManifestItem, collar: float = 0.25) -> FileResult:
    ref_words = read_nlp(item.ref_words)
    ref_tokens = [t for w in ref_words for t in normalize(w.text)]
    hyp_tokens = [t for w in transcript.words for t in normalize(w.text)]
    ec = error_counts(ref_tokens, hyp_tokens)
    cp = cpwer(ref_words, transcript.words)
    result = FileResult(
        id=item.id, split=item.split, ref_words=len(ref_tokens), wer=ec.rate,
        substitutions=ec.substitutions, deletions=ec.deletions, insertions=ec.insertions,
        cer=cer(" ".join(ref_tokens), " ".join(hyp_tokens)), cpwer=cp.rate, der=None,
        ref_speakers=len({w.speaker for w in ref_words}),
        hyp_speakers=len({t.speaker for t in transcript.turns}),
        word_errors=ec.errors, cp_errors=cp.errors,
    )
    if item.ref_rttm:
        d = der(read_rttm(item.ref_rttm), transcript.turns, collar=collar)
        result.der, result.der_missed_s = d.rate, d.missed
        result.der_false_alarm_s, result.der_confusion_s, result.der_total_s = d.false_alarm, d.confusion, d.total
    return result


def evaluate(items: list[ManifestItem], settings: Settings, split: str = "test", collar: float = 0.25,
             oracle_speakers: bool = False, seed: int = 0, progress=None) -> Report:
    report = None
    for item in items:
        ref_words = read_nlp(item.ref_words) if settings.asr_backend == "scripted" else None
        ref_turns = read_rttm(item.ref_rttm) if item.ref_rttm else None
        pipe = build_pipeline(settings, ref_words, ref_turns, seed=seed)
        if report is None:
            report = Report(pipe.describe(), split, collar)
        audio = load_audio(item.audio, SAMPLE_RATE)
        n_spk = len({w.speaker for w in read_nlp(item.ref_words)}) if oracle_speakers else None
        transcript = pipe.run(audio, SAMPLE_RATE, num_speakers=n_spk, meta={"id": item.id})
        report.files.append(score(transcript, item, collar))
        if progress:
            progress(item.id)
    if report is None:
        raise ValueError(f"no manifest items in split {split!r}")
    return report


def write_report(report: Report, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j, m = out / "report.json", out / "report.md"
    j.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    m.write_text(report.markdown(), encoding="utf-8")
    return j, m
