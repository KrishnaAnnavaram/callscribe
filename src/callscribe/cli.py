"""Command line: ``callscribe synth | transcribe | evaluate | tune | stats | manifest | config``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from . import SAMPLE_RATE
from .audio import load_audio
from .config import ASR_BACKENDS, DIARIZERS, Settings, load_dotenv
from .diarize import PyannoteDiarizer, save_params
from .evaluate import evaluate, write_report
from .export import FORMATS, write_outputs
from .pipeline import build_pipeline
from .references import earnings21_manifest, read_manifest, read_nlp, read_rttm, write_manifest
from .stats import speaker_stats, timeline
from .synthetic import write_dataset
from .tune import PYANNOTE_GRID, tune
from .types import Transcript


def _settings(args) -> Settings:
    load_dotenv()
    s = Settings.from_env()
    over = {k: getattr(args, k) for k in ("asr_backend", "diarizer", "diarizer_params")
            if getattr(args, k, None)}
    return replace(s, **over).validate()


def cmd_synth(args) -> int:
    manifest = write_dataset(args.out, args.calls, args.seed)
    items = read_manifest(manifest)
    print(f"{len(items)} synthetic calls in {args.out}: "
          f"{sum(i.split == 'dev' for i in items)} dev, {sum(i.split == 'test' for i in items)} test")
    print(f"manifest: {manifest}")
    return 0


def cmd_transcribe(args) -> int:
    s = _settings(args)
    audio_path = Path(args.audio)
    nlp = Path(args.ref_words) if args.ref_words else audio_path.with_suffix(".nlp")
    rttm = Path(args.ref_rttm) if args.ref_rttm else audio_path.with_suffix(".rttm")
    ref_words = read_nlp(nlp) if s.asr_backend == "scripted" and nlp.exists() else None
    ref_turns = read_rttm(rttm) if s.diarizer == "reference" and rttm.exists() else None
    pipe = build_pipeline(s, ref_words, ref_turns, seed=args.seed)
    audio = load_audio(audio_path, SAMPLE_RATE)
    t = pipe.run(audio, SAMPLE_RATE, num_speakers=args.num_speakers, meta={"id": audio_path.stem})
    paths = write_outputs(t, args.out or s.out_dir, audio_path.stem, args.formats.split(","))
    print(f"{len(t.words)} words, {len({x.speaker for x in t.turns})} speakers, "
          f"{len(t.utterances)} utterances ({pipe.describe()})")
    for p in paths:
        print(f"wrote {p}")
    return 0


def cmd_evaluate(args) -> int:
    s = _settings(args)
    items = read_manifest(args.manifest, args.split)
    report = evaluate(items, s, args.split, args.collar, args.oracle_speakers, args.seed,
                      progress=lambda i: print(f"scored {i}", file=sys.stderr))
    print(report.markdown())
    if args.out:
        j, m = write_report(report, args.out)
        print(f"wrote {j} and {m}")
    return 0


def cmd_tune(args) -> int:
    items = read_manifest(args.manifest, "dev")
    if args.diarizer == "pyannote":
        s = _settings(args)
        result = tune(items, PYANNOTE_GRID, lambda params: PyannoteDiarizer(
            s.pyannote_pipeline, s.hf_token, s.device, params), collar=args.collar)
    else:
        result = tune(items, collar=args.collar)
    for params, d in sorted(result.trials, key=lambda t: t[1])[:5]:
        print(f"DER {d:.4f}  {params}")
    path = save_params(result.best_params, args.out,
                       {"objective": "pooled DER", "split": "dev", "dev_files": len(items),
                        "best_der": round(result.best_der, 4)})
    print(f"best dev DER {result.best_der:.4f}. Saved to {path}. Set CALLSCRIBE_DIARIZER_PARAMS={path}")
    return 0


def cmd_stats(args) -> int:
    t = Transcript.from_dict(json.loads(Path(args.transcript).read_text(encoding="utf-8")))
    print(f"{'speaker':>12} {'talk s':>8} {'share':>6} {'turns':>5} {'words':>6} {'wpm':>6} {'longest s':>9}")
    for st in speaker_stats(t):
        print(f"{st.speaker:>12} {st.talk_time_s:8.1f} {st.share:6.1%} {st.turns:5d} {st.words:6d} "
              f"{st.words_per_minute:6.1f} {st.longest_turn_s:9.1f}")
    print()
    print(timeline(t))
    return 0


def cmd_manifest(args) -> int:
    items = earnings21_manifest(args.root, args.dev_fraction)
    write_manifest(items, args.out)
    with_rttm = sum(1 for i in items if i.ref_rttm)
    print(f"{len(items)} files ({sum(i.split == 'dev' for i in items)} dev), {with_rttm} with RTTM. "
          f"Wrote {args.out}")
    return 0


def cmd_config(args) -> int:
    s = _settings(args)
    for k, v in s.public().items():
        print(f"{k:>18} = {v}")
    return 0


def _backend_flags(p) -> None:
    p.add_argument("--asr-backend", choices=ASR_BACKENDS)
    p.add_argument("--diarizer", choices=DIARIZERS)
    p.add_argument("--diarizer-params", help="JSON file from 'callscribe tune'")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="callscribe", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="write synthetic calls with exact references")
    p.add_argument("--out", required=True)
    p.add_argument("--calls", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("transcribe", help="speaker-attributed transcript of one audio file")
    p.add_argument("audio")
    p.add_argument("--out", help="output folder (CALLSCRIBE_OUT_DIR)")
    p.add_argument("--formats", default=",".join(FORMATS))
    p.add_argument("--num-speakers", type=int)
    p.add_argument("--ref-words", help=".nlp file for the scripted ASR (default: next to the audio)")
    p.add_argument("--ref-rttm", help=".rttm file for the reference diarizer")
    p.add_argument("--seed", type=int, default=0)
    _backend_flags(p)
    p.set_defaults(fn=cmd_transcribe)

    p = sub.add_parser("evaluate", help="WER, CER, cpWER and DER on a manifest split")
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p.add_argument("--collar", type=float, default=0.25)
    p.add_argument("--oracle-speakers", action="store_true", help="give the true speaker count")
    p.add_argument("--out", help="folder for report.json and report.md")
    p.add_argument("--seed", type=int, default=0)
    _backend_flags(p)
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("tune", help="grid search of the diarizer parameters on the dev split")
    p.add_argument("--manifest", required=True)
    p.add_argument("--diarizer", choices=["spectral", "pyannote"], default="spectral")
    p.add_argument("--collar", type=float, default=0.25)
    p.add_argument("--out", default="configs/diarizer_params.json")
    p.set_defaults(fn=cmd_tune)

    p = sub.add_parser("stats", help="talk time for each speaker and a text timeline")
    p.add_argument("transcript", help="a transcript .json file")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("manifest", help="build a manifest from an Earnings-21 folder")
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="manifest.json")
    p.add_argument("--dev-fraction", type=float, default=0.25)
    p.set_defaults(fn=cmd_manifest)

    p = sub.add_parser("config", help="show the settings (the token shows only as set or not set)")
    _backend_flags(p)
    p.set_defaults(fn=cmd_config)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
