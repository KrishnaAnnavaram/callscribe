import json
import subprocess
import sys
from pathlib import Path

import pytest

from callscribe import SAMPLE_RATE
from callscribe.cli import main
from callscribe.config import Settings
from callscribe.evaluate import evaluate, score
from callscribe.metrics import error_counts, normalize
from callscribe.pipeline import build_pipeline
from callscribe.references import read_manifest, read_nlp, read_rttm
from callscribe.tune import tune

REPO = Path(__file__).resolve().parents[1]
SECRET = "hf_" + "Z" * 30


def test_evaluation_scores_the_pipeline_output(dataset):
    """Problem 1: the scored transcript is the pipeline output, not a side transcript."""
    item = read_manifest(dataset, "test")[0]
    s = Settings()
    pipe = build_pipeline(s, read_nlp(item.ref_words), seed=0)
    from callscribe.audio import load_audio

    t = pipe.run(load_audio(item.audio), SAMPLE_RATE)
    direct = error_counts([x for w in read_nlp(item.ref_words) for x in normalize(w.text)],
                          [x for w in t.words for x in normalize(w.text)]).rate
    report = evaluate([item], s)
    assert report.files[0].wer == pytest.approx(direct)
    assert report.pipeline == {"asr": "scripted", "diarizer": "spectral"}
    assert score(t, item).wer == pytest.approx(direct)


def test_scripted_pipeline_needs_reference_words():
    with pytest.raises(ValueError, match="faster-whisper"):
        build_pipeline(Settings())


def test_tuning_uses_dev_only_and_beats_the_default(dataset):
    """Problem 6: thresholds come from a dev split, and the tuned value is better there."""
    dev = read_manifest(dataset, "dev")
    with pytest.raises(ValueError, match="never on the test split"):
        tune(read_manifest(dataset, "all"))
    result = tune(dev)
    default = [d for p, d in result.trials if p == {"min_duration_off": 0.3, "threshold": 0.15}][0]
    assert result.best_der <= default
    assert result.best_der < 0.1


def test_reference_diarizer_gives_zero_der(dataset):
    item = read_manifest(dataset, "test")[0]
    report = evaluate([item], Settings(diarizer="reference"))
    assert report.files[0].der == 0.0
    assert report.files[0].cpwer == pytest.approx(report.files[0].wer)


def test_pooled_numbers_weight_by_reference_size(dataset):
    items = read_manifest(dataset, "test")
    report = evaluate(items, Settings(diarizer="reference"))
    p = report.pooled()
    assert p["files"] == len(items)
    assert p["wer"] == pytest.approx(sum(f.word_errors for f in report.files) / p["ref_words"])
    assert p["der"] == 0.0
    assert "| **pooled** |" in report.markdown()


def test_token_is_never_printed(monkeypatch, capsys):
    """Problem 7: the token is not in the settings repr or in any CLI output."""
    monkeypatch.setenv("HF_TOKEN", SECRET)
    s = Settings.from_env()
    assert s.hf_token == SECRET
    assert SECRET not in repr(s) and SECRET not in str(s.public())
    assert main(["config"]) == 0
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err
    assert "hf_token = set" in out.out


def test_bad_backend_names_are_rejected(monkeypatch):
    monkeypatch.setenv("CALLSCRIBE_ASR_BACKEND", "whisper-tiny")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_no_notebook_with_outputs_in_the_repository():
    """Problems 7 and 8: no notebook can keep a printed token or a second copy of the code."""
    for nb in REPO.rglob("*.ipynb"):
        if ".venv" in nb.parts:
            continue
        cells = json.loads(nb.read_text(encoding="utf-8")).get("cells", [])
        assert not any(c.get("outputs") for c in cells), nb


def test_cli_end_to_end(tmp_path, capsys):
    data = tmp_path / "d"
    assert main(["synth", "--out", str(data), "--calls", "4", "--seed", "1"]) == 0
    manifest = data / "manifest.json"
    assert main(["tune", "--manifest", str(manifest), "--out", str(tmp_path / "p.json")]) == 0
    assert main(["transcribe", str(data / "synth001.wav"), "--out", str(tmp_path / "out"),
                 "--diarizer-params", str(tmp_path / "p.json")]) == 0
    assert (tmp_path / "out" / "synth001.srt").exists()
    assert main(["stats", str(tmp_path / "out" / "synth001.json")]) == 0
    assert main(["evaluate", "--manifest", str(manifest), "--out", str(tmp_path / "rep"),
                 "--diarizer-params", str(tmp_path / "p.json")]) == 0
    rep = json.loads((tmp_path / "rep" / "report.json").read_text())
    assert rep["pooled"]["files"] == 3 and rep["split"] == "test"
    capsys.readouterr()
    assert main(["transcribe", str(tmp_path / "missing.wav")]) == 2


def test_module_entry_point_runs():
    r = subprocess.run([sys.executable, "-m", "callscribe.cli", "--help"], capture_output=True, text=True,
                       env={"PYTHONPATH": str(REPO / "src"), "SYSTEMROOT": __import__("os").environ.get(
                           "SYSTEMROOT", "")})
    assert r.returncode == 0 and "transcribe" in r.stdout


def test_faster_whisper_backend_optional():
    pytest.importorskip("faster_whisper")
