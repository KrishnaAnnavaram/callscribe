import json

import pytest

from callscribe import SAMPLE_RATE
from callscribe.align import assign_speakers, utterances
from callscribe.diarize import (ReferenceDiarizer, SpectralDiarizer, SpectralParams, load_params, merge_nested,
                                merge_turns, relabel, save_params, spectral_from_params)
from callscribe.metrics import der
from callscribe.types import UNKNOWN, Turn, Word


def test_spectral_diarizer_separates_synthetic_voices(call):
    turns = SpectralDiarizer().diarize(call.audio, SAMPLE_RATE, num_speakers=3)
    assert len({t.speaker for t in turns}) == 3
    assert der(call.turns, turns).rate < 0.05


def test_threshold_controls_the_speaker_count(call):
    low = SpectralDiarizer(SpectralParams(threshold=0.02)).diarize(call.audio, SAMPLE_RATE)
    high = SpectralDiarizer(SpectralParams(threshold=0.9)).diarize(call.audio, SAMPLE_RATE)
    assert len({t.speaker for t in low}) >= 3
    assert len({t.speaker for t in high}) == 1


def test_silence_gives_no_turns():
    import numpy as np

    assert SpectralDiarizer().diarize(np.zeros(SAMPLE_RATE * 3, dtype=np.float32), SAMPLE_RATE) == []


def test_merge_and_relabel():
    turns = [Turn(0, 1, "b"), Turn(1.1, 2, "b"), Turn(2.5, 2.6, "a"), Turn(3, 4, "a")]
    merged = merge_turns(turns, min_off=0.3, min_on=0.2)
    assert merged == [Turn(0, 2, "b"), Turn(3, 4, "a")]
    assert [t.speaker for t in relabel(merged)] == ["SPEAKER_00", "SPEAKER_01"]


def test_params_file_round_trip(tmp_path):
    path = save_params({"threshold": 0.03}, tmp_path / "p.json", {"split": "dev"})
    assert load_params(path) == {"threshold": 0.03}
    assert spectral_from_params(load_params(path)).params.threshold == 0.03
    with pytest.raises(ValueError):
        spectral_from_params({"nonsense": 1})
    assert json.loads(path.read_text())["meta"]["split"] == "dev"


def test_merge_nested_for_pyannote_parameters():
    base = {"clustering": {"threshold": 0.7, "method": "centroid"}, "segmentation": {"min_duration_off": 0.0}}
    out = merge_nested(base, {"clustering.threshold": 0.6})
    assert out["clustering"] == {"threshold": 0.6, "method": "centroid"}
    assert base["clustering"]["threshold"] == 0.7


def test_assign_speakers_by_largest_overlap_then_nearest():
    turns = [Turn(0, 2, "A"), Turn(2, 5, "B")]
    ws = [Word(1.5, 2.6, "x"), Word(0.2, 0.4, "y"), Word(5.3, 5.5, "z"), Word(9, 9.2, "far")]
    got = [w.speaker for w in assign_speakers(ws, turns, max_gap=0.5)]
    assert got == ["B", "A", "B", UNKNOWN]


def test_utterances_split_on_speaker_change_and_pause():
    ws = [Word(0, 0.3, "a", "A"), Word(0.4, 0.6, "b", "A"), Word(3, 3.2, "c", "A"), Word(3.3, 3.5, "d", "B")]
    us = utterances(ws, max_pause=1.0)
    assert [(u.speaker, u.text) for u in us] == [("A", "a b"), ("A", "c"), ("B", "d")]


def test_reference_diarizer_returns_the_reference(call):
    assert ReferenceDiarizer(call.turns).diarize(call.audio, SAMPLE_RATE) == call.turns


def test_pyannote_backend_optional():
    pytest.importorskip("pyannote.audio")
    import os

    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")
