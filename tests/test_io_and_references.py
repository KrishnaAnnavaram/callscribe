import shutil

import numpy as np
import pytest

from callscribe.audio import AudioError, load_audio, read_wav, resample, write_wav
from callscribe.export import clock, to_srt, to_txt, write_outputs
from callscribe.references import (ManifestItem, ReferenceError, dev_or_test, earnings21_manifest, read_manifest,
                                   read_nlp, read_rttm, write_manifest, write_rttm)
from callscribe.stats import speaker_stats, timeline
from callscribe.types import Transcript, Turn, Utterance, Word

NLP = """token|speaker|ts|endTs|punctuation|case|tags|wer_tags
Good|1||||UC|[]|[]
afternoon|1||||LC|[]|[]
,|1||||||
Thank|2|3.5|3.9||UC|[]|[]
"""


def test_read_nlp(tmp_path):
    p = tmp_path / "a.nlp"
    p.write_text(NLP, encoding="utf-8")
    ws = read_nlp(p)
    assert [w.text for w in ws] == ["Good", "afternoon", ",", "Thank"]
    assert ws[0].speaker == "1" and ws[3].start == 3.5 and ws[0].start == 0.0
    p.write_text("word|x\n", encoding="utf-8")
    with pytest.raises(ReferenceError):
        read_nlp(p)


def test_rttm_round_trip(tmp_path):
    turns = [Turn(0.5, 2.0, "A"), Turn(2.25, 3.0, "B")]
    p = write_rttm(turns, "call1", tmp_path / "c.rttm")
    assert read_rttm(p) == turns
    assert p.read_text().startswith("SPEAKER call1 1 0.500 1.500")


def test_wav_round_trip_and_resample(tmp_path):
    x = np.sin(2 * np.pi * 440 * np.arange(8000) / 8000).astype(np.float32) * 0.5
    write_wav(tmp_path / "t.wav", x, sr=8000)
    y, sr = read_wav(tmp_path / "t.wav")
    assert sr == 8000 and np.allclose(x, y, atol=1e-3)
    z = load_audio(tmp_path / "t.wav")
    assert len(z) == 16000 and z.dtype == np.float32
    assert len(resample(x, 8000, 8000)) == len(x)
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "none.wav")


def test_mp3_needs_ffmpeg(tmp_path):
    p = tmp_path / "x.mp3"
    p.write_bytes(b"not audio")
    if shutil.which("ffmpeg") is None:
        with pytest.raises(AudioError, match="ffmpeg is not installed"):
            load_audio(p)
    else:
        with pytest.raises(AudioError, match="ffmpeg failed"):
            load_audio(p)


def test_manifest_paths_and_splits(tmp_path):
    write_manifest([ManifestItem("a", "a.wav", "a.nlp", None, "dev"), ManifestItem("b", "b.wav", "b.nlp")],
                   tmp_path / "m.json")
    assert [i.id for i in read_manifest(tmp_path / "m.json", "dev")] == ["a"]
    items = read_manifest(tmp_path / "m.json", "all")
    assert items[1].audio == str(tmp_path / "b.wav")
    assert dev_or_test("4320211") == dev_or_test("4320211")
    share = np.mean([dev_or_test(str(n)) == "dev" for n in range(2000)])
    assert 0.2 < share < 0.3


def test_earnings21_manifest_finds_pairs(tmp_path):
    (tmp_path / "media").mkdir()
    (tmp_path / "refs").mkdir()
    for cid in ("111", "222"):
        (tmp_path / "media" / f"{cid}.mp3").write_bytes(b"")
        (tmp_path / "refs" / f"{cid}.nlp").write_text(NLP, encoding="utf-8")
    (tmp_path / "media" / "333.mp3").write_bytes(b"")  # no reference: skipped
    write_rttm([Turn(0, 1, "1")], "111", tmp_path / "refs" / "111.rttm")
    items = earnings21_manifest(tmp_path)
    assert [i.id for i in items] == ["111", "222"]
    assert items[0].ref_rttm and items[1].ref_rttm is None
    with pytest.raises(ReferenceError):
        earnings21_manifest(tmp_path / "media")


def transcript():
    ws = [Word(0.0, 0.4, "hello", "S0"), Word(0.5, 0.9, "there", "S0"), Word(1.5, 2.0, "hi", "S1")]
    turns = [Turn(0.0, 0.9, "S0"), Turn(1.5, 2.0, "S1")]
    us = [Utterance(0.0, 0.9, "S0", "hello there"), Utterance(1.5, 2.0, "S1", "hi")]
    return Transcript(ws, turns, us, {"id": "t"})


def test_exports(tmp_path):
    t = transcript()
    assert clock(3723.5) == "01:02:03.500"
    assert "00:00:00,000 --> 00:00:00,900" in to_srt(t)
    assert to_txt(t).splitlines()[1] == "S1 [00:00:01.500 - 00:00:02.000]: hi"
    paths = write_outputs(t, tmp_path, "t")
    assert sorted(p.suffix for p in paths) == [".json", ".rttm", ".srt", ".txt"]
    again = Transcript.from_dict(__import__("json").loads((tmp_path / "t.json").read_text()))
    assert again.words == t.words
    with pytest.raises(ValueError):
        write_outputs(t, tmp_path, "t", ["docx"])


def test_speaker_stats_and_timeline():
    st = speaker_stats(transcript())
    assert st[0].speaker == "S0" and st[0].words == 2 and st[0].share == pytest.approx(0.9 / 1.4, abs=1e-3)
    tl = timeline(transcript(), width=20)
    assert tl.count("|") == 4 and "#" in tl
