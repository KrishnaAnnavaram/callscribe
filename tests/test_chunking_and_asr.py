import numpy as np
import pytest

from callscribe import SAMPLE_RATE
from callscribe.asr import ScriptedASR
from callscribe.chunking import ChunkedASR, merge_words, plan_chunks
from callscribe.synthetic import CallSpec, synth_call
from callscribe.types import Word
from callscribe.vad import detect


class WindowASR:
    """Hears only the reference words that are fully inside its audio window (no errors)."""

    name = "window"

    def __init__(self, ref):
        self.inner = ScriptedASR(ref, 0, 0, 0)

    def transcribe(self, x, sr, offset=0.0):
        return self.inner.transcribe(x, sr, offset)


@pytest.fixture(scope="module")
def long_call():
    return synth_call(CallSpec(n_speakers=2, n_turns=40, seed=3))


def test_plan_chunks_partition_the_timeline():
    chunks = plan_chunks(95.0, [(0, 20), (21, 50), (51, 95)], max_len=30, overlap=1.0)
    assert chunks[0].own_start == 0 and chunks[-1].own_end == 95.0
    for a, b in zip(chunks, chunks[1:]):
        assert a.own_end == b.own_start
        assert b.start == pytest.approx(b.own_start - 1.0)
    assert all(c.own_end - c.own_start <= 30 for c in chunks)
    assert chunks[0].own_end == pytest.approx(20.5)  # the cut is in the silence


def test_plan_without_silence_cuts_at_max_len():
    chunks = plan_chunks(70.0, None, max_len=30, overlap=2.0)
    assert [c.own_end for c in chunks] == [30, 60, 70]
    with pytest.raises(ValueError):
        plan_chunks(10, None, max_len=0)


def test_hard_cuts_lose_words_but_vad_chunks_with_overlap_do_not(long_call):
    """Problem 5: fixed 30 s windows cut words. VAD cuts plus overlap keep each word once."""
    ref = long_call.words
    asr = WindowASR(ref)
    x, sr = long_call.audio, SAMPLE_RATE
    assert long_call.duration > 60
    hard = []
    for c in plan_chunks(long_call.duration, None, max_len=7.0, overlap=0.0):
        seg = x[int(c.start * sr):int(c.end * sr)]
        hard.append((c, [w.shifted(c.start) for w in asr.transcribe(seg, sr, offset=c.start)]))
    hard_words = merge_words(hard)
    good = ChunkedASR(asr, max_len=7.0, overlap=1.0).transcribe(x, sr)
    assert len(hard_words) < len(ref)
    assert [w.text for w in good] == [w.text for w in ref]
    assert np.allclose([w.start for w in good], [w.start for w in ref], atol=1e-3)


def test_merge_keeps_a_boundary_word_once():
    from callscribe.chunking import Chunk

    w = Word(9.8, 10.4, "margin")
    a, b = Chunk(0, 10, 0, 11), Chunk(10, 20, 9, 20)
    assert merge_words([(a, [w]), (b, [w])]) == [w]


def test_scripted_asr_is_deterministic_and_chunk_invariant(call):
    asr = ScriptedASR(call.words, 0.1, 0.1, 0.05, seed=2)
    full = asr.transcribe(call.audio, SAMPLE_RATE)
    again = ScriptedASR(call.words, 0.1, 0.1, 0.05, seed=2).transcribe(call.audio, SAMPLE_RATE)
    assert full == again
    chunked = ChunkedASR(asr, max_len=8.0, overlap=1.0).transcribe(call.audio, SAMPLE_RATE)
    assert [w.text for w in chunked] == [w.text for w in full]


def test_scripted_asr_error_rates_are_close_to_the_settings():
    ref = [Word(i * 0.5, i * 0.5 + 0.3, f"w{i % 50}") for i in range(4000)]
    out = ScriptedASR(ref, 0.10, 0.05, 0.0, seed=1).transcribe(np.zeros(2100 * SAMPLE_RATE), SAMPLE_RATE)
    assert len(out) / len(ref) == pytest.approx(0.95, abs=0.02)
    with pytest.raises(ValueError):
        ScriptedASR(ref, 0.8, 0.5, 0)


def test_vad_finds_the_bursts(call):
    regions = detect(call.audio, SAMPLE_RATE)
    covered = sum(any(s - 0.05 <= w.mid <= e + 0.05 for s, e in regions) for w in call.words)
    assert covered / len(call.words) > 0.98
    assert detect(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE) == []
