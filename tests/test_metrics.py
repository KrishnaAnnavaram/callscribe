import itertools

import numpy as np
import pytest

from callscribe.metrics import cer, cpwer, der, edit_distance, error_counts, normalize, wer
from callscribe.types import Turn, Word


def brute_lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def test_normalize():
    assert normalize("Revenue grew 12% & margins, uh, improved!") == [
        "revenue", "grew", "12", "percent", "and", "margins", "improved"]
    assert normalize("It's $1,000.") == ["it's", "1000"]
    assert "uh" in normalize("uh okay", drop_fillers=False)


def test_vectorised_edit_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = list(rng.integers(0, 4, rng.integers(0, 12)))
        b = list(rng.integers(0, 4, rng.integers(0, 12)))
        assert edit_distance(a, b) == brute_lev(a, b)


def test_error_counts_breakdown():
    ref = "the quarter was strong".split()
    hyp = "the quarter was very strong too".split()
    ec = error_counts(ref, hyp)
    assert (ec.substitutions, ec.deletions, ec.insertions, ec.errors) == (0, 0, 2, 2)
    ec = error_counts("a b c".split(), "a x".split())
    assert ec.errors == 2 and ec.substitutions + ec.deletions + ec.insertions == 2
    big = error_counts(list(range(100)), list(range(1, 101)), max_cells=10)
    assert big.errors == 2 and big.substitutions is None


def test_wer_counts_order_and_repeats():
    """Problem 2: a set overlap of 1.0 can hide a bad transcript. WER does not."""
    ref = "revenue grew and margins fell"
    hyp = "fell margins and grew revenue"
    assert set(normalize(ref)) == set(normalize(hyp))
    assert wer(ref, hyp) >= 0.8
    assert wer("profit profit profit", "profit") == pytest.approx(2 / 3)
    assert wer(ref, ref) == 0.0


def test_cer():
    assert cer("abc", "abd") == pytest.approx(1 / 3)
    assert np.isnan(cer("a" * 50, "b", max_chars=10))


def words(spec):
    return [Word(float(n), n + 0.5, text, spk) for n, (text, spk) in enumerate(spec)]


def test_cpwer_is_label_free_but_punishes_wrong_attribution():
    ref = words([("good", "A"), ("morning", "A"), ("thanks", "B"), ("operator", "B")])
    relabelled = [w.with_speaker({"A": "S1", "B": "S0"}[w.speaker]) for w in ref]
    res = cpwer(ref, relabelled)
    assert res.rate == 0.0 and res.mapping == {"S1": "A", "S0": "B"}
    one_speaker = [w.with_speaker("S0") for w in ref]
    # S0 maps to A: 2 insertions. B has no stream: 2 deletions. 4 errors for 4 words.
    assert cpwer(ref, one_speaker).rate == pytest.approx(1.0)
    assert wer(" ".join(w.text for w in ref), " ".join(w.text for w in one_speaker)) == 0.0


def test_der_basics():
    ref = [Turn(0, 10, "A"), Turn(10, 20, "B")]
    assert der(ref, ref, collar=0).rate == 0.0
    swapped = [Turn(0, 10, "x"), Turn(10, 20, "y")]
    assert der(ref, swapped, collar=0).rate == 0.0
    merged = der(ref, [Turn(0, 20, "x")], collar=0)
    assert merged.confusion == pytest.approx(10.0, abs=0.02) and merged.rate == pytest.approx(0.5, abs=0.01)
    missing = der(ref, [Turn(0, 10, "x")], collar=0)
    assert missing.missed == pytest.approx(10.0, abs=0.02)
    fa = der([Turn(0, 10, "A")], [Turn(0, 15, "x")], collar=0)
    assert fa.false_alarm == pytest.approx(5.0, abs=0.02)


def test_der_collar_forgives_boundary_errors():
    ref = [Turn(0, 10, "A"), Turn(10, 20, "B")]
    late = [Turn(0, 10.2, "x"), Turn(10.2, 20, "y")]
    assert der(ref, late, collar=0).rate > 0
    assert der(ref, late, collar=0.25).rate == 0.0


def test_der_overlap_handling():
    ref = [Turn(0, 10, "A"), Turn(5, 10, "B")]
    hyp = [Turn(0, 10, "x")]
    full = der(ref, hyp, collar=0)
    assert full.total == pytest.approx(15.0, abs=0.02)
    assert full.missed == pytest.approx(5.0, abs=0.02)
    assert der(ref, hyp, collar=0, skip_overlap=True).rate == 0.0


@pytest.mark.parametrize("perm", list(itertools.permutations(["p", "q", "r"])))
def test_der_mapping_is_optimal_for_any_label_order(perm):
    ref = [Turn(0, 3, "A"), Turn(3, 6, "B"), Turn(6, 9, "C")]
    hyp = [Turn(t.start, t.end, perm[k]) for k, t in enumerate(ref)]
    assert der(ref, hyp, collar=0).rate == 0.0
