"""Grid search of diarizer parameters on the DEV split. The objective is the pooled DER.

The result goes into a JSON file. ``CALLSCRIBE_DIARIZER_PARAMS`` points the pipeline to it.
Never tune on the test split.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from . import SAMPLE_RATE
from .audio import load_audio
from .diarize import spectral_from_params
from .metrics import der
from .references import ManifestItem, read_rttm

SPECTRAL_GRID = {
    "threshold": [0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.25],
    "min_duration_off": [0.1, 0.3, 0.6],
}

# pyannote 3.x names. Use them with a pyannote diarizer factory.
PYANNOTE_GRID = {
    "clustering.threshold": [0.6, 0.7, 0.8],
    "segmentation.min_duration_off": [0.0, 0.1, 0.3],
}


@dataclass
class TuneResult:
    best_params: dict
    best_der: float
    trials: list[tuple[dict, float]]


def grid(space: dict[str, list]) -> list[dict]:
    keys = sorted(space)
    return [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]


def tune(items: list[ManifestItem], space: dict[str, list] | None = None, factory=spectral_from_params,
         collar: float = 0.25) -> TuneResult:
    """Return the parameters with the lowest pooled DER on ``items`` (all must have an RTTM file)."""
    if any(i.split == "test" for i in items):
        raise ValueError("tune only on dev items, never on the test split")
    usable = [i for i in items if i.ref_rttm]
    if not usable:
        raise ValueError("tuning needs dev items with an RTTM reference")
    audio = {i.id: load_audio(i.audio, SAMPLE_RATE) for i in usable}
    refs = {i.id: read_rttm(i.ref_rttm) for i in usable}
    trials = []
    for params in grid(space or SPECTRAL_GRID):
        diarizer = factory(params)
        err = total = 0.0
        for i in usable:
            d = der(refs[i.id], diarizer.diarize(audio[i.id], SAMPLE_RATE), collar=collar)
            err += d.missed + d.false_alarm + d.confusion
            total += d.total
        trials.append((params, err / total if total else float("inf")))
    best = min(trials, key=lambda t: t[1])
    return TuneResult(best[0], best[1], trials)
