"""The one transcription pipeline. The CLI, the evaluation and the tuning all call ``CallPipeline.run``.

Thus the evaluated transcript is always the output of the configured system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import SAMPLE_RATE
from .align import assign_speakers, utterances
from .asr import ASR, FasterWhisperASR, ScriptedASR
from .config import Settings
from .diarize import Diarizer, PyannoteDiarizer, ReferenceDiarizer, load_params, spectral_from_params
from .types import Transcript, Turn, Word


@dataclass
class CallPipeline:
    asr: ASR
    diarizer: Diarizer
    max_gap: float = 0.5
    max_pause: float = 1.0

    def describe(self) -> dict:
        return {"asr": getattr(self.asr, "name", "?"), "diarizer": getattr(self.diarizer, "name", "?")}

    def run(self, audio: np.ndarray, sr: int = SAMPLE_RATE, num_speakers: int | None = None,
            meta: dict | None = None) -> Transcript:
        t0 = time.perf_counter()
        turns = self.diarizer.diarize(audio, sr, num_speakers)
        t1 = time.perf_counter()
        words = self.asr.transcribe(audio, sr, offset=0.0)
        t2 = time.perf_counter()
        words = assign_speakers(words, turns, self.max_gap)
        info = {
            **self.describe(), **(meta or {}),
            "duration_s": round(len(audio) / sr, 3),
            "diarize_s": round(t1 - t0, 3), "asr_s": round(t2 - t1, 3),
        }
        return Transcript(words, turns, utterances(words, self.max_pause), info)


def build_pipeline(settings: Settings, reference_words: list[Word] | None = None,
                   reference_turns: list[Turn] | None = None, asr_errors: tuple[float, float, float] = (
                       0.05, 0.03, 0.02), seed: int = 0) -> CallPipeline:
    """Make the pipeline that the settings name. The scripted ASR needs reference words."""
    if settings.asr_backend == "scripted":
        if reference_words is None:
            raise ValueError("the scripted ASR needs reference words (a .nlp file next to the audio). "
                             "For real audio, set CALLSCRIBE_ASR_BACKEND=faster-whisper.")
        sub, dele, ins = asr_errors
        asr: ASR = ScriptedASR(reference_words, sub, dele, ins, seed)
    else:
        asr = FasterWhisperASR(settings.asr_model, settings.device, settings.compute_type, settings.language)
    params = load_params(settings.diarizer_params) if settings.diarizer_params else None
    if settings.diarizer == "spectral":
        diarizer: Diarizer = spectral_from_params(params)
    elif settings.diarizer == "reference":
        if reference_turns is None:
            raise ValueError("the reference diarizer needs an RTTM file")
        diarizer = ReferenceDiarizer(reference_turns)
    else:
        diarizer = PyannoteDiarizer(settings.pyannote_pipeline, settings.hf_token, settings.device, params)
    return CallPipeline(asr, diarizer)
