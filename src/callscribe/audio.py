"""Audio input: decode to 16 kHz mono float32 in memory. No converted copy is written to disk.

WAV files are read with the standard library. Other formats (MP3, M4A, ...) are decoded by an
``ffmpeg`` process that writes raw samples to a pipe.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from . import SAMPLE_RATE


class AudioError(RuntimeError):
    pass


def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def resample(x: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    frac = Fraction(sr_out, sr_in).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator).astype(np.float32)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr, ch, width, n = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
        raw = wf.readframes(n)
    if width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
    else:
        raise AudioError(f"{path}: {8 * width}-bit WAV is not supported")
    return to_mono(x.reshape(-1, ch)), sr


def write_wav(path: str | Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def decode_ffmpeg(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise AudioError("ffmpeg is not installed. Install it to read MP3 and other compressed formats.")
    cmd = [exe, "-nostdin", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg failed for {path}: {proc.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(proc.stdout, "<i2").astype(np.float32) / 32768


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return mono float32 samples at ``sr``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".wav":
        x, sr_in = read_wav(path)
        return resample(x, sr_in, sr)
    return decode_ffmpeg(path, sr)


def frame_rms(x: np.ndarray, sr: int, frame: float = 0.025, hop: float = 0.010) -> np.ndarray:
    """RMS energy of each frame. Frame i starts at i * hop seconds."""
    n, h = int(frame * sr), int(hop * sr)
    if len(x) < n:
        return np.zeros(0)
    idx = np.arange(0, len(x) - n + 1, h)
    windows = np.lib.stride_tricks.sliding_window_view(x, n)[idx]
    return np.sqrt((windows.astype(np.float64) ** 2).mean(axis=1))
