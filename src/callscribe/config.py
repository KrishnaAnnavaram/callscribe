"""Settings from environment variables. The Hugging Face token is never printed."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

ASR_BACKENDS = ("scripted", "faster-whisper")
DIARIZERS = ("spectral", "pyannote", "reference")


@dataclass(frozen=True)
class Settings:
    asr_backend: str = "scripted"
    asr_model: str = "large-v3"
    language: str = "en"
    device: str = "cpu"
    compute_type: str = "int8"
    diarizer: str = "spectral"
    pyannote_pipeline: str = "pyannote/speaker-diarization-3.1"
    diarizer_params: str = ""  # path to a JSON file from ``callscribe tune``
    out_dir: str = "out"
    hf_token: str = field(default="", repr=False)

    ENV = {
        "asr_backend": "CALLSCRIBE_ASR_BACKEND",
        "asr_model": "CALLSCRIBE_ASR_MODEL",
        "language": "CALLSCRIBE_LANGUAGE",
        "device": "CALLSCRIBE_DEVICE",
        "compute_type": "CALLSCRIBE_COMPUTE_TYPE",
        "diarizer": "CALLSCRIBE_DIARIZER",
        "pyannote_pipeline": "CALLSCRIBE_PYANNOTE_PIPELINE",
        "diarizer_params": "CALLSCRIBE_DIARIZER_PARAMS",
        "out_dir": "CALLSCRIBE_OUT_DIR",
        "hf_token": "HF_TOKEN",
    }

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        values = {name: env[var] for name, var in cls.ENV.items() if env.get(var)}
        return cls(**values).validate()

    def validate(self) -> "Settings":
        if self.asr_backend not in ASR_BACKENDS:
            raise ValueError(f"CALLSCRIBE_ASR_BACKEND must be one of {ASR_BACKENDS}")
        if self.diarizer not in DIARIZERS:
            raise ValueError(f"CALLSCRIBE_DIARIZER must be one of {DIARIZERS}")
        return self

    def public(self) -> dict[str, str]:
        """All settings for display. The token shows only as set or not set."""
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "hf_token"}
        out["hf_token"] = "set" if self.hf_token else "not set"
        return out


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=VALUE lines into os.environ. Existing variables win. Values are never printed."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
