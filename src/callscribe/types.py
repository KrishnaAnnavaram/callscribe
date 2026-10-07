"""Data types shared by all stages. Times are in seconds from the start of the audio."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    speaker: str | None = None
    prob: float | None = None

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2

    def with_speaker(self, speaker: str) -> "Word":
        return Word(self.start, self.end, self.text, speaker, self.prob)

    def shifted(self, offset: float) -> "Word":
        return Word(self.start + offset, self.end + offset, self.text, self.speaker, self.prob)


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class Transcript:
    words: list[Word]
    turns: list[Turn]
    utterances: list[Utterance]
    meta: dict = field(default_factory=dict)

    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "utterances": [asdict(u) for u in self.utterances],
            "words": [asdict(w) for w in self.words],
            "turns": [asdict(t) for t in self.turns],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(
            words=[Word(**w) for w in d["words"]],
            turns=[Turn(**t) for t in d["turns"]],
            utterances=[Utterance(**u) for u in d["utterances"]],
            meta=d.get("meta", {}),
        )
