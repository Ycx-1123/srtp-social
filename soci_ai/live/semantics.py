from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..domain import PerceptionEvent


@dataclass(frozen=True)
class BiasPattern:
    phrase: str
    category: str
    weight: float
    explanation: str


class MicroaggressionAnalyzer:
    def __init__(self, patterns: list[BiasPattern]):
        self.patterns = patterns

    @classmethod
    def default(cls) -> "MicroaggressionAnalyzer":
        path = Path(__file__).with_name("bias_patterns.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls([BiasPattern(**item) for item in payload])

    @property
    def categories(self) -> set[str]:
        return {pattern.category for pattern in self.patterns}

    def analyze(self, text: str, at_ms: int) -> PerceptionEvent:
        normalized = "".join(text.casefold().split())
        matches = [
            pattern
            for pattern in self.patterns
            if "".join(pattern.phrase.casefold().split()) in normalized
        ]
        matches.sort(key=lambda item: item.weight, reverse=True)
        if matches:
            strongest = matches[0]
            score = min(1.0, strongest.weight + 0.04 * (len(matches) - 1))
            category = strongest.category
            evidence = "；".join(
                f"{item.phrase}：{item.explanation}" for item in matches[:3]
            )
            confidence = min(1.0, 0.88 + 0.03 * len(matches))
        else:
            score = 0.0
            category = "none"
            evidence = ""
            confidence = 0.72 if normalized else 0.0
        return PerceptionEvent(
            at_ms=max(0, int(at_ms)),
            modality="semantic",
            features={"microbias": round(score, 4), "category": category},
            confidence=confidence,
            source="live_semantics",
            evidence=evidence,
        )
