from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AdapterHealth:
    name: str
    available: bool
    active: bool
    reason: str

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)

