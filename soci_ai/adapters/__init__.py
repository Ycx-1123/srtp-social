"""Input adapters for scripted showcase and optional research models."""

from .base import AdapterHealth
from .optional import discover_optional_adapters
from .showcase import ScenarioCatalog

__all__ = ["AdapterHealth", "ScenarioCatalog", "discover_optional_adapters"]

