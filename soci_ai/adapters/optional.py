from __future__ import annotations

import importlib.util

from .base import AdapterHealth


OPTIONAL_PACKAGES = {
    "yolov8": "ultralytics",
    "sensevoice": "funasr",
    "cold_cbbq": "transformers",
    "opencv": "cv2",
}


def discover_optional_adapters(active: bool = False) -> list[AdapterHealth]:
    health: list[AdapterHealth] = []
    for name, package in OPTIONAL_PACKAGES.items():
        available = importlib.util.find_spec(package) is not None
        health.append(
            AdapterHealth(
                name=name,
                available=available,
                active=active and available,
                reason="ready" if available else "package_missing",
            )
        )
    return health

