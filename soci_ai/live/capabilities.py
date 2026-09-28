from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ..config import Settings


CapabilityStatus = Literal["ready", "package_missing", "model_missing", "disabled"]


@dataclass(frozen=True)
class ModelCapability:
    name: str
    status: CapabilityStatus
    detail: str


@dataclass(frozen=True)
class LiveCapabilities:
    camera_capture: Literal["browser"]
    microphone_capture: Literal["browser"]
    opencv: ModelCapability
    yolo: ModelCapability
    facial_actions: ModelCapability
    sensevoice: ModelCapability
    local_only: bool = True
    raw_media_persisted: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _model_capability(name: str, package: str, path: Path | None) -> ModelCapability:
    if not _package_available(package):
        return ModelCapability(name=name, status="package_missing", detail=f"missing:{package}")
    if path is None or not path.exists():
        return ModelCapability(name=name, status="model_missing", detail=str(path or "not_configured"))
    return ModelCapability(name=name, status="ready", detail=str(path))


def detect_live_capabilities(settings: Settings) -> LiveCapabilities:
    opencv = ModelCapability(
        name="opencv",
        status="ready" if _package_available("cv2") else "package_missing",
        detail="browser frame decoding and Haar face detection",
    )
    return LiveCapabilities(
        camera_capture="browser",
        microphone_capture="browser",
        opencv=opencv,
        yolo=_model_capability("yolov8-emotion", "ultralytics", settings.yolo_weights),
        facial_actions=_model_capability(
            "face-landmarker-action-units",
            "mediapipe",
            settings.face_landmarker_model,
        ),
        sensevoice=_model_capability(
            "sensevoice-small",
            "funasr",
            settings.sensevoice_model_dir,
        ),
    )


def print_capabilities(settings: Settings) -> None:
    print(json.dumps(detect_live_capabilities(settings).to_dict(), ensure_ascii=False, indent=2))
