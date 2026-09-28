from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping


RuntimeMode = Literal["showcase", "edge", "research"]


@dataclass(frozen=True)
class Settings:
    root: Path
    mode: RuntimeMode = "showcase"
    allow_network: bool = False
    load_heavy_models: bool = False
    version: str = "0.1.0"
    yolo_weights: Path | None = None
    sensevoice_model_dir: Path | None = None
    face_landmarker_model: Path | None = None
    live_video_fps: float = 3.0

    @classmethod
    def from_env(
        cls,
        root: Path,
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        values = dict(os.environ if environ is None else environ)
        mode = values.get("SOCI_MODE", "showcase").strip().lower()
        if mode not in {"showcase", "edge", "research"}:
            raise ValueError(f"Unsupported SOCI_MODE: {mode}")
        resolved_root = root.resolve()
        project_root = resolved_root.parent

        def resolve_path(key: str, fallback: Path) -> Path:
            raw = values.get(key, "").strip()
            candidate = Path(raw).expanduser() if raw else fallback
            if not candidate.is_absolute():
                candidate = resolved_root / candidate
            return candidate.resolve()

        try:
            requested_fps = float(values.get("SOCI_LIVE_VIDEO_FPS", "3"))
        except ValueError:
            requested_fps = 3.0
        return cls(
            root=resolved_root,
            mode=mode,  # type: ignore[arg-type]
            allow_network=values.get("SOCI_ALLOW_NETWORK", "0") == "1",
            load_heavy_models=mode != "showcase",
            yolo_weights=resolve_path(
                "SOCI_YOLO_WEIGHTS",
                project_root / "runs" / "classify" / "runs" / "cls" / "yolov8n_cls_train" / "weights" / "best.pt",
            ),
            sensevoice_model_dir=resolve_path(
                "SOCI_SENSEVOICE_MODEL_DIR",
                project_root / "radio_project" / "models" / "SenseVoiceSmall",
            ),
            face_landmarker_model=resolve_path(
                "SOCI_FACE_LANDMARKER_MODEL",
                resolved_root / "models" / "face_landmarker.task",
            ),
            live_video_fps=max(1.0, min(6.0, requested_fps)),
        )
